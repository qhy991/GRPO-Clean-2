import os
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
from dataclasses import asdict, dataclass, field
import torch
import gc
import json
from collections import deque
import sys
import math
from transformers.trainer_callback import DefaultFlowCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from enhanced_debugging_and_fixes import (
    EnhancedCurriculumDebugCallback,
    Qwen3CompatibilityFixer,
    RewardStabilityMonitor,
    integrate_enhanced_debugging
)
from curriculum_debug_config import (
    FixedEnhancedCurriculumManager,
    setup_fixed_curriculum_manager
)
from qwen3_prompt_fix import (
    wrap_prompt_for_qwen3, # 假设这是正确的 qwen3 包装函数
    parse_llm_completion_qwen3,
    qwen3_dataset_processing_pipeline,
    setup_qwen3_generation_config
)
# --- BEGIN: PyTorch Safe Unpickling Configuration ---
logger_temp = logging.getLogger(__name__ + "_startup")
try:
    # Ensure numpy is imported as np for the list below
    # import numpy as np # Should be at the top of the file

    from numpy.core.multiarray import _reconstruct
    from numpy import dtype as numpy_dtype
    # numpy.ndarray is accessed directly via np.ndarray if numpy is imported as np
    from numpy.dtypes import UInt32DType

    safe_globals_list = []

    # Add core numpy elements known to be safe
    if callable(_reconstruct):
        safe_globals_list.append(_reconstruct)
    else:
        logger_temp.warning("_reconstruct is not callable.")

    if isinstance(numpy_dtype, type):
        safe_globals_list.append(numpy_dtype)
    else:
        logger_temp.warning("numpy.dtype is not a type.")

    if isinstance(np.ndarray, type): # Assuming import numpy as np
        safe_globals_list.append(np.ndarray)
    else:
        logger_temp.warning("np.ndarray is not a type.")

    if isinstance(UInt32DType, type):
        safe_globals_list.append(UInt32DType)
    else:
        logger_temp.warning("UInt32DType is not a type.")

    logger_temp.info(f"Initial safe_globals_list before adding scalar types: {len(safe_globals_list)} items.")

    # Add common numpy scalar types explicitly
    numpy_scalar_types_to_add = [
        np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
        np.int_, np.uint, np.longlong, np.ulonglong,
        np.half, np.float16, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ]

    added_scalar_types_count = 0
    for nt_class in numpy_scalar_types_to_add:
        if isinstance(nt_class, type):
            safe_globals_list.append(nt_class)
            added_scalar_types_count += 1
        else:
            logger_temp.warning(f"NumPy scalar '{str(nt_class)}' (type: {type(nt_class)}) is not directly a type class, not adding.")

    logger_temp.info(f"Added {added_scalar_types_count} NumPy scalar types to safe_globals_list.")

    if not safe_globals_list:
        logger_temp.warning("safe_globals_list is empty before calling torch.serialization.add_safe_globals.")

    torch.serialization.add_safe_globals(safe_globals_list)
    logger_temp.info(
        f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items."
    )

except ImportError as e:
    logger_temp.warning(
        f"Failed to import NumPy modules for torch safe globals: {e}. "
        "If you encounter RNG or optimizer state loading issues, this might be the cause."
    )
except AttributeError as e:
    logger_temp.warning(
        f"Attribute error accessing NumPy properties for torch safe globals: {e}."
    )
except Exception as e_globals:
    logger_temp.error(f"An unexpected error occurred while setting up torch safe globals: {e_globals}", exc_info=True)
# --- END: PyTorch Safe Unpickling Configuration ---

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    GenerationConfig
)
import wandb # Keep wandb import here for global access if needed by callbacks immediately

from utils import (
    extract_module_info, validate_verilog_code,
    run_iverilog_simulation, validate_and_update_dataset_paths, enhance_prompt_func,
    EnhancedInferenceCallback, assess_code_quality,DetailedInferenceCallback,
    assess_design_complexity, ExperienceBuffer, StepLoggingCallback, monitor_advanced_stage_training
)
from config import EnvConfig, ScriptConfig, EnhancedRewardConfig
from enhanced_curriculum import (
    EnhancedCurriculumManager, CurriculumStageConfig,
    create_default_curriculum_stages, create_custom_curriculum_stages
)
from datetime import datetime

# logger = logging.getLogger(__name__) # THIS WILL BE INITIALIZED INSIDE MAIN

class DetailedWandbCallback(TrainerCallback):
    """增强的 W&B 日志回调"""

    def __init__(self, env_cfg, script_cfg, reward_cfg, experience_buffer=None):
        self.env_cfg = env_cfg
        self.script_cfg = script_cfg
        self.reward_cfg = reward_cfg
        self.experience_buffer = experience_buffer
        self.step_count = 0
        self.recent_rewards = deque(maxlen=100) # Added
        # Ensure logger is available if methods are called before main logger is fully set up
        # This is a fallback, ideally methods are called after full setup.
        self._logger = logging.getLogger(__name__ + ".DetailedWandbCallback")


    def on_init_end(self, args, state, control, **kwargs):
        if not getattr(self.env_cfg, 'wandb_disable', False) and args.local_rank <= 0 : # only on main process
            # import wandb # Already imported globally
            wandb.init(
                project=getattr(self.env_cfg, 'wandb_project', 'grpo-training'),
                name=getattr(self.env_cfg, 'wandb_run_name', None), # This should be set based on run_name
                config={
                    **asdict(self.script_cfg),
                    **asdict(self.reward_cfg),
                    "trainer_args": args.to_dict()
                },
                # resume="allow", # Let main() handle resume logic for WANDB_RUN_ID
                # id=os.getenv("WANDB_RUN_ID", None) # Let main() handle resume logic
            )
            self._logger.info("🚀 W&B 初始化完成")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or getattr(self.env_cfg, 'wandb_disable', False) or args.local_rank > 0:
            return

        try:
            # import wandb # Already imported globally
            if wandb.run is None:
                return

            current_step = getattr(state, 'global_step', 0) or 0
            self.step_count = current_step

            wandb_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    wandb_logs[f"train/{key}"] = value

            if self.experience_buffer:
                buffer_stats = self.experience_buffer.get_stats()
                for key, value in buffer_stats.items():
                    if isinstance(value, (int, float)):
                        wandb_logs[f"experience_buffer/{key}"] = value

            if wandb_logs:
                wandb.log(wandb_logs, step=current_step)

            if self.recent_rewards:
                try:
                    wandb.log({"reward_distribution": wandb.Histogram(np.array(self.recent_rewards))}, step=current_step)
                except Exception as e_hist:
                    self._logger.warning(f"Failed to log reward histogram to W&B: {e_hist}")

        except Exception as e_wandb:
            self._logger.warning(f"W&B 日志记录失败: {e_wandb}")

    def add_reward(self, reward: float):
        if getattr(self.env_cfg, 'wandb_disable', False) or not (hasattr(wandb, 'run') and wandb.run is not None):
            return
        self.recent_rewards.append(reward)

    def log_reward_components(self, reward_components: Dict[str, float], step: Optional[int] = None):
        try:
            if hasattr(wandb, 'run') and wandb.run is not None and not getattr(self.env_cfg, 'wandb_disable', False):
                log_step = step if step is not None else self.step_count
                wandb.log({f"reward_components/{k}": v for k, v in reward_components.items()}, step=log_step)
        except Exception as e:
            self._logger.warning(f"Failed to log reward components: {e}")

    def log_batch_aggregated_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if getattr(self.env_cfg, 'wandb_disable', False) or not metrics:
            return

        try:
            # import wandb # Already imported globally
            if wandb.run is None:
                self._logger.warning("W&B run not initialized. Skipping log_batch_aggregated_metrics.")
                return

            log_step = step
            if log_step is None:
                log_step = self.step_count if hasattr(self, 'step_count') and self.step_count > 0 else None

            sanitized_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    sanitized_metrics[key] = value
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    if not (np.isnan(value) or np.isinf(value)):
                        sanitized_metrics[key] = float(value)

            if sanitized_metrics:
                if log_step is not None:
                    wandb.log(sanitized_metrics, step=log_step)
                else:
                    wandb.log(sanitized_metrics)
                self._logger.debug(f"Logged batch-aggregated metrics to W&B (step {log_step if log_step is not None else 'auto'}): {list(sanitized_metrics.keys())}")
            else:
                self._logger.debug("No valid batch-aggregated metrics to log after sanitization.")

        except ImportError:
            self._logger.warning("W&B module not found. Cannot log batch_aggregated_metrics.")
        except Exception as e_wandb_agg:
            self._logger.error(f"Error logging batch-aggregated metrics to W&B: {e_wandb_agg}", exc_info=True)


class CustomStatePersistenceCallback(TrainerCallback):
    def __init__(self,
                 curriculum_manager: Optional[EnhancedCurriculumManager],
                 experience_buffer: Optional[ExperienceBuffer],
                 script_cfg: ScriptConfig):
        self.curriculum_manager = curriculum_manager
        self.experience_buffer = experience_buffer
        self.script_cfg = script_cfg
        self._logger = logging.getLogger(__name__ + ".CustomStatePersistenceCallback")


    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        current_step = getattr(state, 'global_step', 0) or 0
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{current_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        self._logger.debug(f"CustomStatePersistenceCallback: 正在向 checkpoint 文件夹保存自定义状态: {checkpoint_folder}")

        if self.curriculum_manager:
            curriculum_state = self.curriculum_manager.get_curriculum_state()
            file_path = os.path.join(checkpoint_folder, "enhanced_curriculum_state.json")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(curriculum_state, f, indent=2)
                self._logger.info(f"已保存课程学习状态到: {file_path}")
            except Exception as e:
                self._logger.error(f"保存课程学习状态失败 ({file_path}): {e}")

        if self.experience_buffer and self.script_cfg.enable_experience_replay:
            buffer_state = self.experience_buffer.get_buffer_state()
            file_path = os.path.join(checkpoint_folder, "enhanced_experience_buffer_state.json")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(buffer_state, f, indent=2)
                self._logger.info(f"已保存经验回放池状态到: {file_path}")
            except TypeError as te:
                self._logger.warning(f"经验回放池内容可能不是完全 JSON 可序列化，保存尝试失败 ({file_path}): {te}。请检查经验条目格式。")
            except Exception as e:
                self._logger.error(f"保存经验回放池状态失败 ({file_path}): {e}")


def validate_dataset_for_curriculum(dataset: Dataset, script_cfg: ScriptConfig, logger_ref) -> bool: # Pass logger
    if dataset is None:
        logger_ref.error("Dataset is None, cannot validate for curriculum learning")
        return False

    if len(dataset) == 0:
        logger_ref.error("Empty dataset provided for curriculum learning")
        return False

    required_fields = ['prompt', 'testbench_path', 'expected_total_tests', 'reference_verilog_path']
    missing_fields = [field for field in required_fields if field not in dataset.column_names]

    if missing_fields:
        logger_ref.error(f"Dataset missing required fields for training: {missing_fields}")
        return False

    curriculum_fields = ['level', 'complexity_score']
    missing_curriculum_fields = [field for field in curriculum_fields if field not in dataset.column_names]

    if missing_curriculum_fields:
        logger_ref.warning(f"Dataset missing curriculum learning fields: {missing_curriculum_fields}")
        logger_ref.warning("Curriculum learning may not work optimally")

    if len(dataset) > 0:
        first_example = dataset[0]
        if 'level' in first_example:
            levels = [ex.get('level') for ex in dataset if ex]
            unique_levels = set(l for l in levels if l is not None)
            expected_levels = {'basic', 'intermediate', 'advanced', 'expert'}

            if not unique_levels.intersection(expected_levels) and unique_levels:
                logger_ref.warning(f"Unusual level values found: {unique_levels}")
                logger_ref.warning(f"Expected values: {expected_levels}")

        if 'complexity_score' in first_example:
            complexities = [ex.get('complexity_score', 0) for ex in dataset if ex]
            complexities = [c for c in complexities if isinstance(c, (int, float))]

            if complexities:
                min_complexity = min(complexities)
                max_complexity = max(complexities)

                if min_complexity < 0 or max_complexity > 10:
                    logger_ref.warning(f"Complexity scores outside expected range [0-10]: {min_complexity:.2f} - {max_complexity:.2f}")

    logger_ref.info("Dataset validation for curriculum learning completed")
    return True


def setup_curriculum_manager(script_cfg: ScriptConfig, dataset: Dataset, logger_ref) -> Optional[EnhancedCurriculumManager]: # Pass logger
    if not script_cfg.enable_curriculum:
        logger_ref.info("Curriculum learning disabled.")
        return None

    if dataset is None:
        logger_ref.error("Cannot setup curriculum manager with None dataset")
        return None

    has_level_info = False
    if len(dataset) > 0:
        first_example = dataset[0]
        has_level_info = 'level' in first_example and first_example['level'] is not None and str(first_example['level']).strip() != ""

    if not has_level_info:
        logger_ref.warning("Dataset does not contain valid 'level' field data. Curriculum learning will use complexity-only or default stages.")
        if script_cfg.curriculum_type == "dual_layer" or script_cfg.curriculum_type == "level_only":
            logger_ref.info(f"Switching curriculum type from '{script_cfg.curriculum_type}' to 'complexity_only' due to missing level info.")
            script_cfg.curriculum_type = "complexity_only"

    curriculum_stages_config_list = []

    if script_cfg.curriculum_type == "dual_layer" and has_level_info:
        logger_ref.info(f"Dynamically generating dual_layer curriculum stages with focus: {script_cfg.curriculum_focus_levels}, emphasis: {script_cfg.curriculum_complexity_emphasis}")

        level_counts_dist = {}
        complexity_by_level_dist = {}
        if dataset and len(dataset) > 0:
            for example in dataset:
                level = example.get('level', 'unknown').lower()
                complexity = example.get('complexity_score', 5.0)
                level_counts_dist[level] = level_counts_dist.get(level, 0) + 1
                if level not in complexity_by_level_dist:
                    complexity_by_level_dist[level] = []
                complexity_by_level_dist[level].append(complexity)

        dataset_distribution_for_stages = {
            'level_counts': level_counts_dist,
            'complexity_by_level': complexity_by_level_dist,
            'total_samples': len(dataset) if dataset else 0
        }

        curriculum_stages_config_list = create_custom_curriculum_stages(
            dataset_distribution=dataset_distribution_for_stages,
            focus_levels=script_cfg.curriculum_focus_levels,
            complexity_emphasis=script_cfg.curriculum_complexity_emphasis
        )
        logger_ref.info(f"Generated {len(curriculum_stages_config_list)} custom stages for dual_layer.")
    else:
        if script_cfg.curriculum_type != "dual_layer" and has_level_info :
            logger_ref.info(f"Curriculum type is '{script_cfg.curriculum_type}'. Using default stages as specific logic for this type (other than dual_layer) is not implemented here, or conditions not met.")
        elif not has_level_info and script_cfg.curriculum_type != "complexity_only":
            logger_ref.info(f"Dataset lacks level information for '{script_cfg.curriculum_type}'. Falling back to default (likely complexity-based) stages.")

        curriculum_stages_config_list = create_default_curriculum_stages()
        logger_ref.info(f"Generated {len(curriculum_stages_config_list)} default stages (type: {script_cfg.curriculum_type}).")

    if curriculum_stages_config_list:
        for stage_config_item in curriculum_stages_config_list:
            if isinstance(stage_config_item, CurriculumStageConfig):
                stage_config_item.min_evaluations = 10
            else:
                logger_ref.warning(f"Encountered non-CurriculumStageConfig item in list: {type(stage_config_item)}")
        logger_ref.info(f"Ensured min_evaluations is 10 for all {len(curriculum_stages_config_list)} stages.")
    else:
        logger_ref.warning("No curriculum stages were generated. Curriculum learning might be ineffective.")
        return None

    if not curriculum_stages_config_list:
        logger_ref.error("No curriculum stages defined after attempting generation. Disabling curriculum learning.")
        return None

    return EnhancedCurriculumManager(curriculum_stages_config_list, dataset)


def enhance_dataset_with_level_and_complexity(dataset: Dataset, logger_ref) -> Dataset: # Pass logger
    if not dataset or len(dataset) == 0:
        logger_ref.warning("enhance_dataset_with_level_and_complexity: Received empty or None dataset.")
        return dataset

    def process_example(example):
        if not isinstance(example, dict):
            if hasattr(example, 'keys') and callable(example.keys) and hasattr(example, '__getitem__'):
                logger_ref.debug(f"Converting LazyRow/dict-like object of type {type(example)} to dict in process_example.")
                try:
                    example = dict(example)
                except Exception as e_conv:
                    logger_ref.error(f"Failed to convert dict-like object {type(example)} to dict: {e_conv}. Skipping example.")
                    return None
            else:
                logger_ref.warning(f"Skipping non-dict/non-dict-like example in process_example: {type(example)}")
                return None

        current_level = example.get('level')
        if current_level is None or not isinstance(current_level, str) or not current_level.strip():
            example['level'] = 'intermediate'
        else:
            example['level'] = str(current_level).lower().strip()

        current_complexity = example.get('complexity_score')
        try:
            example['complexity_score'] = float(current_complexity)
        except (ValueError, TypeError):
            example['complexity_score'] = 5.0

        if 'difficulty' in example and 'level' not in example:
            difficulty_to_level = {'A': 'basic', 'B': 'basic', 'C': 'intermediate', 'D': 'intermediate', 'E': 'advanced', 'F': 'expert'}
            example['level'] = difficulty_to_level.get(str(example['difficulty']).upper(), 'intermediate')

        return example

    try:
        enhanced_dataset = dataset.map(process_example, num_proc=1).filter(lambda x: x is not None)
        logger_ref.info("Dataset enhanced/validated with level and complexity information")

        if len(enhanced_dataset) > 0:
            levels = [ex['level'] for ex in enhanced_dataset]
            complexities = [ex['complexity_score'] for ex in enhanced_dataset]

            logger_ref.info("Enhanced dataset distribution:")
            for level_val in sorted(list(set(levels))):
                count = levels.count(level_val)
                level_complexities = [c for ex, c in zip(enhanced_dataset, complexities) if ex['level'] == level_val]
                avg_complexity = np.mean(level_complexities) if level_complexities else 0
                logger_ref.info(f"  {level_val.capitalize()}: {count} samples, avg complexity: {avg_complexity:.2f}")
        else:
            logger_ref.warning("Dataset is empty after enhancement processing.")

        return enhanced_dataset

    except Exception as e:
        logger_ref.error(f"Failed to enhance dataset: {e}", exc_info=True)
        return dataset

def calculate_enhanced_rewards_for_single_prompt(
    prompt_str: str,
    completions_for_this_prompt: List[str],
    current_tb_path: str,
    current_expected_total_from_manifest: int,
    current_ref_verilog_path: str,
    reward_config: EnhancedRewardConfig,
    logger_ref, # Pass logger
    training_step: int = 0,
    wandb_callback: Optional[DetailedWandbCallback] = None,
    output_dir_for_debug: Optional[str] = None
) -> List[Dict[str, Any]]:
    detailed_results_for_prompt: List[Dict[str, Any]] = []
    num_completions = len(completions_for_this_prompt)

    prompt_id_base = prompt_str.split('\n', 1)[0]
    name_match_for_id = re.search(r"module MUST be named `(\w+)`", prompt_str, re.IGNORECASE)
    if name_match_for_id:
        prompt_id_base = f"Mod_{name_match_for_id.group(1)}"

    sanitized_prompt_id_for_file = re.sub(r'[^\w_.)( -]', '', prompt_id_base).strip().replace(' ', '_')[:50]
    if not sanitized_prompt_id_for_file:
        sanitized_prompt_id_for_file = "unknown_prompt"
    prompt_id_for_log = prompt_id_base[:70]

    logger_ref.debug(f"ENHANCED_REWARDS: For '{prompt_id_for_log}', processing {num_completions} completion(s).")

    module_name, req_ports = "", []
    if current_ref_verilog_path and os.path.exists(current_ref_verilog_path):
        module_name, req_ports = extract_module_info(current_ref_verilog_path)

    if not module_name:
        logger_ref.error(f"ENHANCED_REWARDS: '{prompt_id_for_log}': Failed to extract module info from ref Verilog '{current_ref_verilog_path}' or path invalid.")
        error_reward_val = reward_config.get_scaled_reward(reward_config.compilation_failure * 2, training_step)
        error_result_item = {
            "final_reward": error_reward_val,
            "unscaled_components": {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": reward_config.compilation_failure * 2},
            "funnel_metrics": {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
        }
        return [error_result_item] * num_completions

    for j, full_output in enumerate(completions_for_this_prompt):
        log_pref = f"ENHANCED_REWARDS: '{prompt_id_for_log}', Completion {j+1}/{num_completions}"
        current_unscaled_components = {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": 0.0}
        current_funnel_metrics = {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}

        if not isinstance(full_output, str):
            logger_ref.error(f"{log_pref}: Output is not a string, type: {type(full_output)}.")
            total_reward = reward_config.get_scaled_reward(reward_config.compilation_failure * 2, training_step)
            current_unscaled_components["base_compilation"] = reward_config.compilation_failure * 2
            detailed_results_for_prompt.append({
                "final_reward": total_reward,
                "unscaled_components": current_unscaled_components,
                "funnel_metrics": current_funnel_metrics
            })
            continue

        raw_output_len = len(full_output)
        if raw_output_len > 1000:
            logger_ref.debug(f"{log_pref}: Raw model output (len={raw_output_len}):\n{full_output[:500]}\n...\n{full_output[-500:]}")
        else:
            logger_ref.debug(f"{log_pref}: Raw model output (len={raw_output_len}):\n{full_output}")

        _, code = parse_llm_completion_qwen3(
            full_output,
            debug_prompt=prompt_str,
            debug_context={"step": training_step, "sample_idx": j, "model": "qwen3"}
        )
        # Original print statements for debugging, will use logger_ref if needed for structured logging
        # print("*"*100)
        # print(f"{prompt_str=}")
        # print(f"{full_output=}")
        # print("*"*100)
        # Instead, use logger for consistency if these are important operational logs
        logger_ref.debug(f"{log_pref} DEBUG_REWARD_CALC: Prompt='{prompt_str}', Full_Output='{full_output}'")


        if code and code.strip():
            logger_ref.debug(f"{log_pref}: Extracted code:\n{code}")
            current_funnel_metrics["code_extracted"] = True
        else:
            logger_ref.debug(f"{log_pref}: Extracted code: None or empty.")
            current_funnel_metrics["code_extracted"] = False

        if not code or not code.strip():
            penalty_type = reward_config.missing_code_block_penalty if not code else reward_config.compilation_failure
            current_unscaled_components["base_compilation"] = penalty_type
            log_msg = "No Verilog code block found" if not code else "Empty code block"
            logger_ref.warning(f"{log_pref}: {log_msg} in output.")

            if output_dir_for_debug and not current_funnel_metrics["code_extracted"]:
                try:
                    debug_save_dir = os.path.join(output_dir_for_debug, "debug_no_code_outputs")
                    os.makedirs(debug_save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"step_{training_step}_prompt_{sanitized_prompt_id_for_file}_compl_{j}_{timestamp}.txt"
                    filepath = os.path.join(debug_save_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f_debug:
                        f_debug.write(f"--- PROMPT (Original ID for log: {prompt_id_for_log}) ---\n{prompt_str}\n\n")
                        f_debug.write(f"--- COMPLETION INDEX: {j} ---\n\n--- FULL MODEL OUTPUT ---\n{full_output}")
                    logger_ref.info(f"{log_pref}: Saved problematic (no code) output to {filepath}")
                except Exception as e_save:
                    logger_ref.error(f"{log_pref}: Failed to save debug output: {e_save}")

            total_reward = reward_config.get_scaled_reward(current_unscaled_components["base_compilation"], training_step)
            detailed_results_for_prompt.append({
                "final_reward": total_reward,
                "unscaled_components": current_unscaled_components,
                "funnel_metrics": current_funnel_metrics
            })
            continue

        quality_metrics = assess_code_quality(code)
        current_unscaled_components["efficiency"] = (
            quality_metrics.get("efficiency", 0) * reward_config.code_efficiency_bonus +
            quality_metrics.get("structure", 0) * reward_config.synthesis_friendly_bonus -
            max(0, (1 - quality_metrics.get("complexity", 1)) * reward_config.code_complexity_penalty)
        )
        current_unscaled_components["readability"] = quality_metrics.get("readability", 0) * reward_config.code_readability_bonus

        is_valid, err_msg = validate_verilog_code(code, module_name, req_ports)
        if is_valid:
            logger_ref.debug(f"{log_pref}: Verilog validation: is_valid=True, err_msg=\"{err_msg}\"")
        else:
            logger_ref.info(f"{log_pref}: Verilog validation: is_valid=False, err_msg=\"{err_msg}\"")
            current_unscaled_components["base_compilation"] = reward_config.compilation_failure

        if is_valid:
            sim_res = run_iverilog_simulation(
                code, current_tb_path, current_expected_total_from_manifest,
                prompt_id_for_log, j, logger_ref.isEnabledFor(logging.DEBUG)
            )
            logger_ref.debug(f"{log_pref}: Simulation results: {sim_res}")
            current_funnel_metrics["compiled_successfully"] = sim_res["compilation_success"]

            if not sim_res["compilation_success"]:
                current_unscaled_components["base_compilation"] = reward_config.compilation_failure
                logger_ref.info(f"{log_pref}: Compilation FAILED. Error: {sim_res.get('error_message', 'N/A')}")
            else:
                current_unscaled_components["base_compilation"] = reward_config.compilation_success
                current_funnel_metrics["sim_ran_successfully"] = sim_res["simulation_run_success"]
                if not sim_res["simulation_run_success"]:
                    current_unscaled_components["functional"] = reward_config.simulation_crash
                    logger_ref.info(f"{log_pref}: Simulation CRASHED.")
                elif not sim_res["parsing_success"]:
                    current_unscaled_components["functional"] = reward_config.output_parse_error
                    logger_ref.info(f"{log_pref}: Output parsing FAILED.")
                else:
                    p, total_tests_in_output = sim_res["passed_tests"], sim_res["total_tests_in_output"]
                    current_funnel_metrics["passed_tests"] = p
                    if total_tests_in_output > 0:
                        pass_ratio = p / total_tests_in_output
                        base_functional = pass_ratio * reward_config.max_functional_reward
                        if p > 1:
                            bonus_factor = reward_config.test_pass_bonus_multiplier ** (p - 1)
                            base_functional *= min(bonus_factor, 2.0)
                        current_unscaled_components["functional"] = base_functional
                        if sim_res["all_tests_passed_by_tb"] and p == total_tests_in_output:
                            current_unscaled_components["robustness"] = reward_config.all_tests_passed_bonus
                        if p == total_tests_in_output and total_tests_in_output >= 5:
                            current_unscaled_components["robustness"] += reward_config.edge_case_handling_bonus
                    elif sim_res["all_tests_passed_by_tb"]:
                        current_unscaled_components["functional"] = reward_config.max_functional_reward
                        current_unscaled_components["robustness"] = reward_config.all_tests_passed_bonus
                        current_funnel_metrics["passed_tests"] = total_tests_in_output # Assume all expected passed
                    else:
                        current_unscaled_components["functional"] = reward_config.output_parse_error
                    logger_ref.info(f"{log_pref}: Functional tests - Passed: {p}/{total_tests_in_output}, Overall: {sim_res['all_tests_passed_by_tb']}")

        unscaled_total_reward = (
            reward_config.functional_weight * current_unscaled_components["functional"] +
            reward_config.efficiency_weight * current_unscaled_components["efficiency"] +
            reward_config.readability_weight * current_unscaled_components["readability"] +
            reward_config.robustness_weight * current_unscaled_components["robustness"] +
            current_unscaled_components["base_compilation"]
        )
        final_scaled_reward = reward_config.get_scaled_reward(unscaled_total_reward, training_step)

        if wandb_callback:
            wandb_callback.add_reward(final_scaled_reward)

        logger_ref.info(
            f"{log_pref}: Unscaled Rewards - Func:{current_unscaled_components['functional']:.2f} Eff:{current_unscaled_components['efficiency']:.2f} "
            f"Read:{current_unscaled_components['readability']:.2f} Rob:{current_unscaled_components['robustness']:.2f} "
            f"BaseComp:{current_unscaled_components['base_compilation']:.2f}. UnscaledTotal: {unscaled_total_reward:.2f}. FinalScaled: {final_scaled_reward:.2f}"
        )
        detailed_results_for_prompt.append({
            "final_reward": final_scaled_reward,
            "unscaled_components": current_unscaled_components,
            "funnel_metrics": current_funnel_metrics
        })

    if len(detailed_results_for_prompt) != num_completions:
        logger_ref.critical(f"ENHANCED_REWARDS: '{prompt_id_for_log}': Result list length mismatch. Padding.")
        error_reward_val = reward_config.get_scaled_reward(reward_config.compilation_failure * 3, training_step)
        error_result_item = {
            "final_reward": error_reward_val,
            "unscaled_components": {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": reward_config.compilation_failure * 3},
            "funnel_metrics": {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
        }
        detailed_results_for_prompt.extend([error_result_item] * (num_completions - len(detailed_results_for_prompt)))
    return detailed_results_for_prompt


class CurriculumProgressCallback(TrainerCallback):
    def __init__(self, curriculum_manager, trainer_ref, output_dir):
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref # This will be set later by main
        self.performance_history = []
        self.output_dir = output_dir
        self.debug_log_path = os.path.join(output_dir, "curriculum_debug.txt")
        self._logger = logging.getLogger(__name__ + ".CurriculumProgressCallback") # Use instance logger

        # Ensure the directory for the debug log exists
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        with open(self.debug_log_path, 'w') as f:
            f.write(f"=== 课程学习调试日志 - {datetime.now()} ===\n")

    def _write_debug(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_msg = f"[{timestamp}] CURRICULUM: {message}"
        self._logger.info(debug_msg) # Use instance logger
        with open(self.debug_log_path, 'a') as f:
            f.write(debug_msg + "\n")

    def on_evaluate(self, args, state, control, **kwargs):
        if self.curriculum_manager and args.local_rank <= 0:
            current_step = getattr(state, 'global_step', 0) or 0
            current_stage_idx = self.curriculum_manager.current_stage

            avg_test_pass_rate = 0.0
            found_metric = False
            for log_entry in reversed(state.log_history):
                if 'eval_avg_test_pass_rate' in log_entry: # Make sure this metric is being logged by another callback
                    avg_test_pass_rate = log_entry['eval_avg_test_pass_rate']
                    found_metric = True
                    self._write_debug(f"Found 'eval_avg_test_pass_rate' in log_history: {avg_test_pass_rate:.4f}")
                    break

            if not found_metric:
                self._write_debug("WARN: 'eval_avg_test_pass_rate' not found in log_history. Using default 0.0. Curriculum advancement may be affected.")

            performance_estimate = avg_test_pass_rate

            self._write_debug(f"Step: {current_step}")
            self._write_debug(f"Current stage index: {current_stage_idx}")
            self._write_debug(f"Performance estimate (avg_test_pass_rate): {performance_estimate:.4f}")

            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
                threshold = stage_config.performance_threshold
                min_evals = stage_config.min_evaluations

                self._write_debug(f"Stage Name: {stage_config.name}")
                self._write_debug(f"Performance Threshold: {threshold}")
                self._write_debug(f"Configured Min Evaluations for stage: {min_evals}")
                self._write_debug(f"Actual evaluations for this stage so far: {len(getattr(self.curriculum_manager, 'stage_performance_history', []))}")

                try:
                    should_advance = self.curriculum_manager.should_advance_stage(performance_estimate)
                except TypeError as e:
                    self._write_debug(f"ERROR calling should_advance_stage: {e}. Check method signature in EnhancedCurriculumManager.")
                    should_advance = False

                self._write_debug(f"Decision: Should advance stage? {should_advance}")

                if should_advance:
                    advance_success = False
                    if hasattr(self.curriculum_manager, 'advance_stage'):
                        advance_success = self.curriculum_manager.advance_stage()
                    elif hasattr(self.curriculum_manager, 'advance_to_next_stage'):
                        advance_success = self.curriculum_manager.advance_to_next_stage()
                    else:
                        self._write_debug("ERROR: Curriculum manager lacks 'advance_stage' or 'advance_to_next_stage' method.")

                    if advance_success:
                        new_stage_idx = self.curriculum_manager.current_stage
                        new_dataset = self.curriculum_manager.get_current_stage_dataset()
                        self._write_debug(f"✅ Successfully advanced to stage {new_stage_idx}. New dataset size: {len(new_dataset)}")

                        progress_record = {
                            "step": current_step,
                            "old_stage_idx": current_stage_idx,
                            "new_stage_idx": new_stage_idx,
                            "performance_metric (avg_test_pass_rate)": performance_estimate,
                            "new_dataset_size": len(new_dataset),
                            "timestamp": datetime.now().isoformat()
                        }
                        progress_file = os.path.join(self.output_dir, "stage_progress.jsonl")
                        with open(progress_file, 'a') as f:
                            f.write(json.dumps(progress_record) + "\n")

                        # import wandb # Already imported globally
                        if hasattr(wandb, 'run') and wandb.run is not None:
                            old_stage_name = stage_config.name
                            new_stage_name = "Unknown"
                            if new_stage_idx < len(self.curriculum_manager.curriculum_stages):
                                new_stage_name = self.curriculum_manager.curriculum_stages[new_stage_idx].name
                            wandb.log({
                                "curriculum/stage_transition": 1,
                                "curriculum/old_stage_index": current_stage_idx,
                                "curriculum/new_stage_index": new_stage_idx,
                                "curriculum/old_stage_name": old_stage_name,
                                "curriculum/new_stage_name": new_stage_name,
                                "curriculum/performance_metric": performance_estimate
                            }, step=current_step)
                            self._write_debug(f"Logged stage transition to W&B: {old_stage_name} -> {new_stage_name}")
            else:
                self._write_debug("INFO: All curriculum stages completed or current stage index is out of bounds.")
            self._write_debug("-" * 50)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        if not self.curriculum_manager or args.local_rank > 0:
            return

        current_step = getattr(state, 'global_step', 0) or 0

        # import wandb # Already imported globally
        if hasattr(wandb, 'run') and wandb.run is not None:
            current_stage_idx = self.curriculum_manager.current_stage
            stage_name = "Unknown/Completed"
            num_samples_in_stage = 0
            current_stage_performance_threshold = 0.0

            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                current_stage_obj = self.curriculum_manager.curriculum_stages[current_stage_idx]
                stage_name = current_stage_obj.name
                current_stage_performance_threshold = current_stage_obj.performance_threshold
                try:
                    current_dataset = self.curriculum_manager.get_current_stage_dataset()
                    num_samples_in_stage = len(current_dataset) if current_dataset else 0
                except Exception as e_ds_len:
                    self._logger.warning(f"CurriculumProgressCallback: Could not get current stage dataset size for logging: {e_ds_len}")

            latest_perf_estimate = 0.0
            if logs and 'eval_avg_test_pass_rate' in logs:
                latest_perf_estimate = logs['eval_avg_test_pass_rate']
            elif state.log_history:
                for log_entry in reversed(state.log_history):
                    if 'eval_avg_test_pass_rate' in log_entry:
                        latest_perf_estimate = log_entry['eval_avg_test_pass_rate']
                        break
            
            wandb.log({
                "curriculum/current_stage_idx": current_stage_idx,
                "curriculum/current_stage_name_numeric": current_stage_idx,
                "curriculum/num_samples_in_stage": num_samples_in_stage,
                "curriculum/current_stage_perf_threshold": current_stage_performance_threshold,
                "curriculum/latest_eval_avg_test_pass_rate": latest_perf_estimate
            }, step=state.global_step if hasattr(state, 'global_step') else current_step)


        if not hasattr(self, 'last_locally_logged_stage_idx') or self.last_locally_logged_stage_idx != self.curriculum_manager.current_stage or current_step % 50 == 0:
            current_stage_idx_for_local_log = self.curriculum_manager.current_stage
            stage_name_for_local_log = "Unknown/Completed"
            if current_stage_idx_for_local_log < len(self.curriculum_manager.curriculum_stages):
                stage_name_for_local_log = self.curriculum_manager.curriculum_stages[current_stage_idx_for_local_log].name
            
            dataset_size_for_log = 0
            try:
                dataset_size_for_log = len(self.curriculum_manager.get_current_stage_dataset())
            except Exception:
                pass # Already logged warning if it fails elsewhere
            self._write_debug(f"Step {current_step}: Currently in curriculum stage {current_stage_idx_for_local_log} ('{stage_name_for_local_log}'). Dataset size: {dataset_size_for_log}")
            self.last_locally_logged_stage_idx = current_stage_idx_for_local_log


# Placeholder for DynamicDifficultyAdjuster if it's defined elsewhere or simple
class DynamicDifficultyAdjuster:
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self._logger = logging.getLogger(__name__ + ".DynamicDifficultyAdjuster")


    def adjust_current_stage_difficulty(self, performance_history):
        # Placeholder: Actual logic for difficulty adjustment would go here
        if performance_history and len(performance_history) % 10 == 0: # Example: Log every 10 adjustments
            self._logger.info(f"DynamicDifficultyAdjuster: Checked performance history (length {len(performance_history)}). No adjustment logic implemented yet.")
        pass


class OptimizedCurriculumCallback(DefaultFlowCallback):
    """优化的课程学习回调，包含动态难度调整"""

    def __init__(self, curriculum_manager, trainer_ref, output_dir):
        super().__init__()
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref # Will be set later
        self.output_dir = output_dir
        self.difficulty_adjuster = DynamicDifficultyAdjuster(curriculum_manager)
        self.performance_history = []
        self._logger = logging.getLogger(__name__ + ".OptimizedCurriculumCallback")


    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self.curriculum_manager or args.local_rank > 0:
            return

        current_loss = logs.get('train_loss', float('inf')) # GRL usually uses 'loss', GRPO might be 'train/loss' or similar
        if 'loss' in logs: # Prefer 'loss' if available from trainer directly
            current_loss = logs['loss']

        training_step = state.global_step

        performance = 1.0 - min(current_loss, 1.0) # Simple performance metric
        self.performance_history.append({
            'step': training_step,
            'performance': performance,
            'loss': current_loss,
            'stage': self.curriculum_manager.current_stage
        })
        
        # should_advance_to_next_stage in EnhancedCurriculumManager needs to exist and take these params
        # For now, we rely on CurriculumProgressCallback.on_evaluate for stage advancement based on eval metrics
        # This on_log based advancement on training loss might be too noisy or conflict.
        # if self.curriculum_manager.should_advance_to_next_stage(current_loss, training_step): # Assuming method exists
        #     old_stage = self.curriculum_manager.current_stage
        #     success = self.curriculum_manager.advance_to_next_stage()
        #     if success and self.trainer_ref:
        #         new_dataset = self.curriculum_manager.get_current_stage_dataset()
        #         self._logger.info(f"🎯 OptimizedCurriculumCallback: 阶段进阶: {old_stage} → {self.curriculum_manager.current_stage}")
        #         self._logger.info(f"📊 新数据集大小: {len(new_dataset)}")
        #         # Trainer dataset update is complex and usually handled by recreating dataloader or specific trainer methods

        self.difficulty_adjuster.adjust_current_stage_difficulty(self.performance_history)

        if training_step % 50 == 0:
            self._save_curriculum_state()

    def _save_curriculum_state(self):
        if not self.output_dir:
            return

        state_data = {
            'current_stage': self.curriculum_manager.current_stage,
            'performance_history': self.performance_history[-100:],
            'timestamp': datetime.now().isoformat()
        }
        state_file = os.path.join(self.output_dir, 'curriculum_state_detailed.json')
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._logger.warning(f"保存课程状态失败: {e}")

def debug_checkpoint_contents(checkpoint_path, logger_ref): # Pass logger
    if not os.path.exists(checkpoint_path):
        logger_ref.error(f"Checkpoint路径不存在: {checkpoint_path}")
        return

    logger_ref.info(f"🔍 检查checkpoint内容: {checkpoint_path}")
    try:
        files = os.listdir(checkpoint_path)
        logger_ref.info(f"Checkpoint文件列表: {files}")
        key_files = {
            'config.json': '模型配置', 'pytorch_model.bin': 'PyTorch模型权重',
            'model.safetensors': 'SafeTensors模型权重', 'adapter_config.json': 'PEFT适配器配置',
            'adapter_model.bin': 'PEFT适配器权重(bin)', 'adapter_model.safetensors': 'PEFT适配器权重(safetensors)',
            'training_args.bin': '训练参数', 'trainer_state.json': '训练状态',
            'optimizer.pt': '优化器状态', 'scheduler.pt': '调度器状态'
        }
        logger_ref.info("📁 关键文件检查:")
        for filename, description in key_files.items():
            exists = filename in files
            status = "✅" if exists else "❌"
            logger_ref.info(f"    {status} {filename}: {description}")
    except Exception as e:
        logger_ref.error(f"无法读取checkpoint目录: {e}")

class PeriodicStatusReporter:
    def __init__(self, output_dir, report_interval=100):
        self.output_dir = output_dir
        self.report_interval = report_interval
        self.status_log_path = os.path.join(output_dir, "training_status.txt")
        self._logger = logging.getLogger(__name__ + ".PeriodicStatusReporter") # Instance logger


    def report_status(self, step, trainer_state, curriculum_manager=None, experience_buffer=None):
        if step % self.report_interval != 0:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_report = f"\n========================================\n训练状态报告 - 步数 {step}\n时间: {timestamp}\n========================================\n\n📈 训练进度:\n"
        status_report += f"  - 当前步数: {step}\n  - 最大步数: {trainer_state.max_steps if trainer_state.max_steps > 0 else '无限制'}\n"
        if trainer_state.max_steps > 0:
             status_report += f"  - 完成百分比: {(step/trainer_state.max_steps*100):.2f}%\n"
        else:
            status_report += "  - 完成百分比: N/A\n"


        status_report += "\n📚 课程学习状态:"
        if curriculum_manager:
            current_stage = curriculum_manager.current_stage
            total_stages = len(curriculum_manager.curriculum_stages)
            stage_name = "final"
            if current_stage < total_stages:
                stage_name = curriculum_manager.curriculum_stages[current_stage].name
            
            dataset_size = 0
            try:
                dataset_size = len(curriculum_manager.get_current_stage_dataset())
            except Exception: pass

            status_report += f"\n  - 当前阶段: {current_stage}/{total_stages-1}\n  - 阶段名称: {stage_name}\n"
            status_report += f"  - 阶段评估次数: {len(getattr(curriculum_manager, 'stage_performance_history', []))}\n  - 当前阶段数据集大小: {dataset_size}"
        else:
            status_report += "\n  - 课程学习: 未启用"

        status_report += f"\n\n🔄 经验回放状态:"
        if experience_buffer:
            buffer_stats = experience_buffer.get_stats()
            status_report += f"\n  - 缓存大小: {buffer_stats['size']}/{experience_buffer.max_size}\n  - 平均奖励: {buffer_stats['mean_reward']:.2f}\n  - 最高奖励: {buffer_stats['max_reward']:.2f}"
        else:
            status_report += "\n  - 经验回放: 未启用"

        if trainer_state.log_history:
            recent_loss = trainer_state.log_history[-1].get('loss', trainer_state.log_history[-1].get('train_loss', 'N/A'))
            status_report += f"\n\n📊 最近指标:\n  - 训练损失: {recent_loss}\n  - 学习率: {trainer_state.log_history[-1].get('learning_rate', 'N/A')}"
        status_report += f"\n\n========================================\n"
        
        self._logger.info(status_report) # Log to console via logger
        if self.output_dir and os.path.isdir(self.output_dir): # Ensure output_dir is valid
            with open(self.status_log_path, 'a', encoding='utf-8') as f:
                f.write(status_report + "\n")
        else:
            self._logger.warning(f"PeriodicStatusReporter: Output directory '{self.output_dir}' is not valid. Status report not saved to file.")


class DetailedRewardCallback(TrainerCallback):
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        self.reward_history = []
        self._logger = logging.getLogger(__name__ + ".DetailedRewardCallback") # Instance logger
        os.makedirs(self.output_dir, exist_ok=True) # Ensure dir exists

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or args.local_rank > 0: # Only log on main process
            return

        current_step = getattr(state, 'global_step', 0) or 0
        reward_metrics = {}
        for key, value in logs.items():
            if 'reward' in key.lower() and isinstance(value, (int, float)):
                reward_metrics[key] = value

        if reward_metrics:
            self.reward_history.append({'step': current_step, 'metrics': reward_metrics})
            self._logger.info(f"🎯 步数 {current_step}: 奖励指标 = {reward_metrics}")
            if current_step % 100 == 0:
                self._save_reward_history()

    def _save_reward_history(self):
        try:
            reward_history_path = os.path.join(self.output_dir, "reward_history.json")
            with open(reward_history_path, "w", encoding="utf-8") as f:
                json.dump(self.reward_history, f, indent=2)
        except Exception as e_save:
            self._logger.warning(f"保存奖励历史失败: {e_save}")


def main():
    # These variables will be initialized after logger is setup or within the try block
    curriculum_manager = None
    experience_buffer = None
    wandb_callback = None # Will be initialized after logger and other configs
    model = None
    tokenizer = None
    dataset = None
    trainer = None
    dataset_for_trainer = None
    callbacks_list = [] # Will be populated after logger and other configs
    script_cfg = None # To satisfy finally block if error occurs very early
    logger = None # Initialize logger to None first

    # --- Use basic print for very early messages before logger is configured ---
    try:
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        env_cfg, script_cfg_args, reward_cfg_args, grpo_cfg_args = parser.parse_args_into_dataclasses()
        
        # Make them accessible in the broader scope of main, especially for finally block
        script_cfg = script_cfg_args
        grpo_cfg = grpo_cfg_args
        env_cfg_global = env_cfg # if needed globally by callbacks
        reward_cfg_global = reward_cfg_args # if needed globally
        
        # --- Determine output directory and run name (needed for log file path) ---
        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
        if not run_specific_name_from_env:
            print("[PRE-LOG WARNING] WANDB_RUN_NAME environment variable not set. Generating run name from timestamp.")
            run_specific_name_from_env = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)
        env_cfg.wandb_run_name = sanitized_run_name # Set run name in env_cfg for W&B callback

        if not script_cfg.output_dir_base or not isinstance(script_cfg.output_dir_base, str):
            print(f"[PRE-LOG ERROR] ScriptConfig.output_dir_base ('{script_cfg.output_dir_base}') is invalid. Using './grpo_runs'.")
            script_cfg.output_dir_base = "./grpo_runs"

        resume_checkpoint_path_arg = grpo_cfg.resume_from_checkpoint
        is_resuming = (
            resume_checkpoint_path_arg and
            isinstance(resume_checkpoint_path_arg, str) and
            os.path.isdir(resume_checkpoint_path_arg)
        )

        if is_resuming:
            actual_output_dir = os.path.dirname(resume_checkpoint_path_arg)
            # If resuming, the run name for W&B should ideally be the original one.
            # This assumes the directory name IS the original W&B run name.
            sanitized_run_name = os.path.basename(actual_output_dir)
            env_cfg.wandb_run_name = sanitized_run_name # Override for W&B if resuming
            print(f"[PRE-LOG INFO] 🔄 断续训练模式. 使用原始输出目录: {actual_output_dir}. W&B run name set to: {sanitized_run_name}")
        else:
            actual_output_dir = os.path.join(script_cfg.output_dir_base, sanitized_run_name)

        if grpo_cfg.local_rank <= 0: # Create dir only on main process
            os.makedirs(actual_output_dir, exist_ok=True)
            print(f"[PRE-LOG INFO] Actual output directory for this run: {actual_output_dir}")

        grpo_cfg.output_dir = actual_output_dir
        script_cfg.output_dir = actual_output_dir

        # --- Initialize the main logger ---
        log_file_path = os.path.join(actual_output_dir, "enhanced_training_log.txt")
        log_handlers = [logging.StreamHandler(sys.stdout)]
        if grpo_cfg.local_rank <= 0:
            log_mode = "a" if is_resuming else "w"
            file_handler = logging.FileHandler(log_file_path, mode=log_mode, encoding='utf-8')
            log_handlers.append(file_handler)

        logging.basicConfig(
            level=grpo_cfg.get_process_log_level(),
            format=f"[RANK {grpo_cfg.local_rank:02d}] %(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            handlers=log_handlers,
            force=True,
        )
        logger = logging.getLogger(__name__) # Main logger is now initialized
        # --- Logger initialization complete ---

        # 🚀 快速优化
        script_cfg.max_steps = max(getattr(script_cfg, 'max_steps', 300), 300)
        grpo_cfg.learning_rate = 1e-5
        grpo_cfg.eval_steps = 2
        logger.info(f"🔧 快速优化: 最大步数={script_cfg.max_steps}, 学习率={grpo_cfg.learning_rate}")
        def setup_enhanced_debugging_local(script_cfg_local, grpo_cfg_local, curriculum_manager_local, experience_buffer_local, env_cfg_local, reward_cfg_local, is_resuming_local, run_name_local):
            # Pass logger to this function if it needs it, or it can get it from global scope if careful
            _callbacks_list = [StepLoggingCallback()] # Assuming StepLoggingCallback uses its own logger or global

            if curriculum_manager_local:
                # CurriculumProgressCallback initializes its own logger
                detailed_curriculum_callback = CurriculumProgressCallback(
                    curriculum_manager_local, None, script_cfg_local.output_dir
                )
                _callbacks_list.append(detailed_curriculum_callback)

            # DetailedRewardCallback initializes its own logger
            reward_callback = DetailedRewardCallback(script_cfg_local.output_dir)
            _callbacks_list.append(reward_callback)

            _wandb_callback = None
            if grpo_cfg_local.local_rank <= 0 and "wandb" in grpo_cfg_local.report_to:
                wandb_resume_mode = "allow"
                wandb_run_id_to_use = None # This will be set by WANDB_RUN_ID env var if resuming

                if is_resuming_local:
                    # For W&B, if resuming, WANDB_RUN_ID should be set to the original run's ID.
                    # We've set env_cfg_local.wandb_run_name to the directory name.
                    # If W&B uses run name as ID, this might work. Otherwise, WANDB_RUN_ID must be explicit.
                    # os.environ["WANDB_RUN_ID"] = run_name_local # This might be needed if dir name is W&B ID
                    # os.environ["WANDB_RESUME"] = "must" # or "allow"
                    logger.info(f"🔗 W&B resume mode active. Attempting to resume run: {run_name_local} (or from WANDB_RUN_ID env var if set)")

                # DetailedWandbCallback uses env_cfg_local.wandb_run_name for init name
                _wandb_callback = DetailedWandbCallback(env_cfg_local, script_cfg_local, reward_cfg_local, experience_buffer_local)
                _callbacks_list.append(_wandb_callback)
                logger.info(f"✅ W&B回调已创建 - resume模式: {wandb_resume_mode if is_resuming_local else 'new run'}")
            return _callbacks_list, _wandb_callback
        
        # Continue with the rest of the setup using the initialized logger
        # Assign to global-like variables in main's scope
        reward_cfg = reward_cfg_args # Now use the parsed one

        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME") # Already handled, but kept for consistency if logic moves
        # ... (rest of the run name and output_dir logic is now above logger init) ...

        status_reporter = PeriodicStatusReporter(script_cfg.output_dir, report_interval=50) # Uses its own logger

        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING if grpo_cfg.local_rank <=0 else logging.ERROR)

        logger.info(f"=== ENHANCED GRPO TRAINING STARTED (PID: {os.getpid()}) ===")
        # ... (logging of configs and distributed state)
        logger.info(f"Process rank: {grpo_cfg.process_index}, Local rank: {grpo_cfg.local_rank}, Device: {grpo_cfg.device}, N_GPU: {grpo_cfg.n_gpu}, World Size: {grpo_cfg.world_size}")
        logger.info(f"Distributed Training Type: {grpo_cfg.distributed_state.distributed_type}")
        logger.info(f"Actual Output Dir: {actual_output_dir}")
        logger.info(f"Resume from checkpoint path (from args): {grpo_cfg.resume_from_checkpoint}")
        logger.debug(f"EnvConfig: \n{json.dumps(asdict(env_cfg), indent=2)}") # env_cfg is now env_cfg_global
        logger.debug(f"ScriptConfig: \n{json.dumps(asdict(script_cfg), indent=2)}")
        logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(reward_cfg), indent=2)}")
        logger.debug(f"GRPOConfig (TrainingArguments): \n{grpo_cfg.to_json_string()}")


        if grpo_cfg.world_size > 1:
            logger.info("Waiting for all processes at barrier before proceeding...")
            torch.distributed.barrier()
            logger.info("All processes passed barrier.")

        if script_cfg.enable_experience_replay:
            experience_buffer = ExperienceBuffer(max_size=script_cfg.experience_buffer_size)
            logger.info(f"Experience buffer initialized (size: {script_cfg.experience_buffer_size}).")
            if is_resuming: # Use the already defined is_resuming
                buffer_state_path = os.path.join(resume_checkpoint_path_arg, "enhanced_experience_buffer_state.json") # use resume_checkpoint_path_arg
                if os.path.exists(buffer_state_path):
                    try:
                        logger.info(f"Attempting to load experience buffer state from: {buffer_state_path}")
                        with open(buffer_state_path, "r", encoding="utf-8") as f:
                            state_data = json.load(f)
                        experience_buffer.load_buffer_state(state_data)
                    except Exception as e_load_buffer:
                        logger.error(f"Failed to load experience buffer state from {buffer_state_path}: {e_load_buffer}")
                else:
                    logger.warning(f"Experience buffer state file not found at {buffer_state_path}. Starting with an empty buffer.")
            elif grpo_cfg.resume_from_checkpoint: # resume_from_checkpoint was set but not a valid dir for is_resuming
                 logger.warning(f"grpo_cfg.resume_from_checkpoint ('{grpo_cfg.resume_from_checkpoint}') is not a valid directory. Cannot load experience buffer state.")


        quant_config_dict = {
            "load_in_4bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16 if grpo_cfg.bf16 else torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
        quantization_config_arg = None
        if quant_config_dict.get("load_in_4bit", False):
            try:
                import bitsandbytes
                if torch.cuda.is_available():
                    quantization_config_arg = BitsAndBytesConfig(**quant_config_dict)
                    logger.info("BitsAndBytes quantization will be used.")
                else:
                    logger.warning("CUDA not available, BitsAndBytes quantization disabled.")
            except ImportError:
                logger.warning("bitsandbytes not installed, quantization disabled.")
        else:
            logger.info("🔧 量化已禁用 (load_in_4bit=False)")

        model_dtype = torch.bfloat16 if grpo_cfg.bf16 else (torch.float16 if grpo_cfg.fp16 else torch.float32)
        logger.info(f"Loading base model from: {script_cfg.model_name_or_path} with dtype: {model_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            script_cfg.model_name_or_path,
            quantization_config=quantization_config_arg,
            device_map="auto" if grpo_cfg.world_size == 1 and torch.cuda.is_available() and not grpo_cfg.fsdp else None,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            use_cache=False if grpo_cfg.gradient_checkpointing else True,
            cache_dir=script_cfg.cache_dir
        )
        logger.info("Base model loaded.")

        if quantization_config_arg:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grpo_cfg.gradient_checkpointing)
            logger.info(f"Base model prepared for k-bit training. Grad checkpointing: {grpo_cfg.gradient_checkpointing}.")
        elif grpo_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info(f"Gradient checkpointing enabled for non-k-bit model.")

        tokenizer_load_path = script_cfg.model_name_or_path
        # ... (tokenizer loading logic)
        if script_cfg.stage1_adapter_path and os.path.isdir(script_cfg.stage1_adapter_path):
            potential_tokenizer_path = script_cfg.stage1_adapter_path
            if os.path.exists(os.path.join(potential_tokenizer_path, "tokenizer_config.json")):
                tokenizer_load_path = potential_tokenizer_path
                logger.info(f"Loading tokenizer from stage1_adapter_path: {tokenizer_load_path}")
            else:
                logger.warning(f"tokenizer_config.json not found in stage1_adapter_path, using base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path, trust_remote_code=True, use_fast=True, cache_dir=script_cfg.cache_dir
        )
        logger.info(f"Tokenizer loaded from: {tokenizer_load_path}")


        logger.info("🔧 应用Qwen3兼容性修复...")
        model, tokenizer = Qwen3CompatibilityFixer.fix_generation_config(model, tokenizer) # Assumes Qwen3CompatibilityFixer is defined
        logger.info("🔧 设置PEFT适配器...")

        peft_config_stage2 = LoraConfig(
            r=script_cfg.lora_rank, lora_alpha=script_cfg.lora_alpha,
            lora_dropout=script_cfg.lora_dropout, target_modules=script_cfg.lora_target_modules,
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        is_model_peft_already = isinstance(model, PeftModel)
        logger.info(f"模型当前PEFT状态: {is_model_peft_already}")
        is_quantized = getattr(model, 'is_quantized', False) or hasattr(model, 'hf_quantizer')
        logger.info(f"模型量化状态: {is_quantized}")
        peft_applied_successfully = False

        # Use is_resuming and resume_checkpoint_path_arg for clarity
        if is_resuming: # Check if resuming from a checkpoint directory
            logger.info(f"🔄 检测到从checkpoint恢复: {resume_checkpoint_path_arg}")
            peft_files_in_checkpoint = ['adapter_config.json', 'adapter_model.bin', 'adapter_model.safetensors']
            has_peft_in_checkpoint = any(os.path.exists(os.path.join(resume_checkpoint_path_arg, pf)) for pf in peft_files_in_checkpoint)
            logger.info(f"Checkpoint中是否包含PEFT文件: {has_peft_in_checkpoint}")

            if has_peft_in_checkpoint:
                try:
                    logger.info("🔄 从checkpoint加载PEFT适配器...")
                    if not is_model_peft_already: # Only load if not already a PeftModel
                        model = PeftModel.from_pretrained(model, resume_checkpoint_path_arg, is_trainable=True)
                        logger.info("✅ 成功从checkpoint恢复PEFT适配器")
                    else:
                         logger.info("✅ 模型已经是PEFT模型，checkpoint恢复将由Trainer处理PEFT加载")
                    peft_applied_successfully = True
                except Exception as e_peft_resume:
                    logger.error(f"❌ 从checkpoint恢复PEFT适配器失败: {e_peft_resume}")
                    peft_applied_successfully = False # Fall through to create new or load stage1
            else: # No PEFT files in checkpoint, but resuming. This means base model was saved.
                logger.warning("⚠️ Checkpoint中未发现PEFT适配器文件。如果这是PEFT训练的checkpoint，可能有问题。将尝试创建新适配器或加载stage1。")
                # This case might imply that the checkpoint is for the base model, not an adapter.
                # If we intend to continue PEFT training, we'll need to add an adapter.
                peft_applied_successfully = False


        # If not successfully resumed PEFT from checkpoint, try loading stage1 or creating new
        if not peft_applied_successfully:
            if script_cfg.stage1_adapter_path and os.path.isdir(script_cfg.stage1_adapter_path):
                logger.info(f"📂 加载Stage 1适配器: {script_cfg.stage1_adapter_path}")
                if is_model_peft_already: # Should not happen if peft_applied_successfully is False due to resume
                    logger.warning("⚠️ 模型已经是PeftModel，但从checkpoint加载PEFT未成功。将跳过Stage 1适配器加载，这可能不符合预期。")
                    # peft_applied_successfully = True # It's already a PEFT model, so "applied"
                else:
                    try:
                        model_to_load_adapter = model.model if is_quantized and hasattr(model, 'model') else model # BNB specific
                        model = PeftModel.from_pretrained(
                            model_to_load_adapter,
                            script_cfg.stage1_adapter_path,
                            is_trainable=True,
                            # device_map="auto" # only if model is on auto already and quantized
                        )
                        logger.info("✅ 成功加载Stage 1 LoRA适配器")
                        peft_applied_successfully = True
                    except Exception as e_peft_load:
                        logger.error(f"❌ 加载Stage 1 PEFT适配器失败: {e_peft_load}")
                        # Fall through to create new adapter
            
            # If still no PEFT model (neither resumed nor stage1 loaded)
            if not peft_applied_successfully:
                logger.info("🆕 创建新的PEFT适配器...")
                try:
                    if is_quantized and not getattr(model, "_hf_peft_config_loaded", False): # BNB specific condition
                         # model already prepared by prepare_model_for_kbit_training if quantization_config_arg
                         pass
                    model = get_peft_model(model, peft_config_stage2)
                    logger.info("✅ 成功创建新的PEFT适配器")
                    peft_applied_successfully = True
                except Exception as e_peft_create:
                    logger.error(f"❌ 创建PEFT适配器失败: {e_peft_create}", exc_info=True)
                    sys.exit(1)


        final_is_peft = isinstance(model, PeftModel)
        logger.info(f"🎯 最终模型PEFT状态: {final_is_peft}")
        if final_is_peft:
            logger.info("📊 可训练参数统计:")
            try: model.print_trainable_parameters()
            except Exception as e_print:
                logger.warning(f"无法打印可训练参数: {e_print}")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}, 可训练比例: {100 * trainable_params / total_params:.2f}%")
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                logger.error("❌ 没有找到可训练参数！这将导致训练失败")
                # Attempt to fix (this might be specific to how PEFT model was loaded/created)
                try:
                    model.train() # Ensure model is in training mode
                    # If there's a specific adapter name, ensure it's active
                    if hasattr(model, 'active_adapter') and model.active_adapter:
                         model.set_adapter(model.active_adapter)
                    elif hasattr(model, 'peft_config'): # Iterate and try to set adapters
                        for adapter_name in model.peft_config.keys():
                            logger.info(f"Attempting to set adapter '{adapter_name}' for training.")
                            model.set_adapter(adapter_name) # This might enable gradients
                    
                    # Re-check trainable parameters
                    trainable_params_after_fix = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.info(f"修复后可训练参数: {trainable_params_after_fix:,}")
                    if trainable_params_after_fix == 0:
                        logger.error("❌ 仍然没有可训练参数，无法继续训练")
                        sys.exit(1)
                    else:
                        logger.info("✅ 成功启用可训练参数")
                except Exception as e_enable_train:
                    logger.error(f"❌ 启用适配器训练模式或设置适配器失败: {e_enable_train}", exc_info=True)
                    sys.exit(1)
        else: # Should not happen if logic above is correct
            logger.error("❌ 最终模型不是PeftModel，PEFT配置可能完全失败，训练无法继续")
            sys.exit(1)


        try:
            effective_dataset_base_path = script_cfg.dataset_base_path
            if effective_dataset_base_path is None or not str(effective_dataset_base_path).strip():
                effective_dataset_base_path = os.path.dirname(os.path.abspath(script_cfg.dataset_path))
                logger.info(f"dataset_base_path not provided or empty, derived base path from dataset_path: {effective_dataset_base_path}")
            else:
                effective_dataset_base_path = os.path.abspath(effective_dataset_base_path)
                logger.info(f"Using dataset_base_path from args: {effective_dataset_base_path}")

            dataset_raw = load_dataset("json", data_files=script_cfg.dataset_path, split="train", cache_dir=script_cfg.cache_dir)
            logger.info(f"Raw dataset loaded: {len(dataset_raw)} rows. Columns: {dataset_raw.column_names}")
            if dataset_raw is None or len(dataset_raw) == 0:
                logger.error(f"load_dataset returned None or empty for path: {script_cfg.dataset_path}. Exiting.")
                sys.exit(1)
            
            # Using qwen3_dataset_processing_pipeline as per original code
            # Ensure it uses logger correctly if it logs internally
            dataset_dir_for_proc = os.path.dirname(os.path.abspath(script_cfg.dataset_path))
            logger.info(f"Dataset directory for processing: {dataset_dir_for_proc}")
            # Pass the initialized logger to functions that need it
            dataset = qwen3_dataset_processing_pipeline(dataset_raw, dataset_dir_for_proc, script_cfg) # Pass logger
            del dataset_raw; gc.collect()
            
            dataset = enhance_dataset_with_level_and_complexity(dataset, logger) # Pass logger
            logger.info(f"Final processed dataset: {len(dataset)} rows. Columns: {dataset.column_names}")
            if len(dataset) > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"First processed example:\n{json.dumps(dataset[0], indent=2)}")

        except Exception as e_dataset:
            logger.error(f"Dataset loading/processing error: {e_dataset}", exc_info=True)
            sys.exit(1)

        if not dataset or len(dataset) == 0:
            logger.error("Dataset is empty after processing. Exiting.")
            sys.exit(1)

        if not validate_dataset_for_curriculum(dataset, script_cfg, logger): # Pass logger
            logger.error("Dataset validation for curriculum failed. Please check data format. Exiting.")
            sys.exit(1)

        try:
            curriculum_manager = setup_curriculum_manager(script_cfg, dataset, logger) # Pass logger
            if curriculum_manager:
                logger.info(f"✅ 课程学习已启用. 当前阶段: {curriculum_manager.current_stage}, 总阶段数: {len(curriculum_manager.curriculum_stages)}")
                for i, stage_conf in enumerate(curriculum_manager.curriculum_stages):
                    logger.info(f"    阶段{i}: {stage_conf.name} | 等级: {stage_conf.dataset_levels} | 复杂度: {stage_conf.complexity_range}")
                current_stage_dataset_temp = curriculum_manager.get_current_stage_dataset()
                logger.info(f"📊 当前阶段数据集大小: {len(current_stage_dataset_temp)}")
            else:
                logger.warning("⚠️ 课程学习未启用，使用全部数据集")
        except Exception as e_curriculum:
            logger.error(f"❌ 课程管理器创建失败: {e_curriculum}", exc_info=True)
            curriculum_manager = None # Ensure it's None if setup fails

        if curriculum_manager and is_resuming:
            curriculum_state_path = os.path.join(resume_checkpoint_path_arg, "enhanced_curriculum_state.json")
            if os.path.exists(curriculum_state_path):
                try:
                    logger.info(f"Attempting to load curriculum state from: {curriculum_state_path}")
                    with open(curriculum_state_path, "r", encoding="utf-8") as f:
                        state_data_curr = json.load(f)
                    curriculum_manager.load_curriculum_state(state_data_curr)
                except Exception as e_load_curr:
                    logger.error(f"Failed to load curriculum state from {curriculum_state_path}: {e_load_curr}")
            else:
                logger.warning(f"Curriculum state file not found at {curriculum_state_path}. Curriculum will start from initial stage.")
        elif curriculum_manager and grpo_cfg.resume_from_checkpoint: # resume_from_checkpoint was set but not a valid dir
             logger.warning(f"grpo_cfg.resume_from_checkpoint ('{grpo_cfg.resume_from_checkpoint}') is not a valid directory. Cannot load curriculum state.")


        dataset_for_trainer = dataset
        if curriculum_manager:
            dataset_for_trainer = curriculum_manager.get_current_stage_dataset()
            current_stage_idx_trainer = curriculum_manager.current_stage
            current_stage_obj_trainer = curriculum_manager.curriculum_stages[current_stage_idx_trainer] if current_stage_idx_trainer < len(curriculum_manager.curriculum_stages) else None
            current_stage_name_trainer = current_stage_obj_trainer.name if current_stage_obj_trainer else "Unknown/Final"
            logger.info(f"🎯 Curriculum learning active. Using {len(dataset_for_trainer)} examples from stage {current_stage_idx_trainer} ('{current_stage_name_trainer}').")
        else:
            logger.info("Curriculum learning disabled, using full dataset for trainer.")

        if not dataset_for_trainer or len(dataset_for_trainer) == 0:
            logger.error(f"Dataset for trainer is empty (size: {len(dataset_for_trainer) if dataset_for_trainer else 0}). Check curriculum logic or dataset. Exiting.")
            sys.exit(1)
        
        # Call setup_enhanced_debugging_local to get callbacks and wandb_callback instance
        callbacks_list, wandb_callback = setup_enhanced_debugging_local(script_cfg, grpo_cfg, curriculum_manager, experience_buffer, env_cfg, reward_cfg, is_resuming, sanitized_run_name)


        # Existing callbacks_list might have been populated by setup_enhanced_debugging_local
        # Add other callbacks
        callbacks_list_enhanced = list(callbacks_list) # Start with what was returned

        stability_monitor = RewardStabilityMonitor(script_cfg.output_dir) # Uses its own logger
        callbacks_list_enhanced.append(stability_monitor)
        logger.info("✅ 添加奖励稳定性监控回调")

        if curriculum_manager:
            # EnhancedCurriculumDebugCallback uses its own logger
            enhanced_curriculum_cb = EnhancedCurriculumDebugCallback(curriculum_manager, None, script_cfg.output_dir)
            callbacks_list_enhanced.append(enhanced_curriculum_cb)
            logger.info("✅ 添加增强的课程学习调试回调")
        
        # CustomStatePersistenceCallback uses its own logger
        custom_state_pers_cb = CustomStatePersistenceCallback(curriculum_manager, experience_buffer, script_cfg)
        callbacks_list_enhanced.append(custom_state_pers_cb)
        logger.info("✅ 添加自定义状态持久化回调")


        sample_dataset_for_inf_cb = dataset.select(range(min(len(dataset), script_cfg.callback_num_samples * 5))) if len(dataset) > 0 else None
        if sample_dataset_for_inf_cb and len(sample_dataset_for_inf_cb) > 0:
            # DetailedInferenceCallback uses its own logger
            detailed_inference_cb = DetailedInferenceCallback(
                tokenizer=tokenizer, eval_dataset=sample_dataset_for_inf_cb,
                num_samples=script_cfg.callback_num_samples, eval_every_n_steps=script_cfg.callback_eval_every_n_steps,
                max_new_tokens=grpo_cfg.max_completion_length, max_seq_length=script_cfg.max_seq_length,
                experience_buffer=experience_buffer, output_dir=script_cfg.output_dir
            )
            callbacks_list_enhanced.append(detailed_inference_cb)
            logger.info(f"DetailedInferenceCallback initialized with {len(sample_dataset_for_inf_cb)} samples for periodic eval.")
        else:
            logger.warning("DetailedInferenceCallback will not run due to insufficient sample data from the full dataset.")

        # The CurriculumProgressCallback might have been added by setup_enhanced_debugging_local already.
        # If not, or if you want to be sure it's there and distinct from EnhancedCurriculumDebugCallback:
        added_curriculum_progress_cb = False
        for cb_item in callbacks_list_enhanced:
            if isinstance(cb_item, CurriculumProgressCallback):
                added_curriculum_progress_cb = True; break
        if not added_curriculum_progress_cb and curriculum_manager:
            # CurriculumProgressCallback uses its own logger
            curriculum_progress_cb_instance = CurriculumProgressCallback(curriculum_manager, None, script_cfg.output_dir)
            callbacks_list_enhanced.append(curriculum_progress_cb_instance)
            logger.info("✅ 添加 (独立的) CurriculumProgressCallback")


        callbacks_list = callbacks_list_enhanced # Use the final list
        logger.info(f"总共配置了 {len(callbacks_list)} 个回调。")


        trainer_instance_for_reward_func_ref = [None] # Use a list to pass by reference for mutable access in closure

        def enhanced_batch_reward_calculator(
            prompts: List[str], completions: List[str], testbench_path: List[str],
            expected_total_tests: List[int], reference_verilog_path: List[str],
            original_enhanced_prompt: Optional[List[str]] = None, **kwargs: Any
        ) -> Tuple[List[float], Dict[str, Any]]:
            batch_rewards_final_scaled: List[float] = []
            batch_all_unscaled_components: List[Dict[str, float]] = []
            batch_all_funnel_metrics: List[Dict[str, Any]] = []
            num_items_in_batch = len(prompts)

            # These are expected to be passed in kwargs by GRPOTrainer or the wrapper
            current_reward_cfg = kwargs.get('reward_config_obj')
            current_script_cfg = kwargs.get('script_config_obj') # For output_dir_for_debug and experience replay toggle
            current_experience_buffer = kwargs.get('experience_buffer_obj')
            training_step = kwargs.get('training_step', 0)
            # wandb_cb_from_kwargs = kwargs.get('wandb_callback') # wandb_callback is now in main's scope
            output_dir_for_debug_val = current_script_cfg.output_dir if current_script_cfg else None
            
            # Use the logger from the outer scope (main's logger)
            # No need to pass logger_ref if this function is defined within main where logger is available
            # logger_ref_for_reward = kwargs.get('logger_ref') 
            # if logger_ref_for_reward is None: logger_ref_for_reward = logger # Fallback to main's logger

            if current_reward_cfg is None:
                logger.critical("EnhancedBatchRewardCalculator: 'reward_config_obj' 未在kwargs中提供! 返回默认惩罚。")
                return [-10.0] * num_items_in_batch, {}
            if not original_enhanced_prompt:
                 logger.error("EnhancedBatchRewardCalculator: 'original_enhanced_prompt' is None! 奖励计算可能不准确。")
                 # Potentially return error or try to recover, for now, it will proceed and might fail in single_prompt calc

            # ... (rest of input list length checks using logger)
            expected_lengths = {
                "prompts": len(prompts), "completions": len(completions), "testbench_path": len(testbench_path),
                "expected_total_tests": len(expected_total_tests), "reference_verilog_path": len(reference_verilog_path),
            }
            if original_enhanced_prompt is not None:
                expected_lengths["original_enhanced_prompt"] = len(original_enhanced_prompt)
            
            if len(set(expected_lengths.values())) > 1:
                mismatched_lengths_str = ", ".join([f"{k}:{v}" for k, v in expected_lengths.items()])
                logger.error(f"Enhanced batch reward calculator: Mismatch in input list lengths. Details: {mismatched_lengths_str}")
                return [current_reward_cfg.get_scaled_reward(current_reward_cfg.compilation_failure * 3, training_step)] * len(completions), {}

            if num_items_in_batch == 0:
                logger.warning("Enhanced batch reward calculator: Received an empty batch.")
                return [], {}


            for i in range(num_items_in_batch):
                qwen_formatted_prompt_for_buffer = prompts[i]
                current_completion_str = completions[i]
                current_tb = testbench_path[i]
                current_ett = expected_total_tests[i]
                current_ref_v = reference_verilog_path[i]
                
                prompt_for_reward_calculation = ""
                if original_enhanced_prompt and i < len(original_enhanced_prompt) and isinstance(original_enhanced_prompt[i], str) and original_enhanced_prompt[i].strip():
                    prompt_for_reward_calculation = original_enhanced_prompt[i]
                else: # Fallback logic as before
                    logger.warning(f"Item {i}: 'original_enhanced_prompt' not available/invalid. Using Qwen-formatted prompt for reward. Parsing may be affected.")
                    prompt_for_reward_calculation = qwen_formatted_prompt_for_buffer
                    match_user_content = re.search(r"<\|im_start\|>user\n(.*?)\n?<\|im_end\|>", qwen_formatted_prompt_for_buffer, re.DOTALL)
                    if match_user_content:
                        prompt_for_reward_calculation = match_user_content.group(1).strip()
                        logger.debug(f"Item {i}: Extracted user content from Qwen prompt for reward calculation.")
                    else:
                        logger.warning(f"Item {i}: Could not extract user content from Qwen prompt. Reward parsing may fail with full Qwen prompt.")


                results_list_for_item = calculate_enhanced_rewards_for_single_prompt(
                    prompt_str=prompt_for_reward_calculation,
                    completions_for_this_prompt=[current_completion_str],
                    current_tb_path=current_tb,
                    current_expected_total_from_manifest=current_ett,
                    current_ref_verilog_path=current_ref_v,
                    reward_config=current_reward_cfg,
                    logger_ref=logger, # Pass main logger
                    training_step=training_step,
                    wandb_callback=wandb_callback, # wandb_callback from main's scope
                    output_dir_for_debug=output_dir_for_debug_val
                )
                # ... (processing results_list_for_item and adding to experience_buffer, using logger)
                item_detailed_result = {}
                if results_list_for_item:
                    item_detailed_result = results_list_for_item[0]
                else:
                    logger.error(f"calculate_enhanced_rewards_for_single_prompt returned empty list for item {i}. Assigning penalty.")
                    error_reward_val = current_reward_cfg.get_scaled_reward(current_reward_cfg.compilation_failure * 3, training_step)
                    item_detailed_result = {
                        "final_reward": error_reward_val,
                        "unscaled_components": {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": current_reward_cfg.compilation_failure * 3},
                        "funnel_metrics": {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
                    }
                
                batch_rewards_final_scaled.append(item_detailed_result["final_reward"])
                batch_all_unscaled_components.append(item_detailed_result["unscaled_components"])
                batch_all_funnel_metrics.append(item_detailed_result["funnel_metrics"])
                
                if current_experience_buffer and current_script_cfg and current_script_cfg.enable_experience_replay:
                    try:
                        current_experience_buffer.add_experience(
                            prompt=qwen_formatted_prompt_for_buffer,
                            completion=current_completion_str,
                            reward=item_detailed_result["final_reward"],
                            metadata={
                                "training_step": training_step, "testbench": current_tb,
                                "original_enhanced_prompt_preview": prompt_for_reward_calculation[:100]
                            }
                        )
                    except Exception as e_exp:
                        logger.warning(f"Failed to add experience to buffer for item {i}: {e_exp}", exc_info=True)


            aggregated_metrics_for_wandb = {}
            if num_items_in_batch > 0:
                component_keys = ["functional", "efficiency", "readability", "robustness", "base_compilation"]
                for key in component_keys:
                    values = [comp[key] for comp in batch_all_unscaled_components if key in comp]
                    if values:
                        aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_mean"] = np.mean(values)
                        aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_std"] = np.std(values)
                # ... (funnel metrics aggregation using logger)
                total_completions_in_batch = num_items_in_batch
                successful_extractions = sum(1 for fm in batch_all_funnel_metrics if fm["code_extracted"])
                successful_compilations = sum(1 for fm in batch_all_funnel_metrics if fm["compiled_successfully"])
                simulation_runs = sum(1 for fm in batch_all_funnel_metrics if fm["sim_ran_successfully"])

                aggregated_metrics_for_wandb["generation_funnel/successful_extractions_count"] = successful_extractions
                aggregated_metrics_for_wandb["generation_funnel/successful_extractions_ratio"] = successful_extractions / total_completions_in_batch if total_completions_in_batch > 0 else 0
                aggregated_metrics_for_wandb["generation_funnel/successful_compilations_count"] = successful_compilations
                aggregated_metrics_for_wandb["generation_funnel/successful_compilations_ratio"] = successful_compilations / successful_extractions if successful_extractions > 0 else 0
                aggregated_metrics_for_wandb["generation_funnel/simulation_runs_count"] = simulation_runs
                aggregated_metrics_for_wandb["generation_funnel/simulation_runs_ratio"] = simulation_runs / successful_compilations if successful_compilations > 0 else 0
                passed_tests_values = [fm["passed_tests"] for fm in batch_all_funnel_metrics if fm["sim_ran_successfully"] and fm["passed_tests"] != -1]
                aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_runs"] = np.mean(passed_tests_values) if passed_tests_values else 0


            if batch_rewards_final_scaled and training_step > 0 and training_step % 10 == 0:
                reward_stats = {'mean': np.mean(batch_rewards_final_scaled), 'std': np.std(batch_rewards_final_scaled),
                                'min': np.min(batch_rewards_final_scaled), 'max': np.max(batch_rewards_final_scaled),
                                'median': np.median(batch_rewards_final_scaled)}
                positive_rewards = [r for r in batch_rewards_final_scaled if r > 0]
                success_rate = len(positive_rewards) / len(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0
                high_rewards = [r for r in batch_rewards_final_scaled if r > reward_stats['mean'] + reward_stats['std']]
                low_rewards = [r for r in batch_rewards_final_scaled if r < reward_stats['mean'] - reward_stats['std']]
                logger.info(f"""
                📊 步数 {training_step} 奖励深度分析:
                ├─ 基础统计: Mean={reward_stats['mean']:.4f}, Std={reward_stats['std']:.4f}, Median={reward_stats['median']:.4f}, Range=[{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]
                ├─ 成功分析: PositiveRate={success_rate:.2%} ({len(positive_rewards)}/{len(batch_rewards_final_scaled)}), HighRewards={len(high_rewards)}, LowRewards={len(low_rewards)}
                └─ 批次信息: {len(batch_rewards_final_scaled)} 个样本""")
                if current_experience_buffer:
                    try: logger.info(f"📚 经验缓冲区: {current_experience_buffer.get_stats().get('size',0)} 条经验")
                    except Exception: pass
            if training_step > 0 and training_step % 50 == 0 and current_script_cfg:
                 logger.info(f"""
                🎯 步数 {training_step} 训练里程碑: AvgReward={np.mean(batch_rewards_final_scaled):.4f}, Stability={'稳定' if np.std(batch_rewards_final_scaled) < 2.0 else '波动较大'}
                └─ 预计进度: {training_step}/{current_script_cfg.max_steps if current_script_cfg.max_steps > 0 else 300} ({(training_step/(current_script_cfg.max_steps if current_script_cfg and current_script_cfg.max_steps > 0 else 300)*100):.1f}%)""")

            return batch_rewards_final_scaled, aggregated_metrics_for_wandb

        def reward_func_with_context(*args_reward, **kwargs_reward):
            current_training_step = 0
            # Access trainer_instance_for_reward_func_ref[0] which holds the trainer instance
            if trainer_instance_for_reward_func_ref[0] and hasattr(trainer_instance_for_reward_func_ref[0], 'state') and trainer_instance_for_reward_func_ref[0].state:
                current_training_step = trainer_instance_for_reward_func_ref[0].state.global_step
            else:
                logger.debug("reward_func_with_context: trainer_instance or its state is None. training_step defaults to 0.")

            kwargs_reward['training_step'] = current_training_step
            kwargs_reward['script_config_obj'] = script_cfg # from main's scope
            kwargs_reward['reward_config_obj'] = reward_cfg # from main's scope
            kwargs_reward['experience_buffer_obj'] = experience_buffer # from main's scope
            # kwargs_reward['wandb_callback'] = wandb_callback # from main's scope, passed to single_prompt
            # kwargs_reward['logger_ref'] = logger # Pass main logger to actual calculator if needed, but it's in scope

            if args_reward:
                logger.warning(f"reward_func_with_context received unexpected positional arguments: {args_reward}. Ensure GRPOTrainer passes dataset columns as keyword arguments.")

            rewards_list, aggregated_metrics = enhanced_batch_reward_calculator(**kwargs_reward) # Pass all kwargs

            if wandb_callback and aggregated_metrics: # wandb_callback from main's scope
                try:
                    wandb_callback.log_batch_aggregated_metrics(aggregated_metrics, step=current_training_step)
                except Exception as e_cb_log:
                    logger.error(f"Error calling wandb_callback.log_batch_aggregated_metrics: {e_cb_log}", exc_info=True)
            return rewards_list


        logger.info("Initializing GRPOTrainer...")
        trainer = GRPOTrainer(
            model=model,
            args=grpo_cfg,
            train_dataset=dataset_for_trainer,
            reward_funcs=[reward_func_with_context], # Pass the wrapper
            callbacks=callbacks_list,
            # No tokenizer needed here for GRPO's reward calculation if prompts are already tokenized by GRPO
        )
        trainer_instance_for_reward_func_ref[0] = trainer # Store the trainer instance

        # Set trainer_ref for callbacks that need it
        for cb_item in callbacks_list:
            if hasattr(cb_item, 'trainer_ref') and cb_item.trainer_ref is None:
                cb_item.trainer_ref = trainer
                logger.info(f"Set trainer_ref for callback: {type(cb_item).__name__}")


        logger.info("GRPOTrainer initialized successfully.")
        logger.info(f"=== STARTING GRPO TRAINING (Output: {grpo_cfg.output_dir}) ===")
        logger.info(f"Dataset size for current stage: {len(dataset_for_trainer)} examples")

        actual_resume_path_for_train = grpo_cfg.resume_from_checkpoint # This should be the validated path from earlier logic
        if not is_resuming: # If not truly resuming (e.g., path was invalid), don't pass it
            actual_resume_path_for_train = None
        
        logger.info(f"Final resume_from_checkpoint path for Trainer.train(): {actual_resume_path_for_train if actual_resume_path_for_train else 'No specific path (new run or auto-resume from output_dir if checkpoint exists there)'}")

        train_res = trainer.train(resume_from_checkpoint=actual_resume_path_for_train)

        if grpo_cfg.local_rank <= 0:
            final_model_dir = os.path.join(script_cfg.output_dir, "final_model_adapter")
            logger.info(f"Training finished. Saving final model to {final_model_dir}")
            trainer.save_model(final_model_dir) # Saves adapter if PEFT model
            
            enhanced_artifacts_dir = os.path.join(script_cfg.output_dir, "training_artifacts")
            os.makedirs(enhanced_artifacts_dir, exist_ok=True)
            if experience_buffer and script_cfg.enable_experience_replay:
                buffer_stats = experience_buffer.get_stats()
                with open(os.path.join(enhanced_artifacts_dir, "experience_buffer_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(buffer_stats, f, indent=2)
                logger.info(f"Experience buffer stats saved. Final size: {buffer_stats.get('size', 'N/A')}")
            if curriculum_manager:
                final_stage_idx = curriculum_manager.current_stage
                final_stage_name = "Unknown/Final"
                if final_stage_idx < len(curriculum_manager.curriculum_stages):
                    final_stage_name = curriculum_manager.curriculum_stages[final_stage_idx].name
                curriculum_stats = {
                    "final_stage_idx": final_stage_idx,
                    "final_stage_name": final_stage_name,
                    "stages_completed": final_stage_idx + 1, # Assuming 0-indexed
                    "total_stages": len(curriculum_manager.curriculum_stages),
                    "stage_performance_history": getattr(curriculum_manager, 'stage_performance_history', [])
                }
                with open(os.path.join(enhanced_artifacts_dir, "curriculum_progress.json"), "w", encoding="utf-8") as f:
                    json.dump(curriculum_stats, f, indent=2)
                logger.info(f"Curriculum progress saved. Completed {curriculum_stats['stages_completed']}/{curriculum_stats['total_stages']} stages.")

            metrics = train_res.metrics if hasattr(train_res, 'metrics') else {}
            trainer.log_metrics("train_summary", metrics)
            # trainer.save_metrics("train_summary", os.path.join(enhanced_artifacts_dir, "final_train_metrics.json")) # trainer.save_metrics needs path
            metrics_save_path = os.path.join(enhanced_artifacts_dir, "final_train_metrics.json")
            with open(metrics_save_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Final train metrics saved to {metrics_save_path}")

            trainer.save_state()
            logger.info(f"Training artifacts saved to {enhanced_artifacts_dir}")

            # import wandb # Already imported globally
            if wandb.run is not None:
                current_step_final = trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else 0
                wandb.log({
                    "training/final_status": "completed",
                    "training/total_steps_final": metrics.get("train_steps", current_step_final),
                    "training/final_loss_train": metrics.get("train_loss", 0), # Or find the last logged loss
                }, step = current_step_final)
                if experience_buffer and script_cfg.enable_experience_replay:
                    final_buffer_stats_wandb = experience_buffer.get_stats()
                    wandb.log({f"final_experience_buffer/{k}": v for k,v in final_buffer_stats_wandb.items()}, step=current_step_final)


    except Exception as e_main:
        if logger: # Check if logger was initialized
            logger.error(f"❌ 训练过程中发生主要错误: {e_main}", exc_info=True)
        else:
            print(f"[PRE-LOG CRITICAL] ❌ 训练过程中发生主要错误 (logger 未初始化): {e_main}")
            import traceback
            traceback.print_exc()
        
        # Attempt to log to W&B if it was initialized and this is the main process
        # import wandb # Already imported globally
        if 'grpo_cfg' in locals() and grpo_cfg.local_rank <= 0 and hasattr(wandb, 'run') and wandb.run is not None:
            try:
                wandb.log({"training/final_status": "failed", "training/error_message": str(e_main)})
            except Exception as e_wandb_fail:
                if logger: logger.error(f"Failed to log error to W&B: {e_wandb_fail}")
                else: print(f"[PRE-LOG ERROR] Failed to log error to W&B: {e_wandb_fail}")

        # Clean up resources if possible
        try:
            if 'trainer' in locals() and trainer: del trainer
            if 'model' in locals() and model: del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
        except Exception as e_cleanup_error:
            if logger: logger.warning(f"清理资源时出错: {e_cleanup_error}")
            else: print(f"[PRE-LOG WARNING] 清理资源时出错: {e_cleanup_error}")
        
        # It's good practice to re-raise the exception or exit if it's critical
        sys.exit(1) # Exit with error status

    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            output_dir_final = "unknown"
            if script_cfg and hasattr(script_cfg, 'output_dir'):
                output_dir_final = script_cfg.output_dir
            elif 'actual_output_dir' in locals(): # Fallback if script_cfg not fully set
                output_dir_final = actual_output_dir

            if logger: # Check if logger was initialized
                logger.info(f"=== GRPO TRAINING SCRIPT FINISHED (Output: {output_dir_final}) ===")
            else:
                print(f"[PRE-LOG INFO] === GRPO TRAINING SCRIPT FINISHED (Output: {output_dir_final}) ===")

            # import wandb # Already imported globally
            if 'grpo_cfg' in locals() and hasattr(grpo_cfg, 'local_rank') and grpo_cfg.local_rank <= 0 and hasattr(wandb, 'run') and wandb.run is not None:
                wandb.finish()
        except Exception as e_finally:
            if logger: # Check if logger was initialized
                logger.warning(f"最终清理时出错: {e_finally}")
            else:
                print(f"[PRE-LOG WARNING] 最终清理时出错: {e_finally}")
        
        if logger: logger.info("🧹 清理完成")
        else: print("[PRE-LOG INFO] 🧹 清理完成")


if __name__ == "__main__":
    # Ensure QWEN3 function is correctly referenced or defined if not imported elsewhere
    # For example, if wrap_prompt_for_qwen3 was meant for dataset processing:
    def wrap_prompt_for_qwen(example): # Placeholder if not defined in qwen3_prompt_fix
        # This should be your actual QWEN prompt wrapping logic
        # It seems qwen3_dataset_processing_pipeline should handle this.
        # This definition here is just to avoid NameError if it's missing from imports
        # and was intended to be defined locally for the map function in old dataset pipeline.
        # With qwen3_dataset_processing_pipeline, this might not be needed here.
        if 'prompt' in example:
             example['prompt'] = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        return example

    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()