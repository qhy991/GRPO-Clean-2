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
    wrap_prompt_for_qwen3,
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
            # Attempt to get the type if nt_class is an instance (e.g. np.float32 is an instance of numpy.dtype)
            # However, numpy scalar types like np.float32 are themselves types (e.g. type(np.float32) is numpy.dtype)
            # The list numpy_scalar_types_to_add should ideally contain the type objects directly.
            # For example, np.float32 is <class 'numpy.float32'>.
            # Let's assume the elements in numpy_scalar_types_to_add are already the type objects.
            logger_temp.warning(f"NumPy scalar '{str(nt_class)}' (type: {type(nt_class)}) is not directly a type class, not adding.")

    logger_temp.info(f"Added {added_scalar_types_count} NumPy scalar types to safe_globals_list.")

    # Log the list before adding to torch to help debug if error persists
    # for idx, item in enumerate(safe_globals_list):
    #    logger_temp.debug(f"Item {idx} in safe_globals_list: {str(item)} (Type: {type(item)})")

    if not safe_globals_list:
        logger_temp.warning("safe_globals_list is empty before calling torch.serialization.add_safe_globals.")

    torch.serialization.add_safe_globals(safe_globals_list)
    logger_temp.info(
        f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items."
    )
    # logger_temp.debug(f"Current safe globals content: {[str(g) for g in safe_globals_list]}") # Could be very long

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
import wandb
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training

from utils import (
    extract_module_info,  validate_verilog_code,
    run_iverilog_simulation, validate_and_update_dataset_paths, enhance_prompt_func,
    EnhancedInferenceCallback,  assess_code_quality,DetailedInferenceCallback,
    assess_design_complexity, ExperienceBuffer,  StepLoggingCallback, monitor_advanced_stage_training

)
from config import EnvConfig, ScriptConfig, EnhancedRewardConfig 
from enhanced_curriculum import (
    EnhancedCurriculumManager, CurriculumStageConfig,
    create_default_curriculum_stages, create_custom_curriculum_stages
)
from datetime import datetime

logger = logging.getLogger(__name__)

class DetailedWandbCallback(TrainerCallback):
    """增强的 W&B 日志回调"""
    
    def __init__(self, env_cfg, script_cfg, reward_cfg, experience_buffer=None):
        self.env_cfg = env_cfg
        self.script_cfg = script_cfg 
        self.reward_cfg = reward_cfg
        self.experience_buffer = experience_buffer
        self.step_count = 0
        self.recent_rewards = deque(maxlen=100) # Added
        
    def on_init_end(self, args, state, control, **kwargs):
        if not getattr(self.env_cfg, 'wandb_disable', False):
            import wandb
            wandb.init(
                project=getattr(self.env_cfg, 'wandb_project', 'grpo-training'),
                name=getattr(self.env_cfg, 'wandb_run_name', None),
                config={
                    **asdict(self.script_cfg),
                    **asdict(self.reward_cfg),
                    "trainer_args": args.to_dict()
                }
            )
            logger.info("🚀 W&B 初始化完成")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or getattr(self.env_cfg, 'wandb_disable', False):
            return
            
        try:
            import wandb
            if wandb.run is None:
                return
                
            # 🔧 修复：安全获取global_step
            current_step = getattr(state, 'global_step', 0) or 0
            self.step_count = current_step
            
            # 准备日志数据
            wandb_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    wandb_logs[f"train/{key}"] = value
            
            # 添加经验缓冲区统计信息
            if self.experience_buffer:
                buffer_stats = self.experience_buffer.get_stats()
                for key, value in buffer_stats.items():
                    if isinstance(value, (int, float)):
                        wandb_logs[f"experience_buffer/{key}"] = value
            
            # 记录到W&B
            if wandb_logs:
                wandb.log(wandb_logs, step=current_step)

            if self.recent_rewards:
                try:
                    wandb.log({"reward_distribution": wandb.Histogram(np.array(self.recent_rewards))}, step=current_step)
                    # Optional: Clear recent_rewards after logging histogram if it's preferred to log distribution of rewards *since last log*
                    # self.recent_rewards.clear()
                except Exception as e_hist:
                    logger.warning(f"Failed to log reward histogram to W&B: {e_hist}")
                
        except Exception as e_wandb:
            logger.warning(f"W&B 日志记录失败: {e_wandb}")

    def add_reward(self, reward: float):
        if getattr(self.env_cfg, 'wandb_disable', False) or (hasattr(wandb, 'run') and wandb.run is None):
            return
        self.recent_rewards.append(reward)

    def log_reward_components(self, reward_components: Dict[str, float]):
        """Log detailed reward component breakdown."""
        try:
            # import wandb # Already imported if needed
            if hasattr(wandb, 'run') and wandb.run is not None and not getattr(self.env_cfg, 'wandb_disable', False):
                wandb.log({f"reward_components/{k}": v for k, v in reward_components.items()})
        except Exception as e:
            logger.warning(f"Failed to log reward components: {e}")

    # log_reward method is now effectively replaced by add_reward for histogram purposes.
    # If individual reward logging per completion by this callback is still desired (outside GRPOTrainer's own logging),
    # it would need a different name or logic. For now, assuming add_reward covers the primary need.
    # def log_reward(self, reward: float):
    #     """Log individual reward values."""
    #     try:
    #         # import wandb
    #         if hasattr(wandb, 'run') and wandb.run is not None:
    #             wandb.log({"reward": reward}) # This might be redundant if GRPOTrainer logs rewards
    #     except Exception as e:
    #         logger.warning(f"Failed to log reward: {e}")

    def log_batch_aggregated_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Logs batch-aggregated metrics (unscaled rewards, funnel stats) to W&B."""
        if getattr(self.env_cfg, 'wandb_disable', False) or not metrics:
            return

        try:
            import wandb
            if wandb.run is None:
                logger.warning("W&B run not initialized. Skipping log_batch_aggregated_metrics.")
                return

            # Determine the step for logging
            log_step = step
            if log_step is None: # Fallback if step not directly provided
                # Try to get from internal step_count or default to None (W&B might auto-increment)
                log_step = self.step_count if hasattr(self, 'step_count') and self.step_count > 0 else None

            # Filter out non-numeric or problematic values before logging
            sanitized_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    sanitized_metrics[key] = value
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)): # Handle numpy types
                    if not (np.isnan(value) or np.isinf(value)):
                         sanitized_metrics[key] = float(value) # Convert to standard Python float
                # else: skip non-numeric or problematic values like NaN/Inf

            if sanitized_metrics:
                if log_step is not None:
                    wandb.log(sanitized_metrics, step=log_step)
                else: # If step is still None, log without explicit step
                    wandb.log(sanitized_metrics)
                logger.debug(f"Logged batch-aggregated metrics to W&B (step {log_step if log_step is not None else 'auto'}): {list(sanitized_metrics.keys())}")
            else:
                logger.debug("No valid batch-aggregated metrics to log after sanitization.")

        except ImportError:
            logger.warning("W&B module not found. Cannot log batch_aggregated_metrics.")
        except Exception as e_wandb_agg:
            logger.error(f"Error logging batch-aggregated metrics to W&B: {e_wandb_agg}", exc_info=True)


class CustomStatePersistenceCallback(TrainerCallback):
    def __init__(self, 
                 curriculum_manager: Optional[EnhancedCurriculumManager], 
                 experience_buffer: Optional[ExperienceBuffer],
                 script_cfg: ScriptConfig): 
        self.curriculum_manager = curriculum_manager
        self.experience_buffer = experience_buffer
        self.script_cfg = script_cfg 

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 🔧 修复：安全获取global_step
        current_step = getattr(state, 'global_step', 0) or 0
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{current_step}")
        os.makedirs(checkpoint_folder, exist_ok=True) 
        logger.debug(f"CustomStatePersistenceCallback: 正在向 checkpoint 文件夹保存自定义状态: {checkpoint_folder}")

        if self.curriculum_manager:
            curriculum_state = self.curriculum_manager.get_curriculum_state()
            file_path = os.path.join(checkpoint_folder, "enhanced_curriculum_state.json")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(curriculum_state, f, indent=2)
                logger.info(f"已保存课程学习状态到: {file_path}")
            except Exception as e:
                logger.error(f"保存课程学习状态失败 ({file_path}): {e}")

        if self.experience_buffer and self.script_cfg.enable_experience_replay: 
            buffer_state = self.experience_buffer.get_buffer_state()
            file_path = os.path.join(checkpoint_folder, "enhanced_experience_buffer_state.json")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(buffer_state, f, indent=2) 
                logger.info(f"已保存经验回放池状态到: {file_path}")
            except TypeError as te:
                logger.warning(f"经验回放池内容可能不是完全 JSON 可序列化，保存尝试失败 ({file_path}): {te}。请检查经验条目格式。")
            except Exception as e:
                logger.error(f"保存经验回放池状态失败 ({file_path}): {e}")


def validate_dataset_for_curriculum(dataset: Dataset, script_cfg: ScriptConfig) -> bool:
    if dataset is None:
        logger.error("Dataset is None, cannot validate for curriculum learning")
        return False
    
    if len(dataset) == 0:
        logger.error("Empty dataset provided for curriculum learning")
        return False
    
    required_fields = ['prompt', 'testbench_path', 'expected_total_tests', 'reference_verilog_path']
    missing_fields = [field for field in required_fields if field not in dataset.column_names]
    
    if missing_fields:
        logger.error(f"Dataset missing required fields for training: {missing_fields}")
        return False
    
    curriculum_fields = ['level', 'complexity_score']
    missing_curriculum_fields = [field for field in curriculum_fields if field not in dataset.column_names]
    
    if missing_curriculum_fields:
        logger.warning(f"Dataset missing curriculum learning fields: {missing_curriculum_fields}")
        logger.warning("Curriculum learning may not work optimally")
    
    if len(dataset) > 0: 
        first_example = dataset[0]
        if 'level' in first_example:
            levels = [ex.get('level') for ex in dataset if ex] 
            unique_levels = set(l for l in levels if l is not None) 
            expected_levels = {'basic', 'intermediate', 'advanced', 'expert'}
            
            if not unique_levels.intersection(expected_levels) and unique_levels: 
                logger.warning(f"Unusual level values found: {unique_levels}")
                logger.warning(f"Expected values: {expected_levels}")
        
        if 'complexity_score' in first_example:
            complexities = [ex.get('complexity_score', 0) for ex in dataset if ex]
            complexities = [c for c in complexities if isinstance(c, (int, float))]
            
            if complexities:
                min_complexity = min(complexities)
                max_complexity = max(complexities)
                
                if min_complexity < 0 or max_complexity > 10:
                    logger.warning(f"Complexity scores outside expected range [0-10]: {min_complexity:.2f} - {max_complexity:.2f}")
    
    logger.info("Dataset validation for curriculum learning completed")
    return True


def setup_curriculum_manager(script_cfg: ScriptConfig, dataset: Dataset) -> Optional[EnhancedCurriculumManager]:
    if not script_cfg.enable_curriculum:
        logger.info("Curriculum learning disabled.")
        return None
    
    if dataset is None:
        logger.error("Cannot setup curriculum manager with None dataset")
        return None
    
    has_level_info = False
    if len(dataset) > 0:
        first_example = dataset[0]
        # Ensure 'level' exists and is not None. Also handle cases where it might be an empty string.
        has_level_info = 'level' in first_example and first_example['level'] is not None and str(first_example['level']).strip() != ""
    
    if not has_level_info:
        logger.warning("Dataset does not contain valid 'level' field data. Curriculum learning will use complexity-only or default stages.")
        # Avoid forcing complexity_only if user explicitly set another type, let it fall to default.
        if script_cfg.curriculum_type == "dual_layer" or script_cfg.curriculum_type == "level_only":
            logger.info(f"Switching curriculum type from '{script_cfg.curriculum_type}' to 'complexity_only' due to missing level info.")
            script_cfg.curriculum_type = "complexity_only"
    
    curriculum_stages_config_list = []

    if script_cfg.curriculum_type == "dual_layer" and has_level_info:
        logger.info(f"Dynamically generating dual_layer curriculum stages with focus: {script_cfg.curriculum_focus_levels}, emphasis: {script_cfg.curriculum_complexity_emphasis}")

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
            focus_levels=script_cfg.curriculum_focus_levels, # Assumed to be List[str] from HfArgumentParser
            complexity_emphasis=script_cfg.curriculum_complexity_emphasis
        )
        logger.info(f"Generated {len(curriculum_stages_config_list)} custom stages for dual_layer.")
    else:
        if script_cfg.curriculum_type != "dual_layer" and has_level_info : # e.g. level_only, but we are simplifying
            logger.info(f"Curriculum type is '{script_cfg.curriculum_type}'. Using default stages as specific logic for this type (other than dual_layer) is not implemented here, or conditions not met.")
        elif not has_level_info and script_cfg.curriculum_type != "complexity_only": # User might have set dual_layer/level_only but data is missing
            logger.info(f"Dataset lacks level information for '{script_cfg.curriculum_type}'. Falling back to default (likely complexity-based) stages.")
        
        curriculum_stages_config_list = create_default_curriculum_stages() # Handles complexity_only or serves as fallback
        logger.info(f"Generated {len(curriculum_stages_config_list)} default stages (type: {script_cfg.curriculum_type}).")

    if curriculum_stages_config_list:
        for stage_config_item in curriculum_stages_config_list:
            if isinstance(stage_config_item, CurriculumStageConfig):
                stage_config_item.min_evaluations = 10
            else:
                logger.warning(f"Encountered non-CurriculumStageConfig item in list: {type(stage_config_item)}")
        logger.info(f"Ensured min_evaluations is 10 for all {len(curriculum_stages_config_list)} stages.")
    else:
        logger.warning("No curriculum stages were generated. Curriculum learning might be ineffective.")
        return None

    if not curriculum_stages_config_list:
        logger.error("No curriculum stages defined after attempting generation. Disabling curriculum learning.")
        return None

    return EnhancedCurriculumManager(curriculum_stages_config_list, dataset)


def enhance_dataset_with_level_and_complexity(dataset: Dataset) -> Dataset:
    if not dataset or len(dataset) == 0:
        logger.warning("enhance_dataset_with_level_and_complexity: Received empty or None dataset.")
        return dataset

    def process_example(example):
        if not isinstance(example, dict):
            if hasattr(example, 'keys') and callable(example.keys) and hasattr(example, '__getitem__'):
                logger.debug(f"Converting LazyRow/dict-like object of type {type(example)} to dict in process_example.")
                try:
                    example = dict(example)
                except Exception as e_conv:
                    logger.error(f"Failed to convert dict-like object {type(example)} to dict: {e_conv}. Skipping example.")
                    return None
            else:
                logger.warning(f"Skipping non-dict/non-dict-like example in process_example: {type(example)}")
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
        logger.info("Dataset enhanced/validated with level and complexity information")
        
        if len(enhanced_dataset) > 0:
            levels = [ex['level'] for ex in enhanced_dataset]
            complexities = [ex['complexity_score'] for ex in enhanced_dataset]
            
            logger.info("Enhanced dataset distribution:")
            for level_val in sorted(list(set(levels))): 
                count = levels.count(level_val)
                level_complexities = [c for ex, c in zip(enhanced_dataset, complexities) if ex['level'] == level_val]
                avg_complexity = np.mean(level_complexities) if level_complexities else 0
                logger.info(f"  {level_val.capitalize()}: {count} samples, avg complexity: {avg_complexity:.2f}")
        else:
            logger.warning("Dataset is empty after enhancement processing.")
            
        return enhanced_dataset
        
    except Exception as e:
        logger.error(f"Failed to enhance dataset: {e}", exc_info=True)
        return dataset 

def calculate_enhanced_rewards_for_single_prompt(
    prompt_str: str, 
    completions_for_this_prompt: List[str], 
    current_tb_path: str,
    current_expected_total_from_manifest: int, 
    current_ref_verilog_path: str,
    reward_config: EnhancedRewardConfig,
    training_step: int = 0,
    wandb_callback: Optional[DetailedWandbCallback] = None, 
    output_dir_for_debug: Optional[str] = None 
) -> List[Dict[str, Any]]: # Return list of dicts now
    detailed_results_for_prompt: List[Dict[str, Any]] = [] # Changed from prompt_rewards
    num_completions = len(completions_for_this_prompt)
    
    prompt_id_base = prompt_str.split('\n', 1)[0] 
    name_match_for_id = re.search(r"module MUST be named `(\w+)`", prompt_str, re.IGNORECASE)
    if name_match_for_id:
        prompt_id_base = f"Mod_{name_match_for_id.group(1)}"
    
    sanitized_prompt_id_for_file = re.sub(r'[^\w_.)( -]', '', prompt_id_base).strip().replace(' ', '_')[:50]
    if not sanitized_prompt_id_for_file: 
        sanitized_prompt_id_for_file = "unknown_prompt"
    prompt_id_for_log = prompt_id_base[:70]

    logger.debug(f"ENHANCED_REWARDS: For '{prompt_id_for_log}', processing {num_completions} completion(s).")
    
    module_name, req_ports = "", [] 
    if current_ref_verilog_path and os.path.exists(current_ref_verilog_path):
        module_name, req_ports = extract_module_info(current_ref_verilog_path)
    
    if not module_name:
        logger.error(f"ENHANCED_REWARDS: '{prompt_id_for_log}': Failed to extract module info from ref Verilog '{current_ref_verilog_path}' or path invalid.")
        # Return structure consistent with new output type
        error_reward_val = reward_config.get_scaled_reward(reward_config.compilation_failure * 2, training_step)
        error_result_item = {
            "final_reward": error_reward_val,
            "unscaled_components": {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": reward_config.compilation_failure * 2},
            "funnel_metrics": {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
        }
        return [error_result_item] * num_completions

    for j, full_output in enumerate(completions_for_this_prompt):
        log_pref = f"ENHANCED_REWARDS: '{prompt_id_for_log}', Completion {j+1}/{num_completions}"
        # Initialize unscaled reward components for this completion
        current_unscaled_components = {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": 0.0}
        current_funnel_metrics = {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
        
        if not isinstance(full_output, str):
            logger.error(f"{log_pref}: Output is not a string, type: {type(full_output)}.")
            total_reward = reward_config.get_scaled_reward(reward_config.compilation_failure * 2, training_step)
            current_unscaled_components["base_compilation"] = reward_config.compilation_failure * 2 # reflect penalty
            detailed_results_for_prompt.append({
                "final_reward": total_reward,
                "unscaled_components": current_unscaled_components,
                "funnel_metrics": current_funnel_metrics
            })
            continue

        # Log raw model output
        raw_output_len = len(full_output)
        if raw_output_len > 1000: # Log snippet if too long
            logger.debug(f"{log_pref}: Raw model output (len={raw_output_len}):\n{full_output[:500]}\n...\n{full_output[-500:]}")
        else:
            logger.debug(f"{log_pref}: Raw model output (len={raw_output_len}):\n{full_output}")

        _, code = parse_llm_completion_qwen3(
            full_output, 
            debug_prompt=prompt_str,  # 传递原始prompt
            debug_context={"step": training_step, "sample_idx": j, "model": "qwen3"}
        )
        print("*"*100)
        print(f"{prompt_str=}")
        print(f"{full_output=}")
        print("*"*100)
        # Log extracted code
        if code and code.strip():
            logger.debug(f"{log_pref}: Extracted code:\n{code}")
            current_funnel_metrics["code_extracted"] = True
        else:
            logger.debug(f"{log_pref}: Extracted code: None or empty.")
            current_funnel_metrics["code_extracted"] = False # Explicitly set

        if not code or not code.strip(): 
            penalty_type = reward_config.missing_code_block_penalty if not code else reward_config.compilation_failure
            current_unscaled_components["base_compilation"] = penalty_type # Store unscaled penalty
            log_msg = "No Verilog code block found" if not code else "Empty code block"
            logger.warning(f"{log_pref}: {log_msg} in output.")
            
            if output_dir_for_debug and not current_funnel_metrics["code_extracted"]: # Check funnel metric
                try:
                    debug_save_dir = os.path.join(output_dir_for_debug, "debug_no_code_outputs")
                    os.makedirs(debug_save_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"step_{training_step}_prompt_{sanitized_prompt_id_for_file}_compl_{j}_{timestamp}.txt"
                    filepath = os.path.join(debug_save_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f_debug:
                        f_debug.write(f"--- PROMPT (Original ID for log: {prompt_id_for_log}) ---\n{prompt_str}\n\n")
                        f_debug.write(f"--- COMPLETION INDEX: {j} ---\n\n--- FULL MODEL OUTPUT ---\n{full_output}")
                    logger.info(f"{log_pref}: Saved problematic (no code) output to {filepath}")
                except Exception as e_save:
                    logger.error(f"{log_pref}: Failed to save debug output: {e_save}")
            
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
        # Log validation result
        if is_valid:
            logger.debug(f"{log_pref}: Verilog validation: is_valid=True, err_msg=\"{err_msg}\"")
        else:
            logger.info(f"{log_pref}: Verilog validation: is_valid=False, err_msg=\"{err_msg}\"")
            current_unscaled_components["base_compilation"] = reward_config.compilation_failure

        if is_valid: # Only proceed to simulation if valid
            sim_res = run_iverilog_simulation(
                code, current_tb_path, current_expected_total_from_manifest,
                prompt_id_for_log, j, logger.isEnabledFor(logging.DEBUG) 
            )
            # Log simulation result
            logger.debug(f"{log_pref}: Simulation results: {sim_res}")
            current_funnel_metrics["compiled_successfully"] = sim_res["compilation_success"]

            if not sim_res["compilation_success"]:
                current_unscaled_components["base_compilation"] = reward_config.compilation_failure
                logger.info(f"{log_pref}: Compilation FAILED. Error: {sim_res.get('error_message', 'N/A')}")
            else:
                current_unscaled_components["base_compilation"] = reward_config.compilation_success
                current_funnel_metrics["sim_ran_successfully"] = sim_res["simulation_run_success"]
                if not sim_res["simulation_run_success"]:
                    current_unscaled_components["functional"] = reward_config.simulation_crash
                    logger.info(f"{log_pref}: Simulation CRASHED.")
                elif not sim_res["parsing_success"]:
                    current_unscaled_components["functional"] = reward_config.output_parse_error
                    logger.info(f"{log_pref}: Output parsing FAILED.")
                else:
                    p, total_tests_in_output = sim_res["passed_tests"], sim_res["total_tests_in_output"]
                    current_funnel_metrics["passed_tests"] = p
                    if total_tests_in_output > 0:
                        pass_ratio = p / total_tests_in_output
                        base_functional = pass_ratio * reward_config.max_functional_reward
                        if p > 1: # Apply bonus only if more than one test passed
                            bonus_factor = reward_config.test_pass_bonus_multiplier ** (p - 1) # Exponential bonus
                            base_functional *= min(bonus_factor, 2.0) # Cap bonus to 2x
                        current_unscaled_components["functional"] = base_functional
                        if sim_res["all_tests_passed_by_tb"] and p == total_tests_in_output: # All TB tests passed
                            current_unscaled_components["robustness"] = reward_config.all_tests_passed_bonus
                        if p == total_tests_in_output and total_tests_in_output >= 5: # Consider this as good edge case handling
                            current_unscaled_components["robustness"] += reward_config.edge_case_handling_bonus
                    elif sim_res["all_tests_passed_by_tb"]: # No tests in output, but TB says PASS
                        current_unscaled_components["functional"] = reward_config.max_functional_reward
                        current_unscaled_components["robustness"] = reward_config.all_tests_passed_bonus
                        current_funnel_metrics["passed_tests"] = total_tests_in_output # Assume all expected passed if TB says so
                    else: # Parsing success but 0 tests passed or other issues
                        current_unscaled_components["functional"] = reward_config.output_parse_error # Or a specific penalty for 0 tests
                    logger.info(f"{log_pref}: Functional tests - Passed: {p}/{total_tests_in_output}, Overall: {sim_res['all_tests_passed_by_tb']}")
        # else: is_valid was false, base_compilation already set to failure penalty

        unscaled_total_reward = (
            reward_config.functional_weight * current_unscaled_components["functional"] +
            reward_config.efficiency_weight * current_unscaled_components["efficiency"] +
            reward_config.readability_weight * current_unscaled_components["readability"] +
            reward_config.robustness_weight * current_unscaled_components["robustness"] +
            current_unscaled_components["base_compilation"] # This is already a value like +1 or -5
        )
        final_scaled_reward = reward_config.get_scaled_reward(unscaled_total_reward, training_step)
        
        # For per-completion W&B logging (if still desired, now uses unscaled for components)
        if wandb_callback:
            # wandb_callback.log_reward_components(current_unscaled_components) # Log unscaled version
            wandb_callback.add_reward(final_scaled_reward) # Changed: Log final scaled reward for this completion for histogram
        
        logger.info(
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
        logger.critical(f"ENHANCED_REWARDS: '{prompt_id_for_log}': Result list length mismatch. Padding.")
        error_reward_val = reward_config.get_scaled_reward(reward_config.compilation_failure * 3, training_step)
        error_result_item = {
            "final_reward": error_reward_val,
            "unscaled_components": {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": reward_config.compilation_failure * 3},
            "funnel_metrics": {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}
        }
        detailed_results_for_prompt.extend([error_result_item] * (num_completions - len(detailed_results_for_prompt)))
    return detailed_results_for_prompt

# Custom callback for curriculum learning progression (defined at module level)
class CurriculumProgressCallback(TrainerCallback):
    def __init__(self, curriculum_manager, trainer_ref, output_dir):
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.performance_history = []
        self.output_dir = output_dir
        self.debug_log_path = os.path.join(output_dir, "curriculum_debug.txt")
        
        # 创建专门的课程学习日志文件
        with open(self.debug_log_path, 'w') as f:
            f.write(f"=== 课程学习调试日志 - {datetime.now()} ===\n")
    
    def _write_debug(self, message):
        """写入调试信息到专用文件和控制台"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_msg = f"[{timestamp}] CURRICULUM: {message}"
        
        # 控制台输出
        logger.info(debug_msg)
        
        # 写入调试文件
        with open(self.debug_log_path, 'a') as f:
            f.write(debug_msg + "\n")
    
    def on_evaluate(self, args, state, control, **kwargs):
        if self.curriculum_manager and args.local_rank <= 0:
            # 🔧 修复：安全获取global_step
            current_step = getattr(state, 'global_step', 0) or 0
            current_stage_idx = self.curriculum_manager.current_stage # Corrected variable name
            
            avg_test_pass_rate = 0.0 # Default
            found_metric = False
            for log_entry in reversed(state.log_history):
                if 'eval_avg_test_pass_rate' in log_entry:
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
                min_evals = stage_config.min_evaluations # This is now 10, but curriculum_manager handles actual counting

                self._write_debug(f"Stage Name: {stage_config.name}")
                self._write_debug(f"Performance Threshold: {threshold}")
                self._write_debug(f"Configured Min Evaluations for stage: {min_evals}") # This is the config value
                self._write_debug(f"Actual evaluations for this stage so far: {len(getattr(self.curriculum_manager, 'stage_performance_history', []))}")

                try:
                    # Assuming should_advance_stage in EnhancedCurriculumManager now takes performance_estimate
                    # and internally handles min_evaluations check using its history.
                    should_advance = self.curriculum_manager.should_advance_stage(performance_estimate)
                except TypeError as e:
                    self._write_debug(f"ERROR calling should_advance_stage: {e}. Check method signature in EnhancedCurriculumManager.")
                    should_advance = False
                
                self._write_debug(f"Decision: Should advance stage? {should_advance}")

                if should_advance:
                    advance_success = False
                    # Assuming EnhancedCurriculumManager has a method like 'advance_stage' or 'advance_to_next_stage'
                    if hasattr(self.curriculum_manager, 'advance_stage'):
                        advance_success = self.curriculum_manager.advance_stage() # This should internally use the stored performance history
                    elif hasattr(self.curriculum_manager, 'advance_to_next_stage'): # Alternative name
                        advance_success = self.curriculum_manager.advance_to_next_stage()
                    else:
                        self._write_debug("ERROR: Curriculum manager lacks 'advance_stage' or 'advance_to_next_stage' method.")
                    
                    if advance_success:
                        new_stage_idx = self.curriculum_manager.current_stage
                        new_dataset = self.curriculum_manager.get_current_stage_dataset()

                        self._write_debug(f"✅ Successfully advanced to stage {new_stage_idx}. New dataset size: {len(new_dataset)}")

                        progress_record = {
                            "step": current_step,
                            "old_stage_idx": current_stage_idx, # This was before advancing
                            "new_stage_idx": new_stage_idx,
                            "performance_metric (avg_test_pass_rate)": performance_estimate,
                            "new_dataset_size": len(new_dataset),
                            "timestamp": datetime.now().isoformat()
                        }

                        progress_file = os.path.join(self.output_dir, "stage_progress.jsonl")
                        with open(progress_file, 'a') as f:
                            f.write(json.dumps(progress_record) + "\n")

                        if args.local_rank <= 0 and hasattr(wandb, 'run') and wandb.run is not None:
                            old_stage_name = stage_config.name # Name of the stage we are leaving
                            new_stage_name = "Unknown"
                            if new_stage_idx < len(self.curriculum_manager.curriculum_stages):
                                new_stage_name = self.curriculum_manager.curriculum_stages[new_stage_idx].name

                            wandb.log({
                                "curriculum/stage_transition": 1,
                                "curriculum/old_stage_index": current_stage_idx, # Index before advancing
                                "curriculum/new_stage_index": new_stage_idx,
                                "curriculum/old_stage_name": old_stage_name,
                                "curriculum/new_stage_name": new_stage_name,
                                "curriculum/performance_metric": performance_estimate
                            }, step=current_step)
                            self._write_debug(f"Logged stage transition to W&B: {old_stage_name} -> {new_stage_name}")
            else: # current_stage_idx is beyond the defined stages
                self._write_debug("INFO: All curriculum stages completed or current stage index is out of bounds.")

            self._write_debug("-" * 50) # Separator for logs

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        if not self.curriculum_manager or args.local_rank > 0: # ensure manager exists and only on main process
            return

        current_step = getattr(state, 'global_step', 0) or 0

        # Regular W&B logging for current curriculum status
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
                    # This might be expensive. If so, curriculum_manager should cache dataset size per stage.
                    current_dataset = self.curriculum_manager.get_current_stage_dataset()
                    num_samples_in_stage = len(current_dataset) if current_dataset else 0
                except Exception as e_ds_len:
                    logger.warning(f"CurriculumProgressCallback: Could not get current stage dataset size for logging: {e_ds_len}")

            # Get the latest performance estimate if available (e.g. from eval_avg_test_pass_rate logged by DetailedInferenceCallback)
            latest_perf_estimate = 0.0
            # Check if 'eval_avg_test_pass_rate' is in the logs passed to this on_log method (e.g. from trainer state after eval)
            if logs and 'eval_avg_test_pass_rate' in logs:
                 latest_perf_estimate = logs['eval_avg_test_pass_rate']
            elif state.log_history: # Fallback to searching history if not in current logs
                for log_entry in reversed(state.log_history):
                    if 'eval_avg_test_pass_rate' in log_entry:
                        latest_perf_estimate = log_entry['eval_avg_test_pass_rate']
                        break

            wandb.log({
                "curriculum/current_stage_idx": current_stage_idx,
                "curriculum/current_stage_name_numeric": current_stage_idx, # Using index for easier plotting over name
                "curriculum/num_samples_in_stage": num_samples_in_stage,
                "curriculum/current_stage_perf_threshold": current_stage_performance_threshold,
                "curriculum/latest_eval_avg_test_pass_rate": latest_perf_estimate
            }, step=state.global_step)

        # Original local file logging for when stage actually changes
        # Renamed attribute to avoid potential conflicts and make it clear it's for local logging
        if not hasattr(self, 'last_locally_logged_stage_idx') or self.last_locally_logged_stage_idx != self.curriculum_manager.current_stage or current_step % 50 == 0:
            current_stage_idx_for_local_log = self.curriculum_manager.current_stage
            stage_name_for_local_log = "Unknown/Completed"
            if current_stage_idx_for_local_log < len(self.curriculum_manager.curriculum_stages):
                 stage_name_for_local_log = self.curriculum_manager.curriculum_stages[current_stage_idx_for_local_log].name

            self._write_debug(f"Step {current_step}: Currently in curriculum stage {current_stage_idx_for_local_log} ('{stage_name_for_local_log}'). Dataset size: {len(self.curriculum_manager.get_current_stage_dataset())}")
            self.last_locally_logged_stage_idx = current_stage_idx_for_local_log


# 3. 优化的课程学习回调
class OptimizedCurriculumCallback(DefaultFlowCallback):
    """优化的课程学习回调，包含动态难度调整"""
    
    def __init__(self, curriculum_manager, trainer_ref, output_dir):
        super().__init__()
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        self.difficulty_adjuster = DynamicDifficultyAdjuster(curriculum_manager)
        self.performance_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次日志记录时检查课程进展"""
        if logs is None or not self.curriculum_manager:
            return
            
        current_loss = logs.get('train_loss', float('inf'))
        training_step = state.global_step
        
        # 记录性能
        performance = 1.0 - min(current_loss, 1.0)
        self.performance_history.append({
            'step': training_step,
            'performance': performance,
            'loss': current_loss,
            'stage': self.curriculum_manager.current_stage
        })
        
        # 检查是否需要阶段进阶
        if self.curriculum_manager.should_advance_to_next_stage(current_loss, training_step):
            old_stage = self.curriculum_manager.current_stage
            success = self.curriculum_manager.advance_to_next_stage()
            
            if success and self.trainer_ref:
                # 更新训练器的数据集
                new_dataset = self.curriculum_manager.get_current_stage_dataset()
                
                # 🔧 这里需要特殊处理，因为不能直接替换Trainer的数据集
                # 可以通过设置标志让主训练循环知道需要重新初始化
                logger.info(f"🎯 阶段进阶: {old_stage} → {self.curriculum_manager.current_stage}")
                logger.info(f"📊 新数据集大小: {len(new_dataset)}")
        
        # 动态难度调整
        self.difficulty_adjuster.adjust_current_stage_difficulty(self.performance_history)
        
        # 每50步保存课程状态
        if training_step % 50 == 0:
            self._save_curriculum_state()
    
    def _save_curriculum_state(self):
        """保存课程学习状态"""
        if not self.output_dir:
            return
            
        state_data = {
            'current_stage': self.curriculum_manager.current_stage,
            'performance_history': self.performance_history[-100:],  # 只保存最近100条
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = os.path.join(self.output_dir, 'curriculum_state_detailed.json')
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存课程状态失败: {e}")
# 🔧 额外的调试信息函数
def debug_checkpoint_contents(checkpoint_path):
    """调试checkpoint内容"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint路径不存在: {checkpoint_path}")
        return
        
    logger.info(f"🔍 检查checkpoint内容: {checkpoint_path}")
    
    # 列出所有文件
    try:
        files = os.listdir(checkpoint_path)
        logger.info(f"Checkpoint文件列表: {files}")
        
        # 检查关键文件
        key_files = {
            'config.json': '模型配置',
            'pytorch_model.bin': 'PyTorch模型权重',
            'model.safetensors': 'SafeTensors模型权重',
            'adapter_config.json': 'PEFT适配器配置',
            'adapter_model.bin': 'PEFT适配器权重(bin)',
            'adapter_model.safetensors': 'PEFT适配器权重(safetensors)',
            'training_args.bin': '训练参数',
            'trainer_state.json': '训练状态',
            'optimizer.pt': '优化器状态',
            'scheduler.pt': '调度器状态'
        }
        
        logger.info("📁 关键文件检查:")
        for filename, description in key_files.items():
            exists = filename in files
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {filename}: {description}")
            
    except Exception as e:
        logger.error(f"无法读取checkpoint目录: {e}")
# 5. 在训练循环中添加定期状态报告
class PeriodicStatusReporter:
    def __init__(self, output_dir, report_interval=100):
        self.output_dir = output_dir
        self.report_interval = report_interval
        self.status_log_path = os.path.join(output_dir, "training_status.txt")
        
    def report_status(self, step, trainer_state, curriculum_manager=None, experience_buffer=None):
        """生成定期状态报告"""
        if step % self.report_interval != 0:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_report = f"""
========================================
训练状态报告 - 步数 {step}
时间: {timestamp}
========================================

📈 训练进度:
  - 当前步数: {step}
  - 最大步数: {trainer_state.max_steps if trainer_state.max_steps > 0 else '无限制'}
  - 完成百分比: {(step/trainer_state.max_steps*100) if trainer_state.max_steps > 0 else 'N/A'}%

📚 课程学习状态:"""
        
        if curriculum_manager:
            current_stage = curriculum_manager.current_stage
            total_stages = len(curriculum_manager.curriculum_stages)
            
            status_report += f"""
  - 当前阶段: {current_stage}/{total_stages-1}
  - 阶段名称: {curriculum_manager.curriculum_stages[current_stage].name if current_stage < total_stages else 'final'}
  - 阶段进度: {len(curriculum_manager.stage_performance_history)}次评估
  - 数据集大小: {len(curriculum_manager.get_current_stage_dataset())}"""
        else:
            status_report += "\n  - 课程学习: 未启用"
        
        status_report += f"""

🔄 经验回放状态:"""
        
        if experience_buffer:
            buffer_stats = experience_buffer.get_stats()
            status_report += f"""
  - 缓存大小: {buffer_stats['size']}/{experience_buffer.max_size}
  - 平均奖励: {buffer_stats['mean_reward']:.2f}
  - 最高奖励: {buffer_stats['max_reward']:.2f}"""
        else:
            status_report += "\n  - 经验回放: 未启用"
        
        # 最近的损失信息
        if trainer_state.log_history:
            recent_loss = trainer_state.log_history[-1].get('loss', 'N/A')
            status_report += f"""

📊 最近指标:
  - 训练损失: {recent_loss}
  - 学习率: {trainer_state.log_history[-1].get('learning_rate', 'N/A')}"""
        
        status_report += f"""

========================================
"""
        
        # 输出到控制台
        logger.info(status_report)
        
        # 保存到文件
        with open(self.status_log_path, 'a') as f:
            f.write(status_report + "\n")
class DetailedRewardCallback(TrainerCallback):
    """详细奖励监控回调"""
    
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        self.reward_history = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # 🔧 修复：安全获取global_step
        current_step = getattr(state, 'global_step', 0) or 0
        
        # 记录奖励相关指标
        reward_metrics = {}
        for key, value in logs.items():
            if 'reward' in key.lower() and isinstance(value, (int, float)):
                reward_metrics[key] = value
        
        if reward_metrics:
            self.reward_history.append({
                'step': current_step,
                'metrics': reward_metrics
            })
            
            logger.info(f"🎯 步数 {current_step}: 奖励指标 = {reward_metrics}")
            
            # 定期保存奖励历史
            if current_step % 100 == 0:
                self._save_reward_history()
    
    def _save_reward_history(self):
        """保存奖励历史"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            reward_history_path = os.path.join(self.output_dir, "reward_history.json")
            with open(reward_history_path, "w", encoding="utf-8") as f:
                json.dump(self.reward_history, f, indent=2)
        except Exception as e_save:
            logger.warning(f"保存奖励历史失败: {e_save}")
def main():
    # 🔧 修复：在函数开始就初始化所有变量，避免UnboundLocalError
    curriculum_manager = None
    experience_buffer = None
    wandb_callback = None
    model = None
    tokenizer = None
    dataset = None
    trainer = None
    dataset_for_trainer = None
    callbacks_list = []
    
    try:
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        env_cfg, script_cfg, reward_cfg, grpo_cfg = parser.parse_args_into_dataclasses()
        logging.basicConfig(
            level=grpo_cfg.get_process_log_level(), 
            format=f"[RANK {grpo_cfg.local_rank:02d}] %(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            handlers=log_handlers,
            force=True, 
        )

        logger = logging.getLogger(__name__)
        # 🚀 快速优化
        script_cfg.max_steps = max(getattr(script_cfg, 'max_steps', 300), 300)
        grpo_cfg.learning_rate = 1e-5
        grpo_cfg.eval_steps = 2
        logger.info(f"🔧 快速优化: 最大步数={script_cfg.max_steps}, 学习率={grpo_cfg.learning_rate}")
        # 将setup_enhanced_debugging函数定义移到这里，避免变量引用问题
        def setup_enhanced_debugging_local(script_cfg, grpo_cfg, curriculum_manager, experience_buffer):
            """设置增强调试功能"""
            
            callbacks_list = [StepLoggingCallback()]
            
            # 详细的课程学习回调
            if curriculum_manager:
                detailed_curriculum_callback = CurriculumProgressCallback(
                    curriculum_manager, None, script_cfg.output_dir
                )
                callbacks_list.append(detailed_curriculum_callback)
            
            # 详细的奖励监控回调
            reward_callback = DetailedRewardCallback(script_cfg.output_dir)
            callbacks_list.append(reward_callback)
            
            # W&B callback (如果启用)
            wandb_callback = None
            # 🔧 修复wandb接续
            if grpo_cfg.local_rank <= 0 and "wandb" in grpo_cfg.report_to:
                # 设置wandb的恢复参数
                wandb_resume_mode = "allow"  # 允许恢复
                wandb_run_id = None
                
                if is_resuming:
                    # 尝试从原始run_name推导wandb run_id
                    wandb_run_id = sanitized_run_name
                    wandb_resume_mode = "allow"  # 尝试恢复，如果失败则创建新的
                    logger.info(f"🔗 尝试恢复wandb run: {wandb_run_id}")
                
                # 设置环境变量让wandb知道要恢复
                if wandb_run_id:
                    os.environ["WANDB_RUN_ID"] = wandb_run_id
                    os.environ["WANDB_RESUME"] = wandb_resume_mode
                
                wandb_callback = DetailedWandbCallback(env_cfg, script_cfg, reward_cfg, experience_buffer)
                callbacks_list.append(wandb_callback)
                logger.info(f"✅ wandb回调已创建 - resume模式: {wandb_resume_mode}")

            
            return callbacks_list, wandb_callback
        
        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
        if not run_specific_name_from_env:
            logger_temp.warning("WANDB_RUN_NAME environment variable not set. Generating run name from timestamp.")
            run_specific_name_from_env = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)

        if not script_cfg.output_dir_base or not isinstance(script_cfg.output_dir_base, str):
            logger_temp.error(f"ScriptConfig.output_dir_base ('{script_cfg.output_dir_base}') is invalid. Using './grpo_runs'.")
            script_cfg.output_dir_base = "./grpo_runs" 
        # 🔧 简单修复：断续训练使用原始目录
        resume_checkpoint_path = grpo_cfg.resume_from_checkpoint
        is_resuming = (
            resume_checkpoint_path and 
            isinstance(resume_checkpoint_path, str) and 
            os.path.isdir(resume_checkpoint_path)
        )
        
        if is_resuming:
            # 使用checkpoint的父目录作为输出目录
            actual_output_dir = os.path.dirname(resume_checkpoint_path)
            sanitized_run_name = os.path.basename(actual_output_dir)
            logger_temp.info(f"🔄 断续训练模式")
            logger_temp.info(f"📁 使用原始输出目录: {actual_output_dir}")
            logger_temp.info(f"📝 原始run_name: {sanitized_run_name}")
        else:
            # 原来的新训练逻辑保持不变
            run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
            if not run_specific_name_from_env:
                logger_temp.warning("WANDB_RUN_NAME environment variable not set. Generating run name from timestamp.")
                run_specific_name_from_env = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)

            if not script_cfg.output_dir_base or not isinstance(script_cfg.output_dir_base, str):
                logger_temp.error(f"ScriptConfig.output_dir_base ('{script_cfg.output_dir_base}') is invalid. Using './grpo_runs'.")
                script_cfg.output_dir_base = "./grpo_runs" 
            actual_output_dir = os.path.join(script_cfg.output_dir_base, sanitized_run_name)
        
        if grpo_cfg.local_rank <= 0:
            os.makedirs(actual_output_dir, exist_ok=True)
            logger_temp.info(f"Actual output directory for this run: {actual_output_dir}")

        grpo_cfg.output_dir = actual_output_dir
        script_cfg.output_dir = actual_output_dir 

        log_file_path = os.path.join(actual_output_dir, "enhanced_training_log.txt")
        log_handlers = [logging.StreamHandler(sys.stdout)] 
        if grpo_cfg.local_rank <= 0: 
            log_mode = "a" 
            file_handler = logging.FileHandler(log_file_path, mode=log_mode, encoding='utf-8')
            log_handlers.append(file_handler)

        

        # 设置状态报告器
        status_reporter = PeriodicStatusReporter(script_cfg.output_dir, report_interval=50)

        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING if grpo_cfg.local_rank <=0 else logging.ERROR)

        logger.info(f"=== ENHANCED GRPO TRAINING STARTED (PID: {os.getpid()}) ===")
        logger.info(f"Process rank: {grpo_cfg.process_index}, Local rank: {grpo_cfg.local_rank}, Device: {grpo_cfg.device}, N_GPU: {grpo_cfg.n_gpu}, World Size: {grpo_cfg.world_size}")
        logger.info(f"Distributed Training Type: {grpo_cfg.distributed_state.distributed_type}")
        logger.info(f"Actual Output Dir: {actual_output_dir}")
        logger.info(f"Resume from checkpoint path (from args): {grpo_cfg.resume_from_checkpoint}")
        logger.debug(f"EnvConfig: \n{json.dumps(asdict(env_cfg), indent=2)}")
        logger.debug(f"ScriptConfig: \n{json.dumps(asdict(script_cfg), indent=2)}") 
        logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(reward_cfg), indent=2)}")
        logger.debug(f"GRPOConfig (TrainingArguments): \n{grpo_cfg.to_json_string()}")
        
        if grpo_cfg.world_size > 1:
            logger.info("Waiting for all processes at barrier before proceeding...")
            torch.distributed.barrier()
            logger.info("All processes passed barrier.")
        
        # 🔧 修复：确保experience_buffer初始化在使用前
        if script_cfg.enable_experience_replay:
            experience_buffer = ExperienceBuffer(max_size=script_cfg.experience_buffer_size)
            logger.info(f"Experience buffer initialized (size: {script_cfg.experience_buffer_size}).")
            if grpo_cfg.resume_from_checkpoint and isinstance(grpo_cfg.resume_from_checkpoint, str) and os.path.isdir(grpo_cfg.resume_from_checkpoint):
                buffer_state_path = os.path.join(grpo_cfg.resume_from_checkpoint, "enhanced_experience_buffer_state.json")
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
            elif grpo_cfg.resume_from_checkpoint:
                 logger.warning(f"grpo_cfg.resume_from_checkpoint ('{grpo_cfg.resume_from_checkpoint}') is not a valid directory. Cannot load experience buffer state.")

        quant_config_dict = {
            "load_in_4bit": False,  # 设置为False以禁用量化
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16 if grpo_cfg.bf16 else torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
        
        quantization_config_arg = None
        if quant_config_dict.get("load_in_4bit", False):  # 只有当启用4bit时才创建量化配置
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

        # 🔧 应用Qwen3兼容性修复
        logger.info("🔧 应用Qwen3兼容性修复...")
        model, tokenizer = Qwen3CompatibilityFixer.fix_generation_config(model, tokenizer)
        # 🔧 添加PEFT适配器配置
        logger.info("🔧 设置PEFT适配器...")
        
        # 🔧 修复后的PEFT适配器设置
        logger.info("🔧 设置PEFT适配器...")
        
        peft_config_stage2 = LoraConfig(
            r=script_cfg.lora_rank, 
            lora_alpha=script_cfg.lora_alpha, 
            lora_dropout=script_cfg.lora_dropout,
            target_modules=script_cfg.lora_target_modules, 
            bias="none", 
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 检查模型是否已经是PEFT模型
        is_model_peft_already = isinstance(model, PeftModel)
        logger.info(f"模型当前PEFT状态: {is_model_peft_already}")
        
        # 检查模型是否被量化
        is_quantized = getattr(model, 'is_quantized', False) or hasattr(model, 'hf_quantizer')
        logger.info(f"模型量化状态: {is_quantized}")
        
        peft_applied_successfully = False
        
        # 🔧 重要修复：检查是否从checkpoint恢复，并且checkpoint包含PEFT适配器
        resume_checkpoint_path = grpo_cfg.resume_from_checkpoint
        is_resuming_from_checkpoint = (
            resume_checkpoint_path and 
            isinstance(resume_checkpoint_path, str) and 
            os.path.isdir(resume_checkpoint_path)
        )
        
        if is_resuming_from_checkpoint:
            logger.info(f"🔄 检测到从checkpoint恢复: {resume_checkpoint_path}")
            
            # 检查checkpoint中是否包含PEFT适配器文件
            peft_files_in_checkpoint = [
                'adapter_config.json',
                'adapter_model.bin',
                'adapter_model.safetensors'
            ]
            
            has_peft_in_checkpoint = any(
                os.path.exists(os.path.join(resume_checkpoint_path, peft_file))
                for peft_file in peft_files_in_checkpoint
            )
            
            logger.info(f"Checkpoint中是否包含PEFT文件: {has_peft_in_checkpoint}")
            
            if has_peft_in_checkpoint:
                try:
                    logger.info("🔄 从checkpoint加载PEFT适配器...")
                    # 如果模型还不是PEFT模型，先从checkpoint加载PEFT适配器
                    if not is_model_peft_already:
                        model = PeftModel.from_pretrained(
                            model, 
                            resume_checkpoint_path,
                            is_trainable=True
                        )
                        logger.info("✅ 成功从checkpoint恢复PEFT适配器")
                        peft_applied_successfully = True
                    else:
                        logger.info("✅ 模型已经是PEFT模型，checkpoint恢复将由Trainer处理")
                        peft_applied_successfully = True
                        
                except Exception as e_peft_resume:
                    logger.error(f"❌ 从checkpoint恢复PEFT适配器失败: {e_peft_resume}")
                    logger.info("🔄 将尝试创建新的PEFT适配器...")
                    peft_applied_successfully = False
            else:
                logger.warning("⚠️ Checkpoint中未发现PEFT适配器文件，将创建新的适配器")
                peft_applied_successfully = False
        
        # 情况2: 加载Stage 1适配器 (如果没有从checkpoint恢复PEFT)
        elif script_cfg.stage1_adapter_path and os.path.isdir(script_cfg.stage1_adapter_path) and not peft_applied_successfully:
            logger.info(f"📂 加载Stage 1适配器: {script_cfg.stage1_adapter_path}")
            
            if is_model_peft_already:
                logger.warning("⚠️ 模型已经是PeftModel，将跳过Stage 1适配器加载")
                peft_applied_successfully = True
            else:
                try:
                    # 对于量化模型，确保使用正确的方法
                    if is_quantized:
                        logger.info("🔧 为量化模型加载PEFT适配器...")
                        model = PeftModel.from_pretrained(
                            model, 
                            script_cfg.stage1_adapter_path, 
                            is_trainable=True,
                            device_map="auto"  # 对量化模型很重要
                        )
                    else:
                        model = PeftModel.from_pretrained(
                            model, 
                            script_cfg.stage1_adapter_path, 
                            is_trainable=True
                        )
                    
                    logger.info("✅ 成功加载Stage 1 LoRA适配器")
                    peft_applied_successfully = True
                    
                except Exception as e_peft_load:
                    logger.error(f"❌ 加载Stage 1 PEFT适配器失败: {e_peft_load}")
                    logger.info("🔄 将创建新的PEFT适配器...")
                    peft_applied_successfully = False
        
        # 情况3: 创建新的PEFT适配器 (如果以上都没有成功)
        if not peft_applied_successfully:
            logger.info("🆕 创建新的PEFT适配器...")
            
            try:
                if is_quantized:
                    logger.info("🔧 为量化模型创建PEFT适配器...")
                    # 对于量化模型，确保模型已经准备好进行k-bit训练
                    model = prepare_model_for_kbit_training(
                        model, 
                        use_gradient_checkpointing=grpo_cfg.gradient_checkpointing
                    )
                    logger.info("✅ 模型已准备好进行k-bit训练")
                
                # 创建PEFT模型
                model = get_peft_model(model, peft_config_stage2)
                logger.info("✅ 成功创建新的PEFT适配器")
                peft_applied_successfully = True
                
            except Exception as e_peft_create:
                logger.error(f"❌ 创建PEFT适配器失败: {e_peft_create}")
                logger.error("无法为模型创建PEFT适配器，训练无法继续")
                sys.exit(1)
        
        # 🔧 重要：验证最终的PEFT设置
        final_is_peft = isinstance(model, PeftModel)
        logger.info(f"🎯 最终模型PEFT状态: {final_is_peft}")
        
        if final_is_peft:
            logger.info("📊 可训练参数统计:")
            try:
                model.print_trainable_parameters()
            except Exception as e_print:
                logger.warning(f"无法打印可训练参数: {e_print}")
                
                # 手动计算可训练参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"总参数: {total_params:,}")
                logger.info(f"可训练参数: {trainable_params:,}")
                logger.info(f"可训练比例: {100 * trainable_params / total_params:.2f}%")
                
            # 🔧 额外验证：检查是否有可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                logger.error("❌ 没有找到可训练参数！这将导致训练失败")
                logger.info("🔧 尝试启用PEFT适配器的训练模式...")
                try:
                    model.train()  # 确保模型处于训练模式
                    # 如果有enable_adapters方法，调用它
                    if hasattr(model, 'enable_adapters'):
                        model.enable_adapters()
                    # 检查PEFT适配器是否可训练
                    if hasattr(model, 'peft_config'):
                        for adapter_name in model.peft_config.keys():
                            model.set_adapter(adapter_name)
                    
                    # 重新检查
                    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logger.info(f"修复后的可训练参数: {trainable_params_after:,}")
                    
                    if trainable_params_after == 0:
                        logger.error("❌ 仍然没有可训练参数，无法继续训练")
                        sys.exit(1)
                    else:
                        logger.info("✅ 成功启用可训练参数")
                        
                except Exception as e_enable:
                    logger.error(f"❌ 启用适配器训练失败: {e_enable}")
                    sys.exit(1)
        else:
            logger.error("❌ 模型不是PeftModel，这将导致训练失败")
            
            # 🔧 最后的尝试：强制创建PEFT模型
            logger.info("🔄 最后尝试：强制创建PEFT适配器...")
            try:
                model = get_peft_model(model, peft_config_stage2)
                final_is_peft_retry = isinstance(model, PeftModel)
                if final_is_peft_retry:
                    logger.info("✅ 强制创建PEFT适配器成功")
                else:
                    logger.error("❌ 强制创建PEFT适配器仍然失败")
                    sys.exit(1)
            except Exception as e_force:
                logger.error(f"❌ 强制创建PEFT适配器失败: {e_force}")
                sys.exit(1)

        try:
            # Determine effective base path for dataset files
            effective_dataset_base_path = script_cfg.dataset_base_path
            if effective_dataset_base_path is None or not str(effective_dataset_base_path).strip():
                effective_dataset_base_path = os.path.dirname(os.path.abspath(script_cfg.dataset_path))
                logger.info(f"dataset_base_path not provided or empty, derived base path from dataset_path: {effective_dataset_base_path}")
            else:
                effective_dataset_base_path = os.path.abspath(effective_dataset_base_path) # Ensure it's absolute
                logger.info(f"Using dataset_base_path from args: {effective_dataset_base_path}")

            # dataset_dir = os.path.dirname(os.path.abspath(script_cfg.dataset_path)) # Old logic
            # logger.info(f"Dataset directory: {dataset_dir}") # Old log
            
            dataset_raw = load_dataset("json", data_files=script_cfg.dataset_path, split="train", cache_dir=script_cfg.cache_dir)
            logger.info(f"Raw dataset loaded: {len(dataset_raw)} rows. Columns: {dataset_raw.column_names}")

            if dataset_raw is None or len(dataset_raw) == 0:
                logger.error(f"load_dataset returned None or empty for path: {script_cfg.dataset_path}. Exiting.")
                sys.exit(1)

            def full_dataset_processing_pipeline(raw_ds, ds_dir, num_proc_val, overwrite_cache_flag_val, local_rank_val, script_cfg_val):
                logger.info("Starting full dataset processing pipeline...")
                
                def preprocess_dataset_types(example):
                    processed_example = {}
                    required_cols = ["prompt", "testbench_path", "expected_total_tests", "reference_verilog_path"]
                    if local_rank_val <= 0:
                        logger.debug(f"Preprocessing example keys: {list(example.keys())}")
                    for col in required_cols:
                        if col not in example or example[col] is None:
                            logger.debug(f"Missing or None for required column '{col}' in example: {str(example)[:200]}")
                            return None
                        processed_example[col] = str(example[col]) if col in ["prompt", "testbench_path", "reference_verilog_path"] else example[col]
                    
                    for key, value in example.items():
                        if key not in processed_example:
                            processed_example[key] = value
                    try:
                        ett_val = example["expected_total_tests"]
                        if isinstance(ett_val, str): processed_example["expected_total_tests"] = int(ett_val) if ett_val.strip().isdigit() else -1
                        elif isinstance(ett_val, (int, float)): processed_example["expected_total_tests"] = int(ett_val)
                        else: processed_example["expected_total_tests"] = -1
                        if processed_example["expected_total_tests"] < 0: return None
                    except Exception: return None
                    return processed_example

                processed_ds = raw_ds.map(preprocess_dataset_types, num_proc=num_proc_val, load_from_cache_file=not overwrite_cache_flag_val)
                processed_ds = processed_ds.filter(lambda ex: ex is not None, num_proc=num_proc_val, load_from_cache_file=not overwrite_cache_flag_val)
                logger.info(f"Dataset after type preprocessing: {len(processed_ds)} rows.")
                if len(processed_ds) == 0: return processed_ds
                

                validated_examples_list = validate_and_update_dataset_paths(processed_ds, dataset_base_path=ds_dir)
                if not validated_examples_list:
                    logger.error("Dataset empty after path validation. Returning empty dataset.")
                    return Dataset.from_list([])
                validated_ds = Dataset.from_list(validated_examples_list)
                logger.info(f"Dataset after path validation: {len(validated_ds)} rows.")
                if len(validated_ds) == 0: return validated_ds

                prompt_enhanced_ds = validated_ds.map(enhance_prompt_func, num_proc=num_proc_val, load_from_cache_file=not overwrite_cache_flag_val)
                logger.info(f"Dataset after prompt enhancement: {len(prompt_enhanced_ds)} rows.")
                if len(prompt_enhanced_ds) == 0: return prompt_enhanced_ds
                
                qwen_formatted_ds = prompt_enhanced_ds.map(
                    wrap_prompt_for_qwen,
                    num_proc=num_proc_val, # 使用与之前map相同的num_proc
                    load_from_cache_file=not overwrite_cache_flag_val
                )
                logger.info(f"Dataset after Qwen prompt formatting: {len(qwen_formatted_ds)} rows.")
                if len(qwen_formatted_ds) > 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"First Qwen formatted example prompt (first 200 chars):\n{qwen_formatted_ds[0]['prompt'][:200]}...")
                
                # 后续的列选择等操作应该在 qwen_formatted_ds 上进行
                temp_ds_for_col_add = qwen_formatted_ds
                
                final_cols = [
                    "prompt",                       # 这个现在应该是 Qwen 格式化后的 prompt
                    "original_enhanced_prompt",     # 新增：存储 enhance_prompt_func 的直接输出，未被Qwen包装
                    "testbench_path",
                    "expected_total_tests",
                    "reference_verilog_path", 
                    "original_prompt_for_debug",    # 这个是最初始的用户输入,由 enhance_prompt_func 保留
                    "level",
                    "complexity_score",
                    "category",
                    "difficulty",
                    "task_id"
                ]
                current_cols = list(temp_ds_for_col_add.column_names)
                
                # temp_ds_for_col_add = prompt_enhanced_ds
                for col_to_add in final_cols:
                    if col_to_add not in current_cols:
                        default_val_map = {
                            "original_prompt_for_debug": (lambda x: x.get("prompt", "")),
                            "level": "intermediate", "complexity_score": 5.0,
                            "category": "Unknown", "difficulty": "C",
                            "task_id": (lambda x: f"task_{hash(x.get('prompt', random.random()))}")
                        }
                        if col_to_add in default_val_map:
                            val_provider = default_val_map[col_to_add]
                            if callable(val_provider):
                                 temp_ds_for_col_add = temp_ds_for_col_add.map(lambda x: {**x, col_to_add: val_provider(x)})
                            else:
                                 temp_ds_for_col_add = temp_ds_for_col_add.map(lambda x: {**x, col_to_add: val_provider})
                            logger.info(f"Added default column '{col_to_add}'.")
                            current_cols.append(col_to_add)
                        else:
                            logger.warning(f"Required column '{col_to_add}' missing and no default provider.")
                
                cols_to_keep_final = [col for col in final_cols if col in temp_ds_for_col_add.column_names]
                extra_useful_cols = ["simulation_tool_ok", "simulation_overall_pass", "actual_pass_count", 
                                     "actual_fail_count", "source_project_folder"]
                for extra_col in extra_useful_cols:
                    if extra_col in temp_ds_for_col_add.column_names and extra_col not in cols_to_keep_final:
                        cols_to_keep_final.append(extra_col)

                cols_to_remove_final = [col for col in temp_ds_for_col_add.column_names if col not in cols_to_keep_final]
                if cols_to_remove_final:
                    final_ds = temp_ds_for_col_add.remove_columns(cols_to_remove_final)
                    logger.info(f"Removed columns for final dataset: {cols_to_remove_final}")
                else:
                    final_ds = temp_ds_for_col_add
                
                logger.info(f"Dataset after column selection: {len(final_ds)} rows. Columns: {final_ds.column_names}")
                return final_ds

            logger.info("🤖 使用Qwen3优化的数据集处理流程...")
            dataset_dir = os.path.dirname(os.path.abspath(script_cfg.dataset_path))
            logger.info(f"Dataset directory: {dataset_dir}")
            dataset = qwen3_dataset_processing_pipeline(dataset_raw, dataset_dir, script_cfg)
            del dataset_raw; gc.collect()

            dataset = enhance_dataset_with_level_and_complexity(dataset) 

            logger.info(f"Final processed dataset: {len(dataset)} rows. Columns: {dataset.column_names}")
            if len(dataset) > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"First processed example:\n{json.dumps(dataset[0], indent=2)}")

        except Exception as e_dataset:
            logger.error(f"Dataset loading/processing error: {e_dataset}", exc_info=True)
            sys.exit(1)
        
        if not dataset or len(dataset) == 0: 
            logger.error("Dataset is empty after processing. Exiting.")
            sys.exit(1)

        if not validate_dataset_for_curriculum(dataset, script_cfg): 
            logger.error("Dataset validation for curriculum failed. Please check data format. Exiting.")
            sys.exit(1)
        
        # 🔧 修复：确保curriculum_manager在使用前被正确初始化
        try:
            curriculum_manager = setup_curriculum_manager(script_cfg, dataset)
            if curriculum_manager:
                logger.info(f"✅ 课程学习已启用")
                logger.info(f"📊 当前阶段: {curriculum_manager.current_stage}")
                logger.info(f"📊 总阶段数: {len(curriculum_manager.curriculum_stages)}")
                
                # 详细记录每个阶段信息
                for i, stage in enumerate(curriculum_manager.curriculum_stages):
                    logger.info(f"   阶段{i}: {stage.name} | 等级: {stage.dataset_levels} | 复杂度: {stage.complexity_range}")
                    
                # 记录当前阶段的数据集大小
                current_dataset = curriculum_manager.get_current_stage_dataset()
                logger.info(f"📊 当前阶段数据集大小: {len(current_dataset)}")
            else:
                logger.warning("⚠️ 课程学习未启用，使用全部数据集")
        except Exception as e_curriculum:
            logger.error(f"❌ 课程管理器创建失败: {e_curriculum}", exc_info=True)
            curriculum_manager = None
        

        if curriculum_manager and grpo_cfg.resume_from_checkpoint and \
           isinstance(grpo_cfg.resume_from_checkpoint, str) and \
           os.path.isdir(grpo_cfg.resume_from_checkpoint):
            curriculum_state_path = os.path.join(grpo_cfg.resume_from_checkpoint, "enhanced_curriculum_state.json")
            if os.path.exists(curriculum_state_path):
                try:
                    logger.info(f"Attempting to load curriculum state from: {curriculum_state_path}")
                    with open(curriculum_state_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                    curriculum_manager.load_curriculum_state(state_data)
                except Exception as e_load_curr:
                    logger.error(f"Failed to load curriculum state from {curriculum_state_path}: {e_load_curr}")
            else:
                logger.warning(f"Curriculum state file not found at {curriculum_state_path}. Curriculum will start from initial stage.")
        elif curriculum_manager and grpo_cfg.resume_from_checkpoint:
             logger.warning(f"grpo_cfg.resume_from_checkpoint ('{grpo_cfg.resume_from_checkpoint}') is not a valid directory. Cannot load curriculum state.")

        dataset_for_trainer = dataset 
        if curriculum_manager:
            dataset_for_trainer = curriculum_manager.get_current_stage_dataset()
            current_stage_idx = curriculum_manager.current_stage
            current_stage_obj = curriculum_manager.curriculum_stages[current_stage_idx] if current_stage_idx < len(curriculum_manager.curriculum_stages) else None
            current_stage_name = current_stage_obj.name if current_stage_obj else "Unknown/Final"
            logger.info(f"🎯 Curriculum learning active. Using {len(dataset_for_trainer)} examples from stage {current_stage_idx} ('{current_stage_name}').")
        else:
            logger.info("Curriculum learning disabled, using full dataset.")
        
        if not dataset_for_trainer or len(dataset_for_trainer) == 0:
            logger.error(f"Dataset for trainer is empty (size: {len(dataset_for_trainer) if dataset_for_trainer else 0}). Check curriculum logic or dataset. Exiting.")
            sys.exit(1)

        # 🔧 修复：现在使用本地函数设置调试功能
        # 🔧 添加增强的调试功能
        callbacks_list_enhanced = []
        # 🔧 创建稳定性监控
        stability_monitor = RewardStabilityMonitor(script_cfg.output_dir)
        # 添加稳定性监控回调
        callbacks_list_enhanced.append(stability_monitor)
        logger.info("✅ 添加奖励稳定性监控回调")
        
        # 添加增强的课程学习回调
        if curriculum_manager:
            enhanced_curriculum_cb = EnhancedCurriculumDebugCallback(
                curriculum_manager, None, script_cfg.output_dir  # trainer引用稍后设置
            )
            callbacks_list_enhanced.append(enhanced_curriculum_cb)
            logger.info("✅ 添加增强的课程学习调试回调")
        
        # 添加原有的其他回调
        if 'callbacks_list' in locals():
            callbacks_list_enhanced.extend(callbacks_list)
        
        # 使用增强的回调列表
        callbacks_list = callbacks_list_enhanced
        
        custom_state_pers_cb = CustomStatePersistenceCallback(curriculum_manager, experience_buffer, script_cfg)
        callbacks_list.append(custom_state_pers_cb)
        
        sample_dataset_for_inf_cb = dataset.select(range(min(len(dataset), script_cfg.callback_num_samples * 5))) if len(dataset) > 0 else None
        if sample_dataset_for_inf_cb and len(sample_dataset_for_inf_cb) > 0:
            detailed_inference_cb = DetailedInferenceCallback(
                tokenizer=tokenizer,
                eval_dataset=sample_dataset_for_inf_cb,
                num_samples=script_cfg.callback_num_samples,
                eval_every_n_steps=script_cfg.callback_eval_every_n_steps,
                max_new_tokens=grpo_cfg.max_completion_length,
                max_seq_length=script_cfg.max_seq_length,
                experience_buffer=experience_buffer,
                output_dir=script_cfg.output_dir
            )
            callbacks_list.append(detailed_inference_cb)
            logger.info(f"EnhancedInferenceCallback initialized with {len(sample_dataset_for_inf_cb)} samples.")
        else:
            logger.warning("EnhancedInferenceCallback will not run due to insufficient sample data from the full dataset.")

        if curriculum_manager: 
            curriculum_progress_cb_instance = CurriculumProgressCallback(curriculum_manager, None, script_cfg.output_dir) 
            callbacks_list.append(curriculum_progress_cb_instance)

        logger.info("Initializing GRPOTrainer...")
        trainer_instance_for_reward_func = None 
        
        def enhanced_batch_reward_calculator(
            prompts: List[str],  # 这些应该是 Qwen 格式化后的 prompts，即模型的实际输入
            completions: List[str],
            testbench_path: List[str],
            expected_total_tests: List[int],
            reference_verilog_path: List[str],
            original_enhanced_prompt: Optional[List[str]] = None, # 未经Qwen包装的、enhance_prompt_func的输出
            **kwargs: Any
        ) -> Tuple[List[float], Dict[str, Any]]: # Now returns rewards and aggregated metrics
            batch_rewards_final_scaled: List[float] = []

            # For aggregating unscaled components and funnel metrics
            batch_all_unscaled_components: List[Dict[str, float]] = []
            batch_all_funnel_metrics: List[Dict[str, Any]] = []

            num_items_in_batch = len(prompts)
    
            # --- 从 kwargs 中获取必要的上下文对象 ---
            # 假设这些键名与 reward_func_with_context 中添加到 kwargs_reward 的键名一致
            current_reward_cfg: Optional[EnhancedRewardConfig] = kwargs.get('reward_config_obj')
            current_script_cfg: Optional[ScriptConfig] = kwargs.get('script_config_obj')
            current_experience_buffer: Optional[ExperienceBuffer] = kwargs.get('experience_buffer_obj')
            
            training_step: int = kwargs.get('training_step', 0)
            wandb_cb_from_kwargs: Optional[DetailedWandbCallback] = kwargs.get('wandb_callback', None)
            output_dir_for_debug_val: Optional[str] = kwargs.get('output_dir', None)

            # --- 对关键配置对象进行有效性检查 ---
            if current_reward_cfg is None:
                logger.critical("EnhancedBatchRewardCalculator: 'reward_config_obj' 未在kwargs中提供! 奖励计算将使用默认惩罚值。")
                # 在没有有效 reward_cfg 的情况下，很难计算有意义的奖励，可以返回一个固定的惩罚值
                # 或者，如果 reward_cfg 在全局作用域中定义（不推荐），可以尝试回退：
                # global reward_cfg # 声明使用全局变量
                # current_reward_cfg = reward_cfg
                # 这里我们选择返回固定惩罚，因为依赖全局变量是不好的实践
                return [-10.0] * num_items_in_batch, {} # Return empty dict for metrics

            if current_script_cfg is None:
                logger.warning("EnhancedBatchRewardCalculator: 'script_config_obj' 未在kwargs中提供。某些功能（如经验回放的启用判断）可能受影响。")
                # 如果 script_cfg 对核心奖励逻辑不是绝对必要，可以继续；否则也应该处理错误

            # --- 检查输入列表长度是否一致 ---
            expected_lengths = {
                "prompts": len(prompts), "completions": len(completions), "testbench_path": len(testbench_path),
                "expected_total_tests": len(expected_total_tests), "reference_verilog_path": len(reference_verilog_path),
            }
            if original_enhanced_prompt is not None:
                expected_lengths["original_enhanced_prompt"] = len(original_enhanced_prompt)
            else: # 如果 original_enhanced_prompt 未提供，这是一个严重问题
                logger.error("EnhancedBatchRewardCalculator: 'original_enhanced_prompt' is None! "
                            "Reward calculation might be inaccurate as it relies on unwrapped prompts for parsing.")
                # 根据您的策略，这里可以选择返回惩罚，或者在下面循环中处理

            if len(set(expected_lengths.values())) > 1:
                mismatched_lengths_str = ", ".join([f"{k}:{v}" for k, v in expected_lengths.items()])
                logger.error(
                    f"Enhanced batch reward calculator: Mismatch in input list lengths. Details: {mismatched_lengths_str}"
                )
                return [current_reward_cfg.get_scaled_reward(current_reward_cfg.compilation_failure * 3, training_step)] * len(completions), {}

            if num_items_in_batch == 0:
                logger.warning("Enhanced batch reward calculator: Received an empty batch.")
                return [], {}

            for i in range(num_items_in_batch):
                qwen_formatted_prompt_for_buffer = prompts[i] # 模型实际看到的输入
                current_completion_str = completions[i] # Renamed for clarity
                current_tb = testbench_path[i]
                current_ett = expected_total_tests[i]
                current_ref_v = reference_verilog_path[i]

                # 获取用于奖励计算的、未经Qwen包装的增强后提示
                prompt_for_reward_calculation = ""
                if original_enhanced_prompt and i < len(original_enhanced_prompt) and \
                isinstance(original_enhanced_prompt[i], str) and original_enhanced_prompt[i].strip():
                    prompt_for_reward_calculation = original_enhanced_prompt[i]
                else:
                    logger.warning(
                        f"Item {i}: 'original_enhanced_prompt' not available, invalid, or empty. "
                        f"Using Qwen-formatted prompt ('{qwen_formatted_prompt_for_buffer[:70]}...') for reward calculation. "
                        f"This might lead to parsing issues in calculate_enhanced_rewards_for_single_prompt."
                    )
                    # 回退：使用Qwen格式化后的提示，但这可能导致 calculate_enhanced_rewards_for_single_prompt 内部解析错误
                    prompt_for_reward_calculation = qwen_formatted_prompt_for_buffer
                    # 从Qwen格式中尝试提取 user 部分内容，作为近似的“原始增强后提示”
                    match_user_content = re.search(r"<\|im_start\|>user\n(.*?)\n?<\|im_end\|>", qwen_formatted_prompt_for_buffer, re.DOTALL)
                    if match_user_content:
                        prompt_for_reward_calculation = match_user_content.group(1).strip()
                        logger.debug(f"Item {i}: Extracted user content from Qwen prompt for reward calculation.")
                    else:
                        logger.warning(f"Item {i}: Could not extract user content from Qwen prompt. Reward parsing may fail.")

                # calculate_enhanced_rewards_for_single_prompt now returns a list of dicts
                # Since we pass only one completion, we take the first element.
                results_list_for_item = calculate_enhanced_rewards_for_single_prompt(
                    prompt_str=prompt_for_reward_calculation,
                    completions_for_this_prompt=[current_completion_str], # Pass as a list
                    current_tb_path=current_tb,
                    current_expected_total_from_manifest=current_ett,
                    current_ref_verilog_path=current_ref_v,
                    reward_config=current_reward_cfg,
                    training_step=training_step,
                    wandb_callback=wandb_cb_from_kwargs, # This callback is for per-completion details if still needed
                    output_dir_for_debug=output_dir_for_debug_val
                )
                
                item_detailed_result = {}
                if results_list_for_item:
                    item_detailed_result = results_list_for_item[0]
                else: # Should not happen if calculate_enhanced_rewards_for_single_prompt is robust
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
                                "training_step": training_step, 
                                "testbench": current_tb, 
                                "original_enhanced_prompt_preview": prompt_for_reward_calculation[:100]
                            }
                        )
                    except Exception as e_exp:
                        logger.warning(f"Failed to add experience to buffer for item {i}: {e_exp}", exc_info=True)
            
            
            # 🔧 创建稳定化的奖励计算器

            # --- Aggregate metrics for the batch ---
            aggregated_metrics_for_wandb = {}
            if num_items_in_batch > 0:
                # 1. Unscaled Reward Components (Mean and Std)
                component_keys = ["functional", "efficiency", "readability", "robustness", "base_compilation"]
                for key in component_keys:
                    values = [comp[key] for comp in batch_all_unscaled_components if key in comp]
                    if values:
                        aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_mean"] = np.mean(values)
                        aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_std"] = np.std(values)

                # 2. Code Generation Funnel Metrics
                total_completions_in_batch = num_items_in_batch

                successful_extractions = sum(1 for fm in batch_all_funnel_metrics if fm["code_extracted"])
                successful_compilations = sum(1 for fm in batch_all_funnel_metrics if fm["compiled_successfully"]) # Assumes compiled_successfully=True implies code_extracted=True
                simulation_runs = sum(1 for fm in batch_all_funnel_metrics if fm["sim_ran_successfully"]) # Assumes sim_ran_successfully=True implies compiled_successfully=True

                aggregated_metrics_for_wandb["generation_funnel/successful_extractions_count"] = successful_extractions
                aggregated_metrics_for_wandb["generation_funnel/successful_extractions_ratio"] = successful_extractions / total_completions_in_batch if total_completions_in_batch > 0 else 0

                aggregated_metrics_for_wandb["generation_funnel/successful_compilations_count"] = successful_compilations
                # Ratio based on successfully extracted code for more meaningful funnel
                aggregated_metrics_for_wandb["generation_funnel/successful_compilations_ratio"] = successful_compilations / successful_extractions if successful_extractions > 0 else 0

                aggregated_metrics_for_wandb["generation_funnel/simulation_runs_count"] = simulation_runs
                # Ratio based on successfully compiled code
                aggregated_metrics_for_wandb["generation_funnel/simulation_runs_ratio"] = simulation_runs / successful_compilations if successful_compilations > 0 else 0

                passed_tests_values = [fm["passed_tests"] for fm in batch_all_funnel_metrics if fm["sim_ran_successfully"] and fm["passed_tests"] != -1]
                if passed_tests_values:
                    aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_runs"] = np.mean(passed_tests_values)
                else:
                    aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_runs"] = 0

            # Return both the list of final scaled rewards for GRPOTrainer and the aggregated metrics for W&B
            # 每10步输出详细统计
            if batch_rewards_final_scaled and training_step % 10 == 0:  # 每10步监控一次
                # 计算奖励统计
                reward_stats = {
                    'mean': np.mean(batch_rewards_final_scaled),
                    'std': np.std(batch_rewards_final_scaled),
                    'min': np.min(batch_rewards_final_scaled),
                    'max': np.max(batch_rewards_final_scaled),
                    'median': np.median(batch_rewards_final_scaled)
                }
                
                # 计算成功率统计
                positive_rewards = [r for r in batch_rewards_final_scaled if r > 0]
                success_rate = len(positive_rewards) / len(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0
                
                # 分析奖励分布
                high_rewards = [r for r in batch_rewards_final_scaled if r > reward_stats['mean'] + reward_stats['std']]
                low_rewards = [r for r in batch_rewards_final_scaled if r < reward_stats['mean'] - reward_stats['std']]
                
                logger.info(f"""
                📊 步数 {training_step} 奖励深度分析:
                ├─ 基础统计:
                │  ├─ 平均奖励: {reward_stats['mean']:.4f}
                │  ├─ 标准差: {reward_stats['std']:.4f}
                │  ├─ 中位数: {reward_stats['median']:.4f}
                │  └─ 范围: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]
                ├─ 成功分析:
                │  ├─ 正奖励率: {success_rate:.2%} ({len(positive_rewards)}/{len(batch_rewards_final_scaled)})
                │  ├─ 高奖励数: {len(high_rewards)} (>{reward_stats['mean']+reward_stats['std']:.3f})
                │  └─ 低奖励数: {len(low_rewards)} (<{reward_stats['mean']-reward_stats['std']:.3f})
                └─ 批次信息: {len(batch_rewards_final_scaled)} 个样本
                """)
                
                # 如果有经验缓冲区，记录统计信息
                if current_experience_buffer:
                    try:
                        buffer_stats = current_experience_buffer.get_stats()
                        logger.info(f"📚 经验缓冲区: {buffer_stats.get('size', 0)} 条经验")
                    except Exception as e_buffer:
                        logger.debug(f"无法获取经验缓冲区统计: {e_buffer}")
            
            # 每50步进行更深入的分析
            if training_step % 50 == 0 and training_step > 0:
                logger.info(f"""
                🎯 步数 {training_step} 训练里程碑:
                ├─ 当前批次平均奖励: {np.mean(batch_rewards_final_scaled):.4f}
                ├─ 奖励稳定性: {'稳定' if np.std(batch_rewards_final_scaled) < 2.0 else '波动较大'}
                ├─ 建议: {'继续当前策略' if np.mean(batch_rewards_final_scaled) > 0 else '考虑调整策略'}
                └─ 预计完成进度: {training_step}/{current_script_cfg.max_steps if current_script_cfg else '未知'} ({training_step/(current_script_cfg.max_steps if current_script_cfg and current_script_cfg.max_steps > 0 else 300)*100:.1f}%)
                """)
            
            return batch_rewards_final_scaled, aggregated_metrics_for_wandb
        
        def reward_func_with_context(*args_reward, **kwargs_reward):
            # 这些变量需要在 reward_func_with_context 被 GRPOTrainer 调用时，
            # 能够从其定义时的作用域（通常是 main 函数）通过闭包访问到。
            # 为了更安全，我们可以用 .get() 配合默认值，但这通常意味着外部作用域设置有问题。
            # 假设 trainer_instance_for_reward_func, wandb_callback, script_cfg, reward_cfg, experience_buffer
            # 都在此函数定义时的外部作用域中有效。

            current_training_step = 0
            try:
                if trainer_instance_for_reward_func is not None and \
                hasattr(trainer_instance_for_reward_func, 'state') and \
                trainer_instance_for_reward_func.state is not None:
                    current_training_step = trainer_instance_for_reward_func.state.global_step
                else:
                    logger.debug("reward_func_with_context: trainer_instance or its state is None. training_step defaults to 0.")
            except NameError: # 如果 trainer_instance_for_reward_func 未定义
                logger.warning("reward_func_with_context: 'trainer_instance_for_reward_func' not defined in the accessible scope.")
                # current_training_step 保持 0
            
            kwargs_reward['training_step'] = current_training_step
            # --- 处理 script_cfg ---
            # script_cfg 会被用于 output_dir 和 experience_buffer_obj 的逻辑
            local_script_cfg = None
            try:
                local_script_cfg = script_cfg # 尝试访问外部作用域的 script_cfg
                kwargs_reward['output_dir'] = local_script_cfg.output_dir
                kwargs_reward['script_config_obj'] = local_script_cfg # <--- 添加这一行
            except NameError:
                kwargs_reward['output_dir'] = None
                kwargs_reward['script_config_obj'] = None # <--- 确保也设置为 None
                logger.warning("reward_func_with_context: 'script_cfg' not defined. Passing None for output_dir and script_config_obj.")
            except AttributeError: # script_cfg 存在，但没有 output_dir 属性
                kwargs_reward['output_dir'] = None
                if local_script_cfg is not None: # local_script_cfg 可能在 NameError 之前被赋值
                    kwargs_reward['script_config_obj'] = local_script_cfg
                else: # 如果 script_cfg 真的未定义，这里再捕获一次 (虽然上面 NameError 应该已经捕获)
                    try: kwargs_reward['script_config_obj'] = script_cfg
                    except NameError: kwargs_reward['script_config_obj'] = None
                logger.warning("reward_func_with_context: 'script_cfg.output_dir' attribute missing. Passing None for output_dir.")

 
            try:
                kwargs_reward['wandb_callback'] = wandb_callback # 来自外部作用域
            except NameError:
                kwargs_reward['wandb_callback'] = None
                logger.warning("reward_func_with_context: 'wandb_callback' not defined. Passing None.")
                
            try:
                kwargs_reward['output_dir'] = script_cfg.output_dir # 来自外部作用域的 script_cfg
            except NameError:
                kwargs_reward['output_dir'] = None
                logger.warning("reward_func_with_context: 'script_cfg' not defined or has no 'output_dir'. Passing None for output_dir.")

            # 新增：显式传递 reward_cfg 和 experience_buffer
            try:
                kwargs_reward['reward_config_obj'] = reward_cfg # 来自外部作用域
            except NameError:
                kwargs_reward['reward_config_obj'] = None # 或者一个默认的RewardConfig实例
                logger.error("reward_func_with_context: 'reward_cfg' not defined! This is crucial for enhanced_batch_reward_calculator.")
                # 最好是能确保 reward_cfg 总是可用的，否则后续会出错

            try:
                # 只有当 experience_buffer 实际启用并存在时才传递
                if 'experience_buffer' in globals() and experience_buffer is not None and script_cfg.enable_experience_replay:
                    kwargs_reward['experience_buffer_obj'] = experience_buffer # 来自外部作用域
                else:
                    kwargs_reward['experience_buffer_obj'] = None
            except NameError: # script_cfg 或 experience_buffer 可能未定义
                kwargs_reward['experience_buffer_obj'] = None
                logger.debug("reward_func_with_context: 'experience_buffer' or 'script_cfg' not defined, or replay disabled. Passing None for experience_buffer_obj.")
                
            if args_reward:
                logger.warning(
                    f"reward_func_with_context received unexpected positional arguments: {args_reward}. "
                    "These will be ignored if not handled by enhanced_batch_reward_calculator via *args. "
                    "Ensure GRPOTrainer passes dataset columns as keyword arguments."
                )

            # 调用实际的奖励计算函数
        # Note: enhanced_batch_reward_calculator itself calls add_reward on the wandb_callback for each item in batch
        # if wandb_callback is passed into it and used by calculate_enhanced_rewards_for_single_prompt.
        # So, individual rewards are added there. Here we primarily log aggregated batch metrics.
            # rewards_list, aggregated_metrics = stabilized_reward_calculator(*args_reward, **kwargs_reward)
            rewards_list, aggregated_metrics = enhanced_batch_reward_calculator(*args_reward, **kwargs_reward)


            # Log aggregated metrics using wandb_callback if available
        # The individual rewards for the histogram are added within calculate_enhanced_rewards_for_single_prompt -> wandb_callback.add_reward()
        # which is called by enhanced_batch_reward_calculator.
        # Here, we log other batch-level aggregated metrics.
            if 'wandb_callback' in kwargs_reward and kwargs_reward['wandb_callback'] is not None:
                try:
                    current_step_for_direct_log = kwargs_reward.get('training_step', 0)
                    # log_batch_aggregated_metrics is for other types of metrics, not the reward histogram directly.
                    # The reward histogram is built up by add_reward and logged by DetailedWandbCallback.on_log
                    if aggregated_metrics: # Only log if there's something to log
                        kwargs_reward['wandb_callback'].log_batch_aggregated_metrics(aggregated_metrics, step=current_step_for_direct_log)
                except Exception as e_cb_log:
                    logger.error(f"Error calling wandb_callback.log_batch_aggregated_metrics: {e_cb_log}", exc_info=True)

            return rewards_list # GRPOTrainer expects only the list of rewards

        trainer = GRPOTrainer(
            model=model,
            args=grpo_cfg, 
            train_dataset=dataset_for_trainer,
            reward_funcs=[reward_func_with_context],
            callbacks=callbacks_list,
        )
        trainer_instance_for_reward_func = trainer 

        if curriculum_manager: 
            for cb in callbacks_list:
                if isinstance(cb, CurriculumProgressCallback):
                    cb.trainer_ref = trainer
                    logger.info("Set trainer_ref for CurriculumProgressCallback.")
                    break
        
        logger.info("GRPOTrainer initialized successfully.")

        try:
            logger.info(f"=== STARTING GRPO TRAINING (Output: {grpo_cfg.output_dir}) ===")
            logger.info(f"Dataset size for current stage: {len(dataset_for_trainer)} examples")
            
            actual_resume_path = grpo_cfg.resume_from_checkpoint
            if actual_resume_path and not (isinstance(actual_resume_path, str) and os.path.isdir(actual_resume_path)):
                logger.warning(
                    f"Provided resume_from_checkpoint path ('{actual_resume_path}') is invalid or not a directory. "
                    "Training will start from scratch or attempt to resume from Trainer's output_dir if applicable (but output_dir is new)."
                )
                actual_resume_path = None 

            logger.info(f"Final resume_from_checkpoint path for Trainer.train(): {actual_resume_path if actual_resume_path else 'No specific path (new or auto-resume from output_dir if checkpoint exists there)'}")
            
            train_res = trainer.train(resume_from_checkpoint=actual_resume_path) 
            
            if grpo_cfg.local_rank <= 0:
                final_model_dir = os.path.join(script_cfg.output_dir, "final_model_adapter") 
                logger.info(f"Training finished. Saving final model to {final_model_dir}")
                trainer.save_model(final_model_dir) 
                
                enhanced_artifacts_dir = os.path.join(script_cfg.output_dir, "training_artifacts")
                os.makedirs(enhanced_artifacts_dir, exist_ok=True)
                if experience_buffer and script_cfg.enable_experience_replay:
                    buffer_stats = experience_buffer.get_stats()
                    with open(os.path.join(enhanced_artifacts_dir, "experience_buffer_stats.json"), "w", encoding="utf-8") as f:
                        json.dump(buffer_stats, f, indent=2)
                    logger.info(f"Experience buffer stats saved. Final size: {buffer_stats.get('size', 'N/A')}")
                
                if curriculum_manager:
                    curriculum_stats = {
                        "final_stage_idx": curriculum_manager.current_stage,
                        "final_stage_name": curriculum_manager.curriculum_stages[curriculum_manager.current_stage].name if curriculum_manager.current_stage < len(curriculum_manager.curriculum_stages) else "Unknown/Final",
                        "stages_completed": curriculum_manager.current_stage + 1,
                        "total_stages": len(curriculum_manager.curriculum_stages),
                        "stage_performance_history": getattr(curriculum_manager, 'stage_performance_history', [])
                    }
                    with open(os.path.join(enhanced_artifacts_dir, "curriculum_progress.json"), "w", encoding="utf-8") as f:
                        json.dump(curriculum_stats, f, indent=2)
                    logger.info(f"Curriculum progress saved. Completed {curriculum_stats['stages_completed']}/{curriculum_stats['total_stages']} stages.")

                metrics = train_res.metrics if hasattr(train_res, 'metrics') else {}
                trainer.log_metrics("train_summary", metrics) 
                trainer.save_metrics("train_summary", os.path.join(enhanced_artifacts_dir, "final_train_metrics.json"))
                trainer.save_state() 
                logger.info(f"Training artifacts saved to {enhanced_artifacts_dir}")
                
                if wandb.run is not None:
                     wandb.log({
                        "training/final_status": "completed",
                        "training/total_steps_final": metrics.get("train_steps", trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else 0),
                        "training/final_loss_train": metrics.get("train_loss", 0),
                    }, step = trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else None)
                     
                     if experience_buffer and script_cfg.enable_experience_replay:
                        final_buffer_stats_wandb = experience_buffer.get_stats()
                        wandb.log({f"final_experience_buffer/{k}": v for k,v in final_buffer_stats_wandb.items()}, 
                                 step=trainer.state.global_step if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step') else None)

        except Exception as e_train:
            logger.error(f"Training loop failed: {e_train}", exc_info=True)
            if grpo_cfg.local_rank <= 0 and wandb.run is not None: 
                wandb.log({"training/final_status": "failed", "training/error_message": str(e_train)})
            raise e_train
            
    except Exception as e_main:
        logger.error(f"❌ 训练过程中发生错误: {e_main}", exc_info=True)
        
        # 清理资源
        try:
            if 'trainer' in locals() and trainer:
                del trainer
            if 'model' in locals() and model:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e_cleanup:
            logger.warning(f"清理资源时出错: {e_cleanup}")
        
        raise e_main
        
    finally:
        # 最终清理
        try:
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                gc.collect()
            logger.info(f"=== GRPO TRAINING SCRIPT FINISHED (Output: {script_cfg.output_dir if 'script_cfg' in locals() else 'unknown'}) ===")
            if wandb.run is not None and 'grpo_cfg' in locals() and grpo_cfg.local_rank <= 0: 
                wandb.finish()
        except Exception as e_finally:
            logger.warning(f"最终清理时出错: {e_finally}")
            
        logger.info("🧹 清理完成")

if __name__ == "__main__":
    main()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache(); gc.collect()