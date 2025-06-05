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
    # EnhancedCurriculumDebugCallback, # Will be imported from grpo_project.curriculum.callbacks
    Qwen3CompatibilityFixer,
    # RewardStabilityMonitor, # Will be imported from grpo_project.callbacks.monitoring
    integrate_enhanced_debugging
)
# FixedEnhancedCurriculumManager and setup_fixed_curriculum_manager will be imported from grpo_project.curriculum.manager
# from curriculum_debug_config import (
#     FixedEnhancedCurriculumManager,
#     setup_fixed_curriculum_manager
# )
from grpo_project.callbacks.monitoring import RewardStabilityMonitor # Moved from enhanced_debugging_and_fixes
from qwen3_prompt_fix import (
    parse_llm_completion_qwen3, # wrap_prompt_for_qwen3 is moved
    # qwen3_dataset_processing_pipeline is moved
    setup_qwen3_generation_config
)
from grpo_project.data.preprocessing import (
    VerilogDataPreprocessor, # Will be used for qwen3_dataset_processing_pipeline
    enhance_dataset_with_level_and_complexity # Moved from train.py
)
from grpo_project.data.validation import validate_dataset_for_curriculum # Moved from train.py
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

from utils import ( # Callbacks that were in utils.py are now moved to grpo_project.callbacks
    ExperienceBuffer, # This remains in utils for now, to be moved later
    monitor_advanced_stage_training # This remains in utils for now
    # DetailedWandbCallback from utils is now GenericWandbCallback
)
from grpo_project.callbacks.inference import EnhancedInferenceCallback, DetailedInferenceCallback
# Qwen3InferenceCallback was also moved but not directly used in train.py's top-level imports
from grpo_project.callbacks.monitoring import StepLoggingCallback, DetailedRewardCallback
# DetailedWandbCallback from train.py is now in grpo_project.callbacks.wandb_callbacks
from grpo_project.callbacks.wandb_callbacks import DetailedWandbCallback as TrainDetailedWandbCallback # Alias to avoid clash if any other DetailedWandbCallback is imported
from grpo_project.callbacks.persistence import CustomStatePersistenceCallback
from grpo_project.utils import (
    extract_module_info, validate_verilog_code,
    run_iverilog_simulation, validate_and_update_dataset_paths, enhance_prompt_func,
    assess_code_quality, assess_design_complexity,
    debug_checkpoint_contents, PeriodicStatusReporter
)
from grpo_project.utils.logging_utils import setup_global_logging
from grpo_project.rewards import RewardCalculator # New import for RewardCalculator
# from config import EnvConfig, ScriptConfig, EnhancedRewardConfig # Old imports
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig, OptimizedTrainingConfig, apply_optimized_config # New imports
# enhanced_curriculum content is now in grpo_project.curriculum
# from enhanced_curriculum import (
#     EnhancedCurriculumManager, CurriculumStageConfig,
#     create_default_curriculum_stages, create_custom_curriculum_stages
# )
from grpo_project.curriculum.manager import (
    EnhancedCurriculumManager, FixedEnhancedCurriculumManager,
    setup_enhanced_curriculum_manager, setup_fixed_curriculum_manager
)
from grpo_project.curriculum.stages import CurriculumStageConfig, create_default_curriculum_stages, create_custom_curriculum_stages
from grpo_project.curriculum.callbacks import CurriculumProgressCallback, OptimizedCurriculumCallback, EnhancedCurriculumDebugCallback

from datetime import datetime

logger = logging.getLogger(__name__)

# DetailedWandbCallback class definition was here. It has been moved to grpo_project/callbacks/wandb_callbacks.py

# CustomStatePersistenceCallback class definition was here. It has been moved to grpo_project/callbacks/persistence.py


def validate_dataset_for_curriculum(dataset: Dataset, script_cfg: ScriptConfig) -> bool: # Definition is now in grpo_project.data.validation
#     if dataset is None:
#         logger.error("Dataset is None, cannot validate for curriculum learning")
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
    
    # logger.info("Dataset validation for curriculum learning completed")
    # return True


# def setup_curriculum_manager(script_cfg: ScriptConfig, dataset: Dataset) -> Optional[EnhancedCurriculumManager]: # MOVED to grpo_project.curriculum.manager
#     if not script_cfg.enable_curriculum:
#         logger.info("Curriculum learning disabled.")
#         return None
#
#     if dataset is None: # This line and below were part of the original function, ensuring they are commented.
#         logger.error("Cannot setup curriculum manager with None dataset")
#         return None
#
#     has_level_info = False
#     if len(dataset) > 0:
#         first_example = dataset[0]
#         # Ensure 'level' exists and is not None. Also handle cases where it might be an empty string.
#         has_level_info = 'level' in first_example and first_example['level'] is not None and str(first_example['level']).strip() != ""
#
#     if not has_level_info:
#         logger.warning("Dataset does not contain valid 'level' field data. Curriculum learning will use complexity-only or default stages.")
#         # Avoid forcing complexity_only if user explicitly set another type, let it fall to default.
#         if script_cfg.curriculum_type == "dual_layer" or script_cfg.curriculum_type == "level_only":
#             logger.info(f"Switching curriculum type from '{script_cfg.curriculum_type}' to 'complexity_only' due to missing level info.")
#             script_cfg.curriculum_type = "complexity_only"
#
#     curriculum_stages_config_list = []
#
#     if script_cfg.curriculum_type == "dual_layer" and has_level_info:
#         logger.info(f"Dynamically generating dual_layer curriculum stages with focus: {script_cfg.curriculum_focus_levels}, emphasis: {script_cfg.curriculum_complexity_emphasis}")
#
#         level_counts_dist = {}
#         complexity_by_level_dist = {}
#         if dataset and len(dataset) > 0:
#             for example in dataset:
#                 level = example.get('level', 'unknown').lower()
#                 complexity = example.get('complexity_score', 5.0)
#                 level_counts_dist[level] = level_counts_dist.get(level, 0) + 1
#                 if level not in complexity_by_level_dist:
#                     complexity_by_level_dist[level] = []
#                 complexity_by_level_dist[level].append(complexity)
#
#         dataset_distribution_for_stages = {
#             'level_counts': level_counts_dist,
#             'complexity_by_level': complexity_by_level_dist,
#             'total_samples': len(dataset) if dataset else 0
#         }
#
#         curriculum_stages_config_list = create_custom_curriculum_stages(
#             dataset_distribution=dataset_distribution_for_stages,
#             focus_levels=script_cfg.curriculum_focus_levels, # Assumed to be List[str] from HfArgumentParser
#             complexity_emphasis=script_cfg.curriculum_complexity_emphasis
#         )
#         logger.info(f"Generated {len(curriculum_stages_config_list)} custom stages for dual_layer.")
#     else:
#         if script_cfg.curriculum_type != "dual_layer" and has_level_info : # e.g. level_only, but we are simplifying
#             logger.info(f"Curriculum type is '{script_cfg.curriculum_type}'. Using default stages as specific logic for this type (other than dual_layer) is not implemented here, or conditions not met.")
#         elif not has_level_info and script_cfg.curriculum_type != "complexity_only": # User might have set dual_layer/level_only but data is missing
#             logger.info(f"Dataset lacks level information for '{script_cfg.curriculum_type}'. Falling back to default (likely complexity-based) stages.")
#
#         curriculum_stages_config_list = create_default_curriculum_stages() # Handles complexity_only or serves as fallback
#         logger.info(f"Generated {len(curriculum_stages_config_list)} default stages (type: {script_cfg.curriculum_type}).")
#
#     if curriculum_stages_config_list:
#         for stage_config_item in curriculum_stages_config_list:
#             if isinstance(stage_config_item, CurriculumStageConfig):
#                 stage_config_item.min_evaluations = 10
#             else:
#                 logger.warning(f"Encountered non-CurriculumStageConfig item in list: {type(stage_config_item)}")
#         logger.info(f"Ensured min_evaluations is 10 for all {len(curriculum_stages_config_list)} stages.")
#     else:
#         logger.warning("No curriculum stages were generated. Curriculum learning might be ineffective.")
#         return None
#
#     if not curriculum_stages_config_list:
#         logger.error("No curriculum stages defined after attempting generation. Disabling curriculum learning.")
#         return None
#
#     return EnhancedCurriculumManager(curriculum_stages_config_list, dataset)


# def enhance_dataset_with_level_and_complexity(dataset: Dataset) -> Dataset: # Moved to grpo_project.data.preprocessing
    # if not dataset or len(dataset) == 0:
    #     logger.warning("enhance_dataset_with_level_and_complexity: Received empty or None dataset.")
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
        # logger.error(f"Failed to enhance dataset: {e}", exc_info=True)
        # return dataset

# calculate_enhanced_rewards_for_single_prompt - MOVED to grpo_project.rewards.calculator.RewardCalculator._calculate_single_reward

# Custom callback for curriculum learning progression (defined at module level)
# class CurriculumProgressCallback(TrainerCallback): # MOVED to grpo_project.curriculum.callbacks
#     def __init__(self, curriculum_manager, trainer_ref, output_dir):
#         self.curriculum_manager = curriculum_manager
#         self.trainer_ref = trainer_ref
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
            # self.last_locally_logged_stage_idx = current_stage_idx_for_local_log


# 3. 优化的课程学习回调
# class OptimizedCurriculumCallback(DefaultFlowCallback): # MOVED to grpo_project.curriculum.callbacks
#     """优化的课程学习回调，包含动态难度调整"""
    
#     def __init__(self, curriculum_manager, trainer_ref, output_dir):
        super().__init__()
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        # self.output_dir = output_dir
        # self.difficulty_adjuster = DynamicDifficultyAdjuster(curriculum_manager) # DynamicDifficultyAdjuster is not defined here
        # self.performance_history = []
        
    # def on_log(self, args, state, control, logs=None, **kwargs):
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
            # logger.warning(f"保存课程状态失败: {e}")
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
        # self.status_log_path = os.path.join(output_dir, "training_status.txt") # Definition is in reporting_utils.py
        
    # def report_status(self, step, trainer_state, curriculum_manager=None, experience_buffer=None): # Definition is in reporting_utils.py
    #     """生成定期状态报告"""
    #     if step % self.report_interval != 0:
    #         return
        
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    #     status_report = f"""
# ========================================
# 训练状态报告 - 步数 {step}
# 时间: {timestamp}
# ========================================

# 📈 训练进度:
#   - 当前步数: {step}
#   - 最大步数: {trainer_state.max_steps if trainer_state.max_steps > 0 else '无限制'}
#   - 完成百分比: {(step/trainer_state.max_steps*100) if trainer_state.max_steps > 0 else 'N/A'}%

# 📚 课程学习状态:"""
        
    #     if curriculum_manager:
    #         current_stage = curriculum_manager.current_stage
    #         total_stages = len(curriculum_manager.curriculum_stages)
            
    #         status_report += f"""
#   - 当前阶段: {current_stage}/{total_stages-1}
#   - 阶段名称: {curriculum_manager.curriculum_stages[current_stage].name if current_stage < total_stages else 'final'}
#   - 阶段进度: {len(curriculum_manager.stage_performance_history)}次评估
#   - 数据集大小: {len(curriculum_manager.get_current_stage_dataset())}"""
    #     else:
    #         status_report += "\n  - 课程学习: 未启用"
        
    #     status_report += f"""

# 🔄 经验回放状态:"""
        
    #     if experience_buffer:
    #         buffer_stats = experience_buffer.get_stats()
    #         status_report += f"""
#   - 缓存大小: {buffer_stats['size']}/{experience_buffer.max_size}
#   - 平均奖励: {buffer_stats['mean_reward']:.2f}
#   - 最高奖励: {buffer_stats['max_reward']:.2f}"""
    #     else:
    #         status_report += "\n  - 经验回放: 未启用"
        
    #     # 最近的损失信息
    #     if trainer_state.log_history:
    #         recent_loss = trainer_state.log_history[-1].get('loss', 'N/A')
    #         status_report += f"""

# 📊 最近指标:
#   - 训练损失: {recent_loss}
#   - 学习率: {trainer_state.log_history[-1].get('learning_rate', 'N/A')}"""
        
    #     status_report += f"""

# ========================================
# """
        
    #     # 输出到控制台
    #     logger.info(status_report)
        
    #     # 保存到文件
    #     with open(self.status_log_path, 'a') as f:
    #         f.write(status_report + "\n")

# DetailedRewardCallback class definition was here. It has been moved to grpo_project/callbacks/monitoring.py

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
            wandb_callback_instance = None # Renamed to avoid conflict with outer scope wandb_callback
            # 🔧 修复wandb接续
            if grpo_cfg.local_rank <= 0 and "wandb" in grpo_cfg.report_to:
                # 设置wandb的恢复参数
                wandb_resume_mode = "allow"  # 允许恢复
                wandb_run_id = None
                
                if is_resuming: # is_resuming should be defined in the outer scope of main
                    # 尝试从原始run_name推导wandb run_id
                    wandb_run_id = sanitized_run_name # sanitized_run_name should be defined in the outer scope of main
                    wandb_resume_mode = "allow"  # 尝试恢复，如果失败则创建新的
                    logger.info(f"🔗 尝试恢复wandb run: {wandb_run_id}")
                
                # 设置环境变量让wandb知道要恢复
                if wandb_run_id:
                    os.environ["WANDB_RUN_ID"] = wandb_run_id
                    os.environ["WANDB_RESUME"] = wandb_resume_mode
                
                # Use the aliased TrainDetailedWandbCallback
                wandb_callback_instance = TrainDetailedWandbCallback(env_cfg, script_cfg, reward_cfg, experience_buffer)
                callbacks_list.append(wandb_callback_instance)
                logger.info(f"✅ wandb回调已创建 - resume模式: {wandb_resume_mode}")
            
            return callbacks_list, wandb_callback_instance # Return the instance
        
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
        # Setup logging using the new utility function
        setup_global_logging(
            log_level=grpo_cfg.get_process_log_level(),
            log_file_path=log_file_path,
            local_rank=grpo_cfg.local_rank
        )
        # global logger # logger is now setup globally by setup_global_logging
        logger = logging.getLogger(__name__) # Get the logger instance for the current file

        # 设置状态报告器 (Now imported from grpo_project.utils)
        status_reporter = PeriodicStatusReporter(script_cfg.output_dir, report_interval=50)

        # transformers_logger setup is now handled by setup_global_logging

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
            # dataset = qwen3_dataset_processing_pipeline(dataset_raw, dataset_dir, script_cfg) # Now use VerilogDataPreprocessor
            data_preprocessor = VerilogDataPreprocessor(script_cfg_val=script_cfg) # script_cfg_val is script_cfg in this context
            dataset = data_preprocessor.process_dataset_pipeline(dataset_raw, ds_dir=dataset_dir) # ds_dir is dataset_dir

            del dataset_raw; gc.collect()

            # enhance_dataset_with_level_and_complexity is now part of the pipeline in VerilogDataPreprocessor
            # dataset = enhance_dataset_with_level_and_complexity(dataset)

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
            # Use the new setup function from the curriculum module
            curriculum_manager = setup_fixed_curriculum_manager(script_cfg, dataset)
            if curriculum_manager:
                logger.info(f"✅ 课程学习已启用 using curriculum manager: {type(curriculum_manager).__name__}")
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
            # current_reward_cfg, current_script_cfg, current_experience_buffer, etc. are passed via closure or kwargs
            # Instantiate RewardCalculator (simulator can be None for now, or a basic version if available)
            # Note: reward_cfg would be available in the outer scope of main()
            reward_calculator_instance = RewardCalculator(reward_config=kwargs.get('reward_config_obj'), simulator=None)

            # Call the batch reward calculation method
            # The necessary arguments (prompts, completions, etc.) are in kwargs_reward from GRPOTrainer
            # Ensure that the keys used here match what GRPOTrainer provides in kwargs_reward
            # GRPOTrainer typically passes columns of the dataset as keyword arguments.
            # The 'prompts' and 'completions' are standard. Others like 'testbench_path' need to be in the dataset.
            
            # These are the expected keys from the dataset that GRPOTrainer will pass in kwargs_reward
            # (as per the original calculate_enhanced_rewards_for_single_prompt and its usage context)
            # The GRPOTrainer will pass columns of the dataset by their names.
            # We need to ensure the dataset used by GRPOTrainer has these columns.
            # The `enhanced_batch_reward_calculator` is called by `reward_func_with_context`
            # which receives dataset columns in `kwargs_reward`.
            
            # The keys in `kwargs_reward` are directly from the dataset columns.
            # The `RewardCalculator.calculate_batch_rewards` expects specific argument names.
            # We need to map them correctly.
            
            return reward_calculator_instance.calculate_batch_rewards(
                prompts=kwargs_reward.get('prompt', []), # GRPOTrainer passes the 'prompt' column values
                completions=completions, # This `completions` is directly passed to `enhanced_batch_reward_calculator`
                testbench_paths=kwargs_reward.get('testbench_path', []),
                expected_total_tests_list=kwargs_reward.get('expected_total_tests', []),
                reference_verilog_paths=kwargs_reward.get('reference_verilog_path', []),
                original_enhanced_prompts=kwargs_reward.get('original_enhanced_prompt', None), # This should be in dataset
                training_step=kwargs.get('training_step', 0),
                output_dir_for_debug=kwargs.get('output_dir', None)
            )
        
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