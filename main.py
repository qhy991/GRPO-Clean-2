import os
import logging
import numpy as np
import torch
import gc
import sys
from dataclasses import asdict
import json # For logging configs
from typing import Dict, Any, Optional, List
# --- BEGIN: PyTorch Safe Unpickling Configuration ---
logger_temp = logging.getLogger(__name__ + "_startup")
try:
    from numpy.core.multiarray import _reconstruct
    from numpy import dtype as numpy_dtype
    from numpy.dtypes import UInt32DType

    safe_globals_list = []
    if callable(_reconstruct): safe_globals_list.append(_reconstruct)
    else: logger_temp.warning("_reconstruct is not callable.")
    if isinstance(numpy_dtype, type): safe_globals_list.append(numpy_dtype)
    else: logger_temp.warning("numpy.dtype is not a type.")
    if isinstance(np.ndarray, type): safe_globals_list.append(np.ndarray)
    else: logger_temp.warning("np.ndarray is not a type.")
    if isinstance(UInt32DType, type): safe_globals_list.append(UInt32DType)
    else: logger_temp.warning("UInt32DType is not a type.")

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

    if safe_globals_list:
        torch.serialization.add_safe_globals(safe_globals_list)
        logger_temp.info(f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items, including {added_scalar_types_count} scalar types.")
    else:
        logger_temp.warning("safe_globals_list is empty before calling torch.serialization.add_safe_globals.")

except ImportError as e:
    logger_temp.warning(f"Failed to import NumPy modules for torch safe globals: {e}.")
except AttributeError as e:
    logger_temp.warning(f"Attribute error accessing NumPy properties for torch safe globals: {e}.")
except Exception as e_globals:
    logger_temp.error(f"An unexpected error occurred while setting up torch safe globals: {e_globals}", exc_info=True)
# --- END: PyTorch Safe Unpickling Configuration ---

from transformers import HfArgumentParser
from trl import GRPOConfig

# Configuration imports (assuming these are now the central config files)
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from grpo_project.utils.logging import setup_global_logging # Corrected import
from grpo_project.utils.reporting_utils import PeriodicStatusReporter
from grpo_project.core.models import ModelManager
from grpo_project.data.dataset import load_and_prepare_dataset # Main data function
from grpo_project.rewards.calculator import RewardCalculator
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager # Or EnhancedCurriculumManager
from grpo_project.utils import ExperienceBuffer # Assuming this path is correct

# Callbacks
from grpo_project.callbacks.monitoring import StepLoggingCallback, DetailedRewardCallback, RewardStabilityMonitor
from grpo_project.callbacks.persistence import CustomStatePersistenceCallback
from grpo_project.callbacks.inference import DetailedInferenceCallback
from grpo_project.callbacks.wandb import DetailedWandbCallback as TrainDetailedWandbCallback
from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback


# Logger will be configured by setup_global_logging
logger = logging.getLogger(__name__)

class GRPOTrainingPipeline:
    def __init__(self):
        # 1. Load Configurations
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        self.env_cfg, self.script_cfg, self.reward_cfg, self.grpo_cfg = parser.parse_args_into_dataclasses()

        # EnvConfig's __post_init__ already sets up environment variables (proxies, wandb, cache)
        # Now, setup global logging using the parsed configs
        self._setup_logging()

        logger.info("GRPОTrainingPipeline initialized with configurations.")
        self._log_configs()

        # Initialize core components (placeholders for now, will be filled in `train` or here)
        self.model_manager = ModelManager(
            script_cfg=self.script_cfg,
            grpo_cfg=self.grpo_cfg,
            model_name_or_path=self.script_cfg.model_name_or_path,
            cache_dir=self.env_cfg.cache_dir
        )
        # VerilogDataPreprocessor is used within load_and_prepare_dataset
        # DataValidator/validate_dataset_for_curriculum is used within load_and_prepare_dataset

        self.reward_calculator = RewardCalculator(
            reward_config=self.reward_cfg,
            simulator=None # Default VerilogSimulator will be created internally
        )

        self.curriculum_manager = None # Will be initialized after dataset is loaded
        self.experience_buffer = None # Will be initialized if enabled
        self.callbacks = []
        self.status_reporter = PeriodicStatusReporter(self.script_cfg.output_dir, report_interval=50) # output_dir needs to be set

        self.model = None
        self.tokenizer = None
        self.trainer = None # GRPOTrainer from trl

    def _setup_logging(self):
        # Determine actual output directory (logic moved from TrainingOrchestrator)
        # This needs self.env_cfg.output_dir_base and potentially a run name strategy
        # For now, directly use grpo_cfg.output_dir which should be set by TrainingArguments default or user.
        # However, the previous TrainingOrchestrator had more complex logic for this.
        # Let's replicate part of that logic here for consistency.

        import re # For sanitized_run_name
        from datetime import datetime

        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME") # EnvConfig might set this
        if not run_specific_name_from_env:
             run_specific_name_from_env = f"{self.env_cfg.wandb_run_name_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}" \
                if self.env_cfg.wandb_run_name_prefix else f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)

        # Determine if resuming
        is_resuming = (
            self.grpo_cfg.resume_from_checkpoint and
            isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
            os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
        )

        if is_resuming:
            actual_output_dir = os.path.dirname(self.grpo_cfg.resume_from_checkpoint)
            # If resuming, run name might be inferred from checkpoint path's parent dir name
            # This ensures consistency if WANDB_RUN_NAME was based on the original output dir.
            # sanitized_run_name = os.path.basename(actual_output_dir)
        else:
            actual_output_dir = os.path.join(self.env_cfg.output_dir_base, sanitized_run_name)

        if self.grpo_cfg.local_rank <= 0:
            os.makedirs(actual_output_dir, exist_ok=True)

        # Update config objects with the determined output directory
        self.grpo_cfg.output_dir = actual_output_dir
        # ScriptConfig needs an output_dir too for its components (e.g. callbacks, status_reporter)
        # This was previously handled by TrainingOrchestrator setting script_cfg.output_dir
        # setattr(self.script_cfg, 'output_dir', actual_output_dir) # Add output_dir attribute if not present
        self.script_cfg.output_dir = actual_output_dir # Assuming ScriptConfig now has an output_dir field or we add it

        log_file_path = os.path.join(self.grpo_cfg.output_dir, "grpo_pipeline_log.txt")
        setup_global_logging(
            log_level=self.grpo_cfg.get_process_log_level(),
            log_file_path=log_file_path,
            local_rank=self.grpo_cfg.local_rank
        )
        logger.info(f"Global logging set up. Log file: {log_file_path}")
        logger.info(f"Output directory: {self.grpo_cfg.output_dir}")


    def _log_configs(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"EnvConfig: \n{json.dumps(asdict(self.env_cfg), indent=2)}")
            logger.debug(f"ScriptConfig: \n{json.dumps(asdict(self.script_cfg), indent=2)}")
            logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(self.reward_cfg), indent=2)}")
            logger.debug(f"GRPOConfig (TrainingArguments): \n{self.grpo_cfg.to_json_string()}")


    def _initialize_components(self, dataset_processed):
        """ Initialize components that depend on the processed dataset or other configs """
        if self.script_cfg.enable_experience_replay:
            self.experience_buffer = ExperienceBuffer(max_size=self.script_cfg.experience_buffer_size)
            logger.info(f"Experience buffer initialized (size: {self.script_cfg.experience_buffer_size}).")
            if self.grpo_cfg.resume_from_checkpoint and isinstance(self.grpo_cfg.resume_from_checkpoint, str) and os.path.isdir(self.grpo_cfg.resume_from_checkpoint): # Check if resuming
                buffer_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_experience_buffer_state.json")
                if os.path.exists(buffer_state_path):
                    try:
                        logger.info(f"Attempting to load experience buffer state from: {buffer_state_path}")
                        with open(buffer_state_path, "r", encoding="utf-8") as f: state_data = json.load(f)
                        self.experience_buffer.load_buffer_state(state_data)
                    except Exception as e: logger.error(f"Failed to load experience buffer state: {e}")
                else: logger.warning(f"Experience buffer state file not found at {buffer_state_path}.")

        if self.script_cfg.enable_curriculum:
            self.curriculum_manager = setup_fixed_curriculum_manager(self.script_cfg, dataset_processed)
            if self.curriculum_manager:
                logger.info(f"Curriculum learning enabled: {type(self.curriculum_manager).__name__}.")
                if self.grpo_cfg.resume_from_checkpoint and isinstance(self.grpo_cfg.resume_from_checkpoint, str) and os.path.isdir(self.grpo_cfg.resume_from_checkpoint): # Check if resuming
                    curriculum_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_curriculum_state.json")
                    if os.path.exists(curriculum_state_path):
                        try:
                            logger.info(f"Loading curriculum state from: {curriculum_state_path}")
                            with open(curriculum_state_path, "r", encoding="utf-8") as f: state_data = json.load(f)
                            self.curriculum_manager.load_curriculum_state(state_data)
                        except Exception as e: logger.error(f"Failed to load curriculum state: {e}")
                    else: logger.warning(f"Curriculum state file not found at {curriculum_state_path}.")
            else:
                logger.warning("Curriculum manager setup returned None. Curriculum learning disabled.")

        # Setup Callbacks
        self.callbacks.append(StepLoggingCallback())
        self.callbacks.append(DetailedRewardCallback(self.script_cfg.output_dir)) # script_cfg.output_dir must be set
        self.callbacks.append(RewardStabilityMonitor(self.script_cfg.output_dir))

        if self.curriculum_manager:
            self.callbacks.append(CurriculumProgressCallback(self.curriculum_manager, None, self.script_cfg.output_dir))
            self.callbacks.append(EnhancedCurriculumDebugCallback(self.curriculum_manager, None, self.script_cfg.output_dir))

        self.callbacks.append(CustomStatePersistenceCallback(self.curriculum_manager, self.experience_buffer, self.script_cfg))

        # DetailedInferenceCallback needs tokenizer
        if self.tokenizer and dataset_processed:
            sample_dataset_for_inf_cb = dataset_processed.select(
                range(min(len(dataset_processed), self.script_cfg.callback_num_samples * 5))
            ) if len(dataset_processed) > 0 else None

            if sample_dataset_for_inf_cb and len(sample_dataset_for_inf_cb) > 0:
                self.callbacks.append(DetailedInferenceCallback(
                    tokenizer=self.tokenizer, eval_dataset=sample_dataset_for_inf_cb,
                    num_samples=self.script_cfg.callback_num_samples,
                    eval_every_n_steps=self.script_cfg.callback_eval_every_n_steps,
                    max_new_tokens=self.grpo_cfg.max_completion_length,
                    max_seq_length=self.script_cfg.max_seq_length,
                    experience_buffer=self.experience_buffer,
                    output_dir=self.script_cfg.output_dir
                ))
            else:
                logger.warning("DetailedInferenceCallback will not run due to insufficient sample data.")

        if self.grpo_cfg.local_rank <= 0 and "wandb" in self.grpo_cfg.report_to:
            # Wandb setup
            # ... (similar logic as in TrainingOrchestrator for WANDB_RUN_ID, WANDB_RESUME)
            self.callbacks.append(TrainDetailedWandbCallback(self.env_cfg, self.script_cfg, self.reward_cfg, self.experience_buffer))
            logger.info("TrainDetailedWandbCallback added.")

        logger.info(f"Total callbacks prepared: {len(self.callbacks)}")


    def get_reward_function(self):
        # Closure to capture self.reward_calculator, self.script_cfg, etc.
        def reward_fn_closure(prompts: List[str], completions: List[str], **kwargs_from_trainer_dataset) -> List[float]:
            current_training_step = self.trainer.state.global_step if self.trainer and self.trainer.state else 0

            batch_rewards_args = {
                "prompts": prompts,
                "completions": completions,
                "testbench_paths": kwargs_from_trainer_dataset.get('testbench_path', []),
                "expected_total_tests_list": kwargs_from_trainer_dataset.get('expected_total_tests', []),
                "reference_verilog_paths": kwargs_from_trainer_dataset.get('reference_verilog_path', []),
                "original_enhanced_prompts": kwargs_from_trainer_dataset.get('original_enhanced_prompt'),
                "training_step": current_training_step,
                "output_dir_for_debug": self.script_cfg.output_dir,
                # "experience_buffer_obj": self.experience_buffer, # RewardCalculator doesn't take this directly
                # "wandb_callback_obj": next((cb for cb in self.callbacks if isinstance(cb, TrainDetailedWandbCallback)), None),
                # "script_config_obj": self.script_cfg
            }
            # RewardCalculator's calculate_batch_rewards doesn't take wandb_callback or experience_buffer directly.
            # Those are used by the callbacks themselves.
            rewards_list, _ = self.reward_calculator.calculate_batch_rewards(**batch_rewards_args)
            return rewards_list
        return reward_fn_closure

    def train(self):
        logger.info("Starting GRPО training process...")

        # Load and preprocess data
        logger.info("Loading and preparing dataset...")
        # The tokenizer is needed for data preprocessing by VerilogDataPreprocessor
        # So model/tokenizer setup must come before data prep.

        logger.info("Setting up model and tokenizer...")
        self.model, self.tokenizer = self.model_manager.setup_model_and_tokenizer()
        self.model = self.model_manager.apply_peft_adapter(
            model=self.model,
            is_resuming=(self.grpo_cfg.resume_from_checkpoint is not None), # Simplified is_resuming logic
            resume_checkpoint_path=str(self.grpo_cfg.resume_from_checkpoint) if self.grpo_cfg.resume_from_checkpoint else None
        )

        dataset_processed = load_and_prepare_dataset(
            script_cfg=self.script_cfg,
            env_cfg=self.env_cfg,
            tokenizer=self.tokenizer
        )

        # Initialize components that depend on data (curriculum, some callbacks)
        self._initialize_components(dataset_processed) # Initializes curriculum_manager, experience_buffer, callbacks

        dataset_for_trainer = dataset_processed
        if self.curriculum_manager:
            dataset_for_trainer = self.curriculum_manager.get_current_stage_dataset()
            logger.info(f"Using curriculum dataset for trainer: {len(dataset_for_trainer)} samples from stage '{self.curriculum_manager.get_current_stage_name()}'.")
        else:
            logger.info(f"Using full processed dataset for trainer: {len(dataset_for_trainer)} samples.")

        if not dataset_for_trainer or len(dataset_for_trainer) == 0:
            logger.error("Dataset for trainer is empty. Aborting.")
            return

        # Create GRPОTrainer instance
        from trl import GRPOTrainer # Ensure GRPOTrainer is imported
        self.trainer = GRPOTrainer(
            model=self.model,
            args=self.grpo_cfg,
            train_dataset=dataset_for_trainer,
            reward_funcs=[self.get_reward_function()],
            callbacks=self.callbacks
        )

        # Set trainer_ref for callbacks that need it (CurriculumProgressCallback, etc.)
        for cb in self.callbacks:
            if hasattr(cb, 'trainer_ref') and cb.trainer_ref is None:
                cb.trainer_ref = self.trainer
                logger.info(f"Set trainer_ref for {type(cb).__name__}")

        logger.info("GRPОTrainer instance created. Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=self.grpo_cfg.resume_from_checkpoint)
        logger.info("Training finished.")

        # Save artifacts
        if self.grpo_cfg.local_rank <= 0:
            logger.info("Saving final model adapter and training artifacts...")
            final_model_dir = os.path.join(self.grpo_cfg.output_dir, "final_model_adapter")
            self.trainer.save_model(final_model_dir)

            # Further artifact saving (metrics, state, etc.) can be added here
            # (similar to TrainingOrchestrator._save_training_artifacts)
            metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
            self.trainer.log_metrics("train_summary", metrics)
            self.trainer.save_metrics("train_summary", os.path.join(self.grpo_cfg.output_dir, "final_train_metrics.json"))
            self.trainer.save_state()
            logger.info(f"Training artifacts saved to {self.grpo_cfg.output_dir}")

        self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up GRPOTrainingPipeline resources...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Any other specific cleanup for components
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    try:
        pipeline = GRPOTrainingPipeline()
        pipeline.train()
    except Exception as e:
        logger.error(f"Unhandled exception in GRPOTrainingPipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("GRPОTrainingPipeline script execution finished.")
