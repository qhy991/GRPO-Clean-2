# main.py - 完整修复版本
import os
import logging
import numpy as np
import torch
import gc
import sys
from dataclasses import asdict
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# --- BEGIN: PyTorch Safe Unpickling Configuration ---
logger_temp = logging.getLogger(__name__ + "_startup")
try:
    from numpy.core.multiarray import _reconstruct
    from numpy import dtype as numpy_dtype
    from numpy.dtypes import UInt32DType

    safe_globals_list = []
    if callable(_reconstruct): safe_globals_list.append(_reconstruct)
    if isinstance(numpy_dtype, type): safe_globals_list.append(numpy_dtype)
    if isinstance(np.ndarray, type): safe_globals_list.append(np.ndarray)
    if isinstance(UInt32DType, type): safe_globals_list.append(UInt32DType)

    # Add numpy scalar types
    numpy_scalar_types_to_add = [
        np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
        np.int_, np.uint, np.longlong, np.ulonglong,
        np.half, np.float16, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ]
    
    for nt_class in numpy_scalar_types_to_add:
        if isinstance(nt_class, type):
            safe_globals_list.append(nt_class)

    if safe_globals_list:
        torch.serialization.add_safe_globals(safe_globals_list)
        logger_temp.info(f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items.")

except Exception as e_globals:
    logger_temp.error(f"Error setting up torch safe globals: {e_globals}", exc_info=True)
# --- END: PyTorch Safe Unpickling Configuration ---

from transformers import HfArgumentParser
from trl import GRPOConfig

# Configuration imports
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from grpo_project.utils.logging import setup_global_logging
from grpo_project.utils.reporting_utils import PeriodicStatusReporter
from grpo_project.core.models import ModelManager
from grpo_project.data.dataset import load_and_prepare_dataset
from grpo_project.rewards.calculator import RewardCalculator
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
from grpo_project.utils import ExperienceBuffer

# Callbacks
from grpo_project.callbacks.monitoring import StepLoggingCallback, DetailedRewardCallback, RewardStabilityMonitor
from grpo_project.callbacks.persistence import CustomStatePersistenceCallback
from grpo_project.callbacks.inference import DetailedInferenceCallback
from grpo_project.callbacks.wandb import DetailedWandbCallback as TrainDetailedWandbCallback
from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback

logger = logging.getLogger(__name__)

class GRPOTrainingPipeline:
    def __init__(self):
        # 1. Load Configurations
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        self.env_cfg, self.script_cfg, self.reward_cfg, self.grpo_cfg = parser.parse_args_into_dataclasses()

        # Setup logging first
        self._setup_logging()
        logger.info("GRPOTrainingPipeline initialized.")
        self._log_configs()

        # Initialize core components
        self.model_manager = ModelManager(
            script_cfg=self.script_cfg,
            grpo_cfg=self.grpo_cfg,
            model_name_or_path=self.script_cfg.model_name_or_path,
            cache_dir=self.env_cfg.cache_dir
        )

        self.reward_calculator = RewardCalculator(
            reward_config=self.reward_cfg,
            simulator=None
        )

        self.curriculum_manager = None
        self.experience_buffer = None
        self.callbacks = []
        self.status_reporter = PeriodicStatusReporter(self.grpo_cfg.output_dir, report_interval=50)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _setup_logging(self):
        """Setup logging with improved error handling"""
        import re
        from datetime import datetime

        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
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
        else:
            actual_output_dir = os.path.join(self.env_cfg.output_dir_base, sanitized_run_name)

        if self.grpo_cfg.local_rank <= 0:
            os.makedirs(actual_output_dir, exist_ok=True)

        # Update config objects
        self.grpo_cfg.output_dir = actual_output_dir
        self.script_cfg.output_dir = actual_output_dir

        log_file_path = os.path.join(self.grpo_cfg.output_dir, "grpo_pipeline_log.txt")
        setup_global_logging(
            log_level=self.grpo_cfg.get_process_log_level(),
            log_file_path=log_file_path,
            local_rank=self.grpo_cfg.local_rank
        )
        logger.info(f"Global logging set up. Output directory: {self.grpo_cfg.output_dir}")

    def _log_configs(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"EnvConfig: \n{json.dumps(asdict(self.env_cfg), indent=2)}")
            logger.debug(f"ScriptConfig: \n{json.dumps(asdict(self.script_cfg), indent=2)}")
            logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(self.reward_cfg), indent=2)}")

    def _setup_model_and_tokenizer(self):
        """🔧 修复：正确设置模型和tokenizer"""
        try:
            logger.info("Setting up model and tokenizer...")
            
            # 🔧 关键修复：调用ModelManager的方法
            self.model, self.tokenizer = self.model_manager.setup_model_and_tokenizer()
            
            # 应用PEFT适配器
            is_resuming = (
                self.grpo_cfg.resume_from_checkpoint and 
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and 
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
            )
            
            self.model = self.model_manager.apply_peft_adapter(
                model=self.model,
                is_resuming=is_resuming,
                resume_checkpoint_path=str(self.grpo_cfg.resume_from_checkpoint) if is_resuming else None
            )
            
            logger.info("✅ Model and tokenizer setup completed successfully.")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to setup model and tokenizer: {e}", exc_info=True)
            raise

    def _initialize_components(self, dataset_processed):
        """Initialize components that depend on the processed dataset"""
        try:
            logger.info("Initializing training components...")
            
            # Experience buffer setup
            if self.script_cfg.enable_experience_replay:
                self.experience_buffer = ExperienceBuffer(max_size=self.script_cfg.experience_buffer_size)
                logger.info(f"Experience buffer initialized (size: {self.script_cfg.experience_buffer_size}).")
                
                # Restore experience buffer state if resuming
                if self.grpo_cfg.resume_from_checkpoint and os.path.isdir(self.grpo_cfg.resume_from_checkpoint):
                    buffer_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_experience_buffer_state.json")
                    if os.path.exists(buffer_state_path):
                        try:
                            logger.info(f"Loading experience buffer state from: {buffer_state_path}")
                            with open(buffer_state_path, "r", encoding="utf-8") as f:
                                state_data = json.load(f)
                            self.experience_buffer.load_buffer_state(state_data)
                            logger.info("✅ Experience buffer state loaded successfully.")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load experience buffer state: {e}")

            # Curriculum learning setup
            if self.script_cfg.enable_curriculum:
                self.curriculum_manager = setup_fixed_curriculum_manager(self.script_cfg, dataset_processed)
                if self.curriculum_manager:
                    logger.info(f"Curriculum learning enabled: {type(self.curriculum_manager).__name__}.")
                    
                    # Restore curriculum state if resuming
                    if self.grpo_cfg.resume_from_checkpoint and os.path.isdir(self.grpo_cfg.resume_from_checkpoint):
                        curriculum_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_curriculum_state.json")
                        if os.path.exists(curriculum_state_path):
                            try:
                                logger.info(f"Loading curriculum state from: {curriculum_state_path}")
                                with open(curriculum_state_path, "r", encoding="utf-8") as f:
                                    state_data = json.load(f)
                                self.curriculum_manager.load_curriculum_state(state_data)
                                logger.info("✅ Curriculum state loaded successfully.")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load curriculum state: {e}")

            # Setup callbacks
            self._setup_callbacks(dataset_processed)
            logger.info("✅ All components initialized successfully.")
            
        except Exception as e:
            logger.error(f"❌ Error during component initialization: {e}", exc_info=True)
            raise

    def _setup_callbacks(self, dataset_processed):
        """Setup all callbacks"""
        try:
            self.callbacks = []
            
            # Basic monitoring callbacks
            self.callbacks.append(StepLoggingCallback())
            self.callbacks.append(DetailedRewardCallback(self.script_cfg.output_dir))
            self.callbacks.append(RewardStabilityMonitor(self.script_cfg.output_dir))

            # Curriculum callbacks
            if self.curriculum_manager:
                self.callbacks.append(CurriculumProgressCallback(self.curriculum_manager, None, self.script_cfg.output_dir))
                self.callbacks.append(EnhancedCurriculumDebugCallback(self.curriculum_manager, None, self.script_cfg.output_dir))

            # State persistence callback
            self.callbacks.append(CustomStatePersistenceCallback(self.curriculum_manager, self.experience_buffer, self.script_cfg))

            # Inference callback (needs tokenizer)
            if self.tokenizer and dataset_processed and len(dataset_processed) > 0:
                sample_dataset_for_inf_cb = dataset_processed.select(
                    range(min(len(dataset_processed), self.script_cfg.callback_num_samples * 5))
                )

                if len(sample_dataset_for_inf_cb) > 0:
                    self.callbacks.append(DetailedInferenceCallback(
                        tokenizer=self.tokenizer, 
                        eval_dataset=sample_dataset_for_inf_cb,
                        num_samples=self.script_cfg.callback_num_samples,
                        eval_every_n_steps=self.script_cfg.callback_eval_every_n_steps,
                        max_new_tokens=self.grpo_cfg.max_completion_length,
                        max_seq_length=self.script_cfg.max_seq_length,
                        experience_buffer=self.experience_buffer,
                        output_dir=self.script_cfg.output_dir
                    ))
                    logger.info(f"DetailedInferenceCallback added with {len(sample_dataset_for_inf_cb)} samples.")
                else:
                    logger.warning("⚠️ No samples available for DetailedInferenceCallback.")

            # W&B callback
            if self.grpo_cfg.local_rank <= 0 and "wandb" in self.grpo_cfg.report_to:
                self.callbacks.append(TrainDetailedWandbCallback(
                    self.env_cfg, self.script_cfg, self.reward_cfg, self.experience_buffer
                ))
                logger.info("✅ TrainDetailedWandbCallback added.")

            logger.info(f"Total callbacks prepared: {len(self.callbacks)}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up callbacks: {e}", exc_info=True)
            # 至少保证基本的回调
            self.callbacks = [StepLoggingCallback()]
            logger.warning("⚠️ Using minimal callback setup due to errors.")

    def get_reward_function(self):
        """Create reward function closure"""
        def reward_fn_closure(prompts: List[str], completions: List[str], **kwargs_from_trainer_dataset) -> List[float]:
            try:
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
                }
                
                rewards_list, _ = self.reward_calculator.calculate_batch_rewards(**batch_rewards_args)
                return rewards_list
                
            except Exception as e:
                logger.error(f"❌ Error in reward function: {e}", exc_info=True)
                # 返回默认奖励避免训练中断
                return [0.0] * len(prompts)
        
        return reward_fn_closure

    def train(self):
        """Main training function with comprehensive error handling"""
        try:
            logger.info("🚀 Starting GRPO training process...")

            # 1. Setup model and tokenizer - 🔧 修复调用
            logger.info("📝 Step 1: Setting up model and tokenizer...")
            self._setup_model_and_tokenizer()  # 🔧 这里不再解包，因为已经在方法内部设置了self.model和self.tokenizer

            # 2. Load and preprocess data
            logger.info("📝 Step 2: Loading and preparing dataset...")
            dataset_processed = load_and_prepare_dataset(
                script_cfg=self.script_cfg,
                env_cfg=self.env_cfg,
                tokenizer=self.tokenizer
            )

            if not dataset_processed or len(dataset_processed) == 0:
                raise ValueError("❌ Dataset is empty after processing!")

            # 3. Initialize components
            logger.info("📝 Step 3: Initializing training components...")
            self._initialize_components(dataset_processed)

            # 4. Determine training dataset
            dataset_for_trainer = dataset_processed
            if self.curriculum_manager:
                dataset_for_trainer = self.curriculum_manager.get_current_stage_dataset()
                current_stage_name = self.curriculum_manager.get_current_stage_name()
                logger.info(f"📚 Using curriculum dataset: {len(dataset_for_trainer)} samples from stage '{current_stage_name}'.")
            else:
                logger.info(f"📚 Using full processed dataset: {len(dataset_for_trainer)} samples.")

            if not dataset_for_trainer or len(dataset_for_trainer) == 0:
                raise ValueError("❌ Training dataset is empty!")

            # 5. Create trainer
            logger.info("📝 Step 4: Creating GRPOTrainer...")
            from trl import GRPOTrainer
            
            self.trainer = GRPOTrainer(
                model=self.model,
                args=self.grpo_cfg,
                train_dataset=dataset_for_trainer,
                reward_funcs=[self.get_reward_function()],
                callbacks=self.callbacks
            )

            # Set trainer references for callbacks
            for cb in self.callbacks:
                if hasattr(cb, 'trainer_ref') and cb.trainer_ref is None:
                    cb.trainer_ref = self.trainer
                    logger.debug(f"Set trainer_ref for {type(cb).__name__}")

            # 6. Start training
            logger.info("📝 Step 5: Starting training...")
            logger.info(f"🎯 Training with {len(dataset_for_trainer)} examples.")
            
            train_result = self.trainer.train(resume_from_checkpoint=self.grpo_cfg.resume_from_checkpoint)
            logger.info("✅ Training completed successfully!")

            # 7. Save artifacts
            if self.grpo_cfg.local_rank <= 0:
                self._save_training_artifacts(train_result)

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            raise

    def _save_training_artifacts(self, train_result):
        """Save training artifacts"""
        try:
            logger.info("💾 Saving training artifacts...")
            
            # Save final model
            final_model_dir = os.path.join(self.grpo_cfg.output_dir, "final_model_adapter")
            self.trainer.save_model(final_model_dir)
            logger.info(f"✅ Final model saved to: {final_model_dir}")

            # Save metrics and state
            metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
            
            if hasattr(self.trainer, 'log_metrics'):
                self.trainer.log_metrics("train_summary", metrics)
            
            if hasattr(self.trainer, 'save_metrics'):
                metrics_file = os.path.join(self.grpo_cfg.output_dir, "final_train_metrics.json")
                self.trainer.save_metrics("train_summary", metrics_file)
                logger.info(f"✅ Metrics saved to: {metrics_file}")
            
            if hasattr(self.trainer, 'save_state'):
                self.trainer.save_state()
                logger.info("✅ Trainer state saved.")

            logger.info(f"🎉 All training artifacts saved to: {self.grpo_cfg.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving training artifacts: {e}", exc_info=True)

    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up resources...")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU cache cleared.")
            
            # Force garbage collection
            gc.collect()
            logger.info("✅ Garbage collection completed.")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


def main():
    """Main entry point with comprehensive error handling"""
    pipeline = None
    try:
        logger_temp.info("🚀 Initializing GRPO Training Pipeline...")
        pipeline = GRPOTrainingPipeline()
        
        logger_temp.info("🎯 Starting training...")
        pipeline.train()
        
        logger_temp.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger_temp.warning("⚠️ Training interrupted by user (Ctrl+C)")
        return 1
        
    except Exception as e:
        logger_temp.error(f"💥 Fatal error in training pipeline: {e}", exc_info=True)
        return 1
        
    finally:
        if pipeline:
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                logger_temp.warning(f"⚠️ Error during cleanup: {cleanup_error}")
        
        logger_temp.info("🏁 GRPO Training Pipeline execution finished.")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)