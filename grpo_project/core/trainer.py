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
from datetime import datetime
from grpo_project.data.dataset import load_and_prepare_dataset
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
from trl import GRPOConfig, GRPOTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training

# Project-specific imports
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig, OptimizedTrainingConfig, apply_optimized_config
from grpo_project.utils.logging import setup_global_logging # Updated import
from grpo_project.utils.reporting_utils import PeriodicStatusReporter
# Qwen3CompatibilityFixer will be used by ModelManager internally
# ExperienceBuffer import is fine here if used by other parts of orchestrator, or can be moved if only by callbacks
from grpo_project.utils import ExperienceBuffer

# Import for the main data loading entry point
from grpo_project.data.dataset import load_and_prepare_dataset
# Imports for curriculum manager (still needed by orchestrator directly)
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager

# VerilogDataPreprocessor & validate_dataset_for_curriculum are no longer directly used by trainer.
# Callbacks and Reward Calculator will be imported directly in methods where they are used or further down.

# Temporary logger for setup before full logging is configured
logger_temp = logging.getLogger(__name__ + "_startup")

logger = logging.getLogger(__name__) # Will be configured by setup_global_logging

class TrainingOrchestrator:
    def __init__(self, env_cfg: EnvConfig, script_cfg: ScriptConfig, reward_cfg: EnhancedRewardConfig, grpo_cfg: GRPOConfig):
        self.env_cfg = env_cfg
        self.script_cfg = script_cfg
        self.reward_cfg = reward_cfg
        self.grpo_cfg = grpo_cfg

        self.model = None
        self.tokenizer = None
        self.dataset_raw = None
        self.dataset_processed = None # Full processed dataset
        self.dataset_for_trainer = None # Dataset for current curriculum stage / trainer

        self.trainer = None
        self.curriculum_manager = None
        self.experience_buffer = None # Should be initialized in _setup_curriculum_and_replay

        self.callbacks_list = []
        self.wandb_callback = None # Instance of the W&B callback, if used
        self.status_reporter = None
        self.is_resuming = False
        self.sanitized_run_name = ""
        self.actual_output_dir = ""

        # Instantiate ModelManager here
        # It requires script_cfg, grpo_cfg, model_name_or_path, cache_dir
        from .models import ModelManager # Import ModelManager
        self.model_manager = ModelManager(
            script_cfg=self.script_cfg,
            grpo_cfg=self.grpo_cfg,
            model_name_or_path=self.script_cfg.model_name_or_path,
            cache_dir=self.env_cfg.cache_dir # Use env_cfg for cache_dir
        )

        self._setup_paths_and_logging() # Sets up paths and logging first


    def _setup_paths_and_logging(self):
        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
        if not run_specific_name_from_env:
            logger_temp.warning("WANDB_RUN_NAME environment variable not set. Generating run name from timestamp.")
            run_specific_name_from_env = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)

        # output_dir_base is now in EnvConfig
        if not self.env_cfg.output_dir_base or not isinstance(self.env_cfg.output_dir_base, str):
            logger_temp.error(f"EnvConfig.output_dir_base ('{self.env_cfg.output_dir_base}') is invalid. Using './grpo_runs'.")
            self.env_cfg.output_dir_base = "./grpo_runs"

        self.is_resuming = (
            self.grpo_cfg.resume_from_checkpoint and
            isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
            os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
        )

        if self.is_resuming:
            self.actual_output_dir = os.path.dirname(self.grpo_cfg.resume_from_checkpoint)
            self.sanitized_run_name = os.path.basename(self.actual_output_dir) # Update run name from checkpoint dir
            logger_temp.info(f"Resuming training mode. Using original output directory: {self.actual_output_dir}")
        else:
            # Use env_cfg for output_dir_base
            self.actual_output_dir = os.path.join(self.env_cfg.output_dir_base, self.sanitized_run_name)

        if self.grpo_cfg.local_rank <= 0: # main process
            os.makedirs(self.actual_output_dir, exist_ok=True)

        # Critical: Update config objects with the actual output directory
        self.grpo_cfg.output_dir = self.actual_output_dir
        self.script_cfg.output_dir = self.actual_output_dir

        log_file_path = os.path.join(self.actual_output_dir, "training_orchestrator_log.txt")
        setup_global_logging(
            log_level=self.grpo_cfg.get_process_log_level(),
            log_file_path=log_file_path,
            local_rank=self.grpo_cfg.local_rank
        )

        logger.info(f"=== GRPO TRAINING ORCHESTRATOR INITIALIZED (PID: {os.getpid()}) ===")
        logger.info(f"Run Name: {self.sanitized_run_name}")
        logger.info(f"Output Directory: {self.actual_output_dir}")
        logger.info(f"Resuming: {self.is_resuming} from: {self.grpo_cfg.resume_from_checkpoint}")
        # Initial log of configs (can be extensive, use DEBUG level)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"EnvConfig: \n{json.dumps(asdict(self.env_cfg), indent=2)}")
            logger.debug(f"ScriptConfig: \n{json.dumps(asdict(self.script_cfg), indent=2)}")
            logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(self.reward_cfg), indent=2)}")
            logger.debug(f"GRPOConfig (TrainingArguments): \n{self.grpo_cfg.to_json_string()}")

        if self.grpo_cfg.world_size > 1:
            logger.info("Waiting for all processes at barrier before distributed setup...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                logger.info("All processes passed barrier.")
            else:
                logger.warning("Distributed training indicated by world_size > 1, but torch.distributed not initialized before barrier call.")

        # Initialize PeriodicStatusReporter
        self.status_reporter = PeriodicStatusReporter(self.script_cfg.output_dir, report_interval=50)

    def _setup_model_and_tokenizer_with_manager(self):
        """
        Uses ModelManager to load model, tokenizer and apply PEFT.
        """
        logger.info("Setting up model and tokenizer using ModelManager...")
        try:
            # setup_model_and_tokenizer returns model, tokenizer
            self.model, self.tokenizer = self.model_manager.setup_model_and_tokenizer()

            # apply_peft_adapter takes the base model and returns PEFT model
            self.model = self.model_manager.apply_peft_adapter(
                model=self.model,
                is_resuming=self.is_resuming,
                resume_checkpoint_path=str(self.grpo_cfg.resume_from_checkpoint) if self.is_resuming else None
            )
            logger.info("Model and tokenizer setup complete via ModelManager.")
        except Exception as e:
            logger.error(f"Error during model and tokenizer setup via ModelManager: {e}", exc_info=True)
            raise

    def _setup_dataset_and_dependencies(self):
        """
        Loads and prepares the dataset using the dedicated data module.
        Then sets up curriculum learning and experience replay if enabled.
        """
        logger.info("Setting up dataset and related dependencies...")
        try:
            # Use the new centralized function for loading and preparation, pass env_cfg
            self.dataset_processed = load_and_prepare_dataset(
                script_cfg=self.script_cfg,
                env_cfg=self.env_cfg, # Pass env_cfg
                tokenizer=self.tokenizer
            )
            if not self.dataset_processed or len(self.dataset_processed) == 0:
                raise ValueError("load_and_prepare_dataset returned an empty or None dataset.")
            logger.info(f"Dataset successfully loaded and prepared. Total samples: {len(self.dataset_processed)}")

        except Exception as e:
            logger.error(f"Fatal error during dataset setup: {e}", exc_info=True)
            raise # Re-raise critical error to stop execution

        # Setup curriculum and experience replay, which depend on the processed dataset
        if self.script_cfg.enable_experience_replay:
            try:
                # ExperienceBuffer already imported at the top
                self.experience_buffer = ExperienceBuffer(max_size=self.script_cfg.experience_buffer_size)
                logger.info(f"Experience buffer initialized (size: {self.script_cfg.experience_buffer_size}).")
                if self.is_resuming: # self.is_resuming is boolean
                    buffer_state_path = os.path.join(str(self.grpo_cfg.resume_from_checkpoint), "enhanced_experience_buffer_state.json")
                    if os.path.exists(buffer_state_path):
                        try:
                            logger.info(f"Attempting to load experience buffer state from: {buffer_state_path}")
                            with open(buffer_state_path, "r", encoding="utf-8") as f: state_data = json.load(f)
                            self.experience_buffer.load_buffer_state(state_data)
                        except Exception as e: logger.error(f"Failed to load experience buffer state: {e}")
                    else: logger.warning(f"Experience buffer state file not found at {buffer_state_path}.")
            except NameError: # If ExperienceBuffer failed to import
                logger.error("ExperienceBuffer not available. Experience replay disabled.")
                self.experience_buffer = None
            except Exception as e: # Other errors during ExperienceBuffer initialization
                logger.error(f"Error initializing ExperienceBuffer: {e}", exc_info=True)
                self.experience_buffer = None

        if self.script_cfg.enable_curriculum:
            try:
                # setup_fixed_curriculum_manager already imported at the top
                self.curriculum_manager = setup_fixed_curriculum_manager(self.script_cfg, self.dataset_processed)
                if self.curriculum_manager:
                    logger.info(f"Curriculum learning enabled using {type(self.curriculum_manager).__name__}.")
                    if self.is_resuming: # self.is_resuming is boolean
                        curriculum_state_path = os.path.join(str(self.grpo_cfg.resume_from_checkpoint), "enhanced_curriculum_state.json")
                        if os.path.exists(curriculum_state_path):
                            try:
                                logger.info(f"Loading curriculum state from: {curriculum_state_path}")
                                with open(curriculum_state_path, "r", encoding="utf-8") as f: state_data = json.load(f)
                                self.curriculum_manager.load_curriculum_state(state_data)
                            except Exception as e: logger.error(f"Failed to load curriculum state: {e}")
                        else: logger.warning(f"Curriculum state file not found at {curriculum_state_path}.")
                else:
                    logger.warning("Curriculum manager setup returned None. Curriculum learning disabled.")
            except NameError: # If setup_fixed_curriculum_manager failed to import
                logger.error("setup_fixed_curriculum_manager not available. Curriculum learning disabled.")
                self.curriculum_manager = None
            except Exception as e:
                logger.error(f"Error setting up curriculum manager: {e}", exc_info=True)
                self.curriculum_manager = None

        if self.curriculum_manager:
            self.dataset_for_trainer = self.curriculum_manager.get_current_stage_dataset()
            logger.info(f"Using curriculum dataset for trainer: {len(self.dataset_for_trainer)} samples from stage '{self.curriculum_manager.get_current_stage_name()}'.")
        else:
            self.dataset_for_trainer = self.dataset_processed
            logger.info("Using full processed dataset for trainer.")

        if not self.dataset_for_trainer or len(self.dataset_for_trainer) == 0:
            raise ValueError("Dataset for trainer is empty. Cannot proceed.")


    def _setup_callbacks(self):
        # Import necessary callbacks
        from grpo_project.callbacks.monitoring import StepLoggingCallback, DetailedRewardCallback, RewardStabilityMonitor
        from grpo_project.callbacks.persistence import CustomStatePersistenceCallback
        from grpo_project.callbacks.inference import DetailedInferenceCallback
        from grpo_project.callbacks.wandb import DetailedWandbCallback as TrainDetailedWandbCallback # Alias
        from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback

        self.callbacks_list.append(StepLoggingCallback())
        self.callbacks_list.append(DetailedRewardCallback(self.script_cfg.output_dir))
        self.callbacks_list.append(RewardStabilityMonitor(self.script_cfg.output_dir))

        if self.curriculum_manager:
            self.callbacks_list.append(CurriculumProgressCallback(self.curriculum_manager, None, self.script_cfg.output_dir)) # trainer_ref set later
            self.callbacks_list.append(EnhancedCurriculumDebugCallback(self.curriculum_manager, None, self.script_cfg.output_dir)) # trainer_ref set later

        self.callbacks_list.append(CustomStatePersistenceCallback(self.curriculum_manager, self.experience_buffer, self.script_cfg))

        # DetailedInferenceCallback
        sample_dataset_for_inf_cb = self.dataset_processed.select(range(min(len(self.dataset_processed), self.script_cfg.callback_num_samples * 5))) \
            if self.dataset_processed and len(self.dataset_processed) > 0 else None
        if sample_dataset_for_inf_cb and len(sample_dataset_for_inf_cb) > 0:
            self.callbacks_list.append(DetailedInferenceCallback(
                tokenizer=self.tokenizer, eval_dataset=sample_dataset_for_inf_cb,
                num_samples=self.script_cfg.callback_num_samples, eval_every_n_steps=self.script_cfg.callback_eval_every_n_steps,
                max_new_tokens=self.grpo_cfg.max_completion_length, max_seq_length=self.script_cfg.max_seq_length,
                experience_buffer=self.experience_buffer, output_dir=self.script_cfg.output_dir
            ))
            logger.info(f"DetailedInferenceCallback initialized with {len(sample_dataset_for_inf_cb)} samples.")
        else:
            logger.warning("DetailedInferenceCallback will not run due to insufficient sample data.")

        # WandB Callback
        if self.grpo_cfg.local_rank <= 0 and "wandb" in self.grpo_cfg.report_to:
            # Wandb setup (resume logic)
            wandb_resume_mode = "allow"
            wandb_run_id = None
            if self.is_resuming:
                wandb_run_id = self.sanitized_run_name # From output dir name
                logger.info(f"Attempting to resume wandb run with ID: {wandb_run_id}")

            if wandb_run_id:
                os.environ["WANDB_RUN_ID"] = wandb_run_id
                os.environ["WANDB_RESUME"] = wandb_resume_mode

            # try:
            #     import wandb # Ensure wandb is imported
            # except ImportError:
            #     logger.error("wandb not installed, but specified in report_to. Skipping wandb setup.")
            #     self.wandb_callback = None
            # else:
            self.wandb_callback = TrainDetailedWandbCallback(self.env_cfg, self.script_cfg, self.reward_cfg, self.experience_buffer)
            self.callbacks_list.append(self.wandb_callback)
            logger.info(f"TrainDetailedWandbCallback initialized. Resume mode: {wandb_resume_mode}, Run ID: {wandb_run_id or 'new'}")

        logger.info(f"Callbacks setup complete. Total callbacks: {len(self.callbacks_list)}")


    def _get_reward_function_with_context(self):
        # This method encapsulates the reward function and its context.
        # It needs access to self.reward_cfg, self.script_cfg, self.experience_buffer,
        # self.wandb_callback, and potentially self.trainer (for global_step).

        # Assuming RewardCalculator is in grpo_project.rewards, import it here or at top
        try:
            from grpo_project.rewards import RewardCalculator
        except ImportError:
            logger.error("RewardCalculator not found in grpo_project.rewards. Reward calculation will fail.")
            # Define a dummy reward_calculator_instance or raise error to prevent continuation
            # For now, this will cause NameError later if not handled.
            pass # Let it fail later to make missing dependency clear

        reward_calculator_instance = RewardCalculator(
            reward_config=self.reward_cfg,
            simulator=None
        )

        def reward_fn_closure(prompts: List[str], completions: List[str], **kwargs_from_trainer_dataset) -> List[float]:
            # Ensure reward_calculator_instance is available
            if 'reward_calculator_instance' not in locals() and not hasattr(self, 'reward_calculator_instance'):
                 logger.error("Reward calculator instance not found in reward_fn_closure context!")
                 return [-100.0] * len(prompts) # Return a strong negative reward
            current_training_step = 0
            if self.trainer and hasattr(self.trainer, 'state') and self.trainer.state:
                current_training_step = self.trainer.state.global_step

            # Combine all necessary data for the reward calculator's batch method
            # kwargs_from_trainer_dataset will contain columns from the dataset
            # Ensure keys match what calculate_batch_rewards expects
            batch_rewards_args = {
                "prompts": prompts, # Actual prompts fed to the model by GRPOTrainer
                "completions": completions, # Completions from the model
                "testbench_paths": kwargs_from_trainer_dataset.get('testbench_path', []),
                "expected_total_tests_list": kwargs_from_trainer_dataset.get('expected_total_tests', []),
                "reference_verilog_paths": kwargs_from_trainer_dataset.get('reference_verilog_path', []),
                "original_enhanced_prompts": kwargs_from_trainer_dataset.get('original_enhanced_prompt'), # Optional
                # Contextual arguments for reward calculation and logging
                "training_step": current_training_step,
                "output_dir_for_debug": self.script_cfg.output_dir, # For saving debug files
                "experience_buffer_obj": self.experience_buffer, # Pass the buffer object
                "wandb_callback_obj": self.wandb_callback, # Pass the wandb callback for detailed logging
                "script_config_obj": self.script_cfg # Pass script_cfg if needed by reward calc
            }

            # Call the batch reward calculation
            # calculate_batch_rewards should return Tuple[List[float], Dict[str, Any]]
            rewards_list, aggregated_metrics = reward_calculator_instance.calculate_batch_rewards(**batch_rewards_args)

            # Log aggregated metrics using wandb_callback if available and metrics exist
            if self.wandb_callback and aggregated_metrics and hasattr(self.wandb_callback, 'log_batch_aggregated_metrics'):
                try:
                    self.wandb_callback.log_batch_aggregated_metrics(aggregated_metrics, step=current_training_step)
                except Exception as e_wb_log:
                    logger.error(f"Error logging batch aggregated metrics to wandb: {e_wb_log}", exc_info=True)

            return rewards_list # GRPOTrainer expects only the list of rewards

        return reward_fn_closure


    def _initialize_trainer(self):
        logger.info("Initializing GRPOTrainer...")

        reward_function = self._get_reward_function_with_context()

        self.trainer = GRPOTrainer(
            model=self.model,
            args=self.grpo_cfg,
            train_dataset=self.dataset_for_trainer,
            reward_funcs=[reward_function], # GRPOTrainer expects a list of functions
            tokenizer=self.tokenizer, # Pass tokenizer to GRPOTrainer
            callbacks=self.callbacks_list,
        )

        # Set trainer_ref for callbacks that need it
        for cb in self.callbacks_list:
            if hasattr(cb, 'trainer_ref') and cb.trainer_ref is None:
                cb.trainer_ref = self.trainer
                logger.info(f"Set trainer_ref for {type(cb).__name__}")

        logger.info("GRPOTrainer initialized successfully.")


    def setup_all_components(self):
        """ Sequentially sets up all necessary components for training. """
        self._setup_model_and_tokenizer_with_manager()
        # Tokenizer is needed by _setup_dataset_and_dependencies, ensure it's ready.
        if not self.tokenizer: # Should be set by _setup_model_and_tokenizer_with_manager
            logger.error("Tokenizer not available before dataset setup. Critical error.")
            raise RuntimeError("Tokenizer must be initialized before setting up the dataset.")
        self._setup_dataset_and_dependencies() # Handles dataset loading, processing, validation, curriculum, replay

        self._setup_callbacks() # Callbacks depend on components like curriculum_manager, experience_buffer
        self._initialize_trainer() # Trainer needs model, dataset_for_trainer, callbacks, reward_func
        logger.info("All components set up successfully.")


    def run_training(self):
        if not self.trainer:
            logger.error("Trainer not initialized. Call setup_all_components() first.")
            sys.exit(1)

        logger.info(f"=== STARTING GRPO TRAINING (Output: {self.grpo_cfg.output_dir}) ===")
        logger.info(f"Training with {len(self.dataset_for_trainer) if self.dataset_for_trainer else 'N/A'} examples.")

        # Determine actual_resume_path for trainer.train()
        # self.is_resuming (bool) is set in _setup_paths_and_logging
        # self.grpo_cfg.resume_from_checkpoint (str/None) is the initial arg

        resume_arg_for_trainer = self.grpo_cfg.resume_from_checkpoint # This could be a path or None
        if not resume_arg_for_trainer and self.is_resuming:
            # This implies that is_resuming was True because output_dir suggested it,
            # but no explicit checkpoint path was given. Trainer's default for True is to check output_dir.
            resume_arg_for_trainer = True

        logger.info(f"Trainer.train called with resume_from_checkpoint='{resume_arg_for_trainer}'")

        try:
            train_res = self.trainer.train(resume_from_checkpoint=resume_arg_for_trainer)

            if self.grpo_cfg.local_rank <= 0: # Main process
                self._save_training_artifacts(train_res)

        except Exception as e_train:
            logger.error(f"Training loop failed: {e_train}", exc_info=True)
            if self.grpo_cfg.local_rank <= 0 and self.wandb_callback : #and wandb.run:
                # try:
                #    import wandb
                #    if wandb.run:
                #        wandb.log({"training/final_status": "failed", "training/error_message": str(e_train)})
                # except ImportError: pass
                pass # Placeholder for wandb error logging
            raise # Re-throw after logging

    def _save_training_artifacts(self, train_res):
        # This method is called only on local_rank <= 0
        logger.info("Saving training artifacts...")
        final_model_dir = os.path.join(self.script_cfg.output_dir, "final_model_adapter")
        if hasattr(self.trainer, "save_model"):
            self.trainer.save_model(final_model_dir)

        enhanced_artifacts_dir = os.path.join(self.script_cfg.output_dir, "training_artifacts")
        os.makedirs(enhanced_artifacts_dir, exist_ok=True)

        if self.experience_buffer and self.script_cfg.enable_experience_replay and hasattr(self.experience_buffer, "get_stats"):
            buffer_stats = self.experience_buffer.get_stats()
            with open(os.path.join(enhanced_artifacts_dir, "experience_buffer_stats.json"), "w", encoding="utf-8") as f:
                json.dump(buffer_stats, f, indent=2)
            logger.info(f"Experience buffer stats saved.")

        if self.curriculum_manager: # Assuming methods and attributes exist
            curriculum_stats = {
                "final_stage_idx": self.curriculum_manager.current_stage,
                "final_stage_name": self.curriculum_manager.get_current_stage_name(),
                "stages_completed": self.curriculum_manager.current_stage + 1, # If 0-indexed
                "total_stages": len(self.curriculum_manager.curriculum_stages),
                "stage_performance_history": getattr(self.curriculum_manager, 'stage_performance_history', [])
            }
            with open(os.path.join(enhanced_artifacts_dir, "curriculum_progress.json"), "w", encoding="utf-8") as f:
                json.dump(curriculum_stats, f, indent=2)
            logger.info(f"Curriculum progress saved.")

        metrics = train_res.metrics if hasattr(train_res, 'metrics') else {}
        if hasattr(self.trainer, 'log_metrics'): self.trainer.log_metrics("train_summary", metrics)
        if hasattr(self.trainer, 'save_metrics'): self.trainer.save_metrics("train_summary", os.path.join(enhanced_artifacts_dir, "final_train_metrics.json"))
        if hasattr(self.trainer, 'save_state'): self.trainer.save_state()
        logger.info(f"Training artifacts saved to {enhanced_artifacts_dir}")

        # WandB logging for final status (main process only)
        # if self.wandb_callback: # and wandb.run:
        #    try:
        #        import wandb
        #        if wandb.run:
        #            wandb.log({
        #                "training/final_status": "completed",
        #                "training/total_steps_final": metrics.get("train_steps", self.trainer.state.global_step if self.trainer and self.trainer.state else 0),
        #                "training/final_loss_train": metrics.get("train_loss", 0),
        #            }, step = self.trainer.state.global_step if self.trainer and self.trainer.state else None)
        #            if self.experience_buffer and self.script_cfg.enable_experience_replay and hasattr(self.experience_buffer, "get_stats"):
        #                final_buffer_stats_wandb = self.experience_buffer.get_stats()
        #                wandb.log({f"final_experience_buffer/{k}": v for k,v in final_buffer_stats_wandb.items()},
        #                             step=self.trainer.state.global_step if self.trainer and self.trainer.state else None)
        #    except ImportError: pass
        #    except Exception as e_wandb_final: logger.error(f"Wandb final logging error: {e_wandb_final}")
        pass


    def cleanup(self):
        logger.info("Cleaning up resources...")
        try:
            # Explicitly delete large objects
            if hasattr(self, 'trainer') and self.trainer: del self.trainer; self.trainer = None
            if hasattr(self, 'model') and self.model: del self.model; self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer: del self.tokenizer; self.tokenizer = None
            if hasattr(self, 'dataset_processed') and self.dataset_processed: del self.dataset_processed; self.dataset_processed = None
            if hasattr(self, 'dataset_for_trainer') and self.dataset_for_trainer: del self.dataset_for_trainer; self.dataset_for_trainer = None
            if hasattr(self, 'experience_buffer') and self.experience_buffer: del self.experience_buffer; self.experience_buffer = None
            if hasattr(self, 'curriculum_manager') and self.curriculum_manager: del self.curriculum_manager; self.curriculum_manager = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Resources cleaned up.")

            # Finish wandb run on main process
            # if self.grpo_cfg.local_rank <= 0 and self.wandb_callback: # and wandb.run:
            #    try:
            #        import wandb
            #        if wandb.run: wandb.finish()
            #    except ImportError: pass
            #    except Exception as e_wandb_finish: logger.error(f"Wandb finish error: {e_wandb_finish}")
            pass

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}", exc_info=True)

def main_orchestrator_cli():
    # This function will parse args and call the Orchestrator
    # Ensure HfArgumentParser and config classes are available
    # from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig (already imported if at top)
    # from trl import GRPOConfig (already imported)

    parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
    env_cfg, script_cfg, reward_cfg, grpo_cfg = parser.parse_args_into_dataclasses()

    orchestrator = None
    try:
        orchestrator = TrainingOrchestrator(env_cfg, script_cfg, reward_cfg, grpo_cfg)
        orchestrator.setup_all_components()
        orchestrator.run_training()
    except Exception as e:
        # Use global logger if orchestrator or its logger isn't initialized
        current_logger = logger
        if orchestrator and hasattr(orchestrator, 'logger') and orchestrator.logger:
            current_logger = orchestrator.logger # Should be the same instance if setup_global_logging worked

        current_logger.error(f"Unhandled exception in TrainingOrchestrator CLI: {e}", exc_info=True)
        # Potentially log to wandb here if it's initialized and failed early
        sys.exit(1) # Exit with error code
    finally:
        if orchestrator:
            orchestrator.cleanup()
        else: # Basic cleanup if orchestrator instantiation failed
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
        logger.info("Training script finished.")


if __name__ == "__main__":
    main_orchestrator_cli()
