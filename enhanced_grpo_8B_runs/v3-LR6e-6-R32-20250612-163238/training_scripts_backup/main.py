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
import re

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
from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback, OptimizedCurriculumCallback
from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager, get_wandb_sync_manager

logger = logging.getLogger(__name__)

class GRPOTrainingPipeline:
    def __init__(self):
        # 1. Load Configurations
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        self.env_cfg, self.script_cfg, self.reward_cfg, self.grpo_cfg = parser.parse_args_into_dataclasses()
        
        # 🔧 同步长度配置从ScriptConfig到GRPOConfig
        self._sync_length_configs()

        # 🔧 新增：自动配置WandB恢复
        self._configure_wandb_resume()

        # Setup logging first
        self._setup_logging()
        logger.info("GRPOTrainingPipeline initialized.")
        self._log_configs()

        # 🔧 新增：初始化WandB同步管理器
        self._setup_wandb_sync_manager()

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

    def _sync_length_configs(self):
        """🔧 同步长度配置从ScriptConfig到GRPOConfig，避免参数冲突"""
        logger.info("🔧 同步长度配置到GRPOConfig...")
        
        # 将我们的脚本配置同步到GRPO配置
        self.grpo_cfg.max_prompt_length = self.script_cfg.script_max_prompt_length
        self.grpo_cfg.max_completion_length = self.script_cfg.script_max_completion_length
        
        logger.info(f"  ✅ GRPO max_prompt_length: {self.grpo_cfg.max_prompt_length}")
        logger.info(f"  ✅ GRPO max_completion_length: {self.grpo_cfg.max_completion_length}")

    def _setup_stable_environment(self):
        """🔧 设置稳定的训练环境，避免段错误"""
        try:
            logger.info("🔧 设置稳定训练环境...")
            
            # 设置CUDA环境变量以提高稳定性
            stable_envs = {
                "CUDA_LAUNCH_BLOCKING": "1",  # 同步CUDA操作，便于调试
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",  # 限制内存分割
                "NCCL_BLOCKING_WAIT": "1",  # NCCL阻塞等待
            }
            
            for key, value in stable_envs.items():
                if key not in os.environ:
                    os.environ[key] = value
                    logger.info(f"  设置环境变量: {key}={value}")
            
            # 如果启用了梯度检查点，强制禁用以避免段错误
            if hasattr(self.grpo_cfg, 'gradient_checkpointing') and self.grpo_cfg.gradient_checkpointing:
                logger.warning("⚠️ 检测到梯度检查点已启用，为避免段错误自动禁用")
                self.grpo_cfg.gradient_checkpointing = False
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU内存已清理")
            
            logger.info("✅ 稳定训练环境设置完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 稳定环境设置失败: {e}")

    def _configure_wandb_resume(self):
        """🔧 自动配置WandB恢复，无需外部脚本"""
        try:
            # 检查是否从checkpoint恢复
            is_resuming = (
                self.grpo_cfg.resume_from_checkpoint and
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
            )
            
            if not is_resuming:
                logger.info("🚀 开始新的训练，无需WandB恢复配置")
                # 确保清除可能存在的恢复相关环境变量
                for env_var in ["WANDB_RUN_ID", "WANDB_RESUME"]:
                    if env_var in os.environ:
                        del os.environ[env_var]
                        logger.info(f"🧹 清除环境变量: {env_var}")
                return
            
            checkpoint_path = Path(self.grpo_cfg.resume_from_checkpoint)
            logger.info(f"🔄 检测到checkpoint恢复: {checkpoint_path}")
            
            # 尝试从checkpoint目录提取WandB run ID
            run_id, run_url = self._extract_wandb_run_id(checkpoint_path)
            
            if run_id:
                # 设置精确恢复
                os.environ["WANDB_RUN_ID"] = run_id
                os.environ["WANDB_RESUME"] = "must"
                logger.info(f"✅ WandB精确恢复配置:")
                logger.info(f"  - Run ID: {run_id}")
                logger.info(f"  - Resume Mode: must")
                if run_url:
                    logger.info(f"  - Run URL: {run_url}")
            else:
                # 使用自动恢复模式
                os.environ["WANDB_RESUME"] = "allow"
                logger.info("⚠️ 未找到具体的Run ID，使用自动恢复模式")
                logger.info("  - Resume Mode: allow")
            
            logger.info("✅ WandB恢复配置完成")
            
        except Exception as e:
            logger.warning(f"⚠️ WandB恢复配置失败: {e}")
            logger.info("🔄 将使用默认的WandB配置")

    def _setup_wandb_sync_manager(self):
        """🔧 设置WandB同步管理器"""
        try:
            # 从配置中获取项目信息
            project_name = getattr(self.env_cfg, 'wandb_project', 'VerilogGRPO_Enhanced_v3')
            run_name = f"grpo_run_{os.path.basename(self.grpo_cfg.output_dir)}"
            
            # 初始化同步管理器
            sync_manager = initialize_wandb_sync_manager(
                output_dir=self.grpo_cfg.output_dir,
                project_name=project_name,
                run_name=run_name
            )
            
            logger.info("✅ WandB同步管理器初始化成功")
            logger.info(f"  - 项目: {project_name}")
            logger.info(f"  - 运行名称: {run_name}")
            logger.info(f"  - 输出目录: {self.grpo_cfg.output_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ WandB同步管理器初始化失败: {e}")
            logger.info("🔄 将使用原生WandB功能")

    def _setup_wandb_run(self):
        """🔧 设置WandB运行，处理断续训练"""
        try:
            sync_manager = get_wandb_sync_manager()
            if not sync_manager:
                logger.warning("⚠️ WandB同步管理器未找到，跳过WandB运行设置")
                return
                
            # 检查是否从checkpoint恢复
            resume_from_checkpoint = None
            if (self.grpo_cfg.resume_from_checkpoint and 
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)):
                resume_from_checkpoint = self.grpo_cfg.resume_from_checkpoint
                
            # 准备配置
            config = {
                "model_name_or_path": self.script_cfg.model_name_or_path,
                "learning_rate": self.grpo_cfg.learning_rate,
                "per_device_train_batch_size": self.grpo_cfg.per_device_train_batch_size,
                "max_seq_length": self.script_cfg.max_seq_length,
                "callback_eval_every_n_steps": self.script_cfg.callback_eval_every_n_steps,
                "lora_rank": getattr(self.script_cfg, 'lora_rank', None),
                "curriculum_enabled": self.curriculum_manager is not None,
                "resume_from_checkpoint": resume_from_checkpoint,
            }
            
            # 设置WandB运行
            success = sync_manager.setup_wandb_run(
                resume_from_checkpoint=resume_from_checkpoint,
                config=config
            )
            
            if success:
                logger.info("✅ WandB运行设置成功")
            else:
                logger.warning("⚠️ WandB运行设置失败，将使用本地日志")
                
        except Exception as e:
            logger.warning(f"⚠️ WandB运行设置异常: {e}")

    def _extract_wandb_run_id(self, checkpoint_path: Path) -> tuple[Optional[str], Optional[str]]:
        """从checkpoint目录中提取WandB run ID"""
        try:
            # 方法1: 从wandb目录中查找
            parent_dir = checkpoint_path.parent
            wandb_dir = parent_dir / "wandb"
            
            if wandb_dir.exists():
                logger.info(f"🔍 在 {wandb_dir} 中查找WandB run信息...")
                
                # 查找run目录
                run_dirs = list(wandb_dir.glob("run-*"))
                if run_dirs:
                    latest_run_dir = sorted(run_dirs)[-1]
                    logger.info(f"📁 找到run目录: {latest_run_dir.name}")
                    
                    # 提取run ID (格式: run-20231201_123456-abcd1234)
                    run_name = latest_run_dir.name
                    if "-" in run_name:
                        parts = run_name.split("-")
                        if len(parts) >= 3:
                            run_id = parts[-1]  # 最后一部分是run ID
                            logger.info(f"✅ 提取到run ID: {run_id}")
                            
                            # 尝试读取run信息
                            run_info_file = latest_run_dir / "files" / "wandb-metadata.json"
                            run_url = None
                            if run_info_file.exists():
                                try:
                                    with open(run_info_file, 'r') as f:
                                        metadata = json.load(f)
                                        run_url = metadata.get('url', '')
                                        if run_url:
                                            logger.info(f"🔗 找到Run URL: {run_url}")
                                except Exception as e:
                                    logger.warning(f"⚠️ 读取metadata失败: {e}")
                            
                            return run_id, run_url
            
            # 方法2: 从trainer_state.json中查找
            trainer_state_file = checkpoint_path / "trainer_state.json"
            if trainer_state_file.exists():
                logger.info(f"🔍 在 {trainer_state_file} 中查找训练状态...")
                try:
                    with open(trainer_state_file, 'r') as f:
                        state_data = json.load(f)
                        
                    # 查找log_history中的wandb相关信息
                    log_history = state_data.get('log_history', [])
                    for entry in log_history:
                        if isinstance(entry, dict):
                            for key, value in entry.items():
                                if 'wandb' in key.lower() or '_wandb' in str(value):
                                    logger.info(f"📊 找到WandB相关日志: {key}")
                                    break
                                    
                except Exception as e:
                    logger.warning(f"⚠️ 读取trainer_state.json失败: {e}")
            
            # 方法3: 检查环境变量中是否已有run ID
            env_run_id = os.getenv("WANDB_RUN_ID")
            if env_run_id:
                logger.info(f"🔄 使用环境变量中的WandB run ID: {env_run_id}")
                return env_run_id, None
            
            logger.info("❌ 未能找到WandB run ID")
            return None, None
            
        except Exception as e:
            logger.warning(f"⚠️ 提取WandB run ID时出错: {e}")
            return None, None

    def _setup_logging(self):
        """Setup logging with improved error handling"""
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
        # 🔧 新增：显示长度配置信息
        logger.info("📏 长度配置:")
        logger.info(f"  - 总序列长度: {self.script_cfg.max_seq_length}")
        logger.info(f"  - 最大提示长度: {self.script_cfg.script_max_prompt_length} ({self.script_cfg.script_max_prompt_length/self.script_cfg.max_seq_length*100:.1f}%)")
        logger.info(f"  - 最大输出长度: {self.script_cfg.script_max_completion_length} ({self.script_cfg.script_max_completion_length/self.script_cfg.max_seq_length*100:.1f}%)")
        logger.info(f"  - 分配策略: {self.script_cfg.length_allocation_strategy}")
        logger.info(f"  - GRPO max_prompt_length: {self.grpo_cfg.max_prompt_length}")
        logger.info(f"  - GRPO max_completion_length: {self.grpo_cfg.max_completion_length}")
        
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
    def _initialize_trainer(self):
        logger.info("Initializing GRPOTrainer...")

        reward_function = self._get_reward_function_with_context()

        self.trainer = GRPOTrainer(
            model=self.model,
            args=self.grpo_cfg,
            train_dataset=self.dataset_for_trainer,
            reward_funcs=[reward_function],
            tokenizer=self.tokenizer,
            callbacks=self.callbacks,
        )

        # 🔧 重要：设置 trainer_ref 为所有需要的回调
        for cb in self.callbacks:
            if hasattr(cb, 'trainer_ref') and cb.trainer_ref is None:
                cb.trainer_ref = self.trainer
                logger.info(f"✅ Set trainer_ref for {type(cb).__name__}")

        logger.info("GRPOTrainer initialized successfully.")

        # 🔧 额外的课程学习状态验证
        if self.curriculum_manager:
            logger.info("📚 最终课程学习状态验证:")
            logger.info(f"  - 课程管理器类型: {type(self.curriculum_manager).__name__}")
            
            # 验证课程管理器是否有调试日志
            if hasattr(self.curriculum_manager, 'debug_log'):
                logger.info(f"  - 调试日志条目: {len(self.curriculum_manager.debug_log)}")
                if self.curriculum_manager.debug_log:
                    logger.info(f"  - 最新日志: {self.curriculum_manager.debug_log[-1]}")
            
            # 验证当前阶段数据集
            try:
                current_dataset = self.curriculum_manager.get_current_stage_dataset()
                logger.info(f"  - 当前数据集验证: {len(current_dataset)} samples")
            except Exception as e:
                logger.error(f"  - 数据集验证失败: {e}")

            # 验证课程阶段配置
            for i, stage in enumerate(self.curriculum_manager.curriculum_stages):
                status = "🔄 当前" if i == self.curriculum_manager.current_stage else "⏳ 待进入"
                logger.info(f"  - 阶段{i}: {stage.name} ({status})")

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
        """Setup all callbacks with enhanced curriculum debugging"""
        try:
            self.callbacks = []
            
            # Basic monitoring callbacks
            self.callbacks.append(StepLoggingCallback())
            self.callbacks.append(DetailedRewardCallback(self.script_cfg.output_dir))
            self.callbacks.append(RewardStabilityMonitor(self.script_cfg.output_dir))

            # 🔧 修复：增强的课程学习回调设置
            if self.curriculum_manager:
                # 1. 添加基础的课程进度回调
                curriculum_progress_cb = CurriculumProgressCallback(
                    curriculum_manager=self.curriculum_manager, 
                    trainer_ref=None,  # 稍后设置
                    output_dir=self.script_cfg.output_dir,
                    performance_check_interval=self.script_cfg.curriculum_performance_check_interval
                )
                self.callbacks.append(curriculum_progress_cb)
                logger.info("✅ 添加 CurriculumProgressCallback")

                # 2. 添加增强的课程调试回调
                enhanced_curriculum_cb = EnhancedCurriculumDebugCallback(
                    curriculum_manager=self.curriculum_manager, 
                    trainer_ref=None,  # 稍后设置
                    output_dir=self.script_cfg.output_dir
                )
                self.callbacks.append(enhanced_curriculum_cb)
                logger.info("✅ 添加 EnhancedCurriculumDebugCallback")

                # 3. 添加优化的课程回调（如果需要）
                optimized_curriculum_cb = OptimizedCurriculumCallback(
                    curriculum_manager=self.curriculum_manager,
                    trainer_ref=None,  # 稍后设置
                    output_dir=self.script_cfg.output_dir
                )
                self.callbacks.append(optimized_curriculum_cb)
                logger.info("✅ 添加 OptimizedCurriculumCallback")

            # State persistence callback
            self.callbacks.append(CustomStatePersistenceCallback(
                curriculum_manager=self.curriculum_manager, 
                experience_buffer=self.experience_buffer, 
                script_cfg=self.script_cfg
            ))

            # Inference callback (需要 tokenizer)
            # 🚀 使用流式引导推理回调替代原有的DetailedInferenceCallback
            if self.tokenizer and dataset_processed and len(dataset_processed) > 0:
                sample_dataset_for_inf_cb = dataset_processed.select(
                    range(min(len(dataset_processed), self.script_cfg.callback_num_samples * 5))
                )

                if len(sample_dataset_for_inf_cb) > 0:
                    # 导入流式引导功能
                    from grpo_project.utils.streaming_guidance import create_streaming_inference_callback, GuidanceConfig
                    
                    # 配置引导参数
                    guidance_config = GuidanceConfig(
                        min_reasoning_length=self.script_cfg.min_reasoning_length if hasattr(self.script_cfg, 'min_reasoning_length') else 60,
                        guidance_trigger_threshold=self.script_cfg.guidance_trigger_threshold if hasattr(self.script_cfg, 'guidance_trigger_threshold') else 40,
                        max_guidance_attempts=self.script_cfg.max_guidance_attempts if hasattr(self.script_cfg, 'max_guidance_attempts') else 2,
                        guidance_tokens_limit=self.script_cfg.guidance_tokens_limit if hasattr(self.script_cfg, 'guidance_tokens_limit') else 25
                    )
                    
                    # 创建增强的推理回调
                    streaming_callback = create_streaming_inference_callback(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        eval_dataset=sample_dataset_for_inf_cb,
                        num_samples=self.script_cfg.callback_num_samples,
                        eval_every_n_steps=self.script_cfg.callback_eval_every_n_steps,
                        max_new_tokens=self.script_cfg.script_max_completion_length,
                        max_seq_length=self.script_cfg.max_seq_length,
                        experience_buffer=self.experience_buffer,
                        output_dir=self.script_cfg.output_dir,
                        guidance_config=guidance_config
                    )
                    
                    self.callbacks.append(streaming_callback)
                    logger.info(f"✅ 流式引导推理回调已添加 (samples: {len(sample_dataset_for_inf_cb)})")
                else:
                    logger.warning("⚠️ 样本数据不足，跳过流式引导推理回调")

            # 🔧 关键修复：禁用原生WandB回调，使用同步管理器
            if self.grpo_cfg.local_rank <= 0 and "wandb" in self.grpo_cfg.report_to:
                # 创建使用同步管理器的WandB回调
                from grpo_project.callbacks.wandb_sync_callback import SyncedWandbCallback
                wandb_cb = SyncedWandbCallback(
                    env_cfg=self.env_cfg, 
                    script_cfg=self.script_cfg, 
                    reward_cfg=self.reward_cfg, 
                    experience_buffer=self.experience_buffer
                )
                self.callbacks.append(wandb_cb)
                # 存储 wandb_callback 引用供 reward function 使用
                self.wandb_callback = wandb_cb
                logger.info("✅ SyncedWandbCallback added (替代原生WandB).")

            logger.info(f"Total callbacks prepared: {len(self.callbacks)}")
            
            # 🔧 重要：确保课程学习回调有详细的初始化日志
            if self.curriculum_manager:
                logger.info("📚 课程学习回调详细信息:")
                logger.info(f"  - 当前阶段: {self.curriculum_manager.current_stage}")
                logger.info(f"  - 总阶段数: {len(self.curriculum_manager.curriculum_stages)}")
                
                # 记录当前阶段详情
                if self.curriculum_manager.current_stage < len(self.curriculum_manager.curriculum_stages):
                    current_stage = self.curriculum_manager.curriculum_stages[self.curriculum_manager.current_stage]
                    logger.info(f"  - 当前阶段名称: {current_stage.name}")
                    logger.info(f"  - 性能阈值: {current_stage.performance_threshold}")
                    logger.info(f"  - 最小评估次数: {current_stage.min_evaluations}")
                    
                    # 记录数据集大小
                    current_dataset = self.curriculum_manager.get_current_stage_dataset()
                    logger.info(f"  - 当前阶段数据集大小: {len(current_dataset)}")

        except Exception as e:
            logger.error(f"❌ Error setting up callbacks: {e}", exc_info=True)
            # 至少保证基本的回调
            self.callbacks = [StepLoggingCallback()]
            logger.warning("⚠️ Using minimal callback setup due to errors.")
    def get_reward_function(self):
        """Create reward function closure with enhanced debugging"""
        def reward_fn_closure(prompts: List[str], completions: List[str], **kwargs_from_trainer_dataset) -> List[float]:
            try:
                current_training_step = self.trainer.state.global_step if self.trainer and self.trainer.state else 0

                # 🔧 添加详细的奖励计算日志
                if current_training_step % 10 == 0:  # 每10步记录一次
                    logger.info(f"🎯 步数 {current_training_step}: 开始奖励计算")
                    logger.info(f"  - 批次大小: {len(prompts)}")
                    logger.info(f"  - 完成长度: {[len(c) for c in completions[:3]]}{'...' if len(completions) > 3 else ''}")

                batch_rewards_args = {
                    "prompts": prompts,
                    "completions": completions,
                    "testbench_paths": kwargs_from_trainer_dataset.get('testbench_path', []),
                    "expected_total_tests_list": kwargs_from_trainer_dataset.get('expected_total_tests', []),
                    "reference_verilog_paths": kwargs_from_trainer_dataset.get('reference_verilog_path', []),
                    "original_enhanced_prompts": kwargs_from_trainer_dataset.get('original_enhanced_prompt'),
                    "training_step": current_training_step,
                    "output_dir_for_debug": self.script_cfg.output_dir,
                    # 🔧 确保传递正确的 wandb_callback 引用
                    "wandb_callback_obj": getattr(self, 'wandb_callback', None),
                    "experience_buffer_obj": self.experience_buffer,
                    "script_config_obj": self.script_cfg
                }
                
                rewards_list, aggregated_metrics = self.reward_calculator.calculate_batch_rewards(**batch_rewards_args)
                
                # 🔧 添加奖励统计日志
                if current_training_step % 10 == 0:
                    reward_stats = {
                        'mean': np.mean(rewards_list) if rewards_list else 0,
                        'std': np.std(rewards_list) if rewards_list else 0,
                        'min': np.min(rewards_list) if rewards_list else 0,
                        'max': np.max(rewards_list) if rewards_list else 0
                    }
                    logger.info(f"  - 奖励统计: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")
                    logger.info(f"  - 奖励范围: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
                
                return rewards_list
                    
            except Exception as e:
                logger.error(f"❌ Error in reward function at step {current_training_step}: {e}", exc_info=True)
                # 返回默认奖励避免训练中断
                return [0.0] * len(prompts)
        
        return reward_fn_closure

    def train(self):
        """Main training function with comprehensive error handling"""
        try:
            logger.info("🚀 Starting GRPO training process...")
            
            # 🔧 新增：设置稳定的训练环境
            self._setup_stable_environment()
            
            # 🔧 关键：在训练开始前配置WandB恢复
            logger.info("📝 Step 0: Configuring WandB resume settings...")
            self._configure_wandb_resume()
            
            # 🔧 新增：设置WandB同步管理器运行
            logger.info("📝 Step 0.5: Setting up WandB sync manager run...")
            self._setup_wandb_run()

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
                current_stage_name = self.curriculum_manager.get_current_stage_info()['stage_name']
                logger.info(f"📚 Using curriculum dataset: {len(dataset_for_trainer)} samples from stage '{current_stage_name}'.")
            else:
                logger.info(f"📚 Using full processed dataset: {len(dataset_for_trainer)} samples.")

            if not dataset_for_trainer or len(dataset_for_trainer) == 0:
                raise ValueError("❌ Training dataset is empty!")

            # 5. Create trainer
            logger.info("📝 Step 4: Creating GRPOTrainer...")
            from trl import GRPOTrainer
            
            # 🔧 关键：禁用内置WandB，完全使用我们的同步回调
            grpo_cfg_copy = self.grpo_cfg
            if "wandb" in grpo_cfg_copy.report_to:
                # 创建一个不包含wandb的copy
                import copy
                grpo_cfg_copy = copy.deepcopy(self.grpo_cfg)
                grpo_cfg_copy.report_to = [r for r in grpo_cfg_copy.report_to if r != "wandb"]
                logger.info("🔧 禁用GRPOTrainer内置WandB报告，使用同步回调")
            
            self.trainer = GRPOTrainer(
                model=self.model,
                args=grpo_cfg_copy,
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
            
            # 1. 清理WandB同步管理器
            try:
                sync_manager = get_wandb_sync_manager()
                if sync_manager:
                    sync_manager.finish()
                    logger.info("✅ WandB同步管理器已清理")
            except Exception as e:
                logger.warning(f"⚠️ WandB同步管理器清理失败: {e}")
            
            # 2. Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU cache cleared.")
            
            # 3. Force garbage collection
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