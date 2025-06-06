import logging
import os
import json
from collections import deque # For DetailedWandbCallback from train.py
from typing import Dict, Any, Optional, List # General typing
import numpy as np # For DetailedWandbCallback from train.py
import math # For DetailedWandbCallback from train.py

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# Attempt to import from grpo_project, fallback for placeholders if necessary
try:
    from grpo_project.callbacks.base import BaseCallback
    # For the DetailedWandbCallback from train.py, it needs EnvConfig, ScriptConfig, RewardConfig
    from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig as RewardConfig
    # For GenericWandbCallback (from utils.py), it needs ExperienceBuffer
    from grpo_project.utils import ExperienceBuffer # Updated import
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Callbacks.wandb: Could not import from grpo_project or utils. Using placeholders.") # Updated path in log
    from transformers import TrainerCallback as BaseCallback # Fallback
    class EnvConfig: pass
    class ScriptConfig: pass
    class RewardConfig: pass
    class ExperienceBuffer: pass


logger = logging.getLogger(__name__)

# Originally from train.py
class DetailedWandbCallback(BaseCallback): # Changed to inherit from BaseCallback
    """增强的 W&B 日志回调，支持恢复训练时的数据连续性"""

    def __init__(self, env_cfg: EnvConfig, script_cfg: ScriptConfig, reward_cfg: RewardConfig,
                 experience_buffer: Optional[ExperienceBuffer] = None, output_dir: Optional[str] = None):
        super().__init__(output_dir=output_dir) # Pass output_dir to BaseCallback
        self.env_cfg = env_cfg
        self.script_cfg = script_cfg
        self.reward_cfg = reward_cfg
        self.experience_buffer = experience_buffer
        self.step_count = 0
        self.recent_rewards: deque[float] = deque(maxlen=100)
        self.wandb_initialized = False
        self.wandb_run = None
        
        # 🔧 关键：存储WandB run信息用于恢复
        self.wandb_run_id = None
        self.wandb_resume_mode = "allow"  # 允许恢复已存在的run
        
        logger.info(f"DetailedWandbCallback initialized. Output dir: {self.output_dir}")

    def _extract_run_id_from_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """从checkpoint目录中提取WandB run ID"""
        try:
            # 尝试从checkpoint目录的父目录名称中提取run ID
            if checkpoint_path and os.path.exists(checkpoint_path):
                # 查找wandb相关文件
                parent_dir = os.path.dirname(checkpoint_path)
                wandb_dir = os.path.join(parent_dir, "wandb")
                
                if os.path.exists(wandb_dir):
                    # 查找最新的run目录
                    run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
                    if run_dirs:
                        latest_run_dir = sorted(run_dirs)[-1]
                        run_id = latest_run_dir.replace("run-", "").split("-")[0]
                        logger.info(f"🔄 从checkpoint目录提取到WandB run ID: {run_id}")
                        return run_id
                
                # 尝试从trainer_state.json中读取
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    with open(trainer_state_path, 'r') as f:
                        state_data = json.load(f)
                        if 'log_history' in state_data:
                            # 查找包含wandb信息的日志条目
                            for log_entry in state_data['log_history']:
                                if '_wandb' in str(log_entry):
                                    logger.info("🔄 在trainer_state.json中找到WandB相关信息")
                                    break
                
        except Exception as e:
            logger.warning(f"⚠️ 提取WandB run ID时出错: {e}")
        
        return None

    def _initialize_wandb(self, args: TrainingArguments, state: TrainerState):
        """初始化WandB，支持恢复已存在的run"""
        try:
            import wandb
            
            # 🔧 关键：检查是否从checkpoint恢复
            is_resuming = (
                hasattr(args, 'resume_from_checkpoint') and 
                args.resume_from_checkpoint and 
                os.path.exists(args.resume_from_checkpoint)
            )
            
            # 🔧 重要：优先使用main.py中设置的环境变量
            env_run_id = os.getenv("WANDB_RUN_ID")
            env_resume_mode = os.getenv("WANDB_RESUME", "allow")
            
            # 准备WandB配置
            wandb_config = {
                # 训练参数
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_train_epochs": args.num_train_epochs,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "lr_scheduler_type": args.lr_scheduler_type,
                
                # 模型参数
                "model_name_or_path": getattr(self.script_cfg, 'model_name_or_path', 'unknown'),
                "lora_rank": getattr(self.script_cfg, 'lora_rank', 'unknown'),
                "lora_alpha": getattr(self.script_cfg, 'lora_alpha', 'unknown'),
                "max_seq_length": getattr(self.script_cfg, 'max_seq_length', 'unknown'),
                
                # 奖励参数
                "compilation_success": getattr(self.reward_cfg, 'compilation_success', 'unknown'),
                "test_pass_base_reward": getattr(self.reward_cfg, 'test_pass_base_reward', 'unknown'),
                
                # 恢复信息
                "is_resuming": is_resuming,
                "resume_from_checkpoint": args.resume_from_checkpoint if is_resuming else None,
                "auto_resume_configured": env_run_id is not None,
            }
            
            # 🔧 关键：设置WandB初始化参数
            wandb_init_kwargs = {
                "project": getattr(self.env_cfg, 'wandb_project', 'VerilogGRPO_Enhanced_v3'),
                "entity": getattr(self.env_cfg, 'wandb_entity', None),
                "config": wandb_config,
                "resume": env_resume_mode,
                "save_code": True,
                "tags": ["grpo", "verilog", "enhanced"],
            }
            
            # 🔧 关键：使用main.py设置的run ID（如果存在）
            if env_run_id:
                wandb_init_kwargs["id"] = env_run_id
                logger.info(f"🔄 使用main.py配置的WandB run ID: {env_run_id}")
                logger.info(f"🔄 恢复模式: {env_resume_mode}")
            elif is_resuming:
                # 备用方案：尝试从checkpoint目录提取（虽然main.py应该已经处理了）
                extracted_run_id = self._extract_run_id_from_checkpoint(args.resume_from_checkpoint)
                if extracted_run_id:
                    wandb_init_kwargs["id"] = extracted_run_id
                    wandb_init_kwargs["resume"] = "must"
                    logger.info(f"🔄 备用方案：从checkpoint提取WandB run ID: {extracted_run_id}")
                else:
                    logger.warning("⚠️ 无法确定要恢复的WandB run ID，将创建新的run")
            
            # 设置run名称（如果未指定ID）
            if "id" not in wandb_init_kwargs:
                run_name = os.getenv("WANDB_RUN_NAME")
                if run_name:
                    wandb_init_kwargs["name"] = run_name
            
            # 初始化WandB
            self.wandb_run = wandb.init(**wandb_init_kwargs)
            self.wandb_initialized = True
            self.wandb_run_id = self.wandb_run.id
            
            # 🔧 重要：保存run ID到环境变量，供后续恢复使用
            if not env_run_id:  # 只在没有预设时更新
                os.environ["WANDB_RUN_ID"] = self.wandb_run_id
            
            logger.info(f"✅ WandB初始化成功!")
            logger.info(f"  - Run ID: {self.wandb_run_id}")
            logger.info(f"  - Run URL: {self.wandb_run.url}")
            logger.info(f"  - 恢复模式: {env_resume_mode}")
            logger.info(f"  - 自动配置: {env_run_id is not None}")
            
            # 如果是恢复训练，记录恢复信息
            if is_resuming:
                self.wandb_run.log({
                    "resume_info/checkpoint_path": args.resume_from_checkpoint,
                    "resume_info/global_step": state.global_step,
                    "resume_info/epoch": state.epoch,
                    "resume_info/auto_configured": env_run_id is not None,
                }, step=state.global_step)
                
        except Exception as e:
            logger.error(f"❌ WandB初始化失败: {e}", exc_info=True)
            self.wandb_initialized = False

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练初始化结束时的回调"""
        self._initialize_wandb(args, state)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
               logs: Optional[Dict[str, float]] = None, **kwargs):
        """日志记录回调"""
        if not self.wandb_initialized or not logs:
            return
            
        try:
            import wandb
            
            # 过滤和处理日志数据
            wandb_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    wandb_logs[key] = value
            
            # 添加步数信息
            if state.global_step:
                wandb_logs["global_step"] = state.global_step
            if state.epoch:
                wandb_logs["epoch"] = state.epoch
                
            # 添加奖励统计
            if self.recent_rewards:
                wandb_logs.update({
                    "reward_stats/mean": np.mean(self.recent_rewards),
                    "reward_stats/std": np.std(self.recent_rewards),
                    "reward_stats/min": np.min(self.recent_rewards),
                    "reward_stats/max": np.max(self.recent_rewards),
                })
            
            # 记录到WandB
            if wandb_logs:
                self.wandb_run.log(wandb_logs, step=state.global_step)
                
        except Exception as e:
            logger.warning(f"⚠️ WandB日志记录失败: {e}")

    def add_reward(self, reward: float):
        """添加奖励值到历史记录"""
        if isinstance(reward, (int, float)) and not (math.isnan(reward) or math.isinf(reward)):
            self.recent_rewards.append(reward)

    def log_reward_components(self, reward_components: Dict[str, float]):
        """记录奖励组件"""
        if not self.wandb_initialized:
            return
            
        try:
            import wandb
            
            # 过滤有效的奖励组件
            valid_components = {}
            for key, value in reward_components.items():
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    valid_components[f"reward_components/{key}"] = value
            
            if valid_components:
                self.wandb_run.log(valid_components)
                
        except Exception as e:
            logger.warning(f"⚠️ 奖励组件记录失败: {e}")

    def log_batch_aggregated_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录批次聚合指标"""
        if not self.wandb_initialized:
            return
            
        try:
            import wandb
            
            # 处理聚合指标
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    wandb_metrics[f"batch_metrics/{key}"] = value
                elif isinstance(value, dict):
                    # 处理嵌套字典
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)) and not (math.isnan(sub_value) or math.isinf(sub_value)):
                            wandb_metrics[f"batch_metrics/{key}/{sub_key}"] = sub_value
            
            if wandb_metrics:
                self.wandb_run.log(wandb_metrics, step=step)
                
        except Exception as e:
            logger.warning(f"⚠️ 批次指标记录失败: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时的回调"""
        if self.wandb_initialized:
            try:
                import wandb
                
                # 记录最终统计信息
                final_stats = {
                    "final/global_step": state.global_step,
                    "final/epoch": state.epoch,
                    "final/total_flos": state.total_flos,
                }
                
                if self.recent_rewards:
                    final_stats.update({
                        "final/reward_mean": np.mean(self.recent_rewards),
                        "final/reward_std": np.std(self.recent_rewards),
                    })
                
                self.wandb_run.log(final_stats)
                logger.info("✅ WandB最终统计信息已记录")
                
            except Exception as e:
                logger.warning(f"⚠️ WandB最终统计记录失败: {e}")


# Originally from utils.py, renamed to avoid collision
class GenericWandbCallback(DetailedWandbCallback):
    """向后兼容性别名"""
    def __init__(self, env_config: EnvConfig, script_config: ScriptConfig, reward_config: RewardConfig,
                 experience_buffer: Optional[ExperienceBuffer] = None, output_dir: Optional[str] = None):
        super().__init__(env_config, script_config, reward_config, experience_buffer, output_dir)
