"""
同步WandB回调
完全使用WandB同步管理器，解决步数同步问题
"""

import logging
import math
import os
from typing import Dict, Any, Optional, List
from collections import deque

from transformers import TrainingArguments, TrainerState, TrainerControl

try:
    from grpo_project.configs import EnvConfig, ScriptConfig, RewardConfig
    from grpo_project.utils import ExperienceBuffer
    from grpo_project.callbacks.base import BaseCallback
    from grpo_project.core.wandb_sync_manager import get_wandb_sync_manager, safe_wandb_log, update_wandb_step_offset
except ImportError:
    logging.getLogger(__name__).warning("无法导入GRPO项目组件，使用占位符")
    class EnvConfig: pass
    class ScriptConfig: pass
    class RewardConfig: pass
    class ExperienceBuffer: pass
    from transformers import TrainerCallback as BaseCallback
    
    def get_wandb_sync_manager(): return None
    def safe_wandb_log(*args, **kwargs): return False
    def update_wandb_step_offset(*args, **kwargs): pass

logger = logging.getLogger(__name__)

class SyncedWandbCallback(BaseCallback):
    """使用同步管理器的WandB回调，完全替代原生WandB功能"""
    
    def __init__(self, env_cfg: EnvConfig, script_cfg: ScriptConfig, reward_cfg: RewardConfig,
                 experience_buffer: Optional[ExperienceBuffer] = None, output_dir: Optional[str] = None):
        super().__init__(output_dir=output_dir)
        self.env_cfg = env_cfg
        self.script_cfg = script_cfg
        self.reward_cfg = reward_cfg
        self.experience_buffer = experience_buffer
        
        # 奖励历史记录
        self.recent_rewards = deque(maxlen=1000)
        
        logger.info("🔧 SyncedWandbCallback初始化 - 使用WandB同步管理器")

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练初始化结束时的回调"""
        logger.info("📝 SyncedWandbCallback: 训练初始化完成")
        
        # 更新步数偏移
        if state.global_step > 0:
            update_wandb_step_offset(state.global_step)
            logger.info(f"🔄 初始化时更新步数偏移: trainer_step={state.global_step}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
               logs: Optional[Dict[str, float]] = None, **kwargs):
        """日志记录回调 - 完全替代原生WandB"""
        if not logs:
            return
            
        try:
            # 🔧 关键：先更新步数偏移
            update_wandb_step_offset(state.global_step)
            
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
                import numpy as np
                wandb_logs.update({
                    "reward_stats/mean": np.mean(self.recent_rewards),
                    "reward_stats/std": np.std(self.recent_rewards),
                    "reward_stats/min": np.min(self.recent_rewards),
                    "reward_stats/max": np.max(self.recent_rewards),
                })
            
            # 🔧 关键：使用同步管理器记录
            if wandb_logs:
                success = safe_wandb_log(wandb_logs, step=state.global_step, commit=True)
                if success:
                    logger.debug(f"✅ SyncedWandB记录成功: step={state.global_step}, 指标数={len(wandb_logs)}")
                else:
                    logger.warning(f"⚠️ SyncedWandB记录失败: step={state.global_step}")
                
        except Exception as e:
            logger.warning(f"⚠️ SyncedWandB日志记录失败: {e}")

    def add_reward(self, reward: float):
        """添加奖励值到历史记录"""
        if isinstance(reward, (int, float)) and not (math.isnan(reward) or math.isinf(reward)):
            self.recent_rewards.append(reward)

    def log_reward_components(self, reward_components: Dict[str, float], step: Optional[int] = None):
        """记录奖励组件"""
        try:
            # 更新步数偏移
            if step is not None:
                update_wandb_step_offset(step)
            
            # 过滤有效的奖励组件
            valid_components = {}
            for key, value in reward_components.items():
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    valid_components[f"reward_components/{key}"] = value
            
            if valid_components:
                success = safe_wandb_log(valid_components, step=step)
                if success:
                    logger.debug(f"✅ 奖励组件记录成功: step={step}")
                else:
                    logger.debug(f"⚠️ 奖励组件记录失败: step={step}")
                
        except Exception as e:
            logger.warning(f"⚠️ 奖励组件记录失败: {e}")

    def log_batch_aggregated_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录批次聚合指标"""
        try:
            # 更新步数偏移
            if step is not None:
                update_wandb_step_offset(step)
            
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
                success = safe_wandb_log(wandb_metrics, step=step)
                if success:
                    logger.debug(f"✅ 批次指标记录成功: step={step}")
                else:
                    logger.debug(f"⚠️ 批次指标记录失败: step={step}")
                
        except Exception as e:
            logger.warning(f"⚠️ 批次指标记录失败: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时的回调"""
        try:
            # 更新步数偏移
            update_wandb_step_offset(state.global_step)
            
            # 记录最终统计信息
            final_stats = {
                "final/global_step": state.global_step,
                "final/epoch": state.epoch,
                "final/total_flos": state.total_flos,
            }
            
            if self.recent_rewards:
                import numpy as np
                final_stats.update({
                    "final/reward_mean": np.mean(self.recent_rewards),
                    "final/reward_std": np.std(self.recent_rewards),
                })
            
            success = safe_wandb_log(final_stats, step=state.global_step)
            if success:
                logger.info("✅ SyncedWandB最终统计信息已记录")
            else:
                logger.warning("⚠️ SyncedWandB最终统计记录失败")
            
        except Exception as e:
            logger.warning(f"⚠️ SyncedWandB最终统计记录失败: {e}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """每步结束时更新步数偏移"""
        try:
            # 🔧 关键：每步都更新步数偏移，确保同步
            update_wandb_step_offset(state.global_step)
            
            # 每100步记录一次同步状态
            if state.global_step % 100 == 0:
                sync_manager = get_wandb_sync_manager()
                if sync_manager:
                    logger.info(f"🔄 步数同步状态 (第{state.global_step}步): 偏移={sync_manager.step_offset}")
            
        except Exception as e:
            logger.debug(f"步数偏移更新失败: {e}")


# 向后兼容性别名
class DetailedWandbCallback(SyncedWandbCallback):
    """向后兼容性别名"""
    pass 