# enhanced_debugging_and_fixes.py - 修复训练波动和课程学习问题

import logging
import wandb
import json
import os
from typing import Dict, Any, Optional, List
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# 1. 修复课程学习状态监控和日志
class EnhancedCurriculumDebugCallback(TrainerCallback):
    """增强的课程学习调试回调"""
    
    def __init__(self, curriculum_manager, trainer_ref=None, output_dir=None):
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        self.last_logged_stage = -1
        self.stage_change_history = []
        
        # 创建课程学习专用日志文件
        if self.output_dir:
            self.curriculum_log_file = os.path.join(output_dir, "curriculum_detailed_log.txt")
            with open(self.curriculum_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 课程学习详细调试日志 ===\n")
                f.write(f"初始化时间: {datetime.now()}\n")
                f.write(f"课程管理器是否为None: {self.curriculum_manager is None}\n")
                if self.curriculum_manager:
                    f.write(f"总阶段数: {len(self.curriculum_manager.curriculum_stages)}\n")
                    f.write(f"当前阶段: {self.curriculum_manager.current_stage}\n")
                    for i, stage in enumerate(self.curriculum_manager.curriculum_stages):
                        f.write(f"  阶段{i}: {stage.name} - 等级{stage.dataset_levels} - 复杂度{stage.complexity_range}\n")
                f.write("\n")
    
    def _write_curriculum_log(self, message: str):
        """写入课程学习日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # 控制台输出
        logger.info(f"📚 CURRICULUM: {message}")
        
        # 文件输出
        if self.curriculum_log_file:
            try:
                with open(self.curriculum_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            except Exception as e:
                logger.warning(f"无法写入课程日志文件: {e}")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not self.curriculum_manager or logs is None:
            return
        
        current_step = getattr(state, 'global_step', 0) or 0
        current_stage = self.curriculum_manager.current_stage
        
        # 检查阶段是否发生变化
        if current_stage != self.last_logged_stage:
            self._log_stage_change(current_step, current_stage)
            self.last_logged_stage = current_stage
        
        # 每50步记录一次详细状态
        if current_step % 50 == 0:
            self._log_detailed_status(current_step, logs)
        
        # 检查是否需要进阶
        self._check_advancement_conditions(current_step, logs)
    
    def _log_stage_change(self, step: int, new_stage: int):
        """记录阶段变化"""
        if new_stage < len(self.curriculum_manager.curriculum_stages):
            stage_info = self.curriculum_manager.curriculum_stages[new_stage]
            message = f"🎯 阶段变更到 {new_stage}: {stage_info.name}"
            message += f" | 等级: {stage_info.dataset_levels}"
            message += f" | 复杂度: {stage_info.complexity_range}"
            message += f" | 步数: {step}"
            
            self._write_curriculum_log(message)
            
            # 记录变更历史
            self.stage_change_history.append({
                "step": step,
                "new_stage": new_stage,
                "stage_name": stage_info.name,
                "dataset_levels": stage_info.dataset_levels,
                "complexity_range": stage_info.complexity_range,
                "timestamp": datetime.now().isoformat()
            })
            
            # W&B日志（使用数值而非文字）
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.log({
                    "curriculum/current_stage_index": new_stage,
                    "curriculum/total_stages": len(self.curriculum_manager.curriculum_stages),
                    "curriculum/stage_progress_ratio": new_stage / len(self.curriculum_manager.curriculum_stages),
                    "curriculum/stage_change_step": step,
                    # 使用数值编码代替文字
                    "curriculum/level_count": len(stage_info.dataset_levels),
                    "curriculum/complexity_min": stage_info.complexity_range[0],
                    "curriculum/complexity_max": stage_info.complexity_range[1],
                }, step=step)
    
    def _log_detailed_status(self, step: int, logs: Dict[str, Any]):
        """记录详细状态"""
        if not self.curriculum_manager:
            return
            
        current_stage = self.curriculum_manager.current_stage
        performance_history = getattr(self.curriculum_manager, 'stage_performance_history', [])
        
        # 获取当前阶段配置
        if current_stage < len(self.curriculum_manager.curriculum_stages):
            stage_config = self.curriculum_manager.curriculum_stages[current_stage]
            
            message = f"详细状态 - 步数: {step}"
            message += f" | 当前阶段: {current_stage} ({stage_config.name})"
            message += f" | 性能历史长度: {len(performance_history)}"
            message += f" | 最小评估次数: {stage_config.min_evaluations}"
            message += f" | 性能阈值: {stage_config.performance_threshold}"
            
            if performance_history:
                recent_perf = np.mean(performance_history[-3:]) if len(performance_history) >= 3 else performance_history[-1]
                message += f" | 最近性能: {recent_perf:.4f}"
            
            # 获取当前损失
            current_loss = logs.get('train_loss', logs.get('loss', float('inf')))
            if current_loss != float('inf'):
                performance_estimate = max(0, 1.0 - (current_loss / 10.0))
                message += f" | 当前损失: {current_loss:.4f} | 性能估计: {performance_estimate:.4f}"
            
            self._write_curriculum_log(message)
            
            # W&B详细状态日志
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb_data = {
                    "curriculum_detail/performance_history_length": len(performance_history),
                    "curriculum_detail/min_evaluations": stage_config.min_evaluations,
                    "curriculum_detail/performance_threshold": stage_config.performance_threshold,
                }
                
                if performance_history:
                    wandb_data["curriculum_detail/recent_performance"] = recent_perf
                    wandb_data["curriculum_detail/performance_mean"] = np.mean(performance_history)
                    wandb_data["curriculum_detail/performance_std"] = np.std(performance_history)
                
                if current_loss != float('inf'):
                    wandb_data["curriculum_detail/performance_estimate"] = performance_estimate
                
                wandb.log(wandb_data, step=step)
    
    def _check_advancement_conditions(self, step: int, logs: Dict[str, Any]):
        """检查进阶条件"""
        if not self.curriculum_manager:
            return
        
        current_loss = logs.get('train_loss', logs.get('loss', float('inf')))
        if current_loss == float('inf'):
            return
        
        performance_estimate = max(0, 1.0 - (current_loss / 10.0))
        
        try:
            should_advance = self.curriculum_manager.should_advance_stage(performance_estimate)
            
            message = f"进阶检查 - 步数: {step}"
            message += f" | 性能估计: {performance_estimate:.4f}"
            message += f" | 是否应该进阶: {should_advance}"
            
            if should_advance:
                message += " | 🚀 满足进阶条件!"
                old_stage = self.curriculum_manager.current_stage
                
                # 尝试进阶
                if hasattr(self.curriculum_manager, 'advance_stage'):
                    success = self.curriculum_manager.advance_stage()
                else:
                    success = False
                    message += " | ❌ 没有advance_stage方法"
                
                if success:
                    new_stage = self.curriculum_manager.current_stage
                    message += f" | ✅ 成功从阶段{old_stage}进阶到{new_stage}"
                    
                    # 更新训练器数据集
                    if self.trainer_ref and hasattr(self.trainer_ref, 'train_dataset'):
                        try:
                            new_dataset = self.curriculum_manager.get_current_stage_dataset()
                            self.trainer_ref.train_dataset = new_dataset
                            message += f" | 📊 已更新数据集，包含{len(new_dataset)}个样本"
                        except Exception as e:
                            message += f" | ⚠️ 更新数据集失败: {e}"
                else:
                    message += " | ❌ 进阶失败"
            
            self._write_curriculum_log(message)
            
        except Exception as e:
            error_message = f"进阶检查异常 - 步数: {step} | 错误: {e}"
            self._write_curriculum_log(error_message)
            logger.error(f"课程学习进阶检查异常: {e}", exc_info=True)


# 2. 修复Qwen3兼容性问题
class Qwen3CompatibilityFixer:
    """修复Qwen3模型兼容性问题"""
    
    @staticmethod
    def fix_generation_config(model, tokenizer):
        """修复Qwen3的生成配置"""
        from transformers import GenerationConfig
        
        logger.info("🔧 修复Qwen3生成配置...")
        
        # 确保tokenizer设置正确
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("设置pad_token为eos_token")
        
        # 修复模型配置
        model_config = getattr(model, 'config', None)
        if model_config:
            model_config.pad_token_id = tokenizer.pad_token_id
            model_config.eos_token_id = tokenizer.eos_token_id
        
        # 创建适合Qwen3的生成配置
        if not hasattr(model, 'generation_config') or model.generation_config is None:
            model.generation_config = GenerationConfig()
        
        # Qwen3特定配置
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.top_k = 40
        model.generation_config.repetition_penalty = 1.05
        
        logger.info("✅ Qwen3生成配置修复完成")
        return model, tokenizer
    
    @staticmethod
    def create_qwen3_prompt(content: str) -> str:
        """创建Qwen3格式的prompt"""
        # Qwen3使用的对话格式
        system_message = """You are a Verilog expert. Please provide your solution in the following format:

<think>
Your detailed thinking process here
</think>

```verilog
Your complete Verilog code here
```"""
        
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        return prompt


# 3. 增强的奖励稳定性监控
class RewardStabilityMonitor(TrainerCallback):
    """监控奖励稳定性，减少训练波动"""
    
    def __init__(self, output_dir: str, window_size: int = 100):
        self.output_dir = output_dir
        self.window_size = window_size
        self.reward_history = []
        self.loss_history = []
        self.stability_metrics = []
        
        # 创建奖励稳定性日志文件
        self.stability_log_file = os.path.join(output_dir, "reward_stability_log.txt")
        with open(self.stability_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 奖励稳定性监控日志 ===\n")
            f.write(f"初始化时间: {datetime.now()}\n")
            f.write(f"监控窗口大小: {window_size}\n\n")
    
    def add_reward(self, reward: float, step: int):
        """添加奖励值"""
        self.reward_history.append({"reward": reward, "step": step})
        
        # 保持窗口大小
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
    
    def add_loss(self, loss: float, step: int):
        """添加损失值"""
        self.loss_history.append({"loss": loss, "step": step})
        
        # 保持窗口大小
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
    
    def calculate_stability_metrics(self, step: int) -> Dict[str, float]:
        """计算稳定性指标"""
        if len(self.reward_history) < 10:
            return {}
        
        rewards = [item["reward"] for item in self.reward_history]
        losses = [item["loss"] for item in self.loss_history]
        
        metrics = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_cv": np.std(rewards) / (abs(np.mean(rewards)) + 1e-8),  # 变异系数
            "reward_range": np.max(rewards) - np.min(rewards),
            "reward_positive_ratio": np.mean(np.array(rewards) > 0),
        }
        
        if losses:
            metrics.update({
                "loss_mean": np.mean(losses),
                "loss_std": np.std(losses),
                "loss_cv": np.std(losses) / (abs(np.mean(losses)) + 1e-8),
            })
        
        # 计算趋势
        if len(rewards) >= 20:
            recent_rewards = rewards[-10:]
            earlier_rewards = rewards[-20:-10]
            metrics["reward_trend"] = np.mean(recent_rewards) - np.mean(earlier_rewards)
        
        self.stability_metrics.append({"step": step, "metrics": metrics})
        return metrics
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None or args.local_rank > 0:
            return
        
        current_step = getattr(state, 'global_step', 0) or 0
        
        # 记录损失
        if 'train_loss' in logs or 'loss' in logs:
            loss = logs.get('train_loss', logs.get('loss'))
            self.add_loss(loss, current_step)
        
        # 每50步计算和记录稳定性指标
        if current_step % 50 == 0 and current_step > 0:
            metrics = self.calculate_stability_metrics(current_step)
            
            if metrics:
                self._log_stability_metrics(current_step, metrics)
                
                # W&B日志
                if hasattr(wandb, 'run') and wandb.run is not None:
                    wandb_data = {f"stability/{k}": v for k, v in metrics.items()}
                    wandb.log(wandb_data, step=current_step)
                
                # 检查是否需要调整训练参数
                self._check_stability_and_suggest_adjustments(current_step, metrics)
    
    def _log_stability_metrics(self, step: int, metrics: Dict[str, float]):
        """记录稳定性指标"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_message = f"[{timestamp}] 步数: {step}\n"
        log_message += f"  奖励均值: {metrics.get('reward_mean', 0):.4f}\n"
        log_message += f"  奖励标准差: {metrics.get('reward_std', 0):.4f}\n"
        log_message += f"  奖励变异系数: {metrics.get('reward_cv', 0):.4f}\n"
        log_message += f"  正奖励比例: {metrics.get('reward_positive_ratio', 0):.4f}\n"
        if 'reward_trend' in metrics:
            log_message += f"  奖励趋势: {metrics['reward_trend']:.4f}\n"
        log_message += "\n"
        
        logger.info(f"📊 STABILITY: 步数{step} - 奖励CV: {metrics.get('reward_cv', 0):.4f}")
        
        try:
            with open(self.stability_log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)
        except Exception as e:
            logger.warning(f"无法写入稳定性日志: {e}")
    
    def _check_stability_and_suggest_adjustments(self, step: int, metrics: Dict[str, float]):
        """检查稳定性并建议调整"""
        reward_cv = metrics.get('reward_cv', 0)
        reward_positive_ratio = metrics.get('reward_positive_ratio', 0)
        
        suggestions = []
        
        # 高变异系数 -> 训练不稳定
        if reward_cv > 2.0:
            suggestions.append("奖励变异系数过高，建议降低学习率或增加batch size")
        
        # 负奖励过多 -> 奖励设计问题
        if reward_positive_ratio < 0.2:
            suggestions.append("正奖励比例过低，建议检查奖励函数设计")
        
        # 奖励趋势下降 -> 可能过拟合
        if 'reward_trend' in metrics and metrics['reward_trend'] < -1.0:
            suggestions.append("奖励呈下降趋势，可能存在过拟合")
        
        if suggestions:
            suggestion_msg = f"⚠️ 稳定性建议 (步数{step}): " + "; ".join(suggestions)
            logger.warning(suggestion_msg)
            
            try:
                with open(self.stability_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {suggestion_msg}\n\n")
            except Exception as e:
                logger.warning(f"无法写入建议日志: {e}")


# 4. 修复的奖励计算函数（减少波动）
def create_stabilized_reward_calculator(reward_config, stability_monitor: Optional[RewardStabilityMonitor] = None):
    """创建稳定化的奖励计算器"""
    
    def stabilized_reward_calculator(*args, **kwargs):
        """稳定化的奖励计算"""
        # 这里调用原始的奖励计算函数
        try:
            # 假设原始函数返回 (rewards, metrics)
            rewards, metrics = enhanced_batch_reward_calculator(*args, **kwargs)
            
            # 应用稳定化处理
            if rewards:
                # 记录到稳定性监控器
                if stability_monitor:
                    step = kwargs.get('training_step', 0)
                    for reward in rewards:
                        stability_monitor.add_reward(reward, step)
                
                # 应用奖励削峰和平滑
                stabilized_rewards = []
                for reward in rewards:
                    # 削峰处理（限制极值）
                    clipped_reward = np.clip(reward, -15.0, 15.0)
                    
                    # 轻微平滑（减少噪声）
                    if len(stabilized_rewards) > 0:
                        smoothed_reward = 0.9 * clipped_reward + 0.1 * stabilized_rewards[-1]
                    else:
                        smoothed_reward = clipped_reward
                    
                    stabilized_rewards.append(smoothed_reward)
                
                return stabilized_rewards, metrics
            
        except Exception as e:
            logger.error(f"奖励计算异常: {e}", exc_info=True)
            # 返回安全的默认值
            num_items = len(args[0]) if args and len(args) > 0 else 1
            return [-5.0] * num_items, {}
        
        return rewards, metrics
    
    return stabilized_reward_calculator


# 5. 使用示例和集成指导
def integrate_enhanced_debugging(trainer, curriculum_manager, output_dir, model, tokenizer):
    """集成所有调试增强功能"""
    
    logger.info("🔧 集成增强调试功能...")
    
    # 1. 修复Qwen3兼容性
    model, tokenizer = Qwen3CompatibilityFixer.fix_generation_config(model, tokenizer)
    
    # 2. 创建调试回调
    callbacks_to_add = []
    
    # 课程学习调试回调
    if curriculum_manager:
        curriculum_debug_cb = EnhancedCurriculumDebugCallback(
            curriculum_manager, trainer, output_dir
        )
        callbacks_to_add.append(curriculum_debug_cb)
        logger.info("✅ 添加课程学习调试回调")
    
    # 奖励稳定性监控
    stability_monitor = RewardStabilityMonitor(output_dir)
    callbacks_to_add.append(stability_monitor)
    logger.info("✅ 添加奖励稳定性监控")
    
    # 3. 将回调添加到训练器
    for callback in callbacks_to_add:
        trainer.add_callback(callback)
    
    logger.info(f"🎯 成功集成{len(callbacks_to_add)}个调试功能")
    
    return trainer, stability_monitor


# 6. 主要修复点总结
"""
主要修复的问题：

1. 课程学习问题：
   - 课程进阶逻辑修复
   - 详细的课程状态日志
   - W&B使用数值而非文字记录
   - 数据集更新机制修复

2. Qwen3兼容性：
   - 正确的对话格式
   - 生成配置优化
   - tokenizer设置修复

3. 训练稳定性：
   - 奖励削峰和平滑
   - 稳定性指标监控
   - 异常情况处理

4. 调试信息增强：
   - 专用日志文件
   - 详细的状态追踪
   - 异常捕获和处理

使用方法：
1. 在train.py中导入这些函数
2. 在trainer初始化后调用integrate_enhanced_debugging
3. 在奖励函数中使用create_stabilized_reward_calculator
"""