# grpo_project/curriculum/callbacks.py - 修复版本
import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING, List
import numpy as np

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, DefaultFlowCallback

# Attempt to import from grpo_project
try:
    from grpo_project.callbacks.base import BaseCallback
    from .manager import EnhancedCurriculumManager, FixedEnhancedCurriculumManager
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Callbacks.curriculum: Could not import from grpo_project. Using placeholders.")
    from transformers import TrainerCallback as BaseCallback
    class EnhancedCurriculumManager: pass
    class FixedEnhancedCurriculumManager: pass

if TYPE_CHECKING:
    from transformers import Trainer

logger = logging.getLogger(__name__)

class CurriculumProgressCallback(TrainerCallback):
    """增强的课程学习进度回调，产生详细的调试日志"""
    
    def __init__(self, curriculum_manager, trainer_ref: Optional['Trainer'] = None, output_dir: Optional[str] = None, 
                 performance_check_interval: int = 5):
        """
        增强的课程学习进度回调
        
        Args:
            curriculum_manager: 课程管理器实例
            trainer_ref: 训练器引用（可选）
            output_dir: 输出目录
            performance_check_interval: 性能检查间隔步数
        """
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        self.performance_check_interval = performance_check_interval  # 新增：可配置的检查间隔
        self.debug_log_path = os.path.join(output_dir, "curriculum_progress_debug.txt") if output_dir else "curriculum_progress_debug.txt"
        self.last_locally_logged_stage_idx: int = -1
        self.evaluation_count = 0
        self.last_performance_check_step = 0
        self.step_count_in_current_stage = 0  # 当前阶段的步数计数
        
        # 确保输出目录存在
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.debug_log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== CurriculumProgressCallback Debug Log - {datetime.now()} ===\n")
                f.write(f"初始化课程学习调试回调\n")
                f.write(f"调试日志路径: {self.debug_log_path}\n")
                f.write(f"输出目录: {self.output_dir}\n")
                f.write(f"性能检查间隔: 每{self.performance_check_interval}步\n")  # 新增日志
                f.write("="*80 + "\n")
        
        logger.info(f"✅ CurriculumProgressCallback initialized. Debug log: {self.debug_log_path}, Check interval: {self.performance_check_interval} steps")

    def _write_debug(self, message: str):
        """写入调试信息到专用文件和控制台"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_msg = f"[{timestamp}] CURRICULUM: {message}"
        
        # 控制台输出
        logger.info(debug_msg)
        
        # 写入调试文件
        try:
            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(debug_msg + "\n")
        except Exception as e:
            logger.warning(f"Failed to write debug log: {e}")

    def _calculate_performance_from_logs(self, logs: Optional[Dict[str, float]]) -> float:
        """从日志中计算性能指标 - 修复版本"""
        if not logs:
            return 0.0
            
        # 1. 优先使用评估指标
        if 'eval_avg_test_pass_rate' in logs:
            performance = logs['eval_avg_test_pass_rate']
            self._write_debug(f"📊 使用评估指标 eval_avg_test_pass_rate: {performance:.4f}")
            return performance
            
        # 2. 使用reward指标 (GRPO训练的核心指标)
        if 'reward' in logs:
            reward = logs['reward']
            # 将reward转换为性能分数 (假设reward > 0 表示好的性能)
            # 使用sigmoid函数将reward映射到[0,1]范围
            performance = 1.0 / (1.0 + np.exp(-max(0, reward / 5.0)))  # 缓和的sigmoid
            self._write_debug(f"📊 使用reward指标转换: reward={reward:.4f} -> performance={performance:.4f}")
            return performance
            
        # 3. 使用损失指标转换
        if 'loss' in logs:
            loss = logs['loss']
            # 将loss转换为性能分数 (loss越小，性能越好)
            performance = max(0.0, 1.0 - min(loss, 1.0))
            self._write_debug(f"📊 使用loss指标转换: loss={loss:.4f} -> performance={performance:.4f}")
            return performance
            
        if 'train_loss' in logs:
            loss = logs['train_loss']
            performance = max(0.0, 1.0 - min(loss, 1.0))
            self._write_debug(f"📊 使用train_loss指标转换: loss={loss:.4f} -> performance={performance:.4f}")
            return performance
            
        self._write_debug("⚠️ 未找到可用的性能指标，返回默认值 0.0")
        return 0.0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时的详细日志"""
        if not self.curriculum_manager:
            return
            
        self._write_debug("🚀 训练开始 - 课程学习状态初始化")
        
        current_stage_idx = self.curriculum_manager.current_stage
        total_stages = len(self.curriculum_manager.curriculum_stages)
        
        self._write_debug(f"📊 课程学习总览:")
        self._write_debug(f"  - 总阶段数: {total_stages}")
        self._write_debug(f"  - 当前阶段索引: {current_stage_idx}")
        
        # 详细记录每个阶段信息
        for i, stage in enumerate(self.curriculum_manager.curriculum_stages):
            status = "🔄 当前" if i == current_stage_idx else "⏳ 待进入" if i > current_stage_idx else "✅ 已完成"
            self._write_debug(f"  阶段{i}: {stage.name} | {status}")
            self._write_debug(f"    - 等级: {stage.dataset_levels}")
            self._write_debug(f"    - 复杂度范围: {stage.complexity_range}")
            self._write_debug(f"    - 性能阈值: {stage.performance_threshold}")
            self._write_debug(f"    - 最小评估次数: {stage.min_evaluations}")
        
        # 当前阶段详细信息
        if current_stage_idx < total_stages:
            current_stage = self.curriculum_manager.curriculum_stages[current_stage_idx]
            current_dataset = self.curriculum_manager.get_current_stage_dataset()
            
            self._write_debug(f"🎯 当前阶段详情:")
            self._write_debug(f"  - 阶段名称: {current_stage.name}")
            self._write_debug(f"  - 数据集大小: {len(current_dataset)}")
            self._write_debug(f"  - 目标等级: {current_stage.dataset_levels}")
            self._write_debug(f"  - 复杂度范围: {current_stage.complexity_range}")
            self._write_debug(f"  - 需要达到性能: {current_stage.performance_threshold}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """每步开始时的监控"""
        if not self.curriculum_manager:
            return
            
        current_step = getattr(state, 'global_step', 0) or 0
        
        # 每50步详细记录一次状态
        if current_step % 50 == 0 and current_step > 0:
            self._detailed_status_log(current_step, state)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """每步结束时的监控"""
        if not self.curriculum_manager:
            return
            
        current_step = getattr(state, 'global_step', 0) or 0
        
        # 每10步检查一次基本状态
        if current_step % 10 == 0:
            self._basic_status_check(current_step, state)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """评估时的详细处理 - 简化版本"""
        if not self.curriculum_manager or args.local_rank > 0:
            return
            
        current_step = getattr(state, 'global_step', 0) or 0
        self.evaluation_count += 1
        current_stage_idx = self.curriculum_manager.current_stage
        
        self._write_debug(f"📈 第{self.evaluation_count}次评估 (步数: {current_step})")
        
        # 从最新的日志条目中获取性能
        performance_estimate = 0.0
        if state.log_history:
            latest_logs = state.log_history[-1]
            performance_estimate = self._calculate_performance_from_logs(latest_logs)
        
        self._write_debug(f"📊 评估详情:")
        self._write_debug(f"  - 当前阶段: {current_stage_idx}")
        self._write_debug(f"  - 性能估计: {performance_estimate:.4f}")
        
        if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
            stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
            threshold = stage_config.performance_threshold
            min_evals = stage_config.min_evaluations
            
            self._write_debug(f"  - 阶段名称: {stage_config.name}")
            self._write_debug(f"  - 性能阈值: {threshold}")
            self._write_debug(f"  - 最小评估次数: {min_evals}")
            
            # 🔧 简化：让课程管理器处理评估逻辑
            if performance_estimate > 0:
                self._check_and_advance_stage(performance_estimate, current_step)

    def _check_and_advance_stage(self, current_performance: float, current_step: int):
        """检查并执行阶段进阶 - 简化版本，避免双重判断 + 完整epoch支持"""
        current_stage_idx = self.curriculum_manager.current_stage
        
        if current_stage_idx >= len(self.curriculum_manager.curriculum_stages):
            return  # 已完成所有阶段
            
        stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
        
        self._write_debug(f"📊 阶段进阶检查 (步数: {current_step})")
        self._write_debug(f"  - 当前阶段: {current_stage_idx} ({stage_config.name})")
        self._write_debug(f"  - 当前性能: {current_performance:.4f}")
        self._write_debug(f"  - 性能阈值: {stage_config.performance_threshold}")
        
        # 🔧 新增：获取完整的进阶要求检查
        advancement_reqs = self.curriculum_manager.get_stage_advancement_requirements()
        
        self._write_debug(f"📋 进阶要求检查:")
        for req in advancement_reqs['requirements']:
            status = "✅" if req['met'] else "❌"
            self._write_debug(f"  {status} {req['description']}")
            current_val = req['current']
            target_val = req['target']
            if isinstance(current_val, float):
                current_str = f"{current_val:.4f}"
            else:
                current_str = str(current_val)
            if isinstance(target_val, float):
                target_str = f"{target_val:.4f}"
            else:
                target_str = str(target_val)
            self._write_debug(f"    当前: {current_str}")
            self._write_debug(f"    目标: {target_str}")
            if req['type'] == 'full_training' and 'progress_percent' in req:
                self._write_debug(f"    训练进度: {req['progress_percent']:.1f}%")
        
        can_advance = advancement_reqs['can_advance']
        self._write_debug(f"📊 综合进阶判断: {can_advance}")
        
        # 🔧 修复：统一由课程管理器判断，避免双重逻辑
        try:
            old_stage = current_stage_idx
            
            # 让课程管理器做唯一的判断 - 传递当前步数用于训练进度更新
            if self.curriculum_manager.should_advance_stage(current_performance, current_step):
                success = self.curriculum_manager.advance_stage()
                
                if success:
                    new_stage = self.curriculum_manager.current_stage
                    self._write_debug(f"🎯 成功进阶: 阶段{old_stage} -> 阶段{new_stage}")
                    
                    # 🔧 重要：更新新阶段的开始步数
                    self.curriculum_manager.update_stage_start_step(current_step)
                    
                    # 重置阶段计数器
                    self.step_count_in_current_stage = 0
                    
                    if new_stage < len(self.curriculum_manager.curriculum_stages):
                        new_stage_info = self.curriculum_manager.curriculum_stages[new_stage]
                        try:
                            new_dataset = self.curriculum_manager.get_current_stage_dataset()
                            self._write_debug(f"  - 新阶段名称: {new_stage_info.name}")
                            self._write_debug(f"  - 新阶段数据集大小: {len(new_dataset)}")
                            self._write_debug(f"  - 新阶段目标等级: {new_stage_info.dataset_levels}")
                            self._write_debug(f"  - 新阶段要求完整epoch: {getattr(new_stage_info, 'require_full_epoch', True)}")
                        except Exception as e:
                            self._write_debug(f"  - 新阶段信息获取部分失败: {e}")
                    else:
                        self._write_debug("🏆 已完成所有课程阶段！")
                else:
                    self._write_debug("❌ 课程管理器进阶操作失败")
            else:
                # 详细说明为什么不能进阶
                unmet_reqs = [req for req in advancement_reqs['requirements'] if not req['met']]
                if unmet_reqs:
                    self._write_debug("⏳ 未满足的进阶条件:")
                    for req in unmet_reqs:
                        if req['type'] == 'performance':
                            gap = req['target'] - req['current']
                            self._write_debug(f"  - 性能差距: 需提升 {gap:.4f}")
                        elif req['type'] == 'evaluations':
                            remaining = req['target'] - req['current']
                            self._write_debug(f"  - 评估次数: 还需 {remaining} 次")
                        elif req['type'] == 'full_training':
                            remaining_epochs = req['target'] - req['current']
                            self._write_debug(f"  - 训练进度: 还需 {remaining_epochs:.2f} epoch ({req.get('progress_percent', 0):.1f}%)")
                else:
                    self._write_debug("⏳ 课程管理器判断暂不满足进阶条件")
                
        except Exception as e:
            self._write_debug(f"❌ 阶段进阶检查失败: {e}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """日志记录时的处理 - 简化版本，避免重复的性能管理"""
        if not self.curriculum_manager or args.local_rank > 0:
            return

        current_step = getattr(state, 'global_step', 0) or 0
        self.step_count_in_current_stage += 1
        current_stage_idx = self.curriculum_manager.current_stage
        
        # 检查阶段是否发生变化
        if not hasattr(self, 'last_locally_logged_stage_idx') or self.last_locally_logged_stage_idx != current_stage_idx:
            self._stage_change_log(current_step, current_stage_idx)
            self.last_locally_logged_stage_idx = current_stage_idx

        # 每N步记录一次详细状态，并检查性能（N可配置）
        if current_step % self.performance_check_interval == 0 and current_step > 0:
            self._log_curriculum_status(current_step, logs)
            
            # 🔧 简化：直接基于当前性能进行检查，不维护重复的历史
            if logs:
                performance = self._calculate_performance_from_logs(logs)
                if performance > 0:
                    # 直接检查是否可以进阶，让课程管理器管理所有数据
                    self._check_and_advance_stage(performance, current_step)

        # W&B 记录
        self._wandb_log(current_step, logs)

    def _detailed_status_log(self, current_step: int, state: TrainerState):
        """详细状态日志"""
        current_stage_idx = self.curriculum_manager.current_stage
        
        self._write_debug(f"📊 详细状态报告 (步数: {current_step})")
        self._write_debug(f"  - 当前阶段: {current_stage_idx}")
        
        if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
            stage = self.curriculum_manager.curriculum_stages[current_stage_idx]
            dataset = self.curriculum_manager.get_current_stage_dataset()
            
            self._write_debug(f"  - 阶段名称: {stage.name}")
            self._write_debug(f"  - 数据集大小: {len(dataset)}")
            self._write_debug(f"  - 性能阈值: {stage.performance_threshold}")
            
            # 获取最近的损失
            if state.log_history:
                recent_loss = state.log_history[-1].get('train_loss', 'N/A')
                self._write_debug(f"  - 最近训练损失: {recent_loss}")

    def _basic_status_check(self, current_step: int, state: TrainerState):
        """基本状态检查"""
        current_stage_idx = self.curriculum_manager.current_stage
        
        if current_step - self.last_performance_check_step >= 100:  # 每100步详细检查一次
            self._write_debug(f"🔍 基本状态检查 (步数: {current_step})")
            self._write_debug(f"  当前阶段: {current_stage_idx}")
            
            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                stage_name = self.curriculum_manager.curriculum_stages[current_stage_idx].name
                dataset_size = len(self.curriculum_manager.get_current_stage_dataset())
                self._write_debug(f"  阶段名称: {stage_name}")
                self._write_debug(f"  数据集大小: {dataset_size}")
            
            self.last_performance_check_step = current_step

    def _stage_change_log(self, current_step: int, new_stage_idx: int):
        """阶段变更日志"""
        self._write_debug(f"🔄 阶段变更检测 (步数: {current_step})")
        
        if new_stage_idx < len(self.curriculum_manager.curriculum_stages):
            stage_name = self.curriculum_manager.curriculum_stages[new_stage_idx].name
            dataset_size = len(self.curriculum_manager.get_current_stage_dataset())
            self._write_debug(f"  新阶段: {new_stage_idx} ({stage_name})")
            self._write_debug(f"  数据集大小: {dataset_size}")
        else:
            self._write_debug(f"  阶段: {new_stage_idx} (已完成所有阶段)")

    def _log_curriculum_status(self, current_step: int, logs: Optional[Dict[str, float]]):
        """记录课程状态"""
        current_stage_idx = self.curriculum_manager.current_stage
        
        self._write_debug(f"📈 课程状态更新 (步数: {current_step})")
        
        if logs:
            # 改善loss显示逻辑
            train_loss = logs.get('loss') or logs.get('train_loss')
            if train_loss is not None:
                self._write_debug(f"  - 训练损失: {train_loss:.4f}")
            else:
                self._write_debug(f"  - 训练损失: N/A (未在当前日志中)")
            
            learning_rate = logs.get('learning_rate', 'N/A')
            self._write_debug(f"  - 学习率: {learning_rate}")
            
            # 添加reward显示
            reward = logs.get('reward')
            if reward is not None:
                self._write_debug(f"  - 当前奖励: {reward:.4f}")

    def _wandb_log(self, current_step: int, logs: Optional[Dict[str, float]]):
        """W&B 记录 - 简化版本 + 完整epoch训练进度 + 详细数据集使用监控"""
        try:
            import wandb
            if wandb.run is None:
                return
                
            current_stage_idx = self.curriculum_manager.current_stage
            stage_name = "completed"
            dataset_size = 0
            performance_threshold = 0.0
            
            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                stage = self.curriculum_manager.curriculum_stages[current_stage_idx]
                stage_name = stage.name
                dataset_size = len(self.curriculum_manager.get_current_stage_dataset())
                performance_threshold = stage.performance_threshold
            
            # 获取最新的性能估计
            latest_performance = self._calculate_performance_from_logs(logs) if logs else 0.0
            
            # 🔧 修复：从课程管理器获取性能数据
            stage_evaluation_count = 0
            avg_stage_performance = 0.0
            
            if hasattr(self.curriculum_manager, 'stage_performance_history'):
                stage_performances = self.curriculum_manager.stage_performance_history
                stage_evaluation_count = len(stage_performances)
                avg_stage_performance = np.mean(stage_performances) if stage_performances else 0.0
            
            # 🔧 新增：获取完整训练进度信息
            training_status = self.curriculum_manager.get_stage_training_status()
            advancement_reqs = self.curriculum_manager.get_stage_advancement_requirements()
            
            # 🔧 新增：计算数据集使用统计
            full_dataset_size = len(self.curriculum_manager.full_dataset)
            stage_dataset_coverage = (dataset_size / full_dataset_size * 100) if full_dataset_size > 0 else 0
            
            # 🔧 新增：计算累积数据使用情况
            cumulative_samples_trained = 0
            cumulative_coverage_percent = 0
            
            if training_status and training_status.get('status') != 'no_tracker':
                steps_completed = training_status.get('steps_completed', 0)
                estimated_steps_per_epoch = training_status.get('estimated_steps_per_epoch', 1)
                
                # 假设每步处理1个样本（实际可能不同，但用于估算）
                cumulative_samples_trained = steps_completed
                cumulative_coverage_percent = min(100, (steps_completed / estimated_steps_per_epoch) * 100)
            
            wandb_data = {
                "curriculum/current_stage_idx": int(current_stage_idx),
                "curriculum/current_stage_name_numeric": int(current_stage_idx),
                "curriculum/dataset_size": int(dataset_size),
                "curriculum/performance_threshold": float(performance_threshold),
                "curriculum/latest_performance": float(latest_performance),
                "curriculum/evaluation_count": int(stage_evaluation_count),
                "curriculum/stage_step_count": int(self.step_count_in_current_stage),
                "curriculum/avg_stage_performance": float(avg_stage_performance),
                
                # 🔧 新增：数据集使用情况监控
                "curriculum/full_dataset_size": int(full_dataset_size),
                "curriculum/stage_dataset_coverage_percent": float(stage_dataset_coverage),
                "curriculum/cumulative_samples_trained": int(cumulative_samples_trained),
                "curriculum/cumulative_coverage_percent": float(cumulative_coverage_percent),
            }
            
            # 🔧 新增：完整训练进度指标
            if training_status and training_status.get('status') != 'no_tracker':
                epochs_completed = training_status.get('epochs_completed', 0)
                steps_completed = training_status.get('steps_completed', 0)
                progress_percent = training_status.get('progress_percent', 0)
                estimated_steps_per_epoch = training_status.get('estimated_steps_per_epoch', 0)
                
                wandb_data.update({
                    "curriculum/epochs_completed": float(epochs_completed),
                    "curriculum/steps_completed": int(steps_completed),
                    "curriculum/training_progress_percent": float(progress_percent),
                    "curriculum/require_full_epoch": float(training_status.get('require_full_epoch', False)),
                    "curriculum/epoch_requirement_met": float(training_status.get('is_epoch_requirement_met', False)),
                    "curriculum/estimated_steps_per_epoch": int(estimated_steps_per_epoch),
                    
                    # 🔧 新增：详细的数据使用率指标
                    "curriculum/samples_per_step": 1.0,  # 假设每步1个样本
                    "curriculum/estimated_total_samples": int(estimated_steps_per_epoch),
                    "curriculum/samples_remaining": int(max(0, estimated_steps_per_epoch - steps_completed)),
                    "curriculum/epoch_completion_ratio": float(min(1.0, epochs_completed)),
                    
                    # 🔧 新增：阶段数据使用效率
                    "curriculum/stage_data_efficiency": float(steps_completed / dataset_size) if dataset_size > 0 else 0.0,
                    "curriculum/data_reuse_count": float(epochs_completed),
                })
                
                # 🔧 新增：预测剩余训练时间（基于当前进度）
                if epochs_completed > 0 and progress_percent > 0:
                    estimated_remaining_steps = int(max(0, estimated_steps_per_epoch - steps_completed))
                    wandb_data["curriculum/estimated_remaining_steps"] = estimated_remaining_steps
                    
                    # 如果有步数历史，可以估算剩余时间
                    if hasattr(self, 'step_count_in_current_stage') and self.step_count_in_current_stage > 0:
                        steps_per_training_step = float(steps_completed / self.step_count_in_current_stage) if self.step_count_in_current_stage > 0 else 1.0
                        estimated_remaining_training_steps = float(estimated_remaining_steps / steps_per_training_step) if steps_per_training_step > 0 else 0.0
                        wandb_data["curriculum/estimated_remaining_training_steps"] = estimated_remaining_training_steps
            
            # 🔧 新增：进阶要求满足情况
            if advancement_reqs and 'requirements' in advancement_reqs:
                wandb_data["curriculum/can_advance"] = float(advancement_reqs['can_advance'])
                
                # 分别记录各项要求的满足情况
                for req in advancement_reqs['requirements']:
                    req_type = req['type']
                    wandb_data[f"curriculum/{req_type}_requirement_met"] = float(req['met'])
                    wandb_data[f"curriculum/{req_type}_current"] = req['current']
                    wandb_data[f"curriculum/{req_type}_target"] = req['target']
                    
                    # 🔧 新增：计算每项要求的完成度
                    if req['target'] > 0:
                        completion_ratio = min(1.0, req['current'] / req['target'])
                        wandb_data[f"curriculum/{req_type}_completion_ratio"] = completion_ratio
            
            # 🔧 新增：阶段级别的数据分布信息
            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
                
                # 记录阶段配置信息
                wandb_data.update({
                    "curriculum/stage_complexity_min": float(stage_config.complexity_range[0]),
                    "curriculum/stage_complexity_max": float(stage_config.complexity_range[1]),
                    "curriculum/stage_complexity_span": float(stage_config.complexity_range[1] - stage_config.complexity_range[0]),
                    "curriculum/stage_levels_count": int(len(stage_config.dataset_levels)),
                    "curriculum/stage_min_evaluations": int(stage_config.min_evaluations),
                    "curriculum/stage_require_full_epoch": float(getattr(stage_config, 'require_full_epoch', True)),
                    "curriculum/stage_min_steps_per_epoch": int(getattr(stage_config, 'min_steps_per_epoch', 10)),
                })
                
                # 🔧 修复：将数据级别转换为数值编码，避免文字显示
                level_mapping = {
                    'basic': 1.0,
                    'intermediate': 2.0, 
                    'advanced': 3.0,
                    'expert': 4.0,
                    'master': 5.0
                }
                
                # 记录当前阶段包含的级别（数值形式）
                stage_levels_encoded = []
                for level in stage_config.dataset_levels:
                    level_encoded = level_mapping.get(level.lower(), 0.0)
                    stage_levels_encoded.append(level_encoded)
                
                # 记录级别统计
                wandb_data.update({
                    "curriculum/stage_has_basic": float('basic' in [l.lower() for l in stage_config.dataset_levels]),
                    "curriculum/stage_has_intermediate": float('intermediate' in [l.lower() for l in stage_config.dataset_levels]),
                    "curriculum/stage_has_advanced": float('advanced' in [l.lower() for l in stage_config.dataset_levels]),
                    "curriculum/stage_has_expert": float('expert' in [l.lower() for l in stage_config.dataset_levels]),
                    "curriculum/stage_has_master": float('master' in [l.lower() for l in stage_config.dataset_levels]),
                    "curriculum/stage_level_diversity": float(len(set(stage_config.dataset_levels))),
                    "curriculum/stage_min_level": float(min(stage_levels_encoded) if stage_levels_encoded else 0),
                    "curriculum/stage_max_level": float(max(stage_levels_encoded) if stage_levels_encoded else 0),
                })
            
            # 添加更多调试信息 - 确保都是数值类型
            if logs:
                if 'loss' in logs:
                    wandb_data["curriculum/current_loss"] = float(logs['loss'])
                if 'reward' in logs:
                    wandb_data["curriculum/current_reward"] = float(logs['reward'])
                if 'learning_rate' in logs:
                    wandb_data["curriculum/learning_rate"] = float(logs['learning_rate'])
            
            # 🔧 新增：阶段性能趋势 - 确保都是数值类型
            if hasattr(self.curriculum_manager, 'stage_performance_history') and self.curriculum_manager.stage_performance_history:
                history = self.curriculum_manager.stage_performance_history
                if len(history) >= 2:
                    recent_trend = float(history[-1] - history[-2])
                    wandb_data["curriculum/performance_trend"] = recent_trend
                    
                if len(history) >= 3:
                    recent_avg = float(np.mean(history[-3:]))
                    wandb_data["curriculum/recent_3_avg_performance"] = recent_avg
                    
                    # 性能稳定性（最近3次的标准差）
                    recent_std = float(np.std(history[-3:]))
                    wandb_data["curriculum/performance_stability"] = recent_std
                    
                # 🔧 新增：更多性能统计
                wandb_data.update({
                    "curriculum/performance_history_length": int(len(history)),
                    "curriculum/performance_min": float(min(history)),
                    "curriculum/performance_max": float(max(history)),
                    "curriculum/performance_range": float(max(history) - min(history)),
                    "curriculum/performance_latest": float(history[-1]),
                })
                
                # 计算性能改善趋势（如果有足够数据）
                if len(history) >= 5:
                    early_avg = float(np.mean(history[:2]))
                    recent_avg = float(np.mean(history[-2:]))
                    improvement = recent_avg - early_avg
                    wandb_data["curriculum/performance_improvement"] = improvement
            
            # 🔧 新增：阶段名称的数值编码（用于图表显示）
            stage_name_mapping = {
                'foundation': 0.0,
                'elementary': 1.0,
                'intermediate': 2.0,
                'advanced': 3.0,
                'expert': 4.0,
                'comprehensive': 5.0
            }
            
            if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
                stage_name = self.curriculum_manager.curriculum_stages[current_stage_idx].name
                stage_name_encoded = stage_name_mapping.get(stage_name.lower(), current_stage_idx)
                wandb_data["curriculum/stage_name_encoded"] = float(stage_name_encoded)
            else:
                wandb_data["curriculum/stage_name_encoded"] = 6.0  # completed
            
            # 🔧 确保所有核心指标都是数值类型
            wandb_data.update({
                "curriculum/current_stage_idx": int(current_stage_idx),
                "curriculum/current_stage_name_numeric": int(current_stage_idx),
                "curriculum/dataset_size": int(dataset_size),
                "curriculum/performance_threshold": float(performance_threshold),
                "curriculum/latest_performance": float(latest_performance),
                "curriculum/evaluation_count": int(stage_evaluation_count),
                "curriculum/stage_step_count": int(self.step_count_in_current_stage),
                "curriculum/avg_stage_performance": float(avg_stage_performance),
                "curriculum/full_dataset_size": int(full_dataset_size),
                "curriculum/stage_dataset_coverage_percent": float(stage_dataset_coverage),
                "curriculum/cumulative_samples_trained": int(cumulative_samples_trained),
                "curriculum/cumulative_coverage_percent": float(cumulative_coverage_percent),
            })
            
            wandb.log(wandb_data, step=current_step)
            
        except ImportError:
            pass
        except Exception as e:
            self._write_debug(f"⚠️ W&B 记录异常: {e}")


class EnhancedCurriculumDebugCallback(TrainerCallback):
    """增强的课程学习调试回调 - 提供更深入的调试信息"""
    
    def __init__(self, curriculum_manager, trainer_ref: Optional['Trainer'] = None, output_dir: Optional[str] = None):
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        self.last_logged_stage: int = -1
        self.stage_change_history: List[Dict[str, Any]] = []
        self.curriculum_log_file: Optional[str] = None
        self.performance_history: List[Dict[str, Any]] = []
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.curriculum_log_file = os.path.join(self.output_dir, "enhanced_curriculum_debug_log.txt")
            
            # 初始化日志文件
            with open(self.curriculum_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Enhanced Curriculum Debug Log - {datetime.now()} ===\n")
                f.write("详细的课程学习调试信息\n")
                f.write("="*80 + "\n")
        
        logger.info(f"✅ EnhancedCurriculumDebugCallback initialized. Log: {self.curriculum_log_file}")

    def _write_curriculum_log(self, message: str):
        """写入课程调试日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] ENHANCED_CURRICULUM: {message}"
        
        # 控制台输出
        logger.info(log_message)
        
        # 文件输出
        if self.curriculum_log_file:
            try:
                with open(self.curriculum_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            except Exception as e:
                logger.warning(f"Failed to write curriculum log: {e}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时的增强日志"""
        if not self.curriculum_manager:
            return
            
        self._write_curriculum_log("🚀 Enhanced curriculum debugging started")
        
        # 记录课程管理器的详细信息
        if hasattr(self.curriculum_manager, 'debug_log'):
            recent_debug = self.curriculum_manager.debug_log[-10:] if self.curriculum_manager.debug_log else []
            self._write_curriculum_log(f"📋 Curriculum manager debug log entries: {len(self.curriculum_manager.debug_log)}")
            for entry in recent_debug:
                self._write_curriculum_log(f"  - {entry}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """详细的日志处理"""
        if not self.curriculum_manager or args.local_rank > 0:
            return

        current_step = getattr(state, 'global_step', 0) or 0
        current_stage = self.curriculum_manager.current_stage
        
        # 检查阶段变化
        if self.last_logged_stage != current_stage:
            self._log_stage_change(current_step, current_stage)
            self.last_logged_stage = current_stage

        # 每20步记录详细状态
        if current_step % 20 == 0 and current_step > 0:
            self._log_detailed_status(current_step, logs)

        # 记录性能历史
        if logs:
            performance_data = {
                'step': current_step,
                'stage': current_stage,
                'timestamp': datetime.now().isoformat()
            }
            
            # 收集所有可能的性能指标
            performance_keys = ['train_loss', 'eval_loss', 'eval_avg_test_pass_rate', 'learning_rate']
            for key in performance_keys:
                if key in logs:
                    performance_data[key] = logs[key]
            
            self.performance_history.append(performance_data)
            
            # 只保留最近1000条记录
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        # 检查进阶条件
        if current_step % 10 == 0:  # 每10步检查一次
            self._check_advancement_conditions(current_step, logs)

    def _log_stage_change(self, step: int, new_stage: int):
        """记录阶段变化"""
        self._write_curriculum_log(f"🔄 Stage change detected at step {step}")
        self._write_curriculum_log(f"  Old stage: {self.last_logged_stage}")
        self._write_curriculum_log(f"  New stage: {new_stage}")
        
        if new_stage < len(self.curriculum_manager.curriculum_stages):
            stage_info = self.curriculum_manager.curriculum_stages[new_stage]
            dataset = self.curriculum_manager.get_current_stage_dataset()
            
            self._write_curriculum_log(f"  Stage name: {stage_info.name}")
            self._write_curriculum_log(f"  Dataset size: {len(dataset)}")
            self._write_curriculum_log(f"  Target levels: {stage_info.dataset_levels}")
            self._write_curriculum_log(f"  Complexity range: {stage_info.complexity_range}")
            self._write_curriculum_log(f"  Performance threshold: {stage_info.performance_threshold}")
        
        # 记录变化历史
        change_record = {
            'step': step,
            'old_stage': self.last_logged_stage,
            'new_stage': new_stage,
            'timestamp': datetime.now().isoformat()
        }
        self.stage_change_history.append(change_record)

    def _log_detailed_status(self, step: int, logs: Dict[str, Any]):
        """记录详细状态"""
        self._write_curriculum_log(f"📊 Detailed status at step {step}")
        
        current_stage = self.curriculum_manager.current_stage
        self._write_curriculum_log(f"  Current stage: {current_stage}")
        
        if current_stage < len(self.curriculum_manager.curriculum_stages):
            stage = self.curriculum_manager.curriculum_stages[current_stage]
            
            # 阶段信息
            self._write_curriculum_log(f"  Stage info:")
            self._write_curriculum_log(f"    - Name: {stage.name}")
            self._write_curriculum_log(f"    - Performance threshold: {stage.performance_threshold}")
            self._write_curriculum_log(f"    - Min evaluations: {stage.min_evaluations}")
            
            # 当前性能历史
            if hasattr(self.curriculum_manager, 'stage_performance_history'):
                history = self.curriculum_manager.stage_performance_history
                self._write_curriculum_log(f"    - Current stage evaluations: {len(history)}")
                if history:
                    recent = history[-3:]
                    avg_recent = np.mean(recent) if recent else 0
                    self._write_curriculum_log(f"    - Recent performance avg: {avg_recent:.4f}")
            
            # 数据集信息
            dataset = self.curriculum_manager.get_current_stage_dataset()
            self._write_curriculum_log(f"  Dataset size: {len(dataset)}")
        
        # 训练指标
        if logs:
            self._write_curriculum_log(f"  Training metrics:")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self._write_curriculum_log(f"    - {key}: {value:.6f}")

    def _check_advancement_conditions(self, step: int, logs: Dict[str, Any]):
        """检查进阶条件"""
        if not logs:
            return
            
        current_stage = self.curriculum_manager.current_stage
        if current_stage >= len(self.curriculum_manager.curriculum_stages) - 1:
            return  # 已经是最后阶段
        
        # 尝试获取性能指标
        performance_metrics = ['eval_avg_test_pass_rate', 'eval_loss', 'train_loss']
        performance_value = None
        performance_key = None
        
        for key in performance_metrics:
            if key in logs:
                performance_value = logs[key]
                performance_key = key
                break
        
        if performance_value is not None:
            # 转换性能值
            if performance_key == 'eval_loss' or performance_key == 'train_loss':
                performance_estimate = max(0, 1.0 - min(performance_value, 1.0))
            else:
                performance_estimate = performance_value
            
            stage = self.curriculum_manager.curriculum_stages[current_stage]
            threshold = stage.performance_threshold
            
            # 获取历史
            history = getattr(self.curriculum_manager, 'stage_performance_history', [])
            
            self._write_curriculum_log(f"🔍 Advancement check at step {step}")
            self._write_curriculum_log(f"  Performance metric: {performance_key} = {performance_value:.4f}")
            self._write_curriculum_log(f"  Performance estimate: {performance_estimate:.4f}")
            self._write_curriculum_log(f"  Threshold: {threshold:.4f}")
            self._write_curriculum_log(f"  Stage evaluations: {len(history)}/{stage.min_evaluations}")
            
            if len(history) >= 3:  # 至少有3次评估才分析趋势
                recent_trend = history[-3:]
                trend_direction = "improving" if len(recent_trend) > 1 and recent_trend[-1] > recent_trend[0] else "stable/declining"
                self._write_curriculum_log(f"  Recent trend: {trend_direction}")
            
            # 检查是否满足进阶条件
            meets_threshold = performance_estimate >= threshold
            meets_min_eval = len(history) >= stage.min_evaluations
            
            self._write_curriculum_log(f"  Meets threshold: {meets_threshold}")
            self._write_curriculum_log(f"  Meets min evaluations: {meets_min_eval}")
            
            if meets_threshold and meets_min_eval:
                self._write_curriculum_log("✅ Ready for advancement!")
            else:
                missing = []
                if not meets_threshold:
                    missing.append(f"performance ({performance_estimate:.4f} < {threshold:.4f})")
                if not meets_min_eval:
                    missing.append(f"evaluations ({len(history)} < {stage.min_evaluations})")
                self._write_curriculum_log(f"⏳ Not ready: {', '.join(missing)}")


class OptimizedCurriculumCallback(DefaultFlowCallback):
    """优化的课程学习回调"""
    
    def __init__(self, curriculum_manager, trainer_ref: Optional['Trainer'] = None, output_dir: Optional[str] = None):
        super().__init__()
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        # Note: DynamicDifficultyAdjuster is not implemented yet
        # self.difficulty_adjuster = DynamicDifficultyAdjuster(curriculum_manager)
        self.performance_history: List[Dict[str, Any]] = []
        logger.info("OptimizedCurriculumCallback initialized (DynamicDifficultyAdjuster not implemented).")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """处理日志并检查课程进展"""
        if logs is None or not self.curriculum_manager:
            return
            
        # 修复：正确获取loss值，避免无穷大
        current_loss = None
        if 'loss' in logs:
            current_loss = logs['loss']
        elif 'train_loss' in logs:
            current_loss = logs['train_loss']
        else:
            current_loss = 0.0  # 使用0.0而不是无穷大作为默认值
            
        training_step = getattr(state, 'global_step', 0) or 0
        
        # 记录性能 - 修复：处理loss为负数的情况
        if current_loss is not None and current_loss != float('inf'):
            # 对于GRPO，loss可能为负数，使用sigmoid转换
            performance = 1.0 / (1.0 + np.exp(-max(0, -current_loss)))
        else:
            performance = 0.0
        self.performance_history.append({
            'step': training_step,
            'performance': performance,
            'loss': current_loss,
            'stage': self.curriculum_manager.current_stage
        })
        
        # 检查是否需要阶段进阶
        if hasattr(self.curriculum_manager, 'should_advance_to_next_stage'):
            should_advance = self.curriculum_manager.should_advance_to_next_stage(current_loss, training_step)
        else:
            # 回退到简单的进阶检查
            should_advance = (performance > 0.7 and 
                            len(self.performance_history) > 10 and
                            training_step % 100 == 0)
        
        if should_advance:
            old_stage = self.curriculum_manager.current_stage
            
            if hasattr(self.curriculum_manager, 'advance_to_next_stage'):
                success = self.curriculum_manager.advance_to_next_stage()
            elif hasattr(self.curriculum_manager, 'advance_stage'):
                success = self.curriculum_manager.advance_stage()
            else:
                success = False
                logger.warning("Curriculum manager lacks advancement method")
            
            if success:
                logger.info(f"🎯 课程进阶: {old_stage} → {self.curriculum_manager.current_stage}")
                if hasattr(self.curriculum_manager, 'get_current_stage_dataset'):
                    new_dataset = self.curriculum_manager.get_current_stage_dataset()
                    logger.info(f"📊 新数据集大小: {len(new_dataset)}")
        
        # 每50步保存课程状态
        if training_step % 50 == 0:
            self._save_curriculum_state()
    
    def _save_curriculum_state(self):
        """保存课程学习状态"""
        if not self.output_dir:
            return
            
        state_data = {
            'current_stage': self.curriculum_manager.current_stage,
            'performance_history': self.performance_history[-100:],  # 保存最近100条
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = os.path.join(self.output_dir, 'curriculum_state_detailed.json')
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存课程状态失败: {e}")

class DatasetCoverageMonitorCallback(TrainerCallback):
    """数据集覆盖率监控回调 - 确保所有数据都被使用"""
    
    def __init__(self, curriculum_manager, output_dir: Optional[str] = None):
        self.curriculum_manager = curriculum_manager
        self.output_dir = output_dir
        self.used_sample_indices = set()
        self.stage_sample_usage = {}
        self.coverage_log_file = None
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.coverage_log_file = os.path.join(self.output_dir, "dataset_coverage_monitor.txt")
            
            with open(self.coverage_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Dataset Coverage Monitor - {datetime.now()} ===\n")
                f.write("监控数据集使用覆盖率\n")
                f.write("="*80 + "\n")
        
        logger.info(f"✅ DatasetCoverageMonitorCallback initialized. Log: {self.coverage_log_file}")

    def _write_coverage_log(self, message: str):
        """写入覆盖率监控日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] COVERAGE: {message}"
        
        logger.info(log_message)
        
        if self.coverage_log_file:
            try:
                with open(self.coverage_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            except Exception as e:
                logger.warning(f"Failed to write coverage log: {e}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时初始化覆盖率监控"""
        if not self.curriculum_manager:
            return
            
        self._write_coverage_log("🚀 开始监控数据集覆盖率")
        
        # 获取总数据集大小
        total_samples = len(self.curriculum_manager.full_dataset)
        self._write_coverage_log(f"📊 总数据集大小: {total_samples} 样本")
        
        # 分析每个阶段的理论覆盖情况
        if hasattr(self.curriculum_manager, 'coverage_analysis'):
            coverage = self.curriculum_manager.coverage_analysis
            self._write_coverage_log(f"📈 理论覆盖率: {coverage['coverage_ratio']*100:.1f}%")
            self._write_coverage_log(f"📈 理论覆盖样本: {coverage['covered_samples']}/{coverage['total_samples']}")
            
            if coverage['uncovered_count'] > 0:
                self._write_coverage_log(f"⚠️ 警告: {coverage['uncovered_count']} 个样本未被任何阶段覆盖")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """记录当前阶段的数据使用情况"""
        if not self.curriculum_manager or args.local_rank > 0:
            return

        current_step = getattr(state, 'global_step', 0) or 0
        current_stage_idx = self.curriculum_manager.current_stage
        
        # 每100步记录一次覆盖率状态
        if current_step % 100 == 0 and current_step > 0:
            self._log_coverage_status(current_step, current_stage_idx)

    def _log_coverage_status(self, step: int, stage_idx: int):
        """记录覆盖率状态"""
        if stage_idx >= len(self.curriculum_manager.curriculum_stages):
            stage_name = "completed"
            dataset_size = len(self.curriculum_manager.full_dataset)
        else:
            stage = self.curriculum_manager.curriculum_stages[stage_idx]
            stage_name = stage.name
            current_dataset = self.curriculum_manager.get_current_stage_dataset()
            dataset_size = len(current_dataset)
            
            # 记录当前阶段使用的样本
            if stage_name not in self.stage_sample_usage:
                self.stage_sample_usage[stage_name] = set()
            
            # 这里需要更复杂的逻辑来跟踪实际使用的样本索引
            # 目前只记录理论上的数据集大小
        
        total_samples = len(self.curriculum_manager.full_dataset)
        coverage_ratio = dataset_size / total_samples if total_samples > 0 else 0
        
        self._write_coverage_log(f"📊 步数 {step} - 阶段 {stage_name}")
        self._write_coverage_log(f"  - 当前阶段数据集: {dataset_size} 样本")
        self._write_coverage_log(f"  - 当前阶段覆盖率: {coverage_ratio*100:.1f}%")
        
        # 累计覆盖率统计
        total_used = sum(len(usage) for usage in self.stage_sample_usage.values())
        cumulative_coverage = total_used / total_samples if total_samples > 0 else 0
        self._write_coverage_log(f"  - 累计覆盖率: {cumulative_coverage*100:.1f}%")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时生成最终覆盖率报告"""
        if not self.curriculum_manager:
            return
            
        self._write_coverage_log("🏁 训练结束 - 生成最终覆盖率报告")
        
        total_samples = len(self.curriculum_manager.full_dataset)
        
        # 统计每个阶段的使用情况
        self._write_coverage_log("📈 各阶段数据使用统计:")
        for stage_name, usage in self.stage_sample_usage.items():
            count = len(usage)
            ratio = count / total_samples if total_samples > 0 else 0
            self._write_coverage_log(f"  - {stage_name}: {count} 样本 ({ratio*100:.1f}%)")
        
        # 总体覆盖率
        all_used = set()
        for usage in self.stage_sample_usage.values():
            all_used.update(usage)
        
        final_coverage = len(all_used) / total_samples if total_samples > 0 else 0
        unused_count = total_samples - len(all_used)
        
        self._write_coverage_log(f"📊 最终覆盖率统计:")
        self._write_coverage_log(f"  - 总样本数: {total_samples}")
        self._write_coverage_log(f"  - 已使用样本: {len(all_used)} ({final_coverage*100:.1f}%)")
        self._write_coverage_log(f"  - 未使用样本: {unused_count} ({(1-final_coverage)*100:.1f}%)")
        
        if unused_count > 0:
            self._write_coverage_log(f"⚠️ 警告: {unused_count} 个样本在整个训练过程中从未被使用!")
        else:
            self._write_coverage_log("✅ 所有数据样本都被使用了")
        
        # 保存详细报告
        self._save_detailed_coverage_report()

    def _save_detailed_coverage_report(self):
        """保存详细的覆盖率报告"""
        if not self.output_dir:
            return
            
        report_file = os.path.join(self.output_dir, "dataset_coverage_detailed_report.json")
        
        total_samples = len(self.curriculum_manager.full_dataset)
        all_used = set()
        for usage in self.stage_sample_usage.values():
            all_used.update(usage)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": total_samples,
            "total_used_samples": len(all_used),
            "final_coverage_ratio": len(all_used) / total_samples if total_samples > 0 else 0,
            "unused_sample_count": total_samples - len(all_used),
            "stage_usage": {
                stage_name: {
                    "sample_count": len(usage),
                    "coverage_ratio": len(usage) / total_samples if total_samples > 0 else 0,
                    "sample_indices": list(usage)
                }
                for stage_name, usage in self.stage_sample_usage.items()
            },
            "unused_sample_indices": list(set(range(total_samples)) - all_used)
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self._write_coverage_log(f"💾 详细覆盖率报告已保存: {report_file}")
        except Exception as e:
            self._write_coverage_log(f"❌ 保存覆盖率报告失败: {e}")