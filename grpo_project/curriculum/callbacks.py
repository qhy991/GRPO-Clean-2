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
                 performance_check_interval: int = 25):
        self.curriculum_manager = curriculum_manager
        self.trainer_ref = trainer_ref
        self.output_dir = output_dir
        self.performance_check_interval = performance_check_interval  # 新增：可配置的检查间隔
        self.debug_log_path = os.path.join(output_dir, "curriculum_progress_debug.txt") if output_dir else "curriculum_progress_debug.txt"
        self.last_locally_logged_stage_idx: int = -1
        self.evaluation_count = 0
        self.last_performance_check_step = 0
        self.performance_history = []  # 存储性能历史
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
        """评估时的详细处理 - 修复版本"""
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
        
        # 记录到性能历史 - 修复：确保包含stage信息
        self.performance_history.append({
            'step': current_step,
            'performance': performance_estimate,
            'stage': current_stage_idx,  # 修复：明确指定stage
            'timestamp': datetime.now().isoformat()
        })
        
        self._write_debug(f"📊 评估详情:")
        self._write_debug(f"  - 当前阶段: {current_stage_idx}")
        self._write_debug(f"  - 性能估计: {performance_estimate:.4f}")
        
        if current_stage_idx < len(self.curriculum_manager.curriculum_stages):
            stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
            threshold = stage_config.performance_threshold
            min_evals = stage_config.min_evaluations
            
            self._write_debug(f"  - 阶段名称: {stage_config.name}")
            self._write_debug(f"  - 性能阈值: {threshold}")
            self._write_debug(f"  - 最小评估次数要求: {min_evals}")
            self._write_debug(f"  - 当前阶段评估历史长度: {len(self.performance_history)}")
            
            # 检查进阶条件
            self._check_and_advance_stage(performance_estimate, current_step)

    def _check_and_advance_stage(self, current_performance: float, current_step: int):
        """检查并执行阶段进阶 - 修复版本"""
        current_stage_idx = self.curriculum_manager.current_stage
        
        if current_stage_idx >= len(self.curriculum_manager.curriculum_stages):
            return  # 已完成所有阶段
            
        stage_config = self.curriculum_manager.curriculum_stages[current_stage_idx]
        
        # 获取当前阶段的性能历史 - 修复：正确过滤stage
        stage_performances = [p['performance'] for p in self.performance_history 
                            if p.get('stage') == current_stage_idx]  # 修复：使用 == 而不是默认值
        
        self._write_debug(f"📊 当前阶段性能历史: {len(stage_performances)} 条记录")
        
        if len(stage_performances) >= stage_config.min_evaluations:
            recent_performances = stage_performances[-min(3, len(stage_performances)):]
            avg_recent_performance = np.mean(recent_performances)
            
            self._write_debug(f"📊 进阶条件检查:")
            self._write_debug(f"  - 当前性能: {current_performance:.4f}")
            self._write_debug(f"  - 最近平均性能: {avg_recent_performance:.4f}")
            self._write_debug(f"  - 性能阈值: {stage_config.performance_threshold}")
            self._write_debug(f"  - 评估次数: {len(stage_performances)}/{stage_config.min_evaluations}")
            
            if avg_recent_performance >= stage_config.performance_threshold:
                self._write_debug("✅ 满足进阶条件，执行阶段进阶...")
                
                old_stage = current_stage_idx
                try:
                    # 修复：优先使用课程管理器自带的should_advance_stage方法
                    if hasattr(self.curriculum_manager, 'should_advance_stage'):
                        should_advance = self.curriculum_manager.should_advance_stage(current_performance)
                        if should_advance:
                            success = self.curriculum_manager.advance_stage()
                        else:
                            success = False
                            self._write_debug("⏳ 课程管理器判断暂不进阶")
                    elif hasattr(self.curriculum_manager, 'advance_stage'):
                        success = self.curriculum_manager.advance_stage()
                    else:
                        # 手动进阶 - 作为最后的后备方案
                        self.curriculum_manager.current_stage += 1
                        success = True
                    
                    if success:
                        new_stage = self.curriculum_manager.current_stage
                        self._write_debug(f"🎯 成功进阶: 阶段{old_stage} -> 阶段{new_stage}")
                        
                        # 重置阶段计数器
                        self.step_count_in_current_stage = 0
                        
                        if new_stage < len(self.curriculum_manager.curriculum_stages):
                            new_stage_info = self.curriculum_manager.curriculum_stages[new_stage]
                            try:
                                new_dataset = self.curriculum_manager.get_current_stage_dataset()
                                self._write_debug(f"  - 新阶段名称: {new_stage_info.name}")
                                self._write_debug(f"  - 新阶段数据集大小: {len(new_dataset)}")
                                self._write_debug(f"  - 新阶段目标等级: {new_stage_info.dataset_levels}")
                            except Exception as e:
                                self._write_debug(f"  - 新阶段信息获取部分失败: {e}")
                        else:
                            self._write_debug("🏆 已完成所有课程阶段！")
                            
                except Exception as e:
                    self._write_debug(f"❌ 阶段进阶失败: {e}")
            else:
                self._write_debug(f"⏳ 未满足进阶条件 (需要性能 >= {stage_config.performance_threshold:.4f})")
        else:
            self._write_debug(f"⏳ 评估次数不足 ({len(stage_performances)}/{stage_config.min_evaluations})")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """日志记录时的处理 - 修复版本"""
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
            
            # 基于训练日志进行性能评估和可能的阶段进阶
            if logs:
                performance = self._calculate_performance_from_logs(logs)
                if performance > 0:
                    # 记录性能历史 - 修复：确保包含正确的stage信息
                    self.performance_history.append({
                        'step': current_step,
                        'performance': performance,
                        'stage': current_stage_idx,  # 修复：明确指定当前stage
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # 检查是否可以进阶
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
            train_loss = logs.get('train_loss', 'N/A')
            learning_rate = logs.get('learning_rate', 'N/A')
            self._write_debug(f"  - 训练损失: {train_loss}")
            self._write_debug(f"  - 学习率: {learning_rate}")

    def _wandb_log(self, current_step: int, logs: Optional[Dict[str, float]]):
        """W&B 记录 - 修复版本"""
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
            
            # 计算当前阶段的平均性能
            stage_performances = [p['performance'] for p in self.performance_history 
                                if p.get('stage', current_stage_idx) == current_stage_idx]
            avg_stage_performance = np.mean(stage_performances) if stage_performances else 0.0
            
            wandb_data = {
                "curriculum/current_stage_idx": current_stage_idx,
                "curriculum/current_stage_name_numeric": current_stage_idx,
                "curriculum/dataset_size": dataset_size,
                "curriculum/performance_threshold": performance_threshold,
                "curriculum/latest_performance": latest_performance,
                "curriculum/evaluation_count": len(self.performance_history),
                "curriculum/stage_step_count": self.step_count_in_current_stage,
                "curriculum/avg_stage_performance": avg_stage_performance
            }
            
            # 添加更多调试信息
            if logs:
                if 'loss' in logs:
                    wandb_data["curriculum/current_loss"] = logs['loss']
                if 'reward' in logs:
                    wandb_data["curriculum/current_reward"] = logs['reward']
                if 'learning_rate' in logs:
                    wandb_data["curriculum/learning_rate"] = logs['learning_rate']
            
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
            
        current_loss = logs.get('train_loss', float('inf'))
        training_step = getattr(state, 'global_step', 0) or 0
        
        # 记录性能
        performance = 1.0 - min(current_loss, 1.0)
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