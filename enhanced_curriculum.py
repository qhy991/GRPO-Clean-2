# enhanced_curriculum.py - 双层课程学习系统
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from dataclasses import dataclass
import wandb

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStageConfig:
    """课程学习阶段配置"""
    name: str
    dataset_levels: List[str]           # 数据集等级过滤 ['basic', 'intermediate', 'advanced', 'expert']
    complexity_range: Tuple[float, float]  # 复杂度范围 (min, max)
    epochs_ratio: float                 # 该阶段训练epoch比例
    performance_threshold: float = 0.6  # 进入下一阶段的性能阈值
    min_evaluations: int = 5           # 最少评估次数
    description: str = ""              # 阶段描述

class EnhancedCurriculumManager:
    """增强的双层课程学习管理器"""
    
    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        # 🔧 优化进阶条件 - 针对不同阶段使用不同的标准
        self.stage_progression_configs = {
            0: {  # foundation阶段
                "performance_threshold": 0.7,
                "min_evaluations": 8,
                "stability_window": 4
            },
            1: {  # elementary阶段  
                "performance_threshold": 0.65,
                "min_evaluations": 10,
                "stability_window": 5
            },
            2: {  # intermediate阶段
                "performance_threshold": 0.6,
                "min_evaluations": 15,  # 增加最小评估次数
                "stability_window": 6
            },
            3: {  # advanced阶段 - 最严格的条件
                "performance_threshold": 0.55,
                "min_evaluations": 20,  # 大幅增加
                "stability_window": 8,
                "max_stay_steps": 200   # 最大停留步数
            }
        }
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history = []
        self.stage_statistics = []
        
        # 分析数据集分布
        self._analyze_dataset_distribution()
        
        # 验证课程设计
        self._validate_curriculum_design()
        
        logger.info(f"Enhanced Curriculum Manager initialized with {len(curriculum_stages)} stages")
        for i, stage in enumerate(curriculum_stages):
            logger.info(f"  Stage {i}: {stage.name} - Levels: {stage.dataset_levels}, "
                       f"Complexity: {stage.complexity_range}, Ratio: {stage.epochs_ratio}")
    def should_advance_to_next_stage(self, current_loss, training_step):
        """增强的进阶判断逻辑，针对高难度阶段优化"""
        current_stage = self.current_stage
        
        # 如果已经是最后阶段，不再进阶
        if current_stage >= len(self.curriculum_stages) - 1:
            logger.debug(f"已处于最后阶段 ({current_stage})，不再进阶")
            return False
        
        # 🔧 新增：初始化阶段开始步数跟踪
        if not hasattr(self, 'stage_start_steps'):
            self.stage_start_steps = {}
        
        if current_stage not in self.stage_start_steps:
            self.stage_start_steps[current_stage] = training_step
            logger.info(f"🎯 开始跟踪阶段{current_stage}，起始步数: {training_step}")
        
        stage_start_step = self.stage_start_steps[current_stage]
        steps_in_current_stage = training_step - stage_start_step
        
        # 🔧 新增：不同阶段的最小停留时间要求
        min_steps_by_stage = {
            0: 15,   # foundation - 至少15步
            1: 20,   # elementary - 至少20步  
            2: 30,   # intermediate - 至少30步
            3: 50    # advanced - 至少50步
        }
        
        min_steps_required = min_steps_by_stage.get(current_stage, 20)
        
        if steps_in_current_stage < min_steps_required:
            logger.info(f"⏱️ 阶段{current_stage}需要更多训练: {steps_in_current_stage}/{min_steps_required} 步")
            return False
        
        # 获取阶段配置
        config = self.stage_progression_configs.get(current_stage, {})
        
        # 获取配置参数
        performance_threshold = config.get("performance_threshold", 0.6)
        min_evaluations = config.get("min_evaluations", 10)
        stability_window = config.get("stability_window", 5)
        max_stay_steps = config.get("max_stay_steps", None)
        
        # 计算当前性能
        performance_estimate = 1.0 - min(current_loss, 1.0)
        
        # 添加到性能历史
        self.stage_performance_history.append({
            'step': training_step,
            'performance': performance_estimate,
            'loss': current_loss,
            'stage': current_stage,
            'steps_in_stage': steps_in_current_stage  # 新增：记录在当前阶段的步数
        })
        
        # 只保留当前阶段的性能历史
        current_stage_history = [
            h for h in self.stage_performance_history 
            if h['stage'] == current_stage
        ]
        
        # 检查是否有足够的评估次数
        if len(current_stage_history) < min_evaluations:
            logger.debug(f"阶段{current_stage}: 评估次数不足 ({len(current_stage_history)}/{min_evaluations})")
            return False
        
        # 检查最近的性能稳定性
        recent_performance = [h['performance'] for h in current_stage_history[-stability_window:]]
        
        if len(recent_performance) < stability_window:
            logger.debug(f"阶段{current_stage}: 稳定性窗口数据不足 ({len(recent_performance)}/{stability_window})")
            return False
            
        # 计算性能指标
        avg_performance = np.mean(recent_performance)
        performance_std = np.std(recent_performance)
        
        # 🔧 高难度阶段的特殊处理
        if current_stage == 3:  # advanced阶段
            logger.info(f"🔥 高难度阶段检查 - 步数: {training_step}, 阶段内步数: {steps_in_current_stage}")
            
            # 更严格的稳定性要求
            if performance_std > 0.05:  # 标准差不能太大
                logger.info(f"   📊 性能稳定性不足: std={performance_std:.4f} (要求≤0.05)")
                return False
            
            # 更严格的性能要求
            if avg_performance < 0.85:  # 高难度阶段要求更高的平均性能
                logger.info(f"   📈 平均性能不足: {avg_performance:.4f} (要求≥0.85)")
                return False
            
            # 检查是否达到最大停留步数
            if max_stay_steps and steps_in_current_stage >= max_stay_steps:
                logger.info(f"   ⏰ 达到最大停留步数 ({max_stay_steps})，强制保持在高难度阶段")
                return False  # 不进阶，继续在高难度阶段训练
            
            # 高难度阶段要求连续优秀性能
            if len(recent_performance) >= 5:
                excellent_performance_count = sum(1 for p in recent_performance[-5:] if p >= 0.9)
                if excellent_performance_count < 3:  # 最近5次中至少3次优秀
                    logger.info(f"   🎯 优秀性能次数不足: {excellent_performance_count}/5 次≥0.9")
                    return False
            
            logger.info(f"   ✅ 高难度阶段所有检查通过")
        
        # 🔧 中等难度阶段的特殊处理
        elif current_stage == 2:  # intermediate阶段
            # 中等难度阶段需要稍微严格一些
            if performance_std > 0.08:
                logger.info(f"中等难度阶段性能波动过大: std={performance_std:.4f}")
                return False
            
            # 要求至少有一半的评估达到良好水平
            good_performance_count = sum(1 for p in recent_performance if p >= 0.75)
            if good_performance_count < len(recent_performance) * 0.6:
                logger.info(f"中等难度阶段良好性能比例不足: {good_performance_count}/{len(recent_performance)}")
                return False
        
        # 🔧 基础阶段的渐进式要求
        elif current_stage <= 1:  # foundation和elementary阶段
            # 基础阶段相对宽松，但仍需要基本的稳定性
            if performance_std > 0.15:  # 允许更大的波动
                logger.info(f"基础阶段性能波动过大: std={performance_std:.4f}")
                return False
            
            # 基础阶段只要大部分评估达到基本水平即可
            acceptable_performance_count = sum(1 for p in recent_performance if p >= 0.5)
            if acceptable_performance_count < len(recent_performance) * 0.7:
                logger.info(f"基础阶段可接受性能比例不足: {acceptable_performance_count}/{len(recent_performance)}")
                return False
        
        # 标准进阶条件检查
        should_advance = avg_performance >= performance_threshold
        
        # 🔧 新增：进阶决策的详细日志
        current_stage_name = self.curriculum_stages[current_stage].name if current_stage < len(self.curriculum_stages) else "Unknown"
        
        logger.info(f"""
        📋 进阶检查详情 - 阶段{current_stage} ({current_stage_name}):
        ├─ 时间统计:
        │  ├─ 当前步数: {training_step}
        │  ├─ 阶段起始: {stage_start_step}
        │  ├─ 阶段内步数: {steps_in_current_stage}
        │  └─ 最小要求: {min_steps_required}
        ├─ 性能分析:
        │  ├─ 当前性能: {performance_estimate:.4f}
        │  ├─ 平均性能: {avg_performance:.4f} (阈值: {performance_threshold})
        │  ├─ 性能稳定性: {performance_std:.4f}
        │  └─ 最近趋势: {recent_performance[-3:] if len(recent_performance) >= 3 else recent_performance}
        ├─ 评估统计:
        │  ├─ 评估次数: {len(current_stage_history)}/{min_evaluations}
        │  ├─ 稳定性窗口: {len(recent_performance)}/{stability_window}
        │  └─ 历史长度: {len(self.stage_performance_history)}
        └─ 进阶决策: {'✅ 可以进阶' if should_advance else '❌ 继续当前阶段'}
        """)
        
        # 🔧 新增：如果决定进阶，记录详细信息
        if should_advance:
            logger.info(f"""
            🚀 阶段{current_stage}进阶条件满足:
            ├─ 训练充分: {steps_in_current_stage} ≥ {min_steps_required} 步
            ├─ 性能达标: {avg_performance:.4f} ≥ {performance_threshold}
            ├─ 表现稳定: std={performance_std:.4f}
            └─ 准备进入阶段{current_stage + 1}
            """)
        else:
            # 分析为什么不能进阶
            reasons = []
            if steps_in_current_stage < min_steps_required:
                reasons.append(f"训练时间不足({steps_in_current_stage}/{min_steps_required})")
            if avg_performance < performance_threshold:
                reasons.append(f"性能未达标({avg_performance:.4f}<{performance_threshold})")
            if performance_std > (0.05 if current_stage == 3 else 0.1):
                reasons.append(f"性能不稳定(std={performance_std:.4f})")
            
            logger.info(f"⏸️ 继续阶段{current_stage}训练，原因: {', '.join(reasons)}")
        
        return should_advance
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存。"""
        return {
            "current_stage": self.current_stage,  # 假设 current_stage 是阶段索引或可序列化标识
            "stage_performance_history": self.stage_performance_history,
            # 添加其他任何需要持久化的内部状态变量
            # 例如，如果你的 curriculum_stages 列表是动态生成的或修改的，也可能需要保存
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        """从字典加载课程学习管理器的状态。"""
        self.current_stage = state_dict.get("current_stage", 0) # 使用 get 提供默认值以防 key 不存在
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        # 加载其他已保存的状态变量
        logger.info(f"课程学习状态已加载。从阶段 {self.current_stage} (名称: {self.curriculum_stages[self.current_stage].name if self.current_stage < len(self.curriculum_stages) else '未知'}) 恢复。")
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存。"""
        return {
            "current_stage": self.current_stage,  # 假设 current_stage 是阶段索引或可序列化标识
            "stage_performance_history": self.stage_performance_history,
            # 添加其他任何需要持久化的内部状态变量
            # 例如，如果你的 curriculum_stages 列表是动态生成的或修改的，也可能需要保存
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        """从字典加载课程学习管理器的状态。"""
        self.current_stage = state_dict.get("current_stage", 0) # 使用 get 提供默认值以防 key 不存在
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        # 加载其他已保存的状态变量
        logger.info(f"课程学习状态已加载。从阶段 {self.current_stage} (名称: {self.curriculum_stages[self.current_stage].name if self.current_stage < len(self.curriculum_stages) else '未知'}) 恢复。")

    def _analyze_dataset_distribution(self):
        """分析数据集的等级和复杂度分布"""
        if len(self.full_dataset) == 0:
            logger.warning("Empty dataset provided to curriculum manager")
            return
        
        # 统计数据集等级分布
        level_counts = {}
        complexity_by_level = {}
        
        for example in self.full_dataset:
            level = example.get('level', 'unknown').lower()
            complexity = example.get('complexity_score', 5.0)
            
            # 统计等级分布
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # 统计每个等级的复杂度分布
            if level not in complexity_by_level:
                complexity_by_level[level] = []
            complexity_by_level[level].append(complexity)
        
        self.dataset_distribution = {
            'level_counts': level_counts,
            'complexity_by_level': complexity_by_level,
            'total_samples': len(self.full_dataset)
        }
        
        # 打印分布信息
        logger.info("Dataset Distribution Analysis:")
        logger.info(f"  Total samples: {self.dataset_distribution['total_samples']}")
        for level, count in level_counts.items():
            if level in complexity_by_level and complexity_by_level[level]:
                avg_complexity = np.mean(complexity_by_level[level])
                complexity_range = (np.min(complexity_by_level[level]), np.max(complexity_by_level[level]))
                logger.info(f"  {level.capitalize()}: {count} samples, "
                           f"avg complexity: {avg_complexity:.2f}, range: {complexity_range}")
    
    def _validate_curriculum_design(self):
        """验证课程设计的合理性"""
        # 检查所有数据集等级是否都被覆盖
        available_levels = set(self.dataset_distribution['level_counts'].keys())
        covered_levels = set()
        
        for stage in self.curriculum_stages:
            covered_levels.update([level.lower() for level in stage.dataset_levels])
        
        uncovered_levels = available_levels - covered_levels
        if uncovered_levels:
            logger.warning(f"Dataset levels not covered by curriculum: {uncovered_levels}")
        
        # 检查epoch比例总和
        total_ratio = sum(stage.epochs_ratio for stage in self.curriculum_stages)
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Curriculum epochs ratios sum to {total_ratio:.3f}, not 1.0")
    
    def get_current_stage_dataset(self) -> Dataset:
        """获取当前阶段的数据集"""
        if self.current_stage >= len(self.curriculum_stages):
            logger.info("Curriculum completed, using full dataset")
            return self.full_dataset
        
        stage = self.curriculum_stages[self.current_stage]
        
        # 双层过滤：数据集等级 + 复杂度范围
        filtered_indices = []
        level_filter_count = 0
        complexity_filter_count = 0
        
        for i, example in enumerate(self.full_dataset):
            # 第一层过滤：数据集等级
            example_level = example.get('level', 'unknown').lower()
            if example_level not in [level.lower() for level in stage.dataset_levels]:
                continue
            level_filter_count += 1
            
            # 第二层过滤：复杂度范围
            complexity = example.get('complexity_score', 5.0)
            min_complexity, max_complexity = stage.complexity_range
            if not (min_complexity <= complexity <= max_complexity):
                continue
            complexity_filter_count += 1
            
            filtered_indices.append(i)
        
        if not filtered_indices:
            logger.warning(f"No examples found for stage {self.current_stage} ({stage.name}), using full dataset")
            return self.full_dataset
        
        stage_dataset = self.full_dataset.select(filtered_indices)
        
        # 记录统计信息
        stage_stats = {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'total_examples': len(self.full_dataset),
            'level_filtered': level_filter_count,
            'complexity_filtered': complexity_filter_count,
            'final_selected': len(stage_dataset),
            'selection_ratio': len(stage_dataset) / len(self.full_dataset),
            'target_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range
        }
        self.stage_statistics.append(stage_stats)
        
        logger.info(f"Curriculum Stage {self.current_stage} ({stage.name}):")
        logger.info(f"  Target levels: {stage.dataset_levels}")
        logger.info(f"  Complexity range: {stage.complexity_range}")
        logger.info(f"  Selected examples: {len(stage_dataset)}/{len(self.full_dataset)} ({stage_stats['selection_ratio']:.1%})")
        logger.info(f"  Level filtering: {level_filter_count} examples passed")
        logger.info(f"  Complexity filtering: {complexity_filter_count} examples passed")
        
        return stage_dataset
    
    def should_advance_stage(self, recent_performance: float) -> bool:
        """判断是否应该进入下一阶段.

        Args:
            recent_performance: The latest performance metric (e.g., avg_test_pass_rate).
        """
        if self.current_stage >= len(self.curriculum_stages) - 1:
            logger.debug(f"Already at the final stage ({self.current_stage}). Cannot advance further.")
            return False
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(recent_performance) # Add current performance to history for this stage
        
        logger.debug(f"Stage {self.current_stage} ('{stage.name}'): Received performance {recent_performance:.4f}. "
                    f"History size: {len(self.stage_performance_history)}. Min evals required: {stage.min_evaluations}.")

        # Check if minimum number of evaluations for this stage has been met
        if len(self.stage_performance_history) < stage.min_evaluations:
            logger.info(f"Stage {self.current_stage} ('{stage.name}'): Not enough evaluations yet. "
                        f"Have {len(self.stage_performance_history)}, need {stage.min_evaluations}.")
            return False
        
        # If enough evaluations, consider the average of recent performance scores
        # Using all recorded performances for this stage for the average, or a sliding window if preferred.
        # For simplicity here, let's use all available history for the current stage.
        # A more sophisticated approach might use a decaying average or only the last N evaluations *after* min_evaluations is met.

        # Let's use the performance scores collected *after* min_evaluations were met, or just the most recent ones
        # if that's simpler. The current logic uses a window of the last 3, which is fine.

        # Consider the average of the performance history for this stage (or a recent window of it)
        # The history is reset when a stage advances.
        # Let's use the average of all recorded performances for the current stage
        # if len(self.stage_performance_history) >= stage.min_evaluations:
        # The previous logic of using a recent window seems fine.

        # Let's take the average of the performance metrics collected for this stage,
        # but only those collected *after* meeting the min_evaluations count, or a fixed window.
        # The existing logic takes min(3, len(history)) which means if min_evaluations is 5, it will
        # average the last 3 of those 5 (or more). This seems reasonable.

        recent_window_size = min(len(self.stage_performance_history), max(3, stage.min_evaluations)) # Ensure window is at least min_evals if history is long enough, or all history if shorter
        # Or, more simply, average all performances recorded for this stage so far, if count >= min_evaluations
        # Let's stick to a simpler interpretation: average of the last `stage.min_evaluations` (or all if fewer than that many *additional* evals)
        
        # The previous logic: "recent_window = min(3, len(self.stage_performance_history))"
        # This means it only looks at the last 3 evaluations *once min_evaluations condition is met*.
        # This is a common way to do it to ensure sustained performance.

        # Let's refine: average performance of the window that satisfies min_evaluations.
        # If min_evaluations is 10, we should average at least 10 evaluations.
        # The current history already includes the `recent_performance`.

        # We need to ensure we are looking at a stable performance, so averaging the last few
        # (e.g. 3, or up to `min_evaluations`) makes sense.

        # Let's use the average of the last `stage.min_evaluations` scores if available,
        # otherwise all scores if fewer than `stage.min_evaluations` have been recorded (but this case is handled by the check above).
        # If more than `stage.min_evaluations` are present, average the most recent `stage.min_evaluations` ones.
        num_scores_to_average = stage.min_evaluations

        # Ensure we only average available scores if history is shorter than num_scores_to_average (but longer than initial check)
        # This part is actually covered by `len(self.stage_performance_history) < stage.min_evaluations` check.
        # So, if we are here, len(self.stage_performance_history) >= stage.min_evaluations.

        # Average the most recent 'num_scores_to_average' performance scores.
        relevant_performances = self.stage_performance_history[-num_scores_to_average:]
        current_average_performance = np.mean(relevant_performances)

        logger.info(f"Stage {self.current_stage} ('{stage.name}'): Avg performance over last {len(relevant_performances)} evals: {current_average_performance:.4f}. "
                    f"Threshold: {stage.performance_threshold:.4f}.")

        should_advance = current_average_performance >= stage.performance_threshold
        
        if should_advance:
            logger.info(f"Stage {self.current_stage} ('{stage.name}') performance criteria MET. Advancing.")
        else:
            logger.info(f"Stage {self.current_stage} ('{stage.name}') performance criteria NOT YET MET.")

        return should_advance

    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            # 记录当前阶段的最终统计
            final_stats = {
                'completed_stage': self.current_stage,
                'stage_name': self.curriculum_stages[self.current_stage].name,
                'total_evaluations': len(self.stage_performance_history),
                'final_performance': self.stage_performance_history[-1] if self.stage_performance_history else 0,
                'average_performance': np.mean(self.stage_performance_history) if self.stage_performance_history else 0,
                'performance_history': self.stage_performance_history.copy()
            }
            
            self.current_stage += 1
            self.stage_performance_history = []  # 重置性能历史
            
            new_stage = self.curriculum_stages[self.current_stage]
            logger.info(f"🎯 Advanced to curriculum stage {self.current_stage}: {new_stage.name}")
            logger.info(f"   Previous stage stats: {final_stats['total_evaluations']} evaluations, "
                       f"final performance: {final_stats['final_performance']:.3f}")
            logger.info(f"   New stage targets: levels {new_stage.dataset_levels}, "
                       f"complexity {new_stage.complexity_range}")
            
            return True
        
        logger.info("🏆 Curriculum learning completed! Using full dataset.")
        return False
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        if self.current_stage >= len(self.curriculum_stages):
            return {
                'stage_index': self.current_stage,
                'stage_name': 'completed',
                'dataset_levels': 'all',
                'complexity_range': 'all',
                'is_completed': True
            }
        
        stage = self.curriculum_stages[self.current_stage]
        return {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'dataset_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range,
            'epochs_ratio': stage.epochs_ratio,
            'performance_threshold': stage.performance_threshold,
            'current_evaluations': len(self.stage_performance_history),
            'min_evaluations': stage.min_evaluations,
            'is_completed': False
        }
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """获取整体课程进度"""
        total_stages = len(self.curriculum_stages)
        progress_ratio = (self.current_stage + 1) / total_stages if total_stages > 0 else 1.0
        
        return {
            'current_stage': self.current_stage,
            'total_stages': total_stages,
            'progress_ratio': progress_ratio,
            'completed_stages': self.current_stage,
            'stage_statistics': self.stage_statistics,
            'dataset_distribution': self.dataset_distribution
        }
    
    def log_to_wandb(self, step: int):
        """记录到W&B"""
        if not hasattr(wandb, 'run') or wandb.run is None:
            return
        
        current_info = self.get_current_stage_info()
        progress_info = self.get_curriculum_progress()
        
        # 基础信息
        wandb.log({
            'curriculum/current_stage': current_info['stage_index'],
            'curriculum/stage_name': current_info['stage_name'],
            'curriculum/progress_ratio': progress_info['progress_ratio'],
            'curriculum/completed_stages': progress_info['completed_stages'],
            'curriculum/is_completed': current_info['is_completed']
        }, step=step)
        
        # 当前阶段详细信息
        if not current_info['is_completed']:
            wandb.log({
                'curriculum/current_evaluations': current_info['current_evaluations'],
                'curriculum/min_evaluations': current_info['min_evaluations'],
                'curriculum/performance_threshold': current_info['performance_threshold'],
                'curriculum/dataset_levels': str(current_info['dataset_levels']),
                'curriculum/complexity_range': str(current_info['complexity_range'])
            }, step=step)
        
        # 性能历史
        if self.stage_performance_history:
            wandb.log({
                'curriculum/stage_performance_mean': np.mean(self.stage_performance_history),
                'curriculum/stage_performance_latest': self.stage_performance_history[-1],
                'curriculum/stage_performance_trend': np.mean(self.stage_performance_history[-3:]) if len(self.stage_performance_history) >= 3 else self.stage_performance_history[-1]
            }, step=step)


def create_default_curriculum_stages() -> List[CurriculumStageConfig]:
    """创建默认的双层课程学习阶段"""
    stages = [
        CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=(0.0, 3.0),
            epochs_ratio=0.25,
            performance_threshold=0.7,
            min_evaluations=10, # Changed
            description="基础阶段：学习简单的基础级设计"
        ),
        CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=(0.0, 5.0),
            epochs_ratio=0.25,
            performance_threshold=0.65,
            min_evaluations=10, # Changed
            description="初级阶段：基础级+简单中级设计"
        ),
        CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=(3.0, 7.0),
            epochs_ratio=0.25,
            performance_threshold=0.6,
            min_evaluations=10, # Changed
            description="中级阶段：中等复杂度的中级设计"
        ),
        CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=(5.0, 9.0),
            epochs_ratio=0.15,
            performance_threshold=0.55,
            min_evaluations=10, # Changed
            description="高级阶段：复杂的中级和高级设计"
        ),
        CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=(7.0, 10.0),
            epochs_ratio=0.1,
            performance_threshold=0.5,
            min_evaluations=10, # Changed
            description="专家阶段：最复杂的高级和专家级设计"
        )
    ]
    return stages


def create_custom_curriculum_stages(
    dataset_distribution: Dict[str, Any],
    focus_levels: List[str] = None,
    complexity_emphasis: str = "balanced"  # "simple", "balanced", "complex"
) -> List[CurriculumStageConfig]:
    """根据数据集分布创建自定义课程阶段"""
    
    if focus_levels is None:
        focus_levels = ["basic", "intermediate", "advanced", "expert"]
    
    # 根据复杂度偏好调整复杂度范围
    if complexity_emphasis == "simple":
        complexity_ranges = [(0, 3), (0, 4), (2, 6), (4, 8), (6, 10)]
    elif complexity_emphasis == "complex":
        complexity_ranges = [(0, 4), (2, 6), (4, 8), (6, 10), (8, 10)]
    else:  # balanced
        complexity_ranges = [(0, 3), (0, 5), (3, 7), (5, 9), (7, 10)]
    
    stages = []
    
    # 基础阶段
    if "basic" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=complexity_ranges[0],
            epochs_ratio=0.3, # Will be normalized later
            performance_threshold=0.7,
            min_evaluations=10, # Added
            description="基础阶段：最简单的基础级设计"
        ))
    
    # 初级阶段
    if "basic" in focus_levels and "intermediate" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=complexity_ranges[1],
            epochs_ratio=0.25, # Will be normalized
            performance_threshold=0.65,
            min_evaluations=10, # Added
            description="初级阶段：基础到中级的过渡"
        ))
    
    # 中级阶段
    if "intermediate" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=complexity_ranges[2],
            epochs_ratio=0.25, # Will be normalized
            performance_threshold=0.6,
            min_evaluations=10, # Added
            description="中级阶段：中等复杂度设计"
        ))
    
    # 高级阶段
    if "advanced" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=complexity_ranges[3],
            epochs_ratio=0.15, # Will be normalized
            performance_threshold=0.55,
            min_evaluations=10, # Added
            description="高级阶段：复杂设计"
        ))
    
    # 专家阶段
    if "expert" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=complexity_ranges[4],
            epochs_ratio=0.05, # Will be normalized
            performance_threshold=0.5,
            min_evaluations=10, # Added
            description="专家阶段：最复杂设计"
        ))
    
    if not stages: # Ensure at least one stage if focus_levels is empty or misconfigured
        logger.warning("No custom stages generated based on focus_levels. Falling back to a single default stage.")
        stages.append(CurriculumStageConfig(
            name="default_full_range",
            dataset_levels=focus_levels if focus_levels else ["basic", "intermediate", "advanced", "expert"],
            complexity_range=(0.0, 10.0),
            epochs_ratio=1.0,
            performance_threshold=0.6, # Default threshold
            min_evaluations=10,        # Default min_evaluations
            description="Default stage covering all specified levels and full complexity range."
        ))

    # 标准化epoch比例
    total_ratio = sum(stage.epochs_ratio for stage in stages)
    for stage in stages:
        stage.epochs_ratio /= total_ratio
    
    return stages

class DynamicDifficultyAdjuster:
    """动态调整当前阶段的难度子集"""
    
    def __init__(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager
        self.difficulty_adjustment_history = []
    
    def adjust_current_stage_difficulty(self, performance_metrics):
        """根据性能动态调整当前阶段的数据子集"""
        current_stage = self.curriculum_manager.current_stage
        
        if current_stage < 3:  # 只在非最高阶段调整
            return
            
        # 获取当前性能
        recent_avg_performance = np.mean([
            m['performance'] for m in performance_metrics[-10:]  # 最近10次评估
        ]) if len(performance_metrics) >= 10 else 0.5
        
        # 动态调整策略
        if recent_avg_performance > 0.95:  # 性能很好，增加难度
            self._increase_difficulty()
        elif recent_avg_performance < 0.8:  # 性能不佳，稍微降低难度
            self._decrease_difficulty()
    
    def _increase_difficulty(self):
        """增加当前阶段的难度"""
        current_stage_obj = self.curriculum_manager.curriculum_stages[self.curriculum_manager.current_stage]
        
        # 增加复杂度范围的上限
        if current_stage_obj.complexity_range[1] < 10:
            new_max_complexity = min(10, current_stage_obj.complexity_range[1] + 1)
            current_stage_obj.complexity_range = (current_stage_obj.complexity_range[0], new_max_complexity)
            logger.info(f"🔥 增加难度: 复杂度范围调整为 {current_stage_obj.complexity_range}")
    
    def _decrease_difficulty(self):
        """稍微降低当前阶段的难度"""
        current_stage_obj = self.curriculum_manager.curriculum_stages[self.curriculum_manager.current_stage]
        
        # 降低复杂度范围的下限
        if current_stage_obj.complexity_range[0] > 0:
            new_min_complexity = max(0, current_stage_obj.complexity_range[0] - 1)
            current_stage_obj.complexity_range = (new_min_complexity, current_stage_obj.complexity_range[1])
            logger.info(f"🔻 降低难度: 复杂度范围调整为 {current_stage_obj.complexity_range}")
