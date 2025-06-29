import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from datetime import datetime
import json
import os
try:
    from grpo_project.configs import ScriptConfig
    from .stages import CurriculumStageConfig, create_default_curriculum_stages, create_custom_curriculum_stages
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Could not import from grpo_project.configs or .stages. Using placeholders for ScriptConfig/CurriculumStageConfig.")
    class ScriptConfig: pass # type: ignore
    class CurriculumStageConfig: pass # type: ignore
    def create_default_curriculum_stages() -> List[Any]: return []
    def create_custom_curriculum_stages(*args, **kwargs) -> List[Any]: return []


logger = logging.getLogger(__name__)

class EnhancedCurriculumManager:
    """增强的双层课程学习管理器 (from enhanced_curriculum.py)"""

    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        self.stage_progression_configs = {
            0: {"performance_threshold": 0.75, "min_evaluations": 10, "stability_window": 3},
            1: {"performance_threshold": 0.70, "min_evaluations": 20, "stability_window": 3},
            2: {"performance_threshold": 0.65, "min_evaluations": 20, "stability_window": 3},
            3: {"performance_threshold": 0.60, "min_evaluations": 20, "stability_window": 2, "max_stay_steps": 200}
        }
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history: List[Dict[str, Any]] = [] # More specific type
        self.stage_statistics: List[Dict[str, Any]] = []
        self.stage_start_steps: Dict[int, int] = {} # To track steps per stage

        self._analyze_dataset_distribution()
        self._validate_curriculum_design()

        logger.info(f"EnhancedCurriculumManager initialized with {len(curriculum_stages)} stages")
        # ... (rest of init logging)

    def should_advance_to_next_stage(self, current_loss: float, training_step: int) -> bool:
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager.should_advance_to_next_stage)
        # This method uses self.stage_progression_configs, self.curriculum_stages, self.current_stage,
        # self.stage_performance_history, self.stage_start_steps
        # For brevity, the full logic is not pasted here again but should be moved.
        logger.debug(f"Placeholder should_advance_to_next_stage called with loss {current_loss} at step {training_step}")
        if self.current_stage >= len(self.curriculum_stages) -1: return False
        # Simplified logic for stub:
        if len(self.stage_performance_history) > self.curriculum_stages[self.current_stage].min_evaluations:
            if np.mean([p['performance'] for p in self.stage_performance_history[-3:]]) > self.curriculum_stages[self.current_stage].performance_threshold:
                return True
        return False


    def get_curriculum_state(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "current_round": self.current_round,
            "completed_rounds": self.completed_rounds,
            "stage_performance_history": self.stage_performance_history,
            "all_stage_history": self.all_stage_history,
            "round_history": self.round_history,
            "stage_statistics": self.stage_statistics,
            "threshold_multiplier": self.threshold_multiplier,
            "advancement_attempts": self.advancement_attempts,
            "successful_advancements": self.successful_advancements,
            "total_advancement_checks": self.total_advancement_checks
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        self.current_stage = state_dict.get("current_stage", 0)
        self.current_round = state_dict.get("current_round", 1)
        self.completed_rounds = state_dict.get("completed_rounds", 0)
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        self.all_stage_history = state_dict.get("all_stage_history", [])
        self.round_history = state_dict.get("round_history", [])
        self.stage_statistics = state_dict.get("stage_statistics", [])
        self.threshold_multiplier = state_dict.get("threshold_multiplier", 1.0)
        self.advancement_attempts = state_dict.get("advancement_attempts", 0)
        self.successful_advancements = state_dict.get("successful_advancements", 0)
        self.total_advancement_checks = state_dict.get("total_advancement_checks", 0)
        
        self._log_debug(f"📚 FixedEnhancedCurriculumManager 状态已恢复")
        self._log_debug(f"  - 当前阶段: {self.current_stage}")
        self._log_debug(f"  - 当前轮次: {self.current_round}")
        self._log_debug(f"  - 已完成轮次: {self.completed_rounds}")
        self._log_debug(f"  - 进阶统计: {self.successful_advancements}/{self.advancement_attempts}")
        
        logger.info(f"FixedEnhancedCurriculumManager state loaded. Current stage: {self.current_stage}, Round: {self.current_round}")

    def get_current_stage_name(self) -> str:
        """Get the name of the current curriculum stage"""
        if self.current_stage >= len(self.curriculum_stages):
            return "completed"
        return self.curriculum_stages[self.current_stage].name


    def advance_stage(self) -> bool:
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager.advance_stage)
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_performance_history = []
            logger.info(f"Advanced to curriculum stage {self.current_stage}")
            return True
        return False


# Moved from curriculum_debug_config.py
class FixedEnhancedCurriculumManager:
    """修复版本的增强课程学习管理器 - 增强调试日志 + 循环训练功能"""
    
    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history = []
        self.all_stage_history = []
        self.stage_statistics = []
        self.debug_log = []
        
        # 🔧 增强：添加更多调试信息
        self.last_advancement_check_step = 0
        self.total_advancement_checks = 0
        self.advancement_attempts = 0
        self.successful_advancements = 0
        
        # 🔄 新增：循环训练相关变量
        self.current_round = 1  # 当前是第几轮训练
        self.max_rounds = 5     # 最大轮次数 (可配置)
        self.completed_rounds = 0  # 完成的轮次数
        self.round_history = []    # 每轮的完成历史
        self.threshold_multiplier = 1.0  # 阈值倍数，每轮递增
        self.threshold_increment = 0.1   # 每轮阈值增加量
        
        # 🔧 新增：完整epoch训练跟踪
        self.stage_training_tracker = {}  # 跟踪每个阶段的训练进度
        self.current_stage_start_step = 0  # 当前阶段开始的步数
        self.stage_dataset_size = 0  # 当前阶段数据集大小
        self.stage_steps_completed = 0  # 当前阶段已完成步数
        self.stage_epochs_completed = 0  # 当前阶段已完成的epoch数
        
        self._log_debug("🚀 FixedEnhancedCurriculumManager 开始初始化 (支持循环训练+完整epoch要求)")
        self._log_debug(f"📊 课程配置: 总阶段数={len(curriculum_stages)}, 数据集大小={len(dataset)}")
        self._log_debug(f"🔄 循环训练配置: 最大轮次={self.max_rounds}, 阈值递增={self.threshold_increment}")
        
        # 详细记录每个阶段的配置
        for i, stage in enumerate(curriculum_stages):
            self._log_debug(f"  阶段{i} ({stage.name}):")
            self._log_debug(f"    - 等级: {stage.dataset_levels}")
            self._log_debug(f"    - 复杂度: {stage.complexity_range}")
            self._log_debug(f"    - 基础性能阈值: {stage.performance_threshold}")
            self._log_debug(f"    - 最小评估: {stage.min_evaluations}")
            self._log_debug(f"    - 要求完整epoch: {getattr(stage, 'require_full_epoch', False)}")
            self._log_debug(f"    - 最小步数/epoch: {getattr(stage, 'min_steps_per_epoch', 10)}")
        
        # Analyze dataset distribution using the static method
        self.dataset_distribution = FixedEnhancedCurriculumManager._calculate_dataset_distribution(self.full_dataset, self._log_debug)
        self._log_detailed_distribution() # New method to keep __init__ cleaner

        self._validate_curriculum_design()
        
        # 验证当前阶段数据集
        current_dataset = self.get_current_stage_dataset()
        self._log_debug(f"✅ 初始化完成: 当前阶段数据集大小={len(current_dataset)}")
        self._log_debug(f"🔄 准备开始第{self.current_round}轮训练")
        
        # 🔧 初始化当前阶段的训练跟踪
        self._initialize_stage_tracker()

    def _validate_curriculum_design(self):
        """验证课程设计的合理性 - 增强版：检查数据集覆盖率"""
        available_levels = set(self.dataset_distribution['level_counts'].keys())
        covered_levels = set()
        
        for stage in self.curriculum_stages:
            covered_levels.update([level.lower() for level in stage.dataset_levels])
        
        uncovered_levels = available_levels - covered_levels
        if uncovered_levels:
            self._log_debug(f"⚠️ 未覆盖的数据集等级: {uncovered_levels}")
            self._log_debug("🔧 建议：添加comprehensive阶段或修改现有阶段配置")
        
        total_ratio = sum(stage.epochs_ratio for stage in self.curriculum_stages)
        # Ensure epochs_ratio exists and is a float, provide default if not
        total_ratio = sum(getattr(stage, 'epochs_ratio', 0.0) for stage in self.curriculum_stages)

        if abs(total_ratio - 1.0) > 0.01: # Assuming a default of 0.0 if attribute missing
            self._log_debug(f"⚠️ Epoch比例总和: {total_ratio:.3f} (应该接近1.0)")
        
        # 🔧 新增：详细的数据集覆盖率分析
        self._analyze_dataset_coverage()

    def _analyze_dataset_coverage(self):
        """分析数据集覆盖率"""
        self._log_debug("🔍 开始数据集覆盖率分析...")
        
        total_samples = len(self.full_dataset)
        covered_samples = set()
        stage_coverage = {}
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            stage_indices = []
            
            for i, example in enumerate(self.full_dataset):
                # 检查是否符合当前阶段条件
                example_level = example.get('level', 'unknown').lower()
                complexity = example.get('complexity_score', 5.0)
                
                # 等级匹配
                if example_level in [level.lower() for level in stage.dataset_levels]:
                    # 复杂度匹配
                    min_complexity, max_complexity = stage.complexity_range
                    if min_complexity <= complexity <= max_complexity:
                        stage_indices.append(i)
                        covered_samples.add(i)
            
            stage_coverage[stage.name] = {
                'indices': stage_indices,
                'count': len(stage_indices),
                'ratio': len(stage_indices) / total_samples if total_samples > 0 else 0
            }
            
            self._log_debug(f"  阶段 {stage.name}: {len(stage_indices)} 样本 ({len(stage_indices)/total_samples*100:.1f}%)")
        
        # 总覆盖率统计
        total_covered = len(covered_samples)
        uncovered_count = total_samples - total_covered
        coverage_ratio = total_covered / total_samples if total_samples > 0 else 0
        
        self._log_debug(f"📊 总体覆盖率分析:")
        self._log_debug(f"  - 总样本数: {total_samples}")
        self._log_debug(f"  - 已覆盖样本: {total_covered} ({coverage_ratio*100:.1f}%)")
        self._log_debug(f"  - 未覆盖样本: {uncovered_count} ({(1-coverage_ratio)*100:.1f}%)")
        
        if uncovered_count > 0:
            self._log_debug(f"⚠️ 警告：{uncovered_count} 个样本未被任何阶段覆盖!")
            self._analyze_uncovered_samples(covered_samples)
        else:
            self._log_debug("✅ 所有样本都被至少一个阶段覆盖")
        
        # 保存覆盖率信息
        self.coverage_analysis = {
            'total_samples': total_samples,
            'covered_samples': total_covered,
            'coverage_ratio': coverage_ratio,
            'stage_coverage': stage_coverage,
            'uncovered_count': uncovered_count
        }

    def _analyze_uncovered_samples(self, covered_samples: set):
        """分析未覆盖的样本"""
        uncovered_indices = []
        uncovered_levels = {}
        uncovered_complexities = []
        
        for i, example in enumerate(self.full_dataset):
            if i not in covered_samples:
                uncovered_indices.append(i)
                
                level = example.get('level', 'unknown').lower()
                complexity = example.get('complexity_score', 5.0)
                
                uncovered_levels[level] = uncovered_levels.get(level, 0) + 1
                uncovered_complexities.append(complexity)
        
        self._log_debug(f"🔍 未覆盖样本分析:")
        self._log_debug(f"  - 按等级分布:")
        for level, count in uncovered_levels.items():
            self._log_debug(f"    {level}: {count} 样本")
        
        if uncovered_complexities:
            import numpy as np
            self._log_debug(f"  - 复杂度分布:")
            self._log_debug(f"    最小: {np.min(uncovered_complexities):.2f}")
            self._log_debug(f"    最大: {np.max(uncovered_complexities):.2f}")
            self._log_debug(f"    平均: {np.mean(uncovered_complexities):.2f}")
            self._log_debug(f"    中位数: {np.median(uncovered_complexities):.2f}")
        
        # 保存未覆盖样本信息
        self.uncovered_analysis = {
            'indices': uncovered_indices,
            'levels': uncovered_levels,
            'complexities': uncovered_complexities
        }

    def _log_detailed_distribution(self):
        """Logs the detailed dataset distribution after analysis."""
        if not self.dataset_distribution or not self.dataset_distribution.get('total_samples'):
            self._log_debug("Dataset distribution is empty or invalid after calculation.")
            return

        self._log_debug(f"数据集分布分析 - 总样本: {self.dataset_distribution['total_samples']}")
        for level, count in self.dataset_distribution['level_counts'].items():
            if level in self.dataset_distribution['complexity_by_level'] and self.dataset_distribution['complexity_by_level'][level]:
                avg_complexity = np.mean(self.dataset_distribution['complexity_by_level'][level])
                complexity_range_actual = (np.min(self.dataset_distribution['complexity_by_level'][level]), np.max(self.dataset_distribution['complexity_by_level'][level]))
                self._log_debug(f"  {level}: {count}样本, 平均复杂度: {avg_complexity:.2f}, 范围: {complexity_range_actual}")
            else:
                self._log_debug(f"  {level}: {count}样本, 无复杂度信息")


    def _log_debug(self, message: str):
        """记录调试信息 - 增强版本"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_entry = f"[{timestamp}] {message}"
        self.debug_log.append(debug_entry)
        
        # 🔧 确保调试信息输出到正确的 logger
        logger.info(f"📚 CURRICULUM (Fixed): {message}")
        
        # 🔧 额外：每100条调试日志输出一次统计
        if len(self.debug_log) % 100 == 0:
            logger.info(f"📊 课程调试统计: {len(self.debug_log)} 条日志, 当前阶段={self.current_stage}, 当前轮次={self.current_round}")

    def get_current_threshold(self, stage_index: int = None) -> float:
        """获取当前轮次的有效性能阈值"""
        if stage_index is None:
            stage_index = self.current_stage
            
        if stage_index >= len(self.curriculum_stages):
            return 0.9  # 默认高阈值
            
        base_threshold = self.curriculum_stages[stage_index].performance_threshold
        # 第一轮使用原始阈值，后续轮次递增
        current_threshold = base_threshold + (self.current_round - 1) * self.threshold_increment
        
        # 确保阈值不超过0.95（避免过于苛刻）
        return min(current_threshold, 0.95)
    
    def start_new_round(self):
        """开始新一轮训练"""
        self.completed_rounds += 1
        
        # 记录上一轮的完整信息
        round_summary = {
            'round_number': self.current_round,
            'completed_stages': len(self.all_stage_history),
            'total_evaluations': sum(len(h['performance_history']) for h in self.all_stage_history),
            'completion_timestamp': datetime.now().isoformat(),
            'stage_history': self.all_stage_history.copy()
        }
        self.round_history.append(round_summary)
        
        # 开始新轮次
        self.current_round += 1
        self.current_stage = 0
        self.stage_performance_history = []
        
        self._log_debug(f"🔄 完成第{self.completed_rounds}轮训练，开始第{self.current_round}轮")
        self._log_debug(f"📈 新轮次阈值提升: 基础阈值 + {(self.current_round - 1) * self.threshold_increment:.2f}")
        
        # 记录新轮次的阈值情况
        for i, stage in enumerate(self.curriculum_stages):
            new_threshold = self.get_current_threshold(i)
            self._log_debug(f"  阶段{i} ({stage.name}): {stage.performance_threshold:.2f} -> {new_threshold:.2f}")
    
    def should_continue_curriculum(self) -> bool:
        """判断是否应该继续课程学习（未达到最大轮次）"""
        return self.current_round <= self.max_rounds

    @staticmethod
    def _calculate_dataset_distribution(dataset: Dataset, log_fn=None) -> Dict[str, Any]:
        """
        Calculates and returns the dataset distribution.
        Can be used statically. If log_fn is provided, it will log messages.
        """
        if len(dataset) == 0:
            if log_fn: log_fn("❌ _calculate_dataset_distribution: Empty dataset provided.")
            return {'level_counts': {}, 'complexity_by_level': {}, 'total_samples': 0}
        
        level_counts = {}
        complexity_by_level = {}
        
        for example in dataset:
            level = example.get('level', 'unknown').lower()
            # Ensuring complexity_score is float, default to 5.0 if missing or not convertible
            try:
                complexity = float(example.get('complexity_score', 5.0))
            except (ValueError, TypeError):
                complexity = 5.0

            level_counts[level] = level_counts.get(level, 0) + 1
            
            if level not in complexity_by_level:
                complexity_by_level[level] = []
            complexity_by_level[level].append(complexity)
        
        # The original _analyze_dataset_distribution logged details.
        # This static version returns the raw data. Logging can be done by the caller.

        return {
            'level_counts': level_counts,
            'complexity_by_level': complexity_by_level, # raw lists of complexities per level
            'total_samples': len(dataset)
        }

    def should_advance_stage(self, recent_performance: float, current_step: int = None) -> bool:
        """判断是否应该进入下一阶段 - 增强调试版本 + 循环训练支持 + 完整epoch要求"""
        self.total_advancement_checks += 1
        
        # 🔧 更新训练进度
        if current_step is not None:
            self.update_training_progress(current_step)
        
        self._log_debug(f"🔍 第{self.total_advancement_checks}次进阶检查 (轮次{self.current_round})")
        self._log_debug(f"  - 当前性能: {recent_performance:.4f}")
        self._log_debug(f"  - 当前阶段: {self.current_stage}")
        self._log_debug(f"  - 历史长度: {len(self.stage_performance_history)}")
        self._log_debug(f"  - 历史内容: {[f'{p:.4f}' for p in self.stage_performance_history[-5:]]}")  # 显示最近5次
        
        # 🔄 检查是否到达最后阶段 - 但不直接返回False，而是考虑循环
        if self.current_stage >= len(self.curriculum_stages) - 1:
            self._log_debug("📍 已在最后阶段，检查是否应该开始新轮次")
            # 如果还有剩余轮次，将在advance_stage中处理循环
            # 这里先让正常的阈值检查决定是否"进阶"到新轮次
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(recent_performance)
        
        # 🔧 使用动态阈值
        current_threshold = self.get_current_threshold()
        base_threshold = stage.performance_threshold
        
        self._log_debug(f"  - 性能已记录，新历史长度: {len(self.stage_performance_history)}")
        self._log_debug(f"  - 阶段配置: {stage.name}, 基础阈值={base_threshold:.3f}, 当前阈值={current_threshold:.3f}")
        self._log_debug(f"  - 阈值提升: +{current_threshold - base_threshold:.3f} (轮次{self.current_round})")
        
        # 🔧 新增：检查完整epoch训练要求
        is_training_complete = self.is_stage_training_complete()
        training_status = self.get_stage_training_status()
        
        self._log_debug(f"📊 完整训练检查:")
        self._log_debug(f"  - 要求完整epoch: {training_status.get('require_full_epoch', False)}")
        self._log_debug(f"  - 训练进度: {training_status.get('progress_percent', 0):.1f}%")
        self._log_debug(f"  - 已完成epoch: {training_status.get('epochs_completed', 0):.2f}")
        self._log_debug(f"  - 已完成步数: {training_status.get('steps_completed', 0)}")
        self._log_debug(f"  - 完整训练要求满足: {is_training_complete}")
        
        # 需要足够的评估次数
        if len(self.stage_performance_history) < stage.min_evaluations:
            self._log_debug(f"❌ 评估次数不足: {len(self.stage_performance_history)}/{stage.min_evaluations}")
            return False
        
        # 🔧 新增：检查完整训练要求
        if not is_training_complete:
            self._log_debug(f"❌ 完整训练要求未满足")
            self._log_debug(f"  - 需要完成至少1个完整epoch的训练")
            self._log_debug(f"  - 当前进度: {training_status.get('epochs_completed', 0):.2f}/1.0 epoch")
            return False
        
        # 检查最近的性能表现
        recent_window = min(2, len(self.stage_performance_history))
        recent_performances = self.stage_performance_history[-recent_window:]
        recent_avg = np.mean(recent_performances)
        
        self._log_debug(f"  - 最近{recent_window}次性能: {recent_performances}")
        self._log_debug(f"  - 最近平均性能: {recent_avg:.4f}")
        self._log_debug(f"  - 当前轮次阈值: {current_threshold:.4f}")
        
        should_advance = recent_avg >= current_threshold
        
        # 🔧 详细记录决策过程
        if should_advance:
            self._log_debug(f"✅ 满足进阶条件!")
            self._log_debug(f"  - 性能检查: {recent_avg:.4f} >= {current_threshold:.4f} ✅")
            self._log_debug(f"  - 评估检查: {len(self.stage_performance_history)} >= {stage.min_evaluations} ✅")
            self._log_debug(f"  - 完整训练检查: {is_training_complete} ✅")
            if self.current_stage >= len(self.curriculum_stages) - 1:
                if self.should_continue_curriculum():
                    self._log_debug(f"  - 🔄 将触发新轮次 (当前第{self.current_round}轮)")
                else:
                    self._log_debug(f"  - 🏁 所有轮次已完成 (共{self.max_rounds}轮)")
        else:
            self._log_debug(f"❌ 不满足进阶条件")
            if recent_avg < current_threshold:
                improvement_needed = current_threshold - recent_avg
                self._log_debug(f"  - 性能不足: {recent_avg:.4f} < {current_threshold:.4f} (需提升{improvement_needed:.4f})")
            
        self._log_debug(f"  - 进阶决策: {should_advance}")
        return should_advance

    def advance_stage(self) -> bool:
        """进入下一阶段 - 增强调试版本 + 循环训练支持 + 完整epoch跟踪"""
        self.advancement_attempts += 1
        
        self._log_debug(f"🎯 第{self.advancement_attempts}次进阶尝试 (轮次{self.current_round})")
        
        # 记录当前阶段的最终统计
        old_stage = self.current_stage
        old_stage_name = self.curriculum_stages[old_stage].name if old_stage < len(self.curriculum_stages) else "Final"
        
        # 🔧 详细记录进阶前的状态
        self._log_debug(f"📊 进阶前状态统计:")
        self._log_debug(f"  - 离开阶段: {old_stage} ({old_stage_name})")
        self._log_debug(f"  - 该阶段评估次数: {len(self.stage_performance_history)}")
        
        # 🔧 记录完整训练状态
        training_status = self.get_stage_training_status()
        if training_status and training_status.get('status') != 'no_tracker':
            self._log_debug(f"  - 完整训练状态:")
            self._log_debug(f"    已完成epoch: {training_status.get('epochs_completed', 0):.2f}")
            self._log_debug(f"    已完成步数: {training_status.get('steps_completed', 0)}")
            self._log_debug(f"    训练进度: {training_status.get('progress_percent', 0):.1f}%")
            self._log_debug(f"    epoch要求满足: {training_status.get('is_epoch_requirement_met', False)}")
        
        if self.stage_performance_history:
            final_performance = self.stage_performance_history[-1]
            avg_performance = np.mean(self.stage_performance_history)
            best_performance = np.max(self.stage_performance_history)
            worst_performance = np.min(self.stage_performance_history)
            
            self._log_debug(f"  - 最终性能: {final_performance:.4f}")
            self._log_debug(f"  - 平均性能: {avg_performance:.4f}")
            self._log_debug(f"  - 最佳性能: {best_performance:.4f}")
            self._log_debug(f"  - 最差性能: {worst_performance:.4f}")
        
        final_stats = {
            'completed_stage': old_stage,
            'stage_name': old_stage_name,
            'round_number': self.current_round,
            'total_evaluations': len(self.stage_performance_history),
            'final_performance': self.stage_performance_history[-1] if self.stage_performance_history else 0,
            'average_performance': np.mean(self.stage_performance_history) if self.stage_performance_history else 0,
            'performance_history': self.stage_performance_history.copy(),
            'completion_timestamp': datetime.now().isoformat(),
            'threshold_used': self.get_current_threshold(old_stage),
            'training_status': training_status  # 🔧 新增：保存训练状态
        }
        
        # 保存到全部历史
        self.all_stage_history.append(final_stats)
        
        # 🔄 关键修改：处理阶段进阶 vs 轮次循环
        if self.current_stage >= len(self.curriculum_stages) - 1:
            # 在最后一个阶段，检查是否开始新轮次
            if self.should_continue_curriculum():
                # 开始新轮次
                self._log_debug(f"🔄 完成第{self.current_round}轮最后阶段，开始新轮次")
                self.start_new_round()
                new_stage_name = self.curriculum_stages[0].name
                self.successful_advancements += 1
                
                self._log_debug(f"🔄 轮次循环成功!")
                self._log_debug(f"  - 轮次路径: 第{self.current_round-1}轮阶段{old_stage} -> 第{self.current_round}轮阶段0")
                self._log_debug(f"  - 成功进阶次数: {self.successful_advancements}/{self.advancement_attempts}")
                
                # 显示新轮次的阈值情况
                new_threshold = self.get_current_threshold(0)
                base_threshold = self.curriculum_stages[0].performance_threshold
                self._log_debug(f"📈 新轮次第一阶段:")
                self._log_debug(f"  - 阶段名称: {new_stage_name}")
                self._log_debug(f"  - 基础阈值: {base_threshold:.3f}")
                self._log_debug(f"  - 新轮次阈值: {new_threshold:.3f} (+{new_threshold-base_threshold:.3f})")
                
                # 🔧 重新初始化训练跟踪器
                self.current_stage_start_step = 0  # 这需要从外部更新
                self._initialize_stage_tracker()
                
                return True
            else:
                # 所有轮次已完成
                self._log_debug(f"🏁 所有{self.max_rounds}轮训练已完成，课程学习结束")
                return False
        else:
            # 正常阶段进阶
            self.current_stage += 1
            self.stage_performance_history = []  # 重置性能历史
            
            new_stage_name = self.curriculum_stages[self.current_stage].name
            self.successful_advancements += 1
            
            self._log_debug(f"🎉 成功进阶!")
            self._log_debug(f"  - 进阶路径: 第{self.current_round}轮阶段{old_stage}({old_stage_name}) -> 第{self.current_round}轮阶段{self.current_stage}({new_stage_name})")
            self._log_debug(f"  - 成功进阶次数: {self.successful_advancements}/{self.advancement_attempts}")
            self._log_debug(f"  - 前阶段最终性能: {final_stats['final_performance']:.4f}")
            
            # 🔧 详细记录新阶段信息
            new_stage = self.curriculum_stages[self.current_stage]
            new_dataset = self.get_current_stage_dataset()
            new_threshold = self.get_current_threshold()
            
            self._log_debug(f"📈 新阶段详情:")
            self._log_debug(f"  - 阶段名称: {new_stage.name}")
            self._log_debug(f"  - 目标等级: {new_stage.dataset_levels}")
            self._log_debug(f"  - 复杂度范围: {new_stage.complexity_range}")
            self._log_debug(f"  - 基础阈值: {new_stage.performance_threshold:.3f}")
            self._log_debug(f"  - 当前轮次阈值: {new_threshold:.3f}")
            self._log_debug(f"  - 数据集大小: {len(new_dataset)}")
            self._log_debug(f"  - 数据集比例: {len(new_dataset)/len(self.full_dataset)*100:.1f}%")
            self._log_debug(f"  - 要求完整epoch: {getattr(new_stage, 'require_full_epoch', True)}")
            
            # 🔧 重新初始化训练跟踪器
            self.current_stage_start_step = 0  # 这需要从外部更新
            self._initialize_stage_tracker()
            
            return True

    def get_current_stage_dataset(self) -> Dataset:
        """获取当前阶段的数据集 - 增强调试版本"""
        if self.current_stage >= len(self.curriculum_stages):
            self._log_debug("✅ 课程学习完成，使用全部数据集")
            return self.full_dataset
        
        stage = self.curriculum_stages[self.current_stage]
        
        # 🔧 详细的过滤过程日志
        self._log_debug(f"🔍 开始过滤阶段{self.current_stage}数据集")
        self._log_debug(f"  - 阶段名称: {stage.name}")
        self._log_debug(f"  - 目标等级: {stage.dataset_levels}")
        self._log_debug(f"  - 复杂度范围: {stage.complexity_range}")
        self._log_debug(f"  - 原始数据集大小: {len(self.full_dataset)}")
        
        # 双层过滤：数据集等级 + 复杂度范围
        filtered_indices = []
        level_filter_count = 0
        complexity_filter_count = 0
        
        # 🔧 按等级分类统计
        level_stats = {}
        complexity_stats = {}
        
        for i, example in enumerate(self.full_dataset):
            # 第一层过滤：数据集等级
            example_level = example.get('level', 'unknown').lower()
            level_stats[example_level] = level_stats.get(example_level, 0) + 1
            
            if example_level not in [level.lower() for level in stage.dataset_levels]:
                continue
            level_filter_count += 1
            
            # 第二层过滤：复杂度范围
            complexity = example.get('complexity_score', 5.0)
            complexity_key = f"{int(complexity)}-{int(complexity)+1}"
            complexity_stats[complexity_key] = complexity_stats.get(complexity_key, 0) + 1
            
            min_complexity, max_complexity = stage.complexity_range
            if not (min_complexity <= complexity <= max_complexity):
                continue
            complexity_filter_count += 1
            
            filtered_indices.append(i)
        
        # 🔧 详细的过滤统计日志
        self._log_debug(f"📊 过滤统计:")
        self._log_debug(f"  - 原始数据: {len(self.full_dataset)} 样本")
        self._log_debug(f"  - 等级过滤通过: {level_filter_count} 样本")
        self._log_debug(f"  - 复杂度过滤通过: {complexity_filter_count} 样本")
        self._log_debug(f"  - 最终选择: {len(filtered_indices)} 样本")
        
        # 🔧 等级分布统计
        self._log_debug(f"📈 数据集等级分布:")
        for level, count in sorted(level_stats.items()):
            is_target = level in [l.lower() for l in stage.dataset_levels]
            marker = "✅" if is_target else "❌"
            self._log_debug(f"    {marker} {level}: {count} 样本 ({count/len(self.full_dataset)*100:.1f}%)")
        
        # 🔧 复杂度分布统计
        self._log_debug(f"📈 复杂度分布统计:")
        min_c, max_c = stage.complexity_range
        for comp_range, count in sorted(complexity_stats.items()):
            range_start = int(comp_range.split('-')[0])
            is_in_range = min_c <= range_start <= max_c
            marker = "✅" if is_in_range else "❌"
            self._log_debug(f"    {marker} {comp_range}: {count} 样本")
        
        if not filtered_indices:
            self._log_debug(f"❌ 阶段{self.current_stage}没有匹配的样本，使用全部数据集")
            self._log_debug("⚠️ 这可能表明课程配置有问题!")
            return self.full_dataset
        
        stage_dataset = self.full_dataset.select(filtered_indices)
        
        # 记录详细统计
        stage_stats = {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'total_examples': len(self.full_dataset),
            'level_filtered': level_filter_count,
            'complexity_filtered': complexity_filter_count,
            'final_selected': len(stage_dataset),
            'selection_ratio': len(stage_dataset) / len(self.full_dataset),
            'target_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range,
            'level_distribution': level_stats,
            'complexity_distribution': complexity_stats
        }
        self.stage_statistics.append(stage_stats)
        
        self._log_debug(f"✅ 阶段{self.current_stage}数据集过滤完成")
        self._log_debug(f"  - 选择比例: {stage_stats['selection_ratio']:.1%}")
        self._log_debug(f"  - 过滤效率: 等级{level_filter_count/len(self.full_dataset)*100:.1f}% -> 复杂度{complexity_filter_count/level_filter_count*100:.1f}%")
        
        return stage_dataset

    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息 - 增强调试版本"""
        if self.current_stage >= len(self.curriculum_stages):
            return {
                'stage_index': self.current_stage,
                'stage_name': 'completed',
                'dataset_levels': 'all',
                'complexity_range': 'all',
                'is_completed': True,
                'debug_log_recent': self.debug_log[-10:],
                'advancement_stats': {
                    'total_checks': self.total_advancement_checks,
                    'attempts': self.advancement_attempts,
                    'successful': self.successful_advancements
                }
            }
        
        stage = self.curriculum_stages[self.current_stage]
        current_dataset = self.get_current_stage_dataset()
        
        return {
            'stage_index': self.current_stage,
            'stage_name': stage.name,
            'dataset_levels': stage.dataset_levels,
            'complexity_range': stage.complexity_range,
            'epochs_ratio': stage.epochs_ratio,
            'current_round': self.current_round,
            'completed_rounds': self.completed_rounds,
            'base_threshold': stage.performance_threshold,
            'current_threshold': self.get_current_threshold(),
            'performance_threshold': stage.performance_threshold,
            'current_evaluations': len(self.stage_performance_history),
            'min_evaluations': stage.min_evaluations,
            'dataset_size': len(current_dataset),
            'dataset_ratio': len(current_dataset) / len(self.full_dataset) if len(self.full_dataset) > 0 else 0,
            'is_completed': False,
            'debug_log_recent': self.debug_log[-10:],
            'advancement_stats': {
                'total_checks': self.total_advancement_checks,
                'attempts': self.advancement_attempts,
                'successful': self.successful_advancements
            }
        }

    def log_to_wandb(self, step: int):
        """记录到W&B（使用数值而非文字）- 增强调试版本"""
        try:
            import wandb
            if not hasattr(wandb, 'run') or wandb.run is None:
                return
            
            current_info = self.get_current_stage_info()
            progress_info = self.get_curriculum_progress()
            
            # 🔧 增强的W&B数据
            wandb_data = {
                'curriculum/current_stage_index': current_info['stage_index'],
                'curriculum/total_stages': progress_info['total_stages'],
                'curriculum/progress_ratio': progress_info['progress_ratio'],
                'curriculum/completed_stages_count': progress_info['completed_stages'],
                'curriculum/is_completed': float(current_info['is_completed']),
                'curriculum/debug_log_count': len(self.debug_log),
                'curriculum/advancement_checks': self.total_advancement_checks,
                'curriculum/advancement_attempts': self.advancement_attempts,
                'curriculum/successful_advancements': self.successful_advancements
            }
            
            # 当前阶段详细信息（数值）
            if not current_info['is_completed']:
                wandb_data.update({
                    'curriculum/current_evaluations': current_info['current_evaluations'],
                    'curriculum/min_evaluations': current_info['min_evaluations'],
                    'curriculum/performance_threshold': current_info['performance_threshold'],
                    'curriculum/epochs_ratio': current_info['epochs_ratio'],
                    'curriculum/level_count': len(current_info['dataset_levels']),
                    'curriculum/complexity_min': current_info['complexity_range'][0],
                    'curriculum/complexity_max': current_info['complexity_range'][1],
                    'curriculum/dataset_size': current_info['dataset_size'],
                    'curriculum/dataset_ratio': current_info['dataset_ratio']
                })
            
            # 性能历史统计
            if self.stage_performance_history:
                wandb_data.update({
                    'curriculum/stage_performance_mean': np.mean(self.stage_performance_history),
                    'curriculum/stage_performance_latest': self.stage_performance_history[-1],
                    'curriculum/stage_performance_std': np.std(self.stage_performance_history),
                    'curriculum/stage_performance_min': np.min(self.stage_performance_history),
                    'curriculum/stage_performance_max': np.max(self.stage_performance_history),
                    'curriculum/stage_performance_trend': np.mean(self.stage_performance_history[-3:]) if len(self.stage_performance_history) >= 3 else self.stage_performance_history[-1]
                })
            
            wandb.log(wandb_data, step=step)
            self._log_debug(f"📊 W&B指标已记录 (步数: {step}, 数据点: {len(wandb_data)})")
            
        except ImportError:
            pass  # wandb not available
        except Exception as e:
            self._log_debug(f"❌ W&B记录失败: {e}")
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存"""
        return {
            "current_stage": self.current_stage,
            "stage_performance_history": self.stage_performance_history,
            "all_stage_history": self.all_stage_history,
            "debug_log": self.debug_log[-50:],  # 保留最近50条日志
            "stage_statistics": self.stage_statistics
        }
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """获取整体课程进度"""
        total_stages = len(self.curriculum_stages)
        progress_ratio = (self.current_stage + 1) / total_stages if total_stages > 0 else 1.0
        
        return {
            'current_stage': self.current_stage,
            'total_stages': total_stages,
            'progress_ratio': progress_ratio,
            'completed_stages': len(self.all_stage_history),
            'stage_statistics': self.stage_statistics,
            'dataset_distribution': self.dataset_distribution,
            'all_stage_history': self.all_stage_history,
            'debug_summary': {
                'total_debug_entries': len(self.debug_log),
                'recent_entries': self.debug_log[-5:]  # 最近5条
            }
        }
    def save_detailed_log(self, output_dir: str):
        """保存详细的调试日志到文件 - 增强版本"""
        log_file = os.path.join(output_dir, "curriculum_detailed_debug.json")
        
        detailed_data = {
            "curriculum_state": self.get_curriculum_state(),
            "progress_info": self.get_curriculum_progress(),
            "current_stage_info": self.get_current_stage_info(),
            "export_timestamp": datetime.now().isoformat(),
            "debug_summary": {
                "total_stages": len(self.curriculum_stages),
                "current_stage": self.current_stage,
                "stages_completed": len(self.all_stage_history),
                "total_debug_entries": len(self.debug_log),
                "advancement_stats": {
                    "total_checks": self.total_advancement_checks,
                    "attempts": self.advancement_attempts,
                    "successful": self.successful_advancements,
                    "success_rate": self.successful_advancements / max(1, self.advancement_attempts)
                }
            },
            "recent_debug_log": self.debug_log[-50:] if len(self.debug_log) > 50 else self.debug_log,
            "stage_statistics_detailed": self.stage_statistics
        }
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            self._log_debug(f"💾 详细调试日志已保存: {log_file}")
            
            # 🔧 额外保存一个纯文本版本的调试日志
            text_log_file = os.path.join(output_dir, "curriculum_debug_text.log")
            with open(text_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 课程学习调试日志 - {datetime.now()} ===\n\n")
                f.write(f"总体统计:\n")
                f.write(f"  - 当前阶段: {self.current_stage}\n")
                f.write(f"  - 进阶检查次数: {self.total_advancement_checks}\n")
                f.write(f"  - 进阶尝试次数: {self.advancement_attempts}\n")
                f.write(f"  - 成功进阶次数: {self.successful_advancements}\n")
                f.write(f"  - 调试日志条数: {len(self.debug_log)}\n\n")
                
                f.write("详细调试日志:\n")
                f.write("="*50 + "\n")
                for entry in self.debug_log:
                    f.write(entry + "\n")
            
            self._log_debug(f"📝 文本调试日志已保存: {text_log_file}")
            
        except Exception as e:
            self._log_debug(f"❌ 保存调试日志失败: {e}")

    # 🔧 新增：定期状态报告方法
    def log_periodic_status(self, step: int = None):
        """定期记录课程学习状态"""
        self._log_debug(f"📊 定期状态报告 (步数: {step if step else 'N/A'})")
        
        current_info = self.get_current_stage_info()
        
        self._log_debug(f"  - 当前阶段: {current_info['stage_index']} ({current_info['stage_name']})")
        if not current_info['is_completed']:
            self._log_debug(f"  - 数据集: {current_info['dataset_size']} 样本 ({current_info['dataset_ratio']:.1%})")
            self._log_debug(f"  - 评估进度: {current_info['current_evaluations']}/{current_info['min_evaluations']}")
            self._log_debug(f"  - 性能阈值: {current_info['performance_threshold']}")
        
        self._log_debug(f"  - 进阶统计: {self.successful_advancements}/{self.advancement_attempts} 成功")
        
        if self.stage_performance_history:
            recent_perf = self.stage_performance_history[-min(3, len(self.stage_performance_history)):]
            self._log_debug(f"  - 最近性能: {[f'{p:.3f}' for p in recent_perf]}")

    # 🔧 新增：强制调试输出方法
    def force_debug_output(self):
        """强制输出当前所有调试信息"""
        self._log_debug("🔧 强制调试输出开始")
        self._log_debug(f"📊 调试统计: {len(self.debug_log)} 条日志")
        
        # 输出最近的调试日志
        recent_logs = self.debug_log[-20:] if len(self.debug_log) > 20 else self.debug_log
        self._log_debug(f"📝 最近{len(recent_logs)}条调试日志:")
        for i, entry in enumerate(recent_logs, 1):
            self._log_debug(f"  {i:2d}. {entry}")
        
        # 输出当前状态
        self.log_periodic_status()
        
        self._log_debug("🔧 强制调试输出结束")

    def _initialize_stage_tracker(self):
        """初始化阶段训练跟踪"""
        if self.current_stage < len(self.curriculum_stages):
            stage = self.curriculum_stages[self.current_stage]
            current_dataset = self.get_current_stage_dataset()
            self.stage_dataset_size = len(current_dataset)
            
            # 计算该阶段需要的最少步数
            min_steps = getattr(stage, 'min_steps_per_epoch', 10)
            require_full_epoch = getattr(stage, 'require_full_epoch', True)
            
            if require_full_epoch:
                # 计算完整训练一遍需要的步数（假设batch_size=1）
                estimated_steps_per_epoch = max(self.stage_dataset_size, min_steps)
            else:
                estimated_steps_per_epoch = min_steps
            
            self.stage_training_tracker = {
                'stage_name': stage.name,
                'stage_index': self.current_stage,
                'dataset_size': self.stage_dataset_size,
                'require_full_epoch': require_full_epoch,
                'min_steps_per_epoch': min_steps,
                'estimated_steps_per_epoch': estimated_steps_per_epoch,
                'target_epochs': 1 if require_full_epoch else 0,
                'steps_completed': 0,
                'epochs_completed': 0,
                'start_step': self.current_stage_start_step,
                'is_epoch_requirement_met': False
            }
            
            self._log_debug(f"🔧 初始化阶段{self.current_stage}训练跟踪:")
            self._log_debug(f"  - 阶段名称: {stage.name}")
            self._log_debug(f"  - 数据集大小: {self.stage_dataset_size}")
            self._log_debug(f"  - 要求完整epoch: {require_full_epoch}")
            self._log_debug(f"  - 预估每epoch步数: {estimated_steps_per_epoch}")
            self._log_debug(f"  - 最小步数要求: {min_steps}")
        else:
            self.stage_training_tracker = {}

    def update_training_progress(self, current_step: int):
        """更新训练进度"""
        if not self.stage_training_tracker:
            return
            
        # 计算当前阶段已完成的步数
        steps_in_stage = current_step - self.stage_training_tracker['start_step']
        self.stage_training_tracker['steps_completed'] = steps_in_stage
        
        # 计算完成的epoch数
        estimated_steps_per_epoch = self.stage_training_tracker['estimated_steps_per_epoch']
        if estimated_steps_per_epoch > 0:
            epochs_completed = steps_in_stage / estimated_steps_per_epoch
            self.stage_training_tracker['epochs_completed'] = epochs_completed
            
            # 检查是否满足完整epoch要求
            require_full_epoch = self.stage_training_tracker['require_full_epoch']
            target_epochs = self.stage_training_tracker['target_epochs']
            
            if require_full_epoch:
                self.stage_training_tracker['is_epoch_requirement_met'] = epochs_completed >= target_epochs
            else:
                # 如果不要求完整epoch，只要达到最小步数即可
                min_steps = self.stage_training_tracker['min_steps_per_epoch']
                self.stage_training_tracker['is_epoch_requirement_met'] = steps_in_stage >= min_steps
        
        # 每50步记录一次进度
        if steps_in_stage % 50 == 0 and steps_in_stage > 0:
            self._log_training_progress()

    def _log_training_progress(self):
        """记录训练进度"""
        if not self.stage_training_tracker:
            return
            
        tracker = self.stage_training_tracker
        stage_name = tracker['stage_name']
        steps_completed = tracker['steps_completed']
        epochs_completed = tracker['epochs_completed']
        is_met = tracker['is_epoch_requirement_met']
        
        self._log_debug(f"📈 阶段{stage_name}训练进度:")
        self._log_debug(f"  - 已完成步数: {steps_completed}")
        self._log_debug(f"  - 已完成epoch: {epochs_completed:.2f}")
        self._log_debug(f"  - epoch要求满足: {is_met}")
        
        if tracker['require_full_epoch']:
            progress_percent = min(100, epochs_completed * 100)
            self._log_debug(f"  - 完整训练进度: {progress_percent:.1f}%")

    def is_stage_training_complete(self) -> bool:
        """检查当前阶段是否完成了完整训练要求"""
        if not self.stage_training_tracker:
            return True  # 如果没有跟踪器，认为已完成
            
        return self.stage_training_tracker.get('is_epoch_requirement_met', False)

    def get_stage_training_status(self) -> Dict[str, Any]:
        """获取当前阶段训练状态"""
        if not self.stage_training_tracker:
            return {'status': 'no_tracker', 'complete': True}
            
        tracker = self.stage_training_tracker
        return {
            'stage_name': tracker['stage_name'],
            'stage_index': tracker['stage_index'],
            'dataset_size': tracker['dataset_size'],
            'steps_completed': tracker['steps_completed'],
            'epochs_completed': tracker['epochs_completed'],
            'require_full_epoch': tracker['require_full_epoch'],
            'is_epoch_requirement_met': tracker['is_epoch_requirement_met'],
            'estimated_steps_per_epoch': tracker['estimated_steps_per_epoch'],
            'progress_percent': min(100, tracker['epochs_completed'] * 100) if tracker['require_full_epoch'] else 100
        }

    def update_stage_start_step(self, current_step: int):
        """更新当前阶段开始步数 - 用于阶段进阶时重置步数计数"""
        self.current_stage_start_step = current_step
        if self.stage_training_tracker:
            self.stage_training_tracker['start_step'] = current_step
            # 重置步数和epoch计数
            self.stage_training_tracker['steps_completed'] = 0
            self.stage_training_tracker['epochs_completed'] = 0
            self.stage_training_tracker['is_epoch_requirement_met'] = False
            
        self._log_debug(f"🔧 更新阶段开始步数: {current_step}")
        self._log_debug(f"  - 重置训练进度跟踪器")

    def get_stage_advancement_requirements(self) -> Dict[str, Any]:
        """获取当前阶段的进阶要求"""
        if self.current_stage >= len(self.curriculum_stages):
            return {'stage_completed': True, 'requirements': []}
            
        stage = self.curriculum_stages[self.current_stage]
        training_status = self.get_stage_training_status()
        current_threshold = self.get_current_threshold()
        
        requirements = []
        
        # 性能要求
        recent_performance = np.mean(self.stage_performance_history[-2:]) if len(self.stage_performance_history) >= 2 else 0
        performance_met = recent_performance >= current_threshold
        requirements.append({
            'type': 'performance',
            'description': f'平均性能达到 {current_threshold:.3f}',
            'current': recent_performance,
            'target': current_threshold,
            'met': performance_met
        })
        
        # 评估次数要求
        eval_count_met = len(self.stage_performance_history) >= stage.min_evaluations
        requirements.append({
            'type': 'evaluations',
            'description': f'完成至少 {stage.min_evaluations} 次评估',
            'current': len(self.stage_performance_history),
            'target': stage.min_evaluations,
            'met': eval_count_met
        })
        
        # 完整训练要求
        training_complete = self.is_stage_training_complete()
        if training_status.get('require_full_epoch', False):
            requirements.append({
                'type': 'full_training',
                'description': '完成至少1个完整epoch的训练',
                'current': training_status.get('epochs_completed', 0),
                'target': 1.0,
                'met': training_complete,
                'progress_percent': training_status.get('progress_percent', 0)
            })
        
        all_met = all(req['met'] for req in requirements)
        
        return {
            'stage_name': stage.name,
            'stage_index': self.current_stage,
            'can_advance': all_met,
            'requirements': requirements,
            'training_status': training_status
        }


# Moved from train.py (setup_curriculum_manager)
# This function sets up EnhancedCurriculumManager, not FixedEnhancedCurriculumManager.
# We might need to reconcile this with setup_fixed_curriculum_manager.
def setup_enhanced_curriculum_manager(script_cfg: ScriptConfig, dataset: Dataset) -> Optional[EnhancedCurriculumManager]:
    if not script_cfg.enable_curriculum: # type: ignore
        logger.info("Curriculum learning disabled via script_cfg.")
        return None
    # ... (Logic from train.py setup_curriculum_manager, using create_custom_curriculum_stages or create_default_curriculum_stages)
    # This logic needs access to create_custom_curriculum_stages and create_default_curriculum_stages from .stages
    logger.info("Setting up EnhancedCurriculumManager...")
    # Placeholder logic:
    stages = create_default_curriculum_stages() # from .stages
    return EnhancedCurriculumManager(stages, dataset)


# Moved from curriculum_debug_config.py
def setup_fixed_curriculum_manager(script_cfg: ScriptConfig, dataset: Dataset) -> Optional[FixedEnhancedCurriculumManager]:
    if not script_cfg.enable_curriculum: # type: ignore
        logger.info("📚 Curriculum learning (fixed manager) is disabled.")
        return None
    # ... (Logic from curriculum_debug_config.py setup_fixed_curriculum_manager)
    # This logic needs access to create_fixed_curriculum_stages (if that's also moved/created) or other stage creation logic
    # logger.info("Setting up FixedEnhancedCurriculumManager...") # Original log
    # Placeholder logic:
    # stages = create_fixed_curriculum_stages() # This function would also need to be defined/moved
    # stages = create_default_curriculum_stages() # Using default for now as create_fixed_curriculum_stages is not in scope
    # return FixedEnhancedCurriculumManager(stages, dataset) # Original return

    # MODIFIED function:
    if not script_cfg.enable_curriculum:  # type: ignore
        logger.info("📚 Curriculum learning (fixed manager) is disabled by script_cfg.")
        return None

    logger.info("⚙️ Setting up FixedEnhancedCurriculumManager...")
    stages: List[CurriculumStageConfig] = []

    # Priority 1: Use script_cfg.curriculum_stages if provided
    if hasattr(script_cfg, 'curriculum_stages') and script_cfg.curriculum_stages: # type: ignore
        logger.info("Found curriculum_stages in script_cfg. Attempting to use them.")
        try:
            for stage_dict in script_cfg.curriculum_stages: # type: ignore
                # Ensure all required fields for CurriculumStageConfig are present or have defaults
                config = CurriculumStageConfig(
                    name=stage_dict.get('name', 'Unnamed Stage'),
                    dataset_levels=stage_dict.get('dataset_levels', []),
                    complexity_range=tuple(stage_dict.get('complexity_range', (0.0, 10.0))), # Ensure it's a tuple
                    epochs_ratio=stage_dict.get('epochs_ratio', 0.1), # Default ratio if missing
                    performance_threshold=stage_dict.get('performance_threshold', 0.6),
                    min_evaluations=stage_dict.get('min_evaluations', 5),
                    description=stage_dict.get('description', '')
                )
                stages.append(config)
            if stages:
                 logger.info(f"✅ Successfully loaded {len(stages)} stages from script_cfg.curriculum_stages.")
            else:
                logger.warning("⚠️ script_cfg.curriculum_stages was provided but resulted in zero stages. Check configuration.")
        except Exception as e:
            logger.error(f"❌ Error processing script_cfg.curriculum_stages: {e}. Falling back.")
            stages = [] # Reset stages if parsing failed

    # Priority 2: Use create_custom_curriculum_stages
    if not stages: # If stages were not loaded from script_cfg.curriculum_stages
        logger.info("No stages from script_cfg.curriculum_stages or loading failed. Trying create_custom_curriculum_stages...")
        try:
            # Calculate dataset_distribution - using the static helper method
            # Pass logger.info for optional logging within the method if needed, or handle logging outside.
            dataset_dist = FixedEnhancedCurriculumManager._calculate_dataset_distribution(dataset, logger.info)

            if not dataset_dist or not dataset_dist.get('total_samples'):
                logger.warning("⚠️ Dataset distribution analysis resulted in no samples. Cannot create custom stages effectively.")
            else:
                # Ensure all expected attributes exist on script_cfg or provide defaults
                custom_stages_params = {
                    "dataset_distribution": dataset_dist,
                    "focus_levels": getattr(script_cfg, 'curriculum_focus_levels', None), # type: ignore
                    "complexity_emphasis": getattr(script_cfg, 'curriculum_complexity_emphasis', None) # type: ignore
                }
                logger.info(f"Parameters for create_custom_curriculum_stages: focus_levels={custom_stages_params['focus_levels']}, emphasis={custom_stages_params['complexity_emphasis']}")
                stages = create_custom_curriculum_stages(**custom_stages_params) # from .stages

            if stages:
                logger.info(f"✅ Successfully created {len(stages)} custom stages.")
            else:
                logger.warning("⚠️ create_custom_curriculum_stages resulted in zero stages.")
        except Exception as e:
            logger.error(f"❌ Error calling create_custom_curriculum_stages: {e}. Falling back to default.")
            stages = [] # Reset stages

    # Priority 3: Fallback to create_default_curriculum_stages
    if not stages:
        logger.info("No stages from custom creation or it failed. Falling back to create_default_curriculum_stages.")
        try:
            # 🔧 NEW: 传递ScriptConfig中的阈值参数到create_default_curriculum_stages
            performance_thresholds = []
            for i in range(1, 6):
                threshold_attr = f"curriculum_performance_threshold_{i}"
                if hasattr(script_cfg, threshold_attr):
                    threshold_value = getattr(script_cfg, threshold_attr)
                    if threshold_value is not None:
                        performance_thresholds.append(threshold_value)
                        logger.info(f"📊 从ScriptConfig读取阈值: {threshold_attr}={threshold_value}")
            
            min_evaluations = 5  # 默认值
            if hasattr(script_cfg, 'curriculum_min_evaluations') and script_cfg.curriculum_min_evaluations is not None:
                min_evaluations = script_cfg.curriculum_min_evaluations
                logger.info(f"📊 从ScriptConfig读取最小评估次数: curriculum_min_evaluations={min_evaluations}")
            
            # 如果从ScriptConfig中获取了完整的阈值，就使用它们
            if len(performance_thresholds) >= 5:
                stages = create_default_curriculum_stages(
                    performance_thresholds=performance_thresholds[:5],
                    min_evaluations=min_evaluations
                )
                logger.info(f"✅ 使用ScriptConfig中的阈值创建课程阶段: {performance_thresholds[:5]}")
            else:
                # 否则使用默认的，但仍然传递环境变量或其他可用的参数
                stages = create_default_curriculum_stages(min_evaluations=min_evaluations)
                logger.info(f"✅ 使用默认阈值创建课程阶段（最小评估次数: {min_evaluations}）")
            
            if stages:
                logger.info(f"✅ Successfully created {len(stages)} default stages.")
            else:
                logger.error("❌ create_default_curriculum_stages also resulted in zero stages! This is problematic.")
                return None # Cannot proceed without stages
        except Exception as e:
            logger.error(f"❌ Error calling create_default_curriculum_stages: {e}.")
            return None # Cannot proceed if default stage creation fails

    if not stages:
        logger.error("🚫 Failed to create or load any curriculum stages. Cannot initialize FixedEnhancedCurriculumManager.")
        return None

    # Apply environment variable overrides
    # For these environment variables to take effect, ensure they are `export`ed in your shell script (e.g., `export CURRICULUM_PERFORMANCE_THRESHOLD_1=0.75`).
    if stages: # Ensure there are stages to process
        logger.info(f"⚙️ Checking for environment variable overrides for {len(stages)} stages...")
        for i, stage_config in enumerate(stages):
            stage_index = i + 1 # Environment variables are 1-indexed

            # Check for performance_threshold override
            env_threshold_str = os.environ.get(f"CURRICULUM_PERFORMANCE_THRESHOLD_{stage_index}")
            if env_threshold_str:
                try:
                    env_threshold_val = float(env_threshold_str)
                    original_threshold = stage_config.performance_threshold
                    if abs(original_threshold - env_threshold_val) > 1e-6: # Check if different to avoid unnecessary logs
                        logger.info(f"Applying ENV override for STAGE {stage_index} ('{stage_config.name}'): " \
                                     f"CURRICULUM_PERFORMANCE_THRESHOLD_{stage_index}='{env_threshold_str}'. " \
                                     f"Changing performance_threshold from {original_threshold:.4f} to {env_threshold_val:.4f}.")
                        stage_config.performance_threshold = env_threshold_val
                    else:
                        logger.info(f"ENV CURRICULUM_PERFORMANCE_THRESHOLD_{stage_index} ('{env_threshold_str}') matches existing threshold for STAGE {stage_index} ('{stage_config.name}'). No change.")
                except ValueError:
                    logger.warning(f"Invalid value for CURRICULUM_PERFORMANCE_THRESHOLD_{stage_index}: '{env_threshold_str}'. Must be a float. Ignoring.")

            # Check for min_evaluations override
            env_min_eval_str = os.environ.get(f"CURRICULUM_MIN_EVALUATIONS_{stage_index}") # Corrected typo
            if env_min_eval_str: # Check if not None and not empty
                try:
                    env_min_eval_val = int(env_min_eval_str)
                    original_min_eval = stage_config.min_evaluations
                    if original_min_eval != env_min_eval_val:
                        logger.info(f"Applying ENV override for STAGE {stage_index} ('{stage_config.name}'): " \
                                     f"CURRICULUM_MIN_EVALUATIONS_{stage_index}='{env_min_eval_str}'. " \
                                     f"Changing min_evaluations from {original_min_eval} to {env_min_eval_val}.")
                        stage_config.min_evaluations = env_min_eval_val
                    else:
                        logger.info(f"ENV CURRICULUM_MIN_EVALUATIONS_{stage_index} ('{env_min_eval_str}') matches existing min_evaluations for STAGE {stage_index} ('{stage_config.name}'). No change.")
                except ValueError:
                    logger.warning(f"Invalid value for CURRICULUM_MIN_EVALUATIONS_{stage_index}: '{env_min_eval_str}'. Must be an integer. Ignoring.")
    else:
        logger.info("No stages were populated, skipping environment variable override check.")

    logger.info(f"🏁 Initializing FixedEnhancedCurriculumManager with {len(stages)} stages.")
    # Note: The user should be informed that for these environment variables to take effect,
    # they must be `export`ed in their shell script (e.g., `export CURRICULUM_PERFORMANCE_THRESHOLD_1=0.75`).
    # This guidance is included as a comment above and should be part of the commit message or user documentation.
    return FixedEnhancedCurriculumManager(stages, dataset)
