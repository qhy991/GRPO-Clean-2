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
            0: {"performance_threshold": 0.7, "min_evaluations": 8, "stability_window": 4},
            1: {"performance_threshold": 0.65, "min_evaluations": 10, "stability_window": 5},
            2: {"performance_threshold": 0.6, "min_evaluations": 15, "stability_window": 6},
            3: {"performance_threshold": 0.55, "min_evaluations": 20, "stability_window": 8, "max_stay_steps": 200}
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
            "stage_performance_history": self.stage_performance_history,
            "stage_start_steps": self.stage_start_steps,
            "stage_statistics": self.stage_statistics
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        self.current_stage = state_dict.get("current_stage", 0)
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        self.stage_start_steps = state_dict.get("stage_start_steps", {})
        self.stage_statistics = state_dict.get("stage_statistics", [])
        logger.info(f"EnhancedCurriculumManager state loaded. Current stage: {self.current_stage}")

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
    """修复版本的增强课程学习管理器 - 增强调试日志"""
    
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
        
        self._log_debug("🚀 FixedEnhancedCurriculumManager 开始初始化")
        self._log_debug(f"📊 课程配置: 总阶段数={len(curriculum_stages)}, 数据集大小={len(dataset)}")
        
        # 详细记录每个阶段的配置
        for i, stage in enumerate(curriculum_stages):
            self._log_debug(f"  阶段{i} ({stage.name}):")
            self._log_debug(f"    - 等级: {stage.dataset_levels}")
            self._log_debug(f"    - 复杂度: {stage.complexity_range}")
            self._log_debug(f"    - 性能阈值: {stage.performance_threshold}")
            self._log_debug(f"    - 最小评估: {stage.min_evaluations}")
        
        self._analyze_dataset_distribution()
        self._validate_curriculum_design()
        
        # 验证当前阶段数据集
        current_dataset = self.get_current_stage_dataset()
        self._log_debug(f"✅ 初始化完成: 当前阶段数据集大小={len(current_dataset)}")
    def _validate_curriculum_design(self):
        """验证课程设计的合理性"""
        available_levels = set(self.dataset_distribution['level_counts'].keys())
        covered_levels = set()
        
        for stage in self.curriculum_stages:
            covered_levels.update([level.lower() for level in stage.dataset_levels])
        
        uncovered_levels = available_levels - covered_levels
        if uncovered_levels:
            self._log_debug(f"⚠️ 未覆盖的数据集等级: {uncovered_levels}")
        
        total_ratio = sum(stage.epochs_ratio for stage in self.curriculum_stages)
        if abs(total_ratio - 1.0) > 0.01:
            self._log_debug(f"⚠️ Epoch比例总和: {total_ratio:.3f} (应该接近1.0)")
    def _log_debug(self, message: str):
        """记录调试信息 - 增强版本"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        debug_entry = f"[{timestamp}] {message}"
        self.debug_log.append(debug_entry)
        
        # 🔧 确保调试信息输出到正确的 logger
        logger.info(f"📚 CURRICULUM (Fixed): {message}")
        
        # 🔧 额外：每100条调试日志输出一次统计
        if len(self.debug_log) % 100 == 0:
            logger.info(f"📊 课程调试统计: {len(self.debug_log)} 条日志, 当前阶段={self.current_stage}")
    def _analyze_dataset_distribution(self):
        """分析数据集的等级和复杂度分布"""
        if len(self.full_dataset) == 0:
            self._log_debug("❌ 空数据集")
            return
        
        level_counts = {}
        complexity_by_level = {}
        
        for example in self.full_dataset:
            level = example.get('level', 'unknown').lower()
            complexity = example.get('complexity_score', 5.0)
            
            level_counts[level] = level_counts.get(level, 0) + 1
            
            if level not in complexity_by_level:
                complexity_by_level[level] = []
            complexity_by_level[level].append(complexity)
        
        self.dataset_distribution = {
            'level_counts': level_counts,
            'complexity_by_level': complexity_by_level,
            'total_samples': len(self.full_dataset)
        }
        
        # 详细记录分布信息
        self._log_debug(f"数据集分布分析 - 总样本: {len(self.full_dataset)}")
        for level, count in level_counts.items():
            if level in complexity_by_level and complexity_by_level[level]:
                avg_complexity = np.mean(complexity_by_level[level])
                complexity_range = (np.min(complexity_by_level[level]), np.max(complexity_by_level[level]))
                self._log_debug(f"  {level}: {count}样本, 平均复杂度: {avg_complexity:.2f}, 范围: {complexity_range}")
    def should_advance_stage(self, recent_performance: float) -> bool:
        """判断是否应该进入下一阶段 - 增强调试版本"""
        self.total_advancement_checks += 1
        current_step = self.total_advancement_checks  # 简单的步数计数
        
        self._log_debug(f"🔍 第{self.total_advancement_checks}次进阶检查")
        self._log_debug(f"  - 当前性能: {recent_performance:.4f}")
        self._log_debug(f"  - 当前阶段: {self.current_stage}")
        self._log_debug(f"  - 历史长度: {len(self.stage_performance_history)}")
        
        if self.current_stage >= len(self.curriculum_stages) - 1:
            self._log_debug("❌ 已在最后阶段，不能继续进阶")
            return False
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(recent_performance)
        
        self._log_debug(f"  - 性能已记录，新历史长度: {len(self.stage_performance_history)}")
        self._log_debug(f"  - 阶段配置: {stage.name}, 阈值={stage.performance_threshold}, 最小评估={stage.min_evaluations}")
        
        # 需要足够的评估次数
        if len(self.stage_performance_history) < stage.min_evaluations:
            self._log_debug(f"❌ 评估次数不足: {len(self.stage_performance_history)}/{stage.min_evaluations}")
            return False
        
        # 检查最近的性能表现
        recent_window = min(3, len(self.stage_performance_history))
        recent_performances = self.stage_performance_history[-recent_window:]
        recent_avg = np.mean(recent_performances)
        
        self._log_debug(f"  - 最近{recent_window}次性能: {recent_performances}")
        self._log_debug(f"  - 最近平均性能: {recent_avg:.4f}")
        self._log_debug(f"  - 性能阈值: {stage.performance_threshold}")
        
        should_advance = recent_avg >= stage.performance_threshold
        
        # 🔧 详细记录决策过程
        if should_advance:
            self._log_debug(f"✅ 满足进阶条件!")
            self._log_debug(f"  - 性能检查: {recent_avg:.4f} >= {stage.performance_threshold} ✅")
            self._log_debug(f"  - 评估检查: {len(self.stage_performance_history)} >= {stage.min_evaluations} ✅")
        else:
            self._log_debug(f"❌ 不满足进阶条件")
            if recent_avg < stage.performance_threshold:
                self._log_debug(f"  - 性能不足: {recent_avg:.4f} < {stage.performance_threshold}")
            
        self._log_debug(f"  - 进阶决策: {should_advance}")
        return should_advance

    def advance_stage(self) -> bool:
        """进入下一阶段 - 增强调试版本"""
        self.advancement_attempts += 1
        
        self._log_debug(f"🎯 第{self.advancement_attempts}次进阶尝试")
        
        if self.current_stage >= len(self.curriculum_stages) - 1:
            self._log_debug("❌ 已在最后阶段，无法进阶")
            return False
        
        # 记录当前阶段的最终统计
        old_stage = self.current_stage
        old_stage_name = self.curriculum_stages[old_stage].name
        
        # 🔧 详细记录进阶前的状态
        self._log_debug(f"📊 进阶前状态统计:")
        self._log_debug(f"  - 离开阶段: {old_stage} ({old_stage_name})")
        self._log_debug(f"  - 该阶段评估次数: {len(self.stage_performance_history)}")
        
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
            'total_evaluations': len(self.stage_performance_history),
            'final_performance': self.stage_performance_history[-1] if self.stage_performance_history else 0,
            'average_performance': np.mean(self.stage_performance_history) if self.stage_performance_history else 0,
            'performance_history': self.stage_performance_history.copy(),
            'completion_timestamp': datetime.now().isoformat()
        }
        
        # 保存到全部历史
        self.all_stage_history.append(final_stats)
        
        # 进阶到下一阶段
        self.current_stage += 1
        self.stage_performance_history = []  # 重置性能历史
        
        new_stage_name = self.curriculum_stages[self.current_stage].name if self.current_stage < len(self.curriculum_stages) else "Final"
        
        self.successful_advancements += 1
        
        self._log_debug(f"🎉 成功进阶!")
        self._log_debug(f"  - 进阶路径: {old_stage}({old_stage_name}) -> {self.current_stage}({new_stage_name})")
        self._log_debug(f"  - 成功进阶次数: {self.successful_advancements}/{self.advancement_attempts}")
        self._log_debug(f"  - 前阶段最终性能: {final_stats['final_performance']:.4f}")
        
        # 🔧 详细记录新阶段信息
        if self.current_stage < len(self.curriculum_stages):
            new_stage = self.curriculum_stages[self.current_stage]
            new_dataset = self.get_current_stage_dataset()
            
            self._log_debug(f"📈 新阶段详情:")
            self._log_debug(f"  - 阶段名称: {new_stage.name}")
            self._log_debug(f"  - 目标等级: {new_stage.dataset_levels}")
            self._log_debug(f"  - 复杂度范围: {new_stage.complexity_range}")
            self._log_debug(f"  - 性能阈值: {new_stage.performance_threshold}")
            self._log_debug(f"  - 数据集大小: {len(new_dataset)}")
            self._log_debug(f"  - 数据集比例: {len(new_dataset)/len(self.full_dataset)*100:.1f}%")
        else:
            self._log_debug("🎓 所有阶段已完成!")
        
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
    logger.info("Setting up FixedEnhancedCurriculumManager...")
    # Placeholder logic:
    # stages = create_fixed_curriculum_stages() # This function would also need to be defined/moved
    stages = create_default_curriculum_stages() # Using default for now as create_fixed_curriculum_stages is not in scope
    return FixedEnhancedCurriculumManager(stages, dataset)
