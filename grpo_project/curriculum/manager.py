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

    def _analyze_dataset_distribution(self):
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager._analyze_dataset_distribution)
        pass

    def _validate_curriculum_design(self):
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager._validate_curriculum_design)
        pass

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

    def get_current_stage_info(self) -> Dict[str, Any]:
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager.get_current_stage_info)
        return {}

    def get_curriculum_progress(self) -> Dict[str, Any]:
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager.get_curriculum_progress)
        return {}

    def log_to_wandb(self, step: int):
        # ... (Logic from enhanced_curriculum.py EnhancedCurriculumManager.log_to_wandb)
        pass


# Moved from curriculum_debug_config.py
class FixedEnhancedCurriculumManager:
    """修复版本的增强课程学习管理器"""
    
    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history = []
        self.all_stage_history = []
        self.stage_statistics = []
        self.debug_log = []
        
        self._log_debug(f"FixedEnhancedCurriculumManager initialized - Total Stages: {len(curriculum_stages)}")
        self._analyze_dataset_distribution()
        self._validate_curriculum_design()
        
        # 验证当前阶段数据集
        current_dataset = self.get_current_stage_dataset()
        self._log_debug(f"Current stage ({self.current_stage}) dataset size: {len(current_dataset)}")

    def _log_debug(self, message: str):
        """记录调试信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_entry = f"[{timestamp}] {message}"
        self.debug_log.append(debug_entry)
        logger.info(f"📚 CURRICULUM (Fixed): {message}")

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
        
        self._log_debug(f"数据集分布分析 - 总样本: {len(self.full_dataset)}")
        for level, count in level_counts.items():
            if level in complexity_by_level and complexity_by_level[level]:
                avg_complexity = np.mean(complexity_by_level[level])
                self._log_debug(f"  {level}: {count}样本, 平均复杂度: {avg_complexity:.2f}")

    def _validate_curriculum_design(self):
        """验证课程设计的合理性"""
        available_levels = set(self.dataset_distribution['level_counts'].keys())
        covered_levels = set()
        
        for stage in self.curriculum_stages:
            covered_levels.update([level.lower() for level in stage.dataset_levels])
        
        uncovered_levels = available_levels - covered_levels
        if uncovered_levels:
            self._log_debug(f"⚠️ 未覆盖的数据集等级: {uncovered_levels}")

    def get_current_stage_dataset(self) -> Dataset:
        """获取当前阶段的数据集"""
        if self.current_stage >= len(self.curriculum_stages):
            self._log_debug("✅ 课程学习完成，使用全部数据集")
            return self.full_dataset
        
        stage = self.curriculum_stages[self.current_stage]
        
        # 双层过滤：数据集等级 + 复杂度范围
        filtered_indices = []
        level_filter_count = 0
        complexity_filter_count = 0
        
        self._log_debug(f"开始过滤阶段{self.current_stage}数据集 - 目标等级: {stage.dataset_levels}, 复杂度: {stage.complexity_range}")
        
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
            self._log_debug(f"❌ 阶段{self.current_stage}没有匹配的样本，使用全部数据集")
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
            'complexity_range': stage.complexity_range
        }
        self.stage_statistics.append(stage_stats)
        
        self._log_debug(f"阶段{self.current_stage}数据集过滤完成:")
        self._log_debug(f"  等级过滤通过: {level_filter_count}/{len(self.full_dataset)}")
        self._log_debug(f"  复杂度过滤通过: {complexity_filter_count}/{level_filter_count}")
        self._log_debug(f"  最终选择: {len(stage_dataset)}样本 ({stage_stats['selection_ratio']:.1%})")
        
        return stage_dataset

    def get_current_stage_name(self) -> str:
        """Get the name of the current curriculum stage"""
        if self.current_stage >= len(self.curriculum_stages):
            return "completed"
        return self.curriculum_stages[self.current_stage].name

    def should_advance_stage(self, recent_performance: float) -> bool:
        """判断是否应该进入下一阶段"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            self._log_debug("已在最后阶段，不能继续进阶")
            return False
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(recent_performance)
        
        self._log_debug(f"进阶检查 - 当前性能: {recent_performance:.4f}, 历史长度: {len(self.stage_performance_history)}")
        
        # 需要足够的评估次数
        if len(self.stage_performance_history) < stage.min_evaluations:
            self._log_debug(f"评估次数不足: {len(self.stage_performance_history)}/{stage.min_evaluations}")
            return False
        
        # 检查最近的性能表现
        recent_window = min(3, len(self.stage_performance_history))
        recent_avg = np.mean(self.stage_performance_history[-recent_window:])
        
        should_advance = recent_avg >= stage.performance_threshold
        
        self._log_debug(f"最近{recent_window}次平均性能: {recent_avg:.4f}")
        self._log_debug(f"性能阈值: {stage.performance_threshold}")
        self._log_debug(f"是否应该进阶: {should_advance}")
        
        return should_advance

    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            self._log_debug("❌ 已在最后阶段，无法进阶")
            return False
        
        # 记录当前阶段的最终统计
        old_stage = self.current_stage
        old_stage_name = self.curriculum_stages[old_stage].name
        
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
        
        self._log_debug(f"🎯 成功进阶: {old_stage}({old_stage_name}) -> {self.current_stage}({new_stage_name})")
        self._log_debug(f"前阶段最终性能: {final_stats['final_performance']:.4f}")
        
        # 获取新阶段的数据集大小
        if self.current_stage < len(self.curriculum_stages):
            new_dataset = self.get_current_stage_dataset()
            self._log_debug(f"新阶段数据集大小: {len(new_dataset)}")
        
        return True

    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        if self.current_stage >= len(self.curriculum_stages):
            return {
                'stage_index': self.current_stage,
                'stage_name': 'completed',
                'dataset_levels': 'all',
                'complexity_range': 'all',
                'is_completed': True,
                'debug_log_recent': self.debug_log[-10:]
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
            'is_completed': False,
            'debug_log_recent': self.debug_log[-10:]
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
            'dataset_distribution': getattr(self, 'dataset_distribution', {}),
            'all_stage_history': self.all_stage_history,
            'debug_summary': {
                'total_debug_entries': len(self.debug_log),
                'recent_entries': self.debug_log[-5:]
            }
        }

    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存"""
        return {
            "current_stage": self.current_stage,
            "stage_performance_history": self.stage_performance_history,
            "all_stage_history": self.all_stage_history,
            "debug_log": self.debug_log[-50:],
            "stage_statistics": self.stage_statistics
        }

    def load_curriculum_state(self, state_dict: Dict[str, Any]):
        """从字典加载课程学习管理器的状态"""
        self.current_stage = state_dict.get("current_stage", 0)
        self.stage_performance_history = state_dict.get("stage_performance_history", [])
        self.all_stage_history = state_dict.get("all_stage_history", [])
        self.debug_log = state_dict.get("debug_log", [])
        self.stage_statistics = state_dict.get("stage_statistics", [])
        
        self._log_debug(f"状态已加载 - 当前阶段: {self.current_stage}")
        if self.current_stage < len(self.curriculum_stages):
            stage_name = self.curriculum_stages[self.current_stage].name
            self._log_debug(f"恢复到阶段: {stage_name}")

    def log_to_wandb(self, step: int):
        """记录到W&B（使用数值而非文字）"""
        try:
            import wandb
            if not hasattr(wandb, 'run') or wandb.run is None:
                return
            
            current_info = self.get_current_stage_info()
            progress_info = self.get_curriculum_progress()
            
            # 基础数值信息
            wandb_data = {
                'curriculum/current_stage_index': current_info['stage_index'],
                'curriculum/total_stages': progress_info['total_stages'],
                'curriculum/progress_ratio': progress_info['progress_ratio'],
                'curriculum/completed_stages_count': progress_info['completed_stages'],
                'curriculum/is_completed': float(current_info['is_completed'])
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
                })
            
            wandb.log(wandb_data, step=step)
            self._log_debug(f"已记录W&B指标 (步数: {step})")
            
        except ImportError:
            pass  # wandb not available
        except Exception as e:
            self._log_debug(f"W&B记录失败: {e}")

    def save_detailed_log(self, output_dir: str):
        """保存详细的调试日志到文件"""
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
                "total_debug_entries": len(self.debug_log)
            }
        }
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            self._log_debug(f"详细调试日志已保存: {log_file}")
        except Exception as e:
            self._log_debug(f"保存调试日志失败: {e}")


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
