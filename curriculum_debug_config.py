# enhanced_curriculum.py 的修复版本 - 解决课程学习不生效问题

import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
from dataclasses import dataclass
import wandb
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStageConfig:
    """课程学习阶段配置"""
    name: str
    dataset_levels: List[str]
    complexity_range: Tuple[float, float]
    epochs_ratio: float
    performance_threshold: float = 0.6
    min_evaluations: int = 5
    description: str = ""

class FixedEnhancedCurriculumManager:
    """修复版本的增强课程学习管理器"""
    
    def __init__(self, curriculum_stages: List[CurriculumStageConfig], dataset: Dataset):
        self.curriculum_stages = curriculum_stages
        self.full_dataset = dataset
        self.current_stage = 0
        self.stage_performance_history = []  # 当前阶段的性能历史
        self.all_stage_history = []  # 所有阶段的完整历史
        self.stage_statistics = []
        
        # 创建详细的调试日志
        self.debug_log = []
        self._log_debug(f"课程管理器初始化 - 总阶段数: {len(curriculum_stages)}")
        
        # 分析数据集分布
        self._analyze_dataset_distribution()
        
        # 验证课程设计
        self._validate_curriculum_design()
        
        # 初始日志
        for i, stage in enumerate(curriculum_stages):
            self._log_debug(f"阶段{i}: {stage.name} | 等级: {stage.dataset_levels} | 复杂度: {stage.complexity_range} | 阈值: {stage.performance_threshold}")
        
        # 验证当前阶段数据集
        current_dataset = self.get_current_stage_dataset()
        self._log_debug(f"当前阶段({self.current_stage})数据集大小: {len(current_dataset)}")
    
    def _log_debug(self, message: str):
        """记录调试信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_entry = f"[{timestamp}] {message}"
        self.debug_log.append(debug_entry)
        logger.info(f"📚 CURRICULUM: {message}")
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """获取课程学习管理器的当前状态，用于保存"""
        return {
            "current_stage": self.current_stage,
            "stage_performance_history": self.stage_performance_history,
            "all_stage_history": self.all_stage_history,
            "debug_log": self.debug_log[-50:],  # 保留最近50条日志
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
                'debug_log_recent': self.debug_log[-10:]  # 最近10条日志
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
            'debug_log_recent': self.debug_log[-10:]  # 最近10条日志
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

    def log_to_wandb(self, step: int):
        """记录到W&B（使用数值而非文字）"""
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
        
        # 数据集统计
        if hasattr(self, 'dataset_distribution'):
            total_samples = self.dataset_distribution.get('total_samples', 0)
            if total_samples > 0:
                wandb_data['curriculum/total_dataset_size'] = total_samples
                
                # 各等级样本数量
                level_counts = self.dataset_distribution.get('level_counts', {})
                for level, count in level_counts.items():
                    wandb_data[f'curriculum/level_{level}_count'] = count
                    wandb_data[f'curriculum/level_{level}_ratio'] = count / total_samples
        
        # 当前阶段数据集大小
        try:
            current_dataset = self.get_current_stage_dataset()
            wandb_data['curriculum/current_stage_dataset_size'] = len(current_dataset)
            if hasattr(self, 'dataset_distribution') and self.dataset_distribution.get('total_samples', 0) > 0:
                wandb_data['curriculum/current_stage_dataset_ratio'] = len(current_dataset) / self.dataset_distribution['total_samples']
        except Exception as e:
            self._log_debug(f"获取当前数据集大小失败: {e}")
        
        wandb.log(wandb_data, step=step)
        self._log_debug(f"已记录W&B指标 (步数: {step})")

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


# 创建修复版本的默认课程阶段
def create_fixed_curriculum_stages() -> List[CurriculumStageConfig]:
    """创建修复版本的默认课程学习阶段"""
    stages = [
        CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=(0.0, 3.5),
            epochs_ratio=0.25,
            performance_threshold=0.65,  # 降低阈值，更容易进阶
            min_evaluations=3,  # 减少最小评估次数
            description="基础阶段：最简单的基础级设计"
        ),
        CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=(0.0, 5.5),
            epochs_ratio=0.25,
            performance_threshold=0.6,
            min_evaluations=3,
            description="初级阶段：基础级+简单中级设计"
        ),
        CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=(2.0, 7.5),
            epochs_ratio=0.25,
            performance_threshold=0.55,
            min_evaluations=4,
            description="中级阶段：中等复杂度的中级设计"
        ),
        CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=(4.0, 9.0),
            epochs_ratio=0.15,
            performance_threshold=0.5,
            min_evaluations=4,
            description="高级阶段：复杂的中级和高级设计"
        ),
        CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=(6.0, 10.0),
            epochs_ratio=0.1,
            performance_threshold=0.45,  # 最低阈值
            min_evaluations=3,
            description="专家阶段：最复杂的高级和专家级设计"
        )
    ]
    return stages


# 修复setup_curriculum_manager函数
def setup_fixed_curriculum_manager(script_cfg, dataset: Dataset) -> Optional[FixedEnhancedCurriculumManager]:
    """设置修复版本的课程管理器"""
    if not script_cfg.enable_curriculum:
        logger.info("📚 课程学习已禁用")
        return None
    
    if dataset is None or len(dataset) == 0:
        logger.error("❌ 无法使用空数据集设置课程管理器")
        return None
    
    logger.info("🔧 开始设置修复版本的课程学习管理器...")
    
    # 检查数据集是否包含等级信息
    has_level_info = False
    has_complexity_info = False
    
    if len(dataset) > 0:
        first_example = dataset[0]
        has_level_info = 'level' in first_example and first_example['level'] is not None
        has_complexity_info = 'complexity_score' in first_example and first_example['complexity_score'] is not None
        
        logger.info(f"📊 数据集信息检查:")
        logger.info(f"  包含等级信息: {has_level_info}")
        logger.info(f"  包含复杂度信息: {has_complexity_info}")
    
    # 根据配置创建课程阶段
    curriculum_stages_config_list = []
    
    if script_cfg.curriculum_type == "dual_layer" and has_level_info and has_complexity_info:
        # 使用配置文件中的详细阶段定义
        if hasattr(script_cfg, 'curriculum_stages') and isinstance(script_cfg.curriculum_stages, list):
            for stage_dict in script_cfg.curriculum_stages:
                if isinstance(stage_dict, dict):
                    stage_config = CurriculumStageConfig(
                        name=stage_dict.get("name", "Unnamed Stage"),
                        dataset_levels=stage_dict.get("dataset_levels", []),
                        complexity_range=tuple(stage_dict.get("complexity_range", (0, 10))),
                        epochs_ratio=stage_dict.get("epochs_ratio", 0.2),
                        performance_threshold=stage_dict.get("performance_threshold", 0.6),
                        min_evaluations=stage_dict.get("min_evaluations", 5),
                        description=stage_dict.get("description", "")
                    )
                    curriculum_stages_config_list.append(stage_config)
            logger.info(f"✅ 使用配置文件中的{len(curriculum_stages_config_list)}个课程阶段")
        else:
            # 使用修复版本的默认阶段
            curriculum_stages_config_list = create_fixed_curriculum_stages()
            logger.info("✅ 使用修复版本的默认双层课程阶段")
    
    elif script_cfg.curriculum_type == "level_only" and has_level_info:
        curriculum_stages_config_list = [
            CurriculumStageConfig("basic_only", ["basic"], (0, 10), 0.3, 0.65, 3),
            CurriculumStageConfig("basic_intermediate", ["basic", "intermediate"], (0, 10), 0.3, 0.6, 3),
            CurriculumStageConfig("intermediate_advanced", ["intermediate", "advanced"], (0, 10), 0.3, 0.55, 4),
            CurriculumStageConfig("all_levels", ["basic", "intermediate", "advanced", "expert"], (0, 10), 0.1, 0.5, 3)
        ]
        logger.info("✅ 使用等级优先课程阶段")
    
    else:
        # 复杂度优先模式（回退模式）
        curriculum_stages_config_list = [
            CurriculumStageConfig("simple", ["basic", "intermediate", "advanced", "expert"], (0, 4), 0.3, 0.65, 3),
            CurriculumStageConfig("moderate", ["basic", "intermediate", "advanced", "expert"], (0, 7), 0.4, 0.6, 3),
            CurriculumStageConfig("complex", ["basic", "intermediate", "advanced", "expert"], (0, 10), 0.3, 0.55, 4)
        ]
        logger.info("✅ 使用复杂度优先课程阶段（回退模式）")
    
    if not curriculum_stages_config_list:
        logger.error("❌ 没有定义课程阶段，禁用课程学习")
        return None
    
    # 创建修复版本的课程管理器
    try:
        curriculum_manager = FixedEnhancedCurriculumManager(curriculum_stages_config_list, dataset)
        logger.info("✅ 修复版本课程管理器创建成功")
        
        # 验证初始状态
        current_info = curriculum_manager.get_current_stage_info()
        logger.info(f"📊 初始课程状态:")
        logger.info(f"  当前阶段: {current_info['stage_index']} ({current_info['stage_name']})")
        logger.info(f"  等级: {current_info['dataset_levels']}")
        logger.info(f"  复杂度范围: {current_info['complexity_range']}")
        logger.info(f"  性能阈值: {current_info['performance_threshold']}")
        
        # 验证当前阶段数据集
        current_dataset = curriculum_manager.get_current_stage_dataset()
        logger.info(f"  当前阶段数据集大小: {len(current_dataset)}/{len(dataset)} ({len(current_dataset)/len(dataset)*100:.1f}%)")
        
        return curriculum_manager
        
    except Exception as e:
        logger.error(f"❌ 创建课程管理器失败: {e}", exc_info=True)
        return None


# 在train.py中的集成示例
def integrate_fixed_curriculum_in_train():
    """在train.py中集成修复版本课程学习的示例"""
    return """
# 在train.py的main函数中替换原有的课程管理器设置:

# 1. 替换curriculum manager设置
curriculum_manager = setup_fixed_curriculum_manager(script_cfg, dataset)

# 2. 添加详细的课程状态监控
if curriculum_manager:
    logger.info("🎯 课程学习详细状态:")
    progress_info = curriculum_manager.get_curriculum_progress()
    logger.info(f"  数据集分布: {progress_info['dataset_distribution']}")
    
    # 保存初始状态
    curriculum_manager.save_detailed_log(script_cfg.output_dir)

# 3. 在训练循环中添加更频繁的课程检查
class EnhancedCurriculumCallback(TrainerCallback):
    def __init__(self, curriculum_manager, output_dir):
        self.curriculum_manager = curriculum_manager
        self.output_dir = output_dir
        self.check_frequency = 10  # 每10步检查一次
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.curriculum_manager or logs is None:
            return
        
        current_step = getattr(state, 'global_step', 0) or 0
        
        # 更频繁的检查
        if current_step % self.check_frequency == 0:
            current_loss = logs.get('train_loss', logs.get('loss', float('inf')))
            if current_loss != float('inf'):
                performance_estimate = max(0, 1.0 - (current_loss / 10.0))
                
                # 检查是否应该进阶
                should_advance = self.curriculum_manager.should_advance_stage(performance_estimate)
                
                if should_advance:
                    old_stage = self.curriculum_manager.current_stage
                    success = self.curriculum_manager.advance_stage()
                    
                    if success:
                        new_stage = self.curriculum_manager.current_stage
                        logger.info(f"🎯 课程进阶成功: {old_stage} -> {new_stage}")
                        
                        # 保存进阶状态
                        self.curriculum_manager.save_detailed_log(self.output_dir)
                        
                        # 更新训练器数据集
                        if hasattr(self, 'trainer_ref') and self.trainer_ref:
                            try:
                                new_dataset = self.curriculum_manager.get_current_stage_dataset()
                                self.trainer_ref.train_dataset = new_dataset
                                logger.info(f"📊 数据集已更新: {len(new_dataset)} 样本")
                            except Exception as e:
                                logger.error(f"更新数据集失败: {e}")
                
                # 记录到W&B
                self.curriculum_manager.log_to_wandb(current_step)
        
        # 每100步保存详细日志
        if current_step % 100 == 0:
            self.curriculum_manager.save_detailed_log(self.output_dir)

# 4. 使用增强的课程回调
enhanced_curriculum_cb = EnhancedCurriculumCallback(curriculum_manager, script_cfg.output_dir)
callbacks_list.append(enhanced_curriculum_cb)
"""