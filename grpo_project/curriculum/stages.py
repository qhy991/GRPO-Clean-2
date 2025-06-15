import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np # Used in create_custom_curriculum_stages if dataset_distribution analysis is complex

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStageConfig:
    """课程学习阶段配置"""
    name: str
    dataset_levels: List[str]           # 数据集等级过滤 ['basic', 'intermediate', 'advanced', 'expert']
    complexity_range: Tuple[float, float]  # 复杂度范围 (min, max)
    epochs_ratio: float                 # 该阶段训练epoch比例 # This might be better named as steps_ratio or similar if not strictly epochs
    performance_threshold: float = 0.6  # 进入下一阶段的性能阈值
    min_evaluations: int = 5           # 最少评估次数
    description: str = ""              # 阶段描述

def create_default_curriculum_stages(
    performance_thresholds: Optional[List[float]] = None,
    min_evaluations: int = 5
) -> List[CurriculumStageConfig]:
    """创建默认的双层课程学习阶段
    
    Args:
        performance_thresholds: 五个阶段的性能阈值列表 [foundation, elementary, intermediate, advanced, expert]
        min_evaluations: 最小评估次数
    """
    # 🔧 优先从环境变量读取阈值
    env_thresholds = []
    for i in range(1, 6):  # 阶段1-5
        env_key = f"CURRICULUM_PERFORMANCE_THRESHOLD_{i}"
        env_value = os.environ.get(env_key)
        if env_value:
            try:
                env_thresholds.append(float(env_value))
                logger.info(f"📊 从环境变量读取阈值: {env_key}={env_value}")
            except ValueError:
                logger.warning(f"⚠️ 环境变量 {env_key} 值无效: {env_value}, 忽略")
    
    # 从环境变量读取最小评估次数
    env_min_eval = os.environ.get("CURRICULUM_MIN_EVALUATIONS")
    if env_min_eval:
        try:
            min_evaluations = int(env_min_eval)
            logger.info(f"📊 从环境变量读取最小评估次数: CURRICULUM_MIN_EVALUATIONS={env_min_eval}")
        except ValueError:
            logger.warning(f"⚠️ 环境变量 CURRICULUM_MIN_EVALUATIONS 值无效: {env_min_eval}, 使用默认值")
    
    # 优先级：环境变量 > 函数参数 > 默认值
    if env_thresholds and len(env_thresholds) >= 5:
        performance_thresholds = env_thresholds[:5]
        logger.info(f"✅ 使用环境变量中的性能阈值: {performance_thresholds}")
    elif performance_thresholds is None:
        performance_thresholds = [0.65, 0.60, 0.55, 0.50, 0.45]
        logger.info(f"📊 使用默认性能阈值: {performance_thresholds}")
    else:
        logger.info(f"📊 使用函数参数中的性能阈值: {performance_thresholds}")
    
    # 确保有足够的阈值
    if len(performance_thresholds) < 5:
        logger.warning(f"提供的阈值数量不足({len(performance_thresholds)})，使用默认值补充")
        default_thresholds = [0.65, 0.60, 0.55, 0.50, 0.45]
        performance_thresholds.extend(default_thresholds[len(performance_thresholds):])
    
    stages = [
        CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=(0.0, 3.0),
            epochs_ratio=0.25,
            performance_threshold=performance_thresholds[0],
            min_evaluations=min_evaluations,
            description="基础阶段：学习简单的基础级设计"
        ),
        CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=(0.0, 5.0),
            epochs_ratio=0.25,
            performance_threshold=performance_thresholds[1],
            min_evaluations=min_evaluations,
            description="初级阶段：基础级+简单中级设计"
        ),
        CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=(3.0, 7.0),
            epochs_ratio=0.25,
            performance_threshold=performance_thresholds[2],
            min_evaluations=min_evaluations,
            description="中级阶段：中等复杂度的中级设计"
        ),
        CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=(5.0, 9.0),
            epochs_ratio=0.15,
            performance_threshold=performance_thresholds[3],
            min_evaluations=min_evaluations,
            description="高级阶段：复杂的中级和高级设计"
        ),
        CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=(7.0, 10.0),
            epochs_ratio=0.1,
            performance_threshold=performance_thresholds[4],
            min_evaluations=min_evaluations,
            description="专家阶段：最复杂的高级和专家级设计"
        )
    ]
    
    logger.info(f"创建了 {len(stages)} 个课程阶段，性能阈值: {performance_thresholds[:5]}")
    return stages

def create_custom_curriculum_stages(
    dataset_distribution: Dict[str, Any], # Example: {'level_counts': {...}, 'complexity_by_level': {...}, 'total_samples': X}
    focus_levels: Optional[List[str]] = None,
    complexity_emphasis: str = "balanced"  # "simple", "balanced", "complex"
) -> List[CurriculumStageConfig]:
    """根据数据集分布创建自定义课程阶段"""

    if focus_levels is None:
        focus_levels = ["basic", "intermediate", "advanced", "expert"]

    # Default complexity ranges, can be adjusted based on dataset_distribution if needed
    if complexity_emphasis == "simple":
        complexity_ranges = [(0, 3), (0, 4), (2, 6), (4, 8), (6, 10)]
    elif complexity_emphasis == "complex":
        complexity_ranges = [(0, 4), (2, 6), (4, 8), (6, 10), (8, 10)]
    else:  # balanced
        complexity_ranges = [(0, 3), (0, 5), (3, 7), (5, 9), (7, 10)]

    stages = []

    # This is a simplified creation logic, can be made more data-driven using dataset_distribution
    if "basic" in focus_levels:
        stages.append(CurriculumStageConfig(
            name="foundation_custom",
            dataset_levels=["basic"],
            complexity_range=complexity_ranges[0],
            epochs_ratio=0.3,
            performance_threshold=0.65,
            min_evaluations=5,
            description="Custom: 基础阶段"
        ))

    if "intermediate" in focus_levels:
        levels = ["basic", "intermediate"] if "basic" in focus_levels else ["intermediate"]
        stages.append(CurriculumStageConfig(
            name="elementary_to_intermediate_custom",
            dataset_levels=levels,
            complexity_range=complexity_ranges[1 if "basic" in focus_levels else 2],
            epochs_ratio=0.25,
            performance_threshold=0.65,
            min_evaluations=5,
            description="Custom: 初级到中级过渡"
        ))
        if "basic" not in focus_levels: # If only intermediate, add a dedicated intermediate stage
             stages.append(CurriculumStageConfig(
                name="intermediate_focus_custom",
                dataset_levels=["intermediate"],
                complexity_range=complexity_ranges[2],
                epochs_ratio=0.20,
                performance_threshold=0.6,
                min_evaluations=5,
                description="Custom: 中级核心"
            ))


    if "advanced" in focus_levels:
        levels = ["intermediate", "advanced"] if "intermediate" in focus_levels else ["advanced"]
        stages.append(CurriculumStageConfig(
            name="advanced_custom",
            dataset_levels=levels,
            complexity_range=complexity_ranges[3],
            epochs_ratio=0.15,
            performance_threshold=0.55,
            min_evaluations=5,
            description="Custom: 高级阶段"
        ))

    if "expert" in focus_levels:
        levels = ["advanced", "expert"] if "advanced" in focus_levels else ["expert"]
        stages.append(CurriculumStageConfig(
            name="expert_custom",
            dataset_levels=levels,
            complexity_range=complexity_ranges[4],
            epochs_ratio=0.10,
            performance_threshold=0.5,
            min_evaluations=5,
            description="Custom: 专家阶段"
        ))

    if not stages:
        logger.warning("No custom stages generated based on focus_levels. Falling back to a single default stage.")
        stages.append(CurriculumStageConfig(
            name="default_full_range_custom",
            dataset_levels=focus_levels if focus_levels else ["basic", "intermediate", "advanced", "expert"],
            complexity_range=(0.0, 10.0),
            epochs_ratio=1.0,
            performance_threshold=0.6,
            min_evaluations=5,
            description="Default stage covering all specified levels and full complexity range."
        ))

    # Normalize epoch_ratios
    total_ratio = sum(stage.epochs_ratio for stage in stages)
    if total_ratio > 0 :
        for stage in stages:
            stage.epochs_ratio /= total_ratio
    else: # Avoid division by zero if all ratios are 0
        equal_ratio = 1.0 / len(stages) if stages else 1.0
        for stage in stages:
            stage.epochs_ratio = equal_ratio

    logger.info(f"Created {len(stages)} custom curriculum stages.")
    return stages
