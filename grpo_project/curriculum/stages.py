import logging
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

def create_default_curriculum_stages() -> List[CurriculumStageConfig]:
    """创建默认的双层课程学习阶段"""
    stages = [
        CurriculumStageConfig(
            name="foundation",
            dataset_levels=["basic"],
            complexity_range=(0.0, 3.0),
            epochs_ratio=0.25,
            performance_threshold=0.7,
            min_evaluations=10,
            description="基础阶段：学习简单的基础级设计"
        ),
        CurriculumStageConfig(
            name="elementary",
            dataset_levels=["basic", "intermediate"],
            complexity_range=(0.0, 5.0),
            epochs_ratio=0.25,
            performance_threshold=0.65,
            min_evaluations=10,
            description="初级阶段：基础级+简单中级设计"
        ),
        CurriculumStageConfig(
            name="intermediate",
            dataset_levels=["intermediate"],
            complexity_range=(3.0, 7.0),
            epochs_ratio=0.25,
            performance_threshold=0.6,
            min_evaluations=10,
            description="中级阶段：中等复杂度的中级设计"
        ),
        CurriculumStageConfig(
            name="advanced",
            dataset_levels=["intermediate", "advanced"],
            complexity_range=(5.0, 9.0),
            epochs_ratio=0.15,
            performance_threshold=0.55,
            min_evaluations=10,
            description="高级阶段：复杂的中级和高级设计"
        ),
        CurriculumStageConfig(
            name="expert",
            dataset_levels=["advanced", "expert"],
            complexity_range=(7.0, 10.0),
            epochs_ratio=0.1,
            performance_threshold=0.5,
            min_evaluations=10,
            description="专家阶段：最复杂的高级和专家级设计"
        )
    ]
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
            performance_threshold=0.7,
            min_evaluations=10,
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
            min_evaluations=10,
            description="Custom: 初级到中级过渡"
        ))
        if "basic" not in focus_levels: # If only intermediate, add a dedicated intermediate stage
             stages.append(CurriculumStageConfig(
                name="intermediate_focus_custom",
                dataset_levels=["intermediate"],
                complexity_range=complexity_ranges[2],
                epochs_ratio=0.20,
                performance_threshold=0.6,
                min_evaluations=10,
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
            min_evaluations=10,
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
            min_evaluations=10,
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
            min_evaluations=10,
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
