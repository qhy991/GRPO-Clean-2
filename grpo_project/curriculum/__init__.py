from .stages import CurriculumStageConfig, create_default_curriculum_stages, create_custom_curriculum_stages
from .manager import (
    EnhancedCurriculumManager,
    FixedEnhancedCurriculumManager, # Assuming this is the one primarily used or also needed
    setup_enhanced_curriculum_manager, # General setup function
    setup_fixed_curriculum_manager # Setup for the fixed manager
)
from .callbacks import (
    CurriculumProgressCallback,
    OptimizedCurriculumCallback,
    EnhancedCurriculumDebugCallback
)

__all__ = [
    "CurriculumStageConfig",
    "create_default_curriculum_stages",
    "create_custom_curriculum_stages",
    "EnhancedCurriculumManager",
    "FixedEnhancedCurriculumManager",
    "setup_enhanced_curriculum_manager", # Exposing this setup function
    "setup_fixed_curriculum_manager",    # Exposing this setup function
    "CurriculumProgressCallback",
    "OptimizedCurriculumCallback",
    "EnhancedCurriculumDebugCallback"
]
