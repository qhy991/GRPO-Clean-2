from .calculator import RewardCalculator
from .components import FunctionalRewardComponent, CodeQualityRewardComponent

# Re-export key config classes from the central config location for convenience,
# so users of the 'rewards' module can easily access its specific configs.
try:
    from grpo_project.configs.reward import EnhancedRewardConfig, RewardConfig
except ImportError:
    # Define placeholders if configs are not found, to allow linting/type-checking
    # This should not happen in a correctly structured project.
    class EnhancedRewardConfig: pass # type: ignore
    class RewardConfig: pass # type: ignore


__all__ = [
    "RewardCalculator",
    "FunctionalRewardComponent",
    "CodeQualityRewardComponent",
    "EnhancedRewardConfig", # Re-exported
    "RewardConfig"        # Re-exported
]
