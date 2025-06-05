from dataclasses import dataclass, field

@dataclass
class EnhancedRewardConfig:
    """
    Enhanced configuration for reward values used in GRPO with multi-objective optimization.
    """
    # Basic compilation rewards
    compilation_success: float = field(default=2.0, metadata={"help": "Base reward for successful compilation. Increased from 1.0."})
    compilation_failure: float = field(default=-8.0, metadata={"help": "Penalty for compilation failure. Increased magnitude."})
    simulation_crash: float = field(default=-4.0, metadata={"help": "Penalty for simulation crash. Increased magnitude."})
    output_parse_error: float = field(default=-2.0, metadata={"help": "Penalty for output parsing error."})
    missing_code_block_penalty: float = field(default=-6.0, metadata={"help": "Penalty if the generated code block is missing."})

    # Enhanced functional correctness rewards (non-linear)
    test_pass_base_reward: float = field(default=1.5, metadata={"help": "Base reward per passed test case."})
    test_pass_bonus_multiplier: float = field(default=1.3, metadata={"help": "Multiplier for consecutive test passes (exponential bonus)."})
    max_functional_reward: float = field(default=15.0, metadata={"help": "Max reward for functional correctness. Increased from 10.0."})
    all_tests_passed_bonus: float = field(default=5.0, metadata={"help": "Bonus if all test cases pass. Increased from 2.0."})

    # Code quality rewards
    code_efficiency_bonus: float = field(default=2.0, metadata={"help": "Bonus for efficient code (low complexity, good structure)."})
    code_readability_bonus: float = field(default=1.0, metadata={"help": "Bonus for readable, well-structured code."})
    code_complexity_penalty: float = field(default=-1.0, metadata={"help": "Penalty for overly complex code."})

    # Robustness rewards
    edge_case_handling_bonus: float = field(default=1.5, metadata={"help": "Bonus for handling edge cases correctly."})
    synthesis_friendly_bonus: float = field(default=1.0, metadata={"help": "Bonus for synthesis-friendly code."})

    # Penalty configurations
    timeout_penalty: float = field(default=-3.0, metadata={"help": "Penalty for timeout during simulation."})
    resource_usage_penalty: float = field(default=-0.5, metadata={"help": "Penalty for excessive resource usage."})

    # Multi-objective weights
    functional_weight: float = field(default=0.6, metadata={"help": "Weight for functional correctness in total reward."})
    efficiency_weight: float = field(default=0.2, metadata={"help": "Weight for code efficiency in total reward."})
    readability_weight: float = field(default=0.1, metadata={"help": "Weight for code readability in total reward."})
    robustness_weight: float = field(default=0.1, metadata={"help": "Weight for code robustness in total reward."})

    # Dynamic reward scaling
    enable_adaptive_scaling: bool = field(default=True, metadata={"help": "Enable adaptive reward scaling based on training progress."})
    reward_scale_factor: float = field(default=1.0, metadata={"help": "Global reward scale factor for fine-tuning."})
    reward_clipping_range: float = field(default=20.0, metadata={"help": "Clip rewards to [-range, +range] to prevent instability."})

    def get_scaled_reward(self, base_reward: float, training_step: int = 0) -> float:
        """Apply adaptive scaling and clipping to rewards."""
        scaled_reward = base_reward * self.reward_scale_factor

        if self.enable_adaptive_scaling and training_step > 0:
            # Gradually increase reward sensitivity as training progresses
            adaptive_factor = min(1.0 + (training_step / 5000) * 0.1, 1.5)
            scaled_reward *= adaptive_factor

        # Clip to prevent instability
        return max(-self.reward_clipping_range, min(self.reward_clipping_range, scaled_reward))

# Backward compatibility alias
RewardConfig = EnhancedRewardConfig
