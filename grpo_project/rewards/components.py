import logging
from typing import Dict, Any, Optional

# Attempt to import from grpo_project, fallback to local if not found
try:
    from grpo_project.utils.verilog_utils import assess_code_quality
    from grpo_project.configs.reward import EnhancedRewardConfig
    from grpo_project.evaluation.simulator import VerilogSimulator # Import VerilogSimulator
    # VerilogOutputParser might not be directly needed here if VerilogSimulator handles parsing
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import from grpo_project.utils or grpo_project.configs. Using placeholders.")
    # Placeholder for assess_code_quality
    def assess_code_quality(code: str) -> Dict[str, float]:
        logger.warning("Using placeholder assess_code_quality")
        return {"complexity": 0.5, "readability": 0.5, "efficiency": 0.5, "structure": 0.5}
    # Placeholder for EnhancedRewardConfig if not available
    class EnhancedRewardConfig: # type: ignore
        compilation_failure = -5.0
        code_efficiency_bonus = 1.0
        code_readability_bonus = 1.0
        code_complexity_penalty = -1.0
        synthesis_friendly_bonus = 1.0


logger = logging.getLogger(__name__)

class FunctionalRewardComponent:
    def __init__(self, reward_config: EnhancedRewardConfig, simulator: VerilogSimulator):
        self.reward_config = reward_config
        self.simulator = simulator
        logger.info("FunctionalRewardComponent initialized with VerilogSimulator.")

    def calculate(self, generated_code: str, testbench_path: str, expected_tests: int, prompt_identifier: str, completion_idx: int) -> Dict[str, Any]:
        logger.debug(f"FunctionalRewardComponent: Calculating for {prompt_identifier}, completion {completion_idx}")

        sim_results = self.simulator.run_simulation(
            generated_verilog_code=generated_code,
            testbench_file_path=testbench_path,
            expected_total_tests_from_manifest=expected_tests, # Passed for logging/comparison
            prompt_identifier=prompt_identifier,
            completion_idx=completion_idx,
            print_simulation_details=logger.isEnabledFor(logging.DEBUG) # Control verbosity
        )

        functional_reward = 0.0
        # Logic based on sim_results and self.reward_config
        if not sim_results.get("compilation_success"):
            functional_reward = self.reward_config.compilation_failure
        elif not sim_results.get("simulation_run_success"):
            functional_reward = self.reward_config.simulation_crash # Or a specific penalty for sim crash
        elif not sim_results.get("parsing_success"):
            # This case implies simulation ran but output parsing failed
            functional_reward = self.reward_config.output_parse_error
        else:
            # Simulation and parsing were successful, calculate reward based on test results
            passed = sim_results.get("passed_tests", 0)
            total_in_output = sim_results.get("total_tests_in_output", 0)
            all_passed_tb = sim_results.get("all_tests_passed_by_tb", False)

            if total_in_output > 0:
                pass_ratio = passed / total_in_output
                base_functional = pass_ratio * self.reward_config.max_functional_reward
                if passed > 1: # Apply bonus for multiple passes
                    base_functional *= self.reward_config.test_pass_bonus_multiplier ** (passed - 1)
                functional_reward += base_functional
            elif all_passed_tb : # No tests in output, but TB says PASS (e.g. simple design)
                 functional_reward += self.reward_config.max_functional_reward # Give max if TB implies full pass

            if all_passed_tb and passed == total_in_output : # Consider all_tests_passed_bonus
                functional_reward += self.reward_config.all_tests_passed_bonus

            # Consider edge_case_handling_bonus if applicable (e.g. many tests passed)
            if passed == total_in_output and total_in_output >= 5: # Arbitrary condition
                functional_reward += self.reward_config.edge_case_handling_bonus

        # Ensure this key matches what RewardCalculator expects for "sim_details"
        return {"functional_reward": functional_reward, "sim_details": sim_results}


class CodeQualityRewardComponent:
    def __init__(self, reward_config: EnhancedRewardConfig):
        self.reward_config = reward_config
        logger.info("CodeQualityRewardComponent initialized.")

    def calculate(self, code: str) -> Dict[str, float]:
        if not code or not code.strip():
            return {"efficiency_score": 0.0, "readability_score": 0.0, "complexity_penalty_score": 0.0, "synthesis_bonus_score": 0.0}

        quality_metrics = assess_code_quality(code)

        efficiency_score = quality_metrics.get("efficiency", 0.0) * self.reward_config.code_efficiency_bonus
        readability_score = quality_metrics.get("readability", 0.0) * self.reward_config.code_readability_bonus

        # Ensure complexity penalty is negative or zero if complexity is good (score near 1)
        # A high complexity metric (e.g., 0.2 for very complex) should result in a penalty.
        # A low complexity metric (e.g., 0.8 for simple) should result in minimal or no penalty.
        # The existing formula `max(0, (1 - quality_metrics.get("complexity", 1)) * self.reward_config.code_complexity_penalty)`
        # assumes reward_config.code_complexity_penalty is a positive value representing the max penalty.
        # If quality_metrics.get("complexity") is low (e.g. 0.2, very complex), (1-0.2) = 0.8. Penalty = 0.8 * max_penalty
        # If quality_metrics.get("complexity") is high (e.g. 0.9, simple), (1-0.9) = 0.1. Penalty = 0.1 * max_penalty
        # This seems correct if code_complexity_penalty in config is positive.
        complexity_raw_score = quality_metrics.get("complexity", 1.0) # Default to 1 (no penalty) if not found
        # Penalty should increase as raw score decreases (more complex)
        # Let's assume reward_config.code_complexity_penalty is the amount to penalize for max complexity (raw_score = 0)
        # And 0 penalty for min complexity (raw_score = 1)
        complexity_penalty_applied = (1.0 - complexity_raw_score) * self.reward_config.code_complexity_penalty
        # This value should then be subtracted or added if penalty is negative in config.
        # For clarity, let's return it as a positive penalty amount here.

        synthesis_bonus_score = quality_metrics.get("structure", 0.0) * self.reward_config.synthesis_friendly_bonus

        return {
            "efficiency_score": efficiency_score,
            "readability_score": readability_score,
            "complexity_penalty_applied": complexity_penalty_applied, # This is the calculated penalty value
            "synthesis_bonus_score": synthesis_bonus_score
        }
