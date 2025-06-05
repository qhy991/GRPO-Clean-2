import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Attempt to import from grpo_project, fallback to local if not found
try:
    from grpo_project.configs.reward import EnhancedRewardConfig
    from grpo_project.utils.file_ops import extract_module_info
    from grpo_project.utils.parsing import parse_llm_completion_qwen3
    from grpo_project.utils.verilog_utils import validate_verilog_code # assess_code_quality is used by CodeQualityRewardComponent
    # from grpo_project.utils.simulation import run_iverilog_simulation # This will be removed
    from grpo_project.evaluation.simulator import VerilogSimulator # Import the new simulator
    from .components import FunctionalRewardComponent, CodeQualityRewardComponent
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import from grpo_project. Using placeholders for some types.")
    # Define placeholders for missing types if necessary for standalone testing
    class EnhancedRewardConfig: # type: ignore
        compilation_failure = -8.0
        compilation_success = 2.0
        simulation_crash = -4.0
        output_parse_error = -2.0
        missing_code_block_penalty = -6.0
        test_pass_base_reward = 1.5
        test_pass_bonus_multiplier = 1.3
        max_functional_reward = 15.0
        all_tests_passed_bonus = 5.0
        edge_case_handling_bonus = 1.5
        functional_weight = 0.6
        efficiency_weight = 0.2
        readability_weight = 0.1
        robustness_weight = 0.1
        # Default get_scaled_reward
        def get_scaled_reward(self, base_reward: float, training_step: int = 0) -> float: return base_reward

    def extract_module_info(verilog_file: str) -> tuple[str, list[str]]: return "placeholder_module", []
    def parse_llm_completion_qwen3(text: str, debug_prompt: Optional[str]=None, debug_context:Optional[Dict[str,Any]]=None) -> tuple[Optional[str], Optional[str]]: return "parsed_reasoning", "parsed_code"
    def validate_verilog_code(code: str, name: str, ports: list) -> tuple[bool, str]: return True, ""
    def assess_code_quality(code: str) -> Dict[str, float]: return {}
    def run_iverilog_simulation(*args, **kwargs) -> Dict[str, Any]:
        return {"compilation_success": False, "error_message": "Placeholder sim error"}
    class FunctionalRewardComponent: # type: ignore
        def __init__(self, reward_config, simulator): pass
        def calculate(self, *args, **kwargs) -> Dict[str, Any]: return {"functional_reward": 0.0, "sim_details": {}}
    class CodeQualityRewardComponent: # type: ignore
        def __init__(self, reward_config): pass
        def calculate(self, code: str) -> Dict[str, float]: return {}


logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, reward_config: EnhancedRewardConfig, simulator: Optional[Any] = None): # Simulator can be passed later
        self.reward_config = reward_config
        # If simulator is not passed, instantiate a default one.
        self.simulator = simulator if simulator else VerilogSimulator()
        self.functional_component = FunctionalRewardComponent(reward_config, self.simulator)
        self.quality_component = CodeQualityRewardComponent(reward_config)
        logger.info("RewardCalculator initialized.")

    def _calculate_single_reward(
        self,
        prompt_str: str,
        completion_str: str,
        testbench_path: str,
        expected_total_tests: int,
        reference_verilog_path: str,
        training_step: int = 0,
        # wandb_callback: Optional[Any] = None, # wandb_callback from train.py's scope is not directly passed here
        output_dir_for_debug: Optional[str] = None,
        completion_idx: int = 0 # Added for consistency with original function
    ) -> Dict[str, Any]:

        prompt_id_base = prompt_str.split('\n', 1)[0]
        name_match_for_id = re.search(r"module MUST be named `(\w+)`", prompt_str, re.IGNORECASE)
        if name_match_for_id:
            prompt_id_base = f"Mod_{name_match_for_id.group(1)}"

        sanitized_prompt_id_for_file = re.sub(r'[^\w_.)( -]', '', prompt_id_base).strip().replace(' ', '_')[:50]
        if not sanitized_prompt_id_for_file:
            sanitized_prompt_id_for_file = "unknown_prompt"
        prompt_id_for_log = prompt_id_base[:70]

        log_pref = f"REWARD_CALC: '{prompt_id_for_log}', Completion {completion_idx}"

        current_unscaled_components = {"functional": 0.0, "efficiency": 0.0, "readability": 0.0, "robustness": 0.0, "base_compilation": 0.0}
        current_funnel_metrics = {"code_extracted": False, "compiled_successfully": False, "sim_ran_successfully": False, "passed_tests": -1}

        module_name, req_ports = "", []
        if reference_verilog_path and os.path.exists(reference_verilog_path):
            module_name, req_ports = extract_module_info(reference_verilog_path)

        if not module_name:
            logger.error(f"{log_pref}: Failed to extract module info from ref Verilog '{reference_verilog_path}'.")
            error_reward_val = self.reward_config.get_scaled_reward(self.reward_config.compilation_failure * 2, training_step)
            return {
                "final_reward": error_reward_val,
                "unscaled_components": {**current_unscaled_components, "base_compilation": self.reward_config.compilation_failure * 2},
                "funnel_metrics": current_funnel_metrics
            }

        _, code = parse_llm_completion_qwen3(completion_str, debug_prompt=prompt_str, debug_context={"step": training_step, "sample_idx": completion_idx})

        if code and code.strip():
            current_funnel_metrics["code_extracted"] = True
        else:
            penalty_type = self.reward_config.missing_code_block_penalty
            current_unscaled_components["base_compilation"] = penalty_type
            total_reward = self.reward_config.get_scaled_reward(penalty_type, training_step)
            # Debug saving logic (optional, can be added if needed based on output_dir_for_debug)
            return {
                "final_reward": total_reward,
                "unscaled_components": current_unscaled_components,
                "funnel_metrics": current_funnel_metrics
            }

        # Code Quality Rewards
        quality_scores = self.quality_component.calculate(code)
        current_unscaled_components["efficiency"] = quality_scores.get("efficiency_score", 0.0)
        current_unscaled_components["readability"] = quality_scores.get("readability_score", 0.0)
        # The complexity penalty from component is positive, so subtract it or add if config has negative value
        current_unscaled_components["efficiency"] -= quality_scores.get("complexity_penalty_applied", 0.0)
        current_unscaled_components["robustness"] += quality_scores.get("synthesis_bonus_score", 0.0)


        is_valid, err_msg = validate_verilog_code(code, module_name, req_ports)
        if not is_valid:
            current_unscaled_components["base_compilation"] = self.reward_config.compilation_failure
            logger.info(f"{log_pref}: Verilog validation FAILED. Error: {err_msg}")
        else: # Code is structurally valid, proceed to simulation
            logger.debug(f"{log_pref}: Verilog validation SUCCEEDED.")

            # Use FunctionalRewardComponent for simulation and functional reward part
            functional_result = self.functional_component.calculate(
                generated_code=code,
                testbench_path=testbench_path,
                expected_tests=expected_total_tests,
                prompt_identifier=prompt_id_for_log,
                completion_idx=completion_idx
            )

            sim_details = functional_result.get("sim_details", {})
            current_unscaled_components["functional"] = functional_result.get("functional_reward", 0.0)

            # Update funnel metrics from simulation details
            current_funnel_metrics["compiled_successfully"] = sim_details.get("compilation_success", False)
            current_funnel_metrics["sim_ran_successfully"] = sim_details.get("simulation_run_success", False)
            current_funnel_metrics["passed_tests"] = sim_details.get("passed_tests", -1)

            # Base compilation reward is now implicitly handled by functional_reward if it includes compilation failure penalties
            # Or, it can be set explicitly based on compilation_success if preferred.
            if not sim_details.get("compilation_success"):
                current_unscaled_components["base_compilation"] = self.reward_config.compilation_failure
            else:
                current_unscaled_components["base_compilation"] = self.reward_config.compilation_success
                # Robustness bonuses can still be added here based on sim_details from functional_component
                if sim_details.get("all_tests_passed_by_tb", False) and \
                   sim_details.get("passed_tests", 0) == sim_details.get("total_tests_in_output", -1):
                    current_unscaled_components["robustness"] += self.reward_config.all_tests_passed_bonus
                if sim_details.get("passed_tests", 0) == sim_details.get("total_tests_in_output", -1) and \
                   sim_details.get("total_tests_in_output", 0) >= 5:
                    current_unscaled_components["robustness"] += self.reward_config.edge_case_handling_bonus

        # Combine weighted scores
        unscaled_total_reward = (
            self.reward_config.functional_weight * current_unscaled_components["functional"] +
            self.reward_config.efficiency_weight * current_unscaled_components["efficiency"] + # efficiency already includes penalty
            self.reward_config.readability_weight * current_unscaled_components["readability"] +
            self.reward_config.robustness_weight * current_unscaled_components["robustness"] +
            current_unscaled_components["base_compilation"]
        )

        final_scaled_reward = self.reward_config.get_scaled_reward(unscaled_total_reward, training_step)

        return {
            "final_reward": final_scaled_reward,
            "unscaled_components": current_unscaled_components,
            "funnel_metrics": current_funnel_metrics,
            "raw_code": code # Optionally return raw code for experience buffer
        }

    def calculate_batch_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        testbench_paths: List[str],
        expected_total_tests_list: List[int],
        reference_verilog_paths: List[str],
        original_enhanced_prompts: Optional[List[str]] = None, # For debug/logging if needed by _calculate_single_reward
        training_step: int = 0,
        output_dir_for_debug: Optional[str] = None
    ) -> Tuple[List[float], Dict[str, Any]]:

        batch_rewards_final_scaled: List[float] = []
        batch_all_unscaled_components: List[Dict[str, float]] = []
        batch_all_funnel_metrics: List[Dict[str, Any]] = []
        # batch_raw_codes: List[str] = [] # If needed for experience buffer

        num_items_in_batch = len(prompts)

        for i in range(num_items_in_batch):
            # Use original_enhanced_prompt if available for reward calculation context, else the main prompt
            # The `_calculate_single_reward` expects the "user-facing" prompt for its logic (e.g. module name extraction)
            prompt_for_reward_logic = original_enhanced_prompts[i] if original_enhanced_prompts and i < len(original_enhanced_prompts) else prompts[i]

            single_result = self._calculate_single_reward(
                prompt_str=prompt_for_reward_logic, # This is the original_enhanced_prompt or similar
                completion_str=completions[i],
                testbench_path=testbench_paths[i],
                expected_total_tests=expected_total_tests_list[i],
                reference_verilog_path=reference_verilog_paths[i],
                training_step=training_step,
                output_dir_for_debug=output_dir_for_debug,
                completion_idx=i
            )

            batch_rewards_final_scaled.append(single_result["final_reward"])
            batch_all_unscaled_components.append(single_result["unscaled_components"])
            batch_all_funnel_metrics.append(single_result["funnel_metrics"])
            # if "raw_code" in single_result:
            #     batch_raw_codes.append(single_result["raw_code"])

        # Aggregate metrics for the batch (similar to what was in train.py's enhanced_batch_reward_calculator)
        aggregated_metrics_for_wandb = {}
        if num_items_in_batch > 0:
            component_keys = ["functional", "efficiency", "readability", "robustness", "base_compilation"]
            for key in component_keys:
                values = [comp[key] for comp in batch_all_unscaled_components if key in comp]
                if values:
                    aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_mean"] = sum(values)/len(values) # np.mean if numpy is available
                    # Add std dev if needed: aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_std"] = np.std(values)

            funnel_keys = ["code_extracted", "compiled_successfully", "sim_ran_successfully"]
            for key in funnel_keys:
                count = sum(1 for fm in batch_all_funnel_metrics if fm.get(key))
                aggregated_metrics_for_wandb[f"generation_funnel/{key}_count"] = count
                aggregated_metrics_for_wandb[f"generation_funnel/{key}_ratio"] = count / num_items_in_batch if num_items_in_batch > 0 else 0

            passed_tests_values = [fm["passed_tests"] for fm in batch_all_funnel_metrics if fm.get("sim_ran_successfully") and fm["passed_tests"] != -1]
            if passed_tests_values:
                aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_runs"] = sum(passed_tests_values) / len(passed_tests_values)
            else:
                aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_runs"] = 0

        return batch_rewards_final_scaled, aggregated_metrics_for_wandb
