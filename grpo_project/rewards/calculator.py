import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np # Added for statistical calculations
from ..configs.validation_config import ValidationConfig, DEFAULT_VALIDATION_CONFIG

def extract_module_ports_with_types(verilog_file: str) -> tuple[str, list[dict]]:
    """
    从Verilog文件中提取模块名和端口信息（包括类型）
    返回: (module_name, [{"name": "port_name", "type": "input/output/inout"}, ...])
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(verilog_file):
            logger.error(f"Verilog file not found: {verilog_file}")
            return "", []

        with open(verilog_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取模块名 - 修复正则表达式，确保完整匹配模块名
        # 使用更精确的模式，避免匹配到注释中的 "module" 关键字
        # 修改：支持既有端口列表也有无端口列表的模块声明
        module_match = re.search(r'^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:#.*?)?\s*[\(;]', content, re.MULTILINE)
        if not module_match:
            # 备用模式：不要求在行首，同样支持括号或分号
            module_match = re.search(r'\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:#.*?)?\s*[\(;]', content)
            if not module_match:
                logger.error(f"No module declaration found in {verilog_file}")
                return "", []
        module_name = module_match.group(1)

        # 提取端口信息（包括类型）
        ports = []
        
        # 方法1: 从模块声明中提取端口 - 修复正则表达式以支持参数
        port_pattern = r"module\s+" + re.escape(module_name) + r"\s*(?:#\s*\([^)]*\)\s*)?\((.*?)\)\s*;"
        port_match = re.search(port_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if port_match:
            port_text = port_match.group(1)
            # 清理注释
            port_text = re.sub(r"//.*?(\n|$)", "\n", port_text)
            port_text = re.sub(r"/\*.*?\*/", "", port_text, flags=re.DOTALL)
            port_text = port_text.replace("\n", " ").strip()
            
            if port_text:
                # 解析端口声明
                port_declarations = [p.strip() for p in port_text.split(',') if p.strip()]
                for port_decl in port_declarations:
                    # 修复正则表达式以正确处理 reg 类型
                    # 格式: input/output/inout [wire/reg] [位宽] 端口名
                    type_match = re.search(r'\b(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*$', port_decl.strip(), re.IGNORECASE)
                    if type_match:
                        port_type = type_match.group(1).lower()
                        port_name = type_match.group(2)
                        ports.append({"name": port_name, "type": port_type})
                    else:
                        # 如果没有明确类型，尝试从端口名推断
                        parts = port_decl.split()
                        if parts:
                            port_name = parts[-1].strip("(),;")
                            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", port_name):
                                # 默认为input（这是一个合理的默认值）
                                ports.append({"name": port_name, "type": "input"})

        # 方法2: 如果方法1没有找到端口，尝试从独立的端口声明中提取
        if not ports:
            # 修复独立端口声明的正则表达式
            port_decl_pattern = r'\b(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[[^\]]+\]\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*;'
            port_matches = re.findall(port_decl_pattern, content, re.IGNORECASE)
            for port_type, port_name in port_matches:
                ports.append({"name": port_name, "type": port_type.lower()})

        # 去重并排序
        unique_ports = []
        seen_names = set()
        for port in ports:
            if port["name"] not in seen_names:
                unique_ports.append(port)
                seen_names.add(port["name"])
        
        unique_ports.sort(key=lambda x: x["name"])
        
        logger.debug(f"Extracted from {verilog_file}: module='{module_name}', ports={unique_ports}")
        return module_name, unique_ports

    except Exception as e:
        logger.error(f"Error reading or parsing Verilog file {verilog_file}: {e}", exc_info=True)
        return "", []

# Attempt to import from grpo_project, fallback to local if not found
try:
    from grpo_project.configs.reward import EnhancedRewardConfig
    from grpo_project.utils.file_ops import extract_module_info
    from grpo_project.utils.parsing import parse_llm_completion_qwen3
    from grpo_project.utils.verilog_utils import validate_verilog_code, assess_code_quality
    from grpo_project.evaluation.simulator import VerilogSimulator # Import the new simulator
    # Unused component imports FunctionalRewardComponent, CodeQualityRewardComponent are already removed/commented.
    # Unused run_iverilog_simulation import is already removed/commented.
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

    # 添加ValidationConfig占位符
    class ValidationConfig: # type: ignore
        def __init__(self, **kwargs): pass
        @classmethod
        def create_flexible_config(cls): return cls()

    def extract_module_info(verilog_file: str) -> tuple[str, list[str]]: return "placeholder_module", []

    def parse_llm_completion_qwen3(text: str, debug_prompt: Optional[str]=None, debug_context:Optional[Dict[str,Any]]=None) -> tuple[Optional[str], Optional[str]]: 
        # 增强的解析功能 - 处理各种畸形格式
        import re
        if not text or not isinstance(text, str):
            return None, None
        
        text = text.strip()
        reasoning_part = None
        code_part = None
        
        try:
            # 处理畸形的</think>标记（出现在代码块中的情况）
            # 先清理掉错误位置的</think>标记
            cleaned_text = re.sub(r'```verilog\s*</think>\s*\n', '```verilog\n', text, flags=re.IGNORECASE)
            
            # 提取<think>部分 - 更宽松的匹配
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if think_match:
                reasoning_part = think_match.group(1).strip()
                # 移除已匹配的think部分
                text_without_think = cleaned_text[:think_match.start()] + cleaned_text[think_match.end():]
            else:
                # 如果没有完整的think标签，尝试匹配开始标签
                think_start_pattern = r'<think>(.*?)(?=```verilog|$)'
                think_start_match = re.search(think_start_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                if think_start_match:
                    reasoning_part = think_start_match.group(1).strip()
                    text_without_think = cleaned_text[think_start_match.end():]
                else:
                    text_without_think = cleaned_text
            
            # 查找Verilog代码块 - 处理重复的```verilog标记
            # 先尝试标准的代码块格式
            verilog_patterns = [
                r'```verilog\s*(.*?)\s*```',  # 标准格式
                r'```verilog\s*(module.*?endmodule)\s*```?',  # 以module开头的格式，可选结束```
                r'(module\s+\w+.*?endmodule)',  # 直接的module...endmodule
            ]
            
            for pattern in verilog_patterns:
                code_match = re.search(pattern, text_without_think, re.DOTALL | re.IGNORECASE)
                if code_match:
                    code_part = code_match.group(1).strip()
                    # 清理代码中可能残留的错误标记
                    code_part = re.sub(r'</think>\s*\n?', '', code_part)
                    code_part = re.sub(r'```verilog\s*\n?', '', code_part)
                    code_part = code_part.strip()
                    if code_part and 'module' in code_part:
                        break
            
            # 如果仍然没有找到代码，尝试在原始文本中查找
            if not code_part:
                # 最后的努力：在整个文本中查找任何module...endmodule
                fallback_pattern = r'(module\s+\w+.*?endmodule)'
                fallback_match = re.search(fallback_pattern, text, re.DOTALL | re.IGNORECASE)
                if fallback_match:
                    code_part = fallback_match.group(1).strip()
        
        except Exception as e:
            # 如果解析失败，记录错误但不抛出异常
            if debug_context:
                logger = logging.getLogger(__name__)
                logger.warning(f"Parse error in step {debug_context.get('step', 'unknown')}, sample {debug_context.get('sample_idx', 'unknown')}: {e}")
        
        # 验证提取的代码是否有效
        if code_part and code_part.strip():
            # 基本的验证：确保包含module和endmodule
            if 'module' in code_part.lower() and 'endmodule' in code_part.lower():
                return reasoning_part, code_part
            else:
                # 代码无效，返回None
                return reasoning_part, None
        
        return reasoning_part, code_part
    
    def validate_verilog_code(code: str, name: str, ports: list, config=None) -> tuple[bool, str]: return True, ""
    def assess_code_quality(code: str) -> Dict[str, float]: return {"efficiency": 0.0, "readability": 0.0, "complexity": 0.0, "structure": 0.0} # Added assess_code_quality placeholder
    # Removed run_iverilog_simulation placeholder as it's no longer used
    
    class VerilogSimulator: # type: ignore # Added placeholder for VerilogSimulator
        def __init__(self, *args, **kwargs): pass
        def run_simulation(self, *args, **kwargs) -> Dict[str, Any]:
            return {
                "compilation_success": False, 
                "simulation_run_success": False,
                "parsing_success": False,
                "passed_tests": 0,
                "total_tests_in_output": 0,
                "error_message": "Placeholder simulator error",
                "all_tests_passed_by_tb": False
            }
    # Placeholders for FunctionalRewardComponent and CodeQualityRewardComponent are already removed.


logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, reward_config: EnhancedRewardConfig, simulator: Optional[VerilogSimulator] = None): # Changed Optional[Any] to Optional[VerilogSimulator]
        self.reward_config = reward_config
        # If simulator is not passed, instantiate a default one using the (potentially placeholder) VerilogSimulator
        self.simulator = simulator if simulator else VerilogSimulator()
        # self.functional_component and self.quality_component are already removed.
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

        # 从参考Verilog文件中提取模块信息
        module_name, req_ports = "", []
        if reference_verilog_path and os.path.exists(reference_verilog_path):
            module_name, req_ports = extract_module_ports_with_types(reference_verilog_path)

        if not module_name:
            logger.warning(f"{log_pref}: Could not extract module name from reference Verilog file: {reference_verilog_path}")
            module_name = "unknown_module"

        reasoning_part, code_part = parse_llm_completion_qwen3(completion_str, debug_prompt=prompt_str, debug_context={"step": training_step, "sample_idx": completion_idx})

        if code_part and code_part.strip():
            current_funnel_metrics["code_extracted"] = True
        else:
            penalty_type = self.reward_config.missing_code_block_penalty
            current_unscaled_components["base_compilation"] = penalty_type
            total_reward = self.reward_config.get_scaled_reward(penalty_type, training_step)
            # Debug saving logic - 保存到子文件夹
            if output_dir_for_debug:
                # 创建调试文件的子目录
                debug_subdir = os.path.join(output_dir_for_debug, "reward_debug", "missing_code")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_filename = os.path.join(debug_subdir, f"{sanitized_prompt_id_for_file}_completion{completion_idx}_missingcode_{timestamp}.txt")
                try:
                    os.makedirs(debug_subdir, exist_ok=True)
                    with open(debug_filename, "w") as f:
                        f.write(f"Prompt ID: {prompt_id_for_log}\n")
                        f.write(f"Completion Index: {completion_idx}\n")
                        f.write("Reason: No code block found or code is empty.\n")
                        f.write("Original Completion:\n")
                        f.write(completion_str)
                    logger.debug(f"{log_pref}: Saved debug info for missing code to {debug_filename}")
                except Exception as e:
                    logger.error(f"{log_pref}: Failed to save debug info for missing code: {e}")
            return {
                "final_reward": total_reward,
                "unscaled_components": current_unscaled_components,
                "funnel_metrics": current_funnel_metrics,
                "raw_code": "" # Return empty string for raw_code if code is missing
            }

        # Code Quality Assessment (Direct Integration)
        quality_metrics = assess_code_quality(code_part)
        current_unscaled_components["efficiency"] = (
            quality_metrics.get("efficiency", 0) * self.reward_config.code_efficiency_bonus +
            quality_metrics.get("structure", 0) * self.reward_config.synthesis_friendly_bonus -
            max(0, (1 - quality_metrics.get("complexity", 1)) * self.reward_config.code_complexity_penalty)
        )
        current_unscaled_components["readability"] = quality_metrics.get("readability", 0) * self.reward_config.code_readability_bonus
        # Note: The original quality_component also had a "synthesis_bonus_score" that was added to robustness.
        # The provided example for `calculate_enhanced_rewards_for_single_prompt` does not explicitly show this.
        # Assuming the new direct `assess_code_quality` and the efficiency calculation above covers all quality aspects.
        # If a separate synthesis bonus affecting robustness is still needed, it should be added here.
        # For now, following the provided direct integration logic for efficiency and readability.

        # 验证Verilog代码
        # 使用已提取的端口信息（包括类型）
        if req_ports:
            formatted_ports = req_ports  # req_ports 现在已经是正确的格式
        else:
            formatted_ports = []
        
        # 使用灵活的验证配置
        validation_config = ValidationConfig.create_flexible_config()
        is_valid, validation_error = validate_verilog_code(
            code_part, 
            module_name, 
            formatted_ports, 
            validation_config
        )

        if is_valid:
            logger.debug(f"{log_pref}: Verilog validation SUCCEEDED.")
            current_funnel_metrics["compiled_successfully"] = True # Initial assumption, sim_res will confirm

            sim_res = self.simulator.run_simulation(
                generated_verilog_code=code_part,
                testbench_file_path=testbench_path, # Renamed from testbench_path
                expected_total_tests_from_manifest=expected_total_tests, # Renamed from expected_total_tests
                prompt_identifier=prompt_id_for_log,
                completion_idx=completion_idx,
                print_simulation_details=logger.isEnabledFor(logging.DEBUG) # Added print_simulation_details, removed module_name and output_dir_for_debug
            )

            current_funnel_metrics["compiled_successfully"] = sim_res.get("compilation_success", False)

            if not sim_res.get("compilation_success"):
                current_unscaled_components["base_compilation"] = self.reward_config.compilation_failure
                logger.info(f"{log_pref}: Compilation FAILED. Error: {sim_res.get('error_message', 'No error message')}")
            else:
                current_unscaled_components["base_compilation"] = self.reward_config.compilation_success
                current_funnel_metrics["sim_ran_successfully"] = sim_res.get("simulation_run_success", False)

                if not sim_res.get("simulation_run_success"):
                    current_unscaled_components["functional"] = self.reward_config.simulation_crash
                    logger.info(f"{log_pref}: Simulation CRASHED. Details: {sim_res.get('error_message', 'No error message')}")
                elif not sim_res.get("parsing_success"):
                    current_unscaled_components["functional"] = self.reward_config.output_parse_error
                    logger.info(f"{log_pref}: Simulation output parsing FAILED. Details: {sim_res.get('error_message', 'No error message')}")
                else:
                    p = sim_res.get("passed_tests", 0)
                    total_tests_in_output = sim_res.get("total_tests_in_output", 0)
                    current_funnel_metrics["passed_tests"] = p

                    if total_tests_in_output > 0:
                        pass_ratio = p / total_tests_in_output
                        base_functional = pass_ratio * self.reward_config.max_functional_reward
                        if p > 1: # Apply bonus only if more than one test passed
                            bonus_factor = self.reward_config.test_pass_bonus_multiplier ** (p - 1)
                            base_functional *= min(bonus_factor, 2.0) # Cap bonus factor at 2.0
                        current_unscaled_components["functional"] = base_functional
                    elif sim_res.get("all_tests_passed_by_tb", False): # All tests passed but total_tests_in_output is 0 (e.g. no specific test count in output but TB indicates pass)
                         current_unscaled_components["functional"] = self.reward_config.max_functional_reward # Consider this full marks for functional
                         # This case might also imply all tests passed for robustness bonus
                    else:
                        current_unscaled_components["functional"] = 0 # Or a small penalty if expected tests > 0

                    # Robustness Rewards
                    if sim_res.get("all_tests_passed_by_tb", False) and p == total_tests_in_output:
                        current_unscaled_components["robustness"] += self.reward_config.all_tests_passed_bonus
                    # Check for edge case handling bonus (e.g. passing all tests when there are many)
                    if p == total_tests_in_output and total_tests_in_output >= 5: # Assuming 5 or more tests indicates edge case coverage
                        current_unscaled_components["robustness"] += self.reward_config.edge_case_handling_bonus
                    
                    # Log functional test results
                    logger.info(f"{log_pref}: Functional tests: {p}/{total_tests_in_output} passed. All passed by TB: {sim_res.get('all_tests_passed_by_tb', False)}")

        else: # is_valid is false
            current_unscaled_components["base_compilation"] = self.reward_config.compilation_failure
            # current_funnel_metrics["compiled_successfully"] is already False by default
            logger.info(f"{log_pref}: Verilog validation FAILED. Error: {validation_error}")
            # Save debug info for validation failure - 保存到子文件夹
            if output_dir_for_debug:
                # 创建验证错误文件的子目录
                debug_subdir = os.path.join(output_dir_for_debug, "reward_debug", "validation_errors")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_filename = os.path.join(debug_subdir, f"{sanitized_prompt_id_for_file}_completion{completion_idx}_validation_error_{timestamp}.v")
                try:
                    os.makedirs(debug_subdir, exist_ok=True)
                    with open(debug_filename, "w") as f:
                        f.write(f"// Prompt ID: {prompt_id_for_log}\n")
                        f.write(f"// Completion Index: {completion_idx}\n")
                        f.write(f"// Validation Error: {validation_error}\n\n")
                        f.write(code_part)
                    logger.debug(f"{log_pref}: Saved Verilog with validation error to {debug_filename}")
                except Exception as e:
                    logger.error(f"{log_pref}: Failed to save Verilog with validation error: {e}")


        # Final Reward Calculation
        unscaled_total_reward = (
            self.reward_config.functional_weight * current_unscaled_components["functional"] +
            self.reward_config.efficiency_weight * current_unscaled_components["efficiency"] +
            self.reward_config.readability_weight * current_unscaled_components["readability"] +
            self.reward_config.robustness_weight * current_unscaled_components["robustness"] +
            current_unscaled_components["base_compilation"] # This is the base for compilation success/failure
        )

        final_scaled_reward = self.reward_config.get_scaled_reward(unscaled_total_reward, training_step)

        # Logging unscaled rewards and final scaled reward
        logger.info(
            f"{log_pref}: Unscaled Rewards: Functional={current_unscaled_components['functional']:.2f}, "
            f"Efficiency={current_unscaled_components['efficiency']:.2f}, Readability={current_unscaled_components['readability']:.2f}, "
            f"Robustness={current_unscaled_components['robustness']:.2f}, Compilation={current_unscaled_components['base_compilation']:.2f}. "
            f"Total Unscaled: {unscaled_total_reward:.2f}. Final Scaled: {final_scaled_reward:.2f}"
        )
        logger.info(f"{log_pref}: Funnel Metrics: Code Extracted={current_funnel_metrics['code_extracted']}, "
                    f"Compiled Successfully={current_funnel_metrics['compiled_successfully']}, "
                    f"Sim Ran Successfully={current_funnel_metrics['sim_ran_successfully']}, "
                    f"Passed Tests={current_funnel_metrics['passed_tests']}")


        return {
            "final_reward": final_scaled_reward,
            "unscaled_components": current_unscaled_components,
            "funnel_metrics": current_funnel_metrics,
            "raw_code": code_part
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
        output_dir_for_debug: Optional[str] = None,
        wandb_callback_obj: Optional[Any] = None,
        experience_buffer_obj: Optional[Any] = None,
        script_config_obj: Optional[Any] = None,
        **kwargs  # 添加kwargs来捕获多余参数
    ) -> Tuple[List[float], Dict[str, Any]]:

        batch_rewards_final_scaled: List[float] = []
        batch_all_unscaled_components: List[Dict[str, float]] = []
        batch_all_funnel_metrics: List[Dict[str, Any]] = []

        num_items_in_batch = len(prompts)

        # Input Validation
        expected_list_lengths = {
            "completions": len(completions),
            "testbench_paths": len(testbench_paths),
            "expected_total_tests_list": len(expected_total_tests_list),
            "reference_verilog_paths": len(reference_verilog_paths)
        }
        if original_enhanced_prompts is not None:
            expected_list_lengths["original_enhanced_prompts"] = len(original_enhanced_prompts)

        for name, length in expected_list_lengths.items():
            if length != num_items_in_batch:
                logger.error(
                    f"Batch Reward Calculation Error: Mismatch in list lengths. "
                    f"Prompts have {num_items_in_batch} items, but {name} has {length} items."
                )
                # Return a penalty for each completion and empty aggregated metrics
                penalty_reward = self.reward_config.get_scaled_reward(
                    self.reward_config.compilation_failure * 3, training_step # A harsh penalty
                )
                return [penalty_reward] * num_items_in_batch, {}

        if num_items_in_batch == 0:
            logger.warning("calculate_batch_rewards called with an empty batch.")
            return [], {}

        for i in range(num_items_in_batch):
            qwen_formatted_prompt_for_buffer = prompts[i]
            current_completion_str = completions[i]

            # Determine prompt_for_reward_logic (user-facing prompt for _calculate_single_reward)
            prompt_for_reward_logic = ""
            if original_enhanced_prompts and i < len(original_enhanced_prompts) and original_enhanced_prompts[i]:
                prompt_for_reward_logic = original_enhanced_prompts[i]
            else:
                # Fallback: Try to extract user content from Qwen-formatted prompt if original_enhanced_prompt is missing/empty
                # This assumes Qwen format might include "user\n<actual_prompt_content>\nassistant..."
                match = re.search(r"user\n(.*?)\nassistant", qwen_formatted_prompt_for_buffer, re.DOTALL | re.IGNORECASE)
                if match and match.group(1).strip():
                    prompt_for_reward_logic = match.group(1).strip()
                    logger.debug(f"Using extracted user content as prompt_for_reward_logic for item {i}")
                else:
                    # If extraction fails or not possible, use the full Qwen prompt, but log a warning
                    prompt_for_reward_logic = qwen_formatted_prompt_for_buffer
                    logger.warning(
                        f"Item {i}: original_enhanced_prompt not available or empty, and user content extraction failed. "
                        f"Using full Qwen-formatted prompt for reward logic. This might affect module name extraction if not handled by _calculate_single_reward."
                    )
            
            single_result = self._calculate_single_reward(
                prompt_str=prompt_for_reward_logic,
                completion_str=current_completion_str,
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

            # W&B Interaction (per completion)
            if wandb_callback_obj:
                wandb_callback_obj.add_reward(single_result["final_reward"])

            # Experience Buffer Interaction
            if experience_buffer_obj and script_config_obj and script_config_obj.enable_experience_replay:
                experience_buffer_obj.add_experience(
                    prompt=qwen_formatted_prompt_for_buffer,
                    completion=current_completion_str,
                    reward=single_result["final_reward"],
                    metadata={
                        "training_step": training_step,
                        "testbench": testbench_paths[i],
                        "original_enhanced_prompt_preview": prompt_for_reward_logic[:100], # Use the determined prompt
                        "raw_code": single_result.get("raw_code", "")
                    }
                )

        # Aggregate metrics for W&B (after loop)
        aggregated_metrics_for_wandb = {}
        if num_items_in_batch > 0:
            component_keys = ["functional", "efficiency", "readability", "robustness", "base_compilation"]
            for key in component_keys:
                values = [comp[key] for comp in batch_all_unscaled_components if key in comp and comp[key] is not None]
                if values:
                    aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_mean"] = np.mean(values)
                    aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_std"] = np.std(values)
                else:
                    aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_mean"] = 0.0
                    aggregated_metrics_for_wandb[f"reward_components/unscaled_{key}_std"] = 0.0
            
            # Funnel metrics
            successful_extractions = sum(1 for fm in batch_all_funnel_metrics if fm.get("code_extracted", False))
            successful_compilations = sum(1 for fm in batch_all_funnel_metrics if fm.get("compiled_successfully", False))
            simulation_runs = sum(1 for fm in batch_all_funnel_metrics if fm.get("sim_ran_successfully", False))

            aggregated_metrics_for_wandb["generation_funnel/successful_extractions_count"] = successful_extractions
            aggregated_metrics_for_wandb["generation_funnel/successful_extractions_ratio"] = successful_extractions / num_items_in_batch if num_items_in_batch > 0 else 0

            aggregated_metrics_for_wandb["generation_funnel/successful_compilations_count"] = successful_compilations
            aggregated_metrics_for_wandb["generation_funnel/compilation_ratio_vs_extractions"] = successful_compilations / successful_extractions if successful_extractions > 0 else 0
            aggregated_metrics_for_wandb["generation_funnel/compilation_ratio_vs_batch"] = successful_compilations / num_items_in_batch if num_items_in_batch > 0 else 0
            
            aggregated_metrics_for_wandb["generation_funnel/simulation_runs_count"] = simulation_runs
            aggregated_metrics_for_wandb["generation_funnel/simulation_ratio_vs_compilations"] = simulation_runs / successful_compilations if successful_compilations > 0 else 0
            aggregated_metrics_for_wandb["generation_funnel/simulation_ratio_vs_batch"] = simulation_runs / num_items_in_batch if num_items_in_batch > 0 else 0

            passed_tests_values = [fm["passed_tests"] for fm in batch_all_funnel_metrics if fm.get("sim_ran_successfully") and fm.get("passed_tests", -1) != -1]
            if passed_tests_values:
                aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_sim_runs"] = np.mean(passed_tests_values)
                aggregated_metrics_for_wandb["generation_funnel/std_passed_tests_on_success_sim_runs"] = np.std(passed_tests_values)
            else:
                aggregated_metrics_for_wandb["generation_funnel/avg_passed_tests_on_success_sim_runs"] = 0.0
                aggregated_metrics_for_wandb["generation_funnel/std_passed_tests_on_success_sim_runs"] = 0.0
            
            aggregated_metrics_for_wandb["reward/batch_mean_final_scaled_reward"] = np.mean(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0.0
            aggregated_metrics_for_wandb["reward/batch_std_final_scaled_reward"] = np.std(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0.0

        # W&B Interaction (batch aggregated metrics)
        if wandb_callback_obj and aggregated_metrics_for_wandb:
            wandb_callback_obj.log_batch_aggregated_metrics(aggregated_metrics_for_wandb, step=training_step)

        # Detailed periodic logging
        if training_step > 0 and script_config_obj:
            log_interval_reward_stats = getattr(script_config_obj, "log_interval_reward_stats", 10) # Default 10
            log_interval_milestone = getattr(script_config_obj, "log_interval_milestone", 50) # Default 50
            max_steps = getattr(script_config_obj, "max_steps", 1000) # Default 1000, needed for milestone

            if training_step % log_interval_reward_stats == 0:
                reward_mean = np.mean(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0.0
                reward_std = np.std(batch_rewards_final_scaled) if batch_rewards_final_scaled else 0.0
                # Log more detailed stats if needed, using aggregated_metrics_for_wandb
                logger.info(f"Step [{training_step}/{max_steps}] Batch Reward Stats: Mean={reward_mean:.3f}, Std={reward_std:.3f}")
                # Example for logging a specific component mean
                eff_mean = aggregated_metrics_for_wandb.get("reward_components/unscaled_efficiency_mean", 0.0)
                logger.info(f"Step [{training_step}/{max_steps}] Batch Avg Unscaled Efficiency: {eff_mean:.3f}")

            if training_step % log_interval_milestone == 0:
                logger.info(f"Training Milestone: Reached step {training_step} of {max_steps}.")
                # Log summary of funnel
                extraction_ratio = aggregated_metrics_for_wandb.get("generation_funnel/successful_extractions_ratio", 0.0)
                compilation_ratio_batch = aggregated_metrics_for_wandb.get("generation_funnel/compilation_ratio_vs_batch", 0.0)
                simulation_ratio_batch = aggregated_metrics_for_wandb.get("generation_funnel/simulation_ratio_vs_batch", 0.0)
                logger.info(
                    f"Step [{training_step}/{max_steps}] Funnel Summary (Batch Ratios): "
                    f"Extraction={extraction_ratio*100:.1f}%, Compilation={compilation_ratio_batch*100:.1f}%, Simulation={simulation_ratio_batch*100:.1f}%"
                )

        return batch_rewards_final_scaled, aggregated_metrics_for_wandb
