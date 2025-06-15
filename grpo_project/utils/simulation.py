import re
import os
import subprocess
import tempfile
import logging
from typing import Dict, Any, Tuple, List # Added List

# Assuming assess_code_quality is in .verilog_utils or imported via grpo_project.utils
try:
    from .verilog_utils import assess_code_quality
except ImportError:
    logger_placeholder = logging.getLogger(__name__ + "_placeholder")
    logger_placeholder.warning(
        "simulation.py: Could not import 'assess_code_quality' from '.verilog_utils'. "
        "Simulations run via run_iverilog_simulation will have empty quality_metrics."
    )
    def assess_code_quality(verilog_code: str) -> Dict[str, float]: # type: ignore
        return {}

logger = logging.getLogger(__name__)

def parse_simulation_results_from_output(sim_output: str) -> Tuple[int, int, int, bool]:
    """
    Parses simulation output for passed/failed test cases and overall status.
    """
    passed_count, failed_count, total_cases = 0, 0, 0
    is_overall_pass = False

    # More robust regex to find the summary line, allowing for variations in spacing and text
    summary_match = re.search(
        r"Passed:\s*(\d+)\s*,\s*Failed:\s*(\d+).*?"
        r"Total Test Cases:\s*(\d+).*?"
        r"(OVERALL_PASS|OVERALL_FAIL)",
        sim_output,
        re.DOTALL | re.IGNORECASE,
    )

    if summary_match:
        try:
            passed_count = int(summary_match.group(1))
            failed_count = int(summary_match.group(2))
            total_cases = int(summary_match.group(3))
            is_overall_pass = summary_match.group(4).upper() == "OVERALL_PASS"
        except ValueError: # If conversion to int fails
            logger.warning(f"SIM_PARSE: ValueError parsing full summary from '{summary_match.group(0)}'. Fallback if OVERALL_PASS found.")
            # Fallback: if OVERALL_PASS is present, assume it's a pass, but counts might be unreliable
            if re.search(r"OVERALL_PASS", sim_output, re.IGNORECASE):
                is_overall_pass = True
    else:
        # Fallback to individual parsing if the full summary line is not found or malformed
        pc_match = re.search(r"Passed:\s*(\d+)", sim_output, re.IGNORECASE)
        fc_match = re.search(r"Failed:\s*(\d+)", sim_output, re.IGNORECASE)
        tc_match = re.search(r"Total Test Cases:\s*(\d+)", sim_output, re.IGNORECASE)

        if pc_match: passed_count = int(pc_match.group(1))
        if fc_match: failed_count = int(fc_match.group(1))
        if tc_match: total_cases = int(tc_match.group(1))

        # Infer total_cases if not explicitly found but passed/failed are
        if total_cases == 0 and (passed_count > 0 or failed_count > 0):
            total_cases = passed_count + failed_count

        # Infer overall_pass based on common patterns if full summary is missing
        if re.search(r"OVERALL_PASS", sim_output, re.IGNORECASE):
            is_overall_pass = True
            if total_cases == 0 and passed_count > 0 and failed_count == 0: # Ensure total_cases is consistent
                total_cases = passed_count
        # If some tests passed, none failed, AND no explicit "OVERALL_FAIL" tag, consider it a pass.
        elif passed_count > 0 and failed_count == 0 and not re.search(r"OVERALL_FAIL", sim_output, re.IGNORECASE):
            is_overall_pass = True
            if total_cases == 0 : total_cases = passed_count # Make sure total_cases reflects passed if it was 0

    # Final consistency check for total_cases
    if total_cases == 0 and (passed_count > 0 or failed_count > 0) and not is_overall_pass :
        # If overall status isn't pass, and total is 0, sum passed and failed for a more accurate total
        total_cases = passed_count + failed_count
    elif is_overall_pass and total_cases == 0 and passed_count > 0:
        # If overall pass and total is 0, total should be at least number of passed tests
        total_cases = passed_count

    return passed_count, failed_count, total_cases, is_overall_pass


def run_iverilog_simulation(
    generated_verilog_code: str,
    testbench_file_path: str,
    expected_total_tests_from_manifest: int, # Added to match usage in DetailedInferenceCallback
    prompt_identifier: str = "N/A", # For logging context
    completion_idx: int = -1,       # For logging context
    print_simulation_details: bool = False # For verbose debug logging
) -> Dict[str, Any]:
    """
    Runs Icarus Verilog simulation with enhanced error handling and output parsing.
    """
    result: Dict[str, Any] = {
        "compilation_success": False,
        "simulation_run_success": False,
        "parsing_success": False,
        "passed_tests": 0,
        "failed_tests": 0,
        "total_tests_in_output": 0,
        "all_tests_passed_by_tb": False, # Indicates if the "OVERALL_PASS" tag was found
        "error_message": None,
        "raw_stdout": "",
        "raw_stderr": "",
        "quality_metrics": {} # Populated by assess_code_quality
    }

    log_sim_header = f"--- SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---"
    if print_simulation_details:
        logger.debug(f"\n{'='*80}\n{log_sim_header}")

    if not os.path.exists(testbench_file_path):
        result["error_message"] = f"Sim Error: Testbench not found: {testbench_file_path}."
        logger.error(f"UTILS.SIM: {result['error_message']}")
        return result

    if not generated_verilog_code or not generated_verilog_code.strip():
        result["error_message"] = "Sim Error: Empty Verilog code provided."
        logger.info(f"UTILS.SIM: {result['error_message']}")
        return result

    try:
        result["quality_metrics"] = assess_code_quality(generated_verilog_code)
    except Exception as e_quality:
        logger.warning(f"UTILS.SIM: Could not assess code quality: {e_quality}")
        result["quality_metrics"] = {"error": str(e_quality)}

    try:
        with open(testbench_file_path, "r", encoding="utf-8") as tb_f:
            testbench_content = tb_f.read()
    except Exception as e:
        result["error_message"] = f"Sim Error: Could not read testbench {testbench_file_path}: {e}"
        logger.error(f"UTILS.SIM: {result['error_message']}")
        return result

    if print_simulation_details:
        logger.debug(f"\n[Testbench Path]: {testbench_file_path}")
        logger.debug(f"[Generated Verilog Code (DUT)]:\n{generated_verilog_code}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        if print_simulation_details:
            logger.debug(f"[Temp Sim Dir]: {temp_dir}")

        generated_design_file = os.path.join(temp_dir, "design_under_test.v")
        compiled_output_file = os.path.join(temp_dir, "simulation.vvp")

        with open(generated_design_file, "w", encoding="utf-8") as f:
            f.write(generated_verilog_code)

        tb_top_module_name = "testbench" # Default
        match = re.search(r"^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(\s*\))?\s*;", testbench_content, re.MULTILINE | re.IGNORECASE)
        if match:
            tb_top_module_name = match.group(1)
        if print_simulation_details:
            logger.debug(f"[Deduced TB Top Module]: {tb_top_module_name}")

        compile_command = ["iverilog", "-g2012", "-Wall", "-o", compiled_output_file, generated_design_file, testbench_file_path, "-s", tb_top_module_name]
        if print_simulation_details:
            logger.debug(f"\n[Compile Command]: {' '.join(compile_command)}")

        try:
            process_compile = subprocess.run(compile_command, capture_output=True, text=True, timeout=15, errors="replace")
            result["raw_stdout"] = f"IVERILOG COMPILE STDOUT:\n{process_compile.stdout}\n"
            result["raw_stderr"] = f"IVERILOG COMPILE STDERR:\n{process_compile.stderr}\n"

            if print_simulation_details:
                logger.debug("\n[Compiler Output]:")
                if process_compile.stdout: logger.debug(f"  STDOUT:\n{process_compile.stdout}")
                if process_compile.stderr: logger.debug(f"  STDERR:\n{process_compile.stderr}")
                logger.debug(f"  Return Code: {process_compile.returncode}")

            if process_compile.returncode != 0:
                result["error_message"] = f"Icarus Verilog compilation failed. Exit: {process_compile.returncode}. Stderr: {process_compile.stderr[:1000]}"
                if print_simulation_details: logger.debug(f"COMPILATION FAILED. Error: {result['error_message']}")
                return result
            result["compilation_success"] = True
            if print_simulation_details: logger.debug("COMPILATION SUCCEEDED.")

        except subprocess.TimeoutExpired:
            result["error_message"] = "Icarus Verilog compilation timed out."
            if print_simulation_details: logger.debug(f"COMPILATION TIMED OUT. Error: {result['error_message']}")
            return result
        except Exception as e: # Catch other potential errors during compilation
            result["error_message"] = f"Error during iverilog compilation process: {str(e)}"
            if print_simulation_details: logger.debug(f"COMPILATION EXCEPTION. Error: {result['error_message']}")
            return result

        execute_command = ["vvp", compiled_output_file]
        if print_simulation_details: logger.debug(f"\n[Execution Command]: {' '.join(execute_command)}")

        try:
            process_sim = subprocess.run(execute_command, capture_output=True, text=True, timeout=15, errors="replace")
            result["raw_stdout"] += f"\nVVP SIMULATION STDOUT:\n{process_sim.stdout}\n"
            result["raw_stderr"] += f"VVP SIMULATION STDERR:\n{process_sim.stderr}\n"

            if print_simulation_details:
                logger.debug("\n[Simulator Output (VVP)]:")
                if process_sim.stdout: logger.debug(f"  STDOUT:\n{process_sim.stdout}")
                if process_sim.stderr: logger.debug(f"  STDERR:\n{process_sim.stderr}")
                logger.debug(f"  Return Code: {process_sim.returncode}")

            # Check for explicit VVP errors or specific success indicators
            if "error" in process_sim.stderr.lower() or "segmentation fault" in process_sim.stderr.lower():
                result["simulation_run_success"] = False
                vvp_err_msg = f"VVP sim error. Exit: {process_sim.returncode}. Stdout: ...{process_sim.stdout[-500:]}. Stderr: {process_sim.stderr[:1000]}"
                result["error_message"] = (result.get("error_message", "") + "\n" + vvp_err_msg).strip()
            elif process_sim.returncode == 0 or "$finish" in process_sim.stdout.lower() or "vvp finished" in process_sim.stdout.lower():
                result["simulation_run_success"] = True
            else:
                # No explicit error string, but non-zero exit. Could still be a functional failure detected by testbench.
                result["simulation_run_success"] = True # The simulation ran, but might have failed tests.
                logger.warning(f"UTILS.SIM: VVP sim exit code {process_sim.returncode}, but no explicit error string in stderr. Proceeding with output parsing. Stdout: {process_sim.stdout[:200]}")

            if print_simulation_details:
                logger.debug(f"SIMULATION RUN CONSIDERED {'SUCCESSFUL (from VVP perspective)' if result['simulation_run_success'] else 'FAILED (VVP error)'}.")

            if result["compilation_success"] and result["simulation_run_success"]:
                p, f, t, overall_pass_tag = parse_simulation_results_from_output(process_sim.stdout)
                result.update({
                    "passed_tests": p, "failed_tests": f,
                    "total_tests_in_output": t, "all_tests_passed_by_tb": overall_pass_tag
                })
                if print_simulation_details:
                    logger.debug("\n[Parsed Simulation Results]:")
                    logger.debug(f"  Passed: {p}, Failed: {f}, Total in Output: {t}, Overall Pass Tag: {overall_pass_tag}")

                # Determine parsing_success: if we got any numbers or an overall_pass_tag
                if t > 0 or p > 0 or f > 0 or overall_pass_tag:
                    result["parsing_success"] = True
                elif result["simulation_run_success"]: # Sim ran but nothing useful parsed
                    result["parsing_success"] = False
                    err_msg_parse = f"Could not parse VVP output for test counts, or 0 tests reported. Raw VVP stdout: {process_sim.stdout[:500]}"
                    result["error_message"] = (result.get("error_message", "") + "\n" + err_msg_parse).strip()
                    logger.warning(f"UTILS.SIM: {err_msg_parse}")

                # Check consistency with manifest (warning only)
                if result["parsing_success"] and expected_total_tests_from_manifest > 0 and t != expected_total_tests_from_manifest:
                    mismatch_msg = (
                        f"Sim Mismatch! Parsed total tests ({t}) != expected from manifest ({expected_total_tests_from_manifest}) "
                        f"for TB: {testbench_file_path}."
                    )
                    logger.warning(f"UTILS.SIM: {mismatch_msg} Sim Output Preview: {process_sim.stdout[:200]}")
                    # This doesn't necessarily mean a simulation failure, could be testbench issue or parsing limitation.

        except subprocess.TimeoutExpired:
            result["simulation_run_success"] = False
            timeout_specific_message = "VVP simulation timed out."
            result["error_message"] = (result.get("error_message", "") + "\n" + timeout_specific_message).strip()
            if print_simulation_details: logger.debug(f"SIMULATION EXECUTION TIMED OUT. Error: {result['error_message']}")
        except Exception as e: # Catch other potential errors during VVP run or parsing
            result["simulation_run_success"] = False
            exception_specific_message = f"Error during VVP execution or result parsing: {str(e)}"
            result["error_message"] = (result.get("error_message", "") + "\n" + exception_specific_message).strip()
            if print_simulation_details: logger.debug(f"SIMULATION EXECUTION EXCEPTION. Error: {result['error_message']}")

        if print_simulation_details and result.get("error_message"):
            logger.debug(f"FINAL SIMULATION ERROR/TIMEOUT. Error: {result['error_message']}")

    if print_simulation_details:
        logger.debug(f"--- END OF SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---\n{'='*80}\n")
    return result
