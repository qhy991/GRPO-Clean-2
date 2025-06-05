import os
import subprocess
import tempfile
import logging
import re
from typing import Dict, Any

# Assuming parse_simulation_results_from_output will be in .parser
# If it's needed here (e.g. if run_simulation does its own parsing)
# from .parser import VerilogOutputParser # Or just the function if standalone

logger = logging.getLogger(__name__)

class VerilogSimulator:
    def __init__(self):
        logger.info("VerilogSimulator initialized.")
        # Potentially initialize simulator paths or configurations if needed globally

    def run_simulation(
        self,
        generated_verilog_code: str,
        testbench_file_path: str,
        expected_total_tests_from_manifest: int, # Retained for now, though parsing is separate
        prompt_identifier: str = "N/A",
        completion_idx: int = -1,
        print_simulation_details: bool = False
    ) -> Dict[str, Any]:
        """
        Runs Icarus Verilog simulation.
        This method now focuses on executing the simulation and returning results,
        including parsed counts if the parsing logic is kept here.
        Alternatively, it could return raw stdout/stderr for an external parser.
        For this refactoring, we keep the parsing logic from the original run_iverilog_simulation here.
        """
        # Local import for now, or make it a helper if it's only used here
        from .parser import VerilogOutputParser # Assuming VerilogOutputParser is defined
        parser = VerilogOutputParser()


        result = {
            "compilation_success": False, "simulation_run_success": False, "parsing_success": False,
            "passed_tests": 0, "failed_tests": 0, "total_tests_in_output": 0,
            "all_tests_passed_by_tb": False, "error_message": None, "raw_stdout": "", "raw_stderr": ""
            # "quality_metrics" removed as it's not simulator's responsibility
        }

        log_sim_header = f"--- SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---"
        if print_simulation_details:
            logger.debug("\n" + "="*80 + f"\n{log_sim_header}")

        if not os.path.exists(testbench_file_path):
            alt_path = os.path.join(".", testbench_file_path)
            if os.path.exists(alt_path):
                testbench_file_path = alt_path
            else:
                result["error_message"] = f"Sim Error: Testbench not found: {testbench_file_path}."
                logger.error(f"SIMULATOR: {result['error_message']}")
                return result

        if not generated_verilog_code or not generated_verilog_code.strip():
            result["error_message"] = "Sim Error: Empty Verilog code."
            logger.info(f"SIMULATOR: {result['error_message']}")
            return result

        try:
            with open(testbench_file_path, "r", encoding="utf-8") as tb_f:
                testbench_content = tb_f.read()
        except Exception as e:
            result["error_message"] = f"Sim Error: Could not read testbench {testbench_file_path}: {e}"
            logger.error(f"SIMULATOR: {result['error_message']}")
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
                result["raw_stdout"] += f"IVERILOG COMPILE STDOUT:\n{process_compile.stdout}\n"
                result["raw_stderr"] += f"IVERILOG COMPILE STDERR:\n{process_compile.stderr}\n"

                if process_compile.returncode != 0:
                    result["error_message"] = f"Icarus Verilog compilation failed. Exit: {process_compile.returncode}. Stderr: {process_compile.stderr[:1000]}"
                    return result
                result["compilation_success"] = True
            except subprocess.TimeoutExpired:
                result["error_message"] = "Icarus Verilog compilation timed out."
                return result
            except Exception as e:
                result["error_message"] = f"Error during iverilog compilation: {str(e)}"
                return result

            execute_command = ["vvp", compiled_output_file]
            try:
                process_sim = subprocess.run(execute_command, capture_output=True, text=True, timeout=15, errors="replace")
                result["raw_stdout"] += f"\nVVP SIMULATION STDOUT:\n{process_sim.stdout}\n"
                result["raw_stderr"] += f"VVP SIMULATION STDERR:\n{process_sim.stderr}\n"

                if process_sim.returncode == 0 or "$finish" in process_sim.stdout.lower() or "vvp finished" in process_sim.stdout.lower():
                    result["simulation_run_success"] = True
                else: # Some error or non-zero exit from VVP
                    result["simulation_run_success"] = False
                    vvp_err_msg = f"VVP sim error/non-zero exit. Code: {process_sim.returncode}."
                    current_err = result["error_message"]
                    result["error_message"] = f"{current_err + '; ' if current_err else ''}{vvp_err_msg}"

                if result["simulation_run_success"]:
                    # Use the parser method
                    p, f, t, overall_pass = parser.parse_sim_output(process_sim.stdout)
                    result.update({"passed_tests": p, "failed_tests": f, "total_tests_in_output": t, "all_tests_passed_by_tb": overall_pass})

                    if t > 0 or p > 0 or f > 0 or overall_pass: # Some form of parsable output found
                        result["parsing_success"] = True
                    else: # Simulation ran but output was not parsable for test counts
                        result["parsing_success"] = False
                        err_msg_parse = "Could not parse VVP output for test counts, or 0 tests reported."
                        result["error_message"] = (result.get("error_message", "") + "\n" + err_msg_parse).strip()
                        logger.warning(f"SIMULATOR: {err_msg_parse} for {prompt_identifier}")

                    if result["parsing_success"] and expected_total_tests_from_manifest > 0 and t != expected_total_tests_from_manifest:
                        mismatch_msg = (f"Sim Mismatch! Parsed total ({t}) != expected ({expected_total_tests_from_manifest}) for TB: {testbench_file_path}.")
                        logger.warning(f"SIMULATOR: {mismatch_msg}")
                        # Potentially update error_message or add a specific warning field

            except subprocess.TimeoutExpired:
                result["simulation_run_success"] = False
                result["error_message"] = (result.get("error_message", "") + "\n" + "VVP simulation timed out.").strip()
            except Exception as e:
                result["simulation_run_success"] = False
                result["error_message"] = (result.get("error_message", "") + "\n" + f"Error during VVP execution: {str(e)}").strip()

        if print_simulation_details:
            logger.debug(f"--- END OF SIMULATION RUN FOR PROMPT='{prompt_identifier[:60]}...', COMPLETION_IDX={completion_idx} ---\n" + "="*80 + "\n")
        return result
