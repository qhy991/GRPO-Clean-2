import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class VerilogOutputParser:
    def __init__(self):
        logger.info("VerilogOutputParser initialized.")

    def parse_sim_output(self, sim_output: str) -> Tuple[int, int, int, bool]:
        """
        Parses simulation output for passed/failed test cases and overall status.
        """
        passed_count, failed_count, total_cases = 0, 0, 0
        is_overall_pass = False

        # More robust regex to capture variations in summary format
        summary_match = re.search(
            r"Passed:\s*(\d+)\s*,\s*Failed:\s*(\d+)(?:.*?Total Test Cases:\s*(\d+))?.*?(OVERALL_PASS|OVERALL_FAIL)",
            sim_output,
            re.DOTALL | re.IGNORECASE,
        )

        if summary_match:
            try:
                passed_count = int(summary_match.group(1))
                failed_count = int(summary_match.group(2))
                # Total cases might not always be present in this specific regex group if it's before OVERALL_PASS/FAIL
                total_cases_str = summary_match.group(3)
                if total_cases_str:
                    total_cases = int(total_cases_str)
                else: # Fallback if the third group (Total Test Cases) is not captured by this specific match
                    # Try to find it separately or infer it
                    tc_match_fallback = re.search(r"Total Test Cases:\s*(\d+)", sim_output, re.IGNORECASE)
                    if tc_match_fallback:
                        total_cases = int(tc_match_fallback.group(1))
                    elif passed_count > 0 or failed_count > 0 : # Infer if possible
                        total_cases = passed_count + failed_count


                is_overall_pass = summary_match.group(4).upper() == "OVERALL_PASS"

                # If total_cases is still 0 but we have passes/fails, sum them up
                if total_cases == 0 and (passed_count > 0 or failed_count > 0):
                    total_cases = passed_count + failed_count

            except ValueError:
                logger.warning("PARSER: ValueError parsing full summary. Attempting fallback parsing.")
                # Fallback parsing if the main regex fails or groups are problematic
                self._fallback_parse(sim_output, passed_count, failed_count, total_cases, is_overall_pass)

        else: # Main summary regex didn't match, try simpler individual regexes
            self._fallback_parse(sim_output, passed_count, failed_count, total_cases, is_overall_pass)
            # Re-assign values from fallback parsing (passed by reference effectively via list for mutable update)
            parsed_values = self._fallback_parse(sim_output)
            passed_count, failed_count, total_cases, is_overall_pass = parsed_values


        # Final sanity check for total_cases
        if total_cases == 0 and (passed_count > 0 or failed_count > 0):
            total_cases = passed_count + failed_count

        # If OVERALL_PASS is true, and we have passed_count but total_cases is less, adjust total_cases
        if is_overall_pass and passed_count > 0 and total_cases < passed_count :
            logger.debug(f"Adjusting total_cases from {total_cases} to {passed_count} due to OVERALL_PASS and passed_count.")
            total_cases = passed_count


        logger.debug(f"Parsed simulation output: Passed={passed_count}, Failed={failed_count}, Total={total_cases}, OverallPass={is_overall_pass}")
        return passed_count, failed_count, total_cases, is_overall_pass

    def _fallback_parse(self, sim_output: str) -> Tuple[int, int, int, bool] :
        """Helper for fallback parsing if the main regex fails."""
        passed_count, failed_count, total_cases = 0,0,0
        is_overall_pass = False

        pc_match = re.search(r"Passed:\s*(\d+)", sim_output, re.IGNORECASE)
        fc_match = re.search(r"Failed:\s*(\d+)", sim_output, re.IGNORECASE)
        tc_match = re.search(r"Total Test Cases:\s*(\d+)", sim_output, re.IGNORECASE)

        if pc_match: passed_count = int(pc_match.group(1))
        if fc_match: failed_count = int(fc_match.group(1))
        if tc_match: total_cases = int(tc_match.group(1))

        if total_cases == 0 and (passed_count > 0 or failed_count > 0):
            total_cases = passed_count + failed_count

        # Determine overall pass status based on common patterns
        if re.search(r"OVERALL_PASS", sim_output, re.IGNORECASE):
            is_overall_pass = True
            if total_cases == 0 and passed_count > 0 and failed_count == 0 : # If only "OVERALL_PASS" and "Passed: X"
                 total_cases = passed_count
        elif passed_count > 0 and failed_count == 0 and not re.search(r"OVERALL_FAIL", sim_output, re.IGNORECASE):
            # If tests passed and no explicit fail, assume overall pass
            is_overall_pass = True
            if total_cases == 0 : total_cases = passed_count

        return passed_count, failed_count, total_cases, is_overall_pass

    # Placeholder for LLM Verilog code block parsing, if this class were to expand scope
    # For now, this remains in grpo_project.utils.parsing
    # def parse_llm_verilog_output(self, llm_output: str) -> Tuple[Optional[str], Optional[str]]:
    #     from grpo_project.utils.parsing import parse_llm_completion_qwen3 # Example
    #     return parse_llm_completion_qwen3(llm_output)
