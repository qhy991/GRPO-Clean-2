# This file's contents (run_iverilog_simulation and parse_simulation_results_from_output)
# have been moved to grpo_project.evaluation.simulator.VerilogSimulator
# and grpo_project.evaluation.parser.VerilogOutputParser respectively.

# This file can be removed in a later cleanup step if no other simulation-related
# utilities that don't fit into the new evaluation module are added here.

import logging

logger = logging.getLogger(__name__)
logger.info("grpo_project.utils.simulation module is now empty. Functionality moved to grpo_project.evaluation.")
