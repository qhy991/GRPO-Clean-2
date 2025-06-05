import re
import os
import logging
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def validate_verilog_code(
    completion_code: str, module_name: str, required_ports: List[str]
) -> Tuple[bool, str]:
    """
    Validates basic structure of generated Verilog code.
    """
    if not module_name:
        return False, "Validation Error: No valid module name provided."
    if not completion_code or not completion_code.strip():
        return False, "Validation Error: Generated code is empty."

    clean_code = completion_code.strip()

    if not re.match(r"^\s*module\s+", clean_code, re.IGNORECASE):
        return False, "Validation Error: Does not start with 'module'."

    generated_module_match = re.search(r"^\s*module\s+([\w\.]+)\s*(?:#\(.*?\))?\s*\(", clean_code, re.IGNORECASE | re.MULTILINE)
    if not generated_module_match:
        return False, f"Validation Error: Cannot find module declaration for '{module_name}'."

    declared_module_name = generated_module_match.group(1)
    if declared_module_name != module_name:
        return False, f"Validation Error: Declared module name '{declared_module_name}' != expected '{module_name}'."

    if not re.search(r"\bendmodule\b", clean_code, re.IGNORECASE | re.MULTILINE):
        return False, "Validation Error: Missing 'endmodule'."

    if required_ports:
        port_pattern_generated = rf"^\s*module\s+{re.escape(module_name)}\s*(?:#\(.*?\)\s*)?\((.*?)\)\s*;"
        port_match_generated = re.search(port_pattern_generated, clean_code, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if not port_match_generated:
            return False, f"Validation Error: Module '{module_name}' port list not found/malformed."

        port_text_generated = port_match_generated.group(1)
        port_text_generated = re.sub(r"//.*?(\n|$)", "\n", port_text_generated)
        port_text_generated = re.sub(r"/\*.*?\*/", "", port_text_generated, flags=re.DOTALL)
        port_text_generated = port_text_generated.replace("\n", " ").strip()

        missing_ports = [rp for rp in required_ports if not re.search(rf"\b{re.escape(rp)}\b", port_text_generated)]
        if missing_ports:
            return False, f"Validation Error: Missing ports in '{module_name}': {', '.join(missing_ports)}. Declared: '{port_text_generated[:200]}...'"

    return True, ""

def assess_code_quality(verilog_code: str) -> Dict[str, float]:
    """
    Assess various quality metrics of Verilog code.
    Returns a dictionary with quality scores (0-1 range).
    """
    if not verilog_code or not verilog_code.strip():
        return {"complexity": 0, "readability": 0, "efficiency": 0, "structure": 0}

    lines = verilog_code.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    # Calculate complexity score (inverse of actual complexity)
    nested_depth = 0
    max_nested_depth = 0
    current_depth = 0

    always_blocks = 0
    assign_statements = 0
    case_statements = 0
    if_statements = 0

    for line in non_empty_lines:
        line_lower = line.lower()

        # Count nesting depth
        if any(keyword in line_lower for keyword in ['begin', 'case', 'if']):
            current_depth += 1
            max_nested_depth = max(max_nested_depth, current_depth)
        if any(keyword in line_lower for keyword in ['end', 'endcase']):
            current_depth = max(0, current_depth - 1)

        # Count different constructs
        if 'always' in line_lower:
            always_blocks += 1
        if 'assign' in line_lower:
            assign_statements += 1
        if 'case' in line_lower:
            case_statements += 1
        if 'if' in line_lower and 'ifdef' not in line_lower:
            if_statements += 1

    # Calculate scores
    complexity_score = max(0, 1.0 - (max_nested_depth / 10.0))

    # Readability: comments, spacing, naming conventions
    comment_lines = sum(1 for line in lines if '//' in line or '/*' in line)
    comment_ratio = comment_lines / max(1, len(non_empty_lines))
    readability_score = min(1.0, comment_ratio * 2 + 0.3)

    # Efficiency: prefer assign over always for combinational logic
    total_logic = always_blocks + assign_statements
    efficiency_score = 0.7 if total_logic > 0 else 0.0
    if assign_statements > always_blocks * 2:
        efficiency_score += 0.3

    # Structure: balanced use of different constructs
    construct_diversity = len([x for x in [always_blocks, assign_statements, case_statements] if x > 0])
    structure_score = min(1.0, construct_diversity / 3.0 + 0.4)

    return {
        "complexity": complexity_score,
        "readability": readability_score,
        "efficiency": efficiency_score,
        "structure": structure_score
    }

def assess_design_complexity(verilog_file: str) -> float:
    """
    Assess the complexity level of a reference design for curriculum learning.
    Returns complexity score (0-10).
    Updated to handle new paths.
    """
    try:
        # 处理可能的路径问题
        if not os.path.exists(verilog_file):
            alt_path = os.path.join(".", verilog_file)
            if os.path.exists(alt_path):
                verilog_file = alt_path
            else:
                logger.warning(f"Could not find Verilog file for complexity assessment: {verilog_file}")
                return 5.0  # 默认中等复杂度

        with open(verilog_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        # Count complexity indicators
        always_blocks = len(re.findall(r'\balways\b', content, re.IGNORECASE))
        state_machines = len(re.findall(r'\bcase\b.*?\bendcase\b', content, re.IGNORECASE | re.DOTALL))
        sequential_logic = len(re.findall(r'@\s*\(\s*posedge|@\s*\(\s*negedge', content, re.IGNORECASE))
        memory_elements = len(re.findall(r'\breg\b|\bmemory\b', content, re.IGNORECASE))

        # Calculate complexity score
        complexity = (
            len(non_empty_lines) / 10 +
            always_blocks * 0.5 +
            state_machines * 2.0 +
            sequential_logic * 1.5 +
            memory_elements * 1.0
        )

        return min(10.0, complexity)

    except Exception as e:
        logger.warning(f"Could not assess complexity for {verilog_file}: {e}")
        return 5.0
