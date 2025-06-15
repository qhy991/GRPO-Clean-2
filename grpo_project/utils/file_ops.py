import os
import re
import logging
from typing import List, Tuple, Dict, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

def validate_and_update_dataset_paths(dataset: Dataset, dataset_base_path: str = None) -> List[Dict[str, Any]]:
    """
    Validates dataset examples, resolves file paths to absolute paths,
    and ensures analysis.instantiations_found is valid (if present).
    """
    if dataset is None:
        logger.error("UTILS: Dataset provided to validate_and_update_dataset_paths is None.")
        return []

    processed_examples: List[Dict[str, Any]] = []
    required_keys = ['prompt', 'testbench_path', 'expected_total_tests', 'reference_verilog_path']

    for i, example_orig in enumerate(dataset):
        example = example_orig.copy()
        is_valid_example = True
        prompt_log = str(example.get('prompt', 'N/A'))[:70]

        # Check required keys
        for key in required_keys:
            if key not in example or example[key] is None:
                logger.warning(f"UTILS: Dataset row {i} ('{prompt_log}...') missing or None for key '{key}'. Skipping example.")
                is_valid_example = False
                break
        if not is_valid_example:
            continue

        # Validate analysis.instantiations_found if present
        if 'analysis' in example and isinstance(example['analysis'], dict):
            if 'instantiations_found' in example['analysis'] and example['analysis']['instantiations_found']:
                if not isinstance(example['analysis']['instantiations_found'], list):
                    logger.warning(f"UTILS: Invalid instantiations_found type in analysis for row {i} ('{prompt_log}...'): {type(example['analysis']['instantiations_found'])}. Ignoring.")
                    example['analysis']['instantiations_found'] = []
                elif not all(isinstance(m, str) and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", m) for m in example['analysis']['instantiations_found']):
                    logger.warning(f"UTILS: Invalid module names in analysis.instantiations_found for row {i} ('{prompt_log}...'): {example['analysis']['instantiations_found']}. Ignoring.")
                    example['analysis']['instantiations_found'] = []
            else:
                logger.debug(f"UTILS: No instantiations_found in analysis for row {i} ('{prompt_log}...'). Setting to empty list.")
                example['analysis']['instantiations_found'] = []
        else:
            logger.debug(f"UTILS: No analysis field in row {i} ('{prompt_log}...'). Adding empty analysis.")
            example['analysis'] = {'instantiations_found': []}

        # Path validation and update
        tb_path_orig = str(example['testbench_path'])
        ref_path_orig = str(example['reference_verilog_path'])

        if dataset_base_path:
            tb_full_path = os.path.join(dataset_base_path, tb_path_orig)
            if os.path.exists(tb_full_path):
                example['testbench_path'] = tb_full_path
            else:
                logger.warning(f"UTILS: Testbench file not found for row {i} ('{prompt_log}...'): {tb_path_orig} (resolved to: {tb_full_path}). Skipping example.")
                is_valid_example = False

            if is_valid_example:
                ref_full_path = os.path.join(dataset_base_path, ref_path_orig)
                if os.path.exists(ref_full_path):
                    example['reference_verilog_path'] = ref_full_path
                else:
                    logger.warning(f"UTILS: Reference Verilog file not found for row {i} ('{prompt_log}...'): {ref_path_orig} (resolved to: {ref_full_path}). Skipping example.")
                    is_valid_example = False
        else:
            if not os.path.exists(tb_path_orig):
                logger.warning(f"UTILS: Testbench file not found (dataset_base_path not provided): {tb_path_orig} for row {i} ('{prompt_log}...'). Skipping example.")
                is_valid_example = False
            if is_valid_example and not os.path.exists(ref_path_orig):
                logger.warning(f"UTILS: Reference Verilog file not found (dataset_base_path not provided): {ref_path_orig} for row {i} ('{prompt_log}...'). Skipping example.")
                is_valid_example = False

        if is_valid_example:
            processed_examples.append(example)

    if not processed_examples and len(dataset) > 0:
        logger.error("UTILS: No valid examples found after validation and path update. All examples were skipped.")
    elif len(dataset) > 0:
        logger.info(f"UTILS: Dataset validation and path update complete. {len(processed_examples)}/{len(dataset)} examples are valid and have updated paths.")

    return processed_examples

def extract_module_info(verilog_file: str) -> Tuple[str, List[str]]:
    """
    Extracts module name and port list from a Verilog file.
    Updated to handle the new folder-prefixed paths.
    """
    try:
        # 检查文件是否存在（处理可能的路径问题）
        if not os.path.exists(verilog_file):
            # 尝试相对路径
            alt_path = os.path.join(".", verilog_file)
            if os.path.exists(alt_path):
                verilog_file = alt_path
            else:
                logger.error(f"UTILS: Verilog file not found: {verilog_file}")
                return "", []

        with open(verilog_file, "r", encoding="utf-8") as f:
            content = f.read()

        module_match = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", content, re.IGNORECASE)
        if not module_match:
            logger.error(f"UTILS: No module declaration found in {verilog_file}")
            return "", []
        module_name = module_match.group(1)

        port_pattern_text = r"module\s+" + re.escape(module_name) + r"\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;"
        port_match = re.search(port_pattern_text, content, re.IGNORECASE | re.DOTALL)

        ports = []
        if port_match:
            port_text = port_match.group(1)
            port_text = re.sub(r"//.*?(\n|$)", "\n", port_text)
            port_text = re.sub(r"/\*.*?\*/", "", port_text, flags=re.DOTALL)
            port_text = port_text.replace("\n", " ").strip()

            if port_text:
                port_declarations = [p.strip() for p in port_text.split(',') if p.strip()]
                for port_decl_full in port_declarations:
                    parts = port_decl_full.split()
                    if parts:
                        potential_name = parts[-1].strip("(),;")
                        verilog_keywords = {
                            "input", "output", "inout", "reg", "wire", "logic", "signed", "unsigned",
                            "parameter", "localparam", "integer", "real", "time", "genvar",
                            "always", "assign", "begin", "end", "if", "else", "case", "for"
                        }
                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", potential_name) and potential_name.lower() not in verilog_keywords:
                            ports.append(potential_name)
                        elif len(parts) > 1 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", parts[-2].strip("(),;")) and parts[-2].lower() not in verilog_keywords:
                            ports.append(parts[-2].strip("(),;"))

        unique_ports = sorted(list(set(ports)))
        logger.debug(f"UTILS: Extracted from {verilog_file}: module='{module_name}', ports={unique_ports}")
        return module_name, unique_ports

    except FileNotFoundError:
        logger.error(f"UTILS: Verilog file not found: {verilog_file}")
        return "", []
    except Exception as e:
        logger.error(f"UTILS: Error reading or parsing Verilog file {verilog_file}: {e}", exc_info=True)
        return "", []
