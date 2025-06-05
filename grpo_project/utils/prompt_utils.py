import logging
import os
import re
from typing import Dict, Any

# Import extract_module_info from .file_ops, as it's used by enhance_prompt_func
# This creates a dependency: file_ops.py must define extract_module_info
try:
    from .file_ops import extract_module_info
except ImportError:
    # Fallback placeholder if file_ops or extract_module_info isn't available during transitional refactoring
    # This helps avoid outright script crashes if files are processed out of strict order.
    logger_placeholder = logging.getLogger(__name__ + "_placeholder")
    logger_placeholder.warning(
        "prompt_utils: Could not import 'extract_module_info' from '.file_ops'. "
        "Enhance_prompt_func will use a placeholder for module info, which may affect prompt quality."
    )
    def extract_module_info(verilog_file: str) -> tuple[str, list[str]]: # type: ignore
        return "placeholder_module_due_to_import_error", []

logger = logging.getLogger(__name__)

def wrap_prompt_for_qwen3(example: Dict[str, Any]) -> Dict[str, Any]:
    """为Qwen3模型包装prompt的函数"""
    enhanced_content = example.get("prompt")

    # Qwen3的系统提示
    system_message = """You are a Verilog expert. Please provide your solution in the following strict format:

<think>
Your detailed analysis and thinking process here
</think>

```verilog
Your complete Verilog module code here
```

Requirements:
- The <think> section must contain your reasoning
- The ```verilog section must contain complete, compilable Verilog code
- Follow the exact format shown above"""

    if enhanced_content and isinstance(enhanced_content, str):
        # 保留原始的增强后提示
        example["original_enhanced_prompt"] = enhanced_content

        # 构建Qwen3格式的对话提示
        example["prompt"] = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{enhanced_content.strip()}<|im_end|>\n<|im_start|>assistant\n"
    else:
        logger.warning(f"wrap_prompt_for_qwen3: Encountered non-string or empty prompt: {enhanced_content}")
        # Provide a default, safe, Qwen3-formatted prompt
        example["prompt"] = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\nPlease design a Verilog module.<|im_end|>\n<|im_start|>assistant\n"
        example["original_enhanced_prompt"] = example.get("prompt", "Default prompt due to invalid input") # Keep original if it existed

    return example

def enhance_prompt_func(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Improved prompt enhancement function with stricter formatting requirements.
    Relies on extract_module_info.
    """
    original_prompt = str(example.get("prompt", "No original prompt provided in example."))
    ref_path = str(example.get("reference_verilog_path", ""))

    module_name, ports = "", []
    if ref_path and os.path.exists(ref_path):
        try:
            module_name, ports = extract_module_info(ref_path)
        except Exception as e_extract:
            logger.error(f"EnhancePrompt: Error extracting module info from {ref_path}: {e_extract}", exc_info=True)
            # Fallback if extraction fails
            module_name = "extracted_module_error"
            ports = []

    if not module_name:
        logger.warning(f"EnhancePrompt: Could not determine module name from ref_path '{ref_path}'. Using fallback 'generated_module'.")
        module_name = "generated_module"

    port_desc_list = []
    if ports and ref_path and os.path.exists(ref_path): # Ensure ref_path exists for reading content
        try:
            with open(ref_path, "r", encoding="utf-8") as f_ref:
                ref_content = f_ref.read()
            for p_name in ports:
                try:
                    # Regex to find full port declaration (more robust)
                    # This regex tries to capture common Verilog port declaration styles
                    regex = r"(\b(?:input|output|inout)\b\s*(?:(?:reg|wire|logic|signed|unsigned)\s*)?(?:\[[^\]]+\]\s*)?\b" + re.escape(p_name) + r"\b)"
                    match = re.search(regex, ref_content, re.IGNORECASE | re.MULTILINE)
                    if match:
                        full_decl = re.sub(r'\s+', ' ', match.group(1).strip().replace("\n", " "))
                        port_desc_list.append(full_decl)
                    else:
                        port_desc_list.append(f"`{p_name}` (details not auto-detected from ref content)")
                except Exception as e_inner_port:
                    logger.warning(f"EnhancePrompt: Error processing port '{p_name}' from ref_path '{ref_path}': {e_inner_port}")
                    port_desc_list.append(f"`{p_name}` (processing error)")
        except Exception as e_read_ref:
            logger.error(f"EnhancePrompt: Could not read reference file {ref_path} for port details: {e_read_ref}")
            # Fallback if reading ref file fails but ports were somehow extracted
            port_desc_list = [f"`{p}` (reference content unreadable)" for p in ports]

    elif ports: # Ports extracted but ref_path might be missing or unreadable
        port_desc_list = [f"`{p}` (no reference content provided or readable)" for p in ports]

    port_desc = ("; ".join(port_desc_list) if port_desc_list else "as implied by the design context or problem description")

    system_instruction = f"""**CRITICAL: You must strictly adhere to the following format. Violations will result in a score of 0!**
1. First, provide your detailed thought process between <think> and </think> tags.
2. Then, provide the complete Verilog module code between ```verilog and ```.
3. Do not include any other content or formatting.

**Required output format example:**
<think>
I need to design a module named {module_name}...
Step 1: Analyze requirements...
Step 2: Design the interface...
Step 3: Implement the logic...
</think>
```verilog
module {module_name}(...);  // Your implementation code
endmodule
```

**Strict requirements:**
- The module name must be: `{module_name}`
- Must include all required ports: {port_desc}
- The code must be complete and synthesizable
- Must pass all test cases

**Original problem description:**
{original_prompt}

Now, please provide your solution in the strict format specified above:
"""
    # The 'prompt' field for the model input should be the system_instruction.
    # 'original_prompt_for_debug' is kept for logging or if the original prompt was different.
    return {"prompt": system_instruction, "original_prompt_for_debug": original_prompt}
