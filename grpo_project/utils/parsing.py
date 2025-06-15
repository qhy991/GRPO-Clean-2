from .enhanced_parser import create_enhanced_parsing_function
import re
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# --- Constants for Verilog Parsing ---
THINK_START = "<think>"
THINK_END = "</think>"
CODE_BLOCK_START = "```verilog"
CODE_BLOCK_END = "```"

def parse_llm_completion_qwen3(completion_text: str, debug_prompt: Optional[str] = None,
                              debug_context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    专门为Qwen3优化的解析函数
    """
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    if not completion_text or not isinstance(completion_text, str):
        if debug_enabled:
            logger.debug("❌ Qwen3解析: 输入为空或无效")
        return None, None

    completion_text = completion_text.strip()

    if debug_enabled:
        logger.debug("="*60)
        logger.debug("🔍 Qwen3输出解析开始")
        logger.debug(f"输入长度: {len(completion_text)} 字符")
        logger.debug(f"输入预览: {completion_text[:200]}...")
        logger.debug("="*60)

    reasoning_part = None
    code_part = None

    try:
        # 步骤1: 清理Qwen3特有的标记
        cleaned_text = completion_text

        # 移除可能的对话标记残留
        qwen_markers = [
            r'<\|im_start\|>.*?<\|im_end\|>',
            r'<\|im_start\|>assistant\n?',
            r'<\|im_end\|>',
        ]

        for marker in qwen_markers:
            cleaned_text = re.sub(marker, '', cleaned_text, flags=re.DOTALL)

        cleaned_text = cleaned_text.strip()

        if debug_enabled:
            logger.debug(f"清理后文本长度: {len(cleaned_text)} 字符")
            logger.debug(f"清理后预览: {cleaned_text[:200]}...")

        # 步骤2: 提取<think>部分
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)

        if think_match:
            reasoning_part = think_match.group(1).strip()
            if debug_enabled:
                logger.debug(f"✅ 找到<think>块: {len(reasoning_part)} 字符")
                logger.debug(f"推理预览: {reasoning_part[:150]}...")
        else:
            if debug_enabled:
                logger.debug("❌ 未找到<think>块")
                # 检查是否有不完整的think标签
                if '<think>' in cleaned_text.lower():
                    logger.debug("⚠️ 找到开始标签但缺少结束标签")
                    # 尝试提取开始标签后的内容作为推理
                    think_start = cleaned_text.lower().find('<think>')
                    if think_start >= 0:
                        potential_reasoning = cleaned_text[think_start + 7:].split('```')[0].strip()
                        if len(potential_reasoning) > 20:
                            reasoning_part = potential_reasoning
                            if debug_enabled:
                                logger.debug(f"🔄 恢复的推理: {len(reasoning_part)} 字符")

        # 步骤3: 提取Verilog代码块
        verilog_patterns = [
            r'```verilog\s*(.*?)\s*```',  # 标准Verilog块
            r'```\s*(module\s+.*?endmodule)\s*```',  # 通用代码块中的module
            r'```(?:systemverilog|sv)?\s*(.*?)\s*```',  # SystemVerilog或其他变体
        ]

        for i, pattern in enumerate(verilog_patterns):
            code_match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            if code_match:
                code_part = code_match.group(1).strip()
                if debug_enabled:
                    logger.debug(f"✅ 找到代码块(模式{i+1}): {len(code_part)} 字符")
                    logger.debug(f"代码预览: {code_part[:150]}...")
                break

        if not code_part:
            if debug_enabled:
                logger.debug("❌ 未找到标准代码块，尝试直接提取module")

            # 步骤4: 直接提取module...endmodule
            module_pattern = r'(module\s+\w+.*?endmodule)'
            module_match = re.search(module_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)

            if module_match:
                code_part = module_match.group(1).strip()
                if debug_enabled:
                    logger.debug(f"✅ 直接提取module: {len(code_part)} 字符")
            else:
                if debug_enabled:
                    logger.debug("❌ 未找到module...endmodule模式")

        # 步骤5: 最终验证和清理
        if reasoning_part:
            # 清理推理部分
            reasoning_part = re.sub(r'^[\s\n]*', '', reasoning_part)
            reasoning_part = re.sub(r'[\s\n]*$', '', reasoning_part)

            # 移除可能的提示性文字
            reasoning_part = re.sub(r'^(.*thinking.*?[:：]\s*)', '', reasoning_part, flags=re.IGNORECASE)

            if len(reasoning_part) < 10:
                if debug_enabled:
                    logger.debug(f"⚠️ 推理部分太短({len(reasoning_part)}字符)，设为None")
                reasoning_part = None

        if code_part:
            # 清理代码部分
            code_part = re.sub(r'^[\s\n]*', '', code_part)
            code_part = re.sub(r'[\s\n]*$', '', code_part)

            # 验证代码质量
            if 'module' not in code_part.lower():
                if debug_enabled:
                    logger.debug("⚠️ 代码中没有'module'关键字")

            if len(code_part) < 20:
                if debug_enabled:
                    logger.debug(f"⚠️ 代码太短({len(code_part)}字符)，设为None")
                code_part = None

        # 最终结果记录
        if debug_enabled:
            logger.debug("="*40)
            logger.debug("🎯 解析结果:")
            logger.debug(f"  推理部分: {'✅' if reasoning_part else '❌'} ({len(reasoning_part) if reasoning_part else 0} 字符)")
            logger.debug(f"  代码部分: {'✅' if code_part else '❌'} ({len(code_part) if code_part else 0} 字符)")

            if not reasoning_part and not code_part:
                logger.debug("❌ 完全解析失败")
                logger.debug("🔍 诊断信息:")
                logger.debug(f"  包含'<think>': {'<think>' in cleaned_text.lower()}")
                logger.debug(f"  包含'</think>': {'</think>' in cleaned_text.lower()}")
                logger.debug(f"  包含'```verilog': {'```verilog' in cleaned_text.lower()}")
                logger.debug(f"  包含'```': {'```' in cleaned_text}")
                logger.debug(f"  包含'module': {'module' in cleaned_text.lower()}")
                logger.debug(f"  包含'endmodule': {'endmodule' in cleaned_text.lower()}")
            elif reasoning_part and code_part:
                logger.debug("✅ 完全解析成功")
            else:
                logger.debug("⚠️ 部分解析成功")

            logger.debug("="*60)

    except Exception as e:
        logger.error(f"❌ Qwen3解析异常: {e}", exc_info=True)
        if debug_enabled:
            logger.debug("尝试应急恢复...")

        # 应急恢复
        if 'module' in completion_text.lower() and 'endmodule' in completion_text.lower():
            try:
                emergency_match = re.search(r'(module\s+.*?endmodule)', completion_text, re.DOTALL | re.IGNORECASE)
                if emergency_match:
                    code_part = emergency_match.group(1).strip()
                    if debug_enabled:
                        logger.debug(f"🆘 应急恢复成功: {len(code_part)} 字符")
            except Exception as e_emergency:
                if debug_enabled:
                    logger.debug(f"❌ 应急恢复失败: {e_emergency}")

    return reasoning_part, code_part

def parse_llm_completion(completion_text: str, debug_prompt: Optional[str] = None,
                        debug_context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses the LLM's completion text to extract reasoning and code blocks.
    Now supports both old format (VERILOG_REASONING/CODE markers) and new format (<think>/```verilog).
    Returns (reasoning_text, code_text).
    If a block is not found, the corresponding return value will be None.
    If no tags are found, assumes the entire output might be code (as a fallback).

    Args:
        completion_text: The LLM's output text to parse
        debug_prompt: Optional prompt that was used to generate this completion (for debugging)
        debug_context: Optional context dictionary with step, sample_idx, etc.
    """

    # === 调试信息初始化 ===
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    debug_info = {
        "input_length": len(completion_text) if completion_text else 0,
        "has_debug_prompt": debug_prompt is not None,
        "debug_context": debug_context or {},
        "parsing_steps": [],
        "detected_patterns": [],
        "warnings": [],
        "errors": []
    }

    # === 输入验证 ===
    if not completion_text or not isinstance(completion_text, str):
        debug_info["errors"].append("Empty or invalid input")
        if debug_enabled:
            logger.debug("="*80)
            logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
            logger.debug(f"Input validation FAILED: Empty or invalid input (type: {type(completion_text)})")
            if debug_prompt:
                logger.debug(f"Original Prompt (first 300 chars):\n{debug_prompt[:300]}{'...' if len(debug_prompt) > 300 else ''}")
            logger.debug("="*80)
        return None, None

    completion_text = completion_text.strip()
    debug_info["input_length_after_strip"] = len(completion_text)

    if len(completion_text) < 5:  # 太短的输入无意义
        debug_info["errors"].append(f"Input too short ({len(completion_text)} chars)")
        if debug_enabled:
            logger.debug("="*80)
            logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
            logger.debug(f"Input validation FAILED: Too short ({len(completion_text)} chars)")
            if debug_prompt:
                logger.debug(f"Original Prompt (first 300 chars):\n{debug_prompt[:300]}{'...' if len(debug_prompt) > 300 else ''}")
            logger.debug(f"Short completion text: '{completion_text}'")
            logger.debug("="*80)
        return None, None

    # === 调试会话开始 ===
    if debug_enabled:
        logger.debug("="*80)
        logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION START")
        logger.debug(f"Debug Context: {debug_context}")
        logger.debug(f"Completion text length: {len(completion_text)} chars")

        # 打印原始prompt（如果提供）
        if debug_prompt:
            logger.debug("-" * 40)
            logger.debug("ORIGINAL PROMPT:")
            logger.debug("-" * 40)
            # 截断过长的prompt，但保留结构
            if len(debug_prompt) > 1000:
                prompt_preview = debug_prompt[:500] + "\n\n[... TRUNCATED ...]\n\n" + debug_prompt[-500:]
            else:
                prompt_preview = debug_prompt
            logger.debug(f"{prompt_preview}")
            logger.debug("-" * 40)

        # 打印completion文本预览
        logger.debug("COMPLETION TEXT PREVIEW:")
        logger.debug("-" * 40)
        if len(completion_text) > 800:
            completion_preview = completion_text[:400] + "\n\n[... MIDDLE TRUNCATED ...]\n\n" + completion_text[-400:]
        else:
            completion_preview = completion_text
        logger.debug(f"{completion_preview}")
        logger.debug("-" * 40)

    reasoning_part = None
    code_part = None
    detected_format = "no_known_format"
    parsing_success_flags = {
        "think_block": False,
        "verilog_block": False,
        "generic_code_block": False,
        "module_pattern": False,
        "emergency_extraction": False,
        "full_output_fallback": False
    }

    try:
        debug_info["parsing_steps"].append("Starting main parsing logic")

        # === 新格式解析：<think>...</think> ===
        think_match = re.search(
            r"<think>(.*?)</think>",
            completion_text,
            re.DOTALL | re.IGNORECASE
        )
        if think_match:
            reasoning_part = think_match.group(1).strip()
            parsing_success_flags["think_block"] = True
            debug_info["detected_patterns"].append("think_block")
            debug_info["parsing_steps"].append(f"Found <think> block with {len(reasoning_part)} chars")
            if debug_enabled:
                logger.debug(f"✓ FOUND <think> block:")
                logger.debug(f"  Length: {len(reasoning_part)} chars")
                logger.debug(f"  Preview: '{reasoning_part[:150]}{'...' if len(reasoning_part) > 150 else ''}'")
        else:
            debug_info["warnings"].append("<think> block not found")
            debug_info["parsing_steps"].append("No <think> block found")
            if debug_enabled:
                logger.debug("✗ <think> block NOT FOUND")
                # 检查是否有不完整的think标签
                if "<think>" in completion_text.lower():
                    logger.debug("  ⚠️ Found opening <think> tag but no closing </think>")
                elif "</think>" in completion_text.lower():
                    logger.debug("  ⚠️ Found closing </think> tag but no opening <think>")

        # === 新格式解析：```verilog...``` ===
        verilog_code_match = re.search(
            r"```verilog\s*(.*?)\s*```",
            completion_text,
            re.DOTALL | re.IGNORECASE
        )
        if verilog_code_match:
            code_part = verilog_code_match.group(1).strip()
            parsing_success_flags["verilog_block"] = True
            debug_info["detected_patterns"].append("verilog_block")
            debug_info["parsing_steps"].append(f"Found ```verilog block with {len(code_part)} chars")
            if debug_enabled:
                logger.debug(f"✓ FOUND ```verilog block:")
                logger.debug(f"  Length: {len(code_part)} chars")
                logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")
        else:
            debug_info["warnings"].append("```verilog block not found")
            debug_info["parsing_steps"].append("No ```verilog block found, trying fallbacks")
            if debug_enabled:
                logger.debug("✗ ```verilog block NOT FOUND")
                # 检查是否有不完整的代码块
                if "```verilog" in completion_text.lower():
                    logger.debug("  ⚠️ Found opening ```verilog but no closing ```")
                elif "```" in completion_text:
                    logger.debug("  ⚠️ Found generic ``` markers but not verilog-specific")

            # Fallback: 尝试通用的代码块格式 ```...```
            generic_code_match = re.search(
                r"```\s*(module\s+.*?endmodule)\s*```",
                completion_text,
                re.DOTALL | re.IGNORECASE
            )
            if generic_code_match:
                code_part = generic_code_match.group(1).strip()
                parsing_success_flags["generic_code_block"] = True
                debug_info["detected_patterns"].append("generic_code_block")
                debug_info["parsing_steps"].append(f"Found generic code block (fallback) with {len(code_part)} chars")
                detected_format = "fallback_generic_code_block"
                if debug_enabled:
                    logger.debug(f"✓ FOUND generic ``` block (fallback):")
                    logger.debug(f"  Length: {len(code_part)} chars")
                    logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")

        # === 格式检测和日志 ===
        if think_match and verilog_code_match:
            detected_format = "new_format_think_and_code"
        elif think_match and not verilog_code_match:
            detected_format = "new_format_think_only"
        elif not think_match and verilog_code_match:
            detected_format = "new_format_code_only"

        # === 详细的格式缺失警告 ===
        if not think_match or not verilog_code_match:
            missing_parts = []
            if not think_match:
                missing_parts.append("<think>...</think>")
            if not verilog_code_match:
                missing_parts.append("```verilog...```")

            warning_msg = f"Missing format blocks: {', '.join(missing_parts)}"
            debug_info["warnings"].append(warning_msg)

            if debug_enabled:
                logger.debug(f"⚠️ FORMAT WARNING: {warning_msg}")
                # 显示完整的completion用于分析
                logger.debug("FULL COMPLETION TEXT FOR ANALYSIS:")
                logger.debug("-" * 40)
                logger.debug(completion_text)
                logger.debug("-" * 40)

        # === 额外的代码提取fallback ===
        if code_part is None:
            debug_info["parsing_steps"].append("Attempting direct module...endmodule fallback")
            if debug_enabled:
                logger.debug("🔄 Attempting direct module...endmodule fallback")

            module_match = re.search(
                r"(module\s+\w+.*?endmodule)",
                completion_text,
                re.DOTALL | re.IGNORECASE
            )
            if module_match:
                code_part = module_match.group(1).strip()
                parsing_success_flags["module_pattern"] = True
                debug_info["detected_patterns"].append("module_pattern")
                debug_info["parsing_steps"].append(f"Found module pattern (fallback) with {len(code_part)} chars")
                if detected_format in ["no_known_format", "new_format_think_only"]:
                    detected_format = "fallback_module_endmodule"
                if debug_enabled:
                    logger.debug(f"✓ FOUND module pattern (fallback):")
                    logger.debug(f"  Length: {len(code_part)} chars")
                    logger.debug(f"  Preview: '{code_part[:150]}{'...' if len(code_part) > 150 else ''}'")
            else:
                debug_info["warnings"].append("Direct module...endmodule pattern not found")
                if debug_enabled:
                    logger.debug("✗ Direct module...endmodule pattern NOT FOUND")

        # === Final fallback ===
        if code_part is None and reasoning_part is None:
            debug_info["parsing_steps"].append("Attempting final fallback strategies")
            if debug_enabled:
                logger.debug("🔄 Attempting final fallback strategies")

            has_markers = any(marker in completion_text for marker in [
                "<think>", "</think>", "```verilog", "```"
            ])
            debug_info["has_format_markers"] = has_markers

            if not has_markers:
                debug_info["parsing_steps"].append("No format markers found, treating full output as code")
                code_part = completion_text.strip()
                parsing_success_flags["full_output_fallback"] = True
                if code_part:
                    detected_format = "fallback_full_output_as_code"
                if debug_enabled:
                    logger.debug("🔄 No format markers found, treating full output as code")
                    logger.debug(f"  Full output length: {len(code_part)} chars")
            else:
                debug_info["warnings"].append("Format markers present but parsing failed")
                if debug_enabled:
                    logger.debug("⚠️ Format markers were present but parsing failed")

                # 尝试恢复不完整的think标签
                if "</think>" in completion_text and "<think>" not in completion_text:
                    think_end_pos = completion_text.find("</think>")
                    potential_reasoning = completion_text[:think_end_pos].strip()
                    if len(potential_reasoning) > 10:
                        reasoning_part = potential_reasoning
                        debug_info["parsing_steps"].append("Recovered reasoning from incomplete think tags")
                        if detected_format == "no_known_format":
                            detected_format = "fallback_recovered_think_only"
                        if debug_enabled:
                            logger.debug(f"✓ RECOVERED reasoning from incomplete tags:")
                            logger.debug(f"  Length: {len(reasoning_part)} chars")

    except Exception as e:
        error_msg = f"Exception during parsing: {e}"
        debug_info["errors"].append(error_msg)
        debug_info["parsing_steps"].append(f"Exception occurred: {str(e)}")
        if debug_enabled:
            logger.debug(f"❌ EXCEPTION during parsing: {e}")

        logger.error(f"UTILS: parse_llm_completion: Exception during parsing: {e}", exc_info=True)

        # Emergency extraction
        if "module" in completion_text.lower() and "endmodule" in completion_text.lower():
            try:
                debug_info["parsing_steps"].append("Attempting emergency extraction")
                module_match = re.search(r"(module\s+.*?endmodule)", completion_text, re.DOTALL | re.IGNORECASE)
                if module_match:
                    code_part = module_match.group(1).strip()
                    parsing_success_flags["emergency_extraction"] = True
                    debug_info["detected_patterns"].append("emergency_extraction")
                    if detected_format == "no_known_format":
                        detected_format = "fallback_emergency_extraction"
                    if debug_enabled:
                        logger.debug(f"✓ EMERGENCY extraction successful:")
                        logger.debug(f"  Length: {len(code_part)} chars")
            except Exception as e_emergency:
                emergency_error = f"Exception during emergency extraction: {e_emergency}"
                debug_info["errors"].append(emergency_error)
                if debug_enabled:
                    logger.debug(f"❌ Emergency extraction failed: {e_emergency}")
                logger.error(f"UTILS: parse_llm_completion: {emergency_error}", exc_info=True)

    # === 数据质量检查和清理 ===
    original_reasoning_length = len(reasoning_part) if reasoning_part else 0
    original_code_length = len(code_part) if code_part else 0

    if reasoning_part:
        # 清理推理部分
        reasoning_part = re.sub(r"^[\s\n]*", "", reasoning_part)
        reasoning_part = re.sub(r"[\s\n]*$", "", reasoning_part)
        reasoning_part = re.sub(r"^(your thought process|思考过程).*?[:：]\s*", "", reasoning_part, flags=re.IGNORECASE)

        if len(reasoning_part) < 10:
            debug_info["warnings"].append(f"Reasoning too short ({len(reasoning_part)} chars), setting to None")
            reasoning_part = None

    if code_part:
        # 清理代码部分
        code_part = re.sub(r"^[\s\n]*", "", code_part)
        code_part = re.sub(r"[\s\n]*$", "", code_part)

        if "module" not in code_part.lower():
            debug_info["warnings"].append(f"Extracted code doesn't contain 'module' keyword")

        if len(code_part) < 20:
            debug_info["warnings"].append(f"Code too short ({len(code_part)} chars), setting to None")
            code_part = None

    # === 最终调试总结 ===
    debug_info["final_results"] = {
        "reasoning_length": len(reasoning_part) if reasoning_part else 0,
        "code_length": len(code_part) if code_part else 0,
        "detected_format": detected_format,
        "parsing_success_flags": parsing_success_flags,
        "original_reasoning_length": original_reasoning_length,
        "original_code_length": original_code_length
    }

    if debug_enabled:
        logger.debug("="*40)
        logger.debug("PARSING SUMMARY:")
        logger.debug(f"  Detected format: {detected_format}")
        logger.debug(f"  Success flags: {parsing_success_flags}")
        logger.debug(f"  Final reasoning: {len(reasoning_part) if reasoning_part else 0} chars")
        logger.debug(f"  Final code: {len(code_part) if code_part else 0} chars")
        logger.debug(f"  Warnings: {len(debug_info['warnings'])}")
        logger.debug(f"  Errors: {len(debug_info['errors'])}")

        if debug_info["warnings"]:
            logger.debug("  Warning details:")
            for i, warning in enumerate(debug_info["warnings"], 1):
                logger.debug(f"    {i}. {warning}")

        if debug_info["errors"]:
            logger.debug("  Error details:")
            for i, error in enumerate(debug_info["errors"], 1):
                logger.debug(f"    {i}. {error}")

        # 如果解析失败，显示额外诊断信息
        if not reasoning_part and not code_part:
            logger.debug("❌ PARSING COMPLETELY FAILED")
            logger.debug("Diagnostic information:")
            logger.debug(f"  Contains '<think>': {'<think>' in completion_text.lower()}")
            logger.debug(f"  Contains '</think>': {'</think>' in completion_text.lower()}")
            logger.debug(f"  Contains '```verilog': {'```verilog' in completion_text.lower()}")
            logger.debug(f"  Contains '```': {'```' in completion_text}")
            logger.debug(f"  Contains 'module': {'module' in completion_text.lower()}")
            logger.debug(f"  Contains 'endmodule': {'endmodule' in completion_text.lower()}")
        elif reasoning_part and code_part:
            logger.debug("✅ PARSING SUCCESSFUL (both parts found)")
        else:
            logger.debug("⚠️ PARTIAL PARSING SUCCESS")

        logger.debug("="*80)
        logger.debug("UTILS: parse_llm_completion: DEBUGGING SESSION END")
        logger.debug("="*80)

    return reasoning_part, code_part

parse_llm_completion_with_context = parse_llm_completion # Alias for backward compatibility

def parse_llm_completion_with_context_qwen3(completion_text: str, prompt: str = None,
                                           step: int = None, sample_idx: int = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Wrapper function that calls parse_llm_completion_qwen3 with debug context
    """
    debug_context = {}
    if step is not None:
        debug_context["step"] = step
    if sample_idx is not None:
        debug_context["sample_idx"] = sample_idx
    
    return parse_llm_completion_qwen3(completion_text, debug_prompt=prompt, debug_context=debug_context)

def parse_llm_completion_with_context(completion_text: str, prompt: str = None, 
                                    step: int = None, sample_idx: int = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Wrapper function that calls parse_llm_completion with debug context
    """
    debug_context = {}
    if step is not None:
        debug_context["step"] = step
    if sample_idx is not None:
        debug_context["sample_idx"] = sample_idx
    
    return parse_llm_completion(completion_text, debug_prompt=prompt, debug_context=debug_context)

def validate_and_fix_output_format(raw_output: str) -> Tuple[str, bool]:
    """
    验证和修复LLM输出格式
    返回: (修复后的输出, 是否需要修复)
    """
    if not raw_output:
        return raw_output, False
    
    original_output = raw_output
    needs_fix = False
    
    # 移除Qwen对话标记
    if '<|im_start|>' in raw_output or '<|im_end|>' in raw_output:
        raw_output = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', raw_output, flags=re.DOTALL)
        raw_output = re.sub(r'<\|im_start\|>assistant\n?', '', raw_output)
        raw_output = re.sub(r'<\|im_end\|>', '', raw_output)
        needs_fix = True
    
    # 检查是否有think标签
    has_think_start = '<think>' in raw_output.lower()
    has_think_end = '</think>' in raw_output.lower()
    
    # 检查是否有代码块
    has_verilog_block = '```verilog' in raw_output.lower()
    has_code_end = '```' in raw_output and raw_output.rfind('```') > raw_output.find('```verilog')
    
    # 如果缺少think标签，尝试添加
    if not has_think_start and not has_think_end:
        # 查找可能的思考内容（在代码块之前的文本）
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos > 0:
            thinking_content = raw_output[:verilog_pos].strip()
            if len(thinking_content) > 50:  # 有足够的内容
                raw_output = f"<think>\n{thinking_content}\n</think>\n\n{raw_output[verilog_pos:]}"
                needs_fix = True
    
    # 如果只有开始标签，尝试修复结束标签
    elif has_think_start and not has_think_end:
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos > 0:
            think_pos = raw_output.lower().find('<think>')
            think_content = raw_output[think_pos:verilog_pos].strip()
            remaining_content = raw_output[verilog_pos:]
            raw_output = f"{think_content}\n</think>\n\n{remaining_content}"
            needs_fix = True
    
    # 如果缺少verilog代码块标记，尝试添加
    if not has_verilog_block:
        # 查找module...endmodule模式
        module_match = re.search(r'(module\s+.*?endmodule)', raw_output, re.DOTALL | re.IGNORECASE)
        if module_match:
            module_code = module_match.group(1)
            # 替换原始的module代码为带标记的版本
            raw_output = raw_output.replace(module_code, f"```verilog\n{module_code}\n```")
            needs_fix = True
    
    # 如果有开始标记但没有结束标记
    elif has_verilog_block and not has_code_end:
        verilog_pos = raw_output.lower().find('```verilog')
        if verilog_pos >= 0:
            # 查找verilog代码块后的endmodule
            after_verilog = raw_output[verilog_pos + 10:]  # 跳过```verilog
            endmodule_match = re.search(r'(.*?endmodule)', after_verilog, re.DOTALL | re.IGNORECASE)
            if endmodule_match:
                verilog_content = endmodule_match.group(1).strip()
                remaining_content = after_verilog[endmodule_match.end():].strip()
                
                # 重构输出
                before_verilog = raw_output[:verilog_pos]
                if remaining_content:
                    raw_output = f"{before_verilog}```verilog\n{verilog_content}\n```\n{remaining_content}"
                else:
                    raw_output = f"{before_verilog}```verilog\n{verilog_content}\n```"
                needs_fix = True
    
    return raw_output.strip(), needs_fix

# 使用增强的解析器
_enhanced_parse_func, _enhanced_parser = create_enhanced_parsing_function()

def parse_llm_completion_with_context_qwen3(completion_text: str, prompt: str = None,
                                           step: int = None, sample_idx: int = None):
    """使用增强解析器的Qwen3解析函数"""
    return _enhanced_parse_func(completion_text, prompt, step, sample_idx)

# 向后兼容
parse_llm_completion_with_context = parse_llm_completion_with_context_qwen3

def get_parsing_stats():
    """获取解析统计信息"""
    return _enhanced_parser.get_parsing_stats()

def reset_parsing_stats():
    """重置解析统计信息"""
    _enhanced_parser.reset_stats()
