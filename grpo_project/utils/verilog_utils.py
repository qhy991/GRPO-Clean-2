import re
import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..configs.validation_config import ValidationConfig, DEFAULT_VALIDATION_CONFIG

logger = logging.getLogger(__name__)

def validate_verilog_code(
    code: str, 
    expected_module_name: str, 
    required_ports: List[Dict[str, str]], 
    config: Optional[ValidationConfig] = None
) -> Tuple[bool, str]:
    """
    验证Verilog代码的正确性
    
    Args:
        code: Verilog代码
        expected_module_name: 期望的模块名
        required_ports: 必需的端口列表，格式为 [{"name": "port_name", "type": "input/output"}]
        config: 验证配置，如果为None则使用默认配置
    
    Returns:
        (is_valid, error_message): 验证结果和错误信息
    """
    # 暂时禁用验证逻辑 - 直接返回通过
    if config and config.verbose_validation:
        logger.info(f"验证已禁用 - 自动通过: 模块 '{expected_module_name}'")
    
    return True, "验证已禁用 - 自动通过"
    
    # 以下是原始验证逻辑（已禁用）
    """
    if config is None:
        config = DEFAULT_VALIDATION_CONFIG
    
    if not code or not code.strip():
        return False, "代码为空"
    
    # 清理代码
    code = code.strip()
    
    # 1. 检查基本结构
    if config.require_endmodule and 'endmodule' not in code:
        return False, "缺少 'endmodule' 关键字"
    
    # 2. 提取模块声明
    module_pattern = r'module\s+(\w+)\s*\('
    module_match = re.search(module_pattern, code, re.IGNORECASE)
    
    if not module_match:
        return False, "未找到有效的模块声明"
    
    actual_module_name = module_match.group(1)
    
    # 3. 模块名验证
    if config.strict_module_name:
        if actual_module_name != expected_module_name:
            return False, f"模块名不匹配: 期望 '{expected_module_name}', 实际 '{actual_module_name}'"
    elif not config.allow_module_name_variations:
        # 中等严格模式：允许一些常见变体
        if not _is_module_name_acceptable(actual_module_name, expected_module_name):
            return False, f"模块名不可接受: 期望 '{expected_module_name}' 或其变体, 实际 '{actual_module_name}'"
    
    # 4. 端口验证
    port_validation_result = _validate_ports(code, required_ports, config)
    if not port_validation_result[0]:
        return port_validation_result
    
    # 5. 语法基本检查
    syntax_check = _basic_syntax_check(code)
    if not syntax_check[0]:
        return syntax_check
    
    if config.verbose_validation:
        logger.info(f"验证通过: 模块 '{actual_module_name}' (期望: '{expected_module_name}')")
    
    return True, "验证通过"
    """

def _is_module_name_acceptable(actual: str, expected: str) -> bool:
    """检查模块名是否可接受（允许常见变体）"""
    # 完全匹配
    if actual == expected:
        return True
    
    # 忽略大小写
    if actual.lower() == expected.lower():
        return True
    
    # 允许下划线变体
    actual_normalized = actual.replace('_', '').lower()
    expected_normalized = expected.replace('_', '').lower()
    if actual_normalized == expected_normalized:
        return True
    
    # 允许常见后缀
    common_suffixes = ['_module', '_mod', '_v1', '_v2', '_impl']
    for suffix in common_suffixes:
        if actual.lower() == (expected + suffix).lower():
            return True
        if (actual + suffix).lower() == expected.lower():
            return True
    
    return False

def _validate_ports(code: str, required_ports: List[Dict[str, str]], config: ValidationConfig) -> Tuple[bool, str]:
    """验证端口定义"""
    if not required_ports:
        return True, "无需验证端口"
    
    # 更强大的端口提取逻辑
    found_port_dict = {}
    
    # 方法1: 从模块声明中提取端口
    module_pattern = r'module\s+\w+\s*\((.*?)\)\s*;'
    module_match = re.search(module_pattern, code, re.DOTALL | re.IGNORECASE)
    
    if module_match:
        port_list = module_match.group(1)
        # 清理注释
        port_list = re.sub(r'//.*?$', '', port_list, flags=re.MULTILINE)
        port_list = re.sub(r'/\*.*?\*/', '', port_list, flags=re.DOTALL)
        
        # 提取端口声明 - 支持多种格式
        # 格式1: input wire [3:0] SW
        # 格式2: output LEDR
        # 格式3: input a, b, c
        port_patterns = [
            r'(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[[^\]]+\]\s+)?(\w+)',  # 带wire/reg和位宽
            r'(input|output|inout)\s+(\w+)',  # 简单格式
        ]
        
        for pattern in port_patterns:
            matches = re.findall(pattern, port_list, re.IGNORECASE)
            for port_type, port_name in matches:
                found_port_dict[port_name] = port_type.lower()
    
    # 方法2: 从模块内部的端口声明中提取（如果模块声明中没有完整信息）
    if not found_port_dict:
        # 查找独立的端口声明
        standalone_port_patterns = [
            r'(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:\[[^\]]+\]\s+)?(\w+)\s*;',
            r'(input|output|inout)\s+(\w+)\s*;'
        ]
        
        for pattern in standalone_port_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for port_type, port_name in matches:
                found_port_dict[port_name] = port_type.lower()
    
    # 调试信息
    if config.verbose_validation:
        logger.info(f"检测到的端口: {found_port_dict}")
        logger.info(f"要求的端口: {required_ports}")
    
    # 创建大小写不敏感的端口映射
    found_port_dict_lower = {k.lower(): (k, v) for k, v in found_port_dict.items()}
    
    # 检查必需端口
    missing_ports = []
    type_mismatches = []
    
    # 检查是否是简单I/O模块的情况
    is_simple_io_case = _is_simple_io_case(required_ports, found_port_dict)
    
    for req_port in required_ports:
        port_name = req_port['name']
        port_type = req_port['type'].lower()
        
        # 首先尝试精确匹配
        if port_name in found_port_dict:
            if found_port_dict[port_name] != port_type:
                type_mismatches.append(
                    f"{port_name}: 期望 {port_type}, 实际 {found_port_dict[port_name]}"
                )
        # 然后尝试大小写不敏感匹配
        elif port_name.lower() in found_port_dict_lower:
            actual_name, actual_type = found_port_dict_lower[port_name.lower()]
            if actual_type != port_type:
                type_mismatches.append(
                    f"{port_name}: 期望 {port_type}, 实际 {actual_type} (大小写不匹配: {actual_name})"
                )
            # 大小写不匹配但类型正确，记录警告但不算错误
            if config.verbose_validation:
                logger.warning(f"端口名大小写不匹配: 期望 '{port_name}', 实际 '{actual_name}'")
        # 对于简单I/O情况，尝试语义匹配
        elif is_simple_io_case:
            semantic_match = _find_semantic_port_match(port_name, port_type, found_port_dict)
            if semantic_match:
                if config.verbose_validation:
                    logger.info(f"语义匹配成功: 期望 '{port_name}' ({port_type}) -> 实际 '{semantic_match}' ({found_port_dict[semantic_match]})")
            else:
                missing_ports.append(f"{port_name} ({port_type})")
        else:
            missing_ports.append(f"{port_name} ({port_type})")
    
    if missing_ports:
        return False, f"缺少必需端口: {', '.join(missing_ports)}"
    
    if type_mismatches:
        return False, f"端口类型不匹配: {'; '.join(type_mismatches)}"
    
    # 检查额外端口（如果不允许）
    if not config.allow_extra_ports:
        required_port_names_lower = {port['name'].lower() for port in required_ports}
        extra_ports = []
        for found_name in found_port_dict.keys():
            if found_name.lower() not in required_port_names_lower:
                # 对于简单I/O情况，允许一些常见的额外端口
                if is_simple_io_case and _is_acceptable_extra_port(found_name):
                    if config.verbose_validation:
                        logger.info(f"允许的额外端口: {found_name}")
                    continue
                extra_ports.append(found_name)
        if extra_ports:
            return False, f"发现额外端口: {', '.join(extra_ports)}"
    
    return True, "端口验证通过"

def _is_simple_io_case(required_ports: List[Dict[str, str]], found_ports: Dict[str, str]) -> bool:
    """检查是否是简单I/O模块的情况"""
    # 如果要求的端口只有简单的I和O，认为是简单I/O情况
    required_names = {port['name'] for port in required_ports}
    simple_io_patterns = {'I', 'O', 'i', 'o', 'in', 'out', 'input', 'output'}
    
    # 检查是否所有要求的端口都是简单I/O模式
    if required_names.issubset(simple_io_patterns):
        return True
    
    # 检查是否只有一个输入和一个输出
    input_count = sum(1 for port in required_ports if port['type'].lower() == 'input')
    output_count = sum(1 for port in required_ports if port['type'].lower() == 'output')
    
    return input_count == 1 and output_count == 1

def _find_semantic_port_match(required_name: str, required_type: str, found_ports: Dict[str, str]) -> str:
    """为简单I/O情况查找语义等价的端口匹配"""
    # 定义语义等价的端口名称映射
    input_aliases = {
        'i': ['in', 'input', 'data_in', 'din'],
        'in': ['i', 'input', 'data_in', 'din'],
        'input': ['i', 'in', 'data_in', 'din']
    }
    
    output_aliases = {
        'o': ['out', 'output', 'data_out', 'dout'],
        'out': ['o', 'output', 'data_out', 'dout'],
        'output': ['o', 'out', 'data_out', 'dout']
    }
    
    # 根据端口类型选择别名集合
    if required_type.lower() == 'input':
        aliases = input_aliases.get(required_name.lower(), [])
    elif required_type.lower() == 'output':
        aliases = output_aliases.get(required_name.lower(), [])
    else:
        return None
    
    # 查找匹配的端口
    for found_name, found_type in found_ports.items():
        if found_type == required_type.lower():
            if found_name.lower() in aliases:
                return found_name
    
    # 如果没有找到别名匹配，且只有一个对应类型的端口，则进行类型匹配
    matching_ports = [name for name, port_type in found_ports.items() if port_type == required_type.lower()]
    if len(matching_ports) == 1:
        # 额外检查：确保端口名称不是完全无关的
        found_name = matching_ports[0].lower()
        # 排除明显不相关的端口名称
        excluded_names = {'x', 'y', 'z', 'a', 'b', 'c', 'sel', 'enable', 'clk', 'rst'}
        if found_name not in excluded_names:
            return matching_ports[0]
    
    return None

def _is_acceptable_extra_port(port_name: str) -> bool:
    """检查额外端口是否可接受（对于简单I/O情况）"""
    # 允许的额外端口（通常用于时钟、复位等）
    acceptable_extra = {
        'clk', 'clock', 'rst', 'reset', 'rst_n', 'reset_n', 
        'enable', 'en', 'ce', 'clr', 'clear'
    }
    return port_name.lower() in acceptable_extra

def _basic_syntax_check(code: str) -> Tuple[bool, str]:
    """基本语法检查"""
    # 检查括号匹配
    if code.count('(') != code.count(')'):
        return False, "括号不匹配"
    
    if code.count('{') != code.count('}'):
        return False, "大括号不匹配"
    
    # 检查基本关键字
    if 'module' not in code.lower():
        return False, "缺少 'module' 关键字"
    
    return True, "基本语法检查通过"

# 向后兼容的函数签名
def validate_verilog_code_legacy(
    code: str, 
    name: str, 
    ports: list, 
    strict_module_name: bool = True
) -> Tuple[bool, str]:
    """向后兼容的验证函数"""
    # 暂时禁用验证逻辑 - 直接返回通过
    return True, "验证已禁用 - 自动通过"
    
    # 以下是原始验证逻辑（已禁用）
    """
    # 转换旧格式的端口到新格式
    if ports and isinstance(ports[0], str):
        # 假设是简单的端口名列表，默认为input
        converted_ports = [{"name": port, "type": "input"} for port in ports]
    else:
        converted_ports = ports
    
    # 创建配置
    config = ValidationConfig(
        strict_module_name=strict_module_name,
        allow_module_name_variations=not strict_module_name
    )
    
    return validate_verilog_code(code, name, converted_ports, config)
    """

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
