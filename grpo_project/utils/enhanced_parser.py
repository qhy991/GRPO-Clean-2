#!/usr/bin/env python3
"""
增强的LLM输出解析器
解决代码解析失败问题，支持多种输出格式
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class EnhancedOutputParser:
    """增强的输出解析器，支持多种格式和自动修复"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.parsing_stats = {
            'total_attempts': 0,
            'successful_parses': 0,
            'format_fixes_applied': 0,
            'emergency_extractions': 0,
            'complete_failures': 0
        }
    
    def parse_llm_output(self, raw_output: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        解析LLM输出，提取reasoning和code
        
        Args:
            raw_output: 原始LLM输出
            context: 上下文信息（用于调试）
            
        Returns:
            Tuple[reasoning, code]: 解析出的推理过程和代码
        """
        self.parsing_stats['total_attempts'] += 1
        
        if not raw_output or not raw_output.strip():
            self.parsing_stats['complete_failures'] += 1
            return None, None
        
        # 1. 预处理：清理和标准化输出
        cleaned_output = self._preprocess_output(raw_output)
        
        # 2. 尝试标准格式解析
        reasoning, code = self._parse_standard_format(cleaned_output)
        if reasoning and code:
            self.parsing_stats['successful_parses'] += 1
            return reasoning, code
        
        # 3. 尝试格式修复
        fixed_output = self._fix_output_format(cleaned_output)
        if fixed_output != cleaned_output:
            self.parsing_stats['format_fixes_applied'] += 1
            reasoning, code = self._parse_standard_format(fixed_output)
            if reasoning and code:
                self.parsing_stats['successful_parses'] += 1
                return reasoning, code
        
        # 4. 尝试应急提取
        reasoning, code = self._emergency_extraction(cleaned_output)
        if reasoning or code:
            self.parsing_stats['emergency_extractions'] += 1
            if reasoning and code:
                self.parsing_stats['successful_parses'] += 1
            return reasoning, code
        
        # 5. 完全失败
        self.parsing_stats['complete_failures'] += 1
        if self.debug_mode:
            self._log_parsing_failure(raw_output, context)
        
        return None, None
    
    def _preprocess_output(self, raw_output: str) -> str:
        """预处理输出，移除不必要的标记"""
        output = raw_output.strip()
        
        # 移除Qwen对话标记 - 修复版本
        # 先移除完整的对话标记对
        output = re.sub(r'<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>', r'\1', output, flags=re.DOTALL)
        output = re.sub(r'<\|im_start\|>system\s*(.*?)\s*<\|im_end\|>', '', output, flags=re.DOTALL)
        output = re.sub(r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>', '', output, flags=re.DOTALL)
        
        # 移除剩余的单独标记
        output = re.sub(r'<\|im_start\|>assistant\n?', '', output)
        output = re.sub(r'<\|im_start\|>.*?\n?', '', output)
        output = re.sub(r'<\|im_end\|>', '', output)
        
        # 移除其他常见的对话标记
        output = re.sub(r'<\|begin_of_text\|>.*?<\|end_of_text\|>', '', output, flags=re.DOTALL)
        output = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', output, flags=re.DOTALL)
        
        # 清理多余的空白
        output = re.sub(r'\n\s*\n\s*\n', '\n\n', output)
        output = output.strip()
        
        return output
    
    def _parse_standard_format(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """解析标准格式：<think>...</think> 和 ```verilog...```"""
        reasoning = None
        code = None
        
        # 提取thinking部分
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, output, re.DOTALL | re.IGNORECASE)
        if think_match:
            reasoning = think_match.group(1).strip()
        
        # 提取verilog代码
        verilog_pattern = r'```verilog\s*(.*?)```'
        verilog_match = re.search(verilog_pattern, output, re.DOTALL | re.IGNORECASE)
        if verilog_match:
            code = verilog_match.group(1).strip()
        
        # 如果没有找到verilog标记，尝试普通代码块
        if not code:
            code_pattern = r'```\s*(.*?)```'
            code_matches = re.findall(code_pattern, output, re.DOTALL)
            for match in code_matches:
                if 'module' in match.lower() and 'endmodule' in match.lower():
                    code = match.strip()
                    break
        
        return reasoning, code
    
    def _fix_output_format(self, output: str) -> str:
        """尝试修复输出格式"""
        fixed_output = output
        
        # 1. 修复缺失的think标签
        fixed_output = self._fix_think_tags(fixed_output)
        
        # 2. 修复缺失的代码块标记
        fixed_output = self._fix_code_blocks(fixed_output)
        
        # 3. 修复嵌套或重复的标记
        fixed_output = self._fix_nested_tags(fixed_output)
        
        return fixed_output
    
    def _fix_think_tags(self, output: str) -> str:
        """修复think标签"""
        # 检查是否有think开始但没有结束
        has_think_start = '<think>' in output.lower()
        has_think_end = '</think>' in output.lower()
        
        if has_think_start and not has_think_end:
            # 找到代码块的位置，在其前面添加</think>
            verilog_pos = output.lower().find('```verilog')
            if verilog_pos == -1:
                verilog_pos = output.lower().find('```')
            if verilog_pos == -1:
                verilog_pos = output.lower().find('module')
            
            if verilog_pos > 0:
                output = output[:verilog_pos] + '</think>\n\n' + output[verilog_pos:]
        
        elif not has_think_start and not has_think_end:
            # 尝试识别思考内容并添加标签
            verilog_pos = output.lower().find('```verilog')
            if verilog_pos == -1:
                verilog_pos = output.lower().find('```')
            if verilog_pos == -1:
                verilog_pos = output.lower().find('module')
            
            if verilog_pos > 50:  # 有足够的前置内容
                thinking_content = output[:verilog_pos].strip()
                remaining_content = output[verilog_pos:]
                output = f"<think>\n{thinking_content}\n</think>\n\n{remaining_content}"
        
        return output
    
    def _fix_code_blocks(self, output: str) -> str:
        """修复代码块标记"""
        # 如果没有```verilog标记，但有module...endmodule
        if '```verilog' not in output.lower():
            module_pattern = r'(module\s+.*?endmodule)'
            module_match = re.search(module_pattern, output, re.DOTALL | re.IGNORECASE)
            if module_match:
                module_code = module_match.group(1)
                output = output.replace(module_code, f'```verilog\n{module_code}\n```')
        
        # 修复不完整的代码块
        verilog_start = output.lower().find('```verilog')
        if verilog_start >= 0:
            after_start = output[verilog_start + 10:]  # 跳过```verilog
            if '```' not in after_start:
                # 找到endmodule的位置
                endmodule_match = re.search(r'(.*?endmodule)', after_start, re.DOTALL | re.IGNORECASE)
                if endmodule_match:
                    verilog_content = endmodule_match.group(1).strip()
                    remaining = after_start[endmodule_match.end():].strip()
                    before_verilog = output[:verilog_start]
                    if remaining:
                        output = f"{before_verilog}```verilog\n{verilog_content}\n```\n{remaining}"
                    else:
                        output = f"{before_verilog}```verilog\n{verilog_content}\n```"
        
        return output
    
    def _fix_nested_tags(self, output: str) -> str:
        """修复嵌套或重复的标记"""
        # 移除重复的think标签
        output = re.sub(r'<think>\s*<think>', '<think>', output, flags=re.IGNORECASE)
        output = re.sub(r'</think>\s*</think>', '</think>', output, flags=re.IGNORECASE)
        
        # 移除重复的代码块标记
        output = re.sub(r'```verilog\s*```verilog', '```verilog', output, flags=re.IGNORECASE)
        output = re.sub(r'```\s*```', '```', output)
        
        return output
    
    def _emergency_extraction(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """应急提取：使用更宽松的规则"""
        reasoning = None
        code = None
        
        # 1. 尝试提取任何形式的思考内容
        reasoning_patterns = [
            r'<think>(.*?)</think>',
            r'思考[：:](.*?)(?=```|module|$)',
            r'分析[：:](.*?)(?=```|module|$)',
            r'解决方案[：:](.*?)(?=```|module|$)',
            r'^(.*?)(?=```verilog|```|module)',
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                potential_reasoning = match.group(1).strip()
                if len(potential_reasoning) > 20:  # 足够长的内容
                    reasoning = potential_reasoning
                    break
        
        # 2. 尝试提取任何形式的Verilog代码
        code_patterns = [
            r'```verilog\s*(.*?)```',
            r'```\s*(module\s+.*?endmodule).*?```',
            r'(module\s+\w+.*?endmodule)',
            r'```([^`]*module[^`]*endmodule[^`]*)```',
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                potential_code = match.group(1).strip()
                if 'module' in potential_code.lower() and len(potential_code) > 30:
                    code = potential_code
                    break
        
        # 3. 如果还是没找到代码，尝试更宽松的匹配
        if not code:
            lines = output.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line_lower = line.lower().strip()
                if 'module' in line_lower and not in_code_block:
                    in_code_block = True
                    code_lines.append(line)
                elif in_code_block:
                    code_lines.append(line)
                    if 'endmodule' in line_lower:
                        break
            
            if code_lines and len(code_lines) > 3:
                code = '\n'.join(code_lines).strip()
        
        return reasoning, code
    
    def _log_parsing_failure(self, raw_output: str, context: Optional[Dict[str, Any]]):
        """记录解析失败的详细信息"""
        logger.error("="*60)
        logger.error("PARSING FAILURE DETECTED")
        logger.error("="*60)
        
        if context:
            logger.error(f"Context: {context}")
        
        logger.error(f"Raw output length: {len(raw_output)}")
        logger.error(f"Raw output preview: {raw_output[:500]}...")
        
        # 检查常见模式
        patterns_to_check = [
            ('<think>', raw_output.lower().count('<think>')),
            ('</think>', raw_output.lower().count('</think>')),
            ('```verilog', raw_output.lower().count('```verilog')),
            ('```', raw_output.count('```')),
            ('module', raw_output.lower().count('module')),
            ('endmodule', raw_output.lower().count('endmodule')),
        ]
        
        logger.error("Pattern analysis:")
        for pattern, count in patterns_to_check:
            logger.error(f"  {pattern}: {count} occurrences")
        
        logger.error("="*60)
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """获取解析统计信息"""
        total = self.parsing_stats['total_attempts']
        if total == 0:
            return self.parsing_stats
        
        stats = self.parsing_stats.copy()
        stats['success_rate'] = self.parsing_stats['successful_parses'] / total
        stats['fix_rate'] = self.parsing_stats['format_fixes_applied'] / total
        stats['emergency_rate'] = self.parsing_stats['emergency_extractions'] / total
        stats['failure_rate'] = self.parsing_stats['complete_failures'] / total
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.parsing_stats = {
            'total_attempts': 0,
            'successful_parses': 0,
            'format_fixes_applied': 0,
            'emergency_extractions': 0,
            'complete_failures': 0
        }

def create_enhanced_parsing_function():
    """创建增强的解析函数，用于替换现有的解析逻辑"""
    parser = EnhancedOutputParser(debug_mode=True)
    
    def enhanced_parse_llm_completion_with_context(completion_text: str, prompt: str = None,
                                                 step: int = None, sample_idx: int = None) -> Tuple[Optional[str], Optional[str]]:
        """增强的解析函数，兼容现有接口"""
        context = {}
        if step is not None:
            context["step"] = step
        if sample_idx is not None:
            context["sample_idx"] = sample_idx
        if prompt:
            context["prompt_preview"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
        
        return parser.parse_llm_output(completion_text, context)
    
    return enhanced_parse_llm_completion_with_context, parser

# 使用示例
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 标准格式
        """<think>
        I need to design a simple counter module.
        </think>
        
        ```verilog
        module counter(
            input clk,
            input rst,
            output reg [7:0] count
        );
        always @(posedge clk or posedge rst) begin
            if (rst) count <= 0;
            else count <= count + 1;
        end
        endmodule
        ```""",
        
        # 缺少think标签
        """I need to design a simple counter module.
        This will increment on each clock cycle.
        
        ```verilog
        module counter(
            input clk,
            input rst,
            output reg [7:0] count
        );
        always @(posedge clk or posedge rst) begin
            if (rst) count <= 0;
            else count <= count + 1;
        end
        endmodule
        ```""",
        
        # 缺少代码块标记
        """<think>
        I need to design a simple counter module.
        </think>
        
        module counter(
            input clk,
            input rst,
            output reg [7:0] count
        );
        always @(posedge clk or posedge rst) begin
            if (rst) count <= 0;
            else count <= count + 1;
        end
        endmodule""",
        
        # Qwen格式
        """<|im_start|>assistant
        <think>
        I need to design a simple counter module.
        </think>
        
        ```verilog
        module counter(
            input clk,
            input rst,
            output reg [7:0] count
        );
        always @(posedge clk or posedge rst) begin
            if (rst) count <= 0;
            else count <= count + 1;
        end
        endmodule
        ```
        <|im_end|>""",
    ]
    
    parser = EnhancedOutputParser(debug_mode=True)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}")
        print(f"{'='*50}")
        
        reasoning, code = parser.parse_llm_output(test_case)
        
        print(f"解析结果:")
        print(f"  Reasoning: {'✓' if reasoning else '✗'} ({len(reasoning) if reasoning else 0} chars)")
        print(f"  Code: {'✓' if code else '✗'} ({len(code) if code else 0} chars)")
        
        if reasoning:
            print(f"  Reasoning preview: {reasoning[:100]}...")
        if code:
            print(f"  Code preview: {code[:100]}...")
    
    print(f"\n{'='*50}")
    print("解析统计:")
    print(f"{'='*50}")
    stats = parser.get_parsing_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}") 