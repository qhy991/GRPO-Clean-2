from typing import Optional, Dict, Any, List, Tuple
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class StreamState:
    """流式输出状态管理"""
    current_reasoning: str = ""
    current_code: str = ""
    has_think_start: bool = False
    has_think_end: bool = False
    has_code_start: bool = False
    has_code_end: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def reset(self):
        """重置状态"""
        self.__init__()
    
    def update_reasoning(self, text: str) -> bool:
        """更新推理内容并检查是否需要引导"""
        self.current_reasoning += text
        
        # 检查思考标签
        if '<think>' in text.lower():
            self.has_think_start = True
        if '</think>' in text.lower():
            self.has_think_end = True
            
        # 检查推理内容长度
        if len(self.current_reasoning) < 50 and not self.has_code_start:
            return True  # 需要引导
        return False
    
    def update_code(self, text: str) -> bool:
        """更新代码内容并检查是否需要引导"""
        self.current_code += text
        
        # 检查代码块标记
        if '```verilog' in text.lower():
            self.has_code_start = True
        if '```' in text and self.has_code_start:
            self.has_code_end = True
            
        # 检查代码内容
        if self.has_code_start and not self.has_code_end:
            if 'module' not in self.current_code.lower():
                self.warnings.append("代码块中未找到module关键字")
            if len(self.current_code) < 20:
                return True  # 需要引导
        return False
    
    def get_guidance_prompt(self) -> Optional[str]:
        """根据当前状态生成引导提示"""
        if not self.has_think_start and not self.has_code_start:
            return "请开始你的思考过程，使用<think>标签包裹。"
        
        if self.has_think_start and not self.has_think_end and len(self.current_reasoning) < 50:
            return "请继续详细说明你的思考过程。"
            
        if self.has_code_start and not self.has_code_end:
            if 'module' not in self.current_code.lower():
                return "请开始编写Verilog模块代码。"
            if len(self.current_code) < 20:
                return "请继续完善代码实现。"
                
        return None
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return {
            "reasoning_length": len(self.current_reasoning),
            "code_length": len(self.current_code),
            "has_think_tags": self.has_think_start and self.has_think_end,
            "has_code_tags": self.has_code_start and self.has_code_end,
            "warnings": self.warnings,
            "errors": self.errors
        }

class StreamParser:
    """流式输出解析器"""
    def __init__(self):
        self.state = StreamState()
        
    def process_chunk(self, chunk: str) -> Tuple[str, Optional[str]]:
        """
        处理新的输出块
        返回: (处理后的块, 引导提示)
        """
        # 检查是否包含代码块
        if '```verilog' in chunk.lower():
            self.state.has_code_start = True
            return chunk, self.state.get_guidance_prompt()
            
        # 检查是否包含思考标签
        if '<think>' in chunk.lower():
            self.state.has_think_start = True
        if '</think>' in chunk.lower():
            self.state.has_think_end = True
            
        # 根据当前状态更新内容
        if not self.state.has_code_start:
            needs_guidance = self.state.update_reasoning(chunk)
        else:
            needs_guidance = self.state.update_code(chunk)
            
        # 获取引导提示
        guidance = self.state.get_guidance_prompt() if needs_guidance else None
        
        return chunk, guidance
    
    def reset(self):
        """重置解析器状态"""
        self.state.reset()
        
    def get_debug_info(self) -> Dict[str, Any]:
        """获取当前解析状态信息"""
        return self.state.get_debug_info() 