# grpo_project/utils/streaming_guidance.py
import torch
import re
import logging
from typing import List, Dict, Optional, Tuple, Generator, Any
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from dataclasses import dataclass
import time
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GuidanceConfig:
    """流式引导配置"""
    min_reasoning_length: int = 50  # 最小思考长度
    guidance_trigger_threshold: int = 30  # 触发引导的阈值
    max_guidance_attempts: int = 5  # 增加最大引导尝试次数到5次
    guidance_tokens_limit: int = 20  # 每次引导添加的最大token数
    stop_on_code_start: bool = True  # 遇到代码块开始时停止
    save_failed_generations: bool = True  # 是否保存失败的生成
    save_successful_generations: bool = True  # 是否保存成功的生成
    failed_generations_dir: str = "failed_generations"  # 失败生成保存目录
    successful_generations_dir: str = "successful_generations"  # 成功生成保存目录
    
class ThinkingGuidanceManager:
    """思考过程引导管理器"""
    
    def __init__(self, config: GuidanceConfig):
        self.config = config
        self.guidance_phrases = [
            # 分析阶段引导词
            "First, I need to analyze the requirements carefully. ",
            "Let me break down this problem step by step. ",
            "The key requirements for this module are: ",
            "I should start by identifying the input and output signals. ",
            
            # 设计阶段引导词
            "For the design approach, I will implement ",
            "The module architecture should include ",
            "The main logic blocks needed are: ",
            "I need to consider the following design constraints: ",
            
            # 实现阶段引导词
            "For the implementation, I will use ",
            "The core logic can be implemented using ",
            "I should structure the code with these main sections: ",
            "The signal flow will be organized as follows: ",
            
            # 验证阶段引导词
            "To ensure correctness, I should verify that ",
            "The edge cases I need to handle include: ",
            "For proper timing, I must ensure ",
            "The output behavior should satisfy these conditions: "
        ]
        
    def select_guidance_phrase(self, current_text: str, attempt: int) -> str:
        """根据当前文本和尝试次数选择合适的引导词"""
        
        # 根据当前内容选择引导类型
        current_lower = current_text.lower()
        
        # 如果没有分析内容，优先分析引导
        if not any(word in current_lower for word in ['analyze', 'requirement', 'input', 'output']):
            candidates = self.guidance_phrases[:4]  # 分析阶段
        # 如果没有设计内容，优先设计引导  
        elif not any(word in current_lower for word in ['design', 'implement', 'architecture', 'logic']):
            candidates = self.guidance_phrases[4:8]  # 设计阶段
        # 如果没有实现内容，优先实现引导
        elif not any(word in current_lower for word in ['implementation', 'code', 'structure']):
            candidates = self.guidance_phrases[8:12]  # 实现阶段
        # 否则使用验证引导
        else:
            candidates = self.guidance_phrases[12:]  # 验证阶段
            
        # 根据尝试次数选择（避免重复）
        index = attempt % len(candidates)
        return candidates[index]

class StreamingGuidanceGenerator:
    """流式引导生成器"""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 config: GuidanceConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GuidanceConfig()
        self.guidance_manager = ThinkingGuidanceManager(self.config)
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 创建生成保存目录
        if self.config.save_failed_generations:
            os.makedirs(self.config.failed_generations_dir, exist_ok=True)
        if self.config.save_successful_generations:
            os.makedirs(self.config.successful_generations_dir, exist_ok=True)
        
    def _save_generation(self, prompt: str, generated_text: str, guidance_applied: List[Dict], 
                        step: int = 0, is_successful: bool = True, reason: str = None):
        """保存生成结果（成功或失败）"""
        if (is_successful and not self.config.save_successful_generations) or \
           (not is_successful and not self.config.save_failed_generations):
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = self.config.successful_generations_dir if is_successful else self.config.failed_generations_dir
        prefix = "success" if is_successful else "failed"
        
        filename = os.path.join(
            dir_name,
            f"{prefix}_gen_step{step}_{timestamp}.json"
        )
        
        try:
            data = {
                "timestamp": timestamp,
                "step": step,
                "is_successful": is_successful,
                "prompt": prompt,
                "generated_text": generated_text,
                "guidance_applied": guidance_applied,
                "config": {
                    "min_reasoning_length": self.config.min_reasoning_length,
                    "guidance_trigger_threshold": self.config.guidance_trigger_threshold,
                    "max_guidance_attempts": self.config.max_guidance_attempts,
                    "guidance_tokens_limit": self.config.guidance_tokens_limit
                }
            }
            
            if not is_successful and reason:
                data["failure_reason"] = reason
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            status = "成功" if is_successful else "失败"
            logger.info(f"💾 保存{status}生成到: {filename}")
            
        except Exception as e:
            logger.error(f"保存生成结果时出错: {e}")

    def generate_with_guidance(self, 
                             prompt: str,
                             max_new_tokens: int = 512,
                             temperature: float = 0.8,
                             top_p: float = 0.95,
                             step: int = 0,
                             **generation_kwargs) -> Dict[str, Any]:
        """
        带引导的流式生成
        
        Returns:
            Dict包含generated_text, guidance_applied, reasoning_part, code_part等
        """
        
        logger.info(f"🚀 开始流式引导生成，max_tokens={max_new_tokens}")
        
        # 初始化状态
        generated_text = ""
        guidance_applied = []
        reasoning_part = ""
        code_part = ""
        guidance_attempts = 0
        
        # 准备初始输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 流式生成状态跟踪
        in_think_block = False
        think_content = ""
        generated_tokens = 0
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # 生成下一个token
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                # 应用采样策略
                logits = outputs.logits[:, -1, :]
                if temperature > 0:
                    logits = logits / temperature
                    
                # Top-p采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # 采样下一个token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 解码token
                next_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated_text += next_text
                generated_tokens += 1
                
                # 更新输入序列
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                
                # 解析当前状态
                parsing_result = self._parse_current_output(generated_text)
                
                # 检查是否进入think块
                if '<think>' in generated_text and not in_think_block:
                    in_think_block = True
                    logger.debug("🤔 检测到进入think块")
                    
                # 如果在think块中，累积内容
                if in_think_block and not parsing_result['think_closed']:
                    think_content = parsing_result['think_content']
                    
                    # 检查是否需要引导
                    should_guide, reason = self._should_apply_guidance(
                        think_content, generated_tokens, guidance_attempts
                    )
                    
                    if should_guide:
                        logger.info(f"🎯 触发引导 - 原因: {reason}")
                        guidance_text = self._apply_guidance(
                            think_content, guidance_attempts
                        )
                        
                        if guidance_text:
                            # 将引导文本编码并添加到序列中
                            guidance_tokens = self.tokenizer.encode(
                                guidance_text, 
                                add_special_tokens=False,
                                return_tensors="pt"
                            ).to(self.model.device)
                            
                            # 更新序列
                            input_ids = torch.cat([input_ids, guidance_tokens], dim=-1)
                            attention_mask = torch.cat([
                                attention_mask, 
                                torch.ones_like(guidance_tokens)
                            ], dim=-1)
                            
                            generated_text += guidance_text
                            guidance_applied.append({
                                'step': step,
                                'reason': reason,
                                'guidance': guidance_text,
                                'attempt': guidance_attempts
                            })
                            
                            guidance_attempts += 1
                            logger.info(f"✅ 应用引导 (尝试 {guidance_attempts}): {guidance_text}")
                
                # 检查停止条件
                if self._should_stop_generation(generated_text, parsing_result):
                    logger.info(f"🛑 生成停止 - 检测到完整输出或代码块")
                    break
                    
                # 检查think块是否关闭
                if parsing_result['think_closed'] and in_think_block:
                    in_think_block = False
                    reasoning_part = parsing_result['think_content']
                    logger.debug(f"✅ Think块完成，长度: {len(reasoning_part)}")
                    
                # 检查是否达到最大引导次数
                if guidance_attempts >= self.config.max_guidance_attempts:
                    logger.warning(f"⚠️ 达到最大引导次数 ({guidance_attempts})")
        
        # 最终解析
        final_result = self._parse_current_output(generated_text)
        
        result = {
            'generated_text': generated_text,
            'reasoning_part': final_result['think_content'],
            'code_part': final_result['code_content'], 
            'guidance_applied': guidance_applied,
            'guidance_count': len(guidance_applied),
            'final_reasoning_length': len(final_result['think_content']),
            'generation_successful': self._is_generation_successful(final_result),
            'tokens_generated': generated_tokens
        }
        
        logger.info(f"🏁 生成完成: tokens={generated_tokens}, "
                   f"guidance_count={len(guidance_applied)}, "
                   f"reasoning_length={len(final_result['think_content'])}")
        
        # 在生成结束时检查是否成功
        is_successful = self._is_generation_successful(final_result)
        if not is_successful:
            failure_reason = "generation_incomplete"
            if guidance_attempts >= self.config.max_guidance_attempts:
                failure_reason = "max_guidance_attempts_reached"
            elif not final_result['think_closed']:
                failure_reason = "incomplete_thinking"
            elif not final_result['code_closed']:
                failure_reason = "incomplete_code"
            elif len(final_result['think_content']) < self.config.min_reasoning_length:
                failure_reason = "insufficient_reasoning"
            elif 'module' not in final_result['code_content'].lower():
                failure_reason = "missing_module_declaration"
        else:
            failure_reason = None
        
        # 保存生成结果（无论成功还是失败）
        self._save_generation(
            prompt=prompt,
            generated_text=generated_text,
            guidance_applied=guidance_applied,
            step=step,
            is_successful=is_successful,
            reason=failure_reason
        )
        
        return result
    
    def _parse_current_output(self, text: str) -> Dict[str, Any]:
        """解析当前输出状态"""
        
        # 检查think块状态
        think_start = text.find('<think>')
        think_end = text.find('</think>')
        
        think_content = ""
        think_closed = False
        
        if think_start >= 0:
            if think_end > think_start:
                think_content = text[think_start + 7:think_end].strip()
                think_closed = True
            else:
                think_content = text[think_start + 7:].strip()
                think_closed = False
        
        # 检查代码块状态
        code_start = text.find('```verilog')
        code_end = text.rfind('```')
        
        code_content = ""
        code_closed = False
        
        if code_start >= 0:
            if code_end > code_start + 10:  # 确保不是同一个```
                code_content = text[code_start + 10:code_end].strip()
                code_closed = True
            else:
                code_content = text[code_start + 10:].strip()
                code_closed = False
        
        return {
            'think_content': think_content,
            'think_closed': think_closed,
            'code_content': code_content,
            'code_closed': code_closed,
            'has_think_start': think_start >= 0,
            'has_code_start': code_start >= 0
        }
    
    def _should_apply_guidance(self, 
                             think_content: str, 
                             tokens_generated: int,
                             guidance_attempts: int) -> Tuple[bool, str]:
        """判断是否应该应用引导"""
        
        # 检查引导次数限制
        if guidance_attempts >= self.config.max_guidance_attempts:
            return False, "max_attempts_reached"
        
        # 检查think内容长度
        if len(think_content.strip()) < self.config.guidance_trigger_threshold:
            return True, f"think_too_short_{len(think_content)}"
        
        # 检查内容质量
        if self._is_low_quality_thinking(think_content):
            return True, "low_quality_content"
            
        return False, "no_guidance_needed"
    
    def _is_low_quality_thinking(self, think_content: str) -> bool:
        """检查思考内容是否质量较低"""
        
        if not think_content.strip():
            return True
            
        # 检查无意义开头
        bad_starts = ['well', 'i will', 'let me', 'so', 'ok', 'hmm']
        first_words = think_content.lower().strip().split()[:3]
        if any(start in ' '.join(first_words) for start in bad_starts):
            return True
            
        # 检查是否缺少技术词汇
        tech_words = ['module', 'input', 'output', 'logic', 'implement', 'design', 'signal']
        has_tech_words = any(word in think_content.lower() for word in tech_words)
        
        return not has_tech_words
    
    def _apply_guidance(self, current_think: str, attempt: int) -> str:
        """应用引导文本"""
        
        guidance_phrase = self.guidance_manager.select_guidance_phrase(
            current_think, attempt
        )
        
        # 添加适当的连接词
        if current_think.strip() and not current_think.strip().endswith('.'):
            guidance_text = ". " + guidance_phrase
        else:
            guidance_text = guidance_phrase
            
        return guidance_text
    
    def _should_stop_generation(self, generated_text: str, parsing_result: Dict) -> bool:
        """判断是否应该停止生成"""
        
        # 如果检测到完整的think和code块
        if (parsing_result['think_closed'] and 
            parsing_result['code_closed'] and
            len(parsing_result['think_content']) >= self.config.min_reasoning_length):
            return True
            
        # 如果检测到多个```（可能出错）
        if generated_text.count('```') > 2:
            return True
            
        return False
    
    def _is_generation_successful(self, parsing_result: Dict) -> bool:
        """判断生成是否成功"""
        
        return (
            parsing_result['think_closed'] and
            parsing_result['code_closed'] and
            len(parsing_result['think_content']) >= self.config.min_reasoning_length and
            'module' in parsing_result['code_content'].lower()
        )

# grpo_project/utils/streaming_integration.py
class StreamingInferenceIntegration:
    """将流式引导集成到现有推理系统中"""
    
    def __init__(self, model, tokenizer, config: GuidanceConfig = None):
        self.streaming_generator = StreamingGuidanceGenerator(model, tokenizer, config)
        
    def enhance_inference_callback(self, inference_callback):
        """增强现有的推理回调以支持流式引导"""
        
        original_generate = inference_callback._generate_single_sample
        
        def enhanced_generate(model, prompt, step):
            """替换原有的生成方法"""
            
            # 使用流式引导生成
            result = self.streaming_generator.generate_with_guidance(
                prompt=prompt,
                max_new_tokens=inference_callback.max_new_tokens,
                temperature=0.8,
                top_p=0.95
            )
            
            # 转换为原有格式
            return {
                'reasoning': result['reasoning_part'],
                'code': result['code_part'],
                'raw_output': result['generated_text'],
                'generation_time': 0.0,  # 流式生成时间计算较复杂，可以后续添加
                'step': step,
                'guidance_applied': result['guidance_applied'],
                'guidance_count': result['guidance_count'],
                'error': None if result['generation_successful'] else "Generation incomplete"
            }
        
        # 替换方法
        inference_callback._generate_single_sample = enhanced_generate
        return inference_callback

# 使用示例
def create_streaming_inference_callback(model, tokenizer, eval_dataset, **kwargs):
    """创建支持流式引导的推理回调"""
    
    from grpo_project.callbacks.inference import DetailedInferenceCallback
    
    # 创建基础回调，移除guidance_config参数
    base_callback = DetailedInferenceCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        **{k: v for k, v in kwargs.items() if k != 'guidance_config'}
    )
    
    # 创建流式引导配置
    guidance_config = GuidanceConfig(
        min_reasoning_length=60,
        guidance_trigger_threshold=40,
        max_guidance_attempts=5,  # 增加到5次
        save_failed_generations=True,  # 启用失败生成保存
        save_successful_generations=True,  # 启用成功生成保存
        failed_generations_dir=os.path.join(kwargs.get('output_dir', '.'), 'failed_generations'),
        successful_generations_dir=os.path.join(kwargs.get('output_dir', '.'), 'successful_generations')
    )
    
    # 集成流式引导
    integration = StreamingInferenceIntegration(model, tokenizer, guidance_config)
    enhanced_callback = integration.enhance_inference_callback(base_callback)
    
    return enhanced_callback