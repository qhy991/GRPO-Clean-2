# qwen3_prompt_fix.py - 完整版本，包含所有必要的导入

import re
import os
import logging
import json
import time
import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from datasets import Dataset
from transformers import (
    TrainerCallback, 
    TrainingArguments, 
    TrainerState, 
    TrainerControl,
    GenerationConfig
)
from datetime import datetime

# 获取logger
logger = logging.getLogger(__name__)

# 导入必要的工具函数
try:
    from grpo_project.utils.file_ops import validate_and_update_dataset_paths
    from grpo_project.utils.prompt_utils import enhance_prompt_func
    from grpo_project.utils.verilog_utils import assess_code_quality
except ImportError:
    logger.warning("某些grpo_project.utils函数导入失败，将使用简化版本")
    
    # 简化版本的函数
    def validate_and_update_dataset_paths(dataset, dataset_base_path=None):
        """简化版本的路径验证函数"""
        if dataset is None:
            return []
        
        processed_examples = []
        for example in dataset:
            # 基本验证
            required_keys = ['prompt', 'testbench_path', 'expected_total_tests', 'reference_verilog_path']
            if all(key in example and example[key] is not None for key in required_keys):
                processed_examples.append(example)
        
        return processed_examples
    
    def enhance_prompt_func(example):
        """简化版本的提示增强函数"""
        if isinstance(example, dict) and 'prompt' in example:
            # 保持原样或做基本处理
            return {
                "prompt": example['prompt'],
                "original_prompt_for_debug": example.get('prompt', '')
            }
        return example
    
    def assess_code_quality(code):
        """简化版本的代码质量评估"""
        if not code:
            return {"complexity": 0, "readability": 0, "efficiency": 0, "structure": 0}
        
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # 简单的质量评估
        has_module = 'module' in code.lower()
        has_endmodule = 'endmodule' in code.lower()
        has_comments = any('//' in line or '/*' in line for line in lines)
        
        return {
            "complexity": 0.8 if has_module and has_endmodule else 0.3,
            "readability": 0.7 if has_comments else 0.5,
            "efficiency": 0.6,
            "structure": 0.8 if has_module and has_endmodule else 0.3
        }

# 修复1: wrap_prompt_for_qwen3函数
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
        logger.warning(f"遇到非字符串或空prompt: {enhanced_content}")
        example["prompt"] = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\nPlease design a Verilog module.<|im_end|>\n<|im_start|>assistant\n"
        example["original_enhanced_prompt"] = "Default prompt due to invalid input"
    
    return example

# 修复2: 增强的parse_llm_completion函数 - 更好地处理Qwen3输出
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

# 修复3: 更新parse_llm_completion_with_context函数
def parse_llm_completion_with_context_qwen3(completion_text: str, prompt: str = None, 
                                           step: int = None, sample_idx: int = None) -> Tuple[Optional[str], Optional[str]]:
    """
    专门为Qwen3优化的带上下文的解析函数
    """
    debug_context = {
        "model": "qwen3",
        "step": step,
        "sample_idx": sample_idx,
        "prompt_preview": prompt[:100] if prompt else None
    }
    
    return parse_llm_completion_qwen3(completion_text, debug_prompt=prompt, debug_context=debug_context)

# 修复4: Qwen3推理回调的生成配置
class Qwen3InferenceCallback(TrainerCallback):
    """专门为Qwen3优化的推理回调"""
    
    def __init__(self, tokenizer, eval_dataset=None, num_samples=2, eval_every_n_steps=25, 
                 max_new_tokens=512, max_seq_length=4096, output_dir=None):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        
        if output_dir:
            self.samples_dir = os.path.join(output_dir, "qwen3_generated_samples")
            os.makedirs(self.samples_dir, exist_ok=True)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            if model is None or args.local_rank > 0:
                return
            
            logger.info(f"\n🤖 === Qwen3推理回调 - 步数 {state.global_step} ===")
            
            if self.eval_dataset and len(self.eval_dataset) > 0:
                sample_indices = random.sample(range(len(self.eval_dataset)), 
                                             min(self.num_samples, len(self.eval_dataset)))
                
                for i, idx in enumerate(sample_indices):
                    sample = self.eval_dataset[idx]
                    prompt = sample['prompt']
                    
                    logger.info(f"\n📝 Qwen3样本 {i+1}/{len(sample_indices)} (数据集索引: {idx})")
                    logger.info(f"等级: {sample.get('level', 'unknown')}")
                    logger.info(f"复杂度: {sample.get('complexity_score', 'unknown')}")
                    
                    try:
                        result = self._generate_qwen3_sample(model, prompt, state.global_step)
                        
                        logger.info(f"生成时间: {result.get('generation_time', 0):.2f}秒")
                        
                        # 验证生成质量
                        reasoning = result.get('reasoning')
                        code = result.get('code')
                        
                        logger.info(f"推理部分: {'✅' if reasoning else '❌'} ({len(reasoning) if reasoning else 0} 字符)")
                        logger.info(f"代码部分: {'✅' if code else '❌'} ({len(code) if code else 0} 字符)")
                        
                        if code:
                            # 简单验证代码质量
                            has_module = 'module' in code.lower()
                            has_endmodule = 'endmodule' in code.lower()
                            logger.info(f"代码质量: module={'✅' if has_module else '❌'}, endmodule={'✅' if has_endmodule else '❌'}")
                        
                        # 保存样本
                        if self.output_dir:
                            self._save_qwen3_sample(state.global_step, i, sample, result)
                    
                    except Exception as e:
                        logger.error(f"Qwen3生成样本失败: {e}", exc_info=True)
            
            logger.info(f"🤖 === Qwen3推理回调结束 ===\n")
    
    def _generate_qwen3_sample(self, model, prompt, step):
        """生成Qwen3样本"""
        model.eval()
        
        # Qwen3特定的生成配置
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,  # Qwen3推荐的温度
            "top_p": 0.8,        # Qwen3推荐的top_p
            "top_k": 40,         # Qwen3推荐的top_k
            "repetition_penalty": 1.05,  # 轻微的重复惩罚
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 准备输入
        max_prompt_len = self.max_seq_length - self.max_new_tokens
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_prompt_len,
            padding=False
        ).to(model.device)
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            generation_time = time.time() - start_time
            
            # 解码生成的部分
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 使用Qwen3优化的解析
            reasoning, code = parse_llm_completion_qwen3(
                generated_text,
                debug_prompt=prompt,
                debug_context={"step": step, "model": "qwen3"}
            )
            
            model.train()  # 恢复训练模式
            
            return {
                'reasoning': reasoning,
                'code': code,
                'raw_output': generated_text,
                'generation_time': generation_time,
                'step': step,
                'generation_config': generation_config
            }
        
        except Exception as e:
            logger.error(f"Qwen3生成异常: {e}", exc_info=True)
            model.train()
            return {
                'reasoning': None,
                'code': None,
                'raw_output': f"Generation failed: {e}",
                'generation_time': time.time() - start_time,
                'step': step,
                'error': str(e)
            }
    
    def _save_qwen3_sample(self, step, sample_idx, original_sample, generated_result):
        """保存Qwen3生成样本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = str(original_sample.get('task_id', f"sample_{sample_idx}")).replace('/', '_')
        
        filename = f"qwen3_step_{step}_{task_id}_{timestamp}.json"
        filepath = os.path.join(self.samples_dir, filename)
        
        # 获取原始问题描述
        orig_prompt = original_sample.get('original_prompt_for_debug', 
                                        original_sample.get('original_enhanced_prompt', 'N/A'))
        
        sample_data = {
            "metadata": {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "model": "qwen3",
                "sample_idx": sample_idx,
                "task_id": original_sample.get('task_id')
            },
            "original_sample": {
                "level": original_sample.get('level'),
                "complexity_score": original_sample.get('complexity_score'),
                "category": original_sample.get('category'),
                "original_problem": orig_prompt[:500] + "..." if len(orig_prompt) > 500 else orig_prompt,
                "testbench_path": original_sample.get('testbench_path', ''),
            },
            "generation": {
                "reasoning": generated_result.get('reasoning'),
                "code": generated_result.get('code'),
                "raw_output_preview": generated_result.get('raw_output', '')[:300] + "...",
                "generation_time_seconds": generated_result.get('generation_time'),
                "generation_config": generated_result.get('generation_config', {}),
                "error": generated_result.get('error')
            },
            "quality_assessment": {
                "has_reasoning": generated_result.get('reasoning') is not None,
                "has_code": generated_result.get('code') is not None,
                "reasoning_length": len(generated_result.get('reasoning', '')),
                "code_length": len(generated_result.get('code', '')),
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Qwen3样本已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存Qwen3样本失败: {e}")

# 修复5: 在主数据处理流程中使用Qwen3格式
def qwen3_dataset_processing_pipeline(raw_ds, ds_dir, script_cfg_val):
    """专门为Qwen3优化的数据集处理流程"""
    logger.info("🤖 启动Qwen3数据集处理流程...")
    
    # 第一步：基础数据验证和类型处理
    def preprocess_for_qwen3(example):
        if not isinstance(example, dict):
            return None
        
        # 验证必需字段
        required_fields = ["prompt", "testbench_path", "expected_total_tests", "reference_verilog_path"]
        for field in required_fields:
            if field not in example or example[field] is None:
                return None
        
        # 确保数据类型正确
        processed = {}
        for key, value in example.items():
            if key in ["prompt", "testbench_path", "reference_verilog_path"]:
                processed[key] = str(value)
            elif key == "expected_total_tests":
                try:
                    processed[key] = int(value) if isinstance(value, str) and value.isdigit() else int(value)
                except (ValueError, TypeError):
                    return None
            else:
                processed[key] = value
        
        return processed
    
    # 应用预处理
    processed_ds = raw_ds.map(preprocess_for_qwen3, num_proc=1).filter(lambda x: x is not None)
    logger.info(f"Qwen3预处理后: {len(processed_ds)} 行")
    
    if len(processed_ds) == 0:
        return processed_ds
    
    # 第二步：路径验证
    validated_examples = validate_and_update_dataset_paths(processed_ds, ds_dir)
    if not validated_examples:
        return Dataset.from_list([])
    
    validated_ds = Dataset.from_list(validated_examples)
    logger.info(f"Qwen3路径验证后: {len(validated_ds)} 行")
    
    # 第三步：提示增强
    enhanced_ds = validated_ds.map(enhance_prompt_func, num_proc=1)
    logger.info(f"Qwen3提示增强后: {len(enhanced_ds)} 行")
    
    # 第四步：Qwen3格式包装
    qwen3_ds = enhanced_ds.map(wrap_prompt_for_qwen3, num_proc=1)
    logger.info(f"Qwen3格式包装后: {len(qwen3_ds)} 行")
    
    # 第五步：添加缺失的列
    final_cols = [
        "prompt",                       # Qwen3格式化后的prompt
        "original_enhanced_prompt",     # enhance_prompt_func的直接输出
        "testbench_path",
        "expected_total_tests",
        "reference_verilog_path", 
        "original_prompt_for_debug",    # 最初始的用户输入
        "level",
        "complexity_score",
        "category",
        "difficulty",
        "task_id"
    ]
    
    def add_missing_columns(example):
        """添加缺失的列"""
        result = example.copy()
        
        # 添加默认值
        if 'level' not in result or result['level'] is None:
            result['level'] = 'intermediate'
        if 'complexity_score' not in result or result['complexity_score'] is None:
            result['complexity_score'] = 5.0
        if 'category' not in result or result['category'] is None:
            result['category'] = 'Unknown'
        if 'difficulty' not in result or result['difficulty'] is None:
            result['difficulty'] = 'C'
        if 'task_id' not in result or result['task_id'] is None:
            result['task_id'] = f"task_{hash(str(result.get('prompt', random.random())))}"
        if 'original_prompt_for_debug' not in result:
            result['original_prompt_for_debug'] = result.get('original_enhanced_prompt', result.get('prompt', ''))
        
        return result
    
    final_ds = qwen3_ds.map(add_missing_columns, num_proc=1)
    logger.info(f"Qwen3列补全后: {len(final_ds)} 行")
    
    # 验证Qwen3格式
    if len(final_ds) > 0:
        sample_prompt = final_ds[0]['prompt']
        logger.debug(f"样本prompt预览: {sample_prompt[:200]}...")
        
        # 检查是否包含Qwen3的关键标记
        has_system = "<|im_start|>system" in sample_prompt
        has_user = "<|im_start|>user" in sample_prompt
        has_assistant = "<|im_start|>assistant" in sample_prompt
        
        if has_system and has_user and has_assistant:
            logger.info("✅ Qwen3格式验证通过")
        else:
            logger.warning("⚠️ Qwen3格式可能不完整")
            logger.info(f"格式检查: system={has_system}, user={has_user}, assistant={has_assistant}")
    
    return final_ds

# 修复6: 更新生成配置的辅助函数
def setup_qwen3_generation_config(model, tokenizer, script_cfg):
    """设置Qwen3的生成配置"""
    from transformers import GenerationConfig
    
    logger.info("🤖 设置Qwen3生成配置...")
    
    # 确保tokenizer配置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("设置pad_token为eos_token")
    
    # 更新模型配置
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    
    # 设置生成配置
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=script_cfg.max_seq_length // 2,
        do_sample=True,
        temperature=getattr(script_cfg, 'gen_temperature', 0.7),
        top_p=getattr(script_cfg, 'gen_top_p', 0.8),
        top_k=getattr(script_cfg, 'gen_top_k', 40),
        repetition_penalty=getattr(script_cfg, 'gen_repetition_penalty', 1.05),
        length_penalty=getattr(script_cfg, 'gen_length_penalty', 1.0),
    )
    
    model.generation_config = generation_config
    logger.info("✅ Qwen3生成配置设置完成")
    
    return model, tokenizer

# 在主训练脚本中的集成示例
def integrate_qwen3_fixes():
    """在train.py中集成Qwen3修复的示例代码"""
    return """
# 在train.py的main函数中添加以下修复:

# 1. 替换数据集处理函数
dataset = qwen3_dataset_processing_pipeline(dataset_raw, dataset_dir, script_cfg)

# 2. 应用Qwen3兼容性修复
model, tokenizer = setup_qwen3_generation_config(model, tokenizer, script_cfg)

# 3. 使用Qwen3优化的解析函数
# 在parse_llm_completion_with_context的调用处，替换为:
reasoning, code = parse_llm_completion_with_context_qwen3(
    generated_text, prompt=prompt, step=step, sample_idx=sample_idx
)

# 4. 使用Qwen3推理回调
qwen3_inference_cb = Qwen3InferenceCallback(
    tokenizer=tokenizer,
    eval_dataset=sample_dataset_for_inf_cb,
    num_samples=script_cfg.callback_num_samples,
    eval_every_n_steps=script_cfg.callback_eval_every_n_steps,
    output_dir=script_cfg.output_dir
)
callbacks_list.append(qwen3_inference_cb)

# 5. 在奖励计算中使用Qwen3解析
# 在calculate_enhanced_rewards_for_single_prompt函数中:
_, code = parse_llm_completion_qwen3(full_output, debug_prompt=prompt_str)
"""