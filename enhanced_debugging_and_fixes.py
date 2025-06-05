# enhanced_debugging_and_fixes.py - 修复训练波动和课程学习问题

import logging
import wandb
import json
import os
from typing import Dict, Any, Optional, List
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# 1. 修复课程学习状态监控和日志
# EnhancedCurriculumDebugCallback MOVED to grpo_project.curriculum.callbacks


# 2. 修复Qwen3兼容性问题
class Qwen3CompatibilityFixer:
    """修复Qwen3模型兼容性问题"""
    
    @staticmethod
    def fix_generation_config(model, tokenizer):
        """修复Qwen3的生成配置"""
        from transformers import GenerationConfig
        
        logger.info("🔧 修复Qwen3生成配置...")
        
        # 确保tokenizer设置正确
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("设置pad_token为eos_token")
        
        # 修复模型配置
        model_config = getattr(model, 'config', None)
        if model_config:
            model_config.pad_token_id = tokenizer.pad_token_id
            model_config.eos_token_id = tokenizer.eos_token_id
        
        # 创建适合Qwen3的生成配置
        if not hasattr(model, 'generation_config') or model.generation_config is None:
            model.generation_config = GenerationConfig()
        
        # Qwen3特定配置
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.top_k = 40
        model.generation_config.repetition_penalty = 1.05
        
        logger.info("✅ Qwen3生成配置修复完成")
        return model, tokenizer
    
    @staticmethod
    def create_qwen3_prompt(content: str) -> str:
        """创建Qwen3格式的prompt"""
        # Qwen3使用的对话格式
        system_message = """You are a Verilog expert. Please provide your solution in the following format:

<think>
Your detailed thinking process here
</think>

```verilog
Your complete Verilog code here
```"""
        
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        return prompt


# 3. 增强的奖励稳定性监控
# class RewardStabilityMonitor(TrainerCallback): # Already MOVED to grpo_project.callbacks.monitoring in previous step
#     """监控奖励稳定性，减少训练波动"""
# ... (rest of the class was here)

# 4. 修复的奖励计算函数（减少波动）
def create_stabilized_reward_calculator(reward_config, stability_monitor: Optional[RewardStabilityMonitor] = None):
    """创建稳定化的奖励计算器"""
    
    def stabilized_reward_calculator(*args, **kwargs):
        """稳定化的奖励计算"""
        # 这里调用原始的奖励计算函数
        try:
            # 假设原始函数返回 (rewards, metrics)
            rewards, metrics = enhanced_batch_reward_calculator(*args, **kwargs)
            
            # 应用稳定化处理
            if rewards:
                # 记录到稳定性监控器
                if stability_monitor:
                    step = kwargs.get('training_step', 0)
                    for reward in rewards:
                        stability_monitor.add_reward(reward, step)
                
                # 应用奖励削峰和平滑
                stabilized_rewards = []
                for reward in rewards:
                    # 削峰处理（限制极值）
                    clipped_reward = np.clip(reward, -15.0, 15.0)
                    
                    # 轻微平滑（减少噪声）
                    if len(stabilized_rewards) > 0:
                        smoothed_reward = 0.9 * clipped_reward + 0.1 * stabilized_rewards[-1]
                    else:
                        smoothed_reward = clipped_reward
                    
                    stabilized_rewards.append(smoothed_reward)
                
                return stabilized_rewards, metrics
            
        except Exception as e:
            logger.error(f"奖励计算异常: {e}", exc_info=True)
            # 返回安全的默认值
            num_items = len(args[0]) if args and len(args) > 0 else 1
            return [-5.0] * num_items, {}
        
        return rewards, metrics
    
    return stabilized_reward_calculator


# 5. 使用示例和集成指导
def integrate_enhanced_debugging(trainer, curriculum_manager, output_dir, model, tokenizer):
    """集成所有调试增强功能"""
    
    logger.info("🔧 集成增强调试功能...")
    
    # 1. 修复Qwen3兼容性
    model, tokenizer = Qwen3CompatibilityFixer.fix_generation_config(model, tokenizer)
    
    # 2. 创建调试回调
    callbacks_to_add = []
    
    # 课程学习调试回调
    if curriculum_manager:
        curriculum_debug_cb = EnhancedCurriculumDebugCallback(
            curriculum_manager, trainer, output_dir
        )
        callbacks_to_add.append(curriculum_debug_cb)
        logger.info("✅ 添加课程学习调试回调")
    
    # 奖励稳定性监控
    stability_monitor = RewardStabilityMonitor(output_dir)
    callbacks_to_add.append(stability_monitor)
    logger.info("✅ 添加奖励稳定性监控")
    
    # 3. 将回调添加到训练器
    for callback in callbacks_to_add:
        trainer.add_callback(callback)
    
    logger.info(f"🎯 成功集成{len(callbacks_to_add)}个调试功能")
    
    return trainer, stability_monitor


# 6. 主要修复点总结
"""
主要修复的问题：

1. 课程学习问题：
   - 课程进阶逻辑修复
   - 详细的课程状态日志
   - W&B使用数值而非文字记录
   - 数据集更新机制修复

2. Qwen3兼容性：
   - 正确的对话格式
   - 生成配置优化
   - tokenizer设置修复

3. 训练稳定性：
   - 奖励削峰和平滑
   - 稳定性指标监控
   - 异常情况处理

4. 调试信息增强：
   - 专用日志文件
   - 详细的状态追踪
   - 异常捕获和处理

使用方法：
1. 在train.py中导入这些函数
2. 在trainer初始化后调用integrate_enhanced_debugging
3. 在奖励函数中使用create_stabilized_reward_calculator
"""