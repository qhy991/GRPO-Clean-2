#!/usr/bin/env python3
"""
Qwen3 Model Configuration Fix Script
修复Qwen3模型的配置问题，包括聊天模板和tokenizer设置
"""

import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

def get_qwen3_chat_template():
    """获取Qwen3的正确聊天模板"""
    return """{% for message in messages %}
{%- if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""

def fix_qwen3_tokenizer(model_path: str, cache_dir: Optional[str] = None):
    """修复Qwen3 tokenizer配置"""
    print(f"🔧 正在修复Qwen3 tokenizer配置: {model_path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=False  # Qwen3建议使用slow tokenizer
        )
        
        # 设置正确的聊天模板
        tokenizer.chat_template = get_qwen3_chat_template()
        
        # 确保特殊token正确设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 验证token设置
        print(f"✅ EOS token: {tokenizer.eos_token}")
        print(f"✅ PAD token: {tokenizer.pad_token}")
        print(f"✅ Chat template set: {tokenizer.chat_template is not None}")
        
        # 测试聊天模板
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(
                test_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"✅ 聊天模板测试成功:")
            print(f"   输入: {test_messages}")
            print(f"   输出: {repr(formatted)}")
        except Exception as e:
            print(f"❌ 聊天模板测试失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def create_optimized_training_config():
    """创建优化的训练配置"""
    config = {
        "model_config": {
            "model_name_or_path": "/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834",
            "trust_remote_code": True,
            "use_fast_tokenizer": False,
            "torch_dtype": "bfloat16"
        },
        
        "training_config": {
            # 优化的学习率设置
            "learning_rate": 2e-5,  # 提高学习率
            "warmup_ratio": 0.1,    # 减少warmup
            "lr_scheduler_type": "cosine_with_restarts",
            
            # 优化的批处理设置
            "per_device_train_batch_size": 2,  # 增加批处理大小
            "gradient_accumulation_steps": 16,  # 增加梯度累积
            "dataloader_num_workers": 4,
            
            # 优化的生成参数
            "gen_temperature": 0.8,
            "gen_top_k": 50,
            "gen_top_p": 0.9,
            "gen_repetition_penalty": 1.02,
            "gen_length_penalty": 1.0
        },
        
        "reward_config": {
            # 重新平衡奖励函数
            "compilation_success": 3.0,      # 增加成功奖励
            "compilation_failure": -2.0,     # 减少失败惩罚
            "simulation_crash": -2.0,
            "test_pass_base_reward": 2.0,
            "test_pass_bonus_multiplier": 1.5,
            "max_functional_reward": 20.0,   # 增加最大奖励
            "all_tests_passed_bonus": 8.0,   # 增加通过所有测试的奖励
            
            # 权重调整
            "functional_weight": 0.8,        # 增加功能权重
            "efficiency_weight": 0.1,
            "readability_weight": 0.05,
            "robustness_weight": 0.05
        },
        
        "curriculum_config": {
            # 降低课程学习阈值
            "curriculum_performance_threshold_1": 0.4,  # 从0.75降低到0.4
            "curriculum_performance_threshold_2": 0.5,  # 从0.70降低到0.5
            "curriculum_performance_threshold_3": 0.6,  # 从0.65降低到0.6
            "curriculum_min_evaluations": 3,
            "curriculum_performance_check_interval": 3  # 更频繁检查
        },
        
        "guidance_config": {
            # 更积极的指导策略
            "enable_streaming_guidance": True,
            "min_reasoning_length": 80,      # 增加最小推理长度
            "guidance_trigger_threshold": 30, # 降低触发阈值
            "max_guidance_attempts": 3,      # 增加最大尝试次数
            "guidance_tokens_limit": 40      # 增加指导token限制
        }
    }
    
    return config

def generate_fixed_training_command():
    """生成修复后的训练命令"""
    config = create_optimized_training_config()
    
    command = f"""torchrun \\
    --nproc_per_node 2 \\
    --master_addr localhost \\
    --master_port 14524 \\
    /home/qhy/Research/LLM/GRPO-Clean-2/main.py \\
    --hf_endpoint "https://hf-mirror.com" \\
    --http_proxy "http://10.130.148.206:7890" \\
    --https_proxy "http://10.130.148.206:7890" \\
    --wandb_project "VerilogGRPO_Enhanced_8B_Fixed" \\
    --wandb_entity "qhy0227-tsinghua-university" \\
    --wandb_run_name_prefix "qwen3-fixed-v1" \\
    --model_name_or_path "{config['model_config']['model_name_or_path']}" \\
    --stage1_adapter_path "/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/" \\
    --dataset_path "/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl" \\
    --dataset_base_path /home/qhy/Research/LLM/GRPO-RV/dataset \\
    --output_dir_base "./enhanced_grpo_8B_runs_fixed" \\
    --lora_rank 16 \\
    --lora_alpha 32 \\
    --lora_dropout 0.05 \\
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \\
    --max_seq_length 5120 \\
    --script_max_prompt_length 1024 \\
    --script_max_completion_length 4096 \\
    --length_allocation_strategy custom \\
    --callback_num_samples 2 \\
    --callback_eval_every_n_steps 3 \\
    --learning_rate {config['training_config']['learning_rate']} \\
    --warmup_ratio {config['training_config']['warmup_ratio']} \\
    --lr_scheduler_type {config['training_config']['lr_scheduler_type']} \\
    --per_device_train_batch_size {config['training_config']['per_device_train_batch_size']} \\
    --gradient_accumulation_steps {config['training_config']['gradient_accumulation_steps']} \\
    --gen_temperature {config['training_config']['gen_temperature']} \\
    --gen_top_k {config['training_config']['gen_top_k']} \\
    --gen_top_p {config['training_config']['gen_top_p']} \\
    --gen_repetition_penalty {config['training_config']['gen_repetition_penalty']} \\
    --compilation_success {config['reward_config']['compilation_success']} \\
    --compilation_failure {config['reward_config']['compilation_failure']} \\
    --simulation_crash {config['reward_config']['simulation_crash']} \\
    --test_pass_base_reward {config['reward_config']['test_pass_base_reward']} \\
    --test_pass_bonus_multiplier {config['reward_config']['test_pass_bonus_multiplier']} \\
    --max_functional_reward {config['reward_config']['max_functional_reward']} \\
    --all_tests_passed_bonus {config['reward_config']['all_tests_passed_bonus']} \\
    --functional_weight {config['reward_config']['functional_weight']} \\
    --efficiency_weight {config['reward_config']['efficiency_weight']} \\
    --readability_weight {config['reward_config']['readability_weight']} \\
    --robustness_weight {config['reward_config']['robustness_weight']} \\
    --curriculum_performance_threshold_1 {config['curriculum_config']['curriculum_performance_threshold_1']} \\
    --curriculum_performance_threshold_2 {config['curriculum_config']['curriculum_performance_threshold_2']} \\
    --curriculum_performance_threshold_3 {config['curriculum_config']['curriculum_performance_threshold_3']} \\
    --curriculum_min_evaluations {config['curriculum_config']['curriculum_min_evaluations']} \\
    --curriculum_performance_check_interval {config['curriculum_config']['curriculum_performance_check_interval']} \\
    --enable_streaming_guidance {str(config['guidance_config']['enable_streaming_guidance']).lower()} \\
    --min_reasoning_length {config['guidance_config']['min_reasoning_length']} \\
    --guidance_trigger_threshold {config['guidance_config']['guidance_trigger_threshold']} \\
    --max_guidance_attempts {config['guidance_config']['max_guidance_attempts']} \\
    --guidance_tokens_limit {config['guidance_config']['guidance_tokens_limit']} \\
    --num_train_epochs 4 \\
    --logging_strategy "steps" \\
    --logging_steps 3 \\
    --save_strategy "steps" \\
    --save_steps 15 \\
    --save_total_limit 5 \\
    --report_to "wandb" \\
    --remove_unused_columns False \\
    --num_generations 2 \\
    --max_prompt_length 1024 \\
    --max_completion_length 4096 \\
    --optim "adamw_torch" \\
    --ddp_find_unused_parameters False \\
    --seed 42 \\
    --dataloader_pin_memory \\
    --bf16 \\
    --cache_dir "/home/qhy/Research/LLM/GRPO-Clean-2/.enhanced_cache_v2/models"
"""
    
    return command

if __name__ == "__main__":
    print("🔧 Qwen3配置修复工具")
    print("=" * 50)
    
    # 1. 修复tokenizer配置
    model_path = "/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"
    cache_dir = "/home/qhy/Research/LLM/GRPO-Clean-2/.enhanced_cache_v2/models"
    
    success = fix_qwen3_tokenizer(model_path, cache_dir)
    
    if success:
        print("\n✅ Qwen3 tokenizer配置修复成功！")
    else:
        print("\n❌ Qwen3 tokenizer配置修复失败！")
    
    # 2. 生成优化的训练配置
    print("\n📝 生成优化的训练配置...")
    config = create_optimized_training_config()
    
    with open("optimized_qwen3_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 配置已保存到: optimized_qwen3_config.json")
    
    # 3. 生成修复后的训练命令
    print("\n🚀 生成修复后的训练命令...")
    command = generate_fixed_training_command()
    
    with open("fixed_training_command.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Qwen3 优化训练命令\n")
        f.write("# 修复了学习率、批处理大小、奖励函数和课程学习阈值\n\n")
        f.write(command)
    
    print("✅ 训练命令已保存到: fixed_training_command.sh")
    print("\n🎯 主要修复内容:")
    print("   - 学习率: 6e-6 → 2e-5")
    print("   - 批处理大小: 1 → 2")
    print("   - 梯度累积: 8 → 16")
    print("   - 编译成功奖励: 2.0 → 3.0")
    print("   - 编译失败惩罚: -4.0 → -2.0")
    print("   - 课程学习阈值: 0.75 → 0.4")
    print("   - 推理长度要求: 60 → 80")
    
    print("\n🔧 下一步操作:")
    print("   1. 运行此脚本验证tokenizer配置")
    print("   2. 使用生成的 fixed_training_command.sh 重新训练")
    print("   3. 监控训练日志中的编译成功率和奖励变化") 