#!/usr/bin/env python3

import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

print("🔧 设备一致性测试")
print("=" * 60)

# 检查基础环境
print("1. 基础环境检查:")
print(f"   CUDA可用: {torch.cuda.is_available()}")
print(f"   GPU数量: {torch.cuda.device_count()}")

if torch.cuda.device_count() < 2:
    print("❌ 需要至少2张GPU进行测试")
    exit(1)

# 设置设备映射
device_map = {
    'model.embed_tokens': 'cuda:0',
    'model.layers.0': 'cuda:0',
    'model.layers.1': 'cuda:0',
    'model.layers.2': 'cuda:1',
    'model.layers.3': 'cuda:1',
    'model.norm': 'cuda:1',
    'lm_head': 'cuda:1'
}

print("\n2. 模型加载测试:")
model_path = "/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"

try:
    # 加载tokenizer
    print("   加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（模拟多GPU分布）
    print("   加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 让系统自动分配
        max_memory={0: "35GiB", 1: "35GiB"}
    )
    
    print("   模型设备分布:")
    if hasattr(model, 'hf_device_map'):
        device_counts = {}
        for layer, device in model.hf_device_map.items():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        
        for device, count in device_counts.items():
            print(f"     {device}: {count} 层")
    
    # 应用LoRA
    print("   应用LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print("✅ 模型加载成功")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

print("\n3. 设备一致性测试:")

# 创建测试输入
test_text = "Hello, this is a test."
inputs = tokenizer(test_text, return_tensors="pt", padding=True)

print(f"   输入设备: {inputs['input_ids'].device}")

# 测试推理
try:
    print("   测试前向传播...")
    
    # 确保输入在正确设备上
    input_device = next(model.parameters()).device
    print(f"   模型主设备: {input_device}")
    
    # 将输入移动到模型的第一个参数所在设备
    inputs_on_device = {k: v.to(input_device) for k, v in inputs.items()}
    print(f"   输入移动到: {inputs_on_device['input_ids'].device}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs_on_device)
    
    print(f"   输出设备: {outputs.logits.device}")
    print(f"   输出形状: {outputs.logits.shape}")
    print("✅ 前向传播成功")
    
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n4. 生成测试:")

try:
    print("   测试文本生成...")
    
    # 准备生成输入
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 确保输入在正确设备
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   生成文本: {generated_text}")
    print("✅ 文本生成成功")
    
except Exception as e:
    print(f"❌ 文本生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("设备一致性测试完成")

# 内存清理
del model
del tokenizer
torch.cuda.empty_cache()
print("✅ 内存清理完成") 