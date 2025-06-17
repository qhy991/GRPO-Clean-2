#!/usr/bin/env python3

import os
import torch

print("🔍 多GPU环境测试")
print("="*50)

# 1. 检查CUDA可用性
print(f"CUDA可用性: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("❌ CUDA不可用，退出测试")
    exit(1)

# 2. 检查GPU数量
gpu_count = torch.cuda.device_count()
print(f"GPU数量: {gpu_count}")

# 3. 检查环境变量
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

# 4. 检查每个GPU状态
print("\nGPU详细信息:")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / 1024**3
    print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")

# 5. 测试GPU通信
if gpu_count >= 2:
    print("\n🔗 测试GPU通信...")
    try:
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')
        
        # 创建测试数据
        x = torch.randn(100, 100, device=device0)
        print(f"  在GPU 0创建数据: {x.shape}")
        
        # 移动到GPU 1
        y = x.to(device1)
        print(f"  移动到GPU 1: {y.device}")
        
        # 移动回GPU 0
        z = y.to(device0)
        print(f"  移动回GPU 0: {z.device}")
        
        # 验证数据一致性
        is_consistent = torch.allclose(x, z)
        print(f"  数据一致性: {is_consistent}")
        
        if is_consistent:
            print("✅ GPU通信测试成功")
        else:
            print("❌ GPU通信测试失败 - 数据不一致")
            
    except Exception as e:
        print(f"❌ GPU通信测试失败: {e}")
else:
    print("⚠️ GPU数量不足，跳过通信测试")

# 6. 模拟配置检查
print("\n🎯 模拟训练配置:")
use_model_parallel = gpu_count >= 2
print(f"  推荐模型并行: {use_model_parallel}")
print(f"  推荐训练策略: {'model_parallel_single_process' if use_model_parallel else 'single_gpu'}")

print("\n" + "="*50)
print("多GPU环境测试完成") 