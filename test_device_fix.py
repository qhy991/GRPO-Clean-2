#!/usr/bin/env python3
"""简单测试设备修复是否有效"""

import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_device_fix():
    """测试设备修复"""
    print("🔧 测试设备修复...")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 检测到 {gpu_count} 张GPU")
    
    if gpu_count < 2:
        print("⚠️ GPU数量不足，无法测试模型并行")
        return False
    
    # 模拟GRPO中的错误情况
    print("🧪 模拟GRPO错误情况...")
    
    try:
        # 创建不同设备上的张量
        is_eos = torch.tensor([[True, False, True], [False, True, False]]).cuda(1)
        eos_idx = torch.zeros(2, dtype=torch.long).cuda(0)
        
        print(f"  - is_eos设备: {is_eos.device}")
        print(f"  - eos_idx设备: {eos_idx.device}")
        
        # 这会导致错误：Expected all tensors to be on the same device
        try:
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            print("❌ 意外：错误操作成功了")
            return False
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print("✅ 确认：重现了GRPO错误")
                print(f"  错误信息: {e}")
                
                # 测试修复方法
                print("🔧 测试修复方法...")
                
                # 方法1：统一设备
                is_eos_fixed = is_eos.to(eos_idx.device)
                eos_idx[is_eos_fixed.any(dim=1)] = is_eos_fixed.int().argmax(dim=1)[is_eos_fixed.any(dim=1)]
                print("✅ 修复方法1成功：统一设备")
                
                # 方法2：强制到cuda:0
                is_eos_cuda0 = is_eos.to('cuda:0')
                eos_idx_cuda0 = eos_idx.to('cuda:0')
                eos_idx_cuda0[is_eos_cuda0.any(dim=1)] = is_eos_cuda0.int().argmax(dim=1)[is_eos_cuda0.any(dim=1)]
                print("✅ 修复方法2成功：强制cuda:0")
                
                return True
            else:
                print(f"❌ 其他错误: {e}")
                return False
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_import_pipeline():
    """测试导入管道"""
    print("🔧 测试导入训练管道...")
    
    try:
        from main import GRPOTrainingPipeline
        print("✅ 成功导入GRPOTrainingPipeline")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始设备修复测试...")
    
    success = True
    
    # 测试设备修复
    if not test_device_fix():
        success = False
    
    # 测试导入
    if not test_import_pipeline():
        success = False
    
    if success:
        print("\n🎉 所有测试通过！设备修复应该有效")
        print("💡 建议：重新运行训练脚本测试修复效果")
    else:
        print("\n❌ 测试失败！需要进一步调试")
    
    sys.exit(0 if success else 1) 