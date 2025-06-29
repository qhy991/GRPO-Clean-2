#!/usr/bin/env python3
"""设备一致性修复调试脚本"""

import os
import sys
import torch
import logging
from transformers import HfArgumentParser

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from trl import GRPOConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_consistency_fix():
    """测试设备一致性修复逻辑"""
    print("🔧 测试设备一致性修复...")
    
    # 1. 解析配置
    parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
    env_cfg, script_cfg, reward_cfg, grpo_cfg = parser.parse_args_into_dataclasses()
    
    print(f"\n📋 配置信息:")
    print(f"  - use_model_parallel: {getattr(script_cfg, 'use_model_parallel', 'MISSING')}")
    print(f"  - 类型: {type(getattr(script_cfg, 'use_model_parallel', None))}")
    
    # 2. 检查GPU环境
    print(f"\n🚀 GPU环境:")
    if not torch.cuda.is_available():
        print("  ❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"  - GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"    GPU {i}: {props.name}, {memory_gb:.1f}GB总容量, {allocated:.2f}GB已用")
    
    # 3. 检查分布式环境变量
    print(f"\n🔍 分布式环境变量:")
    dist_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    has_dist_vars = False
    for var in dist_vars:
        value = os.environ.get(var)
        if value:
            print(f"  - {var}: {value}")
            has_dist_vars = True
    
    if not has_dist_vars:
        print("  ✅ 无分布式环境变量")
    
    # 4. 模拟多GPU环境检测逻辑
    print(f"\n🔧 模拟多GPU环境检测:")
    
    user_specified_model_parallel = getattr(script_cfg, 'use_model_parallel', None)
    print(f"  - 用户设置的模型并行: {user_specified_model_parallel}")
    
    # 模拟检测逻辑
    if user_specified_model_parallel is False:
        detected_strategy = "single_gpu (用户明确禁用)"
        use_model_parallel = False
    elif user_specified_model_parallel is True:
        if gpu_count >= 2:
            detected_strategy = "model_parallel (用户明确启用)"
            use_model_parallel = True
        else:
            print(f"  ❌ 错误：要求模型并行但GPU不足({gpu_count}<2)")
            return False
    elif user_specified_model_parallel is None and gpu_count >= 2:
        detected_strategy = "model_parallel (自动启用)"
        use_model_parallel = True
    else:
        detected_strategy = "single_gpu (默认)"
        use_model_parallel = False
    
    print(f"  - 检测到的策略: {detected_strategy}")
    print(f"  - 使用模型并行: {use_model_parallel}")
    
    # 5. 模拟训练策略配置
    print(f"\n⚙️ 模拟训练策略配置:")
    
    is_distributed = any(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
    print(f"  - 检测到分布式环境: {is_distributed}")
    
    if use_model_parallel and gpu_count >= 2:
        if is_distributed:
            final_strategy = "model_parallel_only (清理分布式变量)"
            print("  🧹 需要清理分布式环境变量")
        else:
            final_strategy = "model_parallel_single_process"
    elif is_distributed and not use_model_parallel:
        final_strategy = "distributed_data_parallel"
    else:
        final_strategy = "single_gpu"
    
    print(f"  - 最终策略: {final_strategy}")
    
    # 6. 测试张量操作
    if use_model_parallel and gpu_count >= 2:
        print(f"\n🧪 测试跨设备张量操作:")
        try:
            # 创建在不同设备上的张量
            tensor_0 = torch.randn(10, 10).cuda(0)
            tensor_1 = torch.randn(10, 10).cuda(1)
            
            print(f"  - tensor_0 设备: {tensor_0.device}")
            print(f"  - tensor_1 设备: {tensor_1.device}")
            
            # 测试可能导致错误的操作
            try:
                # 这种操作在模型并行中可能会导致设备不一致错误
                result = tensor_0 + tensor_1  # 这会失败
                print("  ❌ 意外：跨设备操作成功了")
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print("  ✅ 确认：检测到跨设备张量错误")
                    print(f"    错误信息: {e}")
                else:
                    print(f"  ⚠️ 其他错误: {e}")
            
            # 测试修复方法
            print("  🔧 测试修复方法:")
            tensor_1_fixed = tensor_1.to(tensor_0.device)
            result = tensor_0 + tensor_1_fixed
            print(f"  ✅ 修复后操作成功，结果设备: {result.device}")
            
        except Exception as e:
            print(f"  ❌ 张量测试失败: {e}")
            return False
    
    print(f"\n✅ 设备一致性修复测试完成")
    return True

def main():
    """主函数"""
    try:
        print("🚀 开始设备一致性修复调试...")
        success = test_device_consistency_fix()
        
        if success:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print("\n❌ 测试失败！")
            return 1
            
    except Exception as e:
        print(f"\n💥 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 