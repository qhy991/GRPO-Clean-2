#!/usr/bin/env python3
"""配置参数传递测试"""

import sys
from transformers import HfArgumentParser
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from trl import GRPOConfig

def test_config_parsing():
    print("🔧 测试配置参数传递...")
    
    # 解析命令行参数
    parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
    env_cfg, script_cfg, reward_cfg, grpo_cfg = parser.parse_args_into_dataclasses()
    
    print("\n📁 数据集配置检查:")
    print(f"  - script_cfg.dataset_path: {getattr(script_cfg, 'dataset_path', 'MISSING')}")
    print(f"  - env_cfg.dataset_base_path: {getattr(env_cfg, 'dataset_base_path', 'MISSING')}")
    
    if hasattr(script_cfg, 'dataset_path') and script_cfg.dataset_path:
        import os
        inferred_base = os.path.dirname(script_cfg.dataset_path)
        print(f"  - 从dataset_path推导的基础路径: {inferred_base}")
        
        # 检查路径是否存在
        if os.path.exists(script_cfg.dataset_path):
            print(f"  ✅ 数据集文件存在: {script_cfg.dataset_path}")
        else:
            print(f"  ❌ 数据集文件不存在: {script_cfg.dataset_path}")
            
        if os.path.exists(inferred_base):
            print(f"  ✅ 推导的基础路径存在: {inferred_base}")
        else:
            print(f"  ❌ 推导的基础路径不存在: {inferred_base}")
    
    base_path = env_cfg.dataset_base_path
    if base_path:
        import os
        if os.path.exists(base_path):
            print(f"  ✅ 明确指定的基础路径存在: {base_path}")
        else:
            print(f"  ❌ 明确指定的基础路径不存在: {base_path}")
    else:
        print("  ⚠️ 没有明确指定基础路径")
    
    print("\n🎯 关键配置:")
    print(f"  - model_name_or_path: {getattr(script_cfg, 'model_name_or_path', 'MISSING')}")
    print(f"  - output_dir_base: {getattr(env_cfg, 'output_dir_base', 'MISSING')}")
    print(f"  - learning_rate: {getattr(grpo_cfg, 'learning_rate', 'MISSING')}")
    
    # 🔧 新增：多GPU配置调试
    print("\n🚀 多GPU配置检查:")
    print(f"  - use_model_parallel: {getattr(script_cfg, 'use_model_parallel', 'MISSING')}")
    print(f"  - 类型: {type(getattr(script_cfg, 'use_model_parallel', None))}")
    
    # 检查GPU环境
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  - 可用GPU数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"    GPU {i}: {props.name}, {memory_gb:.1f}GB")
    else:
        print("  - CUDA不可用")
    
    # 检查环境变量
    print("\n🔍 环境变量检查:")
    import os
    dist_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for var in dist_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"  - {var}: {value}")
    
    print(f"  - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}")
    
    return env_cfg, script_cfg

if __name__ == "__main__":
    try:
        env_cfg, script_cfg = test_config_parsing()
        print("\n✅ 配置解析成功")
    except Exception as e:
        print(f"\n❌ 配置解析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 