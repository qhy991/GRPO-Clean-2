#!/usr/bin/env python3
"""
测试分层抽样配置参数解析脚本
用于验证新添加的参数是否能被HfArgumentParser正确识别
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from transformers import HfArgumentParser, TrainingArguments
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig

# 使用TrainingArguments作为GRPOConfig的替代进行测试
class TestGRPOConfig(TrainingArguments):
    pass

def test_stratified_config():
    """测试分层抽样配置参数"""
    
    # 模拟命令行参数
    test_args = [
        "--model_name_or_path", "/test/model",
        "--dataset_path", "/test/dataset.jsonl",
        "--dataset_sample_ratio", "0.1",
        "--stratify_columns", "level,category",
        "--min_samples_per_category", "2",
        "--sampling_random_seed", "123"
    ]
    
    try:
        # 初始化解析器
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, TestGRPOConfig))
        
        # 解析参数
        env_cfg, script_cfg, reward_cfg, grpo_cfg = parser.parse_args(test_args)
        
        print("✅ 参数解析成功!")
        print("\n🎯 分层抽样配置:")
        print(f"  - dataset_sample_ratio: {script_cfg.dataset_sample_ratio}")
        print(f"  - stratify_columns: {script_cfg.stratify_columns}")
        print(f"  - min_samples_per_category: {script_cfg.min_samples_per_category}")
        print(f"  - sampling_random_seed: {script_cfg.sampling_random_seed}")
        
        # 测试分层抽样逻辑
        if script_cfg.dataset_sample_ratio:
            print(f"\n📊 分层抽样将使用 {script_cfg.dataset_sample_ratio*100:.0f}% 的数据")
            print(f"📂 分层字段: {script_cfg.stratify_columns.split(',')}")
            print(f"🔢 每类最少样本: {script_cfg.min_samples_per_category}")
            print(f"🎲 随机种子: {script_cfg.sampling_random_seed}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 测试分层抽样配置参数...")
    success = test_stratified_config()
    
    if success:
        print("\n✅ 所有测试通过! 分层抽样参数配置正确。")
        print("🚀 现在可以运行训练脚本了。")
    else:
        print("\n❌ 测试失败! 请检查配置参数定义。")
        exit(1) 