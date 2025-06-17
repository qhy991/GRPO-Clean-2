#!/usr/bin/env python3

import os
import torch
import sys
from dataclasses import asdict
from transformers import HfArgumentParser
from trl import GRPOConfig

# Add the project path
sys.path.append('/home/qhy/Research/LLM/GRPO-Clean-2')

from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig

class TestPipeline:
    def __init__(self):
        # 模拟命令行参数
        test_args = [
            '--use_model_parallel', 'true',
            '--model_name_or_path', '/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834',
            '--stage1_adapter_path', '/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/',
            '--dataset_path', '/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl',
            '--output_dir_base', './test_outputs',
            '--max_seq_length', '1024',
            '--script_max_prompt_length', '512',
            '--script_max_completion_length', '512',
            '--per_device_train_batch_size', '1',
            '--gradient_accumulation_steps', '2',
            '--learning_rate', '1e-5',
            '--num_train_epochs', '1',
            '--lora_rank', '32',
            '--lora_alpha', '64'
        ]
        
        # 解析配置
        sys.argv = ['test_config.py'] + test_args
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        self.env_cfg, self.script_cfg, self.reward_cfg, self.grpo_cfg = parser.parse_args_into_dataclasses()
        
        print("🔧 测试训练策略配置")
        print("=" * 50)
        
        # 测试GPU环境检测
        self._detect_multi_gpu_environment()
        print(f"多GPU信息: {self.multi_gpu_info}")
        
        # 测试训练策略配置
        self._configure_training_strategy()
        print(f"训练策略: {self.training_strategy}")
        
        print("=" * 50)

    def _detect_multi_gpu_environment(self):
        """简化版GPU环境检测"""
        try:
            print("🔍 检测多GPU环境...")
            
            if not torch.cuda.is_available():
                print("⚠️ CUDA不可用")
                self.multi_gpu_info = {'gpu_count': 0, 'use_model_parallel': False}
                return
            
            user_specified_model_parallel = getattr(self.script_cfg, 'use_model_parallel', None)
            print(f"🎯 用户指定的模型并行设置: {user_specified_model_parallel}")
            
            gpu_count = torch.cuda.device_count()
            print(f"📊 检测到 {gpu_count} 张GPU")
            
            if user_specified_model_parallel is False:
                print("🎯 用户明确禁用模型并行")
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.multi_gpu_info = {
                    'gpu_count': 1,
                    'total_memory_gb': 79.2,
                    'average_memory_gb': 79.2,
                    'use_model_parallel': False
                }
                return
            
            if gpu_count < 2:
                print(f"⚠️ GPU数量({gpu_count})少于2张，禁用多GPU模式")
                self.multi_gpu_info = {
                    'gpu_count': gpu_count,
                    'total_memory_gb': 79.2 if gpu_count > 0 else 0,
                    'average_memory_gb': 79.2 if gpu_count > 0 else 0,
                    'use_model_parallel': False
                }
                return
            
            # 正常多GPU情况
            use_model_parallel = (user_specified_model_parallel is True or 
                                (user_specified_model_parallel is None and gpu_count >= 2))
            
            self.multi_gpu_info = {
                'gpu_count': gpu_count,
                'total_memory_gb': 79.2 * gpu_count,
                'average_memory_gb': 79.2,
                'use_model_parallel': use_model_parallel
            }
            
            if self.multi_gpu_info['use_model_parallel']:
                print("✅ 多GPU模型并行模式已启用")
            else:
                print("📱 将使用单GPU模式")
                
        except Exception as e:
            print(f"⚠️ 多GPU环境检测失败: {e}")
            self.multi_gpu_info = {'gpu_count': 0, 'use_model_parallel': False}

    def _configure_training_strategy(self):
        """简化版训练策略配置"""
        try:
            print("🔧 配置训练策略...")
            
            # 检测当前环境
            is_distributed = (
                'RANK' in os.environ or 
                'LOCAL_RANK' in os.environ or 
                'WORLD_SIZE' in os.environ or
                getattr(self.grpo_cfg, 'local_rank', -1) >= 0
            )
            
            use_model_parallel = self.multi_gpu_info.get('use_model_parallel', False)
            gpu_count = self.multi_gpu_info.get('gpu_count', 1)
            
            print(f"🔍 环境检测结果:")
            print(f"  - 检测到分布式环境: {is_distributed}")
            print(f"  - 启用模型并行: {use_model_parallel}")
            print(f"  - GPU数量: {gpu_count}")
            
            # 决定训练策略
            if use_model_parallel and gpu_count >= 2:
                if is_distributed:
                    print("⚠️ 检测到分布式环境与模型并行冲突")
                    # 清除分布式环境变量
                    dist_env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
                    for var in dist_env_vars:
                        if var in os.environ:
                            print(f"🧹 清除分布式环境变量: {var}")
                            del os.environ[var]
                    
                    self.training_strategy = "model_parallel_only"
                    print("✅ 配置为纯模型并行模式")
                else:
                    self.training_strategy = "model_parallel_single_process"
                    print("✅ 配置为单进程模型并行模式")
                    
            elif is_distributed and not use_model_parallel:
                self.training_strategy = "distributed_data_parallel"
                print("✅ 配置为分布式数据并行模式")
                
            else:
                self.training_strategy = "single_gpu"
                print("✅ 配置为单GPU训练模式")
                
                if gpu_count > 1:
                    print("🔧 强制单GPU模式，限制使用GPU 0")
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            print(f"🎯 最终训练策略: {self.training_strategy}")
            
        except Exception as e:
            print(f"⚠️ 训练策略配置失败: {e}")
            self.training_strategy = "single_gpu"
            print("🔄 回退到单GPU训练模式")

if __name__ == "__main__":
    test = TestPipeline() 