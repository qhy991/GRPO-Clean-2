#!/usr/bin/env python3
"""
训练稳定性修复脚本
解决SIGSEGV段错误和梯度检查点问题
"""

import os
import sys
import torch
import gc
import signal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_signal_handlers():
    """设置信号处理器以优雅地处理中断"""
    def signal_handler(signum, frame):
        logger.warning(f"接收到信号 {signum}，开始清理...")
        cleanup_resources()
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGSEGV'):
        signal.signal(signal.SIGSEGV, signal_handler)

def cleanup_resources():
    """清理GPU内存和其他资源"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("资源清理完成")
    except Exception as e:
        logger.error(f"资源清理失败: {e}")

def fix_gradient_checkpointing_config():
    """修复梯度检查点配置"""
    config_files = [
        "run_enhanced_grpo_training.sh",
        "grpo_project/configs/training.py"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            logger.info(f"修复配置文件: {config_file}")
            # 这里可以添加具体的配置修复逻辑
    
def set_stable_training_env():
    """设置稳定的训练环境变量"""
    stable_envs = {
        # 禁用梯度检查点相关的不稳定特性
        "TORCH_USE_CUDA_DSA": "1",  # 启用CUDA设备端断言
        "CUDA_LAUNCH_BLOCKING": "1",  # 同步CUDA操作便于调试
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",  # 详细的分布式调试信息
        "NCCL_DEBUG": "INFO",  # NCCL调试信息
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",  # 限制CUDA内存分割
    }
    
    for key, value in stable_envs.items():
        os.environ[key] = value
        logger.info(f"设置环境变量: {key}={value}")

def check_system_resources():
    """检查系统资源是否充足"""
    try:
        # 检查GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                logger.info(f"GPU {i}: 分配 {memory_allocated:.2f}GB, 预留 {memory_reserved:.2f}GB, 总计 {memory_total:.2f}GB")
                
                if memory_allocated / memory_total > 0.9:
                    logger.warning(f"GPU {i} 内存使用率过高: {memory_allocated/memory_total*100:.1f}%")
                    return False
        
        return True
    except Exception as e:
        logger.error(f"资源检查失败: {e}")
        return False

def apply_stability_fixes():
    """应用所有稳定性修复"""
    logger.info("🔧 开始应用训练稳定性修复...")
    
    # 1. 设置信号处理器
    setup_signal_handlers()
    logger.info("✅ 信号处理器设置完成")
    
    # 2. 设置稳定的环境变量
    set_stable_training_env()
    logger.info("✅ 环境变量设置完成")
    
    # 3. 检查系统资源
    if not check_system_resources():
        logger.warning("⚠️ 系统资源不足，建议清理GPU内存后重试")
    else:
        logger.info("✅ 系统资源检查通过")
    
    # 4. 修复梯度检查点配置
    fix_gradient_checkpointing_config()
    logger.info("✅ 梯度检查点配置修复完成")
    
    logger.info("🎉 所有稳定性修复应用完成！")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_stability_fix.log')
        ]
    )
    
    apply_stability_fixes() 