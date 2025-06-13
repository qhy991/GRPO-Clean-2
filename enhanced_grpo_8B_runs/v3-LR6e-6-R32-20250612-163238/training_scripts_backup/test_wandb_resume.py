#!/usr/bin/env python3
"""
测试WandB恢复配置的简单脚本
用于验证main.py中的自动WandB恢复功能是否正常工作
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置基本日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_wandb_resume_configuration():
    """测试WandB恢复配置功能"""
    try:
        logger.info("🧪 开始测试WandB恢复配置...")
        
        # 导入主要的训练管道类
        from main import GRPOTrainingPipeline
        
        # 创建管道实例（仅用于测试配置方法）
        pipeline = GRPOTrainingPipeline()
        
        # 测试场景1: 没有checkpoint的情况
        logger.info("\n📋 测试场景1: 新训练（无checkpoint）")
        original_resume_checkpoint = pipeline.grpo_cfg.resume_from_checkpoint
        pipeline.grpo_cfg.resume_from_checkpoint = None
        
        pipeline._configure_wandb_resume()
        
        # 检查环境变量
        wandb_resume = os.getenv("WANDB_RESUME")
        wandb_run_id = os.getenv("WANDB_RUN_ID")
        
        logger.info(f"  - WANDB_RESUME: {wandb_resume}")
        logger.info(f"  - WANDB_RUN_ID: {wandb_run_id}")
        
        # 测试场景2: 有checkpoint但不存在的情况
        logger.info("\n📋 测试场景2: 指定了不存在的checkpoint")
        pipeline.grpo_cfg.resume_from_checkpoint = "/nonexistent/checkpoint/path"
        
        # 清除之前的环境变量
        if "WANDB_RUN_ID" in os.environ:
            del os.environ["WANDB_RUN_ID"]
        if "WANDB_RESUME" in os.environ:
            del os.environ["WANDB_RESUME"]
        
        pipeline._configure_wandb_resume()
        
        wandb_resume = os.getenv("WANDB_RESUME")
        wandb_run_id = os.getenv("WANDB_RUN_ID")
        
        logger.info(f"  - WANDB_RESUME: {wandb_resume}")
        logger.info(f"  - WANDB_RUN_ID: {wandb_run_id}")
        
        # 测试场景3: 有真实checkpoint的情况（如果存在）
        logger.info("\n📋 测试场景3: 检查真实checkpoint目录")
        
        # 查找可能的checkpoint目录
        possible_checkpoints = []
        output_base = getattr(pipeline.env_cfg, 'output_dir_base', './enhanced_grpo_v3_runs')
        
        if os.path.exists(output_base):
            for item in os.listdir(output_base):
                item_path = os.path.join(output_base, item)
                if os.path.isdir(item_path):
                    # 查找checkpoint子目录
                    for subitem in os.listdir(item_path):
                        if subitem.startswith('checkpoint-'):
                            checkpoint_path = os.path.join(item_path, subitem)
                            if os.path.isdir(checkpoint_path):
                                possible_checkpoints.append(checkpoint_path)
        
        if possible_checkpoints:
            test_checkpoint = possible_checkpoints[0]
            logger.info(f"  - 找到测试checkpoint: {test_checkpoint}")
            
            # 清除之前的环境变量
            if "WANDB_RUN_ID" in os.environ:
                del os.environ["WANDB_RUN_ID"]
            if "WANDB_RESUME" in os.environ:
                del os.environ["WANDB_RESUME"]
            
            pipeline.grpo_cfg.resume_from_checkpoint = test_checkpoint
            pipeline._configure_wandb_resume()
            
            wandb_resume = os.getenv("WANDB_RESUME")
            wandb_run_id = os.getenv("WANDB_RUN_ID")
            
            logger.info(f"  - WANDB_RESUME: {wandb_resume}")
            logger.info(f"  - WANDB_RUN_ID: {wandb_run_id}")
            
            # 测试run ID提取功能
            extracted_run_id, extracted_run_url = pipeline._extract_wandb_run_id(Path(test_checkpoint))
            logger.info(f"  - 提取的run ID: {extracted_run_id}")
            logger.info(f"  - 提取的run URL: {extracted_run_url}")
        else:
            logger.info("  - 未找到可用的checkpoint目录进行测试")
        
        # 恢复原始配置
        pipeline.grpo_cfg.resume_from_checkpoint = original_resume_checkpoint
        
        logger.info("\n✅ WandB恢复配置测试完成!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False

def test_wandb_run_id_extraction():
    """测试WandB run ID提取功能"""
    try:
        logger.info("\n🔍 测试WandB run ID提取功能...")
        
        from main import GRPOTrainingPipeline
        pipeline = GRPOTrainingPipeline()
        
        # 查找真实的checkpoint目录进行测试
        output_base = getattr(pipeline.env_cfg, 'output_dir_base', './enhanced_grpo_v3_runs')
        
        if not os.path.exists(output_base):
            logger.info(f"  - 输出目录不存在: {output_base}")
            return True
        
        checkpoint_found = False
        for item in os.listdir(output_base):
            item_path = os.path.join(output_base, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    if subitem.startswith('checkpoint-'):
                        checkpoint_path = os.path.join(item_path, subitem)
                        if os.path.isdir(checkpoint_path):
                            logger.info(f"  - 测试checkpoint: {checkpoint_path}")
                            
                            run_id, run_url = pipeline._extract_wandb_run_id(Path(checkpoint_path))
                            logger.info(f"    * 提取的run ID: {run_id}")
                            logger.info(f"    * 提取的run URL: {run_url}")
                            
                            # 检查checkpoint目录内容
                            logger.info(f"    * 目录内容:")
                            for file in os.listdir(checkpoint_path):
                                file_path = os.path.join(checkpoint_path, file)
                                if os.path.isfile(file_path):
                                    logger.info(f"      - 文件: {file}")
                                elif os.path.isdir(file_path):
                                    logger.info(f"      - 目录: {file}/")
                            
                            checkpoint_found = True
                            break
                if checkpoint_found:
                    break
        
        if not checkpoint_found:
            logger.info("  - 未找到checkpoint目录进行测试")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ run ID提取测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 开始WandB恢复配置测试...")
    
    success = True
    
    # 运行测试
    success &= test_wandb_resume_configuration()
    success &= test_wandb_run_id_extraction()
    
    if success:
        logger.info("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        logger.error("\n💥 部分测试失败!")
        sys.exit(1) 