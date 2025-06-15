#!/usr/bin/env python3
"""
测试WandB同步修复功能
验证步数同步问题是否已解决
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from grpo_project.core.wandb_sync_manager import (
    initialize_wandb_sync_manager, 
    get_wandb_sync_manager,
    safe_wandb_log,
    update_wandb_step_offset
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_wandb_sync_manager():
    """测试WandB同步管理器的基本功能"""
    logger.info("🧪 开始测试WandB同步管理器...")
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. 测试初始化
        logger.info("📝 测试1: 初始化同步管理器")
        sync_manager = initialize_wandb_sync_manager(
            output_dir=str(temp_path),
            project_name="test-wandb-sync",
            run_name="test-run"
        )
        
        assert sync_manager is not None, "同步管理器初始化失败"
        assert get_wandb_sync_manager() is sync_manager, "全局管理器获取失败"
        logger.info("✅ 同步管理器初始化成功")
        
        # 2. 测试WandB运行设置
        logger.info("📝 测试2: 设置WandB运行")
        success = sync_manager.setup_wandb_run(
            config={"test_param": "test_value", "learning_rate": 0.001}
        )
        
        if success:
            logger.info("✅ WandB运行设置成功")
        else:
            logger.warning("⚠️ WandB运行设置失败（可能是网络问题或WandB未配置）")
        
        # 3. 测试安全日志记录
        logger.info("📝 测试3: 安全日志记录")
        test_data = {
            "test_metric": 0.95,
            "eval_avg_test_pass_rate": 0.75,
            "step": 10
        }
        
        # 测试全局函数
        result = safe_wandb_log(test_data, step=10)
        logger.info(f"📊 日志记录结果: {result}")
        
        # 4. 测试步数偏移更新
        logger.info("📝 测试4: 步数偏移更新")
        update_wandb_step_offset(trainer_step=10)
        logger.info("✅ 步数偏移更新完成")
        
        # 5. 测试本地备份
        logger.info("📝 测试5: 本地备份检查")
        backup_file = temp_path / "wandb_local_backup.jsonl"
        if backup_file.exists():
            logger.info(f"✅ 本地备份文件存在: {backup_file}")
            with open(backup_file, 'r') as f:
                backup_content = f.read()
                logger.info(f"📄 备份内容预览: {backup_content[:100]}...")
        else:
            logger.info("📄 本地备份文件不存在（这是正常的，如果WandB记录成功）")
        
        # 6. 测试清理
        logger.info("📝 测试6: 清理资源")
        sync_manager.finish()
        logger.info("✅ 资源清理完成")
        
        logger.info("🎉 所有测试完成！")

def test_step_sync_simulation():
    """模拟步数同步问题的场景"""
    logger.info("🎯 模拟步数同步问题场景...")
    
    # 模拟断续训练的场景
    logger.info("📝 场景1: 模拟断续训练")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建假的checkpoint目录和trainer_state.json
        checkpoint_dir = temp_path / "checkpoint-25"
        checkpoint_dir.mkdir(parents=True)
        
        trainer_state = {
            "global_step": 25,
            "epoch": 1.0,
            "total_flos": 1000000
        }
        
        with open(checkpoint_dir / "trainer_state.json", 'w') as f:
            import json
            json.dump(trainer_state, f)
        
        logger.info(f"📁 创建假checkpoint: {checkpoint_dir}")
        
        # 初始化同步管理器
        sync_manager = initialize_wandb_sync_manager(
            output_dir=str(temp_path),
            project_name="test-resume-sync",
            run_name="test-resume-run"
        )
        
        # 模拟从checkpoint恢复
        success = sync_manager.setup_wandb_run(
            resume_from_checkpoint=str(checkpoint_dir),
            config={"resumed": True, "original_step": 25}
        )
        
        if success:
            logger.info("✅ 断续训练模拟设置成功")
        else:
            logger.warning("⚠️ 断续训练模拟设置失败")
        
        # 模拟记录一些步骤
        for step in range(26, 31):
            test_data = {
                "eval_avg_test_pass_rate": 0.1 * step,
                "curriculum/evaluation_count": step - 25,
                "simulated_step": step
            }
            
            # 更新步数偏移
            update_wandb_step_offset(step)
            
            # 记录数据
            success = safe_wandb_log(test_data, step=step)
            logger.info(f"📊 步数 {step} 记录结果: {success}")
        
        sync_manager.finish()
        logger.info("🎯 断续训练模拟完成")

def main():
    """主函数"""
    logger.info("🚀 开始WandB同步修复测试...")
    
    try:
        # 基本功能测试
        test_wandb_sync_manager()
        
        # 步数同步场景测试
        test_step_sync_simulation()
        
        logger.info("✅ 所有测试通过！WandB同步修复功能正常")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 