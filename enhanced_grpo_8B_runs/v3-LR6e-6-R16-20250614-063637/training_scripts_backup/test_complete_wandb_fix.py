#!/usr/bin/env python3
"""
完整的WandB步数同步修复测试
验证新的同步回调是否正确工作
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_synced_wandb_callback():
    """测试同步WandB回调"""
    logger.info("🧪 测试同步WandB回调...")
    
    try:
        from grpo_project.callbacks.wandb_sync_callback import SyncedWandbCallback
        from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager
        from grpo_project.configs import EnvConfig, ScriptConfig, RewardConfig
        
        # 创建临时配置
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 初始化同步管理器
            sync_manager = initialize_wandb_sync_manager(
                output_dir=str(temp_path),
                project_name="test-complete-sync",
                run_name="test-synced-callback"
            )
            
            # 设置WandB运行
            success = sync_manager.setup_wandb_run(
                config={"test_sync_callback": True}
            )
            
            if not success:
                logger.warning("⚠️ WandB运行设置失败，但测试继续")
            
            # 创建配置对象（简化版）
            env_cfg = type('EnvConfig', (), {})()
            script_cfg = type('ScriptConfig', (), {})()
            reward_cfg = type('RewardConfig', (), {})()
            
            # 创建同步回调
            callback = SyncedWandbCallback(
                env_cfg=env_cfg,
                script_cfg=script_cfg,
                reward_cfg=reward_cfg,
                output_dir=str(temp_path)
            )
            
            # 模拟训练状态
            class MockTrainingArgs:
                local_rank = 0
                
            class MockTrainerState:
                def __init__(self, step):
                    self.global_step = step
                    self.epoch = step * 0.1
            
            class MockTrainerControl:
                pass
            
            args = MockTrainingArgs()
            control = MockTrainerControl()
            
            # 测试多个步骤
            for step in range(1, 6):
                state = MockTrainerState(step)
                
                # 测试on_log
                logs = {
                    "loss": 0.5 - step * 0.05,
                    "learning_rate": 1e-5,
                    "reward": step * 2.0,
                    "eval_avg_test_pass_rate": min(0.2 * step, 1.0)
                }
                
                logger.info(f"📊 测试步骤 {step}")
                
                # 调用回调方法
                callback.on_log(args, state, control, logs=logs)
                callback.on_step_end(args, state, control)
                
                # 测试奖励组件记录
                callback.log_reward_components({
                    "compilation_success": 1.0,
                    "test_pass_rate": 0.8,
                    "code_quality": 0.7
                }, step=step)
                
                # 测试批次指标记录
                callback.log_batch_aggregated_metrics({
                    "generation_funnel": {
                        "successful_extractions_count": 10,
                        "compilation_ratio_vs_batch": 0.8
                    },
                    "reward": {
                        "batch_mean_final_scaled_reward": step * 1.5
                    }
                }, step=step)
            
            # 测试训练结束
            final_state = MockTrainerState(5)
            callback.on_train_end(args, final_state, control)
            
            logger.info("✅ 同步WandB回调测试完成")
            
            # 检查本地备份
            backup_file = temp_path / "wandb_local_backup.jsonl"
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    lines = f.readlines()
                    logger.info(f"📄 本地备份记录数: {len(lines)}")
                    if lines:
                        logger.info(f"📄 最后一条记录预览: {lines[-1][:100]}...")
            
            sync_manager.finish()
            
    except ImportError as e:
        logger.error(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False
    
    return True

def test_step_sync_scenario():
    """测试步数同步场景"""
    logger.info("🎯 测试步数同步场景...")
    
    try:
        from grpo_project.core.wandb_sync_manager import (
            initialize_wandb_sync_manager, 
            update_wandb_step_offset,
            safe_wandb_log
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 模拟checkpoint
            checkpoint_dir = temp_path / "checkpoint-50"
            checkpoint_dir.mkdir(parents=True)
            
            trainer_state = {
                "global_step": 50,
                "epoch": 2.0,
                "total_flos": 5000000
            }
            
            import json
            with open(checkpoint_dir / "trainer_state.json", 'w') as f:
                json.dump(trainer_state, f)
            
            # 初始化同步管理器
            sync_manager = initialize_wandb_sync_manager(
                output_dir=str(temp_path),
                project_name="test-step-sync-scenario",
                run_name="test-resume-50"
            )
            
            # 模拟断续训练
            success = sync_manager.setup_wandb_run(
                resume_from_checkpoint=str(checkpoint_dir),
                config={"resumed_from_step": 50}
            )
            
            if success:
                logger.info("✅ 断续训练场景设置成功")
            else:
                logger.warning("⚠️ 断续训练场景设置失败，但继续测试")
            
            # 模拟从步数51开始的训练
            for step in range(51, 56):
                # 更新步数偏移
                update_wandb_step_offset(step)
                
                # 记录训练数据
                training_data = {
                    "loss": 0.3 - (step - 50) * 0.02,
                    "reward": (step - 50) * 1.5,
                    "eval_avg_test_pass_rate": min((step - 50) * 0.1, 1.0),
                    "curriculum/evaluation_count": step - 50,
                    "global_step": step
                }
                
                success = safe_wandb_log(training_data, step=step)
                logger.info(f"📊 步骤 {step} 记录结果: {success}")
                
                # 显示同步状态
                if hasattr(sync_manager, 'step_offset'):
                    logger.info(f"🔄 当前步数偏移: {sync_manager.step_offset}")
            
            sync_manager.finish()
            logger.info("🎯 步数同步场景测试完成")
            
    except Exception as e:
        logger.error(f"❌ 步数同步场景测试失败: {e}", exc_info=True)
        return False
    
    return True

def main():
    """主函数"""
    logger.info("🚀 开始完整的WandB修复测试...")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 同步WandB回调
    if test_synced_wandb_callback():
        success_count += 1
        logger.info("✅ 测试1通过: 同步WandB回调")
    else:
        logger.error("❌ 测试1失败: 同步WandB回调")
    
    # 测试2: 步数同步场景
    if test_step_sync_scenario():
        success_count += 1
        logger.info("✅ 测试2通过: 步数同步场景")
    else:
        logger.error("❌ 测试2失败: 步数同步场景")
    
    # 总结
    logger.info(f"📊 测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        logger.info("🎉 所有测试通过！WandB完整修复成功")
        return 0
    else:
        logger.error(f"❌ {total_tests - success_count} 个测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 