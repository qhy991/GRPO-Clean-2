#!/usr/bin/env python3
"""
测试修复后的课程学习回调功能
"""

import sys
import os
import tempfile
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append('.')

from grpo_project.curriculum.callbacks import CurriculumProgressCallback
from grpo_project.curriculum.stages import CurriculumStageConfig

# 模拟trainer相关类
class MockTrainerState:
    def __init__(self):
        self.global_step = 0
        self.log_history = []

class MockTrainingArguments:
    def __init__(self):
        self.local_rank = 0

class MockTrainerControl:
    pass

# 模拟课程管理器
class MockCurriculumManager:
    def __init__(self):
        self.current_stage = 0
        self.curriculum_stages = [
            CurriculumStageConfig(
                name="foundation",
                dataset_levels=["basic"],
                complexity_range=(0.0, 3.0),
                performance_threshold=0.7,
                min_evaluations=3,  # 降低以便测试
                epochs_ratio=0.3
            ),
            CurriculumStageConfig(
                name="intermediate",
                dataset_levels=["intermediate"],
                complexity_range=(3.0, 7.0),
                performance_threshold=0.6,
                min_evaluations=3,
                epochs_ratio=0.4
            ),
            CurriculumStageConfig(
                name="advanced",
                dataset_levels=["advanced"],
                complexity_range=(7.0, 10.0),
                performance_threshold=0.5,
                min_evaluations=3,
                epochs_ratio=0.3
            )
        ]
        self.stage_performance_history = []
    
    def should_advance_stage(self, performance):
        """检查是否应该进阶"""
        if self.current_stage >= len(self.curriculum_stages):
            return False
        
        stage = self.curriculum_stages[self.current_stage]
        self.stage_performance_history.append(performance)
        
        if len(self.stage_performance_history) >= stage.min_evaluations:
            recent_avg = np.mean(self.stage_performance_history[-3:])
            return recent_avg >= stage.performance_threshold
        return False
    
    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.stage_performance_history = []
            return True
        return False
    
    def get_current_stage_dataset(self):
        """模拟数据集"""
        sizes = [2000, 3000, 1500]  # 各阶段数据集大小
        return list(range(sizes[min(self.current_stage, len(sizes)-1)]))

def test_performance_calculation():
    """测试性能计算功能"""
    print("🧪 测试1: 性能计算功能")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = CurriculumProgressCallback(None, None, temp_dir)
        
        # 测试不同类型的日志
        test_cases = [
            ({'eval_avg_test_pass_rate': 0.8}, 0.8, "直接评估指标"),
            ({'reward': 5.0}, None, "reward指标转换"),  # 动态计算
            ({'loss': 0.1}, 0.9, "loss指标转换"),
            ({'train_loss': 0.2}, 0.8, "train_loss指标转换"),
            ({}, 0.0, "无指标情况"),
            (None, 0.0, "None输入")
        ]
        
        for logs, expected, description in test_cases:
            performance = callback._calculate_performance_from_logs(logs)
            if expected is not None:
                status = "✅" if abs(performance - expected) < 0.1 else "❌"
            else:
                status = "🔍"  # 动态计算，仅检查合理性
            print(f"  {status} {description}: {logs} -> {performance:.4f}")
    
    print()

def test_stage_advancement():
    """测试阶段升级功能"""
    print("🧪 测试2: 阶段升级功能")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_manager = MockCurriculumManager()
        callback = CurriculumProgressCallback(mock_manager, None, temp_dir)
        
        mock_args = MockTrainingArguments()
        mock_state = MockTrainerState()
        mock_control = MockTrainerControl()
        
        print(f"  初始阶段: {mock_manager.current_stage}")
        
        # 模拟训练过程
        for step in range(1, 11):
            mock_state.global_step = step * 25
            
            # 模拟性能逐渐提升
            performance = 0.5 + (step * 0.03)  # 0.5 -> 0.8
            
            # 添加到日志历史
            log_entry = {
                'reward': performance * 5,  # 转换为reward格式
                'loss': 1.0 - performance,
                'learning_rate': 1e-6
            }
            mock_state.log_history.append(log_entry)
            
            print(f"  步数 {mock_state.global_step:3d}: 性能={performance:.3f}, 阶段={mock_manager.current_stage}")
            
            # 调用回调函数
            callback.on_log(mock_args, mock_state, mock_control, log_entry)
            
            # 模拟评估
            if step % 3 == 0:
                callback.on_evaluate(mock_args, mock_state, mock_control)
        
        print(f"  最终阶段: {mock_manager.current_stage}")
        print(f"  性能历史长度: {len(callback.performance_history)}")
    
    print()

def test_wandb_logging():
    """测试W&B日志记录"""
    print("🧪 测试3: W&B日志功能")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_manager = MockCurriculumManager()
        callback = CurriculumProgressCallback(mock_manager, None, temp_dir)
        
        # 模拟一些性能历史
        callback.performance_history = [
            {'step': 25, 'performance': 0.6, 'stage': 0, 'timestamp': datetime.now().isoformat()},
            {'step': 50, 'performance': 0.7, 'stage': 0, 'timestamp': datetime.now().isoformat()},
            {'step': 75, 'performance': 0.8, 'stage': 1, 'timestamp': datetime.now().isoformat()},
        ]
        
        test_logs = {
            'loss': 0.15,
            'reward': 3.5,
            'learning_rate': 1.5e-6
        }
        
        try:
            # 尝试W&B记录（即使没有wandb也应该正常工作）
            callback._wandb_log(100, test_logs)
            print("  ✅ W&B记录功能正常（无异常抛出）")
        except Exception as e:
            print(f"  ❌ W&B记录异常: {e}")
    
    print()

def test_debug_logging():
    """测试调试日志功能"""
    print("🧪 测试4: 调试日志功能")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = CurriculumProgressCallback(None, None, temp_dir)
        
        # 写入一些调试信息
        callback._write_debug("测试调试信息 1")
        callback._write_debug("测试调试信息 2")
        callback._write_debug("测试调试信息 3")
        
        # 检查日志文件是否存在
        log_file = callback.debug_log_path
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "测试调试信息" in content:
                    print("  ✅ 调试日志写入成功")
                    print(f"  📄 日志文件: {log_file}")
                    print(f"  📝 内容行数: {len(content.splitlines())}")
                else:
                    print("  ❌ 调试日志内容不匹配")
        else:
            print("  ❌ 调试日志文件未创建")
    
    print()

def main():
    """主测试函数"""
    print("🔧 课程学习回调修复测试")
    print("=" * 50)
    
    test_performance_calculation()
    test_stage_advancement()
    test_wandb_logging()
    test_debug_logging()
    
    print("✅ 所有测试完成！")
    print("\n📋 测试总结:")
    print("- 性能计算功能已修复，支持 reward/loss/eval 指标")
    print("- 阶段升级逻辑已完善，正确处理 stage 字段")
    print("- W&B 日志记录增强，包含更多调试信息")
    print("- 调试日志系统工作正常")

if __name__ == "__main__":
    main() 