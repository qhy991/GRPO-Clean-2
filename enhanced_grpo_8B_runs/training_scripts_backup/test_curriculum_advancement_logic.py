#!/usr/bin/env python3
"""
测试课程学习进阶逻辑
验证为什么满足条件但无法进阶
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from grpo_project.curriculum.stages import create_default_curriculum_stages
from grpo_project.curriculum.manager import FixedEnhancedCurriculumManager

# 创建模拟数据集
class MockDataset:
    def __init__(self, size=1000):
        self.size = size
        self.data = []
        for i in range(size):
            self.data.append({
                'level': 'basic',
                'complexity_score': 2.0,
                'prompt': f'prompt_{i}',
                'completion': f'completion_{i}'
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def select(self, indices):
        selected_data = [self.data[i] for i in indices]
        new_dataset = MockDataset(0)
        new_dataset.data = selected_data
        new_dataset.size = len(selected_data)
        return new_dataset

def test_advancement_logic():
    """测试进阶逻辑"""
    print("🧪 测试课程学习进阶逻辑")
    print("="*50)
    
    # 1. 创建课程阶段配置
    stages = create_default_curriculum_stages()
    foundation_stage = stages[0]
    
    print(f"📊 Foundation阶段配置:")
    print(f"  - 性能阈值: {foundation_stage.performance_threshold}")
    print(f"  - 最小评估: {foundation_stage.min_evaluations}")
    print(f"  - 滑动窗口: 2 (硬编码)")
    
    # 2. 创建课程管理器
    mock_dataset = MockDataset(1000)
    curriculum_manager = FixedEnhancedCurriculumManager(stages, mock_dataset)
    
    # 3. 模拟您的性能序列
    performance_sequence = [
        0.7635,  # 步数230
        0.7173,  # 步数240  
        0.6937,  # 步数250
        0.6303,  # 步数290 (最新)
    ]
    
    print(f"\n🎯 模拟性能序列: {[f'{p:.4f}' for p in performance_sequence]}")
    
    # 4. 逐个添加性能，测试进阶判断
    for i, performance in enumerate(performance_sequence):
        print(f"\n--- 第{i+1}次性能评估: {performance:.4f} ---")
        
        # 检查是否应该进阶
        should_advance = curriculum_manager.should_advance_stage(performance)
        
        print(f"当前历史长度: {len(curriculum_manager.stage_performance_history)}")
        print(f"历史内容: {[f'{p:.4f}' for p in curriculum_manager.stage_performance_history]}")
        
        if len(curriculum_manager.stage_performance_history) >= 2:
            recent_2 = curriculum_manager.stage_performance_history[-2:]
            recent_avg = np.mean(recent_2)
            print(f"最近2次平均: {recent_avg:.4f}")
            print(f"是否 >= 阈值0.65: {recent_avg >= 0.65}")
        
        print(f"进阶判断结果: {'✅ 应该进阶' if should_advance else '❌ 不应该进阶'}")
        
        if should_advance:
            success = curriculum_manager.advance_stage()
            print(f"进阶执行结果: {'✅ 成功' if success else '❌ 失败'}")
            if success:
                print(f"新阶段: {curriculum_manager.current_stage}")
                break
    
    # 5. 分析为什么无法进阶
    print(f"\n🔍 进阶分析:")
    print(f"当前阶段: {curriculum_manager.current_stage}")
    print(f"评估次数: {len(curriculum_manager.stage_performance_history)}")
    print(f"最小评估要求: {foundation_stage.min_evaluations}")
    
    if len(curriculum_manager.stage_performance_history) >= foundation_stage.min_evaluations:
        print("✅ 评估次数满足要求")
    else:
        print("❌ 评估次数不足")
    
    if len(curriculum_manager.stage_performance_history) >= 2:
        recent_2 = curriculum_manager.stage_performance_history[-2:]
        recent_avg = np.mean(recent_2)
        print(f"最近2次平均: {recent_avg:.4f}")
        if recent_avg >= foundation_stage.performance_threshold:
            print("✅ 性能满足要求")
        else:
            print("❌ 性能不满足要求")
    
    # 6. 测试修复方案
    print(f"\n💡 测试修复方案:")
    
    # 添加几个略好的性能
    test_performances = [0.66, 0.67]
    for perf in test_performances:
        print(f"\n添加性能: {perf:.4f}")
        should_advance = curriculum_manager.should_advance_stage(perf)
        print(f"进阶判断: {'✅ 应该进阶' if should_advance else '❌ 不应该进阶'}")
        
        if should_advance:
            success = curriculum_manager.advance_stage()
            print(f"进阶成功! 新阶段: {curriculum_manager.current_stage}")
            break

if __name__ == "__main__":
    test_advancement_logic() 