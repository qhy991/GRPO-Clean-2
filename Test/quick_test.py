#!/usr/bin/env python3
"""快速测试关键组件"""

import sys
import os
sys.path.insert(0, '.')

print("🚀 开始测试VerilogSimulator修复...")

try:
    from grpo_project.evaluation.simulator import VerilogSimulator
    print("✅ 成功导入真正的VerilogSimulator")
    
    # 测试实例化
    simulator = VerilogSimulator()
    print(f"✅ 成功实例化VerilogSimulator: {simulator.__class__}")
    
    # 测试RewardCalculator
    try:
        # 使用占位符配置测试
        class MockConfig:
            compilation_failure = -8.0
            compilation_success = 2.0
            def get_scaled_reward(self, reward, step=0):
                return reward
        
        from grpo_project.rewards.calculator import RewardCalculator
        calc = RewardCalculator(MockConfig())
        
        sim_module = calc.simulator.__class__.__module__
        print(f"✅ RewardCalculator使用的模拟器模块: {sim_module}")
        
        if "evaluation.simulator" in sim_module:
            print("🎉 SUCCESS: RewardCalculator现在使用真正的VerilogSimulator!")
        else:
            print(f"⚠️ WARNING: 可能仍在使用占位符: {sim_module}")
            
    except Exception as e:
        print(f"❌ RewardCalculator测试失败: {e}")

except ImportError as e:
    print(f"❌ 导入VerilogSimulator失败: {e}")
    
print("测试完成") 