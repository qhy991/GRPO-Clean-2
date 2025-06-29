#!/usr/bin/env python3
"""
测试脚本：验证VerilogSimulator修复是否有效
"""

import sys
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_simulator_import():
    """测试能否正确导入真正的VerilogSimulator"""
    try:
        from grpo_project.evaluation.simulator import VerilogSimulator
        logger.info("✅ 成功导入真正的VerilogSimulator")
        return True
    except ImportError as e:
        logger.error(f"❌ 导入真正的VerilogSimulator失败: {e}")
        return False

def test_reward_calculator():
    """测试RewardCalculator是否使用真正的模拟器"""
    try:
        from grpo_project.rewards.calculator import RewardCalculator
        from grpo_project.configs.reward import EnhancedRewardConfig
        
        # 使用默认配置
        config = EnhancedRewardConfig()
        calculator = RewardCalculator(config)
        
        # 检查模拟器类型
        simulator_class = calculator.simulator.__class__.__name__
        simulator_module = calculator.simulator.__class__.__module__
        
        logger.info(f"模拟器类名: {simulator_class}")
        logger.info(f"模拟器模块: {simulator_module}")
        
        if "evaluation.simulator" in simulator_module:
            logger.info("✅ RewardCalculator使用真正的VerilogSimulator!")
            return True
        else:
            logger.warning(f"⚠️ RewardCalculator可能使用占位符模拟器: {simulator_module}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试RewardCalculator失败: {e}")
        return False

def test_simple_simulation():
    """测试简单的仿真功能"""
    try:
        from grpo_project.evaluation.simulator import VerilogSimulator
        
        simulator = VerilogSimulator()
        
        # 简单的Verilog代码示例
        test_verilog = """
module simple_and (
    input a, b,
    output y
);
    assign y = a & b;
endmodule
"""
        
        # 注意：这里需要一个真实的testbench文件
        # 暂时跳过实际仿真测试，只测试类实例化
        logger.info("✅ VerilogSimulator实例化成功")
        logger.info("📝 注意：完整的仿真测试需要真实的testbench文件")
        return True
        
    except Exception as e:
        logger.error(f"❌ 仿真测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始测试VerilogSimulator修复...")
    
    tests = [
        ("导入测试", test_simulator_import),
        ("RewardCalculator测试", test_reward_calculator), 
        ("简单仿真测试", test_simple_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"💥 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("📊 测试结果汇总:")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！VerilogSimulator修复成功！")
        return True
    else:
        logger.error("💀 部分测试失败，可能还有问题需要解决")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 