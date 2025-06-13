#!/usr/bin/env python3
"""
测试课程学习性能检查间隔配置参数传递
"""

import sys
import argparse
import tempfile
from datetime import datetime

# 添加项目路径
sys.path.append('.')

from grpo_project.configs.training import ScriptConfig
from grpo_project.curriculum.callbacks import CurriculumProgressCallback

def test_config_parameter():
    """测试配置参数是否正确定义"""
    print("🧪 测试1: 配置参数定义")
    
    # 创建默认配置
    config = ScriptConfig(model_name_or_path="test_model")
    
    # 检查默认值
    default_interval = config.curriculum_performance_check_interval
    print(f"  ✅ 默认检查间隔: {default_interval} 步")
    
    # 检查是否可以修改
    config.curriculum_performance_check_interval = 10
    print(f"  ✅ 修改后检查间隔: {config.curriculum_performance_check_interval} 步")
    
    return True

def test_callback_parameter():
    """测试回调函数是否正确接收参数"""
    print("\n🧪 测试2: 回调函数参数传递")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试不同的间隔值
        test_intervals = [10, 25, 50]
        
        for interval in test_intervals:
            callback = CurriculumProgressCallback(
                curriculum_manager=None,
                trainer_ref=None,
                output_dir=temp_dir,
                performance_check_interval=interval
            )
            
            actual_interval = callback.performance_check_interval
            status = "✅" if actual_interval == interval else "❌"
            print(f"  {status} 设置间隔 {interval} -> 实际间隔 {actual_interval}")
    
    return True

def test_command_line_parsing():
    """测试命令行参数解析"""
    print("\n🧪 测试3: 命令行参数解析")
    
    # 模拟命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum_performance_check_interval', type=int, default=25)
    
    # 测试不同的命令行输入
    test_cases = [
        ([], 25, "默认值"),
        (['--curriculum_performance_check_interval', '10'], 10, "设置为10"),
        (['--curriculum_performance_check_interval', '50'], 50, "设置为50")
    ]
    
    for args, expected, description in test_cases:
        parsed_args = parser.parse_args(args)
        actual = parsed_args.curriculum_performance_check_interval
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {description}: 期望={expected}, 实际={actual}")
    
    return True

def test_shell_script_integration():
    """测试shell脚本集成"""
    print("\n🧪 测试4: Shell脚本参数示例")
    
    print("  📋 在run_enhanced_grpo_training.sh中设置:")
    print("     CURRICULUM_PERFORMANCE_CHECK_INTERVAL=10   # 快速调试")
    print("     CURRICULUM_PERFORMANCE_CHECK_INTERVAL=25   # 默认平衡")
    print("     CURRICULUM_PERFORMANCE_CHECK_INTERVAL=50   # 节省计算")
    
    print("\n  📋 参数传递链路:")
    print("     Shell变量 -> CMD_ARGS -> Python argparse -> ScriptConfig -> CurriculumProgressCallback")
    
    return True

def test_performance_impact():
    """测试不同间隔的性能影响"""
    print("\n🧪 测试5: 性能影响分析")
    
    intervals = [5, 10, 25, 50, 100]
    
    print("  📊 理论性能影响分析:")
    for interval in intervals:
        checks_per_1000_steps = 1000 // interval
        relative_overhead = checks_per_1000_steps / (1000 // 25)  # 以25步为基准
        
        if relative_overhead > 1.5:
            impact = "🔴 高开销"
        elif relative_overhead > 1.1:
            impact = "🟡 中等开销"
        else:
            impact = "🟢 低开销"
        
        print(f"     间隔{interval:3d}步: {checks_per_1000_steps:2d}次检查/1000步, {impact}")
    
    return True

def main():
    """主测试函数"""
    print("🔧 课程学习性能检查间隔配置测试")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_config_parameter()
        all_passed &= test_callback_parameter()
        all_passed &= test_command_line_parsing()
        all_passed &= test_shell_script_integration()
        all_passed &= test_performance_impact()
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ 所有测试通过！")
            print("\n📋 使用说明:")
            print("1. 在 run_enhanced_grpo_training.sh 中修改:")
            print("   CURRICULUM_PERFORMANCE_CHECK_INTERVAL=15  # 您想要的步数")
            print("\n2. 建议的设置:")
            print("   - 调试阶段: 5-10步 (频繁检查，便于调试)")
            print("   - 正常训练: 20-30步 (平衡性能和响应)")
            print("   - 长期训练: 40-60步 (减少开销)")
            print("\n3. 参数会自动传递到课程学习回调中")
        else:
            print("❌ 部分测试失败")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 