#!/usr/bin/env python3
"""快速测试关键组件"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试关键导入"""
    try:
        from grpo_project.callbacks.inference import DetailedInferenceCallback
        print("✅ DetailedInferenceCallback导入成功")
        
        from grpo_project.utils.parsing import parse_llm_completion_with_context
        print("✅ parse_llm_completion_with_context导入成功")
        
        from grpo_project.utils.simulation import run_iverilog_simulation
        print("✅ run_iverilog_simulation导入成功")
        
        from grpo_project.utils.verilog_utils import assess_code_quality
        print("✅ assess_code_quality导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_parsing():
    """测试解析功能"""
    try:
        from grpo_project.utils.parsing import parse_llm_completion_with_context
        
        test_text = "<think>This is reasoning</think>\n```verilog\nmodule test(); endmodule\n```"
        reasoning, code = parse_llm_completion_with_context(test_text)
        
        if reasoning and code:
            print("✅ 解析功能正常")
            return True
        else:
            print(f"❌ 解析结果异常: reasoning={reasoning is not None}, code={code is not None}")
            return False
    except Exception as e:
        print(f"❌ 解析测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 快速测试开始...")
    
    import_ok = test_imports()
    parsing_ok = test_parsing()
    
    if import_ok and parsing_ok:
        print("🎉 快速测试通过！关键组件应该能正常工作。")
        print("\n📋 修复总结:")
        print("1. ✅ 更新了DetailedInferenceCallback的on_step_end方法")
        print("2. ✅ 添加了完整的_generate_single_sample实现")
        print("3. ✅ 添加了parse_llm_completion_with_context函数")
        print("4. ✅ 修复了导入路径")
        print("5. ✅ 确保生成eval_avg_test_pass_rate指标")
        print("\n🔧 现在训练应该能够:")
        print("   - 正确生成和解析模型输出")
        print("   - 运行仿真测试")
        print("   - 计算测试通过率")
        print("   - 记录eval_avg_test_pass_rate到WandB")
        print("   - 触发课程学习阶段推进")
    else:
        print("❌ 快速测试失败，需要进一步调试") 