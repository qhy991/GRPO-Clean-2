#!/usr/bin/env python3
"""
强制更新课程学习配置脚本
解决配置修改后不生效的问题
"""

import os
import json
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def force_update_curriculum_configs():
    """强制更新所有课程学习相关配置"""
    
    print("🔧 强制更新课程学习配置...")
    
    # 1. 直接修改stages.py中的阈值
    stages_file = "grpo_project/curriculum/stages.py"
    if os.path.exists(stages_file):
        print(f"📝 检查并修复 {stages_file}")
        
        with open(stages_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保foundation阶段的阈值是0.65
        if 'performance_threshold=0.7' in content:
            content = content.replace('performance_threshold=0.7', 'performance_threshold=0.65')
            print("  ✅ 修复foundation阶段阈值: 0.7 -> 0.65")
        
        # 确保最小评估次数是5
        if 'min_evaluations=10' in content:
            content = content.replace('min_evaluations=10', 'min_evaluations=5')
            print("  ✅ 修复最小评估次数: 10 -> 5")
        
        with open(stages_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 2. 创建运行时配置覆盖
    runtime_config = {
        "curriculum_override": {
            "force_foundation_threshold": 0.65,
            "force_min_evaluations": 5,
            "force_window_size": 2,
            "debug_mode": True
        }
    }
    
    with open("runtime_curriculum_override.json", 'w', encoding='utf-8') as f:
        json.dump(runtime_config, f, indent=2, ensure_ascii=False)
    
    print("  ✅ 创建运行时配置覆盖文件")
    
    # 3. 创建临时的强制配置检查脚本
    check_script = '''
# 临时课程配置检查代码
# 添加到main.py中的课程管理器初始化之后

if hasattr(curriculum_manager, 'curriculum_stages') and curriculum_manager.curriculum_stages:
    foundation_stage = curriculum_manager.curriculum_stages[0]
    print(f"🔍 当前foundation阶段配置检查:")
    print(f"  - 性能阈值: {foundation_stage.performance_threshold}")
    print(f"  - 最小评估次数: {foundation_stage.min_evaluations}")
    
    # 强制修改foundation阶段配置
    if foundation_stage.performance_threshold > 0.65:
        print(f"⚠️ 强制修改foundation阶段阈值: {foundation_stage.performance_threshold} -> 0.65")
        foundation_stage.performance_threshold = 0.65
    
    if foundation_stage.min_evaluations > 5:
        print(f"⚠️ 强制修改foundation阶段最小评估: {foundation_stage.min_evaluations} -> 5")
        foundation_stage.min_evaluations = 5
'''
    
    with open("temp_config_check.py", 'w', encoding='utf-8') as f:
        f.write(check_script)
    
    print("  ✅ 创建临时配置检查代码")
    
    # 4. 检查当前配置状态
    print("\n📊 当前配置状态检查:")
    
    try:
        # 尝试导入并检查配置
        sys.path.insert(0, '.')
        from grpo_project.curriculum.stages import create_default_curriculum_stages
        
        stages = create_default_curriculum_stages()
        foundation = stages[0]
        
        print(f"  - Foundation阶段名称: {foundation.name}")
        print(f"  - Foundation性能阈值: {foundation.performance_threshold}")
        print(f"  - Foundation最小评估: {foundation.min_evaluations}")
        
        if foundation.performance_threshold <= 0.65 and foundation.min_evaluations <= 5:
            print("  ✅ 配置修改已生效!")
        else:
            print("  ❌ 配置修改尚未生效，需要重启训练")
            
    except Exception as e:
        print(f"  ⚠️ 配置检查失败: {e}")
    
    # 5. 生成重启建议
    print("\n🚀 下一步操作建议:")
    print("1. 🛑 停止当前训练进程")
    print("2. 🔄 重新启动训练脚本")
    print("3. 👀 观察新的日志输出")
    print("4. 📊 检查foundation阶段阈值是否显示为0.65")
    
    # 6. 创建简化的测试脚本
    test_script = '''#!/usr/bin/env python3
"""
快速测试课程配置是否生效
"""

import sys
sys.path.insert(0, '.')

from grpo_project.curriculum.stages import create_default_curriculum_stages

stages = create_default_curriculum_stages()
foundation = stages[0]

print("🧪 课程配置测试结果:")
print(f"Foundation阶段配置:")
print(f"  - 名称: {foundation.name}")
print(f"  - 性能阈值: {foundation.performance_threshold}")
print(f"  - 最小评估: {foundation.min_evaluations}")
print(f"  - 期望阈值: 0.65")
print(f"  - 期望评估: 5")

if foundation.performance_threshold == 0.65 and foundation.min_evaluations == 5:
    print("✅ 配置修改成功!")
else:
    print("❌ 配置修改失败，请检查stages.py文件")
'''
    
    with open("test_curriculum_config.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("\n📝 创建配置测试脚本: test_curriculum_config.py")
    print("   运行 'python test_curriculum_config.py' 验证配置")

def main():
    """主函数"""
    print("🔧 课程学习配置强制更新工具")
    print("解决配置修改后不生效的问题\n")
    
    force_update_curriculum_configs()
    
    print("\n✅ 配置强制更新完成!")
    print("\n💡 重要提醒:")
    print("- 必须重启训练才能让新配置生效")
    print("- 运行 test_curriculum_config.py 验证配置")
    print("- 观察新训练日志中的阈值显示")

if __name__ == "__main__":
    main() 