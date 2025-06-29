#!/usr/bin/env python3
"""
课程学习进阶问题修复脚本
降低进阶要求，使课程能够正常推进
"""

import json
import os
from grpo_project.configs import ScriptConfig

def fix_curriculum_advancement_config():
    """修复课程进阶配置"""
    print("🔧 修复课程学习进阶配置...")
    
    # 1. 修改配置文件
    config_file = "grpo_project/configs/training.py"
    if os.path.exists(config_file):
        print(f"📝 修改配置文件: {config_file}")
        
        # 读取配置
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改阈值
        content = content.replace('performance_threshold=0.7', 'performance_threshold=0.65')
        content = content.replace('min_evaluations=10', 'min_evaluations=5')
        
        # 写回配置
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 配置文件已修改")
    
    # 2. 创建临时配置覆盖
    override_config = {
        "curriculum_learning": {
            "enabled": True,
            "performance_threshold_override": 0.65,
            "min_evaluations_override": 5,
            "sliding_window_size": 2
        }
    }
    
    with open("curriculum_advancement_override.json", 'w', encoding='utf-8') as f:
        json.dump(override_config, f, indent=2)
    
    print("✅ 进阶配置修复完成")
    print("📋 建议:")
    print("  1. 重新启动训练")
    print("  2. 观察课程是否能正常进阶")
    print("  3. 如果还有问题，可以进一步降低阈值")

if __name__ == "__main__":
    fix_curriculum_advancement_config()
