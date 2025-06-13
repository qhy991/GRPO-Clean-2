#!/usr/bin/env python3
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
