#!/usr/bin/env python3
"""
简化的导入测试脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")  # 只显示前3个路径

# 测试1: 基础导入
print("\n=== 测试1: 基础导入 ===")
try:
    import grpo_project
    print("✅ grpo_project 包导入成功")
except Exception as e:
    print(f"❌ grpo_project 包导入失败: {e}")

# 测试2: 子包导入
print("\n=== 测试2: 子包导入 ===")
try:
    from grpo_project import callbacks
    print("✅ callbacks 子包导入成功")
except Exception as e:
    print(f"❌ callbacks 子包导入失败: {e}")

try:
    from grpo_project import utils
    print("✅ utils 子包导入成功")
except Exception as e:
    print(f"❌ utils 子包导入失败: {e}")

try:
    from grpo_project import evaluation
    print("✅ evaluation 子包导入成功")
except Exception as e:
    print(f"❌ evaluation 子包导入失败: {e}")

# 测试3: 具体类导入
print("\n=== 测试3: 具体类导入 ===")
try:
    from grpo_project.callbacks.base import BaseCallback
    print("✅ BaseCallback 导入成功")
except Exception as e:
    print(f"❌ BaseCallback 导入失败: {e}")

try:
    from grpo_project.utils.replay_buffer import ExperienceBuffer
    print("✅ ExperienceBuffer 导入成功")
except Exception as e:
    print(f"❌ ExperienceBuffer 导入失败: {e}")

try:
    from grpo_project.evaluation.simulator import VerilogSimulator
    print("✅ VerilogSimulator 导入成功")
except Exception as e:
    print(f"❌ VerilogSimulator 导入失败: {e}")

# 测试4: 函数导入
print("\n=== 测试4: 函数导入 ===")
try:
    from grpo_project.utils.parsing import parse_llm_completion_with_context
    print("✅ parse_llm_completion_with_context 导入成功")
except Exception as e:
    print(f"❌ parse_llm_completion_with_context 导入失败: {e}")

try:
    from grpo_project.utils.simulation import run_iverilog_simulation
    print("✅ run_iverilog_simulation 导入成功")
except Exception as e:
    print(f"❌ run_iverilog_simulation 导入失败: {e}")

try:
    from grpo_project.utils.verilog_utils import assess_code_quality
    print("✅ assess_code_quality 导入成功")
except Exception as e:
    print(f"❌ assess_code_quality 导入失败: {e}")

print("\n=== 导入测试完成 ===") 