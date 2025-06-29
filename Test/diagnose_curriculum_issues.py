#!/usr/bin/env python3
"""
课程学习问题诊断脚本
分析loss无穷大、step不匹配、梯度警告等问题
"""

import json
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

def diagnose_curriculum_state_files():
    """诊断课程学习状态文件"""
    print("🔍 诊断课程学习状态文件...")
    
    # 查找所有的curriculum_state_detailed.json文件
    pattern = "**/curriculum_state_detailed.json"
    state_files = list(Path(".").glob(pattern))
    
    if not state_files:
        print("❌ 未找到任何curriculum_state_detailed.json文件")
        return
    
    for state_file in state_files:
        print(f"\n📁 分析文件: {state_file}")
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_stage = data.get('current_stage', 'N/A')
            history = data.get('performance_history', [])
            
            print(f"  - 当前阶段: {current_stage}")
            print(f"  - 性能历史记录数: {len(history)}")
            
            if history:
                # 分析loss值
                loss_values = [entry.get('loss', 'N/A') for entry in history]
                inf_count = sum(1 for loss in loss_values if loss == float('inf') or str(loss).lower() == 'infinity')
                valid_losses = [loss for loss in loss_values if isinstance(loss, (int, float)) and loss != float('inf')]
                
                print(f"  - 无穷大loss记录数: {inf_count}/{len(history)}")
                if valid_losses:
                    print(f"  - 有效loss范围: {min(valid_losses):.4f} ~ {max(valid_losses):.4f}")
                
                # 分析step序列
                steps = [entry.get('step', 0) for entry in history]
                if len(steps) > 1:
                    step_intervals = [steps[i] - steps[i-1] for i in range(1, len(steps))]
                    unique_intervals = set(step_intervals)
                    print(f"  - Step间隔: {unique_intervals}")
                    print(f"  - Step范围: {min(steps)} ~ {max(steps)}")
                
                # 分析性能值
                performances = [entry.get('performance', 0) for entry in history]
                valid_perfs = [p for p in performances if isinstance(p, (int, float)) and not np.isnan(p)]
                if valid_perfs:
                    print(f"  - 性能值范围: {min(valid_perfs):.4f} ~ {max(valid_perfs):.4f}")
                    print(f"  - 平均性能: {np.mean(valid_perfs):.4f}")
                
        except Exception as e:
            print(f"  ❌ 文件读取失败: {e}")

def analyze_training_logs():
    """分析训练日志中的loss模式"""
    print("\n🔍 分析训练日志...")
    
    # 查找所有可能的日志文件
    log_patterns = [
        "**/training_log.txt",
        "**/enhanced_training_log.txt", 
        "**/curriculum_progress_debug.txt"
    ]
    
    log_files = []
    for pattern in log_patterns:
        log_files.extend(list(Path(".").glob(pattern)))
    
    if not log_files:
        print("❌ 未找到训练日志文件")
        return
    
    for log_file in log_files[-3:]:  # 只分析最近的3个日志文件
        print(f"\n📋 分析日志: {log_file}")
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找loss相关的行
            loss_lines = []
            for line_no, line in enumerate(content.split('\n'), 1):
                if any(keyword in line.lower() for keyword in ['loss', 'train_loss', 'eval_loss']):
                    loss_lines.append((line_no, line.strip()))
            
            print(f"  - 包含loss的行数: {len(loss_lines)}")
            
            # 显示最近的几行loss记录
            if loss_lines:
                print("  - 最近的loss记录:")
                for line_no, line in loss_lines[-5:]:
                    print(f"    行{line_no}: {line[:100]}...")
            
        except Exception as e:
            print(f"  ❌ 日志分析失败: {e}")

def check_wandb_files():
    """检查WandB文件中的step冲突"""
    print("\n🔍 检查WandB文件...")
    
    wandb_dirs = list(Path(".").glob("**/wandb"))
    
    if not wandb_dirs:
        print("❌ 未找到wandb目录")
        return
    
    for wandb_dir in wandb_dirs[:2]:  # 只检查最近的2个
        print(f"\n📊 WandB目录: {wandb_dir}")
        
        # 查找summary文件
        summary_files = list(wandb_dir.glob("**/wandb-summary.json"))
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # 检查step相关信息
                step_info = {}
                for key, value in summary.items():
                    if 'step' in key.lower() or '_step' in key:
                        step_info[key] = value
                
                if step_info:
                    print(f"  - Step相关字段: {step_info}")
                
            except Exception as e:
                print(f"  ❌ Summary文件读取失败: {e}")

def provide_fix_recommendations():
    """提供修复建议"""
    print("\n🔧 修复建议:")
    print("="*60)
    
    print("1. **Loss无穷大问题修复**:")
    print("   - 问题原因: OptimizedCurriculumCallback使用float('inf')作为默认值")
    print("   - 修复方案: 已在代码中修复，使用正确的loss获取逻辑")
    print("   - 验证方法: 检查新的curriculum_state_detailed.json文件")
    
    print("\n2. **Step不匹配问题修复**:")
    print("   - 问题原因: 多个回调使用不同的step计数逻辑")
    print("   - 修复方案: 统一使用trainer.state.global_step")
    print("   - 建议配置: CURRICULUM_PERFORMANCE_CHECK_INTERVAL=5")
    
    print("\n3. **梯度警告问题**:")
    print("   - 问题原因: 梯度检查点机制和模型输入的配置问题")
    print("   - 临时方案: 这个警告不会影响训练效果，可以忽略")
    print("   - 长期方案: 需要检查模型输入的requires_grad设置")
    
    print("\n4. **WandB Step冲突修复**:")
    print("   - 问题原因: 多个组件同时记录到WandB导致step冲突")
    print("   - 修复方案: 使用WandB step fix模块（已创建）")
    print("   - 启用方法: 设置WANDB_STEP_FIX_ENABLED=true")
    
    print("\n5. **实时监控建议**:")
    print("   - 查看课程进度: tail -f */curriculum_progress_debug.txt")
    print("   - 监控训练状态: watch -n 5 'grep -r \"current_stage\" */curriculum_state_detailed.json'")
    print("   - WandB面板: 检查curriculum/*指标是否正常更新")

def create_monitoring_script():
    """创建实时监控脚本"""
    monitoring_script = """#!/bin/bash
# 实时监控课程学习状态
echo "🔍 课程学习状态实时监控"
echo "按Ctrl+C退出"
echo "========================"

while true; do
    clear
    echo "⏰ $(date '+%H:%M:%S') - 课程学习状态更新"
    echo "========================"
    
    # 查找最新的状态文件
    LATEST_STATE=$(find . -name "curriculum_state_detailed.json" -type f -printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_STATE" ]; then
        echo "📁 状态文件: $LATEST_STATE"
        
        # 提取关键信息
        CURRENT_STAGE=$(jq -r '.current_stage // "N/A"' "$LATEST_STATE" 2>/dev/null)
        HISTORY_COUNT=$(jq -r '.performance_history | length // 0' "$LATEST_STATE" 2>/dev/null)
        LAST_PERFORMANCE=$(jq -r '.performance_history[-1].performance // "N/A"' "$LATEST_STATE" 2>/dev/null)
        LAST_LOSS=$(jq -r '.performance_history[-1].loss // "N/A"' "$LATEST_STATE" 2>/dev/null)
        LAST_STEP=$(jq -r '.performance_history[-1].step // "N/A"' "$LATEST_STATE" 2>/dev/null)
        
        echo "🎯 当前阶段: $CURRENT_STAGE"
        echo "📊 历史记录数: $HISTORY_COUNT"
        echo "🏆 最新性能: $LAST_PERFORMANCE"
        echo "📉 最新Loss: $LAST_LOSS"
        echo "👣 最新Step: $LAST_STEP"
        
        # 检查异常值
        if [ "$LAST_LOSS" = "Infinity" ] || [ "$LAST_LOSS" = "inf" ]; then
            echo "⚠️  警告: 检测到无穷大Loss值！"
        fi
    else
        echo "❌ 未找到状态文件"
    fi
    
    echo ""
    echo "📈 最近的调试日志 (最后5行):"
    find . -name "curriculum_progress_debug.txt" -type f -exec tail -5 {} \\; 2>/dev/null | tail -5
    
    sleep 5
done
"""
    
    with open("monitor_curriculum.sh", "w") as f:
        f.write(monitoring_script)
    
    os.chmod("monitor_curriculum.sh", 0o755)
    print("\n✅ 创建了实时监控脚本: monitor_curriculum.sh")
    print("   使用方法: ./monitor_curriculum.sh")

def main():
    """主函数"""
    print("🩺 Enhanced GRPO 课程学习问题诊断工具")
    print("="*50)
    
    # 诊断各个组件
    diagnose_curriculum_state_files()
    analyze_training_logs()
    check_wandb_files()
    
    # 提供修复建议
    provide_fix_recommendations()
    
    # 创建监控工具
    create_monitoring_script()
    
    print("\n✅ 诊断完成！")
    print("💡 建议：重新启动训练以应用修复，然后使用监控脚本观察变化")

if __name__ == "__main__":
    main() 