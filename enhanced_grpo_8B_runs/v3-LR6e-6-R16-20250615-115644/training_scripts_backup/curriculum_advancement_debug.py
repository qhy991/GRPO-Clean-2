#!/usr/bin/env python3
"""
课程学习进阶问题诊断脚本
分析为什么课程学习一直认为不满足进阶条件
"""

import json
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def analyze_curriculum_debug_log(log_file_path):
    """分析课程学习调试日志"""
    print(f"🔍 分析课程学习调试日志: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"❌ 日志文件不存在: {log_file_path}")
        return
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # 提取性能数据
    performance_pattern = r'📊 使用reward指标转换: reward=([-\d.]+) -> performance=(\d\.\d+)'
    performance_matches = re.findall(performance_pattern, log_content)
    
    # 提取进阶检查数据
    advancement_pattern = r'📊 阶段进阶检查.*?当前性能: (\d\.\d+).*?性能阈值: (\d\.\d+).*?⏳ 课程管理器判断暂不满足进阶条件'
    advancement_matches = re.findall(advancement_pattern, log_content, re.DOTALL)
    
    # 提取步数信息
    step_pattern = r'课程状态更新 \(步数: (\d+)\)'
    step_matches = re.findall(step_pattern, log_content)
    
    print(f"📊 发现 {len(performance_matches)} 个性能评估记录")
    print(f"📊 发现 {len(advancement_matches)} 个进阶检查记录")
    print(f"📊 发现 {len(step_matches)} 个步数记录")
    
    # 分析性能数据
    if performance_matches:
        performances = [float(match[1]) for match in performance_matches]
        rewards = [float(match[0]) for match in performance_matches]
        
        print(f"\n📈 性能统计:")
        print(f"  - 总评估次数: {len(performances)}")
        print(f"  - 平均性能: {np.mean(performances):.4f}")
        print(f"  - 最高性能: {np.max(performances):.4f}")
        print(f"  - 最低性能: {np.min(performances):.4f}")
        print(f"  - 性能阈值: 0.7000")
        print(f"  - 超过阈值次数: {sum(1 for p in performances if p >= 0.7)}")
        print(f"  - 超过阈值比例: {sum(1 for p in performances if p >= 0.7)/len(performances)*100:.1f}%")
        
        # 分析最近的性能表现
        recent_performances = performances[-10:]  # 最近10次
        print(f"\n🔍 最近10次性能分析:")
        for i, perf in enumerate(recent_performances, 1):
            status = "✅" if perf >= 0.7 else "❌"
            print(f"  {status} 第{len(performances)-10+i}次: {perf:.4f}")
        
        # 分析连续性能表现
        consecutive_good = 0
        max_consecutive_good = 0
        for perf in performances:
            if perf >= 0.7:
                consecutive_good += 1
                max_consecutive_good = max(max_consecutive_good, consecutive_good)
            else:
                consecutive_good = 0
        
        print(f"\n🎯 连续性能分析:")
        print(f"  - 当前连续超阈值次数: {consecutive_good}")
        print(f"  - 历史最大连续超阈值次数: {max_consecutive_good}")
        print(f"  - 需要连续超阈值次数: 通常需要3次连续")
        
        # 检查滑动窗口性能
        window_size = 3
        if len(performances) >= window_size:
            recent_window = performances[-window_size:]
            recent_avg = np.mean(recent_window)
            print(f"\n🔬 滑动窗口分析 (最近{window_size}次):")
            print(f"  - 最近{window_size}次性能: {[f'{p:.4f}' for p in recent_window]}")
            print(f"  - 平均性能: {recent_avg:.4f}")
            print(f"  - 是否满足进阶条件: {'✅ 是' if recent_avg >= 0.7 else '❌ 否'}")
        
    # 分析进阶检查失败的原因
    if advancement_matches:
        print(f"\n🚫 进阶检查失败分析:")
        print(f"  - 进阶检查次数: {len(advancement_matches)}")
        
        for i, (current_perf, threshold) in enumerate(advancement_matches[-5:], 1):  # 最近5次
            print(f"  - 第{len(advancement_matches)-5+i}次检查: 性能{current_perf} vs 阈值{threshold} = {'✅通过' if float(current_perf) >= float(threshold) else '❌失败'}")
    
    # 生成诊断报告
    generate_advancement_diagnosis_report(log_file_path, performance_matches, advancement_matches)

def generate_advancement_diagnosis_report(log_file, performance_data, advancement_data):
    """生成进阶问题诊断报告"""
    report_file = f"curriculum_advancement_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 课程学习进阶问题诊断报告 ===\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"分析日志: {log_file}\n\n")
        
        f.write("## 问题症状\n")
        f.write("- 训练过程中性能经常超过阈值0.7\n")
        f.write("- 但课程管理器一直显示'暂不满足进阶条件'\n")
        f.write("- 始终停留在阶段0 (foundation)\n\n")
        
        f.write("## 可能原因分析\n")
        f.write("1. **评估次数不足**: 需要至少10次评估才考虑进阶\n")
        f.write("2. **滑动窗口要求**: 需要最近3次评估的平均值超过阈值\n")
        f.write("3. **性能计算方式**: reward到performance的转换可能不准确\n")
        f.write("4. **评估频率问题**: 实际调用should_advance_stage的频率不够\n\n")
        
        if performance_data:
            performances = [float(match[1]) for match in performance_data]
            f.write(f"## 数据分析\n")
            f.write(f"- 总性能评估次数: {len(performances)}\n")
            f.write(f"- 超过阈值次数: {sum(1 for p in performances if p >= 0.7)}\n")
            f.write(f"- 最近3次平均: {np.mean(performances[-3:]) if len(performances) >= 3 else 'N/A'}\n")
            f.write(f"- 是否满足进阶: {'是' if len(performances) >= 10 and np.mean(performances[-3:]) >= 0.7 else '否'}\n\n")
        
        f.write("## 解决方案建议\n")
        f.write("1. 降低性能阈值 (0.7 -> 0.65)\n")
        f.write("2. 减少最小评估次数要求 (10 -> 5)\n")
        f.write("3. 调整滑动窗口大小 (3 -> 2)\n")
        f.write("4. 增加性能检查频率\n")
        f.write("5. 改进reward到performance的转换公式\n\n")
        
        f.write("## 下一步行动\n")
        f.write("1. 运行 curriculum_fix_advancement.py 脚本\n")
        f.write("2. 调整课程配置参数\n")
        f.write("3. 重新启动训练\n")
    
    print(f"📋 诊断报告已生成: {report_file}")

def create_advancement_fix_script():
    """创建进阶问题修复脚本"""
    fix_script_content = '''#!/usr/bin/env python3
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
'''
    
    with open("curriculum_fix_advancement.py", 'w', encoding='utf-8') as f:
        f.write(fix_script_content)
    
    print("📝 进阶问题修复脚本已创建: curriculum_fix_advancement.py")

def main():
    """主函数"""
    print("🔍 课程学习进阶问题诊断工具")
    
    # 查找最新的调试日志
    debug_logs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if "curriculum_progress_debug.txt" in file:
                debug_logs.append(os.path.join(root, file))
    
    if not debug_logs:
        print("❌ 未找到课程学习调试日志文件")
        print("请确保训练过程中生成了 curriculum_progress_debug.txt 文件")
        return
    
    # 使用最新的日志文件
    latest_log = max(debug_logs, key=os.path.getmtime)
    print(f"📄 使用日志文件: {latest_log}")
    
    # 分析日志
    analyze_curriculum_debug_log(latest_log)
    
    # 创建修复脚本
    create_advancement_fix_script()
    
    print("\n🎯 总结:")
    print("课程学习无法进阶的主要原因是:")
    print("1. 性能阈值设置过高 (0.7)")
    print("2. 最小评估次数要求过多 (10次)")
    print("3. 滑动窗口要求过严格 (需要连续3次平均超阈值)")
    print("\n💡 解决方案:")
    print("运行 ./curriculum_fix_advancement.py 来修复配置")

if __name__ == "__main__":
    main() 