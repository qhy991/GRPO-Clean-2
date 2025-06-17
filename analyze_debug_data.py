#!/usr/bin/env python3
"""
DEBUG数据分析脚本
用于分析训练过程中保存的所有DEBUG数据
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def analyze_debug_data(debug_output_base):
    """分析DEBUG数据的主函数"""
    
    print(f"🔍 开始分析DEBUG数据: {debug_output_base}")
    
    # 检查目录是否存在
    if not os.path.exists(debug_output_base):
        print(f"❌ DEBUG目录不存在: {debug_output_base}")
        return
    
    # 获取所有时间戳目录
    timestamp_dirs = glob.glob(os.path.join(debug_output_base, "*", "*"))
    timestamp_dirs = [d for d in timestamp_dirs if os.path.isdir(d)]
    
    print(f"📁 找到 {len(timestamp_dirs)} 个时间戳目录")
    
    for timestamp_dir in sorted(timestamp_dirs):
        print(f"\n📊 分析时间戳: {os.path.basename(timestamp_dir)}")
        analyze_timestamp_data(timestamp_dir)
    
    # 生成总体分析报告
    generate_overall_report(debug_output_base, timestamp_dirs)

def analyze_timestamp_data(timestamp_dir):
    """分析单个时间戳的数据"""
    
    # 分析生成样本
    generations_dir = os.path.join(timestamp_dir, "generations")
    if os.path.exists(generations_dir):
        analyze_generations(generations_dir)
    
    # 分析失败样本
    failed_dir = os.path.join(timestamp_dir, "failed_generations") 
    if os.path.exists(failed_dir):
        analyze_failed_generations(failed_dir)
    
    # 分析成功样本
    success_dir = os.path.join(timestamp_dir, "successful_generations")
    if os.path.exists(success_dir):
        analyze_successful_generations(success_dir)
    
    # 分析详细指标
    metrics_dir = os.path.join(timestamp_dir, "detailed_metrics")
    if os.path.exists(metrics_dir):
        analyze_detailed_metrics(metrics_dir)
    
    # 分析奖励详情
    reward_dir = os.path.join(timestamp_dir, "reward_details")
    if os.path.exists(reward_dir):
        analyze_reward_details(reward_dir)

def analyze_generations(generations_dir):
    """分析生成样本"""
    json_files = glob.glob(os.path.join(generations_dir, "*.json"))
    print(f"  📝 生成样本文件: {len(json_files)}")
    
    if json_files:
        total_samples = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_samples += len(data)
                    else:
                        total_samples += 1
            except Exception as e:
                print(f"    ❌ 读取文件失败: {json_file}, 错误: {e}")
        
        print(f"  📊 总生成样本数: {total_samples}")

def analyze_failed_generations(failed_dir):
    """分析失败的生成样本"""
    json_files = glob.glob(os.path.join(failed_dir, "*.json"))
    print(f"  ❌ 失败样本文件: {len(json_files)}")
    
    if json_files:
        failure_reasons = {}
        total_failed = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            total_failed += 1
                            reason = item.get('failure_reason', 'unknown')
                            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    else:
                        total_failed += 1
                        reason = data.get('failure_reason', 'unknown')
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                        
            except Exception as e:
                print(f"    ❌ 读取失败文件错误: {json_file}, 错误: {e}")
        
        print(f"  📊 总失败样本数: {total_failed}")
        print(f"  📋 失败原因统计:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count}")

def analyze_successful_generations(success_dir):
    """分析成功的生成样本"""
    json_files = glob.glob(os.path.join(success_dir, "*.json"))
    print(f"  ✅ 成功样本文件: {len(json_files)}")
    
    if json_files:
        total_success = 0
        reward_scores = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            total_success += 1
                            if 'reward_score' in item:
                                reward_scores.append(item['reward_score'])
                    else:
                        total_success += 1
                        if 'reward_score' in data:
                            reward_scores.append(data['reward_score'])
                            
            except Exception as e:
                print(f"    ❌ 读取成功文件错误: {json_file}, 错误: {e}")
        
        print(f"  📊 总成功样本数: {total_success}")
        if reward_scores:
            print(f"  🏆 平均奖励分数: {sum(reward_scores)/len(reward_scores):.4f}")
            print(f"  🏆 最高奖励分数: {max(reward_scores):.4f}")
            print(f"  🏆 最低奖励分数: {min(reward_scores):.4f}")

def analyze_detailed_metrics(metrics_dir):
    """分析详细指标"""
    json_files = glob.glob(os.path.join(metrics_dir, "*.json"))
    print(f"  📈 指标文件: {len(json_files)}")

def analyze_reward_details(reward_dir):
    """分析奖励详情"""
    json_files = glob.glob(os.path.join(reward_dir, "*.json"))
    print(f"  🎯 奖励详情文件: {len(json_files)}")

def generate_overall_report(debug_output_base, timestamp_dirs):
    """生成总体分析报告"""
    report_file = os.path.join(debug_output_base, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== DEBUG数据分析报告 ===\n")
        f.write(f"分析时间: {datetime.now()}\n")
        f.write(f"数据目录: {debug_output_base}\n")
        f.write(f"时间戳目录数: {len(timestamp_dirs)}\n")
        f.write("\n")
        
        # 这里可以添加更详细的统计分析
        
    print(f"\n📄 分析报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='分析训练DEBUG数据')
    parser.add_argument('--debug_dir', 
                       default='./model_parallel_only_outputs/debug_data',
                       help='DEBUG数据目录路径')
    
    args = parser.parse_args()
    
    analyze_debug_data(args.debug_dir)

if __name__ == "__main__":
    main() 