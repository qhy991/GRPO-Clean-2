#!/usr/bin/env python3
"""
手动DEBUG监控脚本 - 实时分析训练日志并提取生成样本信息
当训练代码没有内置DEBUG功能时使用
"""
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def parse_training_log(log_file):
    """解析训练日志，提取有用信息"""
    if not os.path.exists(log_file):
        return {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取训练指标
    metrics = {
        'steps': [],
        'losses': [],
        'rewards': [],
        'completions': [],
        'warnings': []
    }
    
    # 正则表达式模式（支持多行匹配）
    patterns = {
        'step_metrics': r"{'loss': ([-\d.]+).*?'reward': ([-\d.]+).*?'completions/mean_length': ([\d.]+).*?'epoch': ([\d.]+)}",
        'warnings': r"\[RANK \d+\] \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - WARNING - (.+?)(?=\n|\r|$)",
        'generation_info': r"completions/mean_length': ([\d.]+).*?'completions/min_length': ([\d.]+).*?'completions/max_length': ([\d.]+)"
    }
    
    # 提取步骤指标 (使用 DOTALL 标志匹配换行符)
    for match in re.finditer(patterns['step_metrics'], content, re.DOTALL):
        loss, reward, mean_length, epoch = match.groups()
        metrics['steps'].append({
            'loss': float(loss),
            'reward': float(reward),
            'mean_length': float(mean_length),
            'epoch': float(epoch),
            'timestamp': datetime.now().isoformat()
        })
    
    # 提取警告信息
    for match in re.finditer(patterns['warnings'], content, re.DOTALL):
        warning_msg = match.group(1)
        metrics['warnings'].append({
            'message': warning_msg,
            'timestamp': datetime.now().isoformat()
        })
    
    # 提取生成信息
    for match in re.finditer(patterns['generation_info'], content, re.DOTALL):
        mean_len, min_len, max_len = match.groups()
        metrics['completions'].append({
            'mean_length': float(mean_len),
            'min_length': float(min_len),
            'max_length': float(max_len),
            'timestamp': datetime.now().isoformat()
        })
    
    return metrics

def analyze_training_progress(metrics):
    """分析训练进度"""
    if not metrics['steps']:
        return {}
    
    latest_step = metrics['steps'][-1]
    
    analysis = {
        'total_steps': len(metrics['steps']),
        'latest_metrics': latest_step,
        'avg_reward': sum(s['reward'] for s in metrics['steps']) / len(metrics['steps']),
        'avg_completion_length': sum(c['mean_length'] for c in metrics['completions']) / len(metrics['completions']) if metrics['completions'] else 0,
        'warning_count': len(metrics['warnings']),
        'estimated_generations': len(metrics['steps']) * 16,  # 假设每步16个生成
        'progress_summary': {
            'current_epoch': latest_step['epoch'],
            'reward_trend': 'improving' if len(metrics['steps']) > 5 and metrics['steps'][-1]['reward'] > metrics['steps'][-5]['reward'] else 'stable',
            'completion_quality': 'good' if latest_step.get('mean_length', 0) > 100 else 'needs_improvement'
        }
    }
    
    return analysis

def save_debug_summary(output_dir, metrics, analysis):
    """保存DEBUG摘要"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细指标
    metrics_file = os.path.join(output_dir, f"manual_debug_metrics_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 保存分析结果
    analysis_file = os.path.join(output_dir, f"manual_debug_analysis_{timestamp}.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # 生成可读摘要
    summary_file = os.path.join(output_dir, f"manual_debug_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"🐛 手动DEBUG摘要 - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 训练进度:\n")
        f.write(f"  总步数: {analysis['total_steps']}\n")
        f.write(f"  当前轮次: {analysis['latest_metrics']['epoch']:.3f}\n")
        f.write(f"  估计生成样本数: {analysis['estimated_generations']}\n")
        f.write(f"  警告数量: {analysis['warning_count']}\n\n")
        
        f.write(f"📈 性能指标:\n")
        f.write(f"  最新Loss: {analysis['latest_metrics']['loss']:.4f}\n")
        f.write(f"  最新Reward: {analysis['latest_metrics']['reward']:.4f}\n")
        f.write(f"  平均Reward: {analysis['avg_reward']:.4f}\n")
        f.write(f"  平均生成长度: {analysis['avg_completion_length']:.1f}\n\n")
        
        f.write(f"🎯 质量评估:\n")
        f.write(f"  奖励趋势: {analysis['progress_summary']['reward_trend']}\n")
        f.write(f"  生成质量: {analysis['progress_summary']['completion_quality']}\n\n")
        
        if metrics['warnings']:
            f.write(f"⚠️  最近警告:\n")
            for warning in metrics['warnings'][-5:]:  # 显示最近5个警告
                f.write(f"  - {warning['message'][:100]}...\n")
    
    return metrics_file, analysis_file, summary_file

def monitor_training():
    """监控训练进程"""
    # 查找最新的训练日志
    debug_base = "./model_parallel_only_outputs/debug_data"
    log_dirs = list(Path(debug_base).glob("training_logs/*/"))
    
    if not log_dirs:
        print("❌ 未找到训练日志目录")
        return
    
    latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
    log_file = latest_log_dir / "full_training_log.txt"
    
    print(f"🔍 监控训练日志: {log_file}")
    
    output_dir = latest_log_dir / "manual_debug"
    
    while True:
        try:
            # 解析日志
            metrics = parse_training_log(log_file)
            
            if metrics['steps']:
                # 分析进度
                analysis = analyze_training_progress(metrics)
                
                # 保存DEBUG信息
                metrics_file, analysis_file, summary_file = save_debug_summary(output_dir, metrics, analysis)
                
                # 打印摘要
                print(f"\n🐛 手动DEBUG更新 ({datetime.now().strftime('%H:%M:%S')})")
                print(f"📊 步数: {analysis['total_steps']}, 轮次: {analysis['latest_metrics']['epoch']:.3f}")
                print(f"📈 Loss: {analysis['latest_metrics']['loss']:.4f}, Reward: {analysis['latest_metrics']['reward']:.4f}")
                print(f"📝 生成长度: {analysis['latest_metrics']['mean_length']:.1f}")
                print(f"⚠️  警告: {analysis['warning_count']} 个")
                print(f"💾 已保存: {os.path.basename(summary_file)}")
            else:
                print(f"⏳ 等待训练数据... ({datetime.now().strftime('%H:%M:%S')})")
            
            time.sleep(30)  # 每30秒检查一次
            
        except KeyboardInterrupt:
            print("\n🛑 监控停止")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    print("🔍 启动手动DEBUG监控...")
    print("📝 此脚本将分析训练日志并提取生成信息")
    print("⚡ 每30秒更新一次，按Ctrl+C停止")
    print("-" * 50)
    
    monitor_training() 