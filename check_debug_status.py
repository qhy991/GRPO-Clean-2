#!/usr/bin/env python3
"""
检查DEBUG状态和生成样本保存情况
"""
import os
import json
from pathlib import Path
from datetime import datetime

def check_debug_status():
    print("🔍 DEBUG状态检查")
    print("=" * 50)
    
    # 检查环境变量
    print("\n📊 环境变量检查:")
    debug_vars = [
        'DEBUG_MODE', 'SAVE_ALL_GENERATIONS', 'SAVE_FAILED_GENERATIONS',
        'SAVE_SUCCESSFUL_GENERATIONS', 'SAVE_DETAILED_METRICS',
        'SAVE_MODEL_OUTPUTS', 'SAVE_REWARD_DETAILS',
        'DEBUG_SAMPLE_FREQUENCY', 'DEBUG_OUTPUT_BASE',
        'GENERATIONS_OUTPUT_DIR', 'FAILED_GENERATIONS_DIR',
        'SUCCESSFUL_GENERATIONS_DIR'
    ]
    
    for var in debug_vars:
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")
    
    # 检查目录结构
    print("\n📁 目录结构检查:")
    debug_base = os.environ.get('DEBUG_OUTPUT_BASE', './model_parallel_only_outputs/debug_data')
    
    if os.path.exists(debug_base):
        print(f"  DEBUG基础目录存在: {debug_base}")
        
        # 列出所有子目录和文件
        for root, dirs, files in os.walk(debug_base):
            level = root.replace(debug_base, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"  {indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"  {subindent}{file} ({size} bytes, {mtime})")
    else:
        print(f"  ❌ DEBUG目录不存在: {debug_base}")
    
    # 检查进程信息
    print("\n🔄 进程信息:")
    if os.path.exists(debug_base):
        process_files = list(Path(debug_base).glob("**/process_info_*.json"))
        if process_files:
            latest_file = max(process_files, key=lambda p: p.stat().st_mtime)
            print(f"  最新进程文件: {latest_file}")
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    print(f"  PID: {data.get('pid', '未知')}")
                    print(f"  时间: {data.get('timestamp', '未知')}")
                    print(f"  GPU内存: {data.get('gpu_memory', '未知')}")
            except Exception as e:
                print(f"  ❌ 读取进程文件失败: {e}")
        else:
            print("  ❌ 没有找到进程信息文件")
    
    # 统计生成文件
    print("\n📈 生成文件统计:")
    if os.path.exists(debug_base):
        generations_dir = os.path.join(debug_base, "generations")
        failed_dir = os.path.join(debug_base, "failed_generations")
        successful_dir = os.path.join(debug_base, "successful_generations")
        metrics_dir = os.path.join(debug_base, "detailed_metrics")
        
        for name, path in [
            ("生成样本", generations_dir),
            ("失败样本", failed_dir),
            ("成功样本", successful_dir),
            ("详细指标", metrics_dir)
        ]:
            if os.path.exists(path):
                files = list(Path(path).rglob("*.json"))
                print(f"  {name}: {len(files)} 个文件")
                if files:
                    latest = max(files, key=lambda p: p.stat().st_mtime)
                    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                    print(f"    最新: {latest.name} ({mtime})")
            else:
                print(f"  {name}: 目录不存在")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_debug_status() 