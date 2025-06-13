#!/usr/bin/env python3
"""
整理调试文件脚本 - 将散乱的调试文件移动到对应的子文件夹中
"""

import os
import shutil
import glob
from pathlib import Path

def organize_debug_files(base_dir):
    """整理调试文件到子文件夹"""
    print(f"🚀 开始整理调试文件: {base_dir}")
    
    # 创建子目录
    reward_debug_dir = os.path.join(base_dir, "reward_debug")
    missing_code_dir = os.path.join(reward_debug_dir, "missing_code")
    validation_errors_dir = os.path.join(reward_debug_dir, "validation_errors")
    
    os.makedirs(missing_code_dir, exist_ok=True)
    os.makedirs(validation_errors_dir, exist_ok=True)
    
    moved_files = {"missing_code": 0, "validation_errors": 0}
    
    # 移动 missingcode 文件
    missingcode_pattern = os.path.join(base_dir, "*missingcode*.txt")
    missingcode_files = glob.glob(missingcode_pattern)
    
    for file_path in missingcode_files:
        filename = os.path.basename(file_path)
        new_path = os.path.join(missing_code_dir, filename)
        try:
            shutil.move(file_path, new_path)
            moved_files["missing_code"] += 1
            print(f"✅ 移动: {filename} -> reward_debug/missing_code/")
        except Exception as e:
            print(f"❌ 移动失败: {filename} - {e}")
    
    # 移动 validation_error 文件
    validation_pattern = os.path.join(base_dir, "*validation_error*.v")
    validation_files = glob.glob(validation_pattern)
    
    for file_path in validation_files:
        filename = os.path.basename(file_path)
        new_path = os.path.join(validation_errors_dir, filename)
        try:
            shutil.move(file_path, new_path)
            moved_files["validation_errors"] += 1
            print(f"✅ 移动: {filename} -> reward_debug/validation_errors/")
        except Exception as e:
            print(f"❌ 移动失败: {filename} - {e}")
    
    print(f"\n📊 整理完成:")
    print(f"  - Missing code 文件: {moved_files['missing_code']}")
    print(f"  - Validation error 文件: {moved_files['validation_errors']}")
    print(f"  - 总计移动: {sum(moved_files.values())} 个文件")
    
    return moved_files

def organize_all_run_directories(grpo_runs_dir):
    """整理所有运行目录中的调试文件"""
    if not os.path.exists(grpo_runs_dir):
        print(f"❌ 目录不存在: {grpo_runs_dir}")
        return
    
    total_moved = {"missing_code": 0, "validation_errors": 0}
    
    # 查找所有包含调试文件的运行目录
    for item in os.listdir(grpo_runs_dir):
        item_path = os.path.join(grpo_runs_dir, item)
        if os.path.isdir(item_path):
            # 检查是否有调试文件
            has_debug_files = (
                len(glob.glob(os.path.join(item_path, "*missingcode*.txt"))) > 0 or
                len(glob.glob(os.path.join(item_path, "*validation_error*.v"))) > 0
            )
            
            if has_debug_files:
                print(f"\n📁 处理目录: {item}")
                moved = organize_debug_files(item_path)
                total_moved["missing_code"] += moved["missing_code"]
                total_moved["validation_errors"] += moved["validation_errors"]
    
    print(f"\n🎉 全部整理完成:")
    print(f"  - Missing code 文件: {total_moved['missing_code']}")
    print(f"  - Validation error 文件: {total_moved['validation_errors']}")
    print(f"  - 总计移动: {sum(total_moved.values())} 个文件")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if os.path.isdir(target_dir):
            organize_debug_files(target_dir)
        else:
            print(f"❌ 目录不存在: {target_dir}")
    else:
        # 默认整理所有enhanced_grpo_v3_runs目录
        grpo_runs_dir = "./enhanced_grpo_v3_runs"
        organize_all_run_directories(grpo_runs_dir)
        
        print(f"\n🔧 修改已应用到代码中，未来的调试文件将自动保存到子文件夹:")
        print(f"  - Missing code: reward_debug/missing_code/")
        print(f"  - Validation errors: reward_debug/validation_errors/")
        print(f"\n💡 如果需要整理特定目录，请使用: python organize_debug_files.py <目录路径>") 