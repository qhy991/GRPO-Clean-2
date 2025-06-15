#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time
import re
import os
import logging
import argparse
import glob
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_monitor.log"),
        logging.StreamHandler()
    ]
)

def get_gpu_memory_info():
    """
    使用nvidia-smi命令获取GPU显存信息
    返回: 列表，每个元素是一张GPU卡的可用显存(MB)
    """
    try:
        # 执行nvidia-smi命令获取GPU信息
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        # 解析结果 - 每行表示一张卡的空闲显存(MB)
        memory_free_info = [int(x) for x in result.strip().split('\n')]
        
        return memory_free_info
    except Exception as e:
        logging.error(f"获取GPU信息失败: {e}")
        return []

def find_latest_checkpoint(base_dir):
    """
    在指定目录下找到最新的checkpoint文件夹
    
    参数:
        base_dir: 基础目录路径
    
    返回:
        最新checkpoint的完整路径，如果没有找到则返回None
    """
    try:
        if not os.path.exists(base_dir):
            logging.error(f"基础目录不存在: {base_dir}")
            return None
        
        # 查找所有checkpoint-*文件夹
        checkpoint_pattern = os.path.join(base_dir, "checkpoint-*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)
        
        if not checkpoint_dirs:
            logging.warning(f"在 {base_dir} 中未找到checkpoint文件夹")
            return None
        
        # 按修改时间排序，获取最新的
        latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
        
        logging.info(f"找到最新checkpoint: {latest_checkpoint}")
        return latest_checkpoint
        
    except Exception as e:
        logging.error(f"查找最新checkpoint时出错: {e}")
        return None

def update_script_with_checkpoint(script_path, checkpoint_dir, backup=True):
    """
    更新脚本文件中的RESUME_FROM_CHECKPOINT_DIR变量
    
    参数:
        script_path: 脚本文件路径
        checkpoint_dir: 新的checkpoint目录路径
        backup: 是否创建备份文件
    
    返回:
        更新是否成功
    """
    try:
        # 读取原始脚本内容
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        if backup:
            backup_path = script_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"已创建备份文件: {backup_path}")
        
        # 使用正则表达式替换RESUME_FROM_CHECKPOINT_DIR的值
        pattern = r'RESUME_FROM_CHECKPOINT_DIR\s*=\s*["\'][^"\']*["\']'
        replacement = f'RESUME_FROM_CHECKPOINT_DIR="{checkpoint_dir}"'
        
        updated_content = re.sub(pattern, replacement, content)
        
        # 检查是否找到并替换了变量
        if updated_content == content:
            logging.warning("未找到RESUME_FROM_CHECKPOINT_DIR变量，尝试在文件末尾添加")
            updated_content += f'\nRESUME_FROM_CHECKPOINT_DIR="{checkpoint_dir}"\n'
        
        # 写入更新后的内容
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logging.info(f"已更新脚本中的checkpoint路径为: {checkpoint_dir}")
        return True
        
    except Exception as e:
        logging.error(f"更新脚本失败: {e}")
        return False

def execute_task(script_path, checkpoint_base_dir=None):
    """
    执行指定的shell脚本，如果指定了checkpoint_base_dir，会先更新checkpoint路径
    """
    try:
        # 如果指定了checkpoint基础目录，先更新checkpoint路径
        if checkpoint_base_dir:
            latest_checkpoint = find_latest_checkpoint(checkpoint_base_dir)
            if latest_checkpoint:
                if not update_script_with_checkpoint(script_path, latest_checkpoint):
                    logging.error("更新checkpoint路径失败")
                    return False
            else:
                logging.error("未找到可用的checkpoint，任务可能无法正常执行")
        
        logging.info(f"开始执行任务脚本: {script_path}")
        
        # 确保脚本有执行权限
        os.chmod(script_path, 0o755)
        
        # 执行脚本
        process = subprocess.Popen(
            ['bash', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logging.info("任务执行成功")
            logging.info(f"输出: {stdout.decode('utf-8')}")
            return True
        else:
            logging.error(f"任务执行失败，返回码: {process.returncode}")
            logging.error(f"错误: {stderr.decode('utf-8')}")
            return False
    except Exception as e:
        logging.error(f"执行任务时出错: {e}")
        return False

def monitor_gpu_resources(script_path, memory_threshold=150000, check_interval=60, max_retries=3, checkpoint_base_dir=None):
    """
    监控GPU资源，当可用显存超过阈值时执行任务
    
    参数:
        script_path: 要执行的shell脚本路径
        memory_threshold: 显存阈值(MB)，默认150000MB (150GB)
        check_interval: 检查间隔(秒)，默认60秒
        max_retries: 任务执行失败时的最大重试次数
        checkpoint_base_dir: checkpoint文件夹的基础目录
    """
    logging.info(f"开始监控GPU资源，显存阈值: {memory_threshold}MB")
    logging.info(f"将执行脚本: {script_path}")
    if checkpoint_base_dir:
        logging.info(f"Checkpoint基础目录: {checkpoint_base_dir}")
    
    attempts = 0
    task_executed = False
    
    while not task_executed and attempts < max_retries:
        try:
            # 获取GPU显存信息
            memory_info = get_gpu_memory_info()
            
            if len(memory_info) < 2:
                logging.warning(f"检测到的GPU数量少于2，实际数量: {len(memory_info)}")
                time.sleep(check_interval)
                continue
            
            # 计算总可用显存
            total_free_memory = sum(memory_info)
            
            logging.info(f"GPU #0 可用显存: {memory_info[0]}MB")
            logging.info(f"GPU #1 可用显存: {memory_info[1]}MB")
            logging.info(f"总可用显存: {total_free_memory}MB")
            
            # 检查是否满足条件
            if total_free_memory >= memory_threshold:
                logging.info(f"满足条件: 总可用显存({total_free_memory}MB) >= 阈值({memory_threshold}MB)")
                
                # 执行任务
                if execute_task(script_path, checkpoint_base_dir):
                    task_executed = True
                    logging.info("任务执行成功，停止监控")
                else:
                    attempts += 1
                    logging.warning(f"任务执行失败，尝试次数: {attempts}/{max_retries}")
                    time.sleep(check_interval)
            else:
                logging.info(f"条件未满足: 总可用显存({total_free_memory}MB) < 阈值({memory_threshold}MB)")
                logging.info(f"{check_interval}秒后重新检查...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logging.info("收到中断信号，停止监控")
            break
        except Exception as e:
            logging.error(f"监控过程中出错: {e}")
            time.sleep(check_interval)
    
    if not task_executed and attempts >= max_retries:
        logging.error(f"达到最大重试次数({max_retries})，停止尝试")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='监控GPU资源并在满足条件时执行任务')
    
    parser.add_argument('script', help='要执行的shell脚本路径')
    parser.add_argument('--threshold', type=int, default=40000, 
                        help='显存阈值(MB)，默认40000MB')
    parser.add_argument('--interval', type=int, default=60, 
                        help='检查间隔(秒)，默认60秒')
    parser.add_argument('--retries', type=int, default=10, 
                        help='任务执行失败时的最大重试次数，默认3次')
    parser.add_argument('--checkpoint-dir', type=str, 
                        help='checkpoint文件夹的基础目录，用于自动查找最新checkpoint')
    
    args = parser.parse_args()
    
    # 验证脚本文件存在
    if not os.path.isfile(args.script):
        logging.error(f"脚本文件不存在: {args.script}")
        exit(1)
    
    # 验证checkpoint目录（如果指定了的话）
    if args.checkpoint_dir and not os.path.isdir(args.checkpoint_dir):
        logging.error(f"Checkpoint目录不存在: {args.checkpoint_dir}")
        exit(1)
    
    # 开始监控
    monitor_gpu_resources(
        args.script, 
        memory_threshold=args.threshold,
        check_interval=args.interval,
        max_retries=args.retries,
        checkpoint_base_dir=args.checkpoint_dir
    )