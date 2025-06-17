#!/usr/bin/env python3
"""
简单的DEBUG数据收集器
独立于main.py运行，收集训练过程中的基本信息
"""

import os
import json
import time
import subprocess
import threading
from datetime import datetime
import psutil
import signal
import sys

class SimpleDebugCollector:
    def __init__(self):
        self.debug_base = os.environ.get('DEBUG_OUTPUT_BASE', './model_parallel_only_outputs/debug_data')
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.debug_base, 'training_logs', self.timestamp)
        self.running = True
        
        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        print(f"\n🛑 收到信号 {signum}，正在停止DEBUG收集器...")
        self.running = False
        
    def collect_system_info(self):
        """收集系统信息"""
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        # 收集GPU信息
        try:
            gpu_info = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], text=True).strip().split('\n')
            
            system_info['gpus'] = []
            for gpu_line in gpu_info:
                if gpu_line.strip():
                    parts = [p.strip() for p in gpu_line.split(',')]
                    if len(parts) >= 6:
                        system_info['gpus'].append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'utilization': int(parts[4]),
                            'temperature': int(parts[5])
                        })
        except Exception as e:
            system_info['gpu_error'] = str(e)
            
        return system_info
    
    def monitor_training_process(self):
        """监控训练进程"""
        process_info = {
            'timestamp': datetime.now().isoformat(),
            'training_process': None,
            'python_processes': []
        }
        
        # 查找训练进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'main.py' in cmdline and 'python' in proc.info['name'].lower():
                    process_info['training_process'] = {
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:200] + '...' if len(cmdline) > 200 else cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / (1024 * 1024) if proc.info['memory_info'] else 0
                    }
                    break
                    
                if 'python' in proc.info['name'].lower():
                    process_info['python_processes'].append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': (cmdline[:100] + '...' if len(cmdline) > 100 else cmdline)
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return process_info
    
    def collect_debug_info(self):
        """收集DEBUG信息"""
        while self.running:
            try:
                timestamp_str = datetime.now().strftime('%H%M%S')
                
                # 收集系统信息
                system_info = self.collect_system_info()
                system_file = os.path.join(self.log_dir, f'system_info_{timestamp_str}.json')
                with open(system_file, 'w', encoding='utf-8') as f:
                    json.dump(system_info, f, indent=2, ensure_ascii=False)
                
                # 收集进程信息
                process_info = self.monitor_training_process()
                process_file = os.path.join(self.log_dir, f'process_info_{timestamp_str}.json')
                with open(process_file, 'w', encoding='utf-8') as f:
                    json.dump(process_info, f, indent=2, ensure_ascii=False)
                
                # 打印简要信息
                if process_info['training_process']:
                    training_pid = process_info['training_process']['pid']
                    cpu_usage = process_info['training_process']['cpu_percent']
                    memory_mb = process_info['training_process']['memory_mb']
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练进程 PID:{training_pid}, CPU:{cpu_usage}%, MEM:{memory_mb:.1f}MB")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 未检测到训练进程")
                
                # GPU信息
                if 'gpus' in system_info:
                    for gpu in system_info['gpus']:
                        print(f"  GPU{gpu['index']}: {gpu['utilization']}% 利用率, {gpu['memory_used']}/{gpu['memory_total']}MB 内存, {gpu['temperature']}°C")
                
                time.sleep(30)  # 每30秒收集一次
                
            except Exception as e:
                print(f"❌ 收集DEBUG信息时出错: {e}")
                time.sleep(5)
    
    def start(self):
        """启动DEBUG收集器"""
        print(f"🚀 启动简单DEBUG收集器")
        print(f"📁 DEBUG目录: {self.debug_base}")
        print(f"📝 日志目录: {self.log_dir}")
        print(f"⏱️  收集间隔: 30秒")
        print(f"🛑 按 Ctrl+C 停止")
        print("-" * 60)
        
        try:
            self.collect_debug_info()
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号，正在停止...")
        finally:
            self.running = False
            print("✅ DEBUG收集器已停止")

def main():
    collector = SimpleDebugCollector()
    collector.start()

if __name__ == "__main__":
    main() 