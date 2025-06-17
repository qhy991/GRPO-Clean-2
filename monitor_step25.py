#!/usr/bin/env python3
"""
专门监控步数25和生成样本保存的脚本
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_step25():
    """监控步数25和生成样本保存"""
    print("🔍 监控步数25和生成样本保存...")
    print("📝 预期：当训练达到步数25时，会保存生成样本")
    print("=" * 60)
    
    # 输出目录
    output_base = "./model_parallel_only_outputs/DEBUG-model-parallel-LR2e-5-R64-BS2x8-20250616-230237"
    samples_dir = os.path.join(output_base, "generated_samples_detailed")
    
    # 手动DEBUG目录
    manual_debug_dir = "./model_parallel_only_outputs/debug_data/training_logs/20250616-230237/manual_debug"
    
    last_step = 0
    step25_triggered = False
    
    while True:
        try:
            # 检查最新的手动DEBUG摘要
            if os.path.exists(manual_debug_dir):
                summary_files = list(Path(manual_debug_dir).glob("manual_debug_summary_*.txt"))
                if summary_files:
                    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
                    
                    try:
                        with open(latest_summary, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 提取步数
                        for line in content.split('\n'):
                            if line.strip().startswith('总步数:'):
                                current_step = int(line.split(':')[1].strip())
                                break
                        else:
                            current_step = last_step
                        
                        # 检查步数变化
                        if current_step > last_step:
                            print(f"📊 步数更新: {last_step} → {current_step}")
                            last_step = current_step
                            
                            # 检查是否接近步数25
                            if current_step >= 20 and current_step < 25:
                                print(f"⚠️  接近步数25! 当前步数: {current_step}")
                            elif current_step >= 25 and not step25_triggered:
                                print(f"🎯 达到步数25! 当前步数: {current_step}")
                                step25_triggered = True
                                
                                # 等待几秒让回调执行
                                print("⏳ 等待回调执行...")
                                time.sleep(10)
                                
                                # 检查生成样本
                                check_generated_samples(samples_dir)
                        
                        # 实时检查生成样本目录
                        if step25_triggered or current_step >= 25:
                            sample_files = list(Path(samples_dir).glob("*.json")) if os.path.exists(samples_dir) else []
                            if sample_files:
                                print(f"🎉 发现生成样本文件: {len(sample_files)} 个")
                                for f in sample_files[-3:]:  # 显示最新的3个文件
                                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                                    print(f"  - {f.name} ({mtime})")
                                print("✅ 修复成功！生成样本开始保存！")
                                return True
                        
                    except Exception as e:
                        print(f"❌ 解析摘要文件失败: {e}")
            
            print(f"⏳ 当前步数: {last_step}, 等待步数25... ({datetime.now().strftime('%H:%M:%S')})")
            time.sleep(20)  # 每20秒检查一次
            
        except KeyboardInterrupt:
            print("\n🛑 监控停止")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(5)

def check_generated_samples(samples_dir):
    """检查生成样本目录"""
    print(f"\n🔍 检查生成样本目录: {samples_dir}")
    
    if not os.path.exists(samples_dir):
        print("❌ 生成样本目录不存在")
        return False
    
    sample_files = list(Path(samples_dir).glob("*.json"))
    
    if not sample_files:
        print("❌ 没有找到生成样本文件")
        return False
    
    print(f"✅ 找到 {len(sample_files)} 个生成样本文件:")
    
    for i, sample_file in enumerate(sample_files[:5]):  # 显示前5个文件
        try:
            mtime = datetime.fromtimestamp(sample_file.stat().st_mtime)
            size = sample_file.stat().st_size
            
            print(f"  {i+1}. {sample_file.name}")
            print(f"     时间: {mtime}, 大小: {size} bytes")
            
            # 读取文件内容预览
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            step = data.get('step', 'N/A')
            task_id = data.get('dataset_original_sample_info', {}).get('task_id', 'N/A')
            reasoning = data.get('generated_result', {}).get('reasoning', '')
            code_preview = data.get('generated_result', {}).get('code', '')[:100]
            
            print(f"     步数: {step}, 任务: {task_id}")
            print(f"     推理: {reasoning[:100]}...")
            print(f"     代码: {code_preview}...")
            print()
            
        except Exception as e:
            print(f"     ❌ 读取文件失败: {e}")
    
    return True

if __name__ == "__main__":
    monitor_step25() 