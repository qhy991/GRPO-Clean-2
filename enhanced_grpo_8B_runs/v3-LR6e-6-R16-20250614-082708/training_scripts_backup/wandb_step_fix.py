#!/usr/bin/env python3
"""
WandB Step 同步修复模块
解决 Enhanced GRPO 训练中的 step 不匹配问题

主要问题：
1. 多个回调同时使用不同的 step 值进行 wandb.log
2. 训练器的 global_step 和回调的内部计数器不同步
3. 异步日志记录导致步数乱序

解决方案：
1. 统一步数管理器
2. 按优先级排队日志
3. 批量提交避免冲突
"""

import wandb
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import queue

logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """单个日志条目"""
    data: Dict[str, Any]
    step: int
    timestamp: float
    priority: int  # 0=最高优先级, 数字越大优先级越低
    source: str    # 日志来源 (trainer, callback, inference等)
    commit: bool = True

class WandBStepManager:
    """WandB步数统一管理器
    
    确保所有的 wandb.log 调用都使用一致的步数，避免 step 冲突
    """
    
    def __init__(self, buffer_size: int = 100, flush_interval: float = 2.0):
        self.current_step = 0
        self.last_logged_step = -1
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # 日志缓冲区 - 按步数分组
        self.log_buffer: Dict[int, List[LogEntry]] = defaultdict(list)
        self.pending_logs: deque = deque(maxlen=buffer_size)
        
        # 线程安全
        self.lock = threading.RLock()
        self.flush_timer: Optional[threading.Timer] = None
        
        # 统计信息
        self.stats = {
            'total_logs': 0,
            'dropped_logs': 0,
            'step_conflicts': 0,
            'successful_flushes': 0
        }
        
        logger.info("🔧 WandB步数管理器已初始化")
    
    def update_step(self, new_step: int, source: str = "trainer"):
        """更新当前步数"""
        with self.lock:
            if new_step > self.current_step:
                old_step = self.current_step
                self.current_step = new_step
                logger.debug(f"📈 步数更新: {old_step} -> {new_step} (来源: {source})")
                
                # 如果步数跳跃太大，可能需要刷新缓冲区
                if new_step - old_step > 5:
                    logger.warning(f"⚠️ 检测到步数大幅跳跃: +{new_step - old_step}, 强制刷新缓冲区")
                    self._force_flush()
            elif new_step < self.current_step:
                logger.warning(f"⚠️ 检测到步数回退: {new_step} < {self.current_step} (来源: {source})")
                self.stats['step_conflicts'] += 1
    
    def safe_log(self, 
                 data: Dict[str, Any], 
                 step: Optional[int] = None, 
                 priority: int = 5,
                 source: str = "unknown",
                 commit: bool = True,
                 force_immediate: bool = False) -> bool:
        """安全的WandB日志记录
        
        Args:
            data: 要记录的数据
            step: 步数 (None = 使用当前步数)
            priority: 优先级 (0=最高, 数字越大优先级越低)
            source: 日志来源标识
            commit: 是否立即提交
            force_immediate: 是否强制立即记录 (跳过缓冲)
        
        Returns:
            bool: 是否成功记录/缓冲
        """
        if not wandb.run:
            logger.debug("WandB run 未初始化，跳过日志记录")
            return False
        
        with self.lock:
            # 确定使用的步数
            log_step = step if step is not None else self.current_step
            
            # 检查步数有效性
            if log_step < self.last_logged_step and not force_immediate:
                logger.debug(f"跳过过期步数日志: {log_step} < {self.last_logged_step} (来源: {source})")
                self.stats['dropped_logs'] += 1
                return False
            
            # 创建日志条目
            entry = LogEntry(
                data=data.copy(),
                step=log_step,
                timestamp=time.time(),
                priority=priority,
                source=source,
                commit=commit
            )
            
            # 添加元信息
            entry.data.update({
                '_step_manager_source': source,
                '_step_manager_priority': priority,
                '_step_manager_timestamp': entry.timestamp
            })
            
            self.stats['total_logs'] += 1
            
            if force_immediate:
                # 立即记录
                return self._immediate_log(entry)
            else:
                # 添加到缓冲区
                self.log_buffer[log_step].append(entry)
                self.pending_logs.append(entry)
                
                # 定期刷新
                self._schedule_flush()
                
                # 如果缓冲区满了，强制刷新
                if len(self.pending_logs) >= self.buffer_size:
                    logger.debug("缓冲区已满，强制刷新")
                    self._force_flush()
                
                return True
    
    def _immediate_log(self, entry: LogEntry) -> bool:
        """立即记录单个条目"""
        try:
            wandb.log(entry.data, step=entry.step, commit=entry.commit)
            self.last_logged_step = max(self.last_logged_step, entry.step)
            logger.debug(f"✅ 立即记录成功: step={entry.step}, source={entry.source}")
            return True
        except Exception as e:
            logger.error(f"❌ 立即记录失败: {e}")
            return False
    
    def _schedule_flush(self):
        """安排定时刷新"""
        if self.flush_timer is None or not self.flush_timer.is_alive():
            self.flush_timer = threading.Timer(self.flush_interval, self._flush_buffer)
            self.flush_timer.daemon = True
            self.flush_timer.start()
    
    def _force_flush(self):
        """强制刷新缓冲区"""
        if self.flush_timer and self.flush_timer.is_alive():
            self.flush_timer.cancel()
        self._flush_buffer()
    
    def _flush_buffer(self):
        """刷新日志缓冲区"""
        with self.lock:
            if not self.log_buffer:
                return
            
            try:
                # 按步数排序，然后按优先级排序
                sorted_steps = sorted(self.log_buffer.keys())
                total_entries = 0
                
                for step in sorted_steps:
                    entries = self.log_buffer[step]
                    if not entries:
                        continue
                    
                    # 按优先级排序
                    entries.sort(key=lambda x: (x.priority, x.timestamp))
                    
                    # 合并同一步数的数据
                    merged_data = {}
                    commit_needed = False
                    
                    for entry in entries:
                        merged_data.update(entry.data)
                        if entry.commit:
                            commit_needed = True
                        total_entries += 1
                    
                    # 检查步数冲突
                    if step < self.last_logged_step:
                        logger.warning(f"⚠️ 跳过冲突步数: {step} < {self.last_logged_step}")
                        self.stats['step_conflicts'] += 1
                        continue
                    
                    # 记录合并后的数据
                    try:
                        wandb.log(merged_data, step=step, commit=commit_needed)
                        self.last_logged_step = max(self.last_logged_step, step)
                        logger.debug(f"✅ 批量记录成功: step={step}, entries={len(entries)}")
                    except wandb.errors.Error as e:
                        if "step" in str(e).lower():
                            logger.warning(f"⚠️ WandB步数冲突 (step={step}): {e}")
                            self.stats['step_conflicts'] += 1
                        else:
                            logger.error(f"❌ WandB记录失败 (step={step}): {e}")
                            raise
                
                # 清空缓冲区
                self.log_buffer.clear()
                self.pending_logs.clear()
                self.stats['successful_flushes'] += 1
                
                if total_entries > 0:
                    logger.debug(f"🔄 缓冲区刷新完成: {total_entries} 条日志")
                
            except Exception as e:
                logger.error(f"❌ 缓冲区刷新失败: {e}")
                # 清空缓冲区以避免无限重试
                self.log_buffer.clear()
                self.pending_logs.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                **self.stats,
                'current_step': self.current_step,
                'last_logged_step': self.last_logged_step,
                'buffer_size': len(self.pending_logs),
                'pending_steps': len(self.log_buffer)
            }
    
    def finalize(self):
        """结束时清理"""
        logger.info("🔧 正在清理WandB步数管理器...")
        self._force_flush()
        if self.flush_timer and self.flush_timer.is_alive():
            self.flush_timer.cancel()
        
        stats = self.get_stats()
        logger.info(f"📊 WandB步数管理器统计: {stats}")

# 全局管理器实例
_global_step_manager: Optional[WandBStepManager] = None

def get_step_manager() -> WandBStepManager:
    """获取全局步数管理器"""
    global _global_step_manager
    if _global_step_manager is None:
        _global_step_manager = WandBStepManager()
    return _global_step_manager

def safe_wandb_log(data: Dict[str, Any], 
                   step: Optional[int] = None,
                   source: str = "unknown",
                   priority: int = 5,
                   **kwargs) -> bool:
    """全局安全的WandB日志记录函数
    
    这个函数应该替代所有直接的 wandb.log 调用
    """
    manager = get_step_manager()
    return manager.safe_log(
        data=data,
        step=step,
        source=source,
        priority=priority,
        **kwargs
    )

def update_training_step(step: int, source: str = "trainer"):
    """更新训练步数"""
    manager = get_step_manager()
    manager.update_step(step, source)

def finalize_wandb_logging():
    """训练结束时清理WandB日志"""
    global _global_step_manager
    if _global_step_manager:
        _global_step_manager.finalize()
        _global_step_manager = None

# 修补函数 - 自动替换现有的wandb.log调用
def patch_wandb_log():
    """修补wandb.log函数，自动使用步数管理器"""
    if not hasattr(wandb, '_original_log'):
        wandb._original_log = wandb.log
        
        def patched_log(data, step=None, commit=True, **kwargs):
            # 如果使用了步数管理器，则使用安全记录
            if _global_step_manager is not None:
                return safe_wandb_log(
                    data=data,
                    step=step,
                    source="patched_wandb",
                    priority=10,  # 默认较低优先级
                    commit=commit
                )
            else:
                # 否则使用原始函数
                return wandb._original_log(data, step=step, commit=commit, **kwargs)
        
        wandb.log = patched_log
        logger.info("🔧 已修补 wandb.log 函数")

def unpatch_wandb_log():
    """取消wandb.log修补"""
    if hasattr(wandb, '_original_log'):
        wandb.log = wandb._original_log
        delattr(wandb, '_original_log')
        logger.info("🔧 已取消 wandb.log 修补")

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试WandB步数管理器...")
    
    # 模拟初始化WandB
    import os
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project="step-manager-test")
    
    # 测试步数管理器
    manager = get_step_manager()
    
    # 模拟正常训练日志
    for i in range(10):
        update_training_step(i, "trainer")
        manager.safe_log({"loss": 1.0 - i * 0.1}, source="trainer", priority=0)
        
        # 模拟回调日志
        if i % 3 == 0:
            manager.safe_log({"eval_metric": i * 0.05}, source="callback", priority=1)
    
    # 强制刷新
    manager._force_flush()
    time.sleep(1)
    
    # 输出统计
    print("📊 测试统计:", manager.get_stats())
    
    # 清理
    finalize_wandb_logging()
    wandb.finish()
    print("✅ 测试完成") 