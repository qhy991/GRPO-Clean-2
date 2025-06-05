import os
import logging
import json
from datetime import datetime
from typing import Optional, Any, Dict # Added for type hints

logger = logging.getLogger(__name__)

class PeriodicStatusReporter:
    def __init__(self, output_dir, report_interval=100):
        self.output_dir = output_dir
        self.report_interval = report_interval
        # Ensure output_dir exists for status_log_path
        os.makedirs(self.output_dir, exist_ok=True)
        self.status_log_path = os.path.join(output_dir, "training_status.txt")

    def report_status(self, step: int, trainer_state: Any,
                      curriculum_manager: Optional[Any] = None,
                      experience_buffer: Optional[Any] = None):
        """生成定期状态报告"""
        if step % self.report_interval != 0:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status_report = f"""
========================================
训练状态报告 - 步数 {step}
时间: {timestamp}
========================================

📈 训练进度:
  - 当前步数: {step}
  - 最大步数: {trainer_state.max_steps if hasattr(trainer_state, 'max_steps') and trainer_state.max_steps > 0 else '无限制'}
  - 完成百分比: {(step/trainer_state.max_steps*100) if hasattr(trainer_state, 'max_steps') and trainer_state.max_steps > 0 else 'N/A'}%

📚 课程学习状态:"""

        if curriculum_manager and hasattr(curriculum_manager, 'current_stage') and hasattr(curriculum_manager, 'curriculum_stages'):
            current_stage = curriculum_manager.current_stage
            total_stages = len(curriculum_manager.curriculum_stages)
            stage_name = "Unknown/Final"
            if current_stage < total_stages and hasattr(curriculum_manager.curriculum_stages[current_stage], 'name'):
                 stage_name = curriculum_manager.curriculum_stages[current_stage].name

            dataset_size = 0
            if hasattr(curriculum_manager, 'get_current_stage_dataset'):
                current_ds = curriculum_manager.get_current_stage_dataset()
                if current_ds is not None:
                    dataset_size = len(current_ds)

            status_report += f"""
  - 当前阶段: {current_stage}/{total_stages-1 if total_stages > 0 else 0}
  - 阶段名称: {stage_name}
  - 阶段进度: {len(getattr(curriculum_manager, 'stage_performance_history', []))}次评估
  - 数据集大小: {dataset_size}"""
        else:
            status_report += "\n  - 课程学习: 未启用或信息不完整"

        status_report += f"""

🔄 经验回放状态:"""

        if experience_buffer and hasattr(experience_buffer, 'get_stats') and hasattr(experience_buffer, 'max_size'):
            buffer_stats = experience_buffer.get_stats()
            status_report += f"""
  - 缓存大小: {buffer_stats.get('size', 'N/A')}/{experience_buffer.max_size}
  - 平均奖励: {buffer_stats.get('mean_reward', 0.0):.2f}
  - 最高奖励: {buffer_stats.get('max_reward', 0.0):.2f}"""
        else:
            status_report += "\n  - 经验回放: 未启用或信息不完整"

        # 最近的损失信息
        if hasattr(trainer_state, 'log_history') and trainer_state.log_history:
            recent_log = trainer_state.log_history[-1]
            recent_loss = recent_log.get('loss', 'N/A')
            learning_rate = recent_log.get('learning_rate', 'N/A')
            status_report += f"""

📊 最近指标:
  - 训练损失: {recent_loss}
  - 学习率: {learning_rate}"""
        else:
            status_report += f"""

📊 最近指标:
  - 训练损失: N/A
  - 学习率: N/A"""

        status_report += f"""

========================================
"""

        # 输出到控制台
        logger.info(status_report)

        # 保存到文件
        try:
            with open(self.status_log_path, 'a', encoding='utf-8') as f:
                f.write(status_report + "\n")
        except Exception as e:
            logger.error(f"Failed to write status report to {self.status_log_path}: {e}")

def debug_checkpoint_contents(checkpoint_path: str):
    """调试checkpoint内容"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint路径不存在: {checkpoint_path}")
        return

    logger.info(f"🔍 检查checkpoint内容: {checkpoint_path}")

    # 列出所有文件
    try:
        files = os.listdir(checkpoint_path)
        logger.info(f"Checkpoint文件列表: {files}")

        # 检查关键文件
        key_files: Dict[str, str] = {
            'config.json': '模型配置',
            'pytorch_model.bin': 'PyTorch模型权重',
            'model.safetensors': 'SafeTensors模型权重',
            'adapter_config.json': 'PEFT适配器配置',
            'adapter_model.bin': 'PEFT适配器权重(bin)',
            'adapter_model.safetensors': 'PEFT适配器权重(safetensors)',
            'training_args.bin': '训练参数',
            'trainer_state.json': '训练状态',
            'optimizer.pt': '优化器状态',
            'scheduler.pt': '调度器状态'
        }

        logger.info("📁 关键文件检查:")
        for filename, description in key_files.items():
            exists = filename in files
            status = "✅" if exists else "❌"
            logger.info(f"   {status} {filename}: {description}")

    except Exception as e:
        logger.error(f"无法读取checkpoint目录: {e}")

# --- Performance Monitoring Utilities (Moved from root utils.py) ---
import numpy as np # For AdvancedPerformanceMonitor
# datetime is already imported at the top of the file

class AdvancedPerformanceMonitor:
    """高级性能监控器"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.performance_history: List[Dict[str, Any]] = [] # Type hint for clarity
        self.stage_history: List[Any] = [] # Type hint for clarity
        if output_dir and not os.path.exists(output_dir): # Ensure output_dir exists if provided
            os.makedirs(output_dir, exist_ok=True)

    def log_step_performance(self, step: int, loss: float, rewards: List[float],
                           stage: Optional[int] = None, additional_metrics: Optional[Dict[str, Any]] = None): # Type hints
        """记录单步性能"""

        performance_data: Dict[str, Any] = { # Type hint
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'avg_reward': float(np.mean(rewards)) if rewards else 0.0, # Ensure float
            'reward_std': float(np.std(rewards)) if rewards else 0.0,  # Ensure float
            'min_reward': float(np.min(rewards)) if rewards else 0.0,  # Ensure float
            'max_reward': float(np.max(rewards)) if rewards else 0.0,  # Ensure float
            'batch_size': len(rewards) if rewards else 0,
            'stage': stage
        }

        if additional_metrics:
            performance_data.update(additional_metrics)

        self.performance_history.append(performance_data)

        # 每10步输出摘要
        if step % 10 == 0: # Make this configurable?
            self._log_performance_summary(performance_data)

    def _log_performance_summary(self, current_data: Dict[str, Any]): # Type hint
        """输出性能摘要"""
        step = current_data['step']

        logger.info(f"""
        📊 步数 {step} 性能摘要:
        ├─ 损失: {current_data['loss']:.4f}
        ├─ 平均奖励: {current_data['avg_reward']:.4f}
        ├─ 奖励标准差: {current_data['reward_std']:.4f}
        ├─ 奖励范围: [{current_data['min_reward']:.4f}, {current_data['max_reward']:.4f}]
        ├─ 批次大小: {current_data['batch_size']}
        └─ 当前阶段: {current_data.get('stage', 'Unknown')}
        """)

    def analyze_recent_performance(self, window_size: int = 20) -> Dict[str, float]:
        """分析最近的性能趋势"""
        if len(self.performance_history) < window_size:
            return {} # Return empty if not enough data

        recent_data = self.performance_history[-window_size:]

        losses = [float(d['loss']) for d in recent_data] # Ensure float
        avg_rewards = [float(d['avg_reward']) for d in recent_data] # Ensure float

        # Calculate trend if there are enough points for polyfit (at least 2)
        loss_trend = 0.0
        reward_trend = 0.0
        if len(losses) >= 2:
            loss_trend = float(np.polyfit(range(len(losses)), losses, 1)[0])
        if len(avg_rewards) >= 2:
            reward_trend = float(np.polyfit(range(len(avg_rewards)), avg_rewards, 1)[0])

        analysis: Dict[str, float] = { # Type hint
            'avg_loss': float(np.mean(losses)) if losses else 0.0,
            'loss_trend': loss_trend,
            'avg_reward': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            'reward_trend': reward_trend,
            'loss_stability': float(np.std(losses)) if losses else 0.0,
            'reward_stability': float(np.std(avg_rewards)) if avg_rewards else 0.0
        }

        return analysis

def create_performance_monitor(output_dir: Optional[str] = None) -> AdvancedPerformanceMonitor: # Type hint
    """创建性能监控器的工厂函数"""
    return AdvancedPerformanceMonitor(output_dir)

def monitor_advanced_stage_training(curriculum_manager: Any, performance_history: List[Dict[str, Any]]): # Type hints
    """监控高难度阶段的训练状况"""

    # Assuming curriculum_manager has 'current_stage' attribute
    if not hasattr(curriculum_manager, 'current_stage') or curriculum_manager.current_stage != 3:  # 只监控最高难度阶段 (example stage)
        return

    recent_performance = performance_history[-20:] if len(performance_history) >= 20 else performance_history

    if not recent_performance:
        return

    # Assuming 'performance' key exists in performance_history dicts
    performances = [float(p.get('performance', 0.0)) for p in recent_performance] # Ensure float and provide default

    avg_performance = float(np.mean(performances)) if performances else 0.0
    performance_trend = 0.0
    if len(performances) >= 2:
        performance_trend = float(np.polyfit(range(len(performances)), performances, 1)[0])  # 线性趋势

    logger.info(f"""
    🎯 高难度阶段性能分析:
    - 平均性能: {avg_performance:.4f}
    - 性能趋势: {'上升' if performance_trend > 0.0001 else ('下降' if performance_trend < -0.0001 else '平稳')} ({performance_trend:.6f}/步)
    - 最近性能: {performances[-1]:.4f} if performances else N/A
    - 性能波动: {float(np.std(performances)):.4f} if performances else 0.0
    """)

    # 给出建议 (example thresholds)
    if avg_performance > 0.95 and performance_trend > 0:
        logger.info("💡 建议: 模型在高难度阶段表现优秀，可以考虑增加数据集难度或延长训练")
    elif avg_performance < 0.85:
        logger.info("💡 建议: 高难度阶段性能偏低，可能需要调整学习率或增加训练步数")
    elif performances and float(np.std(performances)) > 0.1:
        logger.info("💡 建议: 性能波动较大，建议降低学习率或增加batch size")
