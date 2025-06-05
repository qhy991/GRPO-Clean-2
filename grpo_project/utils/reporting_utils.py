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
