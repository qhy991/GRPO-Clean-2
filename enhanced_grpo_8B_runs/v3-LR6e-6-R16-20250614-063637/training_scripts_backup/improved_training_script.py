#!/usr/bin/env python3
"""
改进的GRPO训练脚本
解决断续训练时的步数同步和测试数据生成问题
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入GRPO组件
from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager, get_wandb_sync_manager
from grpo_project.callbacks.enhanced_inference_callback import EnhancedInferenceCallback
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_improved_training():
    """
    设置改进的训练配置
    """
    parser = argparse.ArgumentParser(description="改进的GRPO训练脚本")
    
    # 基本参数
    parser.add_argument("--output_dir", type=str, default="./output", 
                       help="输出目录")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct",
                       help="模型名称")
    parser.add_argument("--dataset_path", type=str, default="./data/train_dataset.json",
                       help="数据集路径")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="每设备训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="保存间隔步数")
    
    # 评估参数
    parser.add_argument("--eval_every_n_steps", type=int, default=25,
                       help="评估间隔步数")
    parser.add_argument("--max_eval_samples", type=int, default=8,
                       help="最大评估样本数")
    
    # 恢复训练参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点恢复训练")
    parser.add_argument("--force_new_wandb_run", action="store_true",
                       help="强制创建新的WandB run")
    
    # WandB参数
    parser.add_argument("--wandb_project", type=str, default="grpo-training-improved",
                       help="WandB项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB运行名称")
    
    # 课程学习参数
    parser.add_argument("--enable_curriculum", action="store_true", default=True,
                       help="启用课程学习")
    parser.add_argument("--curriculum_config", type=str, default=None,
                       help="课程学习配置文件")
    
    return parser.parse_args()

def create_training_config(args):
    """
    创建训练配置
    """
    from transformers import TrainingArguments
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否是断续训练
    is_resuming = args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=10,
        evaluation_strategy="steps" if not is_resuming else "no",  # 断续训练时暂时禁用内置评估
        eval_steps=args.eval_every_n_steps if not is_resuming else None,
        save_total_limit=3,
        load_best_model_at_end=False,  # 避免与课程学习冲突
        metric_for_best_model="eval_avg_test_pass_rate",
        greater_is_better=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # 断续训练相关
        resume_from_checkpoint=args.resume_from_checkpoint if is_resuming else None,
        # 日志相关
        report_to=["wandb"] if not args.force_new_wandb_run else [],  # 控制WandB初始化
        run_name=args.wandb_run_name,
    )
    
    return training_args, is_resuming

def setup_wandb_integration(args, is_resuming: bool):
    """
    设置WandB集成
    """
    logger.info("🔄 设置WandB集成...")
    
    # 初始化WandB同步管理器
    sync_manager = initialize_wandb_sync_manager(
        output_dir=args.output_dir,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name or f"grpo_run_{Path(args.output_dir).name}"
    )
    
    # 设置WandB运行
    config = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size,
        "eval_every_n_steps": args.eval_every_n_steps,
        "max_eval_samples": args.max_eval_samples,
        "enable_curriculum": args.enable_curriculum,
        "is_resuming": is_resuming,
    }
    
    success = sync_manager.setup_wandb_run(
        resume_from_checkpoint=args.resume_from_checkpoint if is_resuming else None,
        config=config
    )
    
    if not success:
        logger.warning("⚠️ WandB初始化失败，将使用本地日志")
    
    return sync_manager

def setup_enhanced_callbacks(args):
    """
    设置增强的回调
    """
    logger.info("📞 设置增强回调...")
    
    callbacks = []
    
    # 1. 增强推理回调
    inference_callback = EnhancedInferenceCallback(
        eval_every_n_steps=args.eval_every_n_steps,
        max_samples=args.max_eval_samples
    )
    callbacks.append(inference_callback)
    
    # 2. 其他必要回调可以在这里添加
    # 例如：课程学习回调、保存回调等
    
    logger.info(f"✅ 设置完成，共 {len(callbacks)} 个回调")
    return callbacks

def load_dataset(args):
    """
    加载数据集
    """
    logger.info(f"📊 加载数据集: {args.dataset_path}")
    
    try:
        # 这里应该加载实际的数据集
        # 暂时返回空，实际使用时需要实现
        logger.warning("⚠️ 数据集加载功能需要实现")
        return None, None  # train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"❌ 数据集加载失败: {e}")
        return None, None

def setup_curriculum_learning(args, dataset):
    """
    设置课程学习
    """
    if not args.enable_curriculum:
        logger.info("📚 课程学习已禁用")
        return None
    
    logger.info("📚 设置课程学习...")
    
    try:
        # 这里应该设置实际的课程学习管理器
        # curriculum_manager = setup_fixed_curriculum_manager(script_cfg, dataset)
        logger.warning("⚠️ 课程学习设置功能需要实现")
        return None
        
    except Exception as e:
        logger.error(f"❌ 课程学习设置失败: {e}")
        return None

def create_trainer(args, training_args, model, train_dataset, eval_dataset, callbacks):
    """
    创建训练器
    """
    logger.info("🚀 创建训练器...")
    
    try:
        # 这里应该创建实际的GRPO训练器
        # 暂时返回None，实际使用时需要实现
        logger.warning("⚠️ 训练器创建功能需要实现")
        return None
        
    except Exception as e:
        logger.error(f"❌ 训练器创建失败: {e}")
        return None

def main():
    """
    主函数
    """
    logger.info("🚀 启动改进的GRPO训练")
    
    # 1. 解析参数
    args = setup_improved_training()
    logger.info(f"📋 训练参数: {vars(args)}")
    
    # 2. 创建训练配置
    training_args, is_resuming = create_training_config(args)
    logger.info(f"⚙️ 训练配置完成, 断续训练: {is_resuming}")
    
    # 3. 设置WandB集成
    sync_manager = setup_wandb_integration(args, is_resuming)
    
    # 4. 加载数据集
    train_dataset, eval_dataset = load_dataset(args)
    
    # 5. 设置课程学习
    curriculum_manager = setup_curriculum_learning(args, train_dataset)
    
    # 6. 设置回调
    callbacks = setup_enhanced_callbacks(args)
    
    # 7. 加载模型（需要实现）
    model = None  # 实际使用时需要加载模型
    
    # 8. 创建训练器
    trainer = create_trainer(args, training_args, model, train_dataset, eval_dataset, callbacks)
    
    if trainer is None:
        logger.error("❌ 训练器创建失败，无法继续训练")
        return
    
    # 9. 开始训练
    logger.info("🎯 开始训练...")
    try:
        # trainer.train()
        logger.info("⚠️ 训练逻辑需要实现")
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        return
    
    # 10. 训练完成
    logger.info("✅ 训练完成")
    
    # 11. 保存最终状态
    if sync_manager:
        status = sync_manager.get_sync_status()
        logger.info(f"📊 最终同步状态: {status}")

def create_training_script_template():
    """
    创建训练脚本模板
    """
    script_content = '''#!/bin/bash
# 改进的GRPO训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="grpo-training-improved"

# 基本参数
OUTPUT_DIR="./enhanced_grpo_output"
MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-instruct"
DATASET_PATH="./data/train_dataset.json"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=5e-5
SAVE_STEPS=50

# 评估参数
EVAL_EVERY_N_STEPS=25
MAX_EVAL_SAMPLES=8

# 恢复训练参数（如果需要）
# RESUME_FROM_CHECKPOINT="./enhanced_grpo_output/checkpoint-xxx"

# 运行训练
python improved_training_script.py \\
    --output_dir "$OUTPUT_DIR" \\
    --model_name "$MODEL_NAME" \\
    --dataset_path "$DATASET_PATH" \\
    --num_train_epochs "$NUM_EPOCHS" \\
    --per_device_train_batch_size "$BATCH_SIZE" \\
    --learning_rate "$LEARNING_RATE" \\
    --save_steps "$SAVE_STEPS" \\
    --eval_every_n_steps "$EVAL_EVERY_N_STEPS" \\
    --max_eval_samples "$MAX_EVAL_SAMPLES" \\
    --enable_curriculum \\
    --wandb_project "$WANDB_PROJECT" \\
    --wandb_run_name "enhanced_grpo_$(date +%Y%m%d_%H%M%S)"
    # --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"  # 断续训练时取消注释

echo "✅ 训练脚本执行完成"
'''
    
    with open("run_improved_training.sh", "w") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("run_improved_training.sh", 0o755)
    
    logger.info("📝 已创建改进的训练脚本: run_improved_training.sh")

if __name__ == "__main__":
    # 创建训练脚本模板
    create_training_script_template()
    
    # 运行主程序
    main() 