# GRPO 改进训练系统指南

## 📋 概述

本指南介绍如何使用改进的GRPO训练系统，该系统解决了以下关键问题：

1. **断续训练时的WandB步数同步问题**
2. **推理评估数据缺失问题**
3. **课程学习无法推进问题**
4. **测试数据生成和记录问题**

## 🔧 核心改进组件

### 1. WandB同步管理器 (`wandb_sync_manager.py`)
- 解决断续训练时的步数不一致问题
- 自动检测恢复训练并创建新的WandB run
- 提供安全的日志记录和本地备份功能

### 2. 增强推理回调 (`enhanced_inference_callback.py`)
- 确保`eval_avg_test_pass_rate`指标正确生成
- 提供模拟数据以确保评估能够进行
- 增强的触发条件，避免重复评估

### 3. 改进训练脚本 (`improved_training_script.py`)
- 集成所有改进组件
- 提供完整的命令行参数支持
- 自动处理断续训练场景

## 🚀 快速开始

### 1. 基本使用

```bash
# 新训练
python improved_training_script.py \
    --output_dir "./enhanced_grpo_output" \
    --model_name "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset_path "./data/train_dataset.json" \
    --eval_every_n_steps 25 \
    --wandb_project "grpo-training-improved"
```

### 2. 断续训练

```bash
# 从检查点恢复（自动创建新WandB run）
python improved_training_script.py \
    --output_dir "./enhanced_grpo_output" \
    --resume_from_checkpoint "./enhanced_grpo_output/checkpoint-100" \
    --eval_every_n_steps 25 \
    --wandb_project "grpo-training-improved"
```

### 3. 集成到现有训练脚本

如果你想将改进功能集成到现有的训练脚本中：

```python
# 1. 初始化WandB同步管理器
from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager

sync_manager = initialize_wandb_sync_manager(
    output_dir="./output",
    project_name="your-project",
    run_name="your-run"
)

# 2. 添加增强推理回调
from grpo_project.callbacks.enhanced_inference_callback import EnhancedInferenceCallback

inference_callback = EnhancedInferenceCallback(
    eval_every_n_steps=25,
    max_samples=8
)

# 3. 在训练器中使用
trainer = YourTrainer(
    # ... 其他参数
    callbacks=[inference_callback]
)
```

## 📊 监控和调试

### 1. WandB监控

改进系统会在WandB中记录以下关键指标：

- `eval_avg_test_pass_rate`: 平均测试通过率（课程学习关键指标）
- `curriculum/latest_performance`: 最新性能评估
- `curriculum/evaluation_count`: 评估次数
- `inference/step`: 推理步数
- `_trainer_step`: 训练器实际步数（用于调试）

### 2. 本地日志

当WandB记录失败时，系统会自动备份到本地：
- `wandb_backup/step_XXXXXX.json`: 按步数保存的备份文件
- `wandb_sync_state.json`: WandB同步状态文件

### 3. 调试信息

系统提供详细的调试日志：

```python
# 检查同步状态
from grpo_project.core.wandb_sync_manager import get_wandb_sync_manager

sync_manager = get_wandb_sync_manager()
if sync_manager:
    status = sync_manager.get_sync_status()
    print(f"同步状态: {status}")

# 检查推理回调状态
inference_callback = trainer.callback_handler.callbacks[0]  # 假设是第一个
summary = inference_callback.get_evaluation_summary()
print(f"评估摘要: {summary}")
```

## 🔧 配置选项

### 训练脚本参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--eval_every_n_steps` | 25 | 评估间隔步数 |
| `--max_eval_samples` | 8 | 最大评估样本数 |
| `--force_new_wandb_run` | False | 强制创建新的WandB run |
| `--enable_curriculum` | True | 启用课程学习 |

### 环境变量

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="your-project-name"
export WANDB_DIR="./wandb_logs"  # WandB日志目录
```

## 🔧 故障排除

### 1. WandB步数不同步

**症状**: WandB界面显示的步数与训练日志中的步数不一致

**解决方案**:
```bash
# 选项1: 强制创建新的WandB run
python improved_training_script.py --force_new_wandb_run

# 选项2: 手动设置新的run ID
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
```

### 2. 评估数据缺失

**症状**: 没有看到`eval_avg_test_pass_rate`指标

**解决方案**:
1. 检查推理回调是否正确添加
2. 确认评估间隔设置正确
3. 查看日志中的评估触发信息

```python
# 调试推理回调
logger.setLevel(logging.DEBUG)  # 启用调试日志
```

### 3. 课程学习无法推进

**症状**: 一直停留在foundation阶段

**原因**: 缺少`eval_avg_test_pass_rate`指标或性能未达到阈值

**解决方案**:
1. 确保推理评估正常工作
2. 检查课程推进阈值设置
3. 降低推进阈值进行测试

### 4. 内存或性能问题

**解决方案**:
```bash
# 减少评估样本数
--max_eval_samples 4

# 增加评估间隔
--eval_every_n_steps 50

# 禁用某些功能
--enable_curriculum false
```

## 📈 最佳实践

### 1. 训练策略

1. **新训练**: 使用改进的训练脚本从头开始
2. **断续训练**: 让系统自动创建新的WandB run
3. **监控**: 重点关注`eval_avg_test_pass_rate`指标
4. **调优**: 根据评估结果调整课程推进阈值

### 2. 资源管理

```bash
# 设置合理的保存间隔
--save_steps 50

# 限制检查点数量
--save_total_limit 3

# 使用合适的批次大小
--per_device_train_batch_size 1
```

### 3. 监控设置

```bash
# 设置合理的评估频率
--eval_every_n_steps 25  # 不要太频繁，避免影响训练速度

# 合理的样本数量
--max_eval_samples 8  # 平衡评估准确性和速度
```

## 🔄 从旧系统迁移

如果你正在使用旧的GRPO训练系统：

### 1. 备份现有数据
```bash
cp -r ./old_output ./old_output_backup
cp ./training_logs.txt ./training_logs_backup.txt
```

### 2. 集成新组件
```python
# 在现有训练脚本中添加
from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager, safe_wandb_log
from grpo_project.callbacks.enhanced_inference_callback import EnhancedInferenceCallback

# 替换原有的WandB记录
# 旧: wandb.log(data, step=step)
# 新: safe_wandb_log(data, step)
```

### 3. 验证改进效果
- 检查WandB步数是否同步
- 确认`eval_avg_test_pass_rate`是否正常生成
- 观察课程学习是否能正常推进

## 📞 技术支持

如果遇到问题，请：

1. 检查日志文件中的错误信息
2. 确认所有依赖包版本正确
3. 查看WandB同步状态文件
4. 提供完整的错误日志和配置信息

### 常用调试命令

```bash
# 检查WandB状态
cat ./output/wandb_sync_state.json

# 查看最近的备份文件
ls -la ./output/wandb_backup/

# 运行测试脚本
python test_wandb_sync.py
```

---

**注意**: 这个改进系统是为了解决断续训练和步数同步问题而设计的。在使用时，请确保理解各个组件的作用，并根据实际需求进行调整。 