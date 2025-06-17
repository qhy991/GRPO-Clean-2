# 核心训练模块分析

## 🧠 ModelManager (`models.py`)
- **功能**：统一管理模型加载、量化、PEFT适配器应用
- **特色**：
  - 支持BitsAndBytes量化
  - 智能PEFT适配器恢复（Stage1 → GRPO）
  - Qwen3兼容性修复
  - k-bit训练准备

```python
# 关键流程
setup_model_and_tokenizer()
├── 加载基础模型（支持量化）
├── 设置tokenizer（pad token处理）
├── 应用Qwen3兼容性修复
└── 准备k-bit训练环境

apply_peft_adapter()
├── 恢复checkpoint（如果存在）
├── 加载Stage1适配器（如果提供）
├── 创建新PEFT适配器（如果需要）
└── 验证可训练参数
```

## 🔄 TrainingOrchestrator (`trainer.py`)
- **功能**：统一编排整个训练流程
- **特色**：
  - 组件化设计，易于扩展
  - 完善的错误恢复机制
  - 分布式训练支持

```python
# 核心流程
setup_all_components()
├── _setup_model_and_tokenizer_with_manager()
├── _setup_dataset_and_dependencies()
├── _setup_callbacks()
└── _initialize_trainer()

run_training()
├── 确定训练数据集（课程学习 vs 全量）
├── 创建奖励函数闭包
├── 启动GRPOTrainer
└── 保存训练产物
```

## 🔐 错误恢复系统 (`error_recovery.py`)
- **功能**：智能错误分类和自动恢复
- **支持错误类型**：
  - CUDA内存错误 → 清理缓存
  - Checkpoint损坏 → 回退到历史版本
  - 解析错误 → 保存问题样本
  - 生成错误 → 调整参数

## 📊 WandB同步管理器 (`wandb_sync_manager.py`)
- **功能**：解决断续训练时的步数同步问题
- **关键特性**：
  - 自动检测步数偏移
  - 本地备份机制
  - 智能run ID恢复