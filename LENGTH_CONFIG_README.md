# GRPO训练长度配置指南

## 概述

现在您可以独立控制GRPO训练中的最大提示长度和最大输出长度，不再限制于原来的固定2048输出长度。

## 🔧 配置参数

### 主要参数

- `MAX_SEQ_LENGTH`: 总序列长度 (模型支持的最大长度)
- `MAX_PROMPT_LENGTH`: 输入提示的最大长度
- `MAX_COMPLETION_LENGTH`: 模型输出的最大长度
- `LENGTH_ALLOCATION_STRATEGY`: 长度分配策略

### 分配策略选项

1. **"custom"**: 使用自定义的 `MAX_PROMPT_LENGTH` 和 `MAX_COMPLETION_LENGTH`
2. **"balanced"**: 50/50 分配 (prompt和completion各占一半)
3. **"prompt_heavy"**: 60/40 分配 (更多空间给prompt)
4. **"completion_heavy"**: 40/60 分配 (更多空间给输出)

## 📝 使用方法

### 1. 直接在训练脚本中修改

编辑 `run_enhanced_grpo_training.sh` 文件：

```bash
# 🔧 独立配置prompt和completion长度
MAX_PROMPT_LENGTH=1024     # 提示的最大长度
MAX_COMPLETION_LENGTH=3072 # 输出的最大长度 (现在可以是3072!)
LENGTH_ALLOCATION_STRATEGY="custom"
```

### 2. 使用预设配置

查看配置示例：
```bash
./length_config_examples.sh
```

### 3. 验证配置

运行训练脚本时会自动验证配置：
- 检查 prompt + completion 不超过总序列长度
- 显示百分比分配
- 提供配置建议

## 📊 推荐配置

### 🎯 用于一般Verilog生成 (默认)
```bash
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=1536    # ~37.5%
MAX_COMPLETION_LENGTH=2560 # ~62.5%
```

### 🚀 用于长代码生成
```bash
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=1024    # ~25%
MAX_COMPLETION_LENGTH=3072 # ~75%
```

### 💪 用于超长代码生成
```bash
MAX_SEQ_LENGTH=6144
MAX_PROMPT_LENGTH=1536    # ~25%
MAX_COMPLETION_LENGTH=4608 # ~75%
```

## ⚡ 性能考量

### GPU内存需求 (粗略估算)

| 序列长度 | 单GPU内存 (batch_size=2) | 推荐GPU |
|---------|------------------------|---------|
| 4096    | ~16GB                  | RTX 4090, A100 |
| 6144    | ~24GB                  | A100 40GB |
| 8192    | ~32GB                  | A100 80GB, H100 |

### 训练速度影响

- 更长的序列 = 更慢的训练速度
- 建议根据实际需求选择合适的长度
- 可以从较短的配置开始，逐步增加

## 🔍 验证和调试

### 配置验证

训练开始时会显示：
```
📏 长度配置验证:
  - 总序列长度: 4096
  - 最大提示长度: 1024 (25%)
  - 最大输出长度: 3072 (75%)
  - 总使用长度: 4096
  - 分配策略: custom
✅ 长度配置有效
```

### 在训练日志中查看

训练日志会显示：
```
📏 长度配置:
  - 总序列长度: 4096
  - 最大提示长度: 1024 (25.0%)
  - 最大输出长度: 3072 (75.0%)
  - 分配策略: custom
  - GRPO max_length: 3072
```

## 💡 最佳实践

1. **从较小配置开始**: 先用4096总长度测试
2. **监控GPU内存**: 使用 `nvidia-smi` 检查内存使用
3. **根据任务调整**: 
   - 简单模块: 2560输出长度够用
   - 复杂设计: 3072-4608输出长度
4. **平衡质量和速度**: 更长不一定更好

## 🛠️ 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少 `MAX_SEQ_LENGTH`
   - 减少 `PER_DEVICE_TRAIN_BATCH_SIZE`
   - 启用梯度检查点

2. **配置验证失败**
   - 确保 `MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH <= MAX_SEQ_LENGTH`
   - 检查策略设置是否正确

3. **训练速度太慢**
   - 考虑使用较短的序列长度
   - 增加GPU数量并启用分布式训练

## 📚 更多帮助

- 查看配置示例: `./length_config_examples.sh`
- 查看训练日志了解实际配置
- 根据GPU内存调整批次大小 