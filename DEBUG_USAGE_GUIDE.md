# 🐛 DEBUG模式训练使用指南

## 📋 概述

这个增强的训练脚本提供了全面的DEBUG功能，可以保存训练过程中的所有数据，帮助您详细分析模型的训练状态和性能。

## 🚀 快速开始

### 1. 启动训练（DEBUG模式）
```bash
cd /home/qhy/Research/LLM/GRPO-Clean-2
./run_model_parallel_only.sh
```

### 2. 实时监控训练状态
在另一个终端中运行：
```bash
cd /home/qhy/Research/LLM/GRPO-Clean-2
./monitor_training.sh
```

### 3. 分析DEBUG数据
训练完成后或训练过程中：
```bash
cd /home/qhy/Research/LLM/GRPO-Clean-2
python analyze_debug_data.py
```

## 📁 DEBUG数据目录结构

训练启动后，会在 `./model_parallel_only_outputs/debug_data/` 下创建以下目录结构：

```
debug_data/
├── generations/
│   └── 20241216-213045/          # 时间戳目录
│       ├── step_001_generations.json
│       ├── step_005_generations.json
│       └── ...
├── failed_generations/
│   └── 20241216-213045/
│       ├── step_001_failed.json
│       └── ...
├── successful_generations/
│   └── 20241216-213045/
│       ├── step_001_success.json
│       └── ...
├── detailed_metrics/
│   └── 20241216-213045/
│       ├── step_001_metrics.json
│       └── ...
├── model_outputs/
│   └── 20241216-213045/
│       ├── step_001_outputs.json
│       └── ...
├── reward_details/
│   └── 20241216-213045/
│       ├── step_001_rewards.json
│       └── ...
└── training_logs/
    └── 20241216-213045/
        ├── full_training_log.txt      # 完整训练日志
        ├── error_log.txt              # 错误日志
        ├── gpu_monitor.log            # GPU监控日志
        ├── final_gpu_status.csv       # 最终GPU状态
        └── training_summary.txt       # 训练摘要
```

## 🔧 DEBUG配置参数

### 主要DEBUG参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEBUG_MODE` | `true` | 启用详细debug模式 |
| `SAVE_ALL_GENERATIONS` | `true` | 保存所有生成的样本 |
| `SAVE_FAILED_GENERATIONS` | `true` | 保存失败的生成样本 |
| `SAVE_SUCCESSFUL_GENERATIONS` | `true` | 保存成功的生成样本 |
| `SAVE_DETAILED_METRICS` | `true` | 保存详细的训练指标 |
| `SAVE_MODEL_OUTPUTS` | `true` | 保存模型的原始输出 |
| `SAVE_REWARD_DETAILS` | `true` | 保存奖励计算的详细信息 |
| `DEBUG_SAMPLE_FREQUENCY` | `5` | 每5步保存一次详细样本 |
| `LOGGING_STEPS` | `1` | 每步都记录日志 |

### WandB增强配置
- **项目标签**: `debug,model_parallel,lora,grpo,verilog`
- **运行名称**: `DEBUG-model-parallel-LR2e-5-R64-BS2x8-20241216-213045`
- **保存代码**: ✅ 自动保存训练代码
- **保存模型**: ✅ 自动保存模型检查点
- **监控参数**: ✅ 监控所有模型参数

## 📊 实时监控功能

### 启动监控
```bash
./monitor_training.sh
```

### 监控选项
```bash
./monitor_training.sh --debug_dir /path/to/debug/data --interval 30
```

### 监控内容
- ✅ **训练进程状态**: 检查训练是否正在运行
- 📊 **GPU使用情况**: 内存、利用率、温度
- 🐛 **DEBUG数据统计**: 各类文件数量和最新文件时间
- 📄 **实时日志**: 显示最新的训练日志
- 💾 **磁盘使用**: 监控存储空间使用情况

## 🔍 数据分析功能

### 基本分析
```bash
python analyze_debug_data.py
```

### 指定目录分析
```bash
python analyze_debug_data.py --debug_dir /path/to/debug/data
```

### 分析内容
- 📊 **样本统计**: 总生成数、成功数、失败数
- 📋 **失败原因**: 详细的失败原因分类统计
- 🏆 **奖励分析**: 平均分、最高分、最低分
- 📈 **趋势分析**: 性能变化趋势
- 📄 **报告生成**: 自动生成分析报告

## 📝 数据文件格式

### 生成样本文件 (`generations/*.json`)
```json
{
  "step": 123,
  "timestamp": "2024-12-16T21:30:45",
  "prompt": "设计一个...",
  "generated_text": "module example...",
  "generation_time": 2.34,
  "token_count": 456,
  "raw_output": "..."
}
```

### 失败样本文件 (`failed_generations/*.json`)
```json
{
  "step": 123,
  "timestamp": "2024-12-16T21:30:45",
  "prompt": "设计一个...",
  "generated_text": "module example...",
  "failure_reason": "compilation_error",
  "error_details": "错误详情...",
  "compilation_output": "编译输出..."
}
```

### 成功样本文件 (`successful_generations/*.json`)
```json
{
  "step": 123,
  "timestamp": "2024-12-16T21:30:45",
  "prompt": "设计一个...",
  "generated_text": "module example...",
  "reward_score": 8.5,
  "test_results": {
    "passed": 7,
    "failed": 0,
    "total": 7
  },
  "quality_metrics": {
    "complexity": 0.8,
    "readability": 0.9
  }
}
```

## 🛠️ 故障排除

### 常见问题

#### 1. DEBUG目录创建失败
```bash
# 检查权限
ls -la ./model_parallel_only_outputs/
# 手动创建目录
mkdir -p ./model_parallel_only_outputs/debug_data
```

#### 2. 监控脚本无法运行
```bash
# 添加执行权限
chmod +x monitor_training.sh
chmod +x analyze_debug_data.py
```

#### 3. 磁盘空间不足
```bash
# 检查磁盘使用
df -h ./model_parallel_only_outputs/
# 清理旧的DEBUG数据
rm -rf ./model_parallel_only_outputs/debug_data/old_timestamp/
```

#### 4. GPU监控失败
```bash
# 检查nvidia-smi是否可用
nvidia-smi
# 检查GPU监控进程
ps aux | grep gpu_monitor
```

## 📈 性能优化建议

### 1. 根据DEBUG数据调整参数
- **失败率高**: 降低学习率，增加warmup
- **奖励分数低**: 检查奖励函数配置
- **生成质量差**: 调整生成参数（temperature, top_p）

### 2. 存储管理
- **定期清理**: 删除旧的DEBUG数据
- **压缩存储**: 使用gzip压缩JSON文件
- **选择性保存**: 只保存关键步骤的DEBUG数据

### 3. 监控优化
- **调整刷新间隔**: 根据需要调整监控频率
- **过滤日志**: 只显示关键信息
- **异步监控**: 避免影响训练性能

## 🎯 最佳实践

1. **训练开始前**
   - ✅ 确保有足够的磁盘空间（建议至少50GB）
   - ✅ 检查所有脚本的执行权限
   - ✅ 验证GPU状态正常

2. **训练过程中**
   - ✅ 定期查看监控界面
   - ✅ 检查DEBUG数据是否正常生成
   - ✅ 关注GPU利用率和内存使用

3. **训练结束后**
   - ✅ 运行数据分析脚本
   - ✅ 查看训练摘要报告
   - ✅ 备份重要的DEBUG数据

## 📞 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请：
1. 查看训练日志文件获取详细错误信息
2. 运行分析脚本查看数据统计
3. 检查GPU监控日志确认硬件状态

---

**注意**: DEBUG模式会显著增加磁盘使用量和训练时间，建议在分析问题时启用，正常训练时可以关闭部分DEBUG功能。 