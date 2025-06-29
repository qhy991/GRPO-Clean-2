# 数据集分层抽样功能

## 概述

本功能为GRPO训练添加了**分层抽样(Stratified Sampling)**支持，让你能够使用较小比例的数据进行训练，同时确保各个类别的数据都得到代表性采样，避免类别偏差问题。

## 功能特点

🎯 **均衡采样**: 确保各个类别的数据都有代表性
📊 **智能分布**: 自动分析数据集类别分布
🔧 **灵活配置**: 支持多种分层字段组合
⚡ **高效训练**: 用10%数据快速验证模型效果
🎲 **可重现**: 固定随机种子确保结果一致

## 配置参数

### 在训练脚本中的配置

```bash
# 🎯 分层抽样配置（使用10%数据但保持各类别均衡）
DATASET_SAMPLE_RATIO=0.1        # 采样比例：10%
STRATIFY_COLUMNS="level,category"  # 分层字段
MIN_SAMPLES_PER_CATEGORY=1      # 每个类别最少保留样本数
SAMPLING_RANDOM_SEED=42         # 采样随机种子
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_sample_ratio` | float | `None` | 采样比例，0.1表示10%。设为None则使用全部数据 |
| `stratify_columns` | str | `"level,category"` | 用于分层的字段，多个字段用逗号分隔 |
| `min_samples_per_category` | int | `1` | 每个类别至少保留的样本数 |
| `sampling_random_seed` | int | `42` | 随机种子，确保结果可重现 |

## 支持的分层字段

系统会自动检测数据集中的以下分类字段：

- **`level`**: 难度等级 (basic, intermediate, advanced)
- **`category`**: 类别标签
- **`difficulty`**: 困难程度
- **`complexity_score`**: 复杂度分数（数值，会自动分桶为simple/medium/complex）

## 使用方法

### 1. 启用分层抽样训练

直接运行训练脚本即可：

```bash
bash run_model_parallel_only.sh
```

脚本已经预配置了10%分层抽样。

### 2. 预览数据分布（可选）

在训练前，可以先预览数据集的分布和采样效果：

```bash
python dataset_sampling_preview.py \
  --dataset_path /home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl \
  --sample_ratio 0.1 \
  --stratify_columns level,category
```

### 3. 自定义配置

修改 `run_model_parallel_only.sh` 中的配置：

```bash
# 使用5%数据
DATASET_SAMPLE_RATIO=0.05

# 只按难度等级分层
STRATIFY_COLUMNS="level"

# 每个类别至少保留3个样本
MIN_SAMPLES_PER_CATEGORY=3
```

## 分层抽样原理

1. **类别分析**: 自动分析数据集中各个类别的分布情况
2. **比例分配**: 按原始比例为每个类别分配采样配额
3. **最小保证**: 确保每个类别至少有`min_samples_per_category`个样本
4. **随机采样**: 在每个类别内进行随机采样
5. **结果验证**: 输出详细的采样前后分布对比

## 日志输出示例

```
🎯 开始分层抽样: 原始数据1000条，目标采样比例10.0%
📊 使用分层字段: ['level', 'category']
📈 原始数据类别分布:
  level:basic|category:logic: 300 样本 (30.0%)
  level:intermediate|category:arithmetic: 400 样本 (40.0%)
  level:advanced|category:control: 300 样本 (30.0%)
🎯 目标总采样数: 100
🎲 执行分层采样:
  level:basic|category:logic: 采样 30/300 样本
  level:intermediate|category:arithmetic: 采样 40/400 样本
  level:advanced|category:control: 采样 30/300 样本
✅ 分层采样完成:
  原始数据: 1000 样本
  采样后: 100 样本
  实际采样比例: 10.0%
```

## 优势对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| **随机采样** | 简单快速 | 可能导致某些类别数据过少或缺失 |
| **分层抽样** | 保持类别分布均衡 | 略微复杂，但效果更好 |
| **全量数据** | 信息最完整 | 训练时间长，资源消耗大 |

## 适用场景

✅ **推荐使用分层抽样的场景**:
- 快速验证模型架构和超参数
- 数据集较大，希望加速实验迭代
- 类别分布不均衡的数据集
- 计算资源有限的情况

❌ **不推荐使用的场景**:
- 最终产品模型训练（建议使用全量数据）
- 数据集本身就很小（<1000条）
- 对训练时间不敏感的情况

## 注意事项

1. **数据质量**: 确保采样后的数据仍能代表完整数据集的特征
2. **类别数量**: 如果类别过多且每类样本很少，可能影响采样效果
3. **随机种子**: 固定种子确保实验可重现，但也要注意不同种子的效果
4. **最终训练**: 在模型效果验证后，建议使用全量数据进行最终训练

## 故障排除

### 问题1: 未找到分层字段
```
⚠️ 未找到合适的分层字段，将进行随机采样
```
**解决方案**: 检查数据集是否包含 `level`、`category` 等字段，或者手动指定存在的字段。

### 问题2: 某些类别样本过少
```
某个类别: 采样 1/2 样本
```
**解决方案**: 增加 `min_samples_per_category` 的值，或者调整 `sample_ratio`。

### 问题3: 采样结果不理想
**解决方案**: 尝试不同的 `stratify_columns` 组合，或者调整随机种子。

## 扩展功能

如需更复杂的采样策略，可以修改 `grpo_project/data/dataset.py` 中的 `stratified_sample_dataset` 函数。

---

📝 **更新记录**:
- 2024-XX-XX: 初始版本，支持基础分层抽样
- 功能完善中，欢迎反馈和建议 