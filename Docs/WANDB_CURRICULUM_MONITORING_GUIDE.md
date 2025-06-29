# WandB课程学习监控指标完整指南

## 📊 概述

通过WandB，您现在可以实时监控每个阶段的完整数据集使用情况。我们提供了40+个详细指标来全面跟踪训练进度。

## 🎯 核心数据集使用监控指标

### 📈 数据集覆盖率监控
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/full_dataset_size` | 完整数据集大小 | 50000 | 了解总数据量 |
| `curriculum/dataset_size` | 当前阶段数据集大小 | 8500 | 当前阶段可用数据 |
| `curriculum/stage_dataset_coverage_percent` | 当前阶段数据占总数据比例 | 17.0% | 阶段数据覆盖率 |
| `curriculum/cumulative_samples_trained` | 累积训练样本数 | 6800 | 已训练样本总数 |
| `curriculum/cumulative_coverage_percent` | 当前阶段累积覆盖率 | 80.0% | 阶段内数据使用率 |

### 🔄 Epoch完整性监控
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/epochs_completed` | 已完成epoch数 | 1.24 | 实际训练轮数 |
| `curriculum/training_progress_percent` | 训练进度百分比 | 124.0% | 完整训练进度 |
| `curriculum/epoch_requirement_met` | epoch要求是否满足 | 1.0 | 进阶条件检查 |
| `curriculum/estimated_steps_per_epoch` | 预估每epoch步数 | 8500 | 完整训练所需步数 |
| `curriculum/steps_completed` | 已完成步数 | 10540 | 当前阶段训练步数 |

### ⚡ 数据使用效率指标
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/stage_data_efficiency` | 阶段数据使用效率 | 1.24 | 数据重复使用率 |
| `curriculum/data_reuse_count` | 数据重复使用次数 | 1.24 | 等同于epochs_completed |
| `curriculum/samples_remaining` | 剩余待训练样本 | 0 | 距离完整epoch的差距 |
| `curriculum/epoch_completion_ratio` | epoch完成比例 | 1.0 | 归一化的epoch进度 |

## 🎯 进阶条件监控

### ✅ 三大进阶要求
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/can_advance` | 是否可以进阶 | 1.0 | 综合进阶判断 |
| `curriculum/performance_requirement_met` | 性能要求是否满足 | 1.0 | 性能条件检查 |
| `curriculum/evaluations_requirement_met` | 评估次数要求是否满足 | 1.0 | 评估条件检查 |
| `curriculum/full_training_requirement_met` | 完整训练要求是否满足 | 1.0 | **新增**完整训练检查 |

### 📊 要求完成度
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/performance_completion_ratio` | 性能要求完成度 | 1.04 | 超额完成情况 |
| `curriculum/evaluations_completion_ratio` | 评估要求完成度 | 1.60 | 评估充分程度 |
| `curriculum/full_training_completion_ratio` | 完整训练完成度 | 1.24 | 训练充分程度 |

## 📋 阶段配置信息

### 🏗️ 阶段基本信息
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/current_stage_idx` | 当前阶段索引 | 2 | 阶段位置 |
| `curriculum/stage_complexity_min` | 阶段最小复杂度 | 3.0 | 复杂度范围下限 |
| `curriculum/stage_complexity_max` | 阶段最大复杂度 | 7.0 | 复杂度范围上限 |
| `curriculum/stage_complexity_span` | 复杂度跨度 | 4.0 | 复杂度范围宽度 |
| `curriculum/stage_levels_count` | 阶段包含级别数 | 1 | 数据级别多样性 |

### ⚙️ 阶段训练要求
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/stage_require_full_epoch` | 是否要求完整epoch | 1.0 | 完整训练要求 |
| `curriculum/stage_min_steps_per_epoch` | 每epoch最小步数 | 30 | 最小训练量 |
| `curriculum/stage_min_evaluations` | 最小评估次数 | 5 | 评估要求 |
| `curriculum/performance_threshold` | 性能阈值 | 0.700 | 进阶标准 |

## 📈 性能趋势监控

### 🎯 性能指标
| 指标名称 | 说明 | 示例值 | 用途 |
|---------|------|--------|------|
| `curriculum/latest_performance` | 最新性能 | 0.756 | 当前表现 |
| `curriculum/avg_stage_performance` | 阶段平均性能 | 0.724 | 稳定性评估 |
| `curriculum/recent_3_avg_performance` | 最近3次平均性能 | 0.748 | 短期趋势 |
| `curriculum/performance_trend` | 性能变化趋势 | 0.032 | 改善/下降幅度 |
| `curriculum/performance_stability` | 性能稳定性 | 0.015 | 波动程度 |

## 🔍 实时监控建议

### 📊 关键仪表板配置

1. **数据使用概览**
   ```
   - curriculum/training_progress_percent (主要指标)
   - curriculum/cumulative_coverage_percent
   - curriculum/epochs_completed
   - curriculum/stage_dataset_coverage_percent
   ```

2. **进阶条件监控**
   ```
   - curriculum/can_advance (总体状态)
   - curriculum/performance_requirement_met
   - curriculum/evaluations_requirement_met  
   - curriculum/full_training_requirement_met (新增)
   ```

3. **效率分析**
   ```
   - curriculum/stage_data_efficiency
   - curriculum/estimated_remaining_steps
   - curriculum/estimated_remaining_training_steps
   ```

### 📈 图表建议

1. **训练进度图**
   - X轴：训练步数
   - Y轴：`curriculum/training_progress_percent`
   - 目标线：100%

2. **数据覆盖率图**
   - X轴：训练步数  
   - Y轴：`curriculum/cumulative_coverage_percent`
   - 分阶段显示

3. **进阶条件状态图**
   - 堆叠柱状图显示三个条件的满足情况
   - 颜色编码：绿色(满足)，红色(未满足)

## 🚨 监控告警建议

### ⚠️ 需要关注的情况

1. **训练进度停滞**
   ```
   curriculum/training_progress_percent 长时间无变化
   ```

2. **数据使用效率低**
   ```
   curriculum/stage_data_efficiency < 0.5 (数据重复使用不足)
   ```

3. **性能不稳定**
   ```
   curriculum/performance_stability > 0.1 (波动过大)
   ```

4. **进阶条件不满足**
   ```
   curriculum/can_advance = 0 且持续很长时间
   ```

## 💡 使用示例

### 查看当前阶段数据使用情况
```python
# 在WandB中查看这些指标：
- 当前阶段: foundation (curriculum/current_stage_idx = 0)
- 数据集大小: 8500 (curriculum/dataset_size)
- 训练进度: 124.0% (curriculum/training_progress_percent) ✅
- 已完成epoch: 1.24 (curriculum/epochs_completed) ✅
- 可以进阶: True (curriculum/can_advance) ✅
```

### 分析数据覆盖率
```python
# 通过这些指标了解数据使用情况：
- 总数据量: 50000 (curriculum/full_dataset_size)
- 当前阶段数据: 8500 (curriculum/dataset_size) 
- 阶段覆盖率: 17.0% (curriculum/stage_dataset_coverage_percent)
- 已训练样本: 10540 (curriculum/cumulative_samples_trained)
- 阶段内覆盖: 124.0% (curriculum/cumulative_coverage_percent)
```

这样您就可以通过WandB实时监控每个阶段的完整数据集使用情况，确保没有数据被遗漏！ 