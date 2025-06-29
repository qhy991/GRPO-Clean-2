# WandB图表优化指南 - 数值显示vs文字显示

## 🎯 问题解决

**问题**：WandB中显示文字而不是图表  
**原因**：传入的数据包含字符串类型  
**解决**：所有数据都转换为数值类型（int/float）

## ✅ 已优化的数据类型

### 📊 核心指标（保证图表显示）

| 指标名称 | 数据类型 | 示例值 | 图表类型 |
|---------|----------|--------|----------|
| `curriculum/training_progress_percent` | float | 124.5 | 📈 线性图 |
| `curriculum/epochs_completed` | float | 1.24 | 📈 线性图 |
| `curriculum/steps_completed` | int | 10540 | 📈 线性图 |
| `curriculum/cumulative_coverage_percent` | float | 80.0 | 📈 线性图 |
| `curriculum/can_advance` | float | 1.0 | 📊 柱状图 |

### 🔧 数据级别编码（避免文字显示）

**之前（文字显示）**：
```python
# ❌ 这样会显示为文字
wandb_data["curriculum/stage_level_0"] = "basic"
wandb_data["curriculum/stage_level_1"] = "intermediate"
```

**现在（数值显示）**：
```python
# ✅ 这样会显示为图表
wandb_data["curriculum/stage_has_basic"] = 1.0      # 是否包含basic级别
wandb_data["curriculum/stage_has_intermediate"] = 1.0  # 是否包含intermediate级别
wandb_data["curriculum/stage_min_level"] = 1.0      # 最低级别（1=basic, 2=intermediate...）
wandb_data["curriculum/stage_max_level"] = 2.0      # 最高级别
```

### 📈 阶段名称编码

**级别映射**：
```python
level_mapping = {
    'basic': 1.0,
    'intermediate': 2.0, 
    'advanced': 3.0,
    'expert': 4.0,
    'master': 5.0
}

stage_name_mapping = {
    'foundation': 0.0,
    'elementary': 1.0,
    'intermediate': 2.0,
    'advanced': 3.0,
    'expert': 4.0,
    'comprehensive': 5.0
}
```

## 🎨 推荐的WandB图表配置

### 1. **训练进度监控面板**

```python
# 主要进度图表
charts = [
    {
        "title": "🎯 训练进度总览",
        "metrics": [
            "curriculum/training_progress_percent",  # 主线（目标100%）
            "curriculum/epochs_completed",           # 辅助线
        ],
        "type": "line",
        "y_axis": "进度/Epoch数"
    },
    
    {
        "title": "📊 数据集使用情况", 
        "metrics": [
            "curriculum/cumulative_coverage_percent",
            "curriculum/stage_dataset_coverage_percent"
        ],
        "type": "line",
        "y_axis": "覆盖率(%)"
    }
]
```

### 2. **进阶条件监控面板**

```python
charts = [
    {
        "title": "✅ 进阶条件状态",
        "metrics": [
            "curriculum/can_advance",                    # 总体状态
            "curriculum/performance_requirement_met",    # 性能条件
            "curriculum/evaluations_requirement_met",    # 评估条件  
            "curriculum/full_training_requirement_met"   # 完整训练条件
        ],
        "type": "bar",  # 柱状图显示满足情况
        "y_axis": "满足状态(0/1)"
    },
    
    {
        "title": "📈 要求完成度",
        "metrics": [
            "curriculum/performance_completion_ratio",
            "curriculum/evaluations_completion_ratio", 
            "curriculum/full_training_completion_ratio"
        ],
        "type": "line",
        "y_axis": "完成度倍数"
    }
]
```

### 3. **阶段分析面板**

```python
charts = [
    {
        "title": "🏗️ 阶段配置概览",
        "metrics": [
            "curriculum/stage_name_encoded",         # 阶段进展
            "curriculum/stage_complexity_min",       # 复杂度范围
            "curriculum/stage_complexity_max",
            "curriculum/stage_levels_count"          # 级别多样性
        ],
        "type": "line",
        "y_axis": "配置值"
    },
    
    {
        "title": "📦 数据级别分布",
        "metrics": [
            "curriculum/stage_has_basic",
            "curriculum/stage_has_intermediate", 
            "curriculum/stage_has_advanced",
            "curriculum/stage_has_expert",
            "curriculum/stage_has_master"
        ],
        "type": "area",  # 面积图显示级别包含情况
        "y_axis": "包含状态(0/1)"
    }
]
```

### 4. **性能趋势面板**

```python
charts = [
    {
        "title": "📈 性能趋势分析",
        "metrics": [
            "curriculum/latest_performance",
            "curriculum/avg_stage_performance",
            "curriculum/recent_3_avg_performance",
            "curriculum/performance_threshold"       # 基准线
        ],
        "type": "line",
        "y_axis": "性能值"
    },
    
    {
        "title": "📊 性能稳定性",
        "metrics": [
            "curriculum/performance_trend",          # 变化趋势
            "curriculum/performance_stability",      # 稳定性（标准差）
            "curriculum/performance_improvement"     # 总体改善
        ],
        "type": "line",
        "y_axis": "趋势/稳定性"
    }
]
```

## 🔍 图表显示验证

### ✅ 正确的图表显示特征

1. **线性图**：显示连续的数值变化趋势
2. **柱状图**：显示离散的状态值（0/1）
3. **面积图**：显示累积或分布情况
4. **平滑曲线**：数值连续变化，无突兀跳跃

### ❌ 错误的文字显示特征

1. **表格形式**：显示为文字列表
2. **标签显示**：显示为离散的文字标签
3. **无法绘制趋势线**：数据点无法连接
4. **无法进行数值运算**：无法计算平均值、趋势等

## 💡 最佳实践建议

### 1. **数据类型规范**
```python
# ✅ 推荐
wandb_data = {
    "curriculum/metric_name": float(value),    # 连续数值
    "curriculum/count_name": int(value),       # 整数计数
    "curriculum/flag_name": float(bool_value), # 布尔转浮点
}

# ❌ 避免
wandb_data = {
    "curriculum/metric_name": str(value),      # 字符串
    "curriculum/list_name": [1, 2, 3],        # 列表
    "curriculum/dict_name": {"key": "value"}  # 字典
}
```

### 2. **命名规范**
```python
# 使用前缀分组相关指标
"curriculum/training_*"     # 训练相关
"curriculum/performance_*"  # 性能相关  
"curriculum/stage_*"        # 阶段相关
"curriculum/data_*"         # 数据相关
```

### 3. **数值范围控制**
```python
# 确保数值在合理范围内
progress_percent = min(100.0, max(0.0, calculated_percent))
completion_ratio = min(1.0, max(0.0, calculated_ratio))
```

## 🎯 预期效果

经过优化后，您在WandB中将看到：

- ✅ **流畅的训练进度曲线**：`curriculum/training_progress_percent`从0%到100%+
- ✅ **清晰的epoch进展**：`curriculum/epochs_completed`从0.0到1.0+
- ✅ **直观的进阶状态**：`curriculum/can_advance`在0和1之间切换
- ✅ **详细的数据使用分析**：各种覆盖率和效率指标的变化趋势

所有指标都将以图表形式显示，便于分析训练进展和数据使用情况！ 