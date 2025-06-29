# 课程学习性能检查间隔配置指南

## 📋 概述

现在您可以在训练脚本中自定义课程学习的性能检查间隔！这个参数控制多少步检查一次性能并判断是否可以进阶到下一个课程阶段。

## 🔧 配置方法

### 在Shell脚本中设置

编辑 `run_enhanced_grpo_training.sh` 文件，找到这一行：

```bash
# 🔧 新增：课程学习性能检查间隔配置
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=25  # 默认每25步检查一次
```

修改数值到您想要的步数：

```bash
# 快速调试 - 每10步检查一次
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=10

# 默认平衡 - 每25步检查一次  
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=25

# 节省计算 - 每50步检查一次
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=50
```

## 📊 建议设置

### 🚀 调试阶段：5-10步
```bash
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=10
```
- **优点**：快速响应，便于观察课程进阶过程
- **缺点**：计算开销稍大
- **适用**：开发调试、问题排查

### ⚖️ 正常训练：20-30步
```bash
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=25  # 默认值
```
- **优点**：性能和响应平衡
- **缺点**：无明显缺点
- **适用**：大多数训练场景

### 🏃 长期训练：40-60步
```bash
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=50
```
- **优点**：最小化计算开销
- **缺点**：响应稍慢
- **适用**：大规模长期训练

## 📈 性能影响分析

| 检查间隔 | 每1000步检查次数 | 计算开销 | 推荐场景 |
|---------|----------------|----------|----------|
| 5步     | 200次          | 🔴 高     | 快速调试 |
| 10步    | 100次          | 🟡 中等   | 细致监控 |
| 25步    | 40次           | 🟢 低     | 默认推荐 |
| 50步    | 20次           | 🟢 低     | 长期训练 |
| 100步   | 10次           | 🟢 极低   | 粗粒度监控 |

## 🔄 参数传递链路

```
Shell脚本变量
    ↓
CMD_ARGS传递
    ↓  
Python argparse解析
    ↓
ScriptConfig配置类
    ↓
CurriculumProgressCallback回调
    ↓
实际性能检查执行
```

## 📝 使用示例

### 示例1：快速调试设置
```bash
# 在 run_enhanced_grpo_training.sh 中
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=10

# 训练时会看到更频繁的日志：
# [00:03:25] CURRICULUM: 📈 课程状态更新 (步数: 10)
# [00:03:35] CURRICULUM: 📈 课程状态更新 (步数: 20)
# [00:03:45] CURRICULUM: 📈 课程状态更新 (步数: 30)
```

### 示例2：生产环境设置
```bash
# 在 run_enhanced_grpo_training.sh 中
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=40

# 训练时会看到较少但充足的日志：
# [00:03:25] CURRICULUM: 📈 课程状态更新 (步数: 40)
# [00:04:15] CURRICULUM: 📈 课程状态更新 (步数: 80)
# [00:05:05] CURRICULUM: 📈 课程状态更新 (步数: 120)
```

## 🔍 日志确认

训练开始时，您会在日志中看到确认信息：

```
Enhanced Features Summary:
  ✅ Curriculum performance check interval: every 25 steps
```

在调试日志文件中也会记录：

```
=== CurriculumProgressCallback Debug Log - 2025-01-07 09:30:00 ===
初始化课程学习调试回调
性能检查间隔: 每25步
```

## ⚡ 实时调整

虽然参数在训练脚本中设置，但您可以通过以下方式在不同训练任务中使用不同设置：

### 方法1：修改脚本文件
```bash
# 直接编辑 run_enhanced_grpo_training.sh
vim run_enhanced_grpo_training.sh
# 修改 CURRICULUM_PERFORMANCE_CHECK_INTERVAL 的值
```

### 方法2：环境变量覆盖（如果需要可以实现）
```bash
# 可以考虑未来支持环境变量覆盖
export CURRICULUM_PERFORMANCE_CHECK_INTERVAL=15
./run_enhanced_grpo_training.sh
```

## 🧪 测试验证

运行测试脚本验证配置是否正确：

```bash
python test_curriculum_interval_config.py
```

如果看到 "✅ 所有测试通过！" 说明配置系统工作正常。

## 📊 监控建议

### W&B监控
训练过程中，在W&B面板查看：
- `curriculum/current_stage_idx`：当前课程阶段
- `curriculum/latest_performance`：最新性能指标
- `curriculum/stage_step_count`：当前阶段步数

### 本地日志监控
查看调试日志文件：
```bash
tail -f enhanced_grpo_v3_runs/*/curriculum_progress_debug.txt
```

## 🔧 故障排除

### 问题1：参数没有生效
检查以下几点：
1. 脚本中是否正确设置了变量
2. CMD_ARGS中是否包含了参数传递
3. Python配置类是否正确定义了字段

### 问题2：检查过于频繁
如果发现性能检查过于频繁：
```bash
# 增加检查间隔
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=50
```

### 问题3：课程进阶太慢
如果课程进阶响应太慢：
```bash
# 减少检查间隔
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=15
```

## 🎯 最佳实践

1. **开发阶段**：使用较小值（10-15步）便于调试
2. **正式训练**：使用默认值（25步）或稍大值（30-40步）
3. **大规模训练**：使用较大值（50-100步）减少开销
4. **根据训练速度调整**：训练快可以用小值，训练慢可以用大值
5. **监控实际效果**：观察W&B和日志，根据实际情况调整

配置完成后，您的课程学习系统将按照设定的间隔进行性能检查和阶段进阶判定！ 