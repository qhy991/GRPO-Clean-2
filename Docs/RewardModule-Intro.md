# 多目标奖励系统深度分析

## 🎯 RewardCalculator 核心架构

### 四维奖励体系
```
总奖励 = 功能性(70%) + 效率(15%) + 可读性(10%) + 鲁棒性(5%)
```

#### 1. 功能性奖励 (Functional)
- **编译成功/失败** → ±2.0 / -4.0
- **仿真通过率** → 非线性递增奖励
- **测试覆盖度** → 边缘情况处理奖励
- **完整性检查** → 所有测试通过额外奖励

#### 2. 代码质量奖励 (Quality)
```python
efficiency_score = 效率指标 × 2.0 + 结构指标 × 1.0
readability_score = 可读性指标 × 1.0
complexity_penalty = (1 - 复杂度评分) × (-1.0)
```

#### 3. 鲁棒性奖励 (Robustness)
- **边缘情况处理** → +1.5
- **综合友好性** → +1.0
- **资源使用优化** → -0.5 (惩罚过度使用)

## 🔄 批次奖励计算流程

```python
calculate_batch_rewards()
├── 输入验证（列表长度匹配）
├── 逐样本奖励计算
│   ├── LLM输出解析（thinking + code）
│   ├── Verilog代码验证
│   ├── 仿真执行（Icarus Verilog）
│   ├── 结果解析（通过/失败测试数）
│   └── 多目标奖励聚合
├── 批次统计计算
├── 经验回放存储
└── WandB指标记录
```

## 🛡️ 失败处理与调试

### 自动调试保存
- **缺失代码块** → 保存到 `reward_debug/missing_code/`
- **验证失败** → 保存到 `reward_debug/validation_errors/`
- **解析异常** → 详细上下文信息记录

### 错误恢复策略
```python
# 奖励计算失败时的降级策略
try:
    return calculate_detailed_reward()
except ParsingError:
    return default_parsing_penalty
except SimulationError:
    return compilation_only_reward
except Exception:
    return safe_fallback_reward
```