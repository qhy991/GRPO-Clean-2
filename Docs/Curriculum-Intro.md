# 双层课程学习系统架构

## 📚 CurriculumStageConfig 阶段配置

```python
@dataclass
class CurriculumStageConfig:
    name: str                    # 阶段名称
    dataset_levels: List[str]    # 数据集等级 ['basic', 'intermediate'...]
    complexity_range: Tuple      # 复杂度范围 (min, max)
    epochs_ratio: float          # 训练epoch比例
    performance_threshold: float # 进入下一阶段的性能阈值
    min_evaluations: int        # 最少评估次数
```

## 🎯 默认课程阶段设计

```
foundation (25%)    → basic          (0.0-3.0) → 阈值:0.65
elementary (25%)    → basic+inter    (0.0-5.0) → 阈值:0.60  
intermediate (25%)  → intermediate   (3.0-7.0) → 阈值:0.55
advanced (15%)      → inter+advanced (5.0-9.0) → 阈值:0.50
expert (10%)        → advanced+expert(7.0-10.0)→ 阈值:0.45
```

## 🔄 FixedEnhancedCurriculumManager 

### 核心功能
- **智能数据集过滤** - 双层过滤（等级+复杂度）
- **性能跟踪** - 历史评估记录和趋势分析
- **自动进阶判断** - 基于阈值和最小评估次数
- **详细调试日志** - 每个决策步骤的完整记录

### 进阶决策逻辑
```python
should_advance_stage():
├── 检查当前阶段是否为最后阶段
├── 累积当前性能到历史记录
├── 验证最小评估次数要求
├── 计算最近窗口平均性能
├── 对比性能阈值
└── 返回进阶决策 + 详细原因
```

## 📊 课程学习回调系统

### CurriculumProgressCallback
- **性能检查间隔**：可配置（默认5步）
- **状态监控**：实时记录阶段变化
- **WandB集成**：课程进度可视化

### EnhancedCurriculumDebugCallback  
- **深度调试**：每20步详细状态记录
- **趋势分析**：性能变化趋势检测
- **决策记录**：进阶条件检查详情

## 🎛️ 自适应课程配置

```python
# 数据集驱动的课程生成
create_custom_curriculum_stages(
    dataset_distribution,    # 基于实际数据分布
    focus_levels,           # 重点训练等级
    complexity_emphasis     # 复杂度倾向：simple/balanced/complex
)
```