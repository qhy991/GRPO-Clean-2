# GRPO断续训练参数传递修复指南

## 问题概述

在GRPO训练中进行断续训练时，经常会出现参数传递断层问题，主要表现在：

1. **配置参数不一致**：长度配置、模型配置等在恢复时出现偏差
2. **WandB同步失败**：步数不匹配、run ID丢失等
3. **课程学习状态丢失**：阶段信息、性能历史无法正确恢复
4. **模型状态不完整**：checkpoint文件损坏或缺失关键信息

## 修复工具使用指南

### 1. 基础清理脚本 (`cleanup_before_training.sh`)

**用途**：清理环境变量、验证文件完整性、生成诊断报告

```bash
# 完整修复
./cleanup_before_training.sh

# 仅生成诊断报告
./cleanup_before_training.sh --report-only

# 仅清理临时文件
./cleanup_before_training.sh --clean-only
```

**主要功能**：
- ✅ 清理残留的WandB环境变量
- ✅ 验证checkpoint文件完整性
- ✅ 检查配置文件结构
- ✅ 验证Python依赖
- ✅ 生成详细诊断报告

### 2. Python参数修复工具 (`fix_resume_parameters.py`)

**用途**：深度诊断和修复配置不一致问题

```bash
# 诊断指定checkpoint
python3 fix_resume_parameters.py --checkpoint /path/to/checkpoint

# 自动修复
python3 fix_resume_parameters.py --checkpoint /path/to/checkpoint --auto-fix

# 仅生成报告
python3 fix_resume_parameters.py --report-only
```

**主要功能**：
- 🔍 配置类一致性检查
- 📏 长度配置匹配验证
- 📊 WandB同步状态检查
- 📂 Checkpoint状态验证

### 3. 课程学习状态修复工具 (`fix_curriculum_sync.py`)

**用途**：专门处理课程学习状态同步问题

```bash
# 诊断课程状态
python3 fix_curriculum_sync.py --diagnose

# 创建新的状态文件
python3 fix_curriculum_sync.py --create-fresh --checkpoint /path/to/checkpoint

# 自动修复状态问题
python3 fix_curriculum_sync.py --auto-fix
```

**主要功能**：
- 📚 课程状态文件验证
- 🎯 管理器模块检查
- 🔄 与checkpoint同步验证
- 📞 回调模块完整性检查

## 常见问题及解决方案

### 问题1：长度配置不匹配

**症状**：
```
⚠️ 警告: prompt长度(1536) + completion长度(2560) = 4096 > 最大序列长度(4096)
```

**解决方案**：
1. 检查训练脚本中的长度配置：
   ```bash
   # 在 run_enhanced_grpo_training.sh 中调整
   MAX_PROMPT_LENGTH=1024
   MAX_COMPLETION_LENGTH=3072
   LENGTH_ALLOCATION_STRATEGY="completion_heavy"
   ```

2. 或使用自动分配：
   ```bash
   LENGTH_ALLOCATION_STRATEGY="balanced"  # 50/50分配
   ```

### 问题2：WandB步数不同步

**症状**：
```
⚠️ 检测到步数偏移: WandB=150, Trainer=100, 偏移=50
```

**解决方案**：
1. 清理WandB环境变量：
   ```bash
   unset WANDB_RUN_ID
   unset WANDB_RESUME
   ```

2. 使用正确的run ID恢复：
   ```bash
   export WANDB_RUN_ID="your_run_id"
   export WANDB_RESUME="must"
   ```

### 问题3：课程学习状态丢失

**症状**：
```
❌ 课程状态文件损坏或课程管理器导入失败
```

**解决方案**：
1. 创建新的状态文件：
   ```bash
   python3 fix_curriculum_sync.py --create-fresh --checkpoint /path/to/checkpoint
   ```

2. 验证课程管理器：
   ```python
   from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
   ```

### 问题4：配置类导入失败

**症状**：
```
ImportError: cannot import name 'ScriptConfig' from 'grpo_project.configs'
```

**解决方案**：
1. 检查项目结构：
   ```bash
   ls -la grpo_project/configs/
   # 应该包含: __init__.py, environment.py, training.py, reward.py
   ```

2. 验证Python路径：
   ```python
   import sys
   sys.path.insert(0, '/path/to/project/root')
   ```

## 断续训练最佳实践

### 1. 训练前检查清单

```bash
# 1. 运行清理脚本
./cleanup_before_training.sh

# 2. 验证checkpoint完整性
ls -la /path/to/checkpoint/
# 应包含: trainer_state.json, config.json, pytorch_model.bin或model.safetensors

# 3. 检查配置一致性
python3 fix_resume_parameters.py --checkpoint /path/to/checkpoint

# 4. 验证课程状态
python3 fix_curriculum_sync.py --diagnose
```

### 2. 环境变量设置

正确的环境变量设置顺序：
```bash
# 1. 基础环境
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="VerilogGRPO_Enhanced_v3"
export WANDB_ENTITY="your_entity"

# 2. WandB恢复（如果需要）
export WANDB_RUN_ID="your_run_id"  # 从诊断报告获取
export WANDB_RESUME="must"         # 或 "allow"

# 3. 其他配置
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

### 3. 配置参数优先级

参数传递的优先级顺序：
1. **命令行参数** > **环境变量** > **配置文件**
2. **ScriptConfig** 中的长度配置会同步到 **GRPOConfig**
3. **checkpoint中的状态** 会覆盖部分配置

### 4. 监控恢复过程

关键日志观察点：
```bash
# 1. 配置同步日志
grep "🔧 同步长度配置" log_file.txt

# 2. WandB恢复日志
grep "WandB恢复" log_file.txt

# 3. 课程状态日志
grep "课程学习" log_file.txt

# 4. Checkpoint加载日志
grep "checkpoint" log_file.txt
```

## 高级故障排除

### 复杂问题诊断流程

1. **收集信息**：
   ```bash
   # 生成完整诊断报告
   ./cleanup_before_training.sh --report-only
   python3 fix_resume_parameters.py --checkpoint /path/to/checkpoint --report-only
   python3 fix_curriculum_sync.py --diagnose
   ```

2. **分析日志**：
   ```bash
   # 查看训练日志
   tail -n 100 grpo_pipeline_log.txt
   
   # 查看WandB日志
   cat wandb/debug.log
   
   # 查看课程学习日志
   cat curriculum_progress_debug.txt
   ```

3. **逐步修复**：
   ```bash
   # 先修复高优先级问题
   python3 fix_resume_parameters.py --auto-fix
   
   # 再处理课程状态
   python3 fix_curriculum_sync.py --auto-fix
   
   # 最后清理环境
   ./cleanup_before_training.sh --clean-only
   ```

### 备份与恢复策略

**重要文件备份**：
```bash
# 训练前备份
cp curriculum_state.json curriculum_state.json.backup
cp -r /path/to/checkpoint /path/to/checkpoint.backup

# 配置文件备份
tar -czf config_backup.tar.gz grpo_project/configs/
```

**恢复策略**：
1. 如果修复失败，从备份恢复
2. 考虑从更早的稳定checkpoint重新开始
3. 重置课程学习状态到安全点

## 预防措施

### 1. 定期状态保存

在训练脚本中添加：
```python
# 每100步保存状态
if step % 100 == 0:
    curriculum_manager.save_state()
    sync_manager.update_step_offset(step)
```

### 2. 配置验证

在训练开始前验证：
```python
def validate_config_consistency(script_cfg, grpo_cfg):
    assert script_cfg.script_max_prompt_length + script_cfg.script_max_completion_length <= script_cfg.max_seq_length
    assert grpo_cfg.max_prompt_length == script_cfg.script_max_prompt_length
    assert grpo_cfg.max_completion_length == script_cfg.script_max_completion_length
```

### 3. 自动化检查

在训练脚本开头添加：
```bash
# 自动运行检查
if [ -f "cleanup_before_training.sh" ]; then
    echo "🔧 运行断续训练检查..."
    ./cleanup_before_training.sh --report-only
fi
```

## 总结

通过使用这套工具和遵循最佳实践，可以有效避免和解决GRPO断续训练中的参数传递问题。关键是：

1. **训练前检查**：使用提供的工具诊断潜在问题
2. **正确配置**：确保各层配置参数一致
3. **状态同步**：保持WandB、课程学习等状态同步
4. **监控日志**：及时发现和处理异常

如果遇到本指南未覆盖的问题，请查看生成的诊断报告或检查相关日志文件。 