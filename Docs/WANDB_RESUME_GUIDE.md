# WandB 自动恢复配置指南

## 🎯 概述

现在训练管道已经集成了**自动WandB恢复配置**功能，无需手动设置环境变量或运行额外脚本。当从checkpoint恢复训练时，系统会自动：

1. 🔍 检测checkpoint目录中的WandB run ID
2. 🔧 设置正确的环境变量
3. 🔄 配置WandB恢复模式
4. 📊 确保训练数据连续性

## 🚀 使用方法

### 方法1: 直接运行训练脚本（推荐）

```bash
cd LLM/GRPO-Clean-2
./run_enhanced_grpo_training.sh
```

**系统会自动处理所有WandB恢复配置！**

### 方法2: 手动指定checkpoint

如果需要从特定checkpoint恢复：

```bash
# 修改配置文件中的 resume_from_checkpoint 参数
# 或者在脚本中设置 RESUME_FROM_CHECKPOINT_DIR
export RESUME_FROM_CHECKPOINT_DIR="/path/to/your/checkpoint"
./run_enhanced_grpo_training.sh
```

## 🔧 自动配置流程

### 1. 检测阶段
- ✅ 检查是否指定了 `resume_from_checkpoint`
- ✅ 验证checkpoint目录是否存在
- ✅ 确定是新训练还是恢复训练

### 2. WandB Run ID 提取
系统会按以下顺序尝试提取WandB run ID：

1. **WandB目录方法**: 检查 `checkpoint/wandb/` 目录
2. **trainer_state.json方法**: 从训练状态文件中提取
3. **环境变量方法**: 使用已存在的 `WANDB_RUN_ID`

### 3. 环境变量设置
根据检测结果自动设置：

```bash
# 恢复训练时
export WANDB_RUN_ID="extracted_run_id"
export WANDB_RESUME="must"

# 新训练时
export WANDB_RESUME="allow"
# WANDB_RUN_ID 保持未设置，让WandB生成新ID
```

## 📊 WandB Callback 增强

`DetailedWandbCallback` 已经升级，现在会：

- 🔄 优先使用main.py设置的环境变量
- 📝 记录恢复信息到WandB
- 🛡️ 提供备用方案确保稳定性
- 📊 自动配置标记和元数据

## 🧪 测试功能

运行测试脚本验证配置：

```bash
cd LLM/GRPO-Clean-2
python test_wandb_resume.py
```

测试内容包括：
- ✅ 新训练配置测试
- ✅ 不存在checkpoint处理测试  
- ✅ 真实checkpoint恢复测试
- ✅ Run ID提取功能测试

## 🔍 故障排除

### 问题1: 找不到WandB run ID
**症状**: 日志显示 "❌ 未能找到WandB run ID"

**解决方案**:
1. 检查checkpoint目录是否完整
2. 确认之前的训练确实使用了WandB
3. 系统会自动使用 `allow` 模式继续训练

### 问题2: WandB数据不连续
**症状**: WandB界面显示数据中断

**解决方案**:
1. 确认使用了正确的checkpoint路径
2. 检查 `WANDB_RUN_ID` 环境变量是否正确设置
3. 查看训练日志中的WandB配置信息

### 问题3: 权限错误
**症状**: 无法读取checkpoint文件

**解决方案**:
1. 检查文件权限: `ls -la /path/to/checkpoint`
2. 确保当前用户有读取权限
3. 检查磁盘空间是否充足

## 📝 日志信息解读

### 成功恢复的日志示例：
```
🔄 检测到checkpoint恢复: /path/to/checkpoint-24
🔍 尝试从WandB目录提取run ID...
✅ 找到WandB run ID: abc123def456
🔧 设置环境变量: WANDB_RUN_ID=abc123def456
🔧 设置环境变量: WANDB_RESUME=must
✅ WandB恢复配置完成!
```

### 新训练的日志示例：
```
🆕 开始新的训练会话
🔧 设置环境变量: WANDB_RESUME=allow
✅ WandB恢复配置完成!
```

## 🎉 优势

1. **零配置**: 无需手动设置环境变量
2. **自动化**: 训练脚本自动处理所有配置
3. **稳定性**: 多种备用方案确保可靠性
4. **透明性**: 详细日志显示所有操作
5. **兼容性**: 支持新训练和恢复训练

## 📚 相关文件

- `main.py`: 主训练管道（包含自动配置逻辑）
- `grpo_project/callbacks/wandb.py`: 增强的WandB回调
- `test_wandb_resume.py`: 测试脚本
- `run_enhanced_grpo_training.sh`: 训练启动脚本

---

**现在您可以专注于训练，让系统自动处理WandB恢复配置！** 🚀 