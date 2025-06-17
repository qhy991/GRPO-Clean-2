#!/bin/bash
# restart_with_debug.sh - 重启训练并启用完整DEBUG功能

echo "🛑 重启训练以启用完整DEBUG功能"
echo "=" * 50

# 1. 停止当前训练
echo "🔴 查找并停止当前训练进程..."
TRAINING_PIDS=$(pgrep -f "main.py")
if [ ! -z "$TRAINING_PIDS" ]; then
    echo "发现训练进程: $TRAINING_PIDS"
    echo "是否停止当前训练? (y/N)"
    read -r RESPONSE
    if [[ "$RESPONSE" == "y" ]] || [[ "$RESPONSE" == "Y" ]]; then
        kill $TRAINING_PIDS
        echo "✅ 训练进程已停止"
        sleep 5
    else
        echo "❌ 用户取消，保持当前训练"
        exit 1
    fi
else
    echo "ℹ️  未发现运行中的训练进程"
fi

# 2. 清理旧的DEBUG收集器
DEBUG_PIDS=$(pgrep -f "simple_debug_collector.py")
if [ ! -z "$DEBUG_PIDS" ]; then
    echo "🧹 清理DEBUG收集器进程: $DEBUG_PIDS"
    kill $DEBUG_PIDS
fi

MANUAL_DEBUG_PIDS=$(pgrep -f "manual_debug_monitor.py")
if [ ! -z "$MANUAL_DEBUG_PIDS" ]; then
    echo "🧹 清理手动DEBUG监控: $MANUAL_DEBUG_PIDS"
    kill $MANUAL_DEBUG_PIDS
fi

# 3. 备份当前输出
BACKUP_DIR="./model_parallel_only_outputs/backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "./model_parallel_only_outputs" ]; then
    echo "💾 备份当前输出到: $BACKUP_DIR"
    mv "./model_parallel_only_outputs" "$BACKUP_DIR"
fi

# 4. 验证脚本更新
echo "🔍 验证DEBUG配置..."
if grep -q "export DEBUG_MODE" ./run_model_parallel_only.sh; then
    echo "✅ DEBUG配置已更新"
else
    echo "❌ DEBUG配置未更新，请检查 run_model_parallel_only.sh"
    exit 1
fi

# 5. 重新启动训练
echo ""
echo "🚀 重新启动训练（完整DEBUG功能）..."
echo "📊 新功能:"
echo "  - 保存所有生成样本"
echo "  - 保存失败/成功样本"
echo "  - 详细训练指标"
echo "  - 奖励计算详情"
echo ""

./run_efficient_lora.sh

echo "✅ 训练重启完成" 