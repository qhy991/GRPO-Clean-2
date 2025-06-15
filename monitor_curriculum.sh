#!/bin/bash
# 实时监控课程学习状态
echo "🔍 课程学习状态实时监控"
echo "按Ctrl+C退出"
echo "========================"

while true; do
    clear
    echo "⏰ $(date '+%H:%M:%S') - 课程学习状态更新"
    echo "========================"
    
    # 查找最新的状态文件
    LATEST_STATE=$(find . -name "curriculum_state_detailed.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_STATE" ]; then
        echo "📁 状态文件: $LATEST_STATE"
        
        # 提取关键信息
        CURRENT_STAGE=$(jq -r '.current_stage // "N/A"' "$LATEST_STATE" 2>/dev/null)
        HISTORY_COUNT=$(jq -r '.performance_history | length // 0' "$LATEST_STATE" 2>/dev/null)
        LAST_PERFORMANCE=$(jq -r '.performance_history[-1].performance // "N/A"' "$LATEST_STATE" 2>/dev/null)
        LAST_LOSS=$(jq -r '.performance_history[-1].loss // "N/A"' "$LATEST_STATE" 2>/dev/null)
        LAST_STEP=$(jq -r '.performance_history[-1].step // "N/A"' "$LATEST_STATE" 2>/dev/null)
        
        echo "🎯 当前阶段: $CURRENT_STAGE"
        echo "📊 历史记录数: $HISTORY_COUNT"
        echo "🏆 最新性能: $LAST_PERFORMANCE"
        echo "📉 最新Loss: $LAST_LOSS"
        echo "👣 最新Step: $LAST_STEP"
        
        # 检查异常值
        if [ "$LAST_LOSS" = "Infinity" ] || [ "$LAST_LOSS" = "inf" ]; then
            echo "⚠️  警告: 检测到无穷大Loss值！"
        fi
    else
        echo "❌ 未找到状态文件"
    fi
    
    echo ""
    echo "📈 最近的调试日志 (最后5行):"
    find . -name "curriculum_progress_debug.txt" -type f -exec tail -5 {} \; 2>/dev/null | tail -5
    
    sleep 5
done
