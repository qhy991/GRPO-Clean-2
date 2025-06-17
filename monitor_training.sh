#!/bin/bash
# monitor_training.sh - 实时监控训练状态

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 默认配置
DEBUG_OUTPUT_BASE="./model_parallel_only_outputs/debug_data"
REFRESH_INTERVAL=30

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug_dir)
            DEBUG_OUTPUT_BASE="$2"
            shift 2
            ;;
        --interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --debug_dir DIR    DEBUG数据目录 (默认: $DEBUG_OUTPUT_BASE)"
            echo "  --interval SEC     刷新间隔秒数 (默认: $REFRESH_INTERVAL)"
            echo "  --help            显示此帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

monitor_training() {
    log_info "🚀 开始监控训练状态"
    log_info "📁 DEBUG目录: $DEBUG_OUTPUT_BASE"
    log_info "⏱️  刷新间隔: $REFRESH_INTERVAL 秒"
    echo ""
    
    while true; do
        clear
        echo "========================================================================"
        echo "                        训练状态实时监控"
        echo "========================================================================"
        echo "监控时间: $(date)"
        echo "DEBUG目录: $DEBUG_OUTPUT_BASE"
        echo ""
        
        # 检查训练是否在运行
        if pgrep -f "main.py" > /dev/null; then
            log_info "✅ 训练进程正在运行"
            TRAIN_PID=$(pgrep -f "main.py")
            echo "训练进程PID: $TRAIN_PID"
        else
            log_warning "⚠️  未检测到训练进程"
        fi
        echo ""
        
        # GPU状态
        log_info "📊 GPU状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
        echo ""
        
        # DEBUG数据统计
        if [ -d "$DEBUG_OUTPUT_BASE" ]; then
            log_info "🐛 DEBUG数据统计:"
            
            # 统计各类文件数量
            GENERATION_FILES=$(find "$DEBUG_OUTPUT_BASE" -name "*.json" -path "*/generations/*" 2>/dev/null | wc -l)
            FAILED_FILES=$(find "$DEBUG_OUTPUT_BASE" -name "*.json" -path "*/failed_generations/*" 2>/dev/null | wc -l)
            SUCCESS_FILES=$(find "$DEBUG_OUTPUT_BASE" -name "*.json" -path "*/successful_generations/*" 2>/dev/null | wc -l)
            METRICS_FILES=$(find "$DEBUG_OUTPUT_BASE" -name "*.json" -path "*/detailed_metrics/*" 2>/dev/null | wc -l)
            
            echo "  📝 生成样本文件: $GENERATION_FILES"
            echo "  ❌ 失败样本文件: $FAILED_FILES"
            echo "  ✅ 成功样本文件: $SUCCESS_FILES"
            echo "  📈 指标文件: $METRICS_FILES"
            
            # 最新文件信息
            LATEST_FILE=$(find "$DEBUG_OUTPUT_BASE" -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
            if [ ! -z "$LATEST_FILE" ]; then
                LATEST_TIME=$(stat -c %y "$LATEST_FILE" 2>/dev/null | cut -d'.' -f1)
                echo "  🕒 最新文件: $(basename "$LATEST_FILE") ($LATEST_TIME)"
            fi
        else
            log_warning "⚠️  DEBUG目录不存在: $DEBUG_OUTPUT_BASE"
        fi
        echo ""
        
        # 日志文件监控
        LATEST_LOG=$(find "$DEBUG_OUTPUT_BASE" -name "full_training_log.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ ! -z "$LATEST_LOG" ]; then
            log_info "📄 最新训练日志 (最后10行):"
            tail -10 "$LATEST_LOG" 2>/dev/null | while read line; do
                echo "  $line"
            done
        fi
        echo ""
        
        # 磁盘使用情况
        log_info "💾 磁盘使用情况:"
        df -h "$DEBUG_OUTPUT_BASE" 2>/dev/null | tail -1 | awk '{print "  使用率: " $5 ", 可用空间: " $4}'
        echo ""
        
        echo "========================================================================"
        echo "按 Ctrl+C 停止监控 | 下次刷新: $REFRESH_INTERVAL 秒后"
        echo "========================================================================"
        
        sleep $REFRESH_INTERVAL
    done
}

# 信号处理
cleanup() {
    echo ""
    log_info "🛑 监控已停止"
    exit 0
}
trap cleanup INT TERM

# 启动监控
monitor_training 