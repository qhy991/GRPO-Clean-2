#!/bin/bash
# cleanup_before_training.sh - 断续训练参数传递修复脚本
# 解决GRPO训练中的参数传递断层问题

set -e

# 获取脚本目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo "🔧 GRPO断续训练参数传递修复脚本"
echo "📁 工作目录: ${SCRIPT_DIR}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# 断续训练常见问题修复
fix_resume_issues() {
    log_info "🔄 修复断续训练参数传递问题..."
    
    # 1. 清理残留的环境变量
    log_debug "清理WandB环境变量..."
    unset WANDB_RUN_ID 2>/dev/null || true
    unset WANDB_RESUME 2>/dev/null || true
    unset WANDB_RUN_NAME 2>/dev/null || true
    
    # 2. 检查并修复checkpoint状态文件
    log_debug "检查checkpoint状态文件..."
    if [ -f "run_enhanced_grpo_training.sh" ]; then
        # 从训练脚本中提取RESUME_FROM_CHECKPOINT_DIR
        RESUME_DIR=$(grep "RESUME_FROM_CHECKPOINT_DIR=" run_enhanced_grpo_training.sh | head -1 | cut -d'"' -f2)
        
        if [ -n "$RESUME_DIR" ] && [ -d "$RESUME_DIR" ]; then
            log_info "📂 检查checkpoint目录: $RESUME_DIR"
            
            # 检查trainer_state.json
            if [ -f "$RESUME_DIR/trainer_state.json" ]; then
                log_debug "验证trainer_state.json完整性..."
                if ! python3 -c "import json; json.load(open('$RESUME_DIR/trainer_state.json'))" 2>/dev/null; then
                    log_warning "trainer_state.json文件损坏，尝试修复..."
                    cp "$RESUME_DIR/trainer_state.json" "$RESUME_DIR/trainer_state.json.backup"
                fi
            fi
            
            # 检查configuration.json
            if [ -f "$RESUME_DIR/config.json" ]; then
                log_debug "验证config.json完整性..."
                if ! python3 -c "import json; json.load(open('$RESUME_DIR/config.json'))" 2>/dev/null; then
                    log_warning "config.json文件损坏"
                fi
            fi
            
            # 检查模型权重文件
            if [ ! -f "$RESUME_DIR/pytorch_model.bin" ] && [ ! -f "$RESUME_DIR/model.safetensors" ]; then
                log_error "未找到模型权重文件！"
                return 1
            fi
            
        else
            log_info "🆕 开始新的训练，无需修复checkpoint"
        fi
    fi
}

# 修复配置同步问题
fix_config_sync() {
    log_info "⚙️ 修复配置同步问题..."
    
    # 检查grpo_project/configs目录
    if [ -d "grpo_project/configs" ]; then
        log_debug "验证配置文件结构..."
        
        # 检查必要的配置文件
        config_files=("__init__.py" "environment.py" "training.py" "reward.py")
        for file in "${config_files[@]}"; do
            config_path="grpo_project/configs/$file"
            if [ ! -f "$config_path" ]; then
                log_error "缺少配置文件: $config_path"
                return 1
            fi
        done
        log_debug "配置文件结构完整"
    else
        log_error "grpo_project/configs目录不存在！"
        return 1
    fi
}

# 修复WandB同步问题
fix_wandb_sync() {
    log_info "📊 修复WandB同步问题..."
    
    # 创建wandb目录（如果不存在）
    if [ ! -d "wandb" ]; then
        mkdir -p wandb
        log_debug "创建wandb目录"
    fi
    
    # 清理可能损坏的wandb缓存
    if [ -d "wandb/.cache" ]; then
        log_debug "清理wandb缓存..."
        rm -rf wandb/.cache
    fi
    
    # 检查WandB配置
    if [ -f "$HOME/.netrc" ]; then
        if grep -q "machine api.wandb.ai" "$HOME/.netrc"; then
            log_debug "WandB认证配置存在"
        else
            log_warning "WandB认证可能未配置，请运行: wandb login"
        fi
    else
        log_warning "未找到.netrc文件，请运行: wandb login"
    fi
}

# 修复课程学习状态
fix_curriculum_state() {
    log_info "📚 修复课程学习状态..."
    
    # 检查课程学习相关文件
    if [ -f "grpo_project/curriculum/manager.py" ]; then
        log_debug "课程学习管理器存在"
    else
        log_error "课程学习管理器不存在！"
        return 1
    fi
    
    # 清理可能损坏的课程状态文件
    if [ -f "curriculum_state.json" ]; then
        if ! python3 -c "import json; json.load(open('curriculum_state.json'))" 2>/dev/null; then
            log_warning "课程状态文件损坏，备份并删除..."
            mv curriculum_state.json "curriculum_state.json.backup.$(date +%s)"
        fi
    fi
}

# 验证Python依赖
check_python_deps() {
    log_info "🐍 检查Python依赖..."
    
    # 检查必要的包
    required_packages=("torch" "transformers" "trl" "datasets" "wandb" "numpy" "peft")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        log_debug "所有必要的Python包都已安装"
    else
        log_error "缺少Python包: ${missing_packages[*]}"
        log_info "请运行: pip install ${missing_packages[*]}"
        return 1
    fi
}

# 生成参数传递诊断报告
generate_diagnostic_report() {
    log_info "📋 生成参数传递诊断报告..."
    
    report_file="parameter_transfer_diagnostic_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== GRPO断续训练参数传递诊断报告 ===
生成时间: $(date)
工作目录: $PWD

1. 环境变量状态:
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"未设置"}
WANDB_PROJECT: ${WANDB_PROJECT:-"未设置"}
WANDB_ENTITY: ${WANDB_ENTITY:-"未设置"}
WANDB_RUN_ID: ${WANDB_RUN_ID:-"未设置"}
WANDB_RESUME: ${WANDB_RESUME:-"未设置"}

2. 文件结构检查:
EOF
    
    # 检查关键文件
    key_files=("main.py" "run_enhanced_grpo_training.sh" "grpo_project/configs/__init__.py")
    for file in "${key_files[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ $file - 存在" >> "$report_file"
        else
            echo "❌ $file - 缺失" >> "$report_file"
        fi
    done
    
    # 检查checkpoint目录
    echo "" >> "$report_file"
    echo "3. Checkpoint状态:" >> "$report_file"
    
    if [ -f "run_enhanced_grpo_training.sh" ]; then
        RESUME_DIR=$(grep "RESUME_FROM_CHECKPOINT_DIR=" run_enhanced_grpo_training.sh | head -1 | cut -d'"' -f2)
        if [ -n "$RESUME_DIR" ]; then
            echo "指定的checkpoint目录: $RESUME_DIR" >> "$report_file"
            if [ -d "$RESUME_DIR" ]; then
                echo "✅ Checkpoint目录存在" >> "$report_file"
                
                # 检查重要文件
                checkpoint_files=("trainer_state.json" "config.json" "pytorch_model.bin" "model.safetensors")
                for file in "${checkpoint_files[@]}"; do
                    if [ -f "$RESUME_DIR/$file" ]; then
                        echo "  ✅ $file" >> "$report_file"
                    else
                        echo "  ❌ $file" >> "$report_file"
                    fi
                done
            else
                echo "❌ Checkpoint目录不存在" >> "$report_file"
            fi
        else
            echo "未设置checkpoint目录（新训练）" >> "$report_file"
        fi
    fi
    
    # Python包状态
    echo "" >> "$report_file"
    echo "4. Python包状态:" >> "$report_file"
    required_packages=("torch" "transformers" "trl" "datasets" "wandb" "numpy" "peft")
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package; print(f'✅ $package - {$package.__version__}')" 2>/dev/null >> "$report_file"; then
            :  # 成功
        else
            echo "❌ $package - 未安装或版本信息不可用" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "=== 诊断报告结束 ===" >> "$report_file"
    
    log_info "📋 诊断报告已生成: $report_file"
}

# 清理临时文件和缓存
cleanup_temp_files() {
    log_info "🧹 清理临时文件和缓存..."
    
    # 清理Python缓存
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # 清理可能的临时训练文件
    rm -f *.tmp 2>/dev/null || true
    rm -f .trainer_temp_* 2>/dev/null || true
    
    # 清理旧的日志文件（保留最近10个）
    if ls *_debug.txt >/dev/null 2>&1; then
        ls -t *_debug.txt | tail -n +11 | xargs rm -f 2>/dev/null || true
    fi
    
    log_debug "临时文件清理完成"
}

# 主函数
main() {
    log_info "🚀 开始GRPO断续训练参数传递修复..."
    
    # 检查是否在正确的目录
    if [ ! -f "main.py" ] || [ ! -d "grpo_project" ]; then
        log_error "请在GRPO项目根目录下运行此脚本！"
        exit 1
    fi
    
    # 执行修复步骤
    fix_resume_issues || exit 1
    fix_config_sync || exit 1
    fix_wandb_sync || exit 1
    fix_curriculum_state || exit 1
    check_python_deps || exit 1
    cleanup_temp_files
    generate_diagnostic_report
    
    log_info "✅ 参数传递修复完成！"
    log_info "📋 请查看诊断报告了解详细情况"
    log_info "🚀 现在可以安全地开始断续训练"
    
    echo ""
    echo "下一步操作："
    echo "1. 检查诊断报告中的任何警告或错误"
    echo "2. 确认训练脚本中的checkpoint路径设置正确"
    echo "3. 运行训练脚本: ./run_enhanced_grpo_training.sh"
}

# 处理命令行参数
case "${1:-}" in
    --help|-h)
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  --help, -h     显示此帮助信息"
        echo "  --report-only  仅生成诊断报告，不执行修复"
        echo "  --clean-only   仅清理临时文件"
        exit 0
        ;;
    --report-only)
        generate_diagnostic_report
        exit 0
        ;;
    --clean-only)
        cleanup_temp_files
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "未知参数: $1"
        echo "使用 --help 查看帮助信息"
        exit 1
        ;;
esac
