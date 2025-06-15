#!/bin/bash
# run_enhanced_grpo_training_v2.sh - 支持新数据集格式的增强GRPO训练脚本

# --- Exit on error ---
set -e
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --- Get the directory where the script is located ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_EXECUTABLE="python3"

# --- 集成的断续训练检查功能 ---
# 设置控制变量
SKIP_SAFETY_CHECK=${SKIP_SAFETY_CHECK:-false}  # 可通过环境变量跳过检查
AUTO_FIX_ISSUES=${AUTO_FIX_ISSUES:-true}       # 自动修复发现的问题

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

# 断续训练安全检查函数
run_safety_checks() {
    log_info "🔍 开始断续训练安全检查..."
    
    local total_issues=0
    local critical_issues=0
    local check_passed=true
    
    # 1. 快速结构检查
    log_debug "检查项目文件结构..."
    local missing_files=""
    
    if [ ! -f "${SCRIPT_DIR}/main.py" ]; then
        missing_files="${missing_files} main.py"
    fi
    
    if [ ! -d "${SCRIPT_DIR}/grpo_project" ]; then
        missing_files="${missing_files} grpo_project/"
    fi
    
    if [ ! -f "${SCRIPT_DIR}/grpo_project/configs/__init__.py" ]; then
        missing_files="${missing_files} grpo_project/configs/"
    fi
    
    if [ -n "$missing_files" ]; then
        log_error "❌ 缺少关键文件:$missing_files"
        critical_issues=$((critical_issues + 1))
        check_passed=false
    else
        log_debug "✅ 项目文件结构完整"
    fi
    
    # 2. Python环境检查
    log_debug "检查Python环境..."
    local missing_packages=""
    
    for package in torch transformers trl datasets wandb numpy peft; do
        if ! ${PYTHON_EXECUTABLE} -c "import $package" 2>/dev/null; then
            missing_packages="${missing_packages} $package"
        fi
    done
    
    if [ -n "$missing_packages" ]; then
        log_error "❌ 缺少Python包:$missing_packages"
        log_error "   请运行: pip install$missing_packages"
        critical_issues=$((critical_issues + 1))
        check_passed=false
    else
        log_debug "✅ Python环境检查通过"
    fi
    
    # 3. 运行详细的Python检查（如果基础检查通过）
    if [ "$check_passed" = true ]; then
        log_debug "运行详细配置检查..."
        
        if [ -f "${SCRIPT_DIR}/quick_resume_check.py" ]; then
            # 使用我们的快速检查工具
            if ${PYTHON_EXECUTABLE} "${SCRIPT_DIR}/quick_resume_check.py" > /tmp/quick_check_output.txt 2>&1; then
                log_debug "✅ 详细配置检查通过"
                
                # 显示重要信息
                if grep -q "🚨\|❌" /tmp/quick_check_output.txt; then
                    log_warning "⚠️ 发现一些问题，但可以继续训练："
                    grep "🚨\|❌" /tmp/quick_check_output.txt | head -3 | while read line; do
                        log_warning "   $line"
                    done
                    total_issues=$((total_issues + 1))
                fi
                
                # 清理临时文件
                rm -f /tmp/quick_check_output.txt
            else
                log_warning "⚠️ 详细检查失败，但基础检查通过，可以继续"
                total_issues=$((total_issues + 1))
                
                # 显示错误信息（简化）
                if [ -f /tmp/quick_check_output.txt ]; then
                    log_debug "检查输出："
                    tail -n 5 /tmp/quick_check_output.txt | while read line; do
                        log_debug "   $line"
                    done
                    rm -f /tmp/quick_check_output.txt
                fi
            fi
        else
            log_debug "ℹ️ 未找到详细检查工具，跳过高级检查"
        fi
        
        # 4. 项目模块导入检查
        log_debug "检查项目模块导入..."
        if ! ${PYTHON_EXECUTABLE} -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
print('✅ 项目模块导入成功')
" 2>/dev/null; then
            log_warning "⚠️ 项目模块导入有问题，但可能仍可继续训练"
            total_issues=$((total_issues + 1))
        else
            log_debug "✅ 项目模块导入成功"
        fi
    fi
    
    # 检查结果汇总
    if [ $critical_issues -gt 0 ]; then
        log_error "🚨 发现 $critical_issues 个严重问题，无法继续训练"
        log_error "   请修复这些问题后重新运行"
        log_info "💡 修复建议："
        log_info "   1. 确保在正确的项目目录下"
        log_info "   2. 安装缺失的Python包"
        log_info "   3. 检查项目文件完整性"
        return 1
    elif [ $total_issues -gt 0 ]; then
        log_warning "⚠️ 发现 $total_issues 个非严重问题，但可以继续训练"
        log_info "💡 建议在训练完成后查看详细日志并修复这些问题"
    else
        log_info "✅ 所有安全检查通过！"
    fi
    
    return 0
}

# 运行自动修复（如果启用）
run_auto_fix() {
    if [ "$AUTO_FIX_ISSUES" = true ]; then
        log_info "🔧 运行自动修复..."
        
        # 清理可能残留的环境变量
        unset WANDB_RUN_ID 2>/dev/null || true
        unset WANDB_RESUME 2>/dev/null || true
        unset WANDB_RUN_NAME 2>/dev/null || true
        
        # 清理Python缓存
        find "${SCRIPT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "${SCRIPT_DIR}" -name "*.pyc" -delete 2>/dev/null || true
        
        # 创建必要的目录
        mkdir -p "${SCRIPT_DIR}/wandb" 2>/dev/null || true
        
        log_debug "✅ 自动修复完成"
    fi
}

# 主检查流程
if [ "$SKIP_SAFETY_CHECK" != true ]; then
    echo ""
    echo "🛡️ =================================================="
    echo "     GRPO断续训练安全检查"
    echo "=================================================="
    
    # 运行自动修复
    run_auto_fix
    
    # 运行安全检查
    if run_safety_checks; then
        log_info "🎯 安全检查完成，继续训练..."
        echo ""
    else
        log_error "💥 安全检查失败，停止训练"
        echo ""
        log_info "🔧 如需跳过检查（不推荐），可设置环境变量："
        log_info "   export SKIP_SAFETY_CHECK=true"
        log_info "   然后重新运行训练脚本"
        exit 1
    fi
else
    log_warning "⚠️ 已跳过安全检查（SKIP_SAFETY_CHECK=true）"
    echo ""
fi

# --- Enhanced Environment Setup ---
export WANDB_PROJECT="VerilogGRPO_Enhanced_8B"
export WANDB_ENTITY="qhy0227-tsinghua-university"

# --- WandB Step Sync Fix ---
# 启用WandB步数同步修复模块，解决step不匹配问题
export WANDB_STEP_FIX_ENABLED=true

# Enable better CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- Distributed Training Parameters ---
NUM_GPUS_PER_NODE=2
MASTER_ADDR="localhost"
MASTER_PORT=$(shuf -i 10000-19999 -n 1)

# --- Enhanced Model and Data Configuration ---
BASE_MODEL_NAME_OR_PATH="/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"
# "/home/qhy/Research/LLM/GRPO-RV/QWEN3-4B"
STAGE1_ADAPTER_PATH="/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/"
# "/home/qhy/Research/LLM/GRPO-RV/GRPO-v4/S1/trainer_output/checkpoint-138795"

# 🔥 重要：使用处理后的增强数据集
# 根据你的数据集处理方式选择一个：
DATASET_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset/all-2.jsonl"
DATASET_BASE_PATH=$(dirname "${DATASET_PATH}")

# --- RESUME FROM CHECKPOINT CONFIGURATION ---
# 🔄 设置此变量为你想要从中恢复的 checkpoint 目录的路径
# 例如: RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_8B_runs/v3-LR6e-6-R16-20250612-175657/checkpoint-540"
# 将此留空以开始新的训练。将其设置为一个不存在的路径也会开始新的训练（会有警告）。
RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_v3_runs/v3-_home_qhy_Research_LLM_GRPO-RV_QWEN3-4B-LR1e-5-R64-20250604-232819-2/checkpoint-144"

# 🔧 关键：WandB恢复配置 + 集成checkpoint验证
if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
    if [ -d "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
        log_info "🔄 检测到checkpoint恢复，开始验证..."
        
        # 集成checkpoint完整性检查
        checkpoint_validation_passed=true
        
        # 检查必要文件
        if [ ! -f "${RESUME_FROM_CHECKPOINT_DIR}/trainer_state.json" ]; then
            log_error "❌ 缺少训练状态文件: trainer_state.json"
            checkpoint_validation_passed=false
        fi
        

        

        
        # 验证JSON文件格式
        if [ -f "${RESUME_FROM_CHECKPOINT_DIR}/trainer_state.json" ]; then
            if ! ${PYTHON_EXECUTABLE} -c "import json; json.load(open('${RESUME_FROM_CHECKPOINT_DIR}/trainer_state.json'))" 2>/dev/null; then
                log_error "❌ trainer_state.json文件格式损坏"
                checkpoint_validation_passed=false
            fi
        fi
        
        if [ -f "${RESUME_FROM_CHECKPOINT_DIR}/config.json" ]; then
            if ! ${PYTHON_EXECUTABLE} -c "import json; json.load(open('${RESUME_FROM_CHECKPOINT_DIR}/config.json'))" 2>/dev/null; then
                log_error "❌ config.json文件格式损坏"
                checkpoint_validation_passed=false
            fi
        fi
        
        if [ "$checkpoint_validation_passed" = false ]; then
            log_error "💥 Checkpoint验证失败，无法继续断续训练"
            log_info "💡 建议："
            log_info "   1. 检查checkpoint目录是否完整"
            log_info "   2. 尝试使用更早的checkpoint"
            log_info "   3. 或者开始新的训练（清空RESUME_FROM_CHECKPOINT_DIR）"
            exit 1
        else
            log_info "✅ Checkpoint验证通过"
        fi
        
        echo "🔄 配置WandB恢复..."
        
        # 🔧 修复：使用全局WandB目录而不是checkpoint目录下的wandb
        GLOBAL_WANDB_DIR="${SCRIPT_DIR}/wandb"
        PARENT_DIR=$(dirname "${RESUME_FROM_CHECKPOINT_DIR}")
        
        # 从checkpoint目录路径提取时间戳
        # 例如: v3-LR6e-6-R64-20250609-100431 -> 20250609-100431
        CHECKPOINT_DIR_NAME=$(basename "${PARENT_DIR}")
        echo "📁 检查点目录名: ${CHECKPOINT_DIR_NAME}"
        
        # 提取时间戳 (格式: YYYYMMDD-HHMMSS)
        TIMESTAMP_PATTERN=$(echo "${CHECKPOINT_DIR_NAME}" | grep -o '[0-9]\{8\}-[0-9]\{6\}')
        
        if [ -n "${TIMESTAMP_PATTERN}" ] && [ -d "${GLOBAL_WANDB_DIR}" ]; then
            echo "🕐 从目录名提取时间戳: ${TIMESTAMP_PATTERN}"
            
            # 将时间戳转换为WandB格式 (YYYYMMDD_HHMMSS)
            WANDB_TIMESTAMP_FORMAT=$(echo "${TIMESTAMP_PATTERN}" | sed 's/-/_/')
            echo "🔍 查找WandB run时间戳: ${WANDB_TIMESTAMP_FORMAT}"
            
            # 首先尝试精确匹配
            MATCHING_RUN_DIR=$(find "${GLOBAL_WANDB_DIR}" -name "run-${WANDB_TIMESTAMP_FORMAT}*" -type d | head -1)
            
            # 如果精确匹配失败，尝试匹配日期和小时分钟（允许秒数差异）
            if [ -z "${MATCHING_RUN_DIR}" ]; then
                # 提取日期和时分 (YYYYMMDD_HHMM)
                DATE_HOUR_MIN=$(echo "${WANDB_TIMESTAMP_FORMAT}" | cut -c1-13)
                echo "🔍 扩展搜索，使用日期+时分: ${DATE_HOUR_MIN}*"
                MATCHING_RUN_DIR=$(find "${GLOBAL_WANDB_DIR}" -name "run-${DATE_HOUR_MIN}*" -type d | head -1)
            fi
            
            if [ -n "${MATCHING_RUN_DIR}" ]; then
                # 提取run ID (格式: run-20250609_100438-xpnrguty -> xpnrguty)
                RUN_ID=$(basename "${MATCHING_RUN_DIR}" | sed 's/run-[0-9]*_[0-9]*-//')
                if [ -n "${RUN_ID}" ]; then
                    export WANDB_RUN_ID="${RUN_ID}"
                    export WANDB_RESUME="must"
                    echo "✅ 找到匹配的WandB run: $(basename "${MATCHING_RUN_DIR}")"
                    echo "✅ 提取到WandB run ID: ${RUN_ID}"
                    echo "✅ 设置WandB恢复模式: must"
                else
                    echo "⚠️ 无法从run目录名提取run ID，将尝试自动恢复"
                    export WANDB_RESUME="allow"
                fi
            else
                echo "⚠️ 未找到匹配时间戳的WandB run目录，查找最近的run..."
                # 备选方案：查找最新的run目录
                LATEST_RUN_DIR=$(find "${GLOBAL_WANDB_DIR}" -name "run-*" -type d | sort | tail -1)
                if [ -n "${LATEST_RUN_DIR}" ]; then
                    RUN_ID=$(basename "${LATEST_RUN_DIR}" | sed 's/run-[0-9]*_[0-9]*-//')
                    if [ -n "${RUN_ID}" ]; then
                        export WANDB_RUN_ID="${RUN_ID}"
                        export WANDB_RESUME="allow"  # 使用allow因为时间戳不匹配
                        echo "✅ 使用最新的WandB run: $(basename "${LATEST_RUN_DIR}")"
                        echo "✅ 提取到WandB run ID: ${RUN_ID}"
                        echo "✅ 设置WandB恢复模式: allow"
                    else
                        echo "⚠️ 无法从最新run目录提取run ID，将尝试自动恢复"
                        export WANDB_RESUME="allow"
                    fi
                else
                    echo "⚠️ 未找到任何WandB run目录，将尝试自动恢复"
                    export WANDB_RESUME="allow"
                fi
            fi
        else
            echo "⚠️ 无法从目录名提取时间戳或WandB目录不存在，将尝试自动恢复"
            export WANDB_RESUME="allow"
        fi
        
        # 修改运行名称以表示这是恢复的训练
        if [ -z "${WANDB_RUN_ID}" ]; then
            export WANDB_RUN_NAME="resumed-${WANDB_RUN_NAME}"
        fi
        
        echo "🔄 WandB恢复配置完成:"
        echo "  - WANDB_RUN_ID: ${WANDB_RUN_ID:-'(自动检测)'}"
        echo "  - WANDB_RESUME: ${WANDB_RESUME}"
        echo "  - WANDB_RUN_NAME: ${WANDB_RUN_NAME}"
        echo "  - 全局WandB目录: ${GLOBAL_WANDB_DIR}"
        
    else
        echo "⚠️ 警告: RESUME_FROM_CHECKPOINT_DIR ('${RESUME_FROM_CHECKPOINT_DIR}') 指定的目录不存在。将开始新的训练，并忽略此设置。"
        # RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_8B_runs/v3-LR6e-6-R16-20250612-175657/checkpoint-540" # 清空以避免传递无效路径给Python脚本
        # 确保不设置恢复相关的环境变量
        unset WANDB_RUN_ID
        unset WANDB_RESUME
    fi
else
    echo "🚀 开始新的训练运行"
    # 确保不设置恢复相关的环境变量
    unset WANDB_RUN_ID
    unset WANDB_RESUME
fi

# 检查数据集文件是否存在
if [ ! -f "${DATASET_PATH}" ]; then
    echo "❌ ERROR: Dataset file not found: ${DATASET_PATH}"
    echo ""
    echo "Please process your dataset first using one of these methods:"
    echo ""
    echo "Method 1 - If you have category folders:"
    echo "  python process_category_folders.py --output enhanced_dataset.jsonl"
    echo ""
    echo "Method 2 - If you have 4 separate files:"
    echo "  python merge_four_files.py"
    echo ""
    echo "Method 3 - Auto-detect and process:"
    echo "  python category_merger.py --auto --output enhanced_dataset.jsonl"
    echo ""
    echo "Then update DATASET_PATH in this script to point to the generated file."
    exit 1
fi

OUTPUT_DIR_BASE="./enhanced_grpo_8B_runs"

# --- Enhanced LoRA Configuration ---
LORA_RANK=16         # Increased capacity
LORA_ALPHA=32        # Scaled with rank  
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

# 🔧 新增：独立的长度配置参数
# 总序列长度（不变）
MAX_SEQ_LENGTH=5120

# 🔧 独立配置prompt和completion长度
# 可以根据需要调整这些值
MAX_PROMPT_LENGTH=1024    # 提示的最大长度 (默认: ~37.5%的总长度)
MAX_COMPLETION_LENGTH=4096 # 输出的最大长度 (默认: ~62.5%的总长度)

# 🔧 长度分配策略选择
# 选项: "balanced" (50/50), "prompt_heavy" (60/40), "completion_heavy" (40/60), "custom" (使用上面的自定义值)
LENGTH_ALLOCATION_STRATEGY="custom"

# 🔧 长度配置预设模板 - 可以直接复制使用
# 如果要使用预设，请取消注释相应的配置并注释掉上面的自定义配置

# 预设1: 平衡分配 (适合一般用途)
# LENGTH_ALLOCATION_STRATEGY="balanced"
# MAX_PROMPT_LENGTH=2048
# MAX_COMPLETION_LENGTH=2048

# 预设2: 长输出模式 (适合需要生成长代码的情况)
# LENGTH_ALLOCATION_STRATEGY="completion_heavy"
# MAX_PROMPT_LENGTH=1280   # ~31%
# MAX_COMPLETION_LENGTH=2816 # ~69%

# 预设3: 超长输出模式 (适合生成非常长的代码)
# LENGTH_ALLOCATION_STRATEGY="custom"
# MAX_PROMPT_LENGTH=1024   # ~25%
# MAX_COMPLETION_LENGTH=3072 # ~75%

# 预设4: 巨型输出模式 (需要更大的总序列长度)
# MAX_SEQ_LENGTH=6144
# LENGTH_ALLOCATION_STRATEGY="custom"
# MAX_PROMPT_LENGTH=1536   # ~25%
# MAX_COMPLETION_LENGTH=4608 # ~75%

# 🔧 验证长度配置
validate_length_config() {
    local total_length=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
    echo "📏 长度配置验证:"
    echo "  - 总序列长度: ${MAX_SEQ_LENGTH}"
    echo "  - 最大提示长度: ${MAX_PROMPT_LENGTH} ($(( MAX_PROMPT_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
    echo "  - 最大输出长度: ${MAX_COMPLETION_LENGTH} ($(( MAX_COMPLETION_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
    echo "  - 总使用长度: ${total_length}"
    echo "  - 分配策略: ${LENGTH_ALLOCATION_STRATEGY}"
    
    if [ ${total_length} -gt ${MAX_SEQ_LENGTH} ]; then
        echo "⚠️ 警告: 提示长度 + 输出长度 (${total_length}) > 总序列长度 (${MAX_SEQ_LENGTH})"
        echo "   建议调整配置或增加MAX_SEQ_LENGTH"
        return 1
    else
        echo "✅ 长度配置有效"
        return 0
    fi
}

# --- Enhanced Callback Configuration ---
CALLBACK_NUM_SAMPLES=2
CALLBACK_EVAL_EVERY_N_STEPS=5

# --- Enhanced Generation Parameters ---
GEN_TEMPERATURE=0.8
GEN_TOP_K=50
GEN_TOP_P=0.95
GEN_REPETITION_PENALTY=1.1
GEN_LENGTH_PENALTY=1.0

# --- Enhanced Multi-Objective Reward Configuration ---
# Basic compilation rewards (enhanced)
REWARD_COMPILATION_SUCCESS=2.0
REWARD_COMPILATION_FAILURE=-4.0
REWARD_SIMULATION_CRASH=-4.0
REWARD_OUTPUT_PARSE_ERROR=-2.0
REWARD_MISSING_CODE_BLOCK_PENALTY=-3.0
REWARD_TIMEOUT_PENALTY=-3.0

# Enhanced functional rewards (non-linear)
REWARD_TEST_PASS_BASE=1.5
REWARD_TEST_PASS_BONUS_MULTIPLIER=1.2
REWARD_MAX_FUNCTIONAL=15.0
REWARD_ALL_TESTS_PASSED_BONUS=5.0

# Code quality rewards (new)
REWARD_CODE_EFFICIENCY_BONUS=2.0
REWARD_CODE_READABILITY_BONUS=1.0
REWARD_CODE_COMPLEXITY_PENALTY=-1.0
REWARD_EDGE_CASE_HANDLING_BONUS=1.5
REWARD_SYNTHESIS_FRIENDLY_BONUS=1.0
REWARD_RESOURCE_USAGE_PENALTY=-0.5

# Multi-objective weights
REWARD_FUNCTIONAL_WEIGHT=0.7
REWARD_EFFICIENCY_WEIGHT=0.15
REWARD_READABILITY_WEIGHT=0.1
REWARD_ROBUSTNESS_WEIGHT=0.05

# Adaptive reward scaling
REWARD_ENABLE_ADAPTIVE_SCALING=false
REWARD_SCALE_FACTOR=1.0
REWARD_CLIPPING_RANGE=20.0

# --- Dual-Layer Curriculum Learning Configuration ---
ENABLE_CURRICULUM=true
CURRICULUM_TYPE="dual_layer"

# 🎯 根据你的数据集分布调整这些设置
# 运行数据集处理脚本后，它会给出推荐配置，复制到这里

CURRICULUM_FOCUS_LEVELS="advanced basic intermediate"
CURRICULUM_COMPLEXITY_EMPHASIS="simple"   # 选项: "simple", "balanced", "complex"

# 🔧 新增：课程学习性能检查间隔配置
# 控制多少步检查一次性能并判断是否可以进阶到下一阶段
# 较小的值(如5)：更频繁检查，响应更快，但计算开销稍大
# 较大的值(如25)：检查较少，节省计算，但响应稍慢
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=5   # 每5步检查一次，更频繁监控

# 如果你的数据集主要是基础级别，使用:
# CURRICULUM_FOCUS_LEVELS="basic intermediate"
# CURRICULUM_COMPLEXITY_EMPHASIS="simple"

# 如果你的数据集主要是高级，使用:
# CURRICULUM_FOCUS_LEVELS="intermediate advanced expert"
# CURRICULUM_COMPLEXITY_EMPHASIS="complex"

# --- Experience Replay Configuration ---
ENABLE_EXPERIENCE_REPLAY=true
EXPERIENCE_BUFFER_SIZE=1000
REPLAY_SAMPLE_RATIO=0.2

# --- Enhanced Training Configuration ---
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=6e-6               # More conservative
NUM_TRAIN_EPOCHS=4
MAX_STEPS=-1
WARMUP_RATIO=0.15                # Increased warmup
LR_SCHEDULER_TYPE="cosine"
WEIGHT_DECAY=0.01
LOGGING_STRATEGY="steps"
LOGGING_STEPS=2
SAVE_STRATEGY="steps"
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=5
SEED=42

# --- Enhanced GRPO Configuration ---
NUM_GENERATIONS_GRPO=2
# 注意: MAX_COMPLETION_LENGTH现在使用上面配置的独立参数

# --- Enhanced Performance Settings ---
BF16_ENABLED=true
FP16_ENABLED=false
GRADIENT_CHECKPOINTING_ENABLED=false
OPTIMIZER_TYPE="adamw_torch"

# --- Enhanced DataLoader Configuration ---
NUM_CPU_CORES=$(nproc --all 2>/dev/null || echo 4)
DATALOADER_NUM_WORKERS=0
# $((NUM_CPU_CORES / NUM_GPUS_PER_NODE / 2))
# if [ "${DATALOADER_NUM_WORKERS}" -lt "1" ]; then DATALOADER_NUM_WORKERS=1; fi
DATALOADER_PIN_MEMORY=true

# --- FSDP Configuration (disabled by default) ---
FSDP_ENABLED=false

# --- Dynamic W&B Run Name ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MODEL_NAME_SLUG=${BASE_MODEL_NAME_OR_PATH//\//_}
export WANDB_RUN_NAME="v3-LR${LEARNING_RATE}-R${LORA_RANK}-${TIMESTAMP}"

# --- Script Backup Configuration ---
# 为了记录实验，我们会将训练脚本复制到输出目录
SCRIPT_BACKUP_ENABLED=true
# 降低课程学习的性能阈值，更容易进阶
CURRICULUM_PERFORMANCE_THRESHOLD_1=0.65  # 基础阶段
CURRICULUM_PERFORMANCE_THRESHOLD_2=0.60  # 初级阶段
CURRICULUM_PERFORMANCE_THRESHOLD_3=0.55  # 中级阶段
CURRICULUM_PERFORMANCE_THRESHOLD_4=0.50  # 高级阶段
CURRICULUM_PERFORMANCE_THRESHOLD_5=0.45  # 专家阶段

# 减少最小评估次数
CURRICULUM_MIN_EVALUATIONS=3

# Qwen3优化的生成参数
GEN_TEMPERATURE=0.7
GEN_TOP_P=0.8
GEN_TOP_K=40
GEN_REPETITION_PENALTY=1.05
# --- Pre-flight Checks ---
echo "========================================================================"
echo "                    ENHANCED GRPO v3 TRAINING"
echo "========================================================================"
echo "Script Directory: ${SCRIPT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Base Model: ${BASE_MODEL_NAME_OR_PATH}"
echo "Stage 1 Adapters: ${STAGE1_ADAPTER_PATH}"
if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ]; then # 再次检查，因为可能在上面被清空
    echo "🔄 将从 Checkpoint 恢复训练: ${RESUME_FROM_CHECKPOINT_DIR}"
else
    echo "🚀 开始新的训练运行。"
fi
echo "Output Directory Base: ${OUTPUT_DIR_BASE}" # 注意: 脚本中 OUTPUT_DIR_BASE 的声明在其首次使用之后，建议移到前面

# 🔧 验证长度配置
echo ""
validate_length_config
if [ $? -ne 0 ]; then
    echo "❌ 长度配置验证失败，请检查配置后重新运行"
    exit 1
fi
echo ""

# Check dataset format
echo "🔍 Checking dataset format..."
FIRST_LINE=$(head -1 "${DATASET_PATH}")
if echo "${FIRST_LINE}" | grep -q '"level"' && echo "${FIRST_LINE}" | grep -q '"complexity_score"'; then
    echo "✅ Detected enhanced dataset format with level and complexity_score"
    
    # Extract some statistics
    echo "📊 Dataset preview:"
    python3 -c "
import json
data = []
with open('${DATASET_PATH}') as f:
    for i, line in enumerate(f):
        if i >= 100: break  # Sample first 100 lines
        data.append(json.loads(line))

if data:
    levels = [d.get('level', 'unknown') for d in data]
    complexities = [d.get('complexity_score', 0) for d in data]
    categories = [d.get('category', 'Unknown') for d in data]
    
    print(f'  📈 Sample size: {len(data)} (from first 100 lines)')
    
    print(f'  📊 Level distribution:')
    for level in sorted(set(levels)):
        count = levels.count(level)
        print(f'    {level}: {count} ({count/len(levels)*100:.1f}%)')
    
    if complexities:
        print(f'  🧮 Complexity range: {min(complexities):.1f} - {max(complexities):.1f}')
    
    print(f'  📂 Categories: {len(set(categories))} unique')
    for cat in sorted(set(categories))[:5]:  # Show top 5
        count = categories.count(cat)
        print(f'    {cat}: {count}')
    if len(set(categories)) > 5:
        print(f'    ... and {len(set(categories)) - 5} more categories')
"
else
    echo "⚠️  WARNING: Dataset may be in legacy format (missing level/complexity_score)"
    echo "   The training script will attempt to auto-upgrade the dataset"
    echo "   For best results, consider processing your dataset first:"
    echo "   python process_category_folders.py ${DATASET_PATH} --output enhanced_${DATASET_PATH}"
fi

echo ""

# Check file paths in dataset
echo "🔍 Checking file paths in dataset..."
python3 -c "
import json
import os

with open('${DATASET_PATH}') as f:
    sample = json.loads(f.readline())

ref_path = sample.get('reference_verilog_path', '')
tb_path = sample.get('testbench_path', '')

print(f'  📄 Sample reference path: {ref_path}')
print(f'  📄 Sample testbench path: {tb_path}')

ref_exists = os.path.exists(ref_path)
tb_exists = os.path.exists(tb_path)

print(f'  ✅ Reference file exists: {ref_exists}')
print(f'  ✅ Testbench file exists: {tb_exists}')

if not ref_exists and not tb_exists:
    print('  ⚠️  WARNING: Sample files not found. Check your working directory.')
    print('     Make sure you are running from the correct directory where the')
    print('     category folders or reference files are located.')
"

echo ""
echo "Enhanced Features Summary:"
echo "  ✅ Multi-objective reward system with detailed component tracking"
echo "  ✅ Dual-layer curriculum learning: ${CURRICULUM_TYPE} (${CURRICULUM_FOCUS_LEVELS})"
echo "  ✅ Complexity emphasis: ${CURRICULUM_COMPLEXITY_EMPHASIS}"
echo "  ✅ Curriculum performance check interval: every ${CURRICULUM_PERFORMANCE_CHECK_INTERVAL} steps"
echo "  ✅ Experience replay: ${ENABLE_EXPERIENCE_REPLAY} (buffer size: ${EXPERIENCE_BUFFER_SIZE})"
echo "  ✅ Adaptive reward scaling: ${REWARD_ENABLE_ADAPTIVE_SCALING}"
echo "  ✅ Enhanced LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  ✅ Conservative learning: lr=${LEARNING_RATE}, warmup=${WARMUP_RATIO}"
echo "  ✅ Generation diversity: temp=${GEN_TEMPERATURE}, rep_penalty=${GEN_REPETITION_PENALTY}"
echo "  ✅ Enhanced monitoring: ${CALLBACK_NUM_SAMPLES} samples every ${CALLBACK_EVAL_EVERY_N_STEPS} steps"

echo ""
mkdir -p ${OUTPUT_DIR_BASE}

# --- Script Backup Function ---
backup_training_scripts() {
    local output_dir="$1"
    if [ "$SCRIPT_BACKUP_ENABLED" = true ] && [ -n "$output_dir" ]; then
        echo "📄 正在备份训练脚本到实验文件夹..."
        
        # 创建脚本备份目录
        local script_backup_dir="${output_dir}/training_scripts_backup"
        mkdir -p "$script_backup_dir"
        
        # 复制主要训练脚本
        if [ -f "${BASH_SOURCE[0]}" ]; then
            cp "${BASH_SOURCE[0]}" "$script_backup_dir/"
            echo "  ✅ 复制训练脚本: $(basename ${BASH_SOURCE[0]})"
        fi
        
        # 复制Python主脚本
        if [ -f "$PYTHON_SCRIPT_TO_RUN" ]; then
            cp "$PYTHON_SCRIPT_TO_RUN" "$script_backup_dir/"
            echo "  ✅ 复制Python脚本: $(basename $PYTHON_SCRIPT_TO_RUN)"
        fi
        
        # 复制其他相关Python文件（如果存在）
        for py_file in "${SCRIPT_DIR}"/*.py; do
            if [ -f "$py_file" ] && [ "$py_file" != "$PYTHON_SCRIPT_TO_RUN" ]; then
                cp "$py_file" "$script_backup_dir/"
                echo "  ✅ 复制相关脚本: $(basename $py_file)"
            fi
        done
        
        # 特别复制WandB步数修复模块
        if [ -f "${SCRIPT_DIR}/wandb_step_fix.py" ]; then
            cp "${SCRIPT_DIR}/wandb_step_fix.py" "$script_backup_dir/"
            echo "  ✅ 复制WandB修复模块: wandb_step_fix.py"
        fi
        
        # 保存训练参数到配置文件
        local config_file="${script_backup_dir}/training_config_${TIMESTAMP}.txt"
        cat > "$config_file" << EOF
# Enhanced GRPO v3 Training Configuration
# Generated at: $(date)
# Script: ${BASH_SOURCE[0]}

## Model Configuration
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}"
STAGE1_ADAPTER_PATH="${STAGE1_ADAPTER_PATH}"
DATASET_PATH="${DATASET_PATH}"

## LoRA Configuration
LORA_RANK=${LORA_RANK}
LORA_ALPHA=${LORA_ALPHA}
LORA_DROPOUT=${LORA_DROPOUT}
LORA_TARGET_MODULES="${LORA_TARGET_MODULES}"

## Training Configuration
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
LEARNING_RATE=${LEARNING_RATE}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}
MAX_STEPS=${MAX_STEPS}
WARMUP_RATIO=${WARMUP_RATIO}
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}"
WEIGHT_DECAY=${WEIGHT_DECAY}

## GRPO Configuration
NUM_GENERATIONS_GRPO=${NUM_GENERATIONS_GRPO}
MAX_COMPLETION_LENGTH_GRPO=${MAX_COMPLETION_LENGTH_GRPO}

## Curriculum Learning
ENABLE_CURRICULUM=${ENABLE_CURRICULUM}
CURRICULUM_TYPE="${CURRICULUM_TYPE}"
CURRICULUM_FOCUS_LEVELS="${CURRICULUM_FOCUS_LEVELS}"
CURRICULUM_COMPLEXITY_EMPHASIS="${CURRICULUM_COMPLEXITY_EMPHASIS}"
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=${CURRICULUM_PERFORMANCE_CHECK_INTERVAL}

## Experience Replay
ENABLE_EXPERIENCE_REPLAY=${ENABLE_EXPERIENCE_REPLAY}
EXPERIENCE_BUFFER_SIZE=${EXPERIENCE_BUFFER_SIZE}
REPLAY_SAMPLE_RATIO=${REPLAY_SAMPLE_RATIO}

## Generation Parameters
GEN_TEMPERATURE=${GEN_TEMPERATURE}
GEN_TOP_K=${GEN_TOP_K}
GEN_TOP_P=${GEN_TOP_P}
GEN_REPETITION_PENALTY=${GEN_REPETITION_PENALTY}

## WandB Configuration
WANDB_PROJECT="${WANDB_PROJECT}"
WANDB_ENTITY="${WANDB_ENTITY}"
WANDB_RUN_NAME="${WANDB_RUN_NAME}"
WANDB_STEP_FIX_ENABLED="${WANDB_STEP_FIX_ENABLED}"

## Resume Configuration
RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_8B_runs/v3-LR6e-6-R16-20250612-175657/checkpoint-540"

## Runtime Information
TIMESTAMP=${TIMESTAMP}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}
EOF
        
        echo "  ✅ 保存训练配置: $(basename $config_file)"
        
        # 保存完整的命令行
        echo "$FULL_CMD" > "${script_backup_dir}/full_command_${TIMESTAMP}.txt"
        echo "  ✅ 保存完整命令: full_command_${TIMESTAMP}.txt"
        
        # 创建实验说明文件
        local experiment_info="${script_backup_dir}/experiment_info_${TIMESTAMP}.md"
        cat > "$experiment_info" << EOF
# Enhanced GRPO v3 Training Experiment

## 实验信息
- **开始时间**: $(date)
- **实验名称**: ${WANDB_RUN_NAME}
- **脚本版本**: Enhanced GRPO v3
- **数据集**: ${DATASET_PATH}
- **基础模型**: ${BASE_MODEL_NAME_OR_PATH}

## 关键配置
- **学习率**: ${LEARNING_RATE}
- **LoRA Rank**: ${LORA_RANK}
- **批次大小**: ${PER_DEVICE_TRAIN_BATCH_SIZE}
- **梯度累积步数**: ${GRADIENT_ACCUMULATION_STEPS}
- **课程学习**: ${ENABLE_CURRICULUM}
- **经验回放**: ${ENABLE_EXPERIENCE_REPLAY}
- **WandB步数修复**: ${WANDB_STEP_FIX_ENABLED}

## 文件说明
- \`$(basename ${BASH_SOURCE[0]})\`: 训练启动脚本
- \`$(basename $PYTHON_SCRIPT_TO_RUN)\`: Python主训练脚本  
- \`training_config_${TIMESTAMP}.txt\`: 详细训练参数
- \`full_command_${TIMESTAMP}.txt\`: 完整执行命令
- \`experiment_info_${TIMESTAMP}.md\`: 本实验说明文件

**注意**: 所有训练脚本已自动备份到与模型权重相同的目录中，便于追溯和复现实验。

## 监控链接
- **WandB项目**: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}
- **运行页面**: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${WANDB_RUN_NAME}

## 技术说明

### WandB步数同步修复
本实验使用了WandB步数同步修复模块 (\`wandb_step_fix.py\`)，解决了以下问题：
1. **多回调日志冲突**: 不同的训练回调可能在不同时间记录同一步数的指标
2. **步数不一致**: 训练器的global_step与回调内部计数器不同步
3. **异步日志乱序**: 某些指标可能异步记录，导致步数倒退警告

修复策略：
- 统一步数管理器确保所有日志使用一致的步数
- 按优先级缓冲日志，避免冲突
- 批量提交减少WandB服务器压力

如果仍然看到步数警告，这是正常的 - 修复模块会自动处理这些冲突。

## 备注
请在训练完成后更新此文件，记录实验结果和发现。
EOF
        
        echo "  ✅ 创建实验说明: $(basename $experiment_info)"
        echo "📄 脚本备份完成，保存到: $script_backup_dir"
        
        return 0
    else
        echo "⏭️  跳过脚本备份 (SCRIPT_BACKUP_ENABLED=${SCRIPT_BACKUP_ENABLED})"
        return 1
    fi
}

# --- Build Command Arguments ---
CMD_ARGS=""

# EnvConfig
CMD_ARGS="${CMD_ARGS} --hf_endpoint \"https://hf-mirror.com\""
CMD_ARGS="${CMD_ARGS} --http_proxy \"http://10.130.148.206:7890\""
CMD_ARGS="${CMD_ARGS} --https_proxy \"http://10.130.148.206:7890\""
CMD_ARGS="${CMD_ARGS} --wandb_project \"${WANDB_PROJECT}\""
CMD_ARGS="${CMD_ARGS} --wandb_entity \"${WANDB_ENTITY}\""
CMD_ARGS="${CMD_ARGS} --wandb_run_name_prefix \"enhanced-v3-${MODEL_NAME_SLUG}\""

# Enhanced ScriptConfig
CMD_ARGS="${CMD_ARGS} --model_name_or_path \"${BASE_MODEL_NAME_OR_PATH}\""
CMD_ARGS="${CMD_ARGS} --stage1_adapter_path \"${STAGE1_ADAPTER_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_path \"${DATASET_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_base_path "${DATASET_BASE_PATH}""
CMD_ARGS="${CMD_ARGS} --output_dir_base \"${OUTPUT_DIR_BASE}\""
CMD_ARGS="${CMD_ARGS} --lora_rank ${LORA_RANK}"
CMD_ARGS="${CMD_ARGS} --lora_alpha ${LORA_ALPHA}"
CMD_ARGS="${CMD_ARGS} --lora_dropout ${LORA_DROPOUT}"
if [ -n "${LORA_TARGET_MODULES}" ]; then
    CMD_ARGS="${CMD_ARGS} --lora_target_modules ${LORA_TARGET_MODULES}"
fi
CMD_ARGS="${CMD_ARGS} --max_seq_length ${MAX_SEQ_LENGTH}"
CMD_ARGS="${CMD_ARGS} --script_max_prompt_length ${MAX_PROMPT_LENGTH}"
CMD_ARGS="${CMD_ARGS} --script_max_completion_length ${MAX_COMPLETION_LENGTH}"
CMD_ARGS="${CMD_ARGS} --length_allocation_strategy ${LENGTH_ALLOCATION_STRATEGY}"
CMD_ARGS="${CMD_ARGS} --callback_num_samples ${CALLBACK_NUM_SAMPLES}"
CMD_ARGS="${CMD_ARGS} --callback_eval_every_n_steps ${CALLBACK_EVAL_EVERY_N_STEPS}"
CMD_ARGS="${CMD_ARGS} --gen_temperature ${GEN_TEMPERATURE}"
CMD_ARGS="${CMD_ARGS} --gen_top_k ${GEN_TOP_K}"
CMD_ARGS="${CMD_ARGS} --gen_top_p ${GEN_TOP_P}"
CMD_ARGS="${CMD_ARGS} --gen_repetition_penalty ${GEN_REPETITION_PENALTY}"
CMD_ARGS="${CMD_ARGS} --gen_length_penalty ${GEN_LENGTH_PENALTY}"

# Enhanced Curriculum Learning
CMD_ARGS="${CMD_ARGS} --enable_curriculum ${ENABLE_CURRICULUM}"
CMD_ARGS="${CMD_ARGS} --curriculum_type ${CURRICULUM_TYPE}"
CMD_ARGS="${CMD_ARGS} --curriculum_focus_levels ${CURRICULUM_FOCUS_LEVELS}"
CMD_ARGS="${CMD_ARGS} --curriculum_complexity_emphasis ${CURRICULUM_COMPLEXITY_EMPHASIS}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_check_interval ${CURRICULUM_PERFORMANCE_CHECK_INTERVAL}"

# Experience Replay
CMD_ARGS="${CMD_ARGS} --enable_experience_replay ${ENABLE_EXPERIENCE_REPLAY}"
CMD_ARGS="${CMD_ARGS} --experience_buffer_size ${EXPERIENCE_BUFFER_SIZE}"
CMD_ARGS="${CMD_ARGS} --replay_sample_ratio ${REPLAY_SAMPLE_RATIO}"

# Enhanced RewardConfig
CMD_ARGS="${CMD_ARGS} --compilation_success ${REWARD_COMPILATION_SUCCESS}"
CMD_ARGS="${CMD_ARGS} --compilation_failure ${REWARD_COMPILATION_FAILURE}"
CMD_ARGS="${CMD_ARGS} --simulation_crash ${REWARD_SIMULATION_CRASH}"
CMD_ARGS="${CMD_ARGS} --output_parse_error ${REWARD_OUTPUT_PARSE_ERROR}"
CMD_ARGS="${CMD_ARGS} --missing_code_block_penalty ${REWARD_MISSING_CODE_BLOCK_PENALTY}"
CMD_ARGS="${CMD_ARGS} --timeout_penalty ${REWARD_TIMEOUT_PENALTY}"

# Enhanced functional rewards
CMD_ARGS="${CMD_ARGS} --test_pass_base_reward ${REWARD_TEST_PASS_BASE}"
CMD_ARGS="${CMD_ARGS} --test_pass_bonus_multiplier ${REWARD_TEST_PASS_BONUS_MULTIPLIER}"
CMD_ARGS="${CMD_ARGS} --max_functional_reward ${REWARD_MAX_FUNCTIONAL}"
CMD_ARGS="${CMD_ARGS} --all_tests_passed_bonus ${REWARD_ALL_TESTS_PASSED_BONUS}"

# Code quality rewards
CMD_ARGS="${CMD_ARGS} --code_efficiency_bonus ${REWARD_CODE_EFFICIENCY_BONUS}"
CMD_ARGS="${CMD_ARGS} --code_readability_bonus ${REWARD_CODE_READABILITY_BONUS}"
CMD_ARGS="${CMD_ARGS} --code_complexity_penalty ${REWARD_CODE_COMPLEXITY_PENALTY}"
CMD_ARGS="${CMD_ARGS} --edge_case_handling_bonus ${REWARD_EDGE_CASE_HANDLING_BONUS}"
CMD_ARGS="${CMD_ARGS} --synthesis_friendly_bonus ${REWARD_SYNTHESIS_FRIENDLY_BONUS}"
CMD_ARGS="${CMD_ARGS} --resource_usage_penalty ${REWARD_RESOURCE_USAGE_PENALTY}"

# Multi-objective weights
CMD_ARGS="${CMD_ARGS} --functional_weight ${REWARD_FUNCTIONAL_WEIGHT}"
CMD_ARGS="${CMD_ARGS} --efficiency_weight ${REWARD_EFFICIENCY_WEIGHT}"
CMD_ARGS="${CMD_ARGS} --readability_weight ${REWARD_READABILITY_WEIGHT}"
CMD_ARGS="${CMD_ARGS} --robustness_weight ${REWARD_ROBUSTNESS_WEIGHT}"

# Adaptive scaling
CMD_ARGS="${CMD_ARGS} --enable_adaptive_scaling ${REWARD_ENABLE_ADAPTIVE_SCALING}"
CMD_ARGS="${CMD_ARGS} --reward_scale_factor ${REWARD_SCALE_FACTOR}"
CMD_ARGS="${CMD_ARGS} --reward_clipping_range ${REWARD_CLIPPING_RANGE}"

# Enhanced GRPOConfig (TrainingArguments)

CMD_ARGS="${CMD_ARGS} --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}"
CMD_ARGS="${CMD_ARGS} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
CMD_ARGS="${CMD_ARGS} --learning_rate ${LEARNING_RATE}"
CMD_ARGS="${CMD_ARGS} --num_train_epochs ${NUM_TRAIN_EPOCHS}"
if [ ${MAX_STEPS} -gt 0 ]; then
    CMD_ARGS="${CMD_ARGS} --max_steps ${MAX_STEPS}"
fi
CMD_ARGS="${CMD_ARGS} --warmup_ratio ${WARMUP_RATIO}"
CMD_ARGS="${CMD_ARGS} --lr_scheduler_type \"${LR_SCHEDULER_TYPE}\""
CMD_ARGS="${CMD_ARGS} --weight_decay ${WEIGHT_DECAY}"
CMD_ARGS="${CMD_ARGS} --logging_strategy \"${LOGGING_STRATEGY}\""
CMD_ARGS="${CMD_ARGS} --logging_steps ${LOGGING_STEPS}"
CMD_ARGS="${CMD_ARGS} --save_strategy \"${SAVE_STRATEGY}\""
CMD_ARGS="${CMD_ARGS} --save_steps ${SAVE_STEPS}"
CMD_ARGS="${CMD_ARGS} --save_total_limit ${SAVE_TOTAL_LIMIT}"
CMD_ARGS="${CMD_ARGS} --report_to \"wandb\""
CMD_ARGS="${CMD_ARGS} --remove_unused_columns False"
CMD_ARGS="${CMD_ARGS} --num_generations ${NUM_GENERATIONS_GRPO}"
CMD_ARGS="${CMD_ARGS} --max_prompt_length ${MAX_PROMPT_LENGTH}"  # GRPO参数：最大提示长度
CMD_ARGS="${CMD_ARGS} --max_completion_length ${MAX_COMPLETION_LENGTH}"  # GRPO参数：最大完成长度
CMD_ARGS="${CMD_ARGS} --optim \"${OPTIMIZER_TYPE}\""
CMD_ARGS="${CMD_ARGS} --ddp_find_unused_parameters False"
CMD_ARGS="${CMD_ARGS} --seed ${SEED}"
CMD_ARGS="${CMD_ARGS} --dataloader_num_workers ${DATALOADER_NUM_WORKERS}"
if [ "$DATALOADER_PIN_MEMORY" = true ]; then
    CMD_ARGS="${CMD_ARGS} --dataloader_pin_memory"
fi
# 在构建其他参数之后，添加 resume_from_checkpoint 参数
if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ] && [ -d "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
    CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint \"${RESUME_FROM_CHECKPOINT_DIR}\""
fi

# Enhanced performance settings
if [ "$BF16_ENABLED" = true ]; then CMD_ARGS="${CMD_ARGS} --bf16"; fi
if [ "$FP16_ENABLED" = true ] && [ "$BF16_ENABLED" = false ]; then CMD_ARGS="${CMD_ARGS} --fp16"; fi
if [ "$GRADIENT_CHECKPOINTING_ENABLED" = true ]; then
    CMD_ARGS="${CMD_ARGS} --gradient_checkpointing"
fi

# Enhanced cache directory setup
CACHE_DIR_BASE="${SCRIPT_DIR}/.enhanced_cache_v2"
mkdir -p "${CACHE_DIR_BASE}/datasets"
mkdir -p "${CACHE_DIR_BASE}/models"
CMD_ARGS="${CMD_ARGS} --cache_dir \"${CACHE_DIR_BASE}/models\""

# --- Enhanced Training Execution ---
LAUNCHER=""
PYTHON_SCRIPT_TO_RUN="${SCRIPT_DIR}/main.py" # Updated to use main.py

if [ ${NUM_GPUS_PER_NODE} -gt 1 ] || [ "$FSDP_ENABLED" = true ]; then
  LAUNCHER="torchrun \
    --nproc_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}"
  
  if [ "$FSDP_ENABLED" = true ]; then
    CMD_ARGS="${CMD_ARGS} --fsdp \"full_shard\""
  fi
  FULL_CMD="${LAUNCHER} ${PYTHON_SCRIPT_TO_RUN} ${CMD_ARGS}"
else
  FULL_CMD="${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT_TO_RUN} ${CMD_ARGS}"
fi

echo "========================================================================"
echo "                        STARTING ENHANCED TRAINING"
echo "========================================================================"

# 集成的训练状态摘要
log_info "🎯 训练配置摘要："
log_info "   模型: $(basename "${BASE_MODEL_NAME_OR_PATH}")"
log_info "   数据集: $(basename "${DATASET_PATH}")"
log_info "   最大序列长度: ${MAX_SEQ_LENGTH}"
log_info "   Prompt长度: ${MAX_PROMPT_LENGTH} ($(( MAX_PROMPT_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
log_info "   Completion长度: ${MAX_COMPLETION_LENGTH} ($(( MAX_COMPLETION_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
log_info "   LoRA配置: rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"

if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ] && [ -d "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
    log_info "   训练模式: 🔄 断续训练 (从 $(basename "${RESUME_FROM_CHECKPOINT_DIR}") 恢复)"
    if [ -n "${WANDB_RUN_ID}" ]; then
        log_info "   WandB恢复: ✅ Run ID: ${WANDB_RUN_ID}"
    else
        log_info "   WandB恢复: ⚠️ 自动检测模式"
    fi
else
    log_info "   训练模式: 🆕 新训练"
fi

log_info "   输出目录: ${OUTPUT_DIR_BASE}"
log_info "   WandB项目: ${WANDB_PROJECT}"

echo ""
echo "Training will begin with curriculum stage 0..."
echo "Monitor progress at: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "========================================================================"

# Save full command to file for debugging
echo "${FULL_CMD}" > "${OUTPUT_DIR_BASE}/full_training_command.txt"
echo "Full command saved to: ${OUTPUT_DIR_BASE}/full_training_command.txt"

# --- Backup Training Scripts (Dynamic) ---
# 创建一个延迟备份函数，在训练开始后查找实际输出目录
backup_scripts_to_model_dir() {
    echo "🔍 正在查找实际的模型保存目录..."
    
    # 等待一小段时间让Python脚本创建目录
    sleep 5
    
    # 查找最新创建的训练目录
    ACTUAL_OUTPUT_DIR=""
    if [ -d "${OUTPUT_DIR_BASE}" ]; then
        # 查找包含当前时间戳的目录（最近5分钟内创建的）
        ACTUAL_OUTPUT_DIR=$(find "${OUTPUT_DIR_BASE}" -maxdepth 1 -type d -name "*${TIMESTAMP}*" -newermt "5 minutes ago" | head -1)
        
        # 如果没找到带时间戳的，查找最新的目录
        if [ -z "${ACTUAL_OUTPUT_DIR}" ]; then
            ACTUAL_OUTPUT_DIR=$(find "${OUTPUT_DIR_BASE}" -maxdepth 1 -type d -newer "${OUTPUT_DIR_BASE}/full_training_command.txt" 2>/dev/null | head -1)
        fi
        
        # 最后备选：查找最新修改的目录
        if [ -z "${ACTUAL_OUTPUT_DIR}" ]; then
            ACTUAL_OUTPUT_DIR=$(ls -dt "${OUTPUT_DIR_BASE}"/*/ 2>/dev/null | head -1 | sed 's/\/$//')
        fi
    fi
    
    if [ -n "${ACTUAL_OUTPUT_DIR}" ] && [ -d "${ACTUAL_OUTPUT_DIR}" ]; then
        echo "✅ 找到实际输出目录: ${ACTUAL_OUTPUT_DIR}"
        backup_training_scripts "${ACTUAL_OUTPUT_DIR}"
        
        # 创建一个符号链接到基础目录，方便查找
        LINK_NAME="${OUTPUT_DIR_BASE}/latest_run"
        if [ -L "${LINK_NAME}" ]; then
            rm "${LINK_NAME}"
        fi
        ln -s "$(basename "${ACTUAL_OUTPUT_DIR}")" "${LINK_NAME}"
        echo "✅ 创建符号链接: ${LINK_NAME} -> $(basename "${ACTUAL_OUTPUT_DIR}")"
    else
        echo "⚠️ 未找到实际输出目录，使用基础目录备份"
        backup_training_scripts "${OUTPUT_DIR_BASE}"
    fi
}

# 启动后台备份任务
backup_scripts_to_model_dir &
BACKUP_PID=$!

# --- Execute Training ---
echo "Starting enhanced GRPO v2 training at $(date)..."

# Trap to handle interruption
trap 'echo "Training interrupted at $(date). Cleaning up..."; exit 130' INT TERM

eval "${FULL_CMD}"

status=$?

# 等待备份任务完成
if [ -n "${BACKUP_PID}" ]; then
    echo "⏳ 等待脚本备份任务完成..."
    wait ${BACKUP_PID} 2>/dev/null || true
    echo "✅ 脚本备份任务已完成"
fi

# --- Post-training Summary ---
echo ""
echo "========================================================================"
echo "                      TRAINING COMPLETION SUMMARY"
echo "========================================================================"
echo "Training finished at: $(date)"
echo "Exit status: ${status}"

if [ $status -eq 0 ]; then
    echo "✅ Enhanced GRPO v2 training completed successfully!"
    
    # Check for outputs
    if [ -d "${OUTPUT_DIR_BASE}" ]; then
        FINAL_MODEL_DIR=$(find "${OUTPUT_DIR_BASE}" -name "*final*model*" -type d 2>/dev/null | head -1)
        if [ -n "${FINAL_MODEL_DIR}" ]; then
            echo "✅ Final enhanced model saved to: ${FINAL_MODEL_DIR}"
            MODEL_SIZE=$(du -sh "${FINAL_MODEL_DIR}" 2>/dev/null | cut -f1 || echo "Unknown")
            echo "  Model size: ${MODEL_SIZE}"
        fi
        
        # Check for enhanced artifacts
        ARTIFACTS_DIR=$(find "${OUTPUT_DIR_BASE}" -name "enhanced_artifacts" -type d 2>/dev/null | head -1)
        if [ -d "${ARTIFACTS_DIR}" ]; then
            echo "✅ Enhanced training artifacts saved:"
            find "${ARTIFACTS_DIR}" -name "*.json" -exec basename {} \; 2>/dev/null | sed 's/^/    - /' || echo "    - (artifacts found but couldn't list)"
        fi
        
        # Check logs
        LOG_FILE=$(find "${OUTPUT_DIR_BASE}" -name "*training_log.txt" 2>/dev/null | head -1)
        if [ -f "${LOG_FILE}" ]; then
            LOG_SIZE=$(wc -l < "${LOG_FILE}" 2>/dev/null || echo "0")
            echo "✅ Training log: ${LOG_SIZE} lines"
            
            # Extract final metrics
            if command -v grep &> /dev/null && [ -f "${LOG_FILE}" ]; then
                FINAL_LOSS=$(grep -o "train_loss[^,]*" "${LOG_FILE}" | tail -1 | cut -d':' -f2 | tr -d ' ' 2>/dev/null || echo "N/A")
                echo "  Final training loss: ${FINAL_LOSS}"
            fi
        fi
    fi
    
    echo ""
    echo "🎉 Next steps:"
    echo "1. Check W&B dashboard for detailed metrics and curriculum progression"
    echo "2. Evaluate the model on your test set"
    echo "3. Compare with baseline model performance"
    echo "4. Consider fine-tuning hyperparameters based on the results"
    
else
    echo "❌ Enhanced training failed with exit code ${status}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Check the training log: ${OUTPUT_DIR_BASE}/*/enhanced_training_log.txt"
    echo "2. Verify dataset format: head -1 ${DATASET_PATH} | python -m json.tool"
    echo "3. Check file paths: ensure verilog and testbench files are accessible"
    echo "4. Review GPU memory usage (consider reducing batch size)"
    echo "5. Check W&B logs for detailed error information"
fi

echo "========================================================================"

# Clean up
if command -v nvidia-smi &> /dev/null; then
    echo "Final GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -n ${NUM_GPUS_PER_NODE}
fi

exit ${status}

# ====================================================================
# 集成安全检查功能说明
# ====================================================================
#
# 本脚本已集成断续训练安全检查功能，包括：
#
# 🛡️ 自动安全检查:
#   - 项目文件结构验证
#   - Python环境和依赖检查  
#   - 配置一致性验证
#   - Checkpoint完整性检查
#   - 项目模块导入测试
#
# 🔧 自动修复功能:
#   - 清理残留环境变量
#   - 清理Python缓存
#   - 创建必要目录
#
# 📊 智能摘要显示:
#   - 训练配置概览
#   - 长度配置比例
#   - 恢复状态信息
#
# 🎛️ 控制选项:
#   export SKIP_SAFETY_CHECK=true     # 跳过安全检查（不推荐）
#   export AUTO_FIX_ISSUES=false      # 禁用自动修复
#
# 🚀 使用方法:
#   1. 设置RESUME_FROM_CHECKPOINT_DIR（如需断续训练）
#   2. 直接运行: ./run_enhanced_grpo_training.sh
#   3. 脚本会自动检查并修复常见问题
#   4. 检查通过后自动开始训练
#
# 💡 如果遇到问题:
#   1. 查看检查输出的错误信息
#   2. 使用独立工具详细诊断: python3 quick_resume_check.py
#   3. 运行完整修复: ./cleanup_before_training.sh
#   4. 重置课程状态: python3 fix_curriculum_sync.py --create-fresh
#
# ====================================================================

RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_8B_runs/v3-LR6e-6-R16-20250612-175657/checkpoint-540"

RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_8B_runs/v3-LR6e-6-R16-20250612-175657/checkpoint-540"
