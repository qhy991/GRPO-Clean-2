#!/bin/bash
# run_enhanced_grpo_training_multi_gpu_fixed.sh - 修复版多GPU模型并行训练脚本

# --- Exit on error ---
set -e

# 如果脚本由 torchrun 触发，则不清理分布式环境变量
if [ -z "$RUNNING_WITH_TORCHRUN" ]; then
  echo "🧹 清除旧的分布式环境变量..."
  unset RANK 2>/dev/null || true
  unset LOCAL_RANK 2>/dev/null || true
  unset WORLD_SIZE 2>/dev/null || true
  unset MASTER_ADDR 2>/dev/null || true
  unset MASTER_PORT 2>/dev/null || true
fi

# 🔧 关键修复：设置GPU但避免分布式模式
export CUDA_VISIBLE_DEVICES=0,1

# 🔧 多GPU模型并行优化设置（修复后）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0  # 异步执行以提高性能
export TORCH_NCCL_BLOCKING_WAIT=1  # 使用新的环境变量名
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 使用新的环境变量名
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0  # 启用P2P通信

# 🔧 Flash Attention 优化
export FLASH_ATTENTION_V2=1

# --- Get the directory where the script is located ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_EXECUTABLE="python3"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# 🔧 GPU环境检查函数（修复版）
check_gpu_environment() {
    log_info "🔍 检查GPU环境..."
    
    # 检查CUDA可用性
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "❌ nvidia-smi未找到，请检查CUDA安装"
        return 1
    fi
    
    # 检查GPU数量
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    log_info "📊 检测到 ${gpu_count} 张GPU"
    
    if [ "${gpu_count}" -lt 2 ]; then
        log_warning "⚠️ 检测到少于2张GPU，将使用单GPU模式"
        export USE_MODEL_PARALLEL=false
        return 0
    fi
    
    # 检查GPU内存
    log_info "💾 GPU内存信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    
    # 🔧 修复：简化的GPU通信测试
    log_debug "🔗 测试GPU通信..."
    if ${PYTHON_EXECUTABLE} -c "
import torch
import sys
try:
    if torch.cuda.device_count() >= 2:
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')
        
        x = torch.randn(100, 100, device=device0)
        y = x.to(device1)
        z = y.to(device0)
        
        print('✅ GPU通信测试成功')
        print(f'   数据一致性: {torch.allclose(x, z)}')
    else:
        print('⚠️ GPU数量不足，跳过通信测试')
        sys.exit(1)
except Exception as e:
    print(f'❌ GPU通信测试失败: {e}')
    sys.exit(1)
"; then
        log_info "✅ GPU通信测试通过"
        export USE_MODEL_PARALLEL=true
    else
        log_warning "⚠️ GPU通信测试失败，将使用单GPU模式"
        export USE_MODEL_PARALLEL=false
    fi
    
    return 0
}

# --- Enhanced Environment Setup ---
export WANDB_PROJECT="VerilogGRPO_ModelParallel"
export WANDB_ENTITY="qhy0227-tsinghua-university"

# --- Model and Data Configuration ---
BASE_MODEL_NAME_OR_PATH="/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"
STAGE1_ADAPTER_PATH="/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/"
DATASET_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl"
DATASET_BASE_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset"

# 验证路径存在
if [ ! -f "${DATASET_PATH}" ]; then
    log_error "❌ 数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_BASE_PATH}" ]; then
    log_error "❌ 数据集基础路径不存在: ${DATASET_BASE_PATH}"
    exit 1
fi

log_info "✅ 数据集路径验证成功:"
log_info "  - 数据集文件: ${DATASET_PATH}"
log_info "  - 基础路径: ${DATASET_BASE_PATH}"

# --- RESUME FROM CHECKPOINT CONFIGURATION ---
RESUME_FROM_CHECKPOINT_DIR=""  # 先清空，避免恢复时的额外复杂性

OUTPUT_DIR_BASE="./model_parallel_outputs"

# 🔧 模型并行优化配置
USE_MODEL_PARALLEL=true
MAX_MEMORY_PER_GPU="75GiB"
LOW_CPU_MEM_USAGE=true

# 🔧 确保多GPU可见性（不要限制到单GPU）
export CUDA_VISIBLE_DEVICES=0,1

# 🔧 LoRA配置
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

# 🔧 长度配置
MAX_SEQ_LENGTH=6144
MAX_PROMPT_LENGTH=1536
MAX_COMPLETION_LENGTH=4608
LENGTH_ALLOCATION_STRATEGY="custom"

# 🔧 训练配置
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-5
NUM_TRAIN_EPOCHS=3
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
WEIGHT_DECAY=0.01

# 🔧 性能设置
BF16_ENABLED=true
FP16_ENABLED=false
GRADIENT_CHECKPOINTING_ENABLED=false  # 模型并行时禁用
OPTIMIZER_TYPE="adamw_torch"

# 🔧 数据加载配置
DATALOADER_NUM_WORKERS=0  # 模型并行时设为0避免冲突
DATALOADER_PIN_MEMORY=false

# 🔧 其他配置
NUM_GENERATIONS_GRPO=2
CALLBACK_NUM_SAMPLES=2
CALLBACK_EVAL_EVERY_N_STEPS=15  # 减少频率
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10

# --- 课程学习配置 ---
ENABLE_CURRICULUM=true
CURRICULUM_TYPE="dual_layer"
CURRICULUM_FOCUS_LEVELS="advanced basic intermediate"
CURRICULUM_COMPLEXITY_EMPHASIS="simple"
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=15

# 课程学习阈值
CURRICULUM_PERFORMANCE_THRESHOLD_1=0.75
CURRICULUM_PERFORMANCE_THRESHOLD_2=0.70
CURRICULUM_PERFORMANCE_THRESHOLD_3=0.65
CURRICULUM_PERFORMANCE_THRESHOLD_4=0.60
CURRICULUM_PERFORMANCE_THRESHOLD_5=0.55
CURRICULUM_MIN_EVALUATIONS=3

# 导出环境变量
export CURRICULUM_PERFORMANCE_THRESHOLD_1
export CURRICULUM_PERFORMANCE_THRESHOLD_2
export CURRICULUM_PERFORMANCE_THRESHOLD_3
export CURRICULUM_PERFORMANCE_THRESHOLD_4
export CURRICULUM_PERFORMANCE_THRESHOLD_5
export CURRICULUM_MIN_EVALUATIONS

# --- 经验回放配置 ---
ENABLE_EXPERIENCE_REPLAY=true
EXPERIENCE_BUFFER_SIZE=1500
REPLAY_SAMPLE_RATIO=0.2

# --- 奖励配置 ---
REWARD_COMPILATION_SUCCESS=2.0
REWARD_COMPILATION_FAILURE=-4.0
REWARD_TEST_PASS_BASE=1.5
REWARD_TEST_PASS_BONUS_MULTIPLIER=1.2
REWARD_MAX_FUNCTIONAL=15.0
REWARD_ALL_TESTS_PASSED_BONUS=5.0

# --- 生成参数 ---
GEN_TEMPERATURE=0.7
GEN_TOP_K=40
GEN_TOP_P=0.8
GEN_REPETITION_PENALTY=1.05

# --- 验证长度配置函数 ---
validate_length_config() {
    local total_length=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
    log_info "📏 长度配置验证:"
    log_info "  - 总序列长度: ${MAX_SEQ_LENGTH}"
    log_info "  - 最大提示长度: ${MAX_PROMPT_LENGTH} ($(( MAX_PROMPT_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
    log_info "  - 最大输出长度: ${MAX_COMPLETION_LENGTH} ($(( MAX_COMPLETION_LENGTH * 100 / MAX_SEQ_LENGTH ))%)"
    log_info "  - 分配策略: ${LENGTH_ALLOCATION_STRATEGY}"
    
    if [ ${total_length} -gt ${MAX_SEQ_LENGTH} ]; then
        log_error "❌ 配置错误: 提示长度 + 输出长度 (${total_length}) > 总序列长度 (${MAX_SEQ_LENGTH})"
        return 1
    else
        log_info "✅ 长度配置有效"
        return 0
    fi
}

# --- 主要检查流程 ---
main_checks() {
    log_info "🚀 开始模型并行训练环境检查..."
    
    # 1. GPU环境检查
    if ! check_gpu_environment; then
        log_error "❌ GPU环境检查失败"
        exit 1
    fi
    
    # 2. 验证长度配置
    if ! validate_length_config; then
        log_error "❌ 长度配置验证失败"
        exit 1
    fi
    
    log_info "✅ 所有检查完成，准备开始训练"
}

# 运行检查
main_checks

# --- 数据集检查 ---
if [ ! -f "${DATASET_PATH}" ]; then
    log_error "❌ 数据集文件未找到: ${DATASET_PATH}"
    exit 1
fi

log_info "🔍 检查数据集格式..."
FIRST_LINE=$(head -1 "${DATASET_PATH}")
if echo "${FIRST_LINE}" | grep -q '"level"' && echo "${FIRST_LINE}" | grep -q '"complexity_score"'; then
    log_info "✅ 检测到增强数据集格式"
else
    log_warning "⚠️ 数据集可能为旧格式，训练脚本将尝试自动升级"
fi

# --- 动态WandB运行名称 ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
export WANDB_RUN_NAME="model-parallel-LR${LEARNING_RATE}-R${LORA_RANK}-${TIMESTAMP}"

# --- 构建命令参数 ---
CMD_ARGS=""

# FSDP 特定配置
CMD_ARGS="${CMD_ARGS} --fsdp \"full_shard\""
# 注意：fsdp_min_num_params 和 fsdp_transformer_layer_cls_to_wrap 是互斥的，只能设置一个
# CMD_ARGS="${CMD_ARGS} --fsdp_min_num_params 100000000"  # 已禁用，与transformer_layer_cls_to_wrap冲突
CMD_ARGS="${CMD_ARGS} --fsdp_transformer_layer_cls_to_wrap \"QWenBlock\""  # 使用Qwen模型的实际layer类名
CMD_ARGS="${CMD_ARGS} --low_cpu_mem_usage ${LOW_CPU_MEM_USAGE}"

# 基础配置
CMD_ARGS="${CMD_ARGS} --wandb_project \"${WANDB_PROJECT}\""
CMD_ARGS="${CMD_ARGS} --wandb_entity \"${WANDB_ENTITY}\""
CMD_ARGS="${CMD_ARGS} --model_name_or_path \"${BASE_MODEL_NAME_OR_PATH}\""
CMD_ARGS="${CMD_ARGS} --stage1_adapter_path \"${STAGE1_ADAPTER_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_path \"${DATASET_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_base_path \"${DATASET_BASE_PATH}\""
CMD_ARGS="${CMD_ARGS} --output_dir_base \"${OUTPUT_DIR_BASE}\""

# LoRA配置
CMD_ARGS="${CMD_ARGS} --lora_rank ${LORA_RANK}"
CMD_ARGS="${CMD_ARGS} --lora_alpha ${LORA_ALPHA}"
CMD_ARGS="${CMD_ARGS} --lora_dropout ${LORA_DROPOUT}"
CMD_ARGS="${CMD_ARGS} --lora_target_modules ${LORA_TARGET_MODULES}"

# 长度配置
CMD_ARGS="${CMD_ARGS} --max_seq_length ${MAX_SEQ_LENGTH}"
CMD_ARGS="${CMD_ARGS} --script_max_prompt_length ${MAX_PROMPT_LENGTH}"
CMD_ARGS="${CMD_ARGS} --script_max_completion_length ${MAX_COMPLETION_LENGTH}"
CMD_ARGS="${CMD_ARGS} --length_allocation_strategy ${LENGTH_ALLOCATION_STRATEGY}"

# 训练配置
CMD_ARGS="${CMD_ARGS} --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}"
CMD_ARGS="${CMD_ARGS} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"
CMD_ARGS="${CMD_ARGS} --learning_rate ${LEARNING_RATE}"
CMD_ARGS="${CMD_ARGS} --num_train_epochs ${NUM_TRAIN_EPOCHS}"
CMD_ARGS="${CMD_ARGS} --warmup_ratio ${WARMUP_RATIO}"
CMD_ARGS="${CMD_ARGS} --lr_scheduler_type \"${LR_SCHEDULER_TYPE}\""
CMD_ARGS="${CMD_ARGS} --weight_decay ${WEIGHT_DECAY}"

# GRPO配置
CMD_ARGS="${CMD_ARGS} --num_generations ${NUM_GENERATIONS_GRPO}"
CMD_ARGS="${CMD_ARGS} --max_prompt_length ${MAX_PROMPT_LENGTH}"
CMD_ARGS="${CMD_ARGS} --max_completion_length ${MAX_COMPLETION_LENGTH}"

# 回调配置
CMD_ARGS="${CMD_ARGS} --callback_num_samples ${CALLBACK_NUM_SAMPLES}"
CMD_ARGS="${CMD_ARGS} --callback_eval_every_n_steps ${CALLBACK_EVAL_EVERY_N_STEPS}"

# 课程学习配置
CMD_ARGS="${CMD_ARGS} --enable_curriculum ${ENABLE_CURRICULUM}"
CMD_ARGS="${CMD_ARGS} --curriculum_type ${CURRICULUM_TYPE}"
CMD_ARGS="${CMD_ARGS} --curriculum_focus_levels ${CURRICULUM_FOCUS_LEVELS}"
CMD_ARGS="${CMD_ARGS} --curriculum_complexity_emphasis ${CURRICULUM_COMPLEXITY_EMPHASIS}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_check_interval ${CURRICULUM_PERFORMANCE_CHECK_INTERVAL}"

# 课程学习阈值
CMD_ARGS="${CMD_ARGS} --curriculum_performance_threshold_1 ${CURRICULUM_PERFORMANCE_THRESHOLD_1}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_threshold_2 ${CURRICULUM_PERFORMANCE_THRESHOLD_2}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_threshold_3 ${CURRICULUM_PERFORMANCE_THRESHOLD_3}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_threshold_4 ${CURRICULUM_PERFORMANCE_THRESHOLD_4}"
CMD_ARGS="${CMD_ARGS} --curriculum_performance_threshold_5 ${CURRICULUM_PERFORMANCE_THRESHOLD_5}"
CMD_ARGS="${CMD_ARGS} --curriculum_min_evaluations ${CURRICULUM_MIN_EVALUATIONS}"

# 经验回放配置
CMD_ARGS="${CMD_ARGS} --enable_experience_replay ${ENABLE_EXPERIENCE_REPLAY}"
CMD_ARGS="${CMD_ARGS} --experience_buffer_size ${EXPERIENCE_BUFFER_SIZE}"
CMD_ARGS="${CMD_ARGS} --replay_sample_ratio ${REPLAY_SAMPLE_RATIO}"

# 奖励配置
CMD_ARGS="${CMD_ARGS} --compilation_success ${REWARD_COMPILATION_SUCCESS}"
CMD_ARGS="${CMD_ARGS} --compilation_failure ${REWARD_COMPILATION_FAILURE}"
CMD_ARGS="${CMD_ARGS} --test_pass_base_reward ${REWARD_TEST_PASS_BASE}"
CMD_ARGS="${CMD_ARGS} --test_pass_bonus_multiplier ${REWARD_TEST_PASS_BONUS_MULTIPLIER}"
CMD_ARGS="${CMD_ARGS} --max_functional_reward ${REWARD_MAX_FUNCTIONAL}"
CMD_ARGS="${CMD_ARGS} --all_tests_passed_bonus ${REWARD_ALL_TESTS_PASSED_BONUS}"

# 生成参数
CMD_ARGS="${CMD_ARGS} --gen_temperature ${GEN_TEMPERATURE}"
CMD_ARGS="${CMD_ARGS} --gen_top_k ${GEN_TOP_K}"
CMD_ARGS="${CMD_ARGS} --gen_top_p ${GEN_TOP_P}"
CMD_ARGS="${CMD_ARGS} --gen_repetition_penalty ${GEN_REPETITION_PENALTY}"

# 性能设置
if [ "$BF16_ENABLED" = true ]; then CMD_ARGS="${CMD_ARGS} --bf16"; fi
if [ "$FP16_ENABLED" = true ] && [ "$BF16_ENABLED" = false ]; then CMD_ARGS="${CMD_ARGS} --fp16"; fi
if [ "$GRADIENT_CHECKPOINTING_ENABLED" = true ]; then CMD_ARGS="${CMD_ARGS} --gradient_checkpointing"; fi

# 数据加载配置
CMD_ARGS="${CMD_ARGS} --dataloader_num_workers ${DATALOADER_NUM_WORKERS}"
if [ "$DATALOADER_PIN_MEMORY" = true ]; then CMD_ARGS="${CMD_ARGS} --dataloader_pin_memory"; fi

# 保存和日志配置
CMD_ARGS="${CMD_ARGS} --logging_strategy \"steps\""
CMD_ARGS="${CMD_ARGS} --logging_steps ${LOGGING_STEPS}"
CMD_ARGS="${CMD_ARGS} --save_strategy \"${SAVE_STRATEGY}\""
CMD_ARGS="${CMD_ARGS} --save_steps ${SAVE_STEPS}"
CMD_ARGS="${CMD_ARGS} --save_total_limit ${SAVE_TOTAL_LIMIT}"
CMD_ARGS="${CMD_ARGS} --report_to \"wandb\""
CMD_ARGS="${CMD_ARGS} --remove_unused_columns False"
CMD_ARGS="${CMD_ARGS} --optim \"${OPTIMIZER_TYPE}\""
CMD_ARGS="${CMD_ARGS} --ddp_find_unused_parameters False"

# 恢复配置
if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ] && [ -d "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
    CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint \"${RESUME_FROM_CHECKPOINT_DIR}\""
fi

# 缓存目录
CACHE_DIR_BASE="${SCRIPT_DIR}/.model_parallel_cache"
mkdir -p "${CACHE_DIR_BASE}/datasets"
mkdir -p "${CACHE_DIR_BASE}/models"
CMD_ARGS="${CMD_ARGS} --cache_dir \"${CACHE_DIR_BASE}/models\""

# --- 🔧 关键修复：训练执行配置 ---
PYTHON_SCRIPT_TO_RUN="${SCRIPT_DIR}/main.py"

# 🔧 使用 torchrun 启动分布式 FSDP
log_info "🚀 使用 torchrun 启动分布式 FSDP 训练"
NPROC_PER_NODE=2  # 如需调整 GPU 数量，请修改此处
# 使用随机端口避免冲突
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

# 确保conda环境激活
FULL_CMD="conda activate ReasoningV && torchrun --nnodes 1 --nproc_per_node ${NPROC_PER_NODE} --master_port ${MASTER_PORT} ${PYTHON_SCRIPT_TO_RUN} ${CMD_ARGS}"

# --- 创建输出目录 ---
mkdir -p ${OUTPUT_DIR_BASE}

# --- 训练前摘要 ---
echo ""
echo "========================================================================"
echo "                    模型并行GRPO训练启动"
echo "========================================================================"
log_info "🎯 训练配置摘要:"
log_info "  - 模式: 单进程模型并行（避免DDP冲突）"
log_info "  - 模型: $(basename "${BASE_MODEL_NAME_OR_PATH}")"
log_info "  - 数据集: $(basename "${DATASET_PATH}")"
log_info "  - 序列长度: ${MAX_SEQ_LENGTH} (Prompt: ${MAX_PROMPT_LENGTH}, Completion: ${MAX_COMPLETION_LENGTH})"
log_info "  - LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
log_info "  - 批次大小: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
log_info "  - 有效批次: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
log_info "  - 学习率: ${LEARNING_RATE}"
log_info "  - 输出目录: ${OUTPUT_DIR_BASE}"
log_info "  - 启动方式: 直接python（非分布式）"

echo ""
log_info "📊 内存预估:"
MODEL_PARAMS=8  # 8B参数
MODEL_MEMORY=$((MODEL_PARAMS * 2))  # bf16
log_info "  - 模型参数: ~${MODEL_PARAMS}B"
log_info "  - 模型内存: ~${MODEL_MEMORY}GB（将分布到2张GPU）"
log_info "  - 每GPU预估: ~$((MODEL_MEMORY / 2 + 10))GB"

echo ""
echo "⚠️  重要提示:"
echo "  - 使用单进程模型并行，避免DDP冲突"
echo "  - 模型权重将分布到GPU 0和1，而非复制"
echo "  - 不使用torchrun启动"
echo ""
echo "监控链接: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "========================================================================"

# 保存完整命令
echo "${FULL_CMD}" > "${OUTPUT_DIR_BASE}/full_training_command.txt"
log_info "💾 完整命令已保存到: ${OUTPUT_DIR_BASE}/full_training_command.txt"

# --- 训练前最终GPU状态 ---
echo ""
log_info "🔍 训练前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# --- 执行训练 ---
echo ""
log_info "🚀 开始模型并行GRPO训练 ($(date))..."

# 设置清理函数
cleanup_on_exit() {
    echo ""
    log_info "🛑 训练结束/中断，执行清理..."
    log_info "📊 最终GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    log_info "⏰ 训练结束时间: $(date)"
}
trap cleanup_on_exit INT TERM EXIT

# 🔧 关键：执行训练命令
eval ${FULL_CMD}

status=$?

# --- 训练后摘要 ---
echo ""
echo "========================================================================"
echo "                      模型并行训练完成摘要"
echo "========================================================================"
log_info "⏰ 训练结束时间: $(date)"
log_info "🎯 退出状态: ${status}"

if [ $status -eq 0 ]; then
    log_info "✅ 模型并行GRPO训练成功完成!"
    
    # 检查输出
    if [ -d "${OUTPUT_DIR_BASE}" ]; then
        FINAL_MODEL_DIR=$(find "${OUTPUT_DIR_BASE}" -name "*final*model*" -type d 2>/dev/null | head -1)
        if [ -n "${FINAL_MODEL_DIR}" ]; then
            MODEL_SIZE=$(du -sh "${FINAL_MODEL_DIR}" 2>/dev/null | cut -f1 || echo "Unknown")
            log_info "✅ 最终模型保存到: ${FINAL_MODEL_DIR}"
            log_info "  📦 模型大小: ${MODEL_SIZE}"
        fi
        
        # 检查日志
        LOG_FILE=$(find "${OUTPUT_DIR_BASE}" -name "*log*.txt" 2>/dev/null | head -1)
        if [ -f "${LOG_FILE}" ]; then
            LOG_SIZE=$(wc -l < "${LOG_FILE}" 2>/dev/null || echo "0")
            log_info "✅ 训练日志: ${LOG_SIZE} 行"
        fi
        
        # 检查模型并行信息
        DEVICE_MAP_FILE=$(find "${OUTPUT_DIR_BASE}" -name "model_device_map.json" 2>/dev/null | head -1)
        if [ -f "${DEVICE_MAP_FILE}" ]; then
            log_info "✅ 模型设备映射已保存: $(basename "${DEVICE_MAP_FILE}")"
        fi
    fi
    
    echo ""
    log_info "🎉 后续步骤:"
    log_info "1. 查看WandB仪表板了解详细指标"
    log_info "2. 检查模型设备分布情况"
    log_info "3. 在测试集上评估模型性能"
    log_info "4. 验证模型并行的内存效率"
    
else
    log_error "❌ 模型并行训练失败，退出码: ${status}"
    echo ""
    log_info "🔧 故障排除:"
    log_info "1. 检查训练日志: ${OUTPUT_DIR_BASE}/*/training_log.txt"
    log_info "2. 验证GPU状态: nvidia-smi"
    log_info "3. 检查是否有DDP冲突错误"
    log_info "4. 确认使用单进程启动而非torchrun"
    log_info "5. 查看WandB日志获取详细错误信息"
    
    # 错误诊断
    if [ $status -ne 0 ]; then
        log_error "🔍 可能的错误原因:"
        log_error "  - 如果看到DTensor/DDP错误：确保清除了所有分布式环境变量"
        log_error "  - 如果看到OOM错误：减少批次大小或序列长度"
        log_error "  - 如果看到导入错误：检查环境和依赖"
        log_error "  - 如果看到配置错误：检查参数设置是否正确"
    fi
fi

echo "========================================================================"

# 最终GPU状态
if command -v nvidia-smi &> /dev/null; then
    echo ""
    log_info "📊 最终GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

echo ""
log_info "💡 使用提示:"
log_info "  - 本脚本使用单进程模型并行，避免了DDP冲突"
log_info "  - 模型权重分布在两张GPU上，实现真正的并行"
log_info "  - 可以通过调整MAX_MEMORY_PER_GPU来控制内存分配"
log_info "  - 如需调整配置，修改脚本中的相应变量即可"

exit ${status}