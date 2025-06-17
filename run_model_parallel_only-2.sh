#!/bin/bash
# run_model_parallel_only.sh - 纯模型并行训练脚本（不使用FSDP分布式）

set -e

# 🔧 激活conda环境
echo "🔧 激活ReasoningV环境..."
source /home/qhy/anaconda3/bin/activate ReasoningV

# 清除分布式环境变量
echo "🧹 清除分布式环境变量..."
unset RANK 2>/dev/null || true
unset LOCAL_RANK 2>/dev/null || true  
unset WORLD_SIZE 2>/dev/null || true
unset MASTER_ADDR 2>/dev/null || true
unset MASTER_PORT 2>/dev/null || true

# 🔧 设置GPU设备
export CUDA_VISIBLE_DEVICES=0,1

# 🔧 模型并行优化设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:4"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0

# Flash Attention
export FLASH_ATTENTION_V2=1

# 目录设置
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_EXECUTABLE="python"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'  
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 配置参数
export WANDB_PROJECT="VerilogGRPO_FullTrain_ModelParallel"  # 修改项目名以区分全量训练
export WANDB_ENTITY="qhy0227-tsinghua-university"

BASE_MODEL_NAME_OR_PATH="/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"
STAGE1_ADAPTER_PATH="/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/"
DATASET_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl"
DATASET_BASE_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset"
OUTPUT_DIR_BASE="./full_train_model_parallel_outputs"  # 全量训练输出目录

# 验证路径
if [ ! -f "${DATASET_PATH}" ]; then
    log_error "❌ 数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

# 🔧 纯模型并行配置（关键：不使用FSDP）
USE_MODEL_PARALLEL=true
USE_FSDP=false  # 明确禁用FSDP
MAX_MEMORY_PER_GPU="70GiB"  # 全量训练时稍微减少内存限制
LOW_CPU_MEM_USAGE=true

# 🔧 全量训练配置（禁用LoRA）
USE_LORA=false  # 关键：禁用LoRA以启用全量训练
LORA_RANK=0     # 设为0表示不使用LoRA
LORA_ALPHA=0
LORA_DROPOUT=0.0
LORA_TARGET_MODULES=""  # 空字符串表示不指定LoRA目标模块

# 长度配置
MAX_SEQ_LENGTH=6144
MAX_PROMPT_LENGTH=1536
MAX_COMPLETION_LENGTH=4608
LENGTH_ALLOCATION_STRATEGY="custom"

# 🔧 全量训练配置
PER_DEVICE_TRAIN_BATCH_SIZE=1  # 全量训练时减小批次大小
GRADIENT_ACCUMULATION_STEPS=8  # 增加梯度累积以保持有效批次大小
LEARNING_RATE=5e-6             # 全量训练时使用更小的学习率
NUM_TRAIN_EPOCHS=2             # 全量训练通常需要更少的epochs
WARMUP_RATIO=0.05              # 减少warmup比例
LR_SCHEDULER_TYPE="cosine"
WEIGHT_DECAY=0.01

# 性能设置
BF16_ENABLED=true
FP16_ENABLED=false
GRADIENT_CHECKPOINTING_ENABLED=false  # 模型并行时禁用
OPTIMIZER_TYPE="adamw_torch"

# 数据加载配置
DATALOADER_NUM_WORKERS=0
DATALOADER_PIN_MEMORY=false

# 其他配置
NUM_GENERATIONS_GRPO=2
CALLBACK_NUM_SAMPLES=2
CALLBACK_EVAL_EVERY_N_STEPS=15
SAVE_STRATEGY="steps"
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=10

# 课程学习配置
ENABLE_CURRICULUM=true
CURRICULUM_TYPE="dual_layer"
CURRICULUM_FOCUS_LEVELS="advanced basic intermediate"
CURRICULUM_COMPLEXITY_EMPHASIS="simple"
CURRICULUM_PERFORMANCE_CHECK_INTERVAL=15

# 经验回放配置
ENABLE_EXPERIENCE_REPLAY=true
EXPERIENCE_BUFFER_SIZE=1500
REPLAY_SAMPLE_RATIO=0.2

# 奖励配置
REWARD_COMPILATION_SUCCESS=2.0
REWARD_COMPILATION_FAILURE=-4.0
REWARD_TEST_PASS_BASE=1.5
REWARD_TEST_PASS_BONUS_MULTIPLIER=1.2
REWARD_MAX_FUNCTIONAL=15.0
REWARD_ALL_TESTS_PASSED_BONUS=5.0

# 生成参数
GEN_TEMPERATURE=0.7
GEN_TOP_K=40
GEN_TOP_P=0.8
GEN_REPETITION_PENALTY=1.05

# 动态WandB运行名称
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [ "$USE_LORA" = true ]; then
    export WANDB_RUN_NAME="model-parallel-LoRA-LR${LEARNING_RATE}-R${LORA_RANK}-${TIMESTAMP}"
else
    export WANDB_RUN_NAME="model-parallel-FullTrain-LR${LEARNING_RATE}-${TIMESTAMP}"
fi

# 构建命令参数
CMD_ARGS=""

# 基础配置
CMD_ARGS="${CMD_ARGS} --wandb_project \"${WANDB_PROJECT}\""
CMD_ARGS="${CMD_ARGS} --wandb_entity \"${WANDB_ENTITY}\""
CMD_ARGS="${CMD_ARGS} --model_name_or_path \"${BASE_MODEL_NAME_OR_PATH}\""
CMD_ARGS="${CMD_ARGS} --stage1_adapter_path \"${STAGE1_ADAPTER_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_path \"${DATASET_PATH}\""
CMD_ARGS="${CMD_ARGS} --dataset_base_path \"${DATASET_BASE_PATH}\""
CMD_ARGS="${CMD_ARGS} --output_dir_base \"${OUTPUT_DIR_BASE}\""

# 🔧 关键：明确启用模型并行
CMD_ARGS="${CMD_ARGS} --use_model_parallel true"
CMD_ARGS="${CMD_ARGS} --max_memory_per_gpu \"${MAX_MEMORY_PER_GPU}\""
CMD_ARGS="${CMD_ARGS} --low_cpu_mem_usage ${LOW_CPU_MEM_USAGE}"

# 🔧 全量训练配置（禁用LoRA）
CMD_ARGS="${CMD_ARGS} --use_lora ${USE_LORA}"
if [ "$USE_LORA" = true ]; then
    CMD_ARGS="${CMD_ARGS} --lora_rank ${LORA_RANK}"
    CMD_ARGS="${CMD_ARGS} --lora_alpha ${LORA_ALPHA}"
    CMD_ARGS="${CMD_ARGS} --lora_dropout ${LORA_DROPOUT}"
    CMD_ARGS="${CMD_ARGS} --lora_target_modules ${LORA_TARGET_MODULES}"
else
    echo "[INFO] 🎯 全量训练模式：已禁用LoRA配置"
fi

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

# 缓存目录
CACHE_DIR_BASE="${SCRIPT_DIR}/.model_parallel_cache"
mkdir -p "${CACHE_DIR_BASE}/datasets"
mkdir -p "${CACHE_DIR_BASE}/models"
CMD_ARGS="${CMD_ARGS} --cache_dir \"${CACHE_DIR_BASE}/models\""

# Python脚本路径
PYTHON_SCRIPT_TO_RUN="${SCRIPT_DIR}/main.py"

# 🔧 关键：使用直接python执行，不使用torchrun
FULL_CMD="${PYTHON_EXECUTABLE} ${PYTHON_SCRIPT_TO_RUN} ${CMD_ARGS}"

# 创建输出目录
mkdir -p ${OUTPUT_DIR_BASE}

# 训练前摘要
echo ""
echo "========================================================================"
if [ "$USE_LORA" = true ]; then
    echo "                    模型并行LoRA训练启动"
else
    echo "                    模型并行全量训练启动"
fi
echo "========================================================================"
log_info "🎯 训练配置摘要:"
log_info "  - 模式: 纯模型并行（无FSDP，无DDP）"
log_info "  - 训练类型: $([ "$USE_LORA" = true ] && echo "LoRA微调" || echo "全量训练")"
log_info "  - 启动方式: 直接python（单进程）"
log_info "  - GPU配置: ${CUDA_VISIBLE_DEVICES}"
log_info "  - 模型: $(basename "${BASE_MODEL_NAME_OR_PATH}")"
log_info "  - 数据集: $(basename "${DATASET_PATH}")"
log_info "  - 序列长度: ${MAX_SEQ_LENGTH}"
if [ "$USE_LORA" = true ]; then
    log_info "  - LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
else
    log_info "  - 全量训练: 所有参数可训练"
fi
log_info "  - 批次大小: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
log_info "  - 梯度累积: ${GRADIENT_ACCUMULATION_STEPS}"
log_info "  - 有效批次: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
log_info "  - 学习率: ${LEARNING_RATE}"
log_info "  - 训练轮数: ${NUM_TRAIN_EPOCHS}"
log_info "  - 输出目录: ${OUTPUT_DIR_BASE}"

echo ""
log_info "📊 内存配置:"
log_info "  - 每GPU内存限制: ${MAX_MEMORY_PER_GPU}"
log_info "  - 低CPU内存使用: ${LOW_CPU_MEM_USAGE}"
log_info "  - 梯度检查点: ${GRADIENT_CHECKPOINTING_ENABLED}"

echo ""
echo "========================================================================"

# 保存完整命令
echo "${FULL_CMD}" > "${OUTPUT_DIR_BASE}/full_training_command.txt"
log_info "💾 完整命令已保存到: ${OUTPUT_DIR_BASE}/full_training_command.txt"

# 训练前GPU状态
echo ""
log_info "🔍 训练前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# 清理函数
cleanup_on_exit() {
    echo ""
    log_info "🛑 训练结束/中断，执行清理..."
    log_info "📊 最终GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    log_info "⏰ 训练结束时间: $(date)"
}
trap cleanup_on_exit INT TERM EXIT

# 🔧 执行训练命令
echo ""
log_info "🚀 开始纯模型并行GRPO训练 ($(date))..."

eval ${FULL_CMD}

status=$?

# 训练后摘要
echo ""
echo "========================================================================"
echo "                      纯模型并行训练完成摘要"
echo "========================================================================"
log_info "⏰ 训练结束时间: $(date)"
log_info "🎯 退出状态: ${status}"

if [ $status -eq 0 ]; then
    log_info "✅ 纯模型并行GRPO训练成功完成!"
else
    log_error "❌ 纯模型并行训练失败，退出码: ${status}"
fi

echo "========================================================================"

exit ${status} 