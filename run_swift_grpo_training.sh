#!/bin/bash
# run_swift_grpo_training.sh - 使用Swift进行GRPO多卡训练

# --- Exit on error ---
set -e
export CUDA_VISIBLE_DEVICES=0,1

# --- 激活ReasoningV环境 ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ReasoningV
echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"

# --- 获取脚本目录 ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# --- Swift多卡训练优化设置 ---
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# --- 颜色定义 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Swift和WandB配置 ---
export WANDB_PROJECT="VerilogGRPO_Swift_Enhanced"
export WANDB_ENTITY="qhy0227-tsinghua-university"

# --- Model和数据配置 ---
BASE_MODEL_NAME_OR_PATH="/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834"
STAGE1_ADAPTER_PATH="/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/"
DATASET_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset/swift_all-with-module-2.jsonl"

# --- Swift专用目录设置 ---
OUTPUT_DIR_BASE="./swift_grpo_runs"
mkdir -p ${OUTPUT_DIR_BASE}

# --- 训练参数 ---
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=6e-6
NUM_TRAIN_EPOCHS=4
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=3072

# --- LoRA配置 ---
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# --- 课程学习配置 ---
CURRICULUM_PERFORMANCE_THRESHOLD_1=0.75
CURRICULUM_PERFORMANCE_THRESHOLD_2=0.70
CURRICULUM_PERFORMANCE_THRESHOLD_3=0.65
CURRICULUM_PERFORMANCE_THRESHOLD_4=0.60
CURRICULUM_PERFORMANCE_THRESHOLD_5=0.55

# --- 检查数据集 ---
if [ ! -f "${DATASET_PATH}" ]; then
    log_error "数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

log_info "🚀 启动Swift多卡GRPO训练"
log_info "模型: $(basename "${BASE_MODEL_NAME_OR_PATH}")"
log_info "数据集: $(basename "${DATASET_PATH}")"
log_info "输出目录: ${OUTPUT_DIR_BASE}"

# --- 预训练检查 ---
echo ""
echo "🔍 训练前环境检查..."
echo "Python环境: $(which python)"
echo "Swift版本: $(python -c 'import swift; print(swift.__version__)' 2>/dev/null || echo 'N/A')"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 检查数据集样本
echo "📊 数据集信息:"
DATASET_LINES=$(wc -l < "${DATASET_PATH}" 2>/dev/null || echo "0")
echo "数据集行数: ${DATASET_LINES}"
if [ -f "${DATASET_PATH}" ]; then
    echo "数据集大小: $(du -sh "${DATASET_PATH}" | cut -f1)"
    echo "样本预览:"
    head -1 "${DATASET_PATH}" | python -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    print(f'  conversations数量: {len(data.get(\"conversations\", []))}')
    if 'conversations' in data and len(data['conversations']) > 0:
        print(f'  第一个用户消息长度: {len(data[\"conversations\"][0].get(\"value\", \"\"))}')
        if len(data['conversations']) > 1:
            print(f'  第一个助手回复长度: {len(data[\"conversations\"][1].get(\"value\", \"\"))}')
except Exception as e:
    print(f'  数据格式检查失败: {e}')
" 2>/dev/null || echo "  数据格式检查失败"
else
    log_error "数据集文件不存在: ${DATASET_PATH}"
    exit 1
fi

# --- 生成时间戳 ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXPERIMENT_NAME="swift-grpo-${TIMESTAMP}"

echo "========================================================================"
echo "                    SWIFT GRPO多卡训练开始"
echo "========================================================================"

# --- Swift多卡训练命令 ---
# 使用Swift的分布式训练接口进行GRPO风格训练
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type qwen3-8b \
    --model_id_or_path "${BASE_MODEL_NAME_OR_PATH}" \
    --sft_type lora \
    --tuner_backend peft \
    --dataset_path "${DATASET_PATH}" \
    --train_dataset_sample -1 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --max_length ${MAX_SEQ_LENGTH} \
    --truncation_strategy delete \
    --check_dataset_strategy warning \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout_p ${LORA_DROPOUT} \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --weight_decay 0.01 \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.15 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --output_dir "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}" \
    --ddp_backend nccl \
    --ddp_find_unused_parameters false \
    --dataloader_num_workers 0 \
    --push_to_hub false \
    --hub_model_id "${EXPERIMENT_NAME}" \
    --hub_private_repo true \
    --hub_token none \
    --use_flash_attn true \

    --report_to 'wandb' \
    --run_name "${EXPERIMENT_NAME}" \
    --seed 42 \
    --deepspeed default-zero2

training_status=$?

echo ""
echo "========================================================================"
echo "                      SWIFT训练完成"
echo "========================================================================"
echo "训练结束时间: $(date)"
echo "退出状态: ${training_status}"

if [ $training_status -eq 0 ]; then
    log_info "✅ Swift GRPO训练成功完成！"
    log_info "模型保存位置: ${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}"
    
    # 显示最终模型信息
    if [ -d "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}" ]; then
        MODEL_SIZE=$(du -sh "${OUTPUT_DIR_BASE}/${EXPERIMENT_NAME}" 2>/dev/null | cut -f1 || echo "Unknown")
        log_info "模型大小: ${MODEL_SIZE}"
    fi
else
    log_error "❌ Swift训练失败，退出码: ${training_status}"
    log_error "请检查训练日志和错误信息"
fi

# --- 最终GPU状态 ---
echo ""
echo "最终GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

exit ${training_status} 