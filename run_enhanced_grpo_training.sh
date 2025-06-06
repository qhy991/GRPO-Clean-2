#!/bin/bash
# run_enhanced_grpo_training_v2.sh - 支持新数据集格式的增强GRPO训练脚本

# --- Exit on error ---
set -e
export CUDA_VISIBLE_DEVICES=1
# --- Get the directory where the script is located ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_EXECUTABLE="python3"

# --- Enhanced Environment Setup ---
export WANDB_PROJECT="VerilogGRPO_Enhanced_v3"
export WANDB_ENTITY="qhy0227-tsinghua-university"

# Enable better CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- Distributed Training Parameters ---
NUM_GPUS_PER_NODE=1
MASTER_ADDR="localhost"
MASTER_PORT=$(shuf -i 10000-19999 -n 1)

# --- Enhanced Model and Data Configuration ---
BASE_MODEL_NAME_OR_PATH="/home/qhy/Research/LLM/GRPO-RV/QWEN3-4B"
STAGE1_ADAPTER_PATH="/home/qhy/Research/LLM/GRPO-RV/GRPO-v4/S1/trainer_output/checkpoint-138795"

# 🔥 重要：使用处理后的增强数据集
# 根据你的数据集处理方式选择一个：
DATASET_PATH="/home/qhy/Research/LLM/GRPO-RV/dataset/all-2.jsonl"
DATASET_BASE_PATH=$(dirname "${DATASET_PATH}")

# --- RESUME FROM CHECKPOINT CONFIGURATION ---
# 🔄 设置此变量为你想要从中恢复的 checkpoint 目录的路径
# 例如: RESUME_FROM_CHECKPOINT_DIR="./enhanced_grpo_v2_runs/your_previous_run_output_dir/checkpoint-XXXX"
# 将此留空以开始新的训练。将其设置为一个不存在的路径也会开始新的训练（会有警告）。
RESUME_FROM_CHECKPOINT_DIR="/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_v3_runs/v3-_home_qhy_Research_LLM_GRPO-RV_QWEN3-4B-LR6e-6-R64-20250605-163908-2/checkpoint-32"
# "/home/qhy/Research/LLM/GRPO-Clean-2/enhanced_grpo_v3_runs/v3-_home_qhy_Research_LLM_GRPO-RV_QWEN3-4B-LR1e-5-R64-20250604-232819-2/checkpoint-136"

# 🔧 关键：WandB恢复配置
if [ -n "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
    if [ -d "${RESUME_FROM_CHECKPOINT_DIR}" ]; then
        echo "🔄 检测到checkpoint恢复，配置WandB恢复..."
        
        # 尝试从checkpoint目录提取WandB run ID
        PARENT_DIR=$(dirname "${RESUME_FROM_CHECKPOINT_DIR}")
        WANDB_DIR="${PARENT_DIR}/wandb"
        
        if [ -d "${WANDB_DIR}" ]; then
            # 查找最新的run目录
            LATEST_RUN_DIR=$(find "${WANDB_DIR}" -name "run-*" -type d | sort | tail -1)
            if [ -n "${LATEST_RUN_DIR}" ]; then
                # 提取run ID (格式: run-20231201_123456-abcd1234)
                RUN_ID=$(basename "${LATEST_RUN_DIR}" | sed 's/run-[0-9]*_[0-9]*-//')
                if [ -n "${RUN_ID}" ]; then
                    export WANDB_RUN_ID="${RUN_ID}"
                    export WANDB_RESUME="must"
                    echo "✅ 找到WandB run ID: ${RUN_ID}"
                    echo "✅ 设置WandB恢复模式: must"
                else
                    echo "⚠️ 无法从目录名提取run ID，将尝试自动恢复"
                    export WANDB_RESUME="allow"
                fi
            else
                echo "⚠️ 未找到WandB run目录，将尝试自动恢复"
                export WANDB_RESUME="allow"
            fi
        else
            echo "⚠️ 未找到WandB目录，将尝试自动恢复"
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
        
    else
        echo "⚠️ 警告: RESUME_FROM_CHECKPOINT_DIR ('${RESUME_FROM_CHECKPOINT_DIR}') 指定的目录不存在。将开始新的训练，并忽略此设置。"
        RESUME_FROM_CHECKPOINT_DIR="" # 清空以避免传递无效路径给Python脚本
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

OUTPUT_DIR_BASE="./enhanced_grpo_v3_runs"

# --- Enhanced LoRA Configuration ---
LORA_RANK=64          # Increased capacity
LORA_ALPHA=128        # Scaled with rank  
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

MAX_SEQ_LENGTH=4096

# --- Enhanced Callback Configuration ---
CALLBACK_NUM_SAMPLES=2
CALLBACK_EVAL_EVERY_N_STEPS=10

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
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=6e-6               # More conservative
NUM_TRAIN_EPOCHS=4
MAX_STEPS=-1
WARMUP_RATIO=0.15                # Increased warmup
LR_SCHEDULER_TYPE="cosine"
WEIGHT_DECAY=0.01
LOGGING_STRATEGY="steps"
LOGGING_STEPS=2
SAVE_STRATEGY="steps"
SAVE_STEPS=8
SAVE_TOTAL_LIMIT=3
SEED=42

# --- Enhanced GRPO Configuration ---
NUM_GENERATIONS_GRPO=4
MAX_COMPLETION_LENGTH_GRPO=$((MAX_SEQ_LENGTH / 2))

# --- Enhanced Performance Settings ---
BF16_ENABLED=true
FP16_ENABLED=false
GRADIENT_CHECKPOINTING_ENABLED=true
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
export WANDB_RUN_NAME="v3-${MODEL_NAME_SLUG}-LR${LEARNING_RATE}-R${LORA_RANK}-${TIMESTAMP}-2"
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
echo "  ✅ Experience replay: ${ENABLE_EXPERIENCE_REPLAY} (buffer size: ${EXPERIENCE_BUFFER_SIZE})"
echo "  ✅ Adaptive reward scaling: ${REWARD_ENABLE_ADAPTIVE_SCALING}"
echo "  ✅ Enhanced LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo "  ✅ Conservative learning: lr=${LEARNING_RATE}, warmup=${WARMUP_RATIO}"
echo "  ✅ Generation diversity: temp=${GEN_TEMPERATURE}, rep_penalty=${GEN_REPETITION_PENALTY}"
echo "  ✅ Enhanced monitoring: ${CALLBACK_NUM_SAMPLES} samples every ${CALLBACK_EVAL_EVERY_N_STEPS} steps"

echo ""
mkdir -p ${OUTPUT_DIR_BASE}

# --- Build Command Arguments ---
CMD_ARGS=""

# EnvConfig
CMD_ARGS="${CMD_ARGS} --hf_endpoint \"https://hf-mirror.com\""
CMD_ARGS="${CMD_ARGS} --http_proxy \"http://10.130.149.18:7890\""
CMD_ARGS="${CMD_ARGS} --https_proxy \"http://10.130.149.18:7890\""
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
CMD_ARGS="${CMD_ARGS} --max_completion_length ${MAX_COMPLETION_LENGTH_GRPO}"
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
echo "Training will begin with curriculum stage 0..."
echo "Monitor progress at: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "========================================================================"

# Save full command to file for debugging
echo "${FULL_CMD}" > "${OUTPUT_DIR_BASE}/full_training_command.txt"
echo "Full command saved to: ${OUTPUT_DIR_BASE}/full_training_command.txt"

# --- Execute Training ---
echo "Starting enhanced GRPO v2 training at $(date)..."

# Trap to handle interruption
trap 'echo "Training interrupted at $(date). Cleaning up..."; exit 130' INT TERM

eval "${FULL_CMD}"

status=$?

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