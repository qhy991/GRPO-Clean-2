#!/bin/bash
# 测试动态备份逻辑的小脚本

OUTPUT_DIR_BASE="./test_backup_output"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# 清理旧的测试目录
if [ -d "${OUTPUT_DIR_BASE}" ]; then
    rm -rf "${OUTPUT_DIR_BASE}"
fi

mkdir -p "${OUTPUT_DIR_BASE}"

# 模拟创建一个带时间戳的训练目录
sleep 1
MOCK_TRAINING_DIR="${OUTPUT_DIR_BASE}/enhanced-v3-test-LR6e-6-R64-${TIMESTAMP}"
mkdir -p "${MOCK_TRAINING_DIR}"
echo "模拟模型文件" > "${MOCK_TRAINING_DIR}/pytorch_model.bin"

# 模拟基础文件
echo "mock command" > "${OUTPUT_DIR_BASE}/full_training_command.txt"

echo "🧪 测试环境已创建:"
echo "  - 基础目录: ${OUTPUT_DIR_BASE}"
echo "  - 模拟训练目录: ${MOCK_TRAINING_DIR}"

# 测试查找逻辑
echo ""
echo "🔍 测试目录查找逻辑..."

ACTUAL_OUTPUT_DIR=""
if [ -d "${OUTPUT_DIR_BASE}" ]; then
    # 查找包含当前时间戳的目录（最近5分钟内创建的）
    ACTUAL_OUTPUT_DIR=$(find "${OUTPUT_DIR_BASE}" -maxdepth 1 -type d -name "*${TIMESTAMP}*" -newermt "5 minutes ago" | head -1)
    echo "  - 时间戳匹配查找: ${ACTUAL_OUTPUT_DIR}"
    
    # 如果没找到带时间戳的，查找最新的目录
    if [ -z "${ACTUAL_OUTPUT_DIR}" ]; then
        ACTUAL_OUTPUT_DIR=$(find "${OUTPUT_DIR_BASE}" -maxdepth 1 -type d -newer "${OUTPUT_DIR_BASE}/full_training_command.txt" 2>/dev/null | head -1)
        echo "  - 相对时间查找: ${ACTUAL_OUTPUT_DIR}"
    fi
    
    # 最后备选：查找最新修改的目录
    if [ -z "${ACTUAL_OUTPUT_DIR}" ]; then
        ACTUAL_OUTPUT_DIR=$(ls -dt "${OUTPUT_DIR_BASE}"/*/ 2>/dev/null | head -1 | sed 's/\/$//')
        echo "  - 最新目录查找: ${ACTUAL_OUTPUT_DIR}"
    fi
fi

if [ -n "${ACTUAL_OUTPUT_DIR}" ] && [ -d "${ACTUAL_OUTPUT_DIR}" ]; then
    echo "✅ 成功找到目标目录: ${ACTUAL_OUTPUT_DIR}"
    
    # 创建符号链接测试
    LINK_NAME="${OUTPUT_DIR_BASE}/latest_run"
    if [ -L "${LINK_NAME}" ]; then
        rm "${LINK_NAME}"
    fi
    ln -s "$(basename "${ACTUAL_OUTPUT_DIR}")" "${LINK_NAME}"
    echo "✅ 符号链接创建成功: ${LINK_NAME} -> $(basename "${ACTUAL_OUTPUT_DIR}")"
    
    # 测试在找到的目录中创建文件
    echo "测试备份文件" > "${ACTUAL_OUTPUT_DIR}/test_backup.txt"
    echo "✅ 测试文件已创建: ${ACTUAL_OUTPUT_DIR}/test_backup.txt"
    
else
    echo "❌ 未找到目标目录"
fi

echo ""
echo "📊 测试结果:"
echo "  - 基础目录内容:"
ls -la "${OUTPUT_DIR_BASE}/"
echo ""
echo "  - 训练目录内容:"
if [ -n "${ACTUAL_OUTPUT_DIR}" ] && [ -d "${ACTUAL_OUTPUT_DIR}" ]; then
    ls -la "${ACTUAL_OUTPUT_DIR}/"
fi

echo ""
echo "🧹 清理测试环境..."
rm -rf "${OUTPUT_DIR_BASE}"
echo "✅ 测试完成" 