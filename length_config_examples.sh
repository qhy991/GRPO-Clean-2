#!/bin/bash

# 长度配置示例文件
# 这个文件展示了不同的长度配置选项，可以复制到主训练脚本中使用

echo "📏 GRPO训练长度配置示例"
echo "========================================"
echo ""

# 🔧 配置1: 当前默认配置 (推荐用于一般Verilog生成)
echo "配置1: 默认配置 - 适合一般Verilog模块生成"
echo "MAX_SEQ_LENGTH=4096"
echo "MAX_PROMPT_LENGTH=1536    # ~37.5%"
echo "MAX_COMPLETION_LENGTH=2560 # ~62.5%"
echo "LENGTH_ALLOCATION_STRATEGY=\"custom\""
echo ""

# 🔧 配置2: 平衡配置
echo "配置2: 平衡配置 - 提示和输出各占50%"
echo "MAX_SEQ_LENGTH=4096"
echo "LENGTH_ALLOCATION_STRATEGY=\"balanced\""
echo "# MAX_PROMPT_LENGTH=2048    # 自动设置"
echo "# MAX_COMPLETION_LENGTH=2048 # 自动设置"
echo ""

# 🔧 配置3: 长输出配置 (推荐用于复杂Verilog)
echo "配置3: 长输出配置 - 适合生成复杂/长Verilog代码"
echo "MAX_SEQ_LENGTH=4096"
echo "MAX_PROMPT_LENGTH=1280     # ~31%"
echo "MAX_COMPLETION_LENGTH=2816 # ~69%"
echo "LENGTH_ALLOCATION_STRATEGY=\"custom\""
echo ""

# 🔧 配置4: 超长输出配置
echo "配置4: 超长输出配置 - 适合生成非常长的Verilog模块"
echo "MAX_SEQ_LENGTH=4096"
echo "MAX_PROMPT_LENGTH=1024     # ~25%"
echo "MAX_COMPLETION_LENGTH=3072 # ~75%"
echo "LENGTH_ALLOCATION_STRATEGY=\"custom\""
echo ""

# 🔧 配置5: 扩展序列长度 + 超长输出
echo "配置5: 扩展序列长度 - 适合生成大型Verilog设计"
echo "MAX_SEQ_LENGTH=6144"
echo "MAX_PROMPT_LENGTH=1536     # ~25%"
echo "MAX_COMPLETION_LENGTH=4608 # ~75%"
echo "LENGTH_ALLOCATION_STRATEGY=\"custom\""
echo ""

# 🔧 配置6: 巨型配置 (需要更多GPU内存)
echo "配置6: 巨型配置 - 适合生成超大型Verilog项目"
echo "MAX_SEQ_LENGTH=8192"
echo "MAX_PROMPT_LENGTH=2048     # ~25%"
echo "MAX_COMPLETION_LENGTH=6144 # ~75%"
echo "LENGTH_ALLOCATION_STRATEGY=\"custom\""
echo ""

echo "========================================"
echo "💡 使用方法:"
echo "1. 选择适合您需求的配置"
echo "2. 复制相应的配置行到 run_enhanced_grpo_training.sh"
echo "3. 注释掉原有的长度配置"
echo "4. 运行训练脚本"
echo ""
echo "📊 性能建议:"
echo "- 配置1-3: 适合大多数单GPU/双GPU训练"
echo "- 配置4-5: 建议4张以上GPU"
echo "- 配置6: 建议8张以上高端GPU (A100/H100)"
echo ""
echo "🔧 内存估算 (大致):"
echo "- 4K序列: ~16GB GPU内存 (每张GPU, batch_size=2)"
echo "- 6K序列: ~24GB GPU内存 (每张GPU, batch_size=2)"
echo "- 8K序列: ~32GB GPU内存 (每张GPU, batch_size=2)"
echo ""

# 🔧 显示当前配置函数
show_current_config() {
    if [ -f "run_enhanced_grpo_training.sh" ]; then
        echo "当前训练脚本中的长度配置:"
        grep -A 10 "MAX_SEQ_LENGTH=" run_enhanced_grpo_training.sh | head -4
    else
        echo "⚠️ 未找到训练脚本文件"
    fi
}

# 如果脚本被直接执行，显示当前配置
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    show_current_config
fi 