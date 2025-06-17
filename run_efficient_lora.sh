#!/bin/bash
# run_efficient_lora.sh - 高效LoRA训练快速启动脚本

echo "🚀 启动高效LoRA训练..."
echo "📊 配置亮点:"
echo "  - 批次大小: 8 (4倍提升)"
echo "  - LoRA rank: 128 (2倍提升)"
echo "  - 数据加载: 4 workers + 8倍预取"
echo "  - 设备优化: 减少同步警告"
echo "  - 预期效果: GPU利用率 >80%, 速度提升 2-3倍"
echo ""

# 检查GPU状态
echo "🔍 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "⚡ 开始训练..."

# 执行优化后的训练脚本
./run_model_parallel_only.sh 