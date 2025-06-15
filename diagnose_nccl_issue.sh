#!/bin/bash
# diagnose_nccl_issue.sh - 诊断NCCL和分布式训练问题

echo "🔍 NCCL和分布式训练环境诊断"
echo "=================================="

# 1. GPU基本信息
echo ""
echo "📊 GPU信息:"
nvidia-smi
echo ""

# 2. GPU拓扑结构
echo "🔗 GPU拓扑结构:"
nvidia-smi topo -m
echo ""

# 3. CUDA版本信息
echo "💻 CUDA信息:"
nvcc --version
echo ""

# 4. PyTorch和NCCL版本
echo "🐍 Python环境信息:"
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'NCCL版本: {torch.cuda.nccl.version()}')
print(f'可用GPU数量: {torch.cuda.device_count()}')
print(f'分布式可用: {torch.distributed.is_available()}')
print(f'NCCL后端可用: {torch.distributed.is_nccl_available()}')
"
echo ""

# 5. 测试GPU通信
echo "🧪 测试GPU间通信..."
python3 -c "
import torch
import torch.distributed as dist
import os

if torch.cuda.device_count() >= 2:
    print('✅ 检测到多个GPU，测试基本通信...')
    
    # 简单的张量操作测试
    try:
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')
        
        tensor0 = torch.randn(1000, 1000, device=device0)
        tensor1 = tensor0.to(device1)
        result = tensor1.sum()
        
        print(f'✅ GPU间数据传输测试成功: {result.item():.2f}')
    except Exception as e:
        print(f'❌ GPU间数据传输测试失败: {e}')
else:
    print('⚠️ 只检测到一个GPU，无法测试多GPU通信')
"
echo ""

# 6. 内存使用情况
echo "💾 当前GPU内存使用:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo ""

# 7. 系统资源
echo "🖥️ 系统资源:"
echo "CPU核心数: $(nproc)"
echo "内存信息:"
free -h
echo ""

# 8. 网络接口（对InfiniBand重要）
echo "🌐 网络接口:"
ip addr show | grep -E "^[0-9]+:|inet "
echo ""

# 9. NCCL测试建议
echo "🔧 NCCL问题解决建议:"
echo "1. 如果GPU拓扑显示PCIe连接，考虑设置:"
echo "   export NCCL_P2P_DISABLE=1"
echo ""
echo "2. 如果没有InfiniBand，设置:"
echo "   export NCCL_IB_DISABLE=1"
echo ""
echo "3. 对于调试，启用详细日志:"
echo "   export NCCL_DEBUG=INFO"
echo ""
echo "4. 增加超时时间:"
echo "   export NCCL_TIMEOUT=7200"
echo ""
echo "5. 检查是否有进程占用GPU:"
nvidia-smi pmon -c 1
echo ""

echo "✅ 诊断完成！请检查上述输出中的异常信息。" 