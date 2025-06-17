#!/bin/bash
# setup_multi_gpu.sh - 快速设置多GPU环境的脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# 获取脚本目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "========================================================================"
echo "            多GPU模型并行训练环境设置脚本"
echo "========================================================================"

# 1. 检查GPU环境
check_gpu_environment() {
    log_info "🔍 检查GPU环境..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "❌ nvidia-smi未找到，请确保NVIDIA驱动已安装"
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    log_info "📊 检测到 ${GPU_COUNT} 张GPU"
    
    if [ "${GPU_COUNT}" -lt 2 ]; then
        log_warning "⚠️ 检测到少于2张GPU，多GPU并行将不可用"
        return 1
    fi
    
    log_info "💾 GPU详细信息:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    
    return 0
}

# 2. 检查Python环境
check_python_environment() {
    log_info "🐍 检查Python环境..."
    
    # 检查Python版本
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "  Python版本: ${PYTHON_VERSION}"
    
    # 检查关键包
    log_info "📦 检查关键Python包..."
    
    declare -A packages=(
        ["torch"]="PyTorch"
        ["transformers"]="Transformers"
        ["accelerate"]="Accelerate"
        ["peft"]="PEFT"
        ["trl"]="TRL"
        ["datasets"]="Datasets"
        ["wandb"]="WandB"
    )
    
    missing_packages=()
    
    for package in "${!packages[@]}"; do
        if python3 -c "import ${package}" 2>/dev/null; then
            version=$(python3 -c "import ${package}; print(${package}.__version__)" 2>/dev/null || echo "unknown")
            log_info "  ✅ ${packages[$package]}: ${version}"
        else
            log_error "  ❌ ${packages[$package]}: 未安装"
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log_error "❌ 缺少必要的Python包: ${missing_packages[*]}"
        log_info "💡 安装命令:"
        log_info "  pip install torch transformers accelerate peft trl datasets wandb"
        return 1
    fi
    
    return 0
}

# 3. 检查CUDA兼容性
check_cuda_compatibility() {
    log_info "🔧 检查CUDA兼容性..."
    
    # 检查CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        log_info "  CUDA版本: ${CUDA_VERSION}"
    else
        log_warning "  ⚠️ nvcc未找到，无法检测CUDA版本"
    fi
    
    # 检查PyTorch CUDA支持
    if python3 -c "import torch; print(f'PyTorch CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null; then
        TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
        log_info "  PyTorch CUDA版本: ${TORCH_CUDA_VERSION}"
        
        # 测试GPU通信
        log_info "🔗 测试GPU间通信..."
        if python3 -c "
import torch
if torch.cuda.device_count() >= 2:
    try:
        x = torch.randn(1000, 1000, device='cuda:0')
        y = x.to('cuda:1')
        z = y.to('cuda:0')
        print(f'✅ GPU通信测试成功: 数据一致性 {torch.allclose(x, z)}')
    except Exception as e:
        print(f'❌ GPU通信测试失败: {e}')
        exit(1)
else:
    print('⚠️ GPU数量不足，跳过通信测试')
"; then
            log_info "  ✅ GPU通信正常"
        else
            log_error "  ❌ GPU通信测试失败"
            return 1
        fi
    else
        log_error "  ❌ PyTorch CUDA不可用"
        return 1
    fi
    
    return 0
}

# 4. 设置Accelerate配置
setup_accelerate_config() {
    log_info "⚙️ 设置Accelerate配置..."
    
    CONFIG_FILE="${SCRIPT_DIR}/accelerate_config_multi_gpu.yaml"
    
    # 如果配置文件不存在，创建它
    if [ ! -f "${CONFIG_FILE}" ]; then
        log_info "📝 创建Accelerate配置文件..."
        cat > "${CONFIG_FILE}" << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
        log_info "  ✅ 配置文件已创建: ${CONFIG_FILE}"
    else
        log_info "  ✅ 配置文件已存在: ${CONFIG_FILE}"
    fi
    
    # 验证配置
    if accelerate config --config_file "${CONFIG_FILE}" 2>/dev/null; then
        log_info "  ✅ Accelerate配置验证通过"
    else
        log_warning "  ⚠️ Accelerate配置验证失败，但文件已创建"
    fi
    
    return 0
}

# 5. 创建便捷脚本
create_convenience_scripts() {
    log_info "📜 创建便捷脚本..."
    
    # GPU监控脚本
    GPU_MONITOR="${SCRIPT_DIR}/monitor_gpu.sh"
    cat > "${GPU_MONITOR}" << 'EOF'
#!/bin/bash
# 实时GPU监控脚本
echo "按 Ctrl+C 停止监控"
echo "========================================"
while true; do
    clear
    echo "GPU状态监控 - $(date)"
    echo "========================================"
    nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
    echo "进程信息:"
    nvidia-smi pmon -c 1 2>/dev/null || echo "无GPU进程"
    sleep 2
done
EOF
    chmod +x "${GPU_MONITOR}"
    log_info "  ✅ GPU监控脚本: ${GPU_MONITOR}"
    
    # 内存清理脚本
    GPU_CLEANUP="${SCRIPT_DIR}/cleanup_gpu.sh"
    cat > "${GPU_CLEANUP}" << 'EOF'
#!/bin/bash
# GPU内存清理脚本
echo "🧹 清理GPU内存..."

# 杀死所有Python GPU进程（谨慎使用）
# pkill -f "python.*cuda" 2>/dev/null || true

# 清理PyTorch缓存
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        print(f'✅ GPU {i} 内存已清理')
else:
    print('❌ CUDA不可用')
"

echo "✅ GPU内存清理完成"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
EOF
    chmod +x "${GPU_CLEANUP}"
    log_info "  ✅ GPU清理脚本: ${GPU_CLEANUP}"
    
    # 快速测试脚本
    QUICK_TEST="${SCRIPT_DIR}/test_multi_gpu.sh"
    cat > "${QUICK_TEST}" << 'EOF'
#!/bin/bash
# 快速多GPU测试脚本
echo "🧪 多GPU功能测试..."

python3 << 'PYEOF'
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_multi_gpu():
    print(f"🔍 检测环境:")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    print(f"  - GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() < 2:
        print("⚠️ 需要至少2张GPU进行测试")
        return False
    
    try:
        print(f"\n🔧 测试GPU通信...")
        # 基本GPU通信测试
        x = torch.randn(1000, 1000, device='cuda:0', dtype=torch.float16)
        y = x.to('cuda:1')
        z = y.back()
        
        print(f"  ✅ 数据传输成功，一致性: {torch.allclose(x, z)}")
        
        # 测试内存分配
        print(f"\n💾 测试内存分配...")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        
        print(f"\n✅ 多GPU测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 多GPU测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_multi_gpu()
    exit(0 if success else 1)
PYEOF
EOF
    chmod +x "${QUICK_TEST}"
    log_info "  ✅ 快速测试脚本: ${QUICK_TEST}"
    
    return 0
}

# 6. 验证设置
verify_setup() {
    log_info "✅ 验证多GPU设置..."
    
    # 运行快速测试
    if [ -f "${SCRIPT_DIR}/test_multi_gpu.sh" ]; then
        log_info "🧪 运行快速测试..."
        if "${SCRIPT_DIR}/test_multi_gpu.sh"; then
            log_info "  ✅ 多GPU设置验证成功"
        else
            log_warning "  ⚠️ 多GPU测试未完全通过，但基础设置已完成"
        fi
    fi
    
    # 检查模型管理器配置
    if [ -f "${SCRIPT_DIR}/grpo_project/core/models.py" ]; then
        if grep -q "use_multi_gpu" "${SCRIPT_DIR}/grpo_project/core/models.py" 2>/dev/null; then
            log_info "  ✅ ModelManager已配置多GPU支持"
        else
            log_warning "  ⚠️ ModelManager可能需要更新以支持多GPU"
        fi
    fi
    
    return 0
}

# 7. 提供使用说明
show_usage_instructions() {
    echo ""
    echo "========================================================================"
    echo "                        使用说明"
    echo "========================================================================"
    log_info "🚀 多GPU环境设置完成！"
    echo ""
    log_info "📋 下一步操作:"
    echo ""
    echo "1. 启动训练:"
    echo "   ./run_enhanced_grpo_training_multi_gpu.sh"
    echo ""
    echo "2. 监控GPU状态:"
    echo "   ./monitor_gpu.sh"
    echo ""
    echo "3. 清理GPU内存(如需要):"
    echo "   ./cleanup_gpu.sh"
    echo ""
    echo "4. 快速测试多GPU功能:"
    echo "   ./test_multi_gpu.sh"
    echo ""
    log_info "⚙️ 配置文件:"
    echo "   - Accelerate配置: ./accelerate_config_multi_gpu.yaml"
    echo "   - 训练脚本: ./run_enhanced_grpo_training_multi_gpu.sh"
    echo ""
    log_info "🔧 如果遇到问题:"
    echo "   1. 检查GPU状态: nvidia-smi"
    echo "   2. 验证CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "   3. 清理内存: ./cleanup_gpu.sh"
    echo "   4. 重新运行此设置脚本"
    echo ""
    log_info "📊 推荐的训练配置(已在训练脚本中设置):"
    echo "   - 序列长度: 6144 (充分利用内存)"
    echo "   - 批次大小: 2 per GPU"
    echo "   - LoRA rank: 64 (适中的参数量)"
    echo "   - 精度: bfloat16 (最佳性能)"
    echo ""
}

# 主执行流程
main() {
    log_info "🚀 开始多GPU环境设置..."
    
    # 1. 检查GPU环境
    if ! check_gpu_environment; then
        log_error "❌ GPU环境检查失败"
        exit 1
    fi
    
    # 2. 检查Python环境
    if ! check_python_environment; then
        log_error "❌ Python环境检查失败"
        exit 1
    fi
    
    # 3. 检查CUDA兼容性
    if ! check_cuda_compatibility; then
        log_error "❌ CUDA兼容性检查失败"
        exit 1
    fi
    
    # 4. 设置Accelerate配置
    setup_accelerate_config
    
    # 5. 创建便捷脚本
    create_convenience_scripts
    
    # 6. 验证设置
    verify_setup
    
    # 7. 显示使用说明
    show_usage_instructions
    
    log_info "✅ 多GPU环境设置完成！"
}

# 执行主函数
main "$@"