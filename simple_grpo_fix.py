#!/usr/bin/env python3
"""简单的GRPO设备错误修复"""

import torch
import logging

logger = logging.getLogger(__name__)

def apply_simple_grpo_fix(trainer):
    """应用简单的GRPO修复，专门解决eos_idx设备错误"""
    try:
        logger.info("🔧 应用简单GRPO修复...")
        
        if not hasattr(trainer, '_generate_and_score_completions'):
            logger.warning("⚠️ 未找到 _generate_and_score_completions 方法")
            return False
            
        original_method = trainer._generate_and_score_completions
        
        def patched_method(batch):
            """修复eos_idx设备错误的补丁"""
            try:
                # 直接调用原始方法
                return original_method(batch)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e) and "cuda:1 and cuda:0" in str(e):
                    logger.warning(f"🔧 检测到GRPO eos_idx设备错误，应用紧急修复: {e}")
                    
                    # 方法：强制所有输入到cuda:0，然后重试
                    try:
                        def force_to_cuda0(obj):
                            if torch.is_tensor(obj):
                                return obj.to('cuda:0', non_blocking=True)
                            elif isinstance(obj, dict):
                                return {k: force_to_cuda0(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return type(obj)(force_to_cuda0(item) for item in obj)
                            else:
                                return obj
                        
                        cuda0_batch = force_to_cuda0(batch)
                        logger.info("🔧 已强制所有张量到cuda:0，重试...")
                        
                        with torch.cuda.device('cuda:0'):
                            result = original_method(cuda0_batch)
                        
                        logger.info("✅ 紧急修复成功")
                        return result
                        
                    except Exception as fix_e:
                        logger.error(f"❌ 紧急修复失败: {fix_e}")
                        raise e
                else:
                    raise e
        
        # 应用补丁
        trainer._generate_and_score_completions = patched_method
        logger.info("✅ 简单GRPO修复已应用")
        return True
        
    except Exception as e:
        logger.error(f"❌ 应用简单GRPO修复失败: {e}")
        return False

def test_simple_fix():
    """测试简单修复方法"""
    print("🧪 测试简单GRPO修复...")
    
    # 模拟batch数据
    test_batch = [
        torch.randn(2, 10, device='cuda:0'),
        torch.randn(2, 10, device='cuda:0')
    ]
    
    # 模拟trainer
    class MockTrainer:
        def __init__(self):
            pass
        
        def _generate_and_score_completions(self, batch):
            # 模拟错误
            raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!")
    
    trainer = MockTrainer()
    
    # 应用修复
    success = apply_simple_grpo_fix(trainer)
    
    if success:
        try:
            result = trainer._generate_and_score_completions(test_batch)
            print("✅ 修复测试成功")
        except Exception as e:
            print(f"❌ 修复测试失败: {e}")
    else:
        print("❌ 修复应用失败")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_simple_fix()
    else:
        print("❌ CUDA不可用，无法测试") 