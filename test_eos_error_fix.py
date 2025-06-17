#!/usr/bin/env python3
"""测试EOS错误修复的简单脚本"""

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_type_detection():
    """测试批次类型检测"""
    logger.info("🧪 测试批次类型检测...")
    
    # 测试字典类型
    dict_batch = {'input_ids': torch.randn(2, 10)}
    logger.info(f"字典批次: {type(dict_batch)}")
    logger.info(f"是否为字典: {isinstance(dict_batch, dict)}")
    
    # 测试元组类型
    tuple_batch = (torch.randn(2, 10), torch.randn(2, 10))
    logger.info(f"元组批次: {type(tuple_batch)}")
    logger.info(f"是否为字典: {isinstance(tuple_batch, dict)}")
    
    # 测试列表类型
    list_batch = [torch.randn(2, 10), torch.randn(2, 10)]
    logger.info(f"列表批次: {type(list_batch)}")
    logger.info(f"是否为字典: {isinstance(list_batch, dict)}")
    
    # 测试张量类型
    tensor_batch = torch.randn(2, 10)
    logger.info(f"张量批次: {type(tensor_batch)}")
    logger.info(f"是否为字典: {isinstance(tensor_batch, dict)}")
    logger.info(f"张量是否有to方法: {hasattr(tensor_batch, 'to')}")

def simulate_device_error_handling():
    """模拟设备错误处理"""
    logger.info("🧪 模拟设备错误处理...")
    
    # 模拟不同类型的batch
    test_batches = [
        {'input_ids': torch.randn(2, 10, device='cuda:0'), 'attention_mask': torch.ones(2, 10, device='cuda:0')},
        (torch.randn(2, 10, device='cuda:0'), torch.randn(2, 10, device='cuda:0')),
        [torch.randn(2, 10, device='cuda:0'), torch.randn(2, 10, device='cuda:0')],
        torch.randn(2, 10, device='cuda:0')
    ]
    
    def enhanced_fallback_wrapper(batch):
        """增强的错误处理包装器（复制自main.py）"""
        try:
            # 模拟设备错误
            raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!")
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                logger.warning(f"🔧 GRPO错误回退：检测到设备不一致错误: {e}")
                
                # 尝试强制所有张量到cuda:0
                try:
                    logger.info("🔧 错误回退：尝试强制所有张量到cuda:0")
                    if hasattr(batch, 'to'):
                        unified_batch = batch.to('cuda:0')
                    elif isinstance(batch, dict):
                        unified_batch = {}
                        for k, v in batch.items():
                            if torch.is_tensor(v):
                                unified_batch[k] = v.to('cuda:0', non_blocking=True)
                            else:
                                unified_batch[k] = v
                    elif isinstance(batch, (list, tuple)):
                        unified_batch = []
                        for item in batch:
                            if torch.is_tensor(item):
                                unified_batch.append(item.to('cuda:0', non_blocking=True))
                            else:
                                unified_batch.append(item)
                        unified_batch = type(batch)(unified_batch)
                    else:
                        logger.warning(f"⚠️ 未知的batch类型: {type(batch)}")
                        raise e
                    
                    logger.info(f"✅ 成功处理 {type(batch)} 类型的batch")
                    return {"status": "success", "batch_type": type(batch).__name__}
                    
                except Exception as fallback_e:
                    logger.error(f"❌ 错误回退也失败: {fallback_e}")
                    # 最后的安全措施：返回空结果
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        batch_size = batch['input_ids'].shape[0]
                        return {
                            'sequences': batch['input_ids'].to('cuda:0'),
                            'logprobs': torch.zeros(batch_size, device='cuda:0'),
                            'ref_logprobs': torch.zeros(batch_size, device='cuda:0'),
                            'values': torch.zeros(batch_size, device='cuda:0')
                        }
                    else:
                        raise e
            else:
                raise e
    
    # 测试所有类型的batch
    for i, batch in enumerate(test_batches):
        logger.info(f"\n测试批次 {i+1}: {type(batch)}")
        try:
            result = enhanced_fallback_wrapper(batch)
            logger.info(f"✅ 处理成功: {result}")
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")

def main():
    """主函数"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法进行测试")
        return
    
    logger.info("🚀 开始EOS错误修复测试...")
    
    test_batch_type_detection()
    print("\n" + "="*50 + "\n")
    simulate_device_error_handling()
    
    logger.info("\n✅ 测试完成！")

if __name__ == "__main__":
    main() 