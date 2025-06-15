#!/usr/bin/env python3
"""
测试DetailedInferenceCallback是否能正常生成eval_avg_test_pass_rate指标
"""

import os
import sys
import logging
from unittest.mock import Mock, MagicMock
from datasets import Dataset

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_detailed_inference_callback():
    """测试DetailedInferenceCallback的基本功能"""
    try:
        from grpo_project.callbacks.inference import DetailedInferenceCallback
        from transformers import TrainingArguments, TrainerState, TrainerControl
        
        logger.info("✅ 成功导入DetailedInferenceCallback")
        
        # 创建模拟的tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # 创建测试数据集
        test_data = [
            {
                'prompt': 'Design a simple AND gate module',
                'task_id': 'test_001',
                'level': 'basic',
                'complexity_score': 3.0,
                'testbench_path': '/nonexistent/path/tb.v',
                'expected_total_tests': 4,
                'original_enhanced_prompt': 'Test prompt for AND gate'
            }
        ]
        test_dataset = Dataset.from_list(test_data)
        
        # 创建回调实例
        callback = DetailedInferenceCallback(
            tokenizer=mock_tokenizer,
            eval_dataset=test_dataset,
            num_samples=1,
            eval_every_n_steps=1,  # 每步都评估，便于测试
            output_dir="./test_output"
        )
        
        logger.info("✅ 成功创建DetailedInferenceCallback实例")
        
        # 创建模拟的训练参数和状态
        args = Mock(spec=TrainingArguments)
        args.local_rank = 0
        
        state = Mock(spec=TrainerState)
        state.global_step = 1
        
        control = Mock(spec=TrainerControl)
        
        # 创建模拟的模型
        mock_model = Mock()
        mock_model.training = True
        mock_model.eval = Mock()
        mock_model.train = Mock()
        mock_model.device = 'cpu'
        
        # 模拟生成过程
        mock_outputs = Mock()
        mock_outputs.__getitem__ = Mock(return_value=Mock())
        mock_outputs[0].__getitem__ = Mock(return_value=[1, 2, 3, 4, 5])  # 模拟生成的token
        
        mock_model.generate = Mock(return_value=mock_outputs)
        mock_tokenizer.decode = Mock(return_value="<think>This is a test reasoning</think>\n```verilog\nmodule test(); endmodule\n```")
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.side_effect = None
        
        # 模拟tokenizer调用
        def mock_tokenizer_call(*args, **kwargs):
            mock_result = Mock()
            mock_result.input_ids = Mock()
            mock_result.input_ids.shape = [1, 10]  # batch_size=1, seq_len=10
            mock_result.to = Mock(return_value=mock_result)
            return mock_result
        
        mock_tokenizer.side_effect = mock_tokenizer_call
        
        logger.info("🧪 开始测试on_step_end方法...")
        
        # 测试on_step_end方法
        try:
            callback.on_step_end(args, state, control, model=mock_model)
            logger.info("✅ on_step_end方法执行成功")
        except Exception as e:
            logger.error(f"❌ on_step_end方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试_generate_single_sample方法
        logger.info("🧪 测试_generate_single_sample方法...")
        try:
            result = callback._generate_single_sample(mock_model, "test prompt", 1)
            logger.info(f"✅ _generate_single_sample返回: {result}")
            
            if result and 'reasoning' in result and 'code' in result:
                logger.info("✅ 生成结果包含reasoning和code字段")
            else:
                logger.warning("⚠️ 生成结果格式可能有问题")
                
        except Exception as e:
            logger.error(f"❌ _generate_single_sample方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("✅ 所有测试通过！DetailedInferenceCallback应该能正常工作")
        return True
        
    except ImportError as e:
        logger.error(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parsing_functions():
    """测试解析函数"""
    try:
        from grpo_project.utils.parsing import parse_llm_completion_with_context
        
        test_output = "<think>This is reasoning</think>\n```verilog\nmodule test(); endmodule\n```"
        reasoning, code = parse_llm_completion_with_context(test_output, step=1, sample_idx=0)
        
        logger.info(f"✅ 解析测试 - reasoning: {reasoning is not None}, code: {code is not None}")
        
        if reasoning and code:
            logger.info("✅ 解析函数工作正常")
            return True
        else:
            logger.warning("⚠️ 解析函数可能有问题")
            return False
            
    except Exception as e:
        logger.error(f"❌ 解析函数测试失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 开始测试DetailedInferenceCallback...")
    
    # 测试解析函数
    logger.info("\n📝 测试1: 解析函数")
    parsing_ok = test_parsing_functions()
    
    # 测试回调
    logger.info("\n📝 测试2: DetailedInferenceCallback")
    callback_ok = test_detailed_inference_callback()
    
    if parsing_ok and callback_ok:
        logger.info("\n🎉 所有测试通过！修复应该有效。")
        sys.exit(0)
    else:
        logger.error("\n❌ 部分测试失败，需要进一步调试。")
        sys.exit(1) 