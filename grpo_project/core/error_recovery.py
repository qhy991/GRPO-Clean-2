# grpo_project/core/error_recovery.py
import os
import json
import logging
import torch
import functools
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingErrorHandler:
    """增强的训练错误处理和恢复机制"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.error_log_dir = self.output_dir / "error_logs"
        self.error_log_dir.mkdir(exist_ok=True)
        
        self.recovery_strategies = {
            "memory_error": self._handle_memory_error,
            "cuda_error": self._handle_cuda_error,
            "checkpoint_error": self._handle_checkpoint_error,
            "parsing_error": self._handle_parsing_error,
            "generation_error": self._handle_generation_error,
            "simulation_error": self._handle_simulation_error
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        统一错误处理入口
        
        Args:
            error: 发生的异常
            context: 错误上下文信息
            
        Returns:
            bool: 是否成功恢复
        """
        error_type = self._classify_error(error)
        error_info = self._collect_error_info(error, context, error_type)
        
        # 记录错误信息
        self._log_error(error_info)
        
        # 尝试恢复
        recovery_fn = self.recovery_strategies.get(error_type, self._handle_generic_error)
        return recovery_fn(error, context, error_info)
    
    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        if "cuda" in error_str or "gpu" in error_str:
            return "cuda_error"
        elif "memory" in error_str or "oom" in error_str:
            return "memory_error"
        elif "checkpoint" in error_str or "state_dict" in error_str:
            return "checkpoint_error"
        elif "parse" in error_str or "parsing" in error_str:
            return "parsing_error"
        elif "generation" in error_str or "generate" in error_str:
            return "generation_error"
        elif "simulation" in error_str or "verilog" in error_str:
            return "simulation_error"
        else:
            return "generic_error"
    
    def _collect_error_info(self, error: Exception, context: Dict[str, Any], error_type: str) -> Dict[str, Any]:
        """收集详细错误信息"""
        return {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_class": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "system_info": self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            try:
                info.update({
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved(),
                    "max_memory_allocated": torch.cuda.max_memory_allocated(),
                })
            except Exception as e:
                logger.warning(f"Failed to get CUDA info: {e}")
        
        return info
    
    def _log_error(self, error_info: Dict[str, Any]):
        """记录错误到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.error_log_dir / f"error_{timestamp}.json"
        
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            logger.error(f"Error logged to: {error_file}")
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def _handle_memory_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理内存错误"""
        logger.warning("Handling memory error...")
        
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            logger.info("Forced garbage collection")
            
            return True
        except Exception as e:
            logger.error(f"Memory error recovery failed: {e}")
            return False
    
    def _handle_cuda_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理CUDA错误"""
        logger.warning("Handling CUDA error...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA device reset")
            return True
        except Exception as e:
            logger.error(f"CUDA error recovery failed: {e}")
            return False
    
    def _handle_checkpoint_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理checkpoint错误"""
        logger.warning("Handling checkpoint error...")
        
        try:
            if 'checkpoint_path' in context:
                checkpoint_path = Path(context['checkpoint_path'])
                backup_paths = list(checkpoint_path.parent.glob("checkpoint-*"))
                backup_paths.sort(reverse=True)
                
                for backup_path in backup_paths[:3]:
                    try:
                        if self._validate_checkpoint(backup_path):
                            logger.info(f"Found valid backup checkpoint: {backup_path}")
                            context['checkpoint_path'] = str(backup_path)
                            return True
                    except Exception as e:
                        logger.warning(f"Backup checkpoint {backup_path} invalid: {e}")
                        continue
            
            logger.warning("No valid backup checkpoint found, will start from scratch")
            return True
        except Exception as e:
            logger.error(f"Checkpoint error recovery failed: {e}")
            return False
    
    def _handle_parsing_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理解析错误"""
        logger.warning("Handling parsing error...")
        
        try:
            if 'sample_data' in context:
                problem_sample_dir = self.error_log_dir / "problem_samples"
                problem_sample_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_file = problem_sample_dir / f"parsing_error_{timestamp}.json"
                
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(context['sample_data'], f, indent=2, ensure_ascii=False)
                
                logger.info(f"Problem sample saved to: {sample_file}")
            
            return True
        except Exception as e:
            logger.error(f"Parsing error recovery failed: {e}")
            return False
    
    def _handle_generation_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理生成错误"""
        logger.warning("Handling generation error...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if 'generation_config' in context:
                gen_config = context['generation_config']
                logger.info(f"Generation config: {gen_config}")
            
            return True
        except Exception as e:
            logger.error(f"Generation error recovery failed: {e}")
            return False
    
    def _handle_simulation_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理仿真错误"""
        logger.warning("Handling simulation error...")
        
        try:
            if 'testbench_path' in context:
                tb_path = context['testbench_path']
                if not os.path.exists(tb_path):
                    logger.error(f"Testbench file not found: {tb_path}")
                    return False
                else:
                    logger.info(f"Testbench file exists: {tb_path}")
            
            return True
        except Exception as e:
            logger.error(f"Simulation error recovery failed: {e}")
            return False
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any], error_info: Dict[str, Any]) -> bool:
        """处理通用错误"""
        logger.warning(f"Handling generic error: {type(error).__name__}")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            logger.warning("Unknown error type, recommend manual intervention")
            return False
        except Exception as e:
            logger.error(f"Generic error recovery failed: {e}")
            return False
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """验证checkpoint文件完整性"""
        required_files = [
            "adapter_config.json",
            "training_args.bin",
            "trainer_state.json"
        ]
        
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            if not file_path.exists():
                return False
        
        try:
            with open(checkpoint_path / "trainer_state.json", 'r') as f:
                state = json.load(f)
                return 'global_step' in state
        except Exception:
            return False


def with_error_recovery(error_handler: Optional[TrainingErrorHandler] = None):
    """错误恢复装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # 如果没有传入error_handler，尝试从self获取
                handler = error_handler
                if handler is None and hasattr(self, 'error_handler'):
                    handler = self.error_handler
                
                if handler is None:
                    logger.error(f"No error handler available for {func.__name__}")
                    raise e
                
                context = {
                    "function": func.__name__,
                    "class": self.__class__.__name__ if hasattr(self, '__class__') else 'unknown',
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()) if kwargs else []
                }
                
                logger.error(f"Error in {func.__name__}: {e}")
                
                if handler.handle_error(e, context):
                    logger.info(f"Error recovered in {func.__name__}, retrying...")
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed in {func.__name__}: {retry_error}")
                        raise retry_error
                else:
                    logger.error(f"Error recovery failed in {func.__name__}")
                    raise e
        return wrapper
    return decorator


# 简化版的错误恢复装饰器，不需要error_handler参数
def safe_execution(func: Callable) -> Callable:
    """简单的安全执行装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            
            # 基本的清理操作
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
            
            # 重新抛出异常，让上层处理
            raise e
    return wrapper