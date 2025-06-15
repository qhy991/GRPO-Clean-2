# grpo_project/core/recovery_wrapper.py
import functools
from typing import Callable, Any

def with_error_recovery(error_handler: TrainingErrorHandler):
    """错误恢复装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # 限制长度
                    "kwargs": str(kwargs)[:200]
                }
                
                if error_handler.handle_error(e, context):
                    logger.info(f"Error recovered in {func.__name__}, retrying...")
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed in {func.__name__}: {retry_error}")
                        raise retry_error
                else:
                    logger.error(f"Error recovery failed in {func.__name__}")
                    raise e
        return wrapper
    return decorator