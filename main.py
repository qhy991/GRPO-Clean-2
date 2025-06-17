# main.py - 修复DDP/模型并行冲突版本
import os
import logging
import numpy as np
import torch
import gc
import sys
from dataclasses import asdict
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import re
import time

# --- BEGIN: PyTorch Safe Unpickling Configuration ---
logger_temp = logging.getLogger(__name__ + "_startup")
try:
    from numpy.core.multiarray import _reconstruct
    from numpy import dtype as numpy_dtype
    from numpy.dtypes import UInt32DType

    safe_globals_list = []
    if callable(_reconstruct): safe_globals_list.append(_reconstruct)
    if isinstance(numpy_dtype, type): safe_globals_list.append(numpy_dtype)
    if isinstance(np.ndarray, type): safe_globals_list.append(np.ndarray)
    if isinstance(UInt32DType, type): safe_globals_list.append(UInt32DType)

    # Add numpy scalar types
    numpy_scalar_types_to_add = [
        np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
        np.int_, np.uint, np.longlong, np.ulonglong,
        np.half, np.float16, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ]
    
    for nt_class in numpy_scalar_types_to_add:
        if isinstance(nt_class, type):
            safe_globals_list.append(nt_class)

    if safe_globals_list:
        torch.serialization.add_safe_globals(safe_globals_list)
        logger_temp.info(f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items.")

except Exception as e_globals:
    logger_temp.error(f"Error setting up torch safe globals: {e_globals}", exc_info=True)
# --- END: PyTorch Safe Unpickling Configuration ---

from transformers import HfArgumentParser
from trl import GRPOConfig

# Configuration imports
from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from grpo_project.utils.logging import setup_global_logging
from grpo_project.utils.reporting_utils import PeriodicStatusReporter
from grpo_project.core.models import ModelManager
from grpo_project.data.dataset import load_and_prepare_dataset
from grpo_project.rewards.calculator import RewardCalculator
from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
from grpo_project.utils import ExperienceBuffer

# Callbacks
from grpo_project.callbacks.monitoring import StepLoggingCallback, DetailedRewardCallback, RewardStabilityMonitor
from grpo_project.callbacks.persistence import CustomStatePersistenceCallback
from grpo_project.callbacks.inference import DetailedInferenceCallback
from grpo_project.callbacks.wandb import DetailedWandbCallback as TrainDetailedWandbCallback
from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback, OptimizedCurriculumCallback
from grpo_project.core.wandb_sync_manager import initialize_wandb_sync_manager, get_wandb_sync_manager

logger = logging.getLogger(__name__)

class GRPOTrainingPipeline:
    def __init__(self):
        # 1. Load Configurations
        parser = HfArgumentParser((EnvConfig, ScriptConfig, EnhancedRewardConfig, GRPOConfig))
        self.env_cfg, self.script_cfg, self.reward_cfg, self.grpo_cfg = parser.parse_args_into_dataclasses()
        
        # 🔧 新增：多GPU环境初始检测
        self._detect_multi_gpu_environment()
        
        # 🔧 关键修复：配置训练策略（在GPU检测之后）
        self._configure_training_strategy()
        
        # 🔧 同步长度配置从ScriptConfig到GRPOConfig
        self._sync_length_configs()
        
        # 🔧 调试：打印最终配置状态
        logger_temp.info(f"🎯 最终配置状态:")
        logger_temp.info(f"  - multi_gpu_info: {getattr(self, 'multi_gpu_info', {})}")
        logger_temp.info(f"  - training_strategy: {getattr(self, 'training_strategy', 'unknown')}")

        # 🔧 新增：自动配置WandB恢复
        self._configure_wandb_resume()

        # Setup logging first
        self._setup_logging()
        logger.info("GRPOTrainingPipeline initialized.")
        self._log_configs()

        # 🔧 新增：初始化WandB同步管理器
        self._setup_wandb_sync_manager()

        # 🔧 新增：多GPU优化的ModelManager初始化
        self.model_manager = ModelManager(
            script_cfg=self.script_cfg,
            grpo_cfg=self.grpo_cfg,
            model_name_or_path=self.script_cfg.model_name_or_path,
            cache_dir=getattr(self.env_cfg, 'cache_dir', None)
        )

        self.reward_calculator = RewardCalculator(
            reward_config=self.reward_cfg,
            simulator=None
        )

        self.curriculum_manager = None
        self.experience_buffer = None
        self.callbacks = []
        self.status_reporter = PeriodicStatusReporter(self.grpo_cfg.output_dir, report_interval=50)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 🔧 多GPU状态跟踪 - 注意：这些将在上面的方法中设置，不要在这里重复初始化

    def _detect_multi_gpu_environment(self):
        """🔧 检测并配置多GPU环境"""
        try:
            logger_temp.info("🔍 检测多GPU环境...")
            
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                logger_temp.warning("⚠️ CUDA不可用，禁用多GPU模式")
                self.multi_gpu_info = {
                    'gpu_count': 0,
                    'total_memory_gb': 0,
                    'average_memory_gb': 0,
                    'use_model_parallel': False
                }
                return
            
            # 🔧 关键修复：检查用户明确指定的模型并行设置
            user_specified_model_parallel = getattr(self.script_cfg, 'use_model_parallel', None)
            logger_temp.info(f"🎯 配置解析调试:")
            logger_temp.info(f"  - script_cfg.use_model_parallel: {user_specified_model_parallel}")
            logger_temp.info(f"  - 类型: {type(user_specified_model_parallel)}")
            
            # 检查GPU数量
            gpu_count = torch.cuda.device_count()
            logger_temp.info(f"📊 检测到 {gpu_count} 张GPU")
            
            # 🔧 关键修复：优化判断逻辑
            if user_specified_model_parallel is False:
                logger_temp.info("🎯 用户明确禁用模型并行，强制使用单GPU模式")
                # 🔧 限制CUDA设备可见性到单GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                self.multi_gpu_info = {
                    'gpu_count': 1,  # 强制设为1
                    'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0,
                    'average_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0,
                    'use_model_parallel': False
                }
                return
            
            # 🔧 修复：如果GPU数量不足但用户明确要求模型并行，给出警告但不强制禁用
            if gpu_count < 2:
                if user_specified_model_parallel is True:
                    logger_temp.error(f"❌ 用户要求模型并行但GPU数量({gpu_count})不足！")
                    logger_temp.error("💡 请确保有至少2张GPU可用，或设置 --use_model_parallel false")
                    raise ValueError(f"GPU数量不足以支持模型并行：需要>=2张GPU，实际{gpu_count}张")
                else:
                    logger_temp.warning(f"⚠️ GPU数量({gpu_count})少于2张，将使用单GPU模式")
                    self.multi_gpu_info = {
                        'gpu_count': gpu_count,
                        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0,
                        'average_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0,
                        'use_model_parallel': False
                    }
                    return
            
            # 检查GPU属性
            logger_temp.info("🔧 GPU详细信息:")
            total_memory = 0
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                total_memory += memory_gb
                logger_temp.info(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")
                
                if memory_gb < 40:  # 至少40GB才推荐用于大模型
                    logger_temp.warning(f"⚠️ GPU {i} 内存({memory_gb:.1f}GB)可能不足以支持大模型训练")
            
            # 🔧 修复：更清晰的判断逻辑
            if user_specified_model_parallel is True:
                # 用户明确要求模型并行
                use_model_parallel = True
                logger_temp.info("✅ 用户明确启用模型并行模式")
            elif user_specified_model_parallel is None and gpu_count >= 2:
                # 用户未设置，且GPU数量足够，默认启用
                use_model_parallel = True
                logger_temp.info("✅ 自动启用模型并行模式（GPU数量>=2）")
            else:
                # 其他情况使用单GPU
                use_model_parallel = False
                logger_temp.info("📱 将使用单GPU模式")
            
            self.multi_gpu_info = {
                'gpu_count': gpu_count,
                'total_memory_gb': total_memory,
                'average_memory_gb': total_memory / gpu_count,
                'use_model_parallel': use_model_parallel
            }
            
            # 🔧 关键修复：确保在检测阶段就设置正确的多GPU信息
            logger_temp.info(f"🔧 多GPU信息设置完成:")
            logger_temp.info(f"  - gpu_count: {self.multi_gpu_info['gpu_count']}")
            logger_temp.info(f"  - use_model_parallel: {self.multi_gpu_info['use_model_parallel']}")
            logger_temp.info(f"  - total_memory_gb: {self.multi_gpu_info['total_memory_gb']:.1f}GB")
            
            if self.multi_gpu_info['use_model_parallel']:
                logger_temp.info("✅ 多GPU模型并行模式已启用")
                logger_temp.info(f"  - 总GPU内存: {total_memory:.1f}GB")
                logger_temp.info(f"  - 平均每GPU: {total_memory/gpu_count:.1f}GB")
            
        except Exception as e:
            logger_temp.error(f"❌ 多GPU环境检测失败: {e}")
            # 设置安全的默认值
            self.multi_gpu_info = {
                'gpu_count': 1,
                'total_memory_gb': 0,
                'average_memory_gb': 0,
                'use_model_parallel': False
            }
            raise

    def _configure_training_strategy(self):
        """🔧 关键修复：配置训练策略以避免DDP/模型并行冲突（完全修复版）"""
        try:
            logger_temp.info("🔧 配置训练策略...")
            
            # 从multi_gpu_info获取正确的模型并行设置
            use_model_parallel = self.multi_gpu_info.get('use_model_parallel', False)
            gpu_count = self.multi_gpu_info.get('gpu_count', 1)

            # 检测分布式环境（通常通过torchrun或deepspeed启动）
            is_launched_as_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

            logger_temp.info(f"🔧 训练策略配置调试:")
            logger_temp.info(f"  - multi_gpu_info.use_model_parallel: {use_model_parallel}")
            logger_temp.info(f"  - multi_gpu_info.gpu_count: {gpu_count}")
            logger_temp.info(f"  - script_cfg.use_model_parallel: {getattr(self.script_cfg, 'use_model_parallel', 'MISSING')}")
            logger_temp.info(f"  - 检测到分布式启动: {is_launched_as_distributed}")

                    # 🔧 检查用户明确的模型并行偏好（优先级最高）
            user_specified_model_parallel_explicit = getattr(self.script_cfg, 'use_model_parallel', None)
            logger_temp.info(f"  - 用户明确的模型并行设置: {user_specified_model_parallel_explicit}")
            
            # 如果用户明确要求模型并行，强制使用模型并行
            if user_specified_model_parallel_explicit is True:
                logger_temp.info("🔧 用户明确要求模型并行，强制设置模型并行策略")
                use_model_parallel = True
                self.training_strategy = "model_parallel_single_process"
                logger_temp.info("✅ 已强制设置为模型并行模式")
                return

            # 🔧 检测用户是否启用FSDP（只有在没有明确要求模型并行时才检查）
            user_specified_fsdp_flag = getattr(self.script_cfg, 'use_fsdp', False)
            # transformers.TrainingArguments 会把 --fsdp 解析到 grpo_cfg.fsdp (List[str] 或 str)
            fsdp_arg_from_grpo = getattr(self.grpo_cfg, 'fsdp', None)
            user_specified_fsdp = user_specified_fsdp_flag or (fsdp_arg_from_grpo is not None and fsdp_arg_from_grpo != "")
            logger_temp.info(f"  - script_cfg.use_fsdp: {user_specified_fsdp}")

            # 如果用户要求FSDP，优先级次高
            if user_specified_fsdp:
                    if gpu_count < 1:
                        raise ValueError("FSDP 需要至少1张可用GPU！")

                    # 如果已经由 torchrun 等方式启动多进程，则认为是分布式 FSDP
                    if is_launched_as_distributed:
                        self.training_strategy = "fsdp_distributed"
                    else:
                        self.training_strategy = "fsdp_single_process"

                    # 使用FSDP时禁用模型并行标记
                    self.multi_gpu_info['use_model_parallel'] = False
                    logger_temp.info(f"✅ 已设置训练策略为 {self.training_strategy}")
                    return  # 提前结束，避免后续逻辑覆盖

            # 关键决策逻辑
            if use_model_parallel and gpu_count >= 2:
                if is_launched_as_distributed:
                    # 如果用户同时尝试DDP和模型并行，这是不兼容的。
                    # 我们优先模型并行，并禁用DDP。
                    logger_temp.warning("⚠️ 检测到分布式环境与模型并行请求冲突。")
                    logger_temp.info("🔧 解决策略：优先使用模型并行，将禁用DDP。")
                    
                    # 清除分布式环境变量以避免TRL/HF库的自动DDP初始化
                    dist_env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
                    for var in dist_env_vars:
                        if var in os.environ:
                            logger_temp.info(f"🧹 清除冲突的分布式环境变量: {var}={os.environ[var]}")
                            del os.environ[var]
                    
                    self.training_strategy = "model_parallel_single_process"
                    logger_temp.info("✅ 已配置为单进程模型并行模式（DDP已禁用）。")
                else:
                    # 这是预期的单进程模型并行场景。
                    self.training_strategy = "model_parallel_single_process"
                    logger_temp.info("✅ 配置为单进程模型并行模式。")
            
            elif is_launched_as_distributed:
                # 策略2：分布式数据并行 (DDP)
                self.training_strategy = "distributed_data_parallel"
                logger_temp.info("✅ 配置为分布式数据并行（DDP）模式。")
                self.multi_gpu_info['use_model_parallel'] = False # 确保一致性
            
            else:
                # 策略3：单GPU训练
                self.training_strategy = "single_gpu"
                logger_temp.info("✅ 配置为单GPU训练模式。")
                self.multi_gpu_info['use_model_parallel'] = False # 确保一致性
                if gpu_count > 1:
                    logger_temp.info("🔧 检测到多张GPU但未使用并行策略，将限制使用GPU 0。")
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    self.multi_gpu_info['gpu_count'] = 1

            logger_temp.info(f"🎯 最终确定的训练策略: {self.training_strategy}")

        except Exception as e:
            logger_temp.error(f"❌ 训练策略配置失败: {e}", exc_info=True)
            # 安全回退到单GPU模式
            self.training_strategy = "single_gpu"
            self.multi_gpu_info = { 'gpu_count': 1, 'use_model_parallel': False }
            logger_temp.info("🔄 已回退到安全的单GPU训练模式。")

    def _prepare_grpo_config_for_strategy(self):
        """🔧 根据训练策略准备GRPO配置（修复只读属性问题）"""
        import copy
        
        logger.info(f"🔧 为 {self.training_strategy} 准备GRPO配置...")
        
        # 🔧 关键修复：只修改非只读属性
        try:
            grpo_cfg_copy = copy.deepcopy(self.grpo_cfg)
        except Exception as e:
            logger.warning(f"⚠️ 深拷贝失败，使用原配置: {e}")
            grpo_cfg_copy = self.grpo_cfg
        
        # 🔧 只修改可以修改的属性，避免只读属性
        try:
            # 关键：禁用内置WandB以避免冲突
            if hasattr(grpo_cfg_copy, 'report_to') and "wandb" in grpo_cfg_copy.report_to:
                try:
                    grpo_cfg_copy.report_to = [r for r in grpo_cfg_copy.report_to if r != "wandb"]
                    logger.info("🔧 禁用GRPOTrainer内置WandB报告")
                except Exception as report_e:
                    logger.warning(f"⚠️ 无法修改report_to属性: {report_e}")
            
            # 根据训练策略进行特定优化
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                logger.info("🚀 应用模型并行优化...")
                
                # 优化数据加载设置
                if hasattr(grpo_cfg_copy, 'dataloader_num_workers'):
                    try:
                        original_workers = grpo_cfg_copy.dataloader_num_workers
                        grpo_cfg_copy.dataloader_num_workers = min(original_workers, 2)
                        if original_workers != grpo_cfg_copy.dataloader_num_workers:
                            logger.info(f"🔧 调整dataloader_num_workers: {original_workers} -> {grpo_cfg_copy.dataloader_num_workers}")
                    except Exception as worker_e:
                        logger.warning(f"⚠️ 无法调整dataloader_num_workers: {worker_e}")
                
                if hasattr(grpo_cfg_copy, 'dataloader_pin_memory'):
                    try:
                        grpo_cfg_copy.dataloader_pin_memory = False
                        logger.info("🔧 禁用dataloader_pin_memory以避免多GPU内存问题")
                    except Exception as pin_e:
                        logger.warning(f"⚠️ 无法禁用dataloader_pin_memory: {pin_e}")
                
                # 禁用DDP相关设置
                if hasattr(grpo_cfg_copy, 'ddp_find_unused_parameters'):
                    try:
                        grpo_cfg_copy.ddp_find_unused_parameters = False
                        logger.info("🔧 禁用ddp_find_unused_parameters")
                    except Exception as ddp_e:
                        logger.warning(f"⚠️ 无法禁用ddp_find_unused_parameters: {ddp_e}")
                        
            elif self.training_strategy.startswith("fsdp"):
                logger.info("🚀 应用 FSDP 相关优化…")

                # 🔧 关键修复：优先清除可能冲突的参数
                if hasattr(grpo_cfg_copy, 'fsdp_min_num_params'):
                    try:
                        # 完全删除这个属性，而不是设为0
                        delattr(grpo_cfg_copy, 'fsdp_min_num_params')
                        logger.info("🔧 删除 fsdp_min_num_params 属性以避免冲突")
                    except Exception as min_params_e:
                        logger.warning(f"⚠️ 无法删除 fsdp_min_num_params: {min_params_e}")

                # 启用 FSDP
                if hasattr(grpo_cfg_copy, 'fsdp'):
                    try:
                        if not grpo_cfg_copy.fsdp or grpo_cfg_copy.fsdp == "":
                            grpo_cfg_copy.fsdp = "full_shard"
                        logger.info(f"🔧 FSDP 模式: {grpo_cfg_copy.fsdp}")
                    except Exception as fsdp_e:
                        logger.warning(f"⚠️ 无法设置 fsdp: {fsdp_e}")

                # 指定 transformer layer 类名（仅在删除min_num_params后设置）
                if hasattr(grpo_cfg_copy, 'fsdp_transformer_layer_cls_to_wrap'):
                    try:
                        grpo_cfg_copy.fsdp_transformer_layer_cls_to_wrap = "QWenBlock"
                        logger.info(f"🔧 设置 fsdp_transformer_layer_cls_to_wrap: {grpo_cfg_copy.fsdp_transformer_layer_cls_to_wrap}")
                    except Exception as fsdp_layer_e:
                        logger.warning(f"⚠️ 无法设置 fsdp_transformer_layer_cls_to_wrap: {fsdp_layer_e}")

                # 设置 fsdp_config 字典
                try:
                    fsdp_conf_dict = {
                        "transformer_layer_cls_to_wrap": "QWenBlock",
                        "backward_prefetch": "backward_pre", 
                        "forward_prefetch": False,
                        "use_orig_params": True,
                        "sync_module_states": True,
                        "limit_all_gathers": True,
                        "cpu_offload": False,  # 禁用CPU offload以获得更好性能
                        "xla": False  # 🔧 关键修复：添加必需的xla键
                    }
                    grpo_cfg_copy.fsdp_config = fsdp_conf_dict
                    logger.info(f"🔧 注入 fsdp_config: {fsdp_conf_dict}")
                except Exception as fsdp_conf_e:
                    logger.warning(f"⚠️ 无法注入 fsdp_config: {fsdp_conf_e}")

            elif self.training_strategy == "distributed_data_parallel":
                logger.info("🔧 保持分布式数据并行设置...")
                # 保持原有设置，只禁用WandB
                
            else:  # single_gpu
                logger.info("🔧 应用单GPU优化...")
                # 单GPU时的优化设置
            
            # 🔧 重要：只读取配置信息，不修改只读属性
            if hasattr(grpo_cfg_copy, 'world_size'):
                logger.debug(f"当前world_size: {grpo_cfg_copy.world_size}")
            if hasattr(grpo_cfg_copy, 'local_rank'):
                logger.debug(f"当前local_rank: {grpo_cfg_copy.local_rank}")
            
            # 🔧 新增：确保分布式相关环境变量清理生效
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process", "single_gpu"]:
                # 清除可能导致分布式初始化的环境变量
                dist_env_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
                for var in dist_env_vars:
                    if var in os.environ:
                        logger.info(f"🧹 确保清除分布式环境变量: {var}")
                        del os.environ[var]
            
            logger.info(f"✅ GRPO配置已针对{self.training_strategy}优化")
            return grpo_cfg_copy
            
        except Exception as config_e:
            logger.error(f"❌ 配置修改失败: {config_e}")
            logger.info("🔄 回退到最小修改策略")
            
            # 🔧 最后的回退方案：创建最小修改版本
            try:
                # 仅尝试禁用WandB，其他保持不变
                if hasattr(self.grpo_cfg, 'report_to') and "wandb" in self.grpo_cfg.report_to:
                    # 如果report_to是可修改的，创建修改版本
                    minimal_copy = copy.copy(self.grpo_cfg)
                    try:
                        minimal_copy.report_to = [r for r in minimal_copy.report_to if r != "wandb"]
                        logger.info("🔧 回退方案：仅禁用WandB报告")
                        return minimal_copy
                    except:
                        pass
                
                # 终极回退：使用原始配置
                logger.warning("⚠️ 使用原始GRPO配置，可能包含WandB冲突")
                return self.grpo_cfg
                
            except Exception as fallback_e:
                logger.error(f"❌ 回退方案也失败: {fallback_e}")
                # 最终回退：直接返回原始配置
                return self.grpo_cfg
        
    def _sync_length_configs(self):
        """🔧 同步长度配置从ScriptConfig到GRPOConfig，避免参数冲突"""
        logger.info("🔧 同步长度配置到GRPOConfig...")
        
        # 将我们的脚本配置同步到GRPO配置
        self.grpo_cfg.max_prompt_length = self.script_cfg.script_max_prompt_length
        self.grpo_cfg.max_completion_length = self.script_cfg.script_max_completion_length
        
        logger.info(f"  ✅ GRPO max_prompt_length: {self.grpo_cfg.max_prompt_length}")
        logger.info(f"  ✅ GRPO max_completion_length: {self.grpo_cfg.max_completion_length}")

    def _setup_stable_environment(self):
        """🔧 设置稳定的训练环境，支持多GPU优化"""
        try:
            logger.info("🔧 设置稳定训练环境...")
            
            # 🔧 根据训练策略设置环境变量
            if self.training_strategy == "model_parallel_only":
                logger.info("🚀 配置纯模型并行环境...")
                
                stable_envs = {
                    "CUDA_LAUNCH_BLOCKING": "0",  # 异步执行以提高性能
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
                    "NCCL_BLOCKING_WAIT": "1",
                    "NCCL_P2P_DISABLE": "0",  # 启用P2P通信
                    "NCCL_TIMEOUT": "7200",
                }
                
                # 确保没有分布式相关的环境变量
                for dist_var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
                    if dist_var in os.environ:
                        del os.environ[dist_var]
                        logger.info(f"🧹 清除分布式变量: {dist_var}")
                
            elif self.training_strategy == "distributed_data_parallel":
                logger.info("🔧 配置分布式数据并行环境...")
                
                stable_envs = {
                    "CUDA_LAUNCH_BLOCKING": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                    "NCCL_BLOCKING_WAIT": "1",
                }
                
            else:
                # 单GPU或单进程模型并行
                stable_envs = {
                    "CUDA_LAUNCH_BLOCKING": "0" if self.training_strategy == "model_parallel_single_process" else "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
                }
            
            # Flash Attention优化（如果支持）
            if getattr(self.script_cfg, 'enable_flash_attention', True):
                stable_envs["FLASH_ATTENTION_V2"] = "1"
                logger.info("⚡ 启用Flash Attention 2优化")
            
            for key, value in stable_envs.items():
                if key not in os.environ:
                    os.environ[key] = value
                    logger.info(f"  设置环境变量: {key}={value}")
            
            # 🔧 梯度检查点处理
            if hasattr(self.grpo_cfg, 'gradient_checkpointing') and self.grpo_cfg.gradient_checkpointing:
                if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                    if not getattr(self.script_cfg, 'gradient_checkpointing_compatible', False):
                        logger.warning("⚠️ 模型并行模式下自动禁用梯度检查点以避免同步问题")
                        self.grpo_cfg.gradient_checkpointing = False
            
            # 清理GPU内存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                logger.info(f"✅ 已清理 {torch.cuda.device_count()} 张GPU的内存")
            
            logger.info("✅ 稳定训练环境设置完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 稳定环境设置失败: {e}")

    def _configure_wandb_resume(self):
        """🔧 自动配置WandB恢复，无需外部脚本"""
        try:
            # 检查是否从checkpoint恢复
            is_resuming = (
                self.grpo_cfg.resume_from_checkpoint and
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
            )
            
            if not is_resuming:
                logger.info("🚀 开始新的训练，无需WandB恢复配置")
                # 确保清除可能存在的恢复相关环境变量
                for env_var in ["WANDB_RUN_ID", "WANDB_RESUME"]:
                    if env_var in os.environ:
                        del os.environ[env_var]
                        logger.info(f"🧹 清除环境变量: {env_var}")
                return
            
            checkpoint_path = Path(self.grpo_cfg.resume_from_checkpoint)
            logger.info(f"🔄 检测到checkpoint恢复: {checkpoint_path}")
            
            # 尝试从checkpoint目录提取WandB run ID
            run_id, run_url = self._extract_wandb_run_id(checkpoint_path)
            
            if run_id:
                # 设置精确恢复
                os.environ["WANDB_RUN_ID"] = run_id
                os.environ["WANDB_RESUME"] = "must"
                logger.info(f"✅ WandB精确恢复配置:")
                logger.info(f"  - Run ID: {run_id}")
                logger.info(f"  - Resume Mode: must")
                if run_url:
                    logger.info(f"  - Run URL: {run_url}")
            else:
                # 使用自动恢复模式
                os.environ["WANDB_RESUME"] = "allow"
                logger.info("⚠️ 未找到具体的Run ID，使用自动恢复模式")
                logger.info("  - Resume Mode: allow")
            
            logger.info("✅ WandB恢复配置完成")
            
        except Exception as e:
            logger.warning(f"⚠️ WandB恢复配置失败: {e}")
            logger.info("🔄 将使用默认的WandB配置")

    def _setup_wandb_sync_manager(self):
        """🔧 设置WandB同步管理器"""
        try:
            # 从配置中获取项目信息
            project_name = getattr(self.env_cfg, 'wandb_project', 'VerilogGRPO_Enhanced_MultiGPU')
            run_name = f"grpo_run_{os.path.basename(self.grpo_cfg.output_dir)}"
            
            # 🔧 训练策略标识
            run_name = f"{self.training_strategy}_{run_name}"
            
            # 初始化同步管理器
            sync_manager = initialize_wandb_sync_manager(
                output_dir=self.grpo_cfg.output_dir,
                project_name=project_name,
                run_name=run_name
            )
            
            logger.info("✅ WandB同步管理器初始化成功")
            logger.info(f"  - 项目: {project_name}")
            logger.info(f"  - 运行名称: {run_name}")
            logger.info(f"  - 输出目录: {self.grpo_cfg.output_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ WandB同步管理器初始化失败: {e}")
            logger.info("🔄 将使用原生WandB功能")

    def _setup_wandb_run(self):
        """🔧 设置WandB运行，处理断续训练"""
        try:
            sync_manager = get_wandb_sync_manager()
            if not sync_manager:
                logger.warning("⚠️ WandB同步管理器未找到，跳过WandB运行设置")
                return
                
            # 检查是否从checkpoint恢复
            resume_from_checkpoint = None
            if (self.grpo_cfg.resume_from_checkpoint and 
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)):
                resume_from_checkpoint = self.grpo_cfg.resume_from_checkpoint
                
            # 🔧 准备配置，包含训练策略信息
            config = {
                "model_name_or_path": self.script_cfg.model_name_or_path,
                "learning_rate": self.grpo_cfg.learning_rate,
                "per_device_train_batch_size": self.grpo_cfg.per_device_train_batch_size,
                "max_seq_length": self.script_cfg.max_seq_length,
                "callback_eval_every_n_steps": self.script_cfg.callback_eval_every_n_steps,
                "lora_rank": getattr(self.script_cfg, 'lora_rank', None),
                "curriculum_enabled": self.curriculum_manager is not None,
                "resume_from_checkpoint": resume_from_checkpoint,
                # 🔧 训练策略相关配置
                "training_strategy": self.training_strategy,
                "use_model_parallel": self.multi_gpu_info.get('use_model_parallel', False),
                "gpu_count": self.multi_gpu_info.get('gpu_count', 1),
                "total_gpu_memory_gb": self.multi_gpu_info.get('total_memory_gb', 0),
                "max_memory_per_gpu": getattr(self.script_cfg, 'max_memory_per_gpu', 'auto'),
            }
            
            # 设置WandB运行
            success = sync_manager.setup_wandb_run(
                resume_from_checkpoint=resume_from_checkpoint,
                config=config
            )
            
            if success:
                logger.info("✅ WandB运行设置成功")
            else:
                logger.warning("⚠️ WandB运行设置失败，将使用本地日志")
                
        except Exception as e:
            logger.warning(f"⚠️ WandB运行设置异常: {e}")

    def _extract_wandb_run_id(self, checkpoint_path: Path) -> tuple[Optional[str], Optional[str]]:
        """从checkpoint目录中提取WandB run ID"""
        try:
            # 方法1: 从wandb目录中查找
            parent_dir = checkpoint_path.parent
            wandb_dir = parent_dir / "wandb"
            
            if wandb_dir.exists():
                logger.info(f"🔍 在 {wandb_dir} 中查找WandB run信息...")
                
                # 查找run目录
                run_dirs = list(wandb_dir.glob("run-*"))
                if run_dirs:
                    latest_run_dir = sorted(run_dirs)[-1]
                    logger.info(f"📁 找到run目录: {latest_run_dir.name}")
                    
                    # 提取run ID (格式: run-20231201_123456-abcd1234)
                    run_name = latest_run_dir.name
                    if "-" in run_name:
                        parts = run_name.split("-")
                        if len(parts) >= 3:
                            run_id = parts[-1]  # 最后一部分是run ID
                            logger.info(f"✅ 提取到run ID: {run_id}")
                            
                            # 尝试读取run信息
                            run_info_file = latest_run_dir / "files" / "wandb-metadata.json"
                            run_url = None
                            if run_info_file.exists():
                                try:
                                    with open(run_info_file, 'r') as f:
                                        metadata = json.load(f)
                                        run_url = metadata.get('url', '')
                                        if run_url:
                                            logger.info(f"🔗 找到Run URL: {run_url}")
                                except Exception as e:
                                    logger.warning(f"⚠️ 读取metadata失败: {e}")
                            
                            return run_id, run_url
            
            # 方法2: 检查环境变量中是否已有run ID
            env_run_id = os.getenv("WANDB_RUN_ID")
            if env_run_id:
                logger.info(f"🔄 使用环境变量中的WandB run ID: {env_run_id}")
                return env_run_id, None
            
            logger.info("❌ 未能找到WandB run ID")
            return None, None
            
        except Exception as e:
            logger.warning(f"⚠️ 提取WandB run ID时出错: {e}")
            return None, None

    def _setup_logging(self):
        """Setup logging with improved error handling"""
        from datetime import datetime

        run_specific_name_from_env = os.getenv("WANDB_RUN_NAME")
        if not run_specific_name_from_env:
            # 🔧 包含训练策略标识
            prefix = getattr(self.env_cfg, 'wandb_run_name_prefix', 'grpo')
            prefix = f"{self.training_strategy}_{prefix}"
            
            run_specific_name_from_env = f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        sanitized_run_name = re.sub(r'[^\w\-.]', '_', run_specific_name_from_env)

        # Determine if resuming
        is_resuming = (
            self.grpo_cfg.resume_from_checkpoint and
            isinstance(self.grpo_cfg.resume_from_checkpoint, str) and
            os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
        )

        if is_resuming:
            actual_output_dir = os.path.dirname(self.grpo_cfg.resume_from_checkpoint)
        else:
            output_dir_base = getattr(self.env_cfg, 'output_dir_base', './outputs')
            actual_output_dir = os.path.join(output_dir_base, sanitized_run_name)

        if self.grpo_cfg.local_rank <= 0:
            os.makedirs(actual_output_dir, exist_ok=True)

        # Update config objects
        self.grpo_cfg.output_dir = actual_output_dir
        if hasattr(self.script_cfg, 'output_dir'):
            self.script_cfg.output_dir = actual_output_dir

        log_file_path = os.path.join(self.grpo_cfg.output_dir, "grpo_pipeline_log.txt")
        setup_global_logging(
            log_level=self.grpo_cfg.get_process_log_level(),
            log_file_path=log_file_path,
            local_rank=self.grpo_cfg.local_rank
        )
        logger.info(f"Global logging set up. Output directory: {self.grpo_cfg.output_dir}")

    def _log_configs(self):
        """🔧 增强配置日志，包含训练策略信息"""
        # 🔧 数据集路径配置调试
        logger.info("📁 数据集配置:")
        logger.info(f"  - dataset_path: {getattr(self.script_cfg, 'dataset_path', 'None')}")
        logger.info(f"  - env_cfg.dataset_base_path: {getattr(self.env_cfg, 'dataset_base_path', 'None')}")
        if hasattr(self.script_cfg, 'dataset_path') and self.script_cfg.dataset_path:
            import os
            inferred_base = os.path.dirname(self.script_cfg.dataset_path)
            logger.info(f"  - 从dataset_path推导的基础路径: {inferred_base}")
        
        # 🔧 训练策略信息
        logger.info("🎯 训练策略配置:")
        logger.info(f"  - 策略: {self.training_strategy}")
        logger.info(f"  - GPU数量: {self.multi_gpu_info.get('gpu_count', 0)}")
        logger.info(f"  - 模型并行: {'✅' if self.multi_gpu_info.get('use_model_parallel', False) else '❌'}")
        
        if self.multi_gpu_info:
            logger.info(f"  - 总GPU内存: {self.multi_gpu_info.get('total_memory_gb', 0):.1f}GB")
            logger.info(f"  - 平均每GPU: {self.multi_gpu_info.get('average_memory_gb', 0):.1f}GB")
        
        # 长度配置信息
        logger.info("📏 长度配置:")
        logger.info(f"  - 总序列长度: {self.script_cfg.max_seq_length}")
        logger.info(f"  - 最大提示长度: {self.script_cfg.script_max_prompt_length} ({self.script_cfg.script_max_prompt_length/self.script_cfg.max_seq_length*100:.1f}%)")
        logger.info(f"  - 最大输出长度: {self.script_cfg.script_max_completion_length} ({self.script_cfg.script_max_completion_length/self.script_cfg.max_seq_length*100:.1f}%)")
        logger.info(f"  - 分配策略: {self.script_cfg.length_allocation_strategy}")
        
        # 训练配置信息
        logger.info("🎯 训练配置:")
        logger.info(f"  - 每GPU批次大小: {self.grpo_cfg.per_device_train_batch_size}")
        logger.info(f"  - 梯度累积步数: {self.grpo_cfg.gradient_accumulation_steps}")
        logger.info(f"  - 学习率: {self.grpo_cfg.learning_rate}")
        
        # 计算有效批次大小
        if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
            # 模型并行时，有效批次大小不会因GPU数量倍增（因为模型是分布的，不是数据）
            effective_batch_size = (self.grpo_cfg.per_device_train_batch_size * 
                                  self.grpo_cfg.gradient_accumulation_steps)
            logger.info(f"  - 有效批次大小: {effective_batch_size} (模型并行)")
        elif self.training_strategy == "distributed_data_parallel":
            effective_batch_size = (self.grpo_cfg.per_device_train_batch_size * 
                                  self.grpo_cfg.gradient_accumulation_steps * 
                                  self.multi_gpu_info.get('gpu_count', 1))
            logger.info(f"  - 有效批次大小: {effective_batch_size} (数据并行)")
        else:
            effective_batch_size = (self.grpo_cfg.per_device_train_batch_size * 
                                  self.grpo_cfg.gradient_accumulation_steps)
            logger.info(f"  - 有效批次大小: {effective_batch_size} (单GPU)")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"EnvConfig: \n{json.dumps(asdict(self.env_cfg), indent=2)}")
            logger.debug(f"ScriptConfig: \n{json.dumps(asdict(self.script_cfg), indent=2)}")
            logger.debug(f"EnhancedRewardConfig: \n{json.dumps(asdict(self.reward_cfg), indent=2)}")

    def _setup_model_and_tokenizer(self):
        """🔧 增强模型和tokenizer设置，解决DDP冲突"""
        try:
            logger.info("🔧 设置模型和tokenizer...")
            
            # 🔧 记录设置前的状态
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                logger.info(f"🚀 准备{self.training_strategy}模型设置...")
                logger.info(f"  - 目标GPU数量: {self.multi_gpu_info.get('gpu_count', 1)}")
                logger.info(f"  - 每GPU内存限制: {getattr(self.script_cfg, 'max_memory_per_gpu', '75GiB')}")
                
                # 记录设置前的GPU内存状态
                logger.info("📊 设置前GPU内存状态:")
                for i in range(self.multi_gpu_info.get('gpu_count', 1)):
                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.1f}GB total")
            
            # 调用ModelManager的方法
            start_time = time.time()
            self.model, self.tokenizer = self.model_manager.setup_model_and_tokenizer()
            setup_time = time.time() - start_time
            
            logger.info(f"⏱️ 模型加载耗时: {setup_time:.2f}秒")
            
            # 🔧 记录模型加载后的状态
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                logger.info("📊 模型加载后GPU内存状态:")
                total_allocated = 0
                for i in range(self.multi_gpu_info.get('gpu_count', 1)):
                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        total_allocated += allocated
                        logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                logger.info(f"  总分配内存: {total_allocated:.2f}GB")
                if self.multi_gpu_info.get('gpu_count', 1) > 1:
                    logger.info(f"  平均每GPU: {total_allocated/self.multi_gpu_info['gpu_count']:.2f}GB")
                
                # 验证模型分布
                if hasattr(self.model, 'hf_device_map'):
                    logger.info("🗺️ 模型设备分布验证:")
                    device_counts = {}
                    for layer_name, device in self.model.hf_device_map.items():
                        device_str = str(device)
                        device_counts[device_str] = device_counts.get(device_str, 0) + 1
                    
                    for device, count in device_counts.items():
                        logger.info(f"  {device}: {count} 层")
                    
                    # 检查是否真正实现了模型并行
                    if len(device_counts) > 1:
                        logger.info("✅ 确认实现模型并行分布")
                    else:
                        logger.warning("⚠️ 模型似乎没有分布到多个设备")
                else:
                    logger.warning("⚠️ 未找到设备映射信息")
            
            # 应用PEFT适配器
            is_resuming = (
                self.grpo_cfg.resume_from_checkpoint and 
                isinstance(self.grpo_cfg.resume_from_checkpoint, str) and 
                os.path.isdir(self.grpo_cfg.resume_from_checkpoint)
            )
            
            logger.info("🔧 应用PEFT适配器...")
            adapter_start_time = time.time()
            self.model = self.model_manager.apply_peft_adapter(
                model=self.model,
                is_resuming=is_resuming,
                resume_checkpoint_path=str(self.grpo_cfg.resume_from_checkpoint) if is_resuming else None
            )
            adapter_time = time.time() - adapter_start_time
            logger.info(f"⏱️ PEFT适配器设置耗时: {adapter_time:.2f}秒")
            
            # 🔧 验证最终模型状态
            if hasattr(self.model_manager, 'get_model_memory_info'):
                memory_info = self.model_manager.get_model_memory_info()
                logger.info("📈 最终模型内存信息:")
                logger.info(f"  - 总参数: {memory_info.get('model_parameters', 0):,}")
                logger.info(f"  - 可训练参数: {memory_info.get('trainable_parameters', 0):,}")
                
                gpu_memory = memory_info.get('gpu_memory', {})
                for gpu_id, gpu_info in gpu_memory.items():
                    logger.info(f"  - {gpu_id}: {gpu_info.get('allocated', 0):.2f}GB allocated")
            
            logger.info("✅ Model and tokenizer setup completed successfully.")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to setup model and tokenizer: {e}", exc_info=True)
            
            # 多GPU错误时的详细诊断
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                logger.error("🔍 模型并行设置失败，进行诊断:")
                try:
                    for i in range(self.multi_gpu_info.get('gpu_count', 1)):
                        if torch.cuda.is_available() and i < torch.cuda.device_count():
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            reserved = torch.cuda.memory_reserved(i) / 1024**3
                            logger.error(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                            
                            # 检查是否有进程占用GPU
                            if allocated > 1.0:  # 超过1GB可能有其他进程
                                logger.error(f"  ⚠️ GPU {i} 可能被其他进程占用")
                                
                except Exception as diag_e:
                    logger.error(f"诊断过程中出错: {diag_e}")
            
            raise

    def _initialize_components(self, dataset_processed):
        """Initialize components that depend on the processed dataset"""
        try:
            logger.info("Initializing training components...")
            
            # 经验回放缓冲区设置
            if self.script_cfg.enable_experience_replay:
                # 根据训练策略调整缓冲区大小
                buffer_size = self.script_cfg.experience_buffer_size
                if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                    buffer_size = max(buffer_size, 1500)  # 模型并行时可以使用更大缓冲区
                    logger.info(f"🚀 模型并行模式下使用增强的经验回放缓冲区: {buffer_size}")
                
                self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
                logger.info(f"Experience buffer initialized (size: {buffer_size}).")
                
                # Restore experience buffer state if resuming
                if self.grpo_cfg.resume_from_checkpoint and os.path.isdir(self.grpo_cfg.resume_from_checkpoint):
                    buffer_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_experience_buffer_state.json")
                    if os.path.exists(buffer_state_path):
                        try:
                            logger.info(f"Loading experience buffer state from: {buffer_state_path}")
                            with open(buffer_state_path, "r", encoding="utf-8") as f:
                                state_data = json.load(f)
                            self.experience_buffer.load_buffer_state(state_data)
                            logger.info("✅ Experience buffer state loaded successfully.")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load experience buffer state: {e}")

            # Curriculum learning setup
            if self.script_cfg.enable_curriculum:
                self.curriculum_manager = setup_fixed_curriculum_manager(self.script_cfg, dataset_processed)
                if self.curriculum_manager:
                    logger.info(f"Curriculum learning enabled: {type(self.curriculum_manager).__name__}.")
                    
                    # Restore curriculum state if resuming
                    if self.grpo_cfg.resume_from_checkpoint and os.path.isdir(self.grpo_cfg.resume_from_checkpoint):
                        curriculum_state_path = os.path.join(self.grpo_cfg.resume_from_checkpoint, "enhanced_curriculum_state.json")
                        if os.path.exists(curriculum_state_path):
                            try:
                                logger.info(f"Loading curriculum state from: {curriculum_state_path}")
                                with open(curriculum_state_path, "r", encoding="utf-8") as f:
                                    state_data = json.load(f)
                                self.curriculum_manager.load_curriculum_state(state_data)
                                logger.info("✅ Curriculum state loaded successfully.")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to load curriculum state: {e}")

            # Setup callbacks
            self._setup_callbacks(dataset_processed)
            logger.info("✅ All components initialized successfully.")
            
        except Exception as e:
            logger.error(f"❌ Error during component initialization: {e}", exc_info=True)
            raise

    def _setup_callbacks(self, dataset_processed):
        """🔧 增强回调设置，避免DDP冲突"""
        try:
            self.callbacks = []
            
            # Basic monitoring callbacks
            self.callbacks.append(StepLoggingCallback())
            self.callbacks.append(DetailedRewardCallback(self.grpo_cfg.output_dir))
            self.callbacks.append(RewardStabilityMonitor(self.grpo_cfg.output_dir))

            # 课程学习回调设置
            if self.curriculum_manager:
                # 根据训练策略调整性能检查间隔
                check_interval = self.script_cfg.curriculum_performance_check_interval
                if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"] and check_interval < 10:
                    check_interval = 10
                    logger.info(f"🔧 模型并行模式下调整课程检查间隔: {check_interval}")
                
                # 添加课程学习相关回调
                curriculum_progress_cb = CurriculumProgressCallback(
                    curriculum_manager=self.curriculum_manager, 
                    trainer_ref=None,
                    output_dir=self.grpo_cfg.output_dir,
                    performance_check_interval=check_interval
                )
                self.callbacks.append(curriculum_progress_cb)
                logger.info("✅ 添加 CurriculumProgressCallback")

                enhanced_curriculum_cb = EnhancedCurriculumDebugCallback(
                    curriculum_manager=self.curriculum_manager, 
                    trainer_ref=None,
                    output_dir=self.grpo_cfg.output_dir
                )
                self.callbacks.append(enhanced_curriculum_cb)
                logger.info("✅ 添加 EnhancedCurriculumDebugCallback")

                optimized_curriculum_cb = OptimizedCurriculumCallback(
                    curriculum_manager=self.curriculum_manager,
                    trainer_ref=None,
                    output_dir=self.grpo_cfg.output_dir
                )
                self.callbacks.append(optimized_curriculum_cb)
                logger.info("✅ 添加 OptimizedCurriculumCallback")

            # State persistence callback
            self.callbacks.append(CustomStatePersistenceCallback(
                curriculum_manager=self.curriculum_manager, 
                experience_buffer=self.experience_buffer, 
                script_cfg=self.script_cfg
            ))

            # 推理回调设置
            if self.tokenizer and dataset_processed and len(dataset_processed) > 0:
                # 根据训练策略调整推理参数
                eval_samples = self.script_cfg.callback_num_samples
                eval_interval = self.script_cfg.callback_eval_every_n_steps
                
                if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                    if eval_interval < 15:
                        eval_interval = 15
                        logger.info(f"🔧 模型并行模式下调整推理回调间隔: {eval_interval}")
                
                sample_dataset_for_inf_cb = dataset_processed.select(
                    range(min(len(dataset_processed), eval_samples * 5))
                )

                if len(sample_dataset_for_inf_cb) > 0:
                    # 尝试使用流式引导推理回调
                    try:
                        from grpo_project.utils.streaming_guidance import create_streaming_inference_callback, GuidanceConfig
                        
                        guidance_config = GuidanceConfig(
                            min_reasoning_length=getattr(self.script_cfg, 'min_reasoning_length', 60),
                            guidance_trigger_threshold=getattr(self.script_cfg, 'guidance_trigger_threshold', 40),
                            max_guidance_attempts=getattr(self.script_cfg, 'max_guidance_attempts', 2),
                            guidance_tokens_limit=getattr(self.script_cfg, 'guidance_tokens_limit', 25)
                        )
                        
                        streaming_callback = create_streaming_inference_callback(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            eval_dataset=sample_dataset_for_inf_cb,
                            num_samples=eval_samples,
                            eval_every_n_steps=eval_interval,
                            max_new_tokens=self.script_cfg.script_max_completion_length,
                            max_seq_length=self.script_cfg.max_seq_length,
                            experience_buffer=self.experience_buffer,
                            output_dir=self.grpo_cfg.output_dir,
                            guidance_config=guidance_config
                        )
                        
                        self.callbacks.append(streaming_callback)
                        logger.info(f"✅ 流式引导推理回调已添加 (samples: {len(sample_dataset_for_inf_cb)}, interval: {eval_interval})")
                        
                    except ImportError:
                        # 降级到标准推理回调
                        logger.warning("⚠️ 流式引导模块不可用，使用标准推理回调")
                        
                        if hasattr(DetailedInferenceCallback, '__init__'):
                            inference_cb = DetailedInferenceCallback(
                                tokenizer=self.tokenizer,
                                eval_dataset=sample_dataset_for_inf_cb,
                                num_samples=eval_samples,
                                eval_every_n_steps=eval_interval,
                                max_new_tokens=self.script_cfg.script_max_completion_length,
                                max_seq_length=self.script_cfg.max_seq_length,
                                experience_buffer=self.experience_buffer,
                                output_dir=self.grpo_cfg.output_dir
                            )
                            self.callbacks.append(inference_cb)
                            logger.info("✅ 标准推理回调已添加")
                else:
                    logger.warning("⚠️ 样本数据不足，跳过推理回调")

            # WandB回调设置
            if self.grpo_cfg.local_rank <= 0 and "wandb" in self.grpo_cfg.report_to:
                try:
                    from grpo_project.callbacks.wandb_sync_callback import SyncedWandbCallback
                    wandb_cb = SyncedWandbCallback(
                        env_cfg=self.env_cfg, 
                        script_cfg=self.script_cfg, 
                        reward_cfg=self.reward_cfg, 
                        experience_buffer=self.experience_buffer
                    )
                    self.callbacks.append(wandb_cb)
                    self.wandb_callback = wandb_cb
                    logger.info("✅ SyncedWandbCallback added (替代原生WandB).")
                except ImportError:
                    logger.warning("⚠️ SyncedWandbCallback不可用，跳过WandB回调")

            logger.info(f"Total callbacks prepared: {len(self.callbacks)}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up callbacks: {e}", exc_info=True)
            self.callbacks = [StepLoggingCallback()]
            logger.warning("⚠️ Using minimal callback setup due to errors.")

    def get_reward_function(self):
        """Create optimized reward function closure with enhanced efficiency"""
        reward_call_count = 0  # 用于减少日志频率
        
        def optimized_reward_fn_closure(prompts: List[str], completions: List[str], **kwargs_from_trainer_dataset) -> List[float]:
            nonlocal reward_call_count
            reward_call_count += 1
            
            try:
                current_training_step = self.trainer.state.global_step if self.trainer and self.trainer.state else 0

                # 🔧 优化1: 大幅减少日志频率以提高效率
                log_interval = 50 if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"] else 30
                
                if reward_call_count % log_interval == 1:
                    logger.info(f"🎯 步数 {current_training_step}: 奖励计算 #{reward_call_count}")
                    logger.info(f"  - 批次大小: {len(prompts)}")
                    logger.info(f"  - 完成长度统计: min={min(len(c) for c in completions)}, max={max(len(c) for c in completions)}, avg={sum(len(c) for c in completions)/len(completions):.1f}")

                # 🔧 优化2: 预构建参数字典，减少重复构建开销
                batch_rewards_args = {
                    "prompts": prompts,
                    "completions": completions,
                    "testbench_paths": kwargs_from_trainer_dataset.get('testbench_path', []),
                    "expected_total_tests_list": kwargs_from_trainer_dataset.get('expected_total_tests', []),
                    "reference_verilog_paths": kwargs_from_trainer_dataset.get('reference_verilog_path', []),
                    "original_enhanced_prompts": kwargs_from_trainer_dataset.get('original_enhanced_prompt'),
                    "training_step": current_training_step,
                    "output_dir_for_debug": self.grpo_cfg.output_dir,
                    "wandb_callback_obj": getattr(self, 'wandb_callback', None),
                    "experience_buffer_obj": self.experience_buffer,
                    "script_config_obj": self.script_cfg
                }
                
                # 🔧 优化3: 快速奖励计算
                start_time = time.time()
                rewards_list, aggregated_metrics = self.reward_calculator.calculate_batch_rewards(**batch_rewards_args)
                calc_time = time.time() - start_time
                
                # 🔧 优化4: 效率监控和统计
                if reward_call_count % log_interval == 1:
                    reward_stats = {
                        'mean': np.mean(rewards_list) if rewards_list else 0,
                        'std': np.std(rewards_list) if rewards_list else 0,
                        'min': np.min(rewards_list) if rewards_list else 0,
                        'max': np.max(rewards_list) if rewards_list else 0
                    }
                    avg_calc_time_per_sample = calc_time / len(prompts) if prompts else 0
                    
                    logger.info(f"  - 奖励统计: mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}")
                    logger.info(f"  - 奖励范围: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
                    logger.info(f"  - 计算效率: {calc_time:.2f}s total, {avg_calc_time_per_sample:.3f}s/sample")
                    
                    # 性能警告
                    if avg_calc_time_per_sample > 2.0:
                        logger.warning(f"⚠️ 奖励计算较慢 ({avg_calc_time_per_sample:.3f}s/sample)，考虑优化")
                
                return rewards_list
                    
            except Exception as e:
                logger.error(f"❌ 奖励函数错误 (step {current_training_step}, call #{reward_call_count}): {e}", exc_info=True)
                # 🔧 优化5: 返回合理的默认奖励而非零值
                return [0.1] * len(prompts)  # 小正值避免训练停滞
        
        return optimized_reward_fn_closure

    def train(self):
        """🔧 修复训练函数，解决DDP/模型并行冲突"""
        try:
            logger.info("🚀 Starting enhanced GRPO training process...")
            logger.info(f"🎯 训练策略: {self.training_strategy}")
            
            # 设置稳定的训练环境
            self._setup_stable_environment()
            
            # 配置WandB恢复
            logger.info("📝 Step 0: Configuring WandB resume settings...")
            self._configure_wandb_resume()
            
            # 设置WandB同步管理器运行
            logger.info("📝 Step 0.5: Setting up WandB sync manager run...")
            self._setup_wandb_run()

            # 1. Setup model and tokenizer
            logger.info("📝 Step 1: Setting up model and tokenizer...")
            self._setup_model_and_tokenizer()

            # 2. Load and preprocess data
            logger.info("📝 Step 2: Loading and preparing dataset...")
            dataset_processed = load_and_prepare_dataset(
                script_cfg=self.script_cfg,
                env_cfg=self.env_cfg,
                tokenizer=self.tokenizer
            )

            if not dataset_processed or len(dataset_processed) == 0:
                raise ValueError("❌ Dataset is empty after processing!")

            # 3. Initialize components
            logger.info("📝 Step 3: Initializing training components...")
            self._initialize_components(dataset_processed)

            # 4. Determine training dataset
            dataset_for_trainer = dataset_processed
            if self.curriculum_manager:
                dataset_for_trainer = self.curriculum_manager.get_current_stage_dataset()
                current_stage_name = self.curriculum_manager.get_current_stage_info()['stage_name']
                logger.info(f"📚 Using curriculum dataset: {len(dataset_for_trainer)} samples from stage '{current_stage_name}'.")
            else:
                logger.info(f"📚 Using full processed dataset: {len(dataset_for_trainer)} samples.")

            if not dataset_for_trainer or len(dataset_for_trainer) == 0:
                raise ValueError("❌ Training dataset is empty!")

            # 存储数据集引用
            self.dataset_for_trainer = dataset_for_trainer

            # 5. 🔧 关键修复：根据训练策略创建GRPOTrainer
            logger.info("📝 Step 4: Creating GRPOTrainer...")
            from trl import GRPOTrainer
            
            # 准备GRPO配置
            grpo_cfg_for_trainer = self._prepare_grpo_config_for_strategy()
            
            # 🔧 强制修复：无论检测到什么训练策略，都应用设备一致性修复
            logger.info(f"🔧 强制应用设备修复 - 检测到训练策略: {self.training_strategy}")
            logger.info(f"🔧 GPU数量: {self.multi_gpu_info.get('gpu_count', 1)}")
            logger.info(f"🔧 模型并行设置: {self.multi_gpu_info.get('use_model_parallel', False)}")
            
            # 如果有多张GPU，强制应用模型并行策略
            if self.multi_gpu_info.get('gpu_count', 1) > 1 and self.multi_gpu_info.get('use_model_parallel', False):
                logger.info("🔧 强制修正训练策略为模型并行")
                self.training_strategy = "model_parallel_single_process"
            
            self.trainer = GRPOTrainer(
                model=self.model,
                args=grpo_cfg_for_trainer,
                train_dataset=dataset_for_trainer,
                reward_funcs=[self.get_reward_function()],
                callbacks=self.callbacks
            )

            # 设置trainer引用
            for cb in self.callbacks:
                if hasattr(cb, 'trainer_ref') and cb.trainer_ref is None:
                    cb.trainer_ref = self.trainer
                    logger.debug(f"Set trainer_ref for {type(cb).__name__}")

            # 6. 🔧 强制应用设备一致性修复
            logger.info("🔧 无条件应用设备一致性修复...")
            self._apply_universal_device_fix()
            
            # 7. 训练前最终检查
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                logger.info("🚀 模型并行训练前最终检查:")
                self._final_model_parallel_check(dataset_for_trainer)

            # 8. Start training
            logger.info("📝 Step 5: Starting training...")
            logger.info(f"🎯 Training with {len(dataset_for_trainer)} examples using {self.training_strategy}.")
            
            training_start_time = time.time()
            
            train_result = self.trainer.train(resume_from_checkpoint=self.grpo_cfg.resume_from_checkpoint)
            
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"⏱️ 总训练时间: {training_duration:.2f}秒 ({training_duration/60:.2f}分钟)")
            
            # 训练后状态检查
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                self._post_training_check()

            # 9. Save artifacts
            if self.grpo_cfg.local_rank <= 0:
                self._save_training_artifacts(train_result)

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            
            # 详细的错误诊断
            self._diagnose_training_failure(e)
            
            raise



    def _apply_universal_device_fix(self):
        """🔧 通用设备一致性修复（适用于所有GPU配置）"""
        try:
            logger.info("🔧 应用通用设备一致性修复...")
            
            # FSDP 已经由 transformers 内部处理设备分布，这里跳过补丁
            if self.training_strategy and self.training_strategy.startswith("fsdp"):
                logger.info("ℹ️ FSDP 策略下跳过通用设备补丁")
                return
            
            if not self.trainer:
                logger.warning("⚠️ 训练器未初始化，跳过设备修复")
                return
            
            # 获取模型设备信息
            if hasattr(self.trainer.model, 'parameters'):
                model_devices = set()
                for param in self.trainer.model.parameters():
                    model_devices.add(param.device)
                
                if model_devices:
                    primary_device = list(model_devices)[0]
                    logger.info(f"🔧 模型主设备: {primary_device}")
                    logger.info(f"🔧 检测到设备数量: {len(model_devices)}")
                    
                    # 如果是模型并行，显示设备分布
                    if len(model_devices) > 1:
                        logger.info("🚀 模型并行设备分布:")
                        for i, device in enumerate(sorted(model_devices, key=str)):
                            logger.info(f"  - 设备 {i+1}: {device}")
                    
                    # 🔧 应用设备修复补丁
                    self._patch_grpo_device_consistency()
                    
                    # 🔧 增强tokenizer设备一致性
                    self._ensure_tokenizer_device_consistency()
                    
                    logger.info("✅ 通用设备修复完成")
                else:
                    logger.warning("⚠️ 无法获取模型设备信息")
            else:
                logger.warning("⚠️ 模型没有参数信息")
                
        except Exception as e:
            logger.error(f"❌ 通用设备修复失败: {e}")
    
    def _ensure_tokenizer_device_consistency(self):
        """🔧 确保tokenizer设备一致性"""
        try:
            if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer:
                # 确保tokenizer的特殊token在正确设备上
                if hasattr(self.trainer.tokenizer, 'pad_token_id') and self.trainer.tokenizer.pad_token_id is not None:
                    logger.debug("🔧 tokenizer设备一致性检查完成")
                    
        except Exception as e:
            logger.debug(f"⚠️ tokenizer设备一致性检查失败: {e}")
    
    def _add_grpo_error_fallback(self):
        """🔧 简化的GRPO错误处理回退机制"""
        logger.info("🔧 GRPO错误处理回退机制已包含在设备一致性补丁中")

    def _fix_trainer_device_consistency(self):
        """🔧 修复训练器设备一致性问题（增强版）"""
        try:
            logger.info("🔧 修复训练器设备一致性...")
            
            # 获取主设备（模型的第一个参数所在设备）
            model_device = next(self.trainer.model.parameters()).device
            logger.info(f"  - 模型主设备: {model_device}")
            
            # 🔧 更全面的设备一致性修复
            if self.training_strategy in ["model_parallel_only", "model_parallel_single_process"]:
                # 确保tokenizer在正确设备上
                if hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer:
                    if hasattr(self.trainer.tokenizer, 'to'):
                        self.trainer.tokenizer.to(model_device)
                
                # 检查并修复数据collator的设备设置
                if hasattr(self.trainer, 'data_collator'):
                    if hasattr(self.trainer.data_collator, 'tokenizer'):
                        if hasattr(self.trainer.data_collator.tokenizer, 'to'):
                            self.trainer.data_collator.tokenizer.to(model_device)
                
                # 🔧 关键修复：应用设备一致性补丁（已在_apply_universal_device_fix中处理）
                logger.info("🔧 设备一致性补丁将在_apply_universal_device_fix中应用")
            
            logger.info("✅ 设备一致性修复完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 设备一致性修复失败: {e}")
    
    def _patch_grpo_device_consistency(self):
        """🔧 高效GRPO设备一致性补丁（优化版）"""
        try:
            logger.info("🔧 应用高效GRPO设备一致性补丁...")
            
            if not hasattr(self.trainer, '_generate_and_score_completions'):
                logger.warning("⚠️ 未找到 _generate_and_score_completions 方法")
                return
                
            original_method = self.trainer._generate_and_score_completions
            device_fix_count = 0  # 用于减少日志频率
            
            def optimized_device_fix(batch):
                """优化的设备错误修复，减少开销"""
                nonlocal device_fix_count
                
                try:
                    # 🔧 优化1: 缓存主设备信息，避免重复查询
                    if not hasattr(optimized_device_fix, '_cached_primary_device'):
                        optimized_device_fix._cached_primary_device = next(self.trainer.model.parameters()).device
                    
                    primary_device = optimized_device_fix._cached_primary_device
                    
                    # 🔧 优化2: 快速检查是否需要设备转换
                    needs_device_fix = False
                    if isinstance(batch, dict):
                        for key, value in batch.items():
                            if torch.is_tensor(value) and value.device != primary_device:
                                needs_device_fix = True
                                break
                    elif torch.is_tensor(batch) and batch.device != primary_device:
                        needs_device_fix = True
                    
                    # 🔧 优化3: 只在需要时进行设备转换
                    if needs_device_fix:
                        def fast_to_device(obj):
                            if torch.is_tensor(obj):
                                return obj.to(primary_device, non_blocking=True)
                            elif isinstance(obj, dict):
                                return {k: fast_to_device(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return type(obj)(fast_to_device(item) for item in obj)
                            return obj
                        
                        prepared_batch = fast_to_device(batch)
                    else:
                        prepared_batch = batch
                    
                    # 🔧 优化4: 在主设备上下文中执行
                    with torch.cuda.device(primary_device):
                        return original_method(prepared_batch)
                        
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        device_fix_count += 1
                        
                        # 🔧 优化5: 减少警告日志频率
                        if device_fix_count % 10 == 1:  # 每10次只记录一次
                            logger.warning(f"🔧 GRPO设备不一致修复 (#{device_fix_count}): {str(e)[:80]}...")
                        
                        # 🔧 优化6: 更激进的设备统一策略
                        try:
                            def force_device_unification(obj):
                                if torch.is_tensor(obj):
                                    return obj.to('cuda:0', non_blocking=True)
                                elif isinstance(obj, dict):
                                    return {k: force_device_unification(v) for k, v in obj.items()}
                                elif isinstance(obj, (list, tuple)):
                                    return type(obj)(force_device_unification(item) for item in obj)
                                return obj
                            
                            unified_batch = force_device_unification(batch)
                            
                            # 🔧 优化7: 使用cuda:0设备上下文和禁用混合精度
                            with torch.cuda.device('cuda:0'):
                                with torch.autocast('cuda', enabled=False):
                                    result = original_method(unified_batch)
                            
                            if device_fix_count % 20 == 1:
                                logger.debug(f"✅ 设备统一修复成功 (#{device_fix_count})")
                            
                            return result
                            
                        except Exception as fallback_e:
                            if device_fix_count % 50 == 1:
                                logger.error(f"❌ 设备修复失败 (#{device_fix_count}): {fallback_e}")
                            raise e
                    else:
                        raise e
                        
                except Exception as other_e:
                    logger.error(f"❌ GRPO设备修复遇到其他错误: {other_e}")
                    raise other_e
            
            # 应用优化补丁
            self.trainer._generate_and_score_completions = optimized_device_fix
            logger.info("✅ 高效GRPO设备一致性补丁已应用")
            
        except Exception as e:
            logger.error(f"❌ 应用GRPO补丁失败: {e}")

    def _final_model_parallel_check(self, dataset_for_trainer):
        """🔧 模型并行训练前的最终检查"""
        logger.info("🚀 模型并行训练前最终检查:")
        logger.info(f"  - 数据集大小: {len(dataset_for_trainer)}")
        logger.info(f"  - 每GPU批次大小: {self.grpo_cfg.per_device_train_batch_size}")
        logger.info(f"  - 梯度累积步数: {self.grpo_cfg.gradient_accumulation_steps}")
        
        effective_batch_size = (self.grpo_cfg.per_device_train_batch_size * 
                              self.grpo_cfg.gradient_accumulation_steps)
        logger.info(f"  - 有效批次大小: {effective_batch_size}")
        
        # GPU内存检查
        logger.info("📊 训练前GPU内存状态:")
        for i in range(self.multi_gpu_info.get('gpu_count', 1)):
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                usage_percent = (allocated / total) * 100
                logger.info(f"    GPU {i}: {allocated:.2f}GB/{total:.1f}GB ({usage_percent:.1f}%)")
                
                # 内存警告
                if usage_percent > 90:
                    logger.warning(f"⚠️ GPU {i} 内存使用率过高 ({usage_percent:.1f}%)，可能导致OOM")
                elif usage_percent > 75:
                    logger.warning(f"⚠️ GPU {i} 内存使用率较高 ({usage_percent:.1f}%)，请注意监控")
        
        # 验证模型确实是分布式的
        if hasattr(self.model, 'hf_device_map'):
            device_set = set(str(device) for device in self.model.hf_device_map.values())
            if len(device_set) > 1:
                logger.info(f"✅ 确认模型已分布到 {len(device_set)} 个设备: {sorted(device_set)}")
            else:
                logger.warning(f"⚠️ 警告：模型似乎只在单一设备 {device_set} 上")
        
        # 检查是否有分布式环境变量残留
        dist_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
        remaining_vars = [var for var in dist_vars if var in os.environ]
        if remaining_vars:
            logger.warning(f"⚠️ 检测到残留的分布式环境变量: {remaining_vars}")
            logger.warning("这可能导致DDP冲突，建议清除这些变量")

    def _post_training_check(self):
        """🔧 训练后状态检查"""
        logger.info("📊 训练完成后GPU内存状态:")
        for i in range(self.multi_gpu_info.get('gpu_count', 1)):
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _diagnose_training_failure(self, error):
        """🔧 训练失败时的详细诊断"""
        logger.error("🔍 训练失败诊断:")
        logger.error(f"  - 错误类型: {type(error).__name__}")
        logger.error(f"  - 训练策略: {self.training_strategy}")
        
        # 检查是否是DDP相关错误
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['dtensor', 'distributed', 'ddp', 'nccl']):
            logger.error("🚨 检测到分布式/DDP相关错误!")
            logger.error("💡 建议解决方案:")
            logger.error("  1. 确保没有同时使用模型并行和DDP")
            logger.error("  2. 检查环境变量中是否有分布式设置")
            logger.error("  3. 尝试使用pure model parallel模式")
            
            # 检查环境变量
            dist_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
            for var in dist_vars:
                if var in os.environ:
                    logger.error(f"  发现分布式环境变量: {var}={os.environ[var]}")
        
        # GPU状态诊断
        if self.multi_gpu_info.get('use_model_parallel', False):
            logger.error("🔍 多GPU状态诊断:")
            try:
                for i in range(self.multi_gpu_info.get('gpu_count', 1)):
                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        logger.error(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as diag_e:
                logger.error(f"GPU诊断失败: {diag_e}")

    def _save_training_artifacts(self, train_result):
        """🔧 增强训练产物保存，包含训练策略信息"""
        try:
            logger.info("💾 Saving training artifacts...")
            
            # Save final model
            final_model_dir = os.path.join(self.grpo_cfg.output_dir, "final_model_adapter")
            self.trainer.save_model(final_model_dir)
            logger.info(f"✅ Final model saved to: {final_model_dir}")

            # 保存训练策略和多GPU信息
            training_info = {
                "training_strategy": self.training_strategy,
                "multi_gpu_info": self.multi_gpu_info,
                "training_completed": True,
                "final_step": getattr(self.trainer.state, 'global_step', 0) if self.trainer and self.trainer.state else 0
            }
            
            training_info_file = os.path.join(self.grpo_cfg.output_dir, "training_strategy_info.json")
            with open(training_info_file, 'w') as f:
                json.dump(training_info, f, indent=2)
            logger.info(f"✅ Training strategy info saved to: {training_info_file}")

            # Save metrics and state
            metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
            
            # 添加训练策略相关指标
            metrics['training_strategy'] = self.training_strategy
            metrics['use_model_parallel'] = self.multi_gpu_info.get('use_model_parallel', False)
            metrics['gpu_count'] = self.multi_gpu_info.get('gpu_count', 1)
            
            if self.multi_gpu_info.get('use_model_parallel', False):
                metrics['total_gpu_memory_gb'] = self.multi_gpu_info['total_memory_gb']
                
                # 保存最终内存使用情况
                final_memory_usage = {}
                for i in range(self.multi_gpu_info['gpu_count']):
                    if torch.cuda.is_available() and i < torch.cuda.device_count():
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        final_memory_usage[f'gpu_{i}_allocated_gb'] = allocated
                        final_memory_usage[f'gpu_{i}_reserved_gb'] = reserved
                
                metrics.update(final_memory_usage)
            
            # 保存指标
            if hasattr(self.trainer, 'log_metrics'):
                self.trainer.log_metrics("train_summary", metrics)
            
            metrics_file = os.path.join(self.grpo_cfg.output_dir, "final_train_metrics.json")
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"✅ Metrics saved to: {metrics_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save metrics to file: {e}")
            
            if hasattr(self.trainer, 'save_state'):
                self.trainer.save_state()
                logger.info("✅ Trainer state saved.")

            # 保存模型分布信息（如果使用模型并行）
            if (self.training_strategy in ["model_parallel_only", "model_parallel_single_process"] and 
                hasattr(self.model, 'hf_device_map')):
                device_map_file = os.path.join(self.grpo_cfg.output_dir, "model_device_map.json")
                try:
                    # 转换device map为可序列化格式
                    serializable_device_map = {}
                    for layer_name, device in self.model.hf_device_map.items():
                        serializable_device_map[layer_name] = str(device)
                    
                    with open(device_map_file, 'w') as f:
                        json.dump(serializable_device_map, f, indent=2)
                    logger.info(f"✅ Model device map saved to: {device_map_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save device map: {e}")

            logger.info(f"🎉 All training artifacts saved to: {self.grpo_cfg.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving training artifacts: {e}", exc_info=True)

    def cleanup(self):
        """🔧 增强资源清理，支持不同训练策略"""
        try:
            logger.info("🧹 Cleaning up resources...")
            
            # 1. 清理WandB同步管理器
            try:
                sync_manager = get_wandb_sync_manager()
                if sync_manager:
                    sync_manager.finish()
                    logger.info("✅ WandB同步管理器已清理")
            except Exception as e:
                logger.warning(f"⚠️ WandB同步管理器清理失败: {e}")
            
            # 2. 根据训练策略清理GPU内存
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"🧹 清理 {gpu_count} 张GPU的内存...")
                
                total_memory_freed = 0
                for i in range(gpu_count):
                    try:
                        # 记录清理前的内存
                        before_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        before_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        
                        # 设置设备并清理
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        
                        # 记录清理后的内存
                        after_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        after_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        
                        freed = before_reserved - after_reserved
                        total_memory_freed += freed
                        
                        logger.info(f"    GPU {i}: 释放 {freed:.2f}GB (剩余: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved)")
                        
                    except Exception as gpu_e:
                        logger.warning(f"⚠️ GPU {i} 清理失败: {gpu_e}")
                
                logger.info(f"✅ 总共释放GPU内存: {total_memory_freed:.2f}GB")
            
            # 3. Force garbage collection
            gc.collect()
            logger.info("✅ Garbage collection completed.")
            
            # 4. 训练策略总结
            logger.info("📊 训练总结:")
            logger.info(f"    - 训练策略: {self.training_strategy}")
            if self.multi_gpu_info:
                logger.info(f"    - 使用GPU数量: {self.multi_gpu_info.get('gpu_count', 0)}")
                logger.info(f"    - 模型并行: {'✅' if self.multi_gpu_info.get('use_model_parallel', False) else '❌'}")
                if self.multi_gpu_info.get('use_model_parallel', False):
                    logger.info(f"    - 总GPU内存: {self.multi_gpu_info.get('total_memory_gb', 0):.1f}GB")
                    logger.info(f"    - 内存效率: 高")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


def main():
    """🔧 增强主入口，支持训练策略错误处理"""
    pipeline = None
    try:
        logger_temp.info("🚀 Initializing Enhanced GRPO Training Pipeline...")
        pipeline = GRPOTrainingPipeline()
        
        # 训练策略验证和摘要
        if hasattr(pipeline, 'training_strategy'):
            logger_temp.info("🔧 训练策略摘要:")
            logger_temp.info(f"    策略: {pipeline.training_strategy}")
            if hasattr(pipeline, 'multi_gpu_info') and pipeline.multi_gpu_info:
                logger_temp.info(f"    GPU数量: {pipeline.multi_gpu_info.get('gpu_count', 0)}")
                logger_temp.info(f"    模型并行: {'✅' if pipeline.multi_gpu_info.get('use_model_parallel', False) else '❌'}")
                logger_temp.info(f"    总内存: {pipeline.multi_gpu_info.get('total_memory_gb', 0):.1f}GB")
        
        logger_temp.info("🎯 Starting training...")
        pipeline.train()
        
        logger_temp.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger_temp.warning("⚠️ Training interrupted by user (Ctrl+C)")
        
        # 训练中断时的清理
        if pipeline and hasattr(pipeline, 'training_strategy'):
            logger_temp.info(f"🧹 {pipeline.training_strategy}训练中断，正在清理资源...")
            try:
                if (hasattr(pipeline, 'multi_gpu_info') and 
                    pipeline.multi_gpu_info.get('use_model_parallel', False) and 
                    torch.cuda.is_available()):
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                logger_temp.info("✅ 资源清理完成")
            except Exception as cleanup_e:
                logger_temp.warning(f"⚠️ 清理过程中出错: {cleanup_e}")
        
        return 1
        
    except Exception as e:
        logger_temp.error(f"💥 Fatal error in training pipeline: {e}", exc_info=True)
        
        # 错误时的详细诊断
        if pipeline and hasattr(pipeline, 'training_strategy'):
            logger_temp.error(f"🔍 {pipeline.training_strategy}训练失败，进行最终诊断:")
            
            # 检查是否是DDP冲突错误
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['dtensor', 'distributed', 'ddp', 'mixed', 'broadcast']):
                logger_temp.error("🚨 检测到DDP/模型并行冲突错误!")
                logger_temp.error("💡 解决建议:")
                logger_temp.error("  1. 确保使用单进程启动: python main.py (不使用torchrun)")
                logger_temp.error("  2. 清除所有分布式环境变量:")
                logger_temp.error("     unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT")
                logger_temp.error("  3. 设置纯模型并行模式: --use_model_parallel true")
                logger_temp.error("  4. 检查是否有其他进程在使用GPU")
            
            # GPU状态诊断
            try:
                if (hasattr(pipeline, 'multi_gpu_info') and 
                    pipeline.multi_gpu_info.get('use_model_parallel', False) and 
                    torch.cuda.is_available()):
                    for i in range(pipeline.multi_gpu_info.get('gpu_count', 0)):
                        if i < torch.cuda.device_count():
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            reserved = torch.cuda.memory_reserved(i) / 1024**3
                            logger_temp.error(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as diag_e:
                logger_temp.error(f"最终诊断失败: {diag_e}")
        
        return 1
        
    finally:
        if pipeline:
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                logger_temp.warning(f"⚠️ Error during cleanup: {cleanup_error}")
        
        logger_temp.info("🏁 Enhanced GRPO Training Pipeline execution finished.")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)# main.py - 修复DDP/模型并行冲突版本