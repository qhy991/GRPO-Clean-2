# grpo_project/validation/config_validator.py
import logging
import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import fields
from pathlib import Path

from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
from trl import GRPOConfig

logger = logging.getLogger(__name__)

class ConfigValidator:
    """配置验证器，确保训练配置的合理性"""
    
    def __init__(self):
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def validate_all_configs(
        self, 
        env_cfg: EnvConfig, 
        script_cfg: ScriptConfig, 
        reward_cfg: EnhancedRewardConfig, 
        grpo_cfg: GRPOConfig
    ) -> Tuple[bool, List[str], List[str]]:
        """
        验证所有配置
        
        Returns:
            Tuple[bool, List[str], List[str]]: (是否通过验证, 错误列表, 警告列表)
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info("🔍 Starting configuration validation...")
        
        # 验证各个配置模块
        self._validate_env_config(env_cfg)
        self._validate_script_config(script_cfg)
        self._validate_reward_config(reward_cfg)
        self._validate_grpo_config(grpo_cfg)
        
        # 跨配置验证
        self._validate_config_compatibility(env_cfg, script_cfg, reward_cfg, grpo_cfg)
        
        # 系统资源验证
        self._validate_system_resources(grpo_cfg)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error(f"❌ Configuration validation failed with {len(self.validation_errors)} errors")
            
        if self.validation_warnings:
            logger.warning(f"⚠️ {len(self.validation_warnings)} warnings found")
            
        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
    
    def _validate_env_config(self, env_cfg: EnvConfig):
        """验证环境配置"""
        # 验证输出目录路径
        if not env_cfg.output_dir_base:
            self.validation_errors.append("output_dir_base cannot be empty")
        else:
            try:
                Path(env_cfg.output_dir_base).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.validation_errors.append(f"Cannot create output_dir_base '{env_cfg.output_dir_base}': {e}")
        
        # 验证缓存目录
        if env_cfg.cache_dir:
            try:
                Path(env_cfg.cache_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.validation_warnings.append(f"Cannot create cache_dir '{env_cfg.cache_dir}': {e}")
        
        # 验证代理设置格式
        if env_cfg.http_proxy and not env_cfg.http_proxy.startswith(('http://', 'https://')):
            self.validation_warnings.append("http_proxy should start with 'http://' or 'https://'")
            
        if env_cfg.https_proxy and not env_cfg.https_proxy.startswith(('http://', 'https://')):
            self.validation_warnings.append("https_proxy should start with 'http://' or 'https://'")
    
    def _validate_script_config(self, script_cfg: ScriptConfig):
        """验证脚本配置"""
        # 验证模型路径
        if not script_cfg.model_name_or_path:
            self.validation_errors.append("model_name_or_path is required")
        
        # 验证数据集路径
        if not script_cfg.dataset_path and not script_cfg.dataset_name:
            self.validation_errors.append("Either dataset_path or dataset_name must be provided")
            
        if script_cfg.dataset_path and not os.path.exists(script_cfg.dataset_path):
            self.validation_errors.append(f"Dataset file not found: {script_cfg.dataset_path}")
        
        # 验证Stage1适配器路径
        if script_cfg.stage1_adapter_path and not os.path.exists(script_cfg.stage1_adapter_path):
            self.validation_warnings.append(f"Stage1 adapter path not found: {script_cfg.stage1_adapter_path}")
        
        # 验证LoRA参数
        if script_cfg.lora_rank <= 0:
            self.validation_errors.append("lora_rank must be positive")
            
        if script_cfg.lora_alpha <= 0:
            self.validation_errors.append("lora_alpha must be positive")
            
        if not 0 <= script_cfg.lora_dropout <= 1:
            self.validation_errors.append("lora_dropout must be between 0 and 1")
        
        # 验证序列长度
        if script_cfg.max_seq_length <= 0:
            self.validation_errors.append("max_seq_length must be positive")
            
        if script_cfg.max_seq_length > 32768:
            self.validation_warnings.append("max_seq_length > 32768 may cause memory issues")
        
        # 验证回调参数
        if script_cfg.callback_num_samples <= 0:
            self.validation_warnings.append("callback_num_samples should be positive for meaningful evaluation")
            
        if script_cfg.callback_eval_every_n_steps <= 0:
            self.validation_errors.append("callback_eval_every_n_steps must be positive")
        
        # 验证生成参数
        if not 0 < script_cfg.gen_temperature <= 2.0:
            self.validation_warnings.append("gen_temperature should be between 0 and 2.0")
            
        if not 0 < script_cfg.gen_top_p <= 1.0:
            self.validation_warnings.append("gen_top_p should be between 0 and 1.0")
            
        if script_cfg.gen_top_k <= 0:
            self.validation_warnings.append("gen_top_k should be positive")
    
    def _validate_reward_config(self, reward_cfg: EnhancedRewardConfig):
        """验证奖励配置"""
        # 验证奖励权重总和
        total_weight = (reward_cfg.functional_weight + reward_cfg.efficiency_weight + 
                       reward_cfg.readability_weight + reward_cfg.robustness_weight)
        
        if abs(total_weight - 1.0) > 0.01:
            self.validation_warnings.append(f"Reward weights sum to {total_weight:.3f}, not 1.0")
        
        # 验证奖励范围
        if reward_cfg.reward_clipping_range <= 0:
            self.validation_errors.append("reward_clipping_range must be positive")
            
        if reward_cfg.max_functional_reward <= 0:
            self.validation_warnings.append("max_functional_reward should be positive")
        
        # 验证惩罚值（应该是负数）
        penalty_fields = ['compilation_failure', 'simulation_crash', 'output_parse_error', 
                         'missing_code_block_penalty', 'timeout_penalty', 'resource_usage_penalty']
        
        for field_name in penalty_fields:
            field_value = getattr(reward_cfg, field_name, 0)
            if field_value > 0:
                self.validation_warnings.append(f"{field_name} is positive ({field_value}), should typically be negative")
    
    def _validate_grpo_config(self, grpo_cfg: GRPOConfig):
        """验证GRPO训练配置"""
        # 验证学习率
        if grpo_cfg.learning_rate <= 0:
            self.validation_errors.append("learning_rate must be positive")
            
        if grpo_cfg.learning_rate > 1e-2:
            self.validation_warnings.append(f"learning_rate ({grpo_cfg.learning_rate}) seems high for fine-tuning")
        
        # 验证批次大小
        if grpo_cfg.per_device_train_batch_size <= 0:
            self.validation_errors.append("per_device_train_batch_size must be positive")
            
        if grpo_cfg.gradient_accumulation_steps <= 0:
            self.validation_errors.append("gradient_accumulation_steps must be positive")
        
        # 验证保存和评估频率
        if grpo_cfg.save_steps <= 0:
            self.validation_warnings.append("save_steps should be positive to enable checkpointing")
            
        if grpo_cfg.eval_steps and grpo_cfg.eval_steps <= 0:
            self.validation_warnings.append("eval_steps should be positive for evaluation")
        
        # 验证输出目录
        if not grpo_cfg.output_dir:
            self.validation_errors.append("output_dir is required")
        
        # 验证恢复checkpoint路径
        if (grpo_cfg.resume_from_checkpoint and 
            isinstance(grpo_cfg.resume_from_checkpoint, str) and
            not os.path.exists(grpo_cfg.resume_from_checkpoint)):
            self.validation_warnings.append(f"resume_from_checkpoint path not found: {grpo_cfg.resume_from_checkpoint}")
    
    def _validate_config_compatibility(self, env_cfg: EnvConfig, script_cfg: ScriptConfig, 
                                     reward_cfg: EnhancedRewardConfig, grpo_cfg: GRPOConfig):
        """验证配置间的兼容性"""
        # 验证序列长度与完成长度的关系
        if hasattr(grpo_cfg, 'max_completion_length'):
            if grpo_cfg.max_completion_length >= script_cfg.max_seq_length:
                self.validation_errors.append(
                    f"max_completion_length ({grpo_cfg.max_completion_length}) must be < max_seq_length ({script_cfg.max_seq_length})"
                )
        
        # 验证batch size与gradient accumulation的组合
        effective_batch_size = grpo_cfg.per_device_train_batch_size * grpo_cfg.gradient_accumulation_steps
        if grpo_cfg.world_size > 1:
            effective_batch_size *= grpo_cfg.world_size
            
        if effective_batch_size > 128:
            self.validation_warnings.append(f"Large effective batch size ({effective_batch_size}) may cause memory issues")
        
        # 验证经验回放与batch size的关系
        if script_cfg.enable_experience_replay:
            if script_cfg.experience_buffer_size < effective_batch_size * 10:
                self.validation_warnings.append(
                    "experience_buffer_size should be at least 10x the effective batch size for meaningful replay"
                )
    
    def _validate_system_resources(self, grpo_cfg: GRPOConfig):
        """验证系统资源"""
        # 验证CUDA可用性
        if grpo_cfg.device and grpo_cfg.device.startswith('cuda'):
            if not torch.cuda.is_available():
                self.validation_errors.append("CUDA device specified but CUDA is not available")
            else:
                device_count = torch.cuda.device_count()
                if grpo_cfg.world_size > device_count:
                    self.validation_warnings.append(
                        f"world_size ({grpo_cfg.world_size}) > available GPUs ({device_count})"
                    )
        
        # 验证内存需求（粗略估算）
        if torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                
                # 粗略估算模型内存需求（基于批次大小和序列长度）
                estimated_memory = (grpo_cfg.per_device_train_batch_size * 
                                  grpo_cfg.gradient_accumulation_steps * 
                                  4 * 1e-9)  # 4 bytes per parameter, rough estimate
                
                if estimated_memory > total_memory * 0.8:
                    self.validation_warnings.append(
                        f"Estimated memory usage ({estimated_memory:.1f}GB) may exceed 80% of GPU memory ({total_memory:.1f}GB)"
                    )
            except Exception as e:
                logger.debug(f"Could not estimate memory usage: {e}")