import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map
from typing import Optional, Any, Dict, List
import json

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, script_cfg, grpo_cfg, model_name_or_path: str, cache_dir: Optional[str] = None):
        self.script_cfg = script_cfg
        self.grpo_cfg = grpo_cfg # This is TrainingArguments/GRPOConfig
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None
        
        # 🔧 新增：多GPU配置检测
        self.num_gpus = torch.cuda.device_count()
        self.use_multi_gpu = self.num_gpus >= 2
        
        if self.use_multi_gpu:
            logger.info(f"🚀 检测到 {self.num_gpus} 张GPU，启用模型并行模式")
        else:
            logger.info(f"📱 检测到 {self.num_gpus} 张GPU，使用单GPU模式")

    def _get_multi_gpu_config(self):
        """🔧 配置多GPU模型并行参数"""
        if not self.use_multi_gpu:
            return None, None
            
        # 为每张A100配置内存限制（保留5GB缓冲）
        max_memory = {}
        for i in range(self.num_gpus):
            max_memory[i] = "75GiB"  # 每张80G A100留5G缓冲
            
        logger.info(f"🔧 多GPU内存配置: {max_memory}")
        
        # 设备映射策略
        device_map = "auto"  # 让accelerate自动分配
        
        return device_map, max_memory

    def _apply_model_optimizations(self, model):
        """🔧 应用模型优化"""
        try:
            # 启用Flash Attention (如果可用)
            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                logger.info("✅ 启用 Flash Attention 2")
            
            # 优化模型配置
            if hasattr(model.config, 'torch_dtype'):
                model.config.torch_dtype = torch.bfloat16 if self.grpo_cfg.bf16 else torch.float16
                
            # 设置生成配置
            if hasattr(model, 'generation_config'):
                model.generation_config.do_sample = True
                model.generation_config.temperature = 0.7
                model.generation_config.top_p = 0.9
                model.generation_config.max_new_tokens = 1024
                
            return model
            
        except Exception as e:
            logger.warning(f"⚠️ 模型优化失败: {e}")
            return model

    def setup_model_and_tokenizer(self):
        """
        Loads the base model and tokenizer with multi-GPU support.
        Applies quantization and compatibility fixes.
        """
        logger.info(f"Loading base model from: {self.model_name_or_path}")
        model_dtype = torch.bfloat16 if self.grpo_cfg.bf16 else (torch.float16 if self.grpo_cfg.fp16 else torch.float32)

        # 🔧 量化配置（多GPU时需要特殊处理）
        quantization_config_arg = None
        if getattr(self.script_cfg, 'use_quantization', False):
            if self.use_multi_gpu:
                logger.warning("⚠️ 多GPU模式下禁用量化以避免兼容性问题")
            else:
                try:
                    quant_config_dict = {
                        "load_in_4bit": getattr(self.script_cfg, 'load_in_4bit', True),
                        "bnb_4bit_quant_type": getattr(self.script_cfg, 'bnb_4bit_quant_type', "nf4"),
                        "bnb_4bit_compute_dtype": model_dtype,
                        "bnb_4bit_use_double_quant": getattr(self.script_cfg, 'bnb_4bit_use_double_quant', True),
                    }
                    valid_bnb_keys = BitsAndBytesConfig.__annotations__.keys()
                    filtered_quant_config = {k: v for k, v in quant_config_dict.items() if k in valid_bnb_keys}

                    quantization_config_arg = BitsAndBytesConfig(**filtered_quant_config)
                    logger.info(f"BitsAndBytes quantization enabled: {filtered_quant_config}")
                except ImportError:
                    logger.warning("bitsandbytes not installed, quantization disabled.")
                except Exception as e_quant:
                    logger.warning(f"Failed to create BitsAndBytesConfig: {e_quant}")

        # 🔧 获取多GPU配置
        device_map, max_memory = self._get_multi_gpu_config()
        
        # 🔧 模型加载参数配置
        model_kwargs = {
            "quantization_config": quantization_config_arg,
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "use_cache": False if self.grpo_cfg.gradient_checkpointing else True,
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": True,  # 重要：减少CPU内存使用
        }
        
        # 🔧 多GPU配置
        if self.use_multi_gpu and not quantization_config_arg and not getattr(self.grpo_cfg, 'fsdp', None):
            model_kwargs.update({
                "device_map": device_map,
                "max_memory": max_memory,
            })
            logger.info("🚀 使用多GPU模型并行配置")
        elif torch.cuda.is_available() and self.grpo_cfg.world_size == 1 and not getattr(self.grpo_cfg, 'fsdp', None):
            model_kwargs["device_map"] = "auto"
            logger.info("📱 使用单GPU自动设备映射")
        else:
            logger.info("💻 使用默认设备配置")

        # 🔧 加载模型
        try:
            logger.info("🔄 开始加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            )
            logger.info("✅ 模型加载成功")
            
            # 🔧 应用模型优化
            self.model = self._apply_model_optimizations(self.model)
            
            # 🔧 记录模型设备分布情况
            self._log_model_device_info()
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            # 降级到CPU加载，然后手动分发
            logger.info("🔄 尝试CPU加载后手动分发...")
            model_kwargs.pop('device_map', None)
            model_kwargs.pop('max_memory', None)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            )
            
            if self.use_multi_gpu:
                self.model = self._manual_device_dispatch()

        # k-bit training preparation
        model_is_quantized = (quantization_config_arg is not None and
                              hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit) or \
                             (hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit)

        if model_is_quantized:
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=self.grpo_cfg.gradient_checkpointing
            )
            logger.info(f"Base model prepared for k-bit training. Grad checkpointing: {self.grpo_cfg.gradient_checkpointing}.")
        elif self.grpo_cfg.gradient_checkpointing and not self.use_multi_gpu:
            # 🔧 多GPU时谨慎使用梯度检查点
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for non-k-bit model.")

        # 🔧 加载tokenizer
        self._load_tokenizer()

        # Apply Qwen3CompatibilityFixer if available
        self._apply_compatibility_fixes()

        return self.model, self.tokenizer

    def _manual_device_dispatch(self):
        """🔧 手动设备分发（备用方案）"""
        try:
            logger.info("🔧 执行手动设备分发...")
            
            # 推断设备映射
            device_map = infer_auto_device_map(
                self.model,
                max_memory={i: "75GiB" for i in range(self.num_gpus)},
                no_split_module_classes=["LlamaDecoderLayer", "QWenBlock", "TransformerBlock"]
            )
            
            logger.info(f"🗺️ 设备映射: {json.dumps(device_map, indent=2)}")
            
            # 分发模型
            model = dispatch_model(self.model, device_map=device_map)
            logger.info("✅ 手动设备分发完成")
            
            return model
            
        except Exception as e:
            logger.warning(f"⚠️ 手动设备分发失败: {e}")
            return self.model

    def _log_model_device_info(self):
        """🔧 记录模型设备分布信息"""
        try:
            if hasattr(self.model, 'hf_device_map'):
                device_map = self.model.hf_device_map
                logger.info("📍 模型设备分布:")
                
                # 统计每个设备的层数
                device_counts = {}
                for layer_name, device in device_map.items():
                    device_str = str(device)
                    device_counts[device_str] = device_counts.get(device_str, 0) + 1
                
                for device, count in device_counts.items():
                    logger.info(f"  - {device}: {count} 层")
                    
                # 检查GPU内存使用
                if torch.cuda.is_available():
                    for i in range(self.num_gpus):
                        if i < torch.cuda.device_count():
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            reserved = torch.cuda.memory_reserved(i) / 1024**3
                            logger.info(f"  - GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
        except Exception as e:
            logger.debug(f"设备信息记录失败: {e}")

    def _load_tokenizer(self):
        """🔧 加载tokenizer"""
        tokenizer_load_path = self.model_name_or_path
        if self.script_cfg.stage1_adapter_path and os.path.isdir(self.script_cfg.stage1_adapter_path):
            potential_tokenizer_path = self.script_cfg.stage1_adapter_path
            if os.path.exists(os.path.join(potential_tokenizer_path, "tokenizer_config.json")):
                tokenizer_load_path = potential_tokenizer_path
                logger.info(f"Loading tokenizer from stage1_adapter_path: {tokenizer_load_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path, 
            trust_remote_code=True, 
            use_fast=True, 
            cache_dir=self.cache_dir
        )
        
        # 🔧 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("设置 pad_token = eos_token")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("添加新的 pad_token: [PAD]")
        
        logger.info(f"Tokenizer loaded from: {tokenizer_load_path}")

    def _apply_compatibility_fixes(self):
        """🔧 应用兼容性修复"""
        try:
            from grpo_project.utils.model_utils import Qwen3CompatibilityFixer
            self.model, self.tokenizer = Qwen3CompatibilityFixer.fix_generation_config(self.model, self.tokenizer)
            logger.info("Applied Qwen3CompatibilityFixer.")
        except ImportError:
            logger.debug("Qwen3CompatibilityFixer not found, skipping this fix.")
        except Exception as e_compat:
            logger.warning(f"Error during Qwen3CompatibilityFixer application: {e_compat}, skipping this fix.")

    def apply_peft_adapter(self, model, is_resuming: bool, resume_checkpoint_path: Optional[str] = None):
        """
        Applies PEFT adapter to the model with multi-GPU support.
        Handles resuming from checkpoint, loading stage1 adapter, or creating a new one.
        """
        if model is None:
            logger.error("Model is None, cannot apply PEFT adapter.")
            raise ValueError("Model not loaded before applying PEFT adapter.")

        # 🔧 多GPU环境下的PEFT配置
        peft_config = LoraConfig(
            r=self.script_cfg.lora_rank,
            lora_alpha=self.script_cfg.lora_alpha,
            lora_dropout=self.script_cfg.lora_dropout,
            target_modules=self.script_cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        peft_applied_successfully = False
        is_model_already_peft = isinstance(model, PeftModel)

        # 🔧 处理断续训练
        if is_resuming and resume_checkpoint_path:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
            if os.path.exists(os.path.join(resume_checkpoint_path, 'adapter_config.json')):
                try:
                    if not is_model_already_peft:
                        # 🔧 多GPU环境下加载PEFT模型
                        model = PeftModel.from_pretrained(
                            model, 
                            resume_checkpoint_path, 
                            is_trainable=True,
                            # 对于多GPU分布式模型，PEFT会自动处理设备映射
                        )
                        logger.info("PEFT adapter loaded from checkpoint for base model.")
                        peft_applied_successfully = True
                    else:
                        logger.info("Model is already PEFT. Trainer will handle adapter state from checkpoint.")
                        peft_applied_successfully = True
                except Exception as e:
                    logger.error(f"Failed to load PEFT adapter from checkpoint {resume_checkpoint_path}: {e}.")
            else:
                logger.warning(f"No adapter_config.json found in checkpoint {resume_checkpoint_path}.")

        # 🔧 处理Stage 1适配器加载
        if not peft_applied_successfully and self.script_cfg.stage1_adapter_path and \
           os.path.isdir(self.script_cfg.stage1_adapter_path):
            if is_model_already_peft:
                logger.warning("Model is already PEFT, skipping Stage 1 adapter loading.")
            else:
                try:
                    logger.info(f"Loading Stage 1 PEFT adapter from: {self.script_cfg.stage1_adapter_path}")
                    model = PeftModel.from_pretrained(
                        model, 
                        self.script_cfg.stage1_adapter_path, 
                        is_trainable=True
                    )
                    logger.info("Stage 1 PEFT adapter loaded successfully.")
                    peft_applied_successfully = True
                except Exception as e:
                    logger.error(f"Failed to load Stage 1 PEFT adapter: {e}. Will attempt to create a new one.")

        # 🔧 创建新的PEFT适配器
        if not peft_applied_successfully and not is_model_already_peft:
            try:
                logger.info("Creating new PEFT adapter...")
                model = get_peft_model(model, peft_config)
                logger.info("New PEFT adapter created successfully.")
                
                # 🔧 多GPU环境下的PEFT优化
                if self.use_multi_gpu:
                    self._optimize_peft_for_multi_gpu(model)
                    
            except Exception as e:
                logger.error(f"Failed to create new PEFT adapter: {e}", exc_info=True)
                raise RuntimeError("Could not apply PEFT adapter to model.") from e

        # 🔧 验证PEFT模型
        if isinstance(model, PeftModel):
            model.print_trainable_parameters()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                logger.error("CRITICAL: No trainable parameters found after PEFT setup.")
                self._fix_peft_trainable_params(model)
        elif not peft_applied_successfully:
            logger.error("Model is not a PeftModel and no PEFT adapter was successfully applied.")

        return model

    def _optimize_peft_for_multi_gpu(self, model):
        """🔧 多GPU环境下的PEFT优化"""
        try:
            # 确保PEFT适配器在正确的设备上
            if hasattr(model, 'peft_config'):
                for adapter_name in model.peft_config.keys():
                    # 设置适配器为可训练
                    model.set_adapter(adapter_name)
                    logger.info(f"✅ 设置适配器 {adapter_name} 为当前活跃适配器")
            
            # 检查模型参数分布
            logger.info("🔍 检查PEFT参数分布:")
            device_param_count = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    device = str(param.device)
                    device_param_count[device] = device_param_count.get(device, 0) + param.numel()
            
            for device, count in device_param_count.items():
                logger.info(f"  - {device}: {count:,} 可训练参数")
                
        except Exception as e:
            logger.warning(f"⚠️ PEFT多GPU优化失败: {e}")

    def _fix_peft_trainable_params(self, model):
        """🔧 修复PEFT可训练参数问题"""
        try:
            model.train()
            if hasattr(model, 'enable_adapters'): 
                model.enable_adapters()
            
            # 确保所有适配器都设置为可训练
            for adapter_name in model.peft_config.keys():
                model.set_adapter(adapter_name)
                
            trainable_params_after_fix = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params_after_fix > 0:
                logger.info(f"Successfully enabled trainable PEFT parameters: {trainable_params_after_fix}")
                model.print_trainable_parameters()
            else:
                raise RuntimeError("Still no trainable PEFT parameters after attempting fix.")
                
        except Exception as e_fix:
            raise RuntimeError(f"Failed to enable trainable PEFT parameters: {e_fix}") from e_fix

    def get_model_memory_info(self):
        """🔧 获取模型内存使用信息"""
        info = {
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "gpu_memory": {}
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info["gpu_memory"][f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024**3,
                    "reserved": torch.cuda.memory_reserved(i) / 1024**3,
                    "max_memory": torch.cuda.get_device_properties(i).total_memory / 1024**3
                }
        
        return info