import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig, # Added for Qwen3CompatibilityFixer potentially
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from typing import Optional, Any, Dict, List
# Assuming these config classes are passed in or accessible
# from grpo_project.configs import ScriptConfig, TrainingConfig (GRPOConfig)
# from grpo_project.utils.model_utils import Qwen3CompatibilityFixer # Or wherever it is

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, script_cfg, grpo_cfg, model_name_or_path: str, cache_dir: Optional[str] = None):
        self.script_cfg = script_cfg
        self.grpo_cfg = grpo_cfg # This is TrainingArguments/GRPOConfig
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None

    def setup_model_and_tokenizer(self):
        """
        Loads the base model and tokenizer.
        Applies quantization and compatibility fixes.
        """
        logger.info(f"Loading base model from: {self.model_name_or_path}")
        model_dtype = torch.bfloat16 if self.grpo_cfg.bf16 else (torch.float16 if self.grpo_cfg.fp16 else torch.float32)

        quantization_config_arg = None
        # Assuming script_cfg has a flag like use_quantization and relevant bnb params
        if getattr(self.script_cfg, 'use_quantization', False):
            try:
                quant_config_dict = {
                    "load_in_4bit": getattr(self.script_cfg, 'load_in_4bit', True),
                    "bnb_4bit_quant_type": getattr(self.script_cfg, 'bnb_4bit_quant_type', "nf4"),
                    "bnb_4bit_compute_dtype": model_dtype, # Ensure compute_dtype is compatible
                    "bnb_4bit_use_double_quant": getattr(self.script_cfg, 'bnb_4bit_use_double_quant', True),
                }
                # Filter for valid BitsAndBytesConfig args if necessary
                valid_bnb_keys = BitsAndBytesConfig.__annotations__.keys()
                filtered_quant_config = {k: v for k, v in quant_config_dict.items() if k in valid_bnb_keys}

                quantization_config_arg = BitsAndBytesConfig(**filtered_quant_config)
                logger.info(f"BitsAndBytes quantization will be used with config: {filtered_quant_config}")
            except ImportError:
                logger.warning("bitsandbytes not installed, quantization disabled.")
            except Exception as e_quant:
                logger.warning(f"Failed to create BitsAndBytesConfig, quantization disabled: {e_quant}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=quantization_config_arg,
            device_map="auto" if self.grpo_cfg.world_size == 1 and torch.cuda.is_available() and not self.grpo_cfg.fsdp else None,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            use_cache=False if self.grpo_cfg.gradient_checkpointing else True,
            cache_dir=self.cache_dir
        )
        logger.info("Base model loaded.")

        # k-bit training preparation
        model_is_quantized = (quantization_config_arg is not None and
                              hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit) or \
                             (hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit)

        if model_is_quantized:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=self.grpo_cfg.gradient_checkpointing)
            logger.info(f"Base model prepared for k-bit training. Grad checkpointing: {self.grpo_cfg.gradient_checkpointing}.")
        elif self.grpo_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for non-k-bit model.")

        # Load tokenizer
        tokenizer_load_path = self.model_name_or_path
        if self.script_cfg.stage1_adapter_path and os.path.isdir(self.script_cfg.stage1_adapter_path):
            potential_tokenizer_path = self.script_cfg.stage1_adapter_path
            if os.path.exists(os.path.join(potential_tokenizer_path, "tokenizer_config.json")):
                tokenizer_load_path = potential_tokenizer_path
                logger.info(f"Loading tokenizer from stage1_adapter_path: {tokenizer_load_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path, trust_remote_code=True, use_fast=True, cache_dir=self.cache_dir
        )
        logger.info(f"Tokenizer loaded from: {tokenizer_load_path}")

        # Apply Qwen3CompatibilityFixer if available
        try:
            from grpo_project.utils.model_utils import Qwen3CompatibilityFixer
            self.model, self.tokenizer = Qwen3CompatibilityFixer.fix_generation_config(self.model, self.tokenizer)
            logger.info("Applied Qwen3CompatibilityFixer.")
        except ImportError:
            logger.debug("Qwen3CompatibilityFixer not found, skipping this fix.") # Changed to debug
        except Exception as e_compat:
            logger.warning(f"Error during Qwen3CompatibilityFixer application: {e_compat}, skipping this fix.")

        return self.model, self.tokenizer

    def apply_peft_adapter(self, model, is_resuming: bool, resume_checkpoint_path: Optional[str] = None):
        """
        Applies PEFT adapter to the model.
        Handles resuming from checkpoint, loading stage1 adapter, or creating a new one.
        """
        if model is None:
            logger.error("Model is None, cannot apply PEFT adapter.")
            raise ValueError("Model not loaded before applying PEFT adapter.")

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

        if is_resuming and resume_checkpoint_path:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_path}")
            if os.path.exists(os.path.join(resume_checkpoint_path, 'adapter_config.json')):
                try:
                    if not is_model_already_peft:
                        model = PeftModel.from_pretrained(model, resume_checkpoint_path, is_trainable=True)
                        logger.info("PEFT adapter loaded from checkpoint for base model.")
                        peft_applied_successfully = True
                    else:
                        logger.info("Model is already PEFT. Trainer will handle adapter state from checkpoint.")
                        peft_applied_successfully = True
                except Exception as e:
                    logger.error(f"Failed to load PEFT adapter from checkpoint {resume_checkpoint_path}: {e}.")
            else:
                logger.warning(f"No adapter_config.json found in checkpoint {resume_checkpoint_path}.")

        if not peft_applied_successfully and self.script_cfg.stage1_adapter_path and \
           os.path.isdir(self.script_cfg.stage1_adapter_path):
            if is_model_already_peft:
                logger.warning("Model is already PEFT, skipping Stage 1 adapter loading.")
            else:
                try:
                    logger.info(f"Loading Stage 1 PEFT adapter from: {self.script_cfg.stage1_adapter_path}")
                    model = PeftModel.from_pretrained(model, self.script_cfg.stage1_adapter_path, is_trainable=True)
                    logger.info("Stage 1 PEFT adapter loaded successfully.")
                    peft_applied_successfully = True
                except Exception as e:
                    logger.error(f"Failed to load Stage 1 PEFT adapter: {e}. Will attempt to create a new one.")

        if not peft_applied_successfully and not is_model_already_peft:
            try:
                logger.info("Creating new PEFT adapter...")
                model = get_peft_model(model, peft_config)
                logger.info("New PEFT adapter created successfully.")
            except Exception as e:
                logger.error(f"Failed to create new PEFT adapter: {e}", exc_info=True)
                raise RuntimeError("Could not apply PEFT adapter to model.") from e

        if isinstance(model, PeftModel):
            model.print_trainable_parameters()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                logger.error("CRITICAL: No trainable parameters found after PEFT setup.")
                try:
                    model.train()
                    if hasattr(model, 'enable_adapters'): model.enable_adapters()
                    # Ensure all adapters are set to trainable if multiple exist
                    for adapter_name in model.peft_config.keys():
                         model.set_adapter(adapter_name) # This might make only one active, check peft docs
                         # model.peft_config[adapter_name].is_trainable = True # Might be needed

                    trainable_params_after_fix = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    if trainable_params_after_fix > 0:
                        logger.info(f"Successfully enabled trainable PEFT parameters: {trainable_params_after_fix}")
                        model.print_trainable_parameters()
                    else:
                        raise RuntimeError("Still no trainable PEFT parameters after attempting fix.")
                except Exception as e_fix:
                    raise RuntimeError(f"Failed to enable trainable PEFT parameters: {e_fix}") from e_fix
        elif not peft_applied_successfully : # If it's not PEFT and we expected it to be
             logger.error("Model is not a PeftModel and no PEFT adapter was successfully applied.")
             # This case might be an error depending on whether PEFT is mandatory.
             # For now, just log, but could raise error.

        return model
