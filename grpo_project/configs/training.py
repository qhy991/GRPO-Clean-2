from dataclasses import dataclass, field
import os
import torch
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class ScriptConfig:
    """
    Enhanced configuration for training script with multi-GPU model parallel support.
    """
    # --- Core Model Configuration ---
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained BASE model or model identifier from huggingface.co/models for GRPO training."}
    )
    
    # --- Dataset Configuration ---
    dataset_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to a local dataset manifest JSONL file. Used if dataset_name is not provided."}
    )
    dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library). E.g., 'c4', 'wikitext'."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split: str = field(
        default="train", 
        metadata={"help": "The dataset split to use (e.g., 'train', 'validation')."}
    )
    
    # --- Stage1 Adapter Configuration ---
    stage1_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapters from the first stage of training."}
    )

    # 🔧 多GPU模型并行配置
    use_model_parallel: bool = field(
        default=True,
        metadata={"help": "Enable multi-GPU model parallelism for distributing model weights across GPUs instead of replicating them."}
    )
    max_memory_per_gpu: str = field(
        default="75GiB",
        metadata={"help": "Maximum memory allocation per GPU. Default: 75GiB for 80GB A100s with 5GB buffer."}
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Enable low CPU memory usage mode during model loading."}
    )
    device_map_strategy: str = field(
        default="auto",
        metadata={"help": "Device mapping strategy: 'auto' (automatic), 'balanced' (manual balancing), 'custom' (user-defined)."}
    )
    force_device_map: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Custom device mapping dictionary. Only used when device_map_strategy='custom'."}
    )
    disable_quantization_for_multi_gpu: bool = field(
        default=True,
        metadata={"help": "Automatically disable quantization when using multi-GPU to avoid compatibility issues."}
    )

    # 🔧 增强的LoRA配置（针对多GPU优化）
    lora_rank: int = field(
        default=64, 
        metadata={"help": "LoRA rank for GRPO stage. Increased for multi-GPU to utilize additional memory."}
    )
    lora_alpha: int = field(
        default=128, 
        metadata={"help": "LoRA alpha for GRPO stage. Scaled with rank (typically 2x rank)."}
    )
    lora_dropout: float = field(
        default=0.05, 
        metadata={"help": "LoRA dropout for GRPO stage."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of modules to target for LoRA in GRPO stage."}
    )

    # 🔧 多GPU优化的长度配置
    max_seq_length: int = field(
        default=6144, 
        metadata={"help": "Maximum sequence length. Increased for multi-GPU to utilize additional memory."}
    )
    script_max_prompt_length: int = field(
        default=1536, 
        metadata={"help": "Maximum length for input prompts. Default: 1536 tokens (~25% of 6144)."}
    )
    script_max_completion_length: int = field(
        default=4608, 
        metadata={"help": "Maximum length for model completions. Default: 4608 tokens (~75% of 6144)."}
    )
    
    length_allocation_strategy: str = field(
        default="custom", 
        metadata={"help": "Length allocation strategy: 'balanced', 'prompt_heavy', 'completion_heavy', 'custom'."}
    )

    # 🚀 流式引导配置
    enable_streaming_guidance: bool = field(
        default=True, 
        metadata={"help": "Enable streaming reasoning guidance for better code generation."}
    )
    min_reasoning_length: int = field(
        default=60, 
        metadata={"help": "Minimum reasoning length requirement for guidance trigger."}
    )
    guidance_trigger_threshold: int = field(
        default=40, 
        metadata={"help": "Reasoning length threshold to trigger guidance intervention."}
    )
    max_guidance_attempts: int = field(
        default=2, 
        metadata={"help": "Maximum number of guidance attempts per generation."}
    )
    guidance_tokens_limit: int = field(
        default=25, 
        metadata={"help": "Maximum tokens per guidance intervention."}
    )

    # 🔧 多GPU优化的回调配置
    callback_num_samples: int = field(
        default=3, 
        metadata={"help": "Number of samples to generate in InferenceCallback."}
    )
    callback_eval_every_n_steps: int = field(
        default=15, 
        metadata={"help": "Frequency of running InferenceCallback. Reduced for multi-GPU to avoid communication overhead."}
    )

    # 生成参数配置
    gen_temperature: float = field(
        default=0.7, 
        metadata={"help": "Temperature for generation during GRPO. Optimized for code generation."}
    )
    gen_top_k: int = field(
        default=40, 
        metadata={"help": "Top-k for generation during GRPO."}
    )
    gen_top_p: float = field(
        default=0.8, 
        metadata={"help": "Top-p (nucleus) for generation during GRPO."}
    )
    gen_repetition_penalty: float = field(
        default=1.05, 
        metadata={"help": "Repetition penalty to avoid repetitive code generation."}
    )
    gen_length_penalty: float = field(
        default=1.0, 
        metadata={"help": "Length penalty for generation."}
    )

    # 🔧 多GPU优化的课程学习配置
    enable_curriculum: bool = field(
        default=True, 
        metadata={"help": "Enable dual-layer curriculum learning strategy."}
    )
    curriculum_type: str = field(
        default="dual_layer", 
        metadata={"help": "Type of curriculum: 'dual_layer', 'complexity_only', 'level_only'."}
    )
    curriculum_focus_levels: List[str] = field(
        default_factory=lambda: ["basic", "intermediate", "advanced", "expert"],
        metadata={"help": "Dataset levels to include in curriculum learning."}
    )
    curriculum_complexity_emphasis: str = field(
        default="balanced",
        metadata={"help": "Complexity emphasis: 'simple', 'balanced', 'complex'."}
    )
    curriculum_performance_check_interval: int = field(
        default=10,
        metadata={"help": "Steps between performance checks for curriculum advancement. Increased for multi-GPU to reduce communication."}
    )

    # 课程学习性能阈值参数
    curriculum_performance_threshold_1: Optional[float] = field(
        default=None,
        metadata={"help": "Performance threshold for curriculum stage 1 (foundation)."}
    )
    curriculum_performance_threshold_2: Optional[float] = field(
        default=None,
        metadata={"help": "Performance threshold for curriculum stage 2 (elementary)."}
    )
    curriculum_performance_threshold_3: Optional[float] = field(
        default=None,
        metadata={"help": "Performance threshold for curriculum stage 3 (intermediate)."}
    )
    curriculum_performance_threshold_4: Optional[float] = field(
        default=None,
        metadata={"help": "Performance threshold for curriculum stage 4 (advanced)."}
    )
    curriculum_performance_threshold_5: Optional[float] = field(
        default=None,
        metadata={"help": "Performance threshold for curriculum stage 5 (expert)."}
    )
    curriculum_min_evaluations: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum evaluations before advancing to next curriculum stage."}
    )

    # 🔧 多GPU优化的经验回放配置
    enable_experience_replay: bool = field(
        default=True, 
        metadata={"help": "Enable experience replay buffer."}
    )
    experience_buffer_size: int = field(
        default=1500, 
        metadata={"help": "Size of experience replay buffer. Increased for multi-GPU."}
    )
    replay_sample_ratio: float = field(
        default=0.2, 
        metadata={"help": "Ratio of replay samples in each batch."}
    )

    # 🔧 多GPU性能优化配置
    enable_flash_attention: bool = field(
        default=True,
        metadata={"help": "Enable Flash Attention 2 if available for better memory efficiency."}
    )
    optimize_memory_usage: bool = field(
        default=True,
        metadata={"help": "Enable memory usage optimizations for multi-GPU training."}
    )
    reduce_communication_overhead: bool = field(
        default=True,
        metadata={"help": "Reduce multi-GPU communication overhead by optimizing callback frequency."}
    )
    gradient_checkpointing_compatible: bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient checkpointing with multi-GPU (may cause sync issues)."}
    )

    # 量化配置（多GPU时通常禁用）
    use_quantization: bool = field(
        default=False,
        metadata={"help": "Enable quantization. Automatically disabled for multi-GPU if disable_quantization_for_multi_gpu=True."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Enable 4-bit quantization."}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4-bit quantization type for BitsAndBytes."}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Enable double quantization for BitsAndBytes."}
    )

    # 课程学习阶段配置
    curriculum_stages: Optional[List[Dict[str, Any]]] = field(
        default=None, 
        metadata={"help": "Detailed dual-layer curriculum stages configuration."}
    )

    def __post_init__(self):
        """配置后处理和验证"""
        
        # 🔧 多GPU量化检查
        if self.use_model_parallel and self.disable_quantization_for_multi_gpu and self.use_quantization:
            print("⚠️ 多GPU模式下自动禁用量化以避免兼容性问题")
            self.use_quantization = False
            self.load_in_4bit = False
        
        # 🔧 自动调整长度配置
        if self.length_allocation_strategy != "custom":
            if self.length_allocation_strategy == "balanced":
                self.script_max_prompt_length = self.max_seq_length // 2
                self.script_max_completion_length = self.max_seq_length // 2
            elif self.length_allocation_strategy == "prompt_heavy":
                self.script_max_prompt_length = int(self.max_seq_length * 0.6)
                self.script_max_completion_length = self.max_seq_length - self.script_max_prompt_length
            elif self.length_allocation_strategy == "completion_heavy":
                self.script_max_prompt_length = int(self.max_seq_length * 0.4)
                self.script_max_completion_length = self.max_seq_length - self.script_max_prompt_length
        
        # 🔧 验证长度配置
        total_length = self.script_max_prompt_length + self.script_max_completion_length
        if total_length > self.max_seq_length:
            print(f"⚠️ 警告: prompt长度({self.script_max_prompt_length}) + completion长度({self.script_max_completion_length}) = {total_length} > 最大序列长度({self.max_seq_length})")
            print(f"   自动调整completion长度为: {self.max_seq_length - self.script_max_prompt_length}")
            self.script_max_completion_length = self.max_seq_length - self.script_max_prompt_length
        
        # 🔧 多GPU环境检查和配置调整
        if self.use_model_parallel:
            self._validate_multi_gpu_environment()
            self._adjust_config_for_multi_gpu()
    
    def _validate_multi_gpu_environment(self):
        """验证多GPU环境"""
        try:
            if not torch.cuda.is_available():
                print("⚠️ CUDA不可用，禁用多GPU模式")
                self.use_model_parallel = False
                return
            
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                print(f"⚠️ 检测到{gpu_count}张GPU，少于2张，禁用多GPU模式")
                self.use_model_parallel = False
                return
            
            print(f"✅ 多GPU环境验证通过，检测到{gpu_count}张GPU")
            
            # 验证GPU内存
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")
                
                if memory_gb < 40:  # 至少40GB
                    print(f"⚠️ GPU {i} 内存不足({memory_gb:.1f}GB)，可能影响大模型训练")
            
        except Exception as e:
            print(f"⚠️ 多GPU环境检查失败: {e}")
            self.use_model_parallel = False
    
    def _adjust_config_for_multi_gpu(self):
        """根据多GPU环境调整配置"""
        if not self.use_model_parallel:
            return
        
        # 🔧 调整回调频率以减少通信开销
        if self.reduce_communication_overhead:
            if self.callback_eval_every_n_steps < 10:
                print(f"📊 多GPU模式下调整回调频率: {self.callback_eval_every_n_steps} -> 10")
                self.callback_eval_every_n_steps = 10
            
            if self.curriculum_performance_check_interval < 10:
                print(f"📚 多GPU模式下调整课程检查频率: {self.curriculum_performance_check_interval} -> 10")
                self.curriculum_performance_check_interval = 10
        
        # 🔧 根据GPU数量调整经验回放缓冲区大小
        try:
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                # 多GPU可以支持更大的缓冲区
                original_size = self.experience_buffer_size
                self.experience_buffer_size = max(self.experience_buffer_size, 1500)
                if self.experience_buffer_size != original_size:
                    print(f"💾 多GPU模式下调整经验回放缓冲区大小: {original_size} -> {self.experience_buffer_size}")
        except:
            pass
    
    def validate_config(self) -> bool:
        """验证配置的完整性和合理性"""
        errors = []
        warnings = []
        
        # 验证必要字段
        if not self.model_name_or_path:
            errors.append("model_name_or_path 不能为空")
        
        # 验证长度配置
        if self.script_max_prompt_length <= 0:
            errors.append(f"script_max_prompt_length ({self.script_max_prompt_length}) 必须大于0")
        
        if self.script_max_completion_length <= 0:
            errors.append(f"script_max_completion_length ({self.script_max_completion_length}) 必须大于0")
        
        total_length = self.script_max_prompt_length + self.script_max_completion_length
        if total_length > self.max_seq_length:
            errors.append(f"总长度 ({total_length}) 超过最大序列长度 ({self.max_seq_length})")
        
        # 验证LoRA配置
        if self.lora_rank <= 0 or self.lora_rank > 1024:
            errors.append(f"lora_rank ({self.lora_rank}) 应该在 1-1024 之间")
        
        if self.lora_alpha <= 0:
            errors.append(f"lora_alpha ({self.lora_alpha}) 必须大于0")
        
        # 验证多GPU配置
        if self.use_model_parallel:
            try:
                if torch.cuda.device_count() < 2:
                    warnings.append("启用了多GPU模式但GPU数量少于2")
            except:
                warnings.append("无法检测GPU数量")
        
        # 验证课程学习配置
        if self.enable_curriculum:
            if self.curriculum_performance_check_interval <= 0:
                errors.append(f"curriculum_performance_check_interval ({self.curriculum_performance_check_interval}) 必须大于0")
        
        # 输出结果
        if errors:
            print("❌ 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        if warnings:
            print("⚠️ 配置警告:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("✅ 配置验证通过")
        return True
    
    def get_effective_batch_size(self, per_device_batch_size: int, gradient_accumulation_steps: int) -> int:
        """计算有效批次大小"""
        num_gpus = 1
        if self.use_model_parallel:
            try:
                # 对于模型并行，有效批次大小 = per_device_batch_size * gradient_accumulation_steps * num_gpus
                num_gpus = max(1, torch.cuda.device_count())
            except:
                pass
        
        return per_device_batch_size * gradient_accumulation_steps * num_gpus
    
    def get_memory_estimation(self) -> Dict[str, Any]:
        """估算内存需求"""
        # 基于常见的8B参数模型估算
        model_params = 8e9  # 8B参数
        
        # 计算模型权重内存
        if self.use_quantization and not (self.use_model_parallel and self.disable_quantization_for_multi_gpu):
            bytes_per_param = 1 if self.load_in_4bit else 2  # 4bit or 8bit
        else:
            bytes_per_param = 2  # bfloat16
        
        model_memory_gb = (model_params * bytes_per_param) / (1024**3)
        
        # 计算激活内存（粗略估算）
        batch_size = 1  # 默认假设
        activation_memory_gb = (self.max_seq_length * self.max_seq_length * batch_size * 4) / (1024**3)
        
        # 计算总内存需求
        total_memory_per_gpu = model_memory_gb + activation_memory_gb + 8  # 8GB缓冲
        
        # 多GPU时分摊模型内存
        if self.use_model_parallel:
            try:
                num_gpus = max(1, torch.cuda.device_count())
                model_memory_per_gpu = model_memory_gb / num_gpus
                total_memory_per_gpu = model_memory_per_gpu + activation_memory_gb + 8
            except:
                num_gpus = 1
        else:
            num_gpus = 1
            model_memory_per_gpu = model_memory_gb
        
        return {
            "model_params": f"{model_params/1e9:.1f}B",
            "total_model_memory_gb": model_memory_gb,
            "model_memory_per_gpu_gb": model_memory_per_gpu,
            "activation_memory_gb": activation_memory_gb,
            "total_memory_per_gpu_gb": total_memory_per_gpu,
            "num_gpus": num_gpus,
            "recommended_max_memory": f"{int(total_memory_per_gpu + 5)}GiB",
            "memory_efficiency": "High" if self.use_model_parallel else "Standard"
        }
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("           训练配置摘要")
        print("="*60)
        
        print(f"🎯 模型配置:")
        print(f"  - 模型路径: {self.model_name_or_path}")
        print(f"  - 多GPU模式: {'✅ 启用' if self.use_model_parallel else '❌ 禁用'}")
        print(f"  - 量化: {'✅ 启用' if self.use_quantization else '❌ 禁用'}")
        
        print(f"\n📏 长度配置:")
        print(f"  - 总序列长度: {self.max_seq_length}")
        print(f"  - 提示长度: {self.script_max_prompt_length} ({self.script_max_prompt_length/self.max_seq_length*100:.1f}%)")
        print(f"  - 完成长度: {self.script_max_completion_length} ({self.script_max_completion_length/self.max_seq_length*100:.1f}%)")
        print(f"  - 分配策略: {self.length_allocation_strategy}")
        
        print(f"\n🔧 LoRA配置:")
        print(f"  - Rank: {self.lora_rank}")
        print(f"  - Alpha: {self.lora_alpha}")
        print(f"  - Dropout: {self.lora_dropout}")
        print(f"  - 目标模块: {len(self.lora_target_modules)} 个")
        
        print(f"\n📚 课程学习:")
        print(f"  - 启用: {'✅' if self.enable_curriculum else '❌'}")
        print(f"  - 类型: {self.curriculum_type}")
        print(f"  - 检查间隔: {self.curriculum_performance_check_interval} 步")
        
        print(f"\n💾 经验回放:")
        print(f"  - 启用: {'✅' if self.enable_experience_replay else '❌'}")
        print(f"  - 缓冲区大小: {self.experience_buffer_size}")
        
        # 内存估算
        memory_info = self.get_memory_estimation()
        print(f"\n🧠 内存估算:")
        print(f"  - 模型参数: {memory_info['model_params']}")
        print(f"  - 每GPU模型内存: {memory_info['model_memory_per_gpu_gb']:.1f}GB")
        print(f"  - 每GPU总内存: {memory_info['total_memory_per_gpu_gb']:.1f}GB")
        print(f"  - 推荐配置: {memory_info['recommended_max_memory']}")
        print(f"  - 内存效率: {memory_info['memory_efficiency']}")
        
        print("="*60 + "\n")


@dataclass
class OptimizedTrainingConfig:
    """多GPU优化的训练配置"""

    # 🔧 多GPU优化的基础训练参数
    max_steps: int = 500  # 增加训练步数以充分利用多GPU
    learning_rate: float = 1e-5
    eval_steps: int = 5    # 减少评估频率以降低通信开销
    save_steps: int = 25   # 减少保存频率
    warmup_steps: int = 30 # 适当增加warmup

    # 🔧 多GPU优化的课程学习参数
    curriculum_performance_threshold: float = 0.65  # 适当提高阈值
    curriculum_progression_patience: int = 20       # 增加耐心值
    advanced_stage_min_steps: int = 80              # 高难度阶段更多步数

    # 🔧 多GPU优化的监控参数
    detailed_logging_interval: int = 15   # 减少日志频率
    performance_analysis_interval: int = 75  # 减少分析频率

    # 多GPU特定优化
    enable_gradient_sync_optimization: bool = True
    reduce_callback_frequency: bool = True
    optimize_communication: bool = True

def apply_optimized_config(script_cfg: ScriptConfig, grpo_cfg, opt_config: OptimizedTrainingConfig = None):
    """将多GPU优化配置应用到现有配置对象"""
    if opt_config is None:
        opt_config = OptimizedTrainingConfig()

    # 应用到 script_cfg
    if hasattr(script_cfg, 'max_steps'):
        script_cfg.max_steps = opt_config.max_steps
    
    # 多GPU优化调整
    if script_cfg.use_model_parallel and opt_config.reduce_callback_frequency:
        # 减少回调频率以降低多GPU通信开销
        script_cfg.callback_eval_every_n_steps = max(
            script_cfg.callback_eval_every_n_steps, 
            15
        )
        script_cfg.curriculum_performance_check_interval = max(
            script_cfg.curriculum_performance_check_interval,
            15
        )
        print("🔧 多GPU模式下已优化回调频率")

    # 应用到 grpo_cfg
    grpo_cfg.learning_rate = opt_config.learning_rate
    grpo_cfg.eval_steps = opt_config.eval_steps
    grpo_cfg.save_steps = opt_config.save_steps
    grpo_cfg.warmup_steps = opt_config.warmup_steps
    
    # 多GPU特定的grpo配置调整
    if script_cfg.use_model_parallel:
        # 禁用可能导致同步问题的特性
        if hasattr(grpo_cfg, 'gradient_checkpointing'):
            if not script_cfg.gradient_checkpointing_compatible:
                grpo_cfg.gradient_checkpointing = False
                print("🔧 多GPU模式下已禁用梯度检查点")
        
        # 优化数据加载
        if hasattr(grpo_cfg, 'dataloader_num_workers'):
            grpo_cfg.dataloader_num_workers = min(grpo_cfg.dataloader_num_workers, 2)
            print("🔧 多GPU模式下已优化数据加载器")

    return script_cfg, grpo_cfg