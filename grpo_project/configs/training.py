from dataclasses import dataclass, field
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class ScriptConfig:
    """
    Configuration for the training script execution, paths, and non-GRPO model/data parameters.
    """
    # --- Fields WITHOUT default values ---
    model_name_or_path: str = field(metadata={"help": "Path to pretrained BASE model or model identifier from huggingface.co/models for GRPO training."})
    dataset_path: str = field(metadata={"help": "Path to the dataset manifest JSONL file."})

    # --- Fields WITH default values ---
    # curriculum_stages is now Optional and has a default, so it belongs in this section.
    curriculum_stages: Optional[List[Dict[str, Any]]] = field(default=None, metadata={"help": "Detailed dual-layer curriculum stages configuration. Populated dynamically by the script if not provided."})
    # dataset_base_path, output_dir_base, cache_dir moved to EnvConfig
    stage1_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapters from the first stage of training. If provided, these will be loaded and training will continue on them."}
    )

    # Enhanced LoRA config
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank for GRPO stage. Increased from 8 for better capacity."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha for GRPO stage. Scaled with rank."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout for GRPO stage."})
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of modules to target for LoRA in GRPO stage."}
    )

    max_seq_length: int = field(default=4096, metadata={"help": "Maximum sequence length for tokenizer and model."})

    # Enhanced callback config
    callback_num_samples: int = field(default=3, metadata={"help": "Number of samples to generate in InferenceCallback."})
    callback_eval_every_n_steps: int = field(default=25, metadata={"help": "Frequency of running InferenceCallback. Reduced for more monitoring."})

    # Enhanced generation parameters
    gen_temperature: float = field(default=0.8, metadata={"help": "Temperature for generation during GRL. Increased for diversity."})
    gen_top_k: int = field(default=50, metadata={"help": "Top-k for generation during GRL."})
    gen_top_p: float = field(default=0.95, metadata={"help": "Top-p (nucleus) for generation during GRL."})
    gen_repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty to avoid repetitive code."})
    gen_length_penalty: float = field(default=1.0, metadata={"help": "Length penalty for generation."})

    # Curriculum learning config (Enhanced with dual-layer support)
    enable_curriculum: bool = field(default=True, metadata={"help": "Enable dual-layer curriculum learning strategy."})
    curriculum_type: str = field(default="dual_layer", metadata={"help": "Type of curriculum: 'dual_layer', 'complexity_only', 'level_only'"})
    curriculum_focus_levels: List[str] = field(
        default_factory=lambda: ["basic", "intermediate", "advanced", "expert"],
        metadata={"help": "Dataset levels to include in curriculum learning."}
    )
    curriculum_complexity_emphasis: str = field(
        default="balanced",
        metadata={"help": "Complexity emphasis: 'simple', 'balanced', 'complex'"}
    )

    # Experience replay config
    enable_experience_replay: bool = field(default=True, metadata={"help": "Enable experience replay buffer."})
    experience_buffer_size: int = field(default=1000, metadata={"help": "Size of experience replay buffer."})
    replay_sample_ratio: float = field(default=0.2, metadata={"help": "Ratio of replay samples in each batch."})

    # --- Field initialized in __post_init__ ---
    # output_dir will now be set by the TrainingOrchestrator using EnvConfig.output_dir_base and run-specific name.
    # So, it's removed as a field from here, or TrainingOrchestrator will set it directly.
    # For now, removing its dynamic creation logic from this __post_init__.
    # The TrainingOrchestrator will be responsible for self.script_cfg.output_dir = self.actual_output_dir

    def __post_init__(self):
        # The output_dir logic based on output_dir_base is removed.
        # TrainingOrchestrator will determine and set script_cfg.output_dir.

        # Cache dir logic is also removed as it's now in EnvConfig.
        # If there are other initializations for ScriptConfig, they would remain here.
        pass # Keep pass if no other post-initialization logic remains

@dataclass
class OptimizedTrainingConfig:
    """优化的训练配置"""

    # 基础训练参数
    max_steps: int = 300
    learning_rate: float = 1e-5
    eval_steps: int = 2
    save_steps: int = 10
    warmup_steps: int = 20

    # 课程学习参数
    curriculum_performance_threshold: float = 0.55
    curriculum_progression_patience: int = 15
    advanced_stage_min_steps: int = 50  # 高难度阶段最少训练步数

    # 监控参数
    detailed_logging_interval: int = 10
    performance_analysis_interval: int = 50

def apply_optimized_config(script_cfg, grpo_cfg, opt_config: OptimizedTrainingConfig = None):
    """将优化配置应用到现有配置对象"""
    if opt_config is None:
        opt_config = OptimizedTrainingConfig()

    # 应用到 script_cfg
    if hasattr(script_cfg, 'max_steps'):
        script_cfg.max_steps = opt_config.max_steps
    if hasattr(script_cfg, 'curriculum_performance_threshold'):
        script_cfg.curriculum_performance_threshold = opt_config.curriculum_performance_threshold

    # 应用到 grpo_cfg
    grpo_cfg.learning_rate = opt_config.learning_rate
    grpo_cfg.eval_steps = opt_config.eval_steps
    grpo_cfg.save_steps = opt_config.save_steps
    grpo_cfg.warmup_steps = opt_config.warmup_steps

    return script_cfg, grpo_cfg
