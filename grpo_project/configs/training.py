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
    # dataset_path can be primary for local files, dataset_name for hub datasets.
    dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to a local dataset manifest JSONL file. Used if dataset_name is not provided."})

    # --- Fields WITH default values ---
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library). E.g., 'c4', 'wikitext'. If provided, dataset_path might be ignored or used for local loading."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    dataset_split: str = field(default="train", metadata={"help": "The dataset split to use (e.g., 'train', 'validation')."})
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

    def __post_init__(self):
        # Note: output_dir is handled by GRPOConfig (TrainingArguments)
        # We'll access it via grpo_cfg.output_dir in the pipeline
        pass
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
