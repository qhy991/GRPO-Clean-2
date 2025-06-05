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
    dataset_base_path: Optional[str] = field(default=None, metadata={"help": "Absolute base path for the dataset. If provided, relative paths in the dataset manifest for 'testbench_path' and 'reference_verilog_path' will be resolved against this."})
    stage1_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapters from the first stage of training. If provided, these will be loaded and training will continue on them."}
    )
    output_dir_base: str = field(default="grpo_verilog_runs_enhanced", metadata={"help": "Base directory for all outputs. A timestamped subdirectory will be created here."})

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to cache downloaded models and datasets. If None, Hugging Face defaults will be used."}
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
    output_dir: str = field(init=False, metadata={"help": "Full path to the output directory for this run (dynamically created)."})

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name_parts = []
        if self.model_name_or_path:
            safe_model_name_parts.append(self.model_name_or_path.split('/')[-1])
        if self.stage1_adapter_path:
             safe_model_name_parts.append("s1adapted")

        safe_model_name = "_".join(safe_model_name_parts) if safe_model_name_parts else "unknown_model"
        self.output_dir = os.path.join(self.output_dir_base, f"enhanced_grpo_{safe_model_name}_{timestamp}")

        if self.cache_dir:
            os.environ['HF_HOME'] = self.cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
            os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
            os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)
            print(f"INFO: HF_HOME and TRANSFORMERS_CACHE set to: {self.cache_dir}")
            print(f"INFO: HF_DATASETS_CACHE set to: {os.environ['HF_DATASETS_CACHE']}")

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
