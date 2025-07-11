#!/bin/bash
# Qwen3 优化训练命令
# 修复了学习率、批处理大小、奖励函数和课程学习阈值

torchrun \
    --nproc_per_node 2 \
    --master_addr localhost \
    --master_port 14524 \
    /home/qhy/Research/LLM/GRPO-Clean-2/main.py \
    --hf_endpoint "https://hf-mirror.com" \
    --http_proxy "http://10.130.148.206:7890" \
    --https_proxy "http://10.130.148.206:7890" \
    --wandb_project "VerilogGRPO_Enhanced_8B_Fixed" \
    --wandb_entity "qhy0227-tsinghua-university" \
    --wandb_run_name_prefix "qwen3-fixed-v1" \
    --model_name_or_path "/home/share/Qwen3-8B/models--Qwen--Qwen3-8B/snapshots/a80f5e57cce20e57b65145f4213844dec1a80834" \
    --stage1_adapter_path "/home/share/ReasoningV/Qwen3/ckpts/sft-qwen3-lora-5epoch-origen200k-S1-20250429_211831/checkpoint-34695/" \
    --dataset_path "/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl" \
    --dataset_base_path /home/qhy/Research/LLM/GRPO-RV/dataset \
    --output_dir_base "./enhanced_grpo_8B_runs_fixed" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_seq_length 5120 \
    --script_max_prompt_length 1024 \
    --script_max_completion_length 4096 \
    --length_allocation_strategy custom \
    --callback_num_samples 2 \
    --callback_eval_every_n_steps 3 \
    --learning_rate 2e-05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine_with_restarts \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gen_temperature 0.8 \
    --gen_top_k 50 \
    --gen_top_p 0.9 \
    --gen_repetition_penalty 1.02 \
    --compilation_success 3.0 \
    --compilation_failure -2.0 \
    --simulation_crash -2.0 \
    --test_pass_base_reward 2.0 \
    --test_pass_bonus_multiplier 1.5 \
    --max_functional_reward 20.0 \
    --all_tests_passed_bonus 8.0 \
    --functional_weight 0.8 \
    --efficiency_weight 0.1 \
    --readability_weight 0.05 \
    --robustness_weight 0.05 \
    --curriculum_performance_threshold_1 0.4 \
    --curriculum_performance_threshold_2 0.5 \
    --curriculum_performance_threshold_3 0.6 \
    --curriculum_min_evaluations 3 \
    --curriculum_performance_check_interval 3 \
    --enable_streaming_guidance true \
    --min_reasoning_length 80 \
    --guidance_trigger_threshold 30 \
    --max_guidance_attempts 3 \
    --guidance_tokens_limit 40 \
    --num_train_epochs 4 \
    --logging_strategy "steps" \
    --logging_steps 3 \
    --save_strategy "steps" \
    --save_steps 15 \
    --save_total_limit 5 \
    --report_to "wandb" \
    --remove_unused_columns False \
    --num_generations 2 \
    --max_prompt_length 1024 \
    --max_completion_length 4096 \
    --optim "adamw_torch" \
    --ddp_find_unused_parameters False \
    --seed 42 \
    --dataloader_pin_memory \
    --bf16 \
    --cache_dir "/home/qhy/Research/LLM/GRPO-Clean-2/.enhanced_cache_v2/models"
