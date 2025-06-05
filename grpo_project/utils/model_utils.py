import logging
from transformers import GenerationConfig

logger = logging.getLogger(__name__)

def setup_qwen3_generation_config(model, tokenizer, script_cfg):
    """设置Qwen3的生成配置"""

    logger.info("🤖 设置Qwen3生成配置...")

    # 确保tokenizer配置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("设置pad_token为eos_token")

    # 更新模型配置
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    # 设置生成配置
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=script_cfg.max_seq_length // 2, # Ensure this is a reasonable value
        do_sample=True,
        temperature=getattr(script_cfg, 'gen_temperature', 0.7),
        top_p=getattr(script_cfg, 'gen_top_p', 0.8),
        top_k=getattr(script_cfg, 'gen_top_k', 40),
        repetition_penalty=getattr(script_cfg, 'gen_repetition_penalty', 1.05),
        length_penalty=getattr(script_cfg, 'gen_length_penalty', 1.0),
    )

    model.generation_config = generation_config
    logger.info("✅ Qwen3生成配置设置完成")

    return model, tokenizer
