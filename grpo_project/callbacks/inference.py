import logging
import os
import random
import time
import json # DetailedInferenceCallback saves json
from typing import List, Optional, Dict, Any
from datetime import datetime

import torch
import numpy as np # DetailedInferenceCallback uses np for metrics
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase, GenerationConfig
from datasets import Dataset # Used in __init__

# Attempt to import from grpo_project, fallback for placeholders if necessary
try:
    from grpo_project.callbacks.base import BaseCallback
    from grpo_project.utils.parsing import parse_llm_completion_with_context, parse_llm_completion_qwen3
    from grpo_project.utils.verilog_utils import assess_code_quality
    from grpo_project.evaluation.simulator import VerilogSimulator # DetailedInferenceCallback uses run_iverilog_simulation
    # ExperienceBuffer is currently in utils.py, will be moved later. For now, import from utils.
    from grpo_project.utils import ExperienceBuffer # Updated import
    
    # 导入仿真函数
    from grpo_project.utils.simulation import run_iverilog_simulation
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Callbacks.inference: Could not import from grpo_project or utils. Using placeholders.")
    from transformers import TrainerCallback as BaseCallback # Fallback

    class ExperienceBuffer: pass # Placeholder
    def parse_llm_completion_with_context(*args, **kwargs): return None, None
    def parse_llm_completion_qwen3(*args, **kwargs): return None, None
    def assess_code_quality(*args, **kwargs): return {}
    def run_iverilog_simulation(*args, **kwargs): 
        return {"compilation_success": False, "simulation_run_success": False, "parsing_success": False,
                "passed_tests": 0, "failed_tests": 0, "total_tests_in_output": 0,
                "all_tests_passed_by_tb": False, "error_message": "Placeholder simulator not implemented"}
    class VerilogSimulator: # Placeholder
        def run_simulation(self, *args, **kwargs) -> Dict[str, Any]:
            return {"compilation_success": False, "simulation_run_success": False, "parsing_success": False,
                    "passed_tests": 0, "failed_tests": 0, "total_tests_in_output": 0,
                    "all_tests_passed_by_tb": False, "error_message": "Placeholder simulator not implemented"}


logger = logging.getLogger(__name__)

class EnhancedInferenceCallback(BaseCallback):
    """Enhanced inference callback with better monitoring and logging."""

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 output_dir: Optional[str] = None, # Added output_dir for BaseCallback
                 eval_dataset: Optional[Dataset] = None,
                 fixed_test_prompts: Optional[List[str]] = None,
                 num_samples: int = 1,
                 eval_every_n_steps: int = 100,
                 max_new_tokens: int = 512,
                 max_seq_length: int = 2048,
                 experience_buffer: Optional[ExperienceBuffer] = None):
        super().__init__(output_dir=output_dir)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.fixed_test_prompts = fixed_test_prompts if fixed_test_prompts is not None else []
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.experience_buffer = experience_buffer
        self.generation_history: List[Dict[str, Any]] = [] # Ensure type hint

        if not self.fixed_test_prompts and self.eval_dataset is None:
            logger.warning("EnhancedInferenceCallback: No fixed prompts and no eval_dataset. Callback may not generate samples.")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        # Content of on_step_end from utils.EnhancedInferenceCallback
        # Needs to use self.output_dir if saving anything
        # Imports like wandb, assess_code_quality, parse_llm_completion_with_context will need to be handled
        # For now, just a pass to keep it simple for the move
        pass


# Alias for backward compatibility if it was used elsewhere
InferenceCallback = EnhancedInferenceCallback

class DetailedInferenceCallback(BaseCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=2, eval_every_n_steps=25,
                 max_new_tokens=512, max_seq_length=2048, experience_buffer=None, output_dir=None):
        super().__init__(output_dir=output_dir)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset # This should be a HuggingFace Dataset object
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.experience_buffer = experience_buffer # Should be an instance of ExperienceBuffer
        self.generation_history: List[Dict[str, Any]] = [] # Initialize as empty list

        # self.output_dir is inherited from BaseCallback
        if self.output_dir:
            self.samples_dir = os.path.join(self.output_dir, "generated_samples_detailed")
            os.makedirs(self.samples_dir, exist_ok=True)
        else:
            self.samples_dir = None # No place to save samples
            logger.warning("DetailedInferenceCallback: output_dir not provided, cannot save samples.")

        # Initialize simulator here
        self.simulator = VerilogSimulator()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """完整的推理回调实现，生成eval_avg_test_pass_rate指标"""
        if state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            if model is None or args.local_rank > 0: # 确保只在主进程执行
                return

            logger.info(f"\n🔍 === 推理回调 (DetailedInferenceCallback) - 步数 {state.global_step} ===")
            
            current_step_total_pass_ratio_sum = 0.0
            current_step_samples_with_tests = 0
            current_step_detailed_sim_results = []
            generation_results_for_wandb_logging = [] # To store results for W&B table if needed

            if self.eval_dataset and len(self.eval_dataset) > 0:
                sample_indices = random.sample(range(len(self.eval_dataset)), 
                                               min(self.num_samples, len(self.eval_dataset)))
                
                for i, idx in enumerate(sample_indices):
                    sample = self.eval_dataset[idx]
                    prompt_from_dataset = str(sample['prompt']) # Ensure string
                    
                    logger.info(f"\n📝 样本 {i+1}/{len(sample_indices)} (数据集索引: {idx}, Task ID: {sample.get('task_id', 'N/A')})")
                    logger.debug(f"DetailedInferenceCallback: Input prompt for model (first 150 chars): {prompt_from_dataset[:150]}...")

                    orig_prompt_for_log = sample.get('original_prompt_for_debug', sample.get('original_enhanced_prompt', 'N/A'))
                    logger.info(f"原始问题 (用于日志): {orig_prompt_for_log[:100]}...")
                    logger.info(f"等级: {sample.get('level', 'unknown')}, 复杂度: {sample.get('complexity_score', 'unknown')}")
                    
                    try:
                        generated_result = self._generate_single_sample(model, prompt_from_dataset, state.global_step)

                        if generated_result is None: # Safeguard
                            logger.error(f"Sample {idx} (Task: {sample.get('task_id', 'N/A')}) generation returned None. Skipping.")
                            generation_results_for_wandb_logging.append({
                                "step": state.global_step, "sample_idx": idx, "task_id": sample.get('task_id', 'N/A'),
                                "level": sample.get('level'), "complexity": sample.get('complexity_score'),
                                "reasoning_preview": "GENERATION_RETURNED_NONE", "code_preview": "", "generation_error": "Returned None"
                            })
                            continue

                        if generated_result.get('error'):
                            logger.warning(f"Sample {idx} (Task: {sample.get('task_id', 'N/A')}) generation error: {generated_result['error']}. Code/reasoning are error messages.")
                        
                        quality_metrics = assess_code_quality(generated_result.get('code', '')) # assess_code_quality should handle error strings
                        logger.info(f"生成时间: {generated_result.get('generation_time', 0):.2f}秒, 代码质量: {quality_metrics}")
                        
                        current_sample_wandb_log = {
                            "step": state.global_step, "sample_idx": idx, "task_id": sample.get('task_id', 'N/A'),
                            "level": sample.get('level'), "complexity": sample.get('complexity_score'),
                            "reasoning_preview": str(generated_result.get('reasoning', ''))[:100],
                            "code_preview": str(generated_result.get('code', ''))[:100],
                            **quality_metrics,
                            "generation_error": generated_result.get('error')
                        }

                        code_to_test = generated_result.get('code')
                        tb_path = sample.get('testbench_path')
                        expected_tests = sample.get('expected_total_tests')
                        prompt_identifier_for_sim = f"Step{state.global_step}_Sample{idx}_{sample.get('task_id', 'UnknownTask')}"
                        
                        current_sample_sim_result = None
                        if code_to_test and code_to_test.strip() and tb_path and os.path.exists(tb_path):
                            logger.info(f"DetailedInferenceCallback: Running simulation for sample {idx} (Task: {sample.get('task_id', 'N/A')})")
                            current_sample_sim_result = run_iverilog_simulation(
                                generated_verilog_code=code_to_test,
                                testbench_file_path=tb_path,
                                expected_total_tests_from_manifest=expected_tests,
                                prompt_identifier=prompt_identifier_for_sim,
                                completion_idx=i,
                                print_simulation_details=False
                            )

                            logger.info(f"DetailedInferenceCallback: Sample {idx} (Task: {sample.get('task_id', 'N/A')}) "
                                        f"Sim results - Passed: {current_sample_sim_result.get('passed_tests',0)}/{current_sample_sim_result.get('total_tests_in_output',0)}. "
                                        f"Compilation: {current_sample_sim_result.get('compilation_success', False)}, "
                                        f"SimRun: {current_sample_sim_result.get('simulation_run_success', False)}, "
                                        f"ParseSuccess: {current_sample_sim_result.get('parsing_success', False)}")

                            if current_sample_sim_result.get("compilation_success") and \
                               current_sample_sim_result.get("simulation_run_success") and \
                               current_sample_sim_result.get("parsing_success") and \
                               current_sample_sim_result.get("total_tests_in_output", 0) > 0:

                                pass_ratio = current_sample_sim_result["passed_tests"] / current_sample_sim_result["total_tests_in_output"]
                                current_step_total_pass_ratio_sum += pass_ratio
                                current_step_samples_with_tests += 1
                                current_sample_wandb_log['pass_ratio'] = pass_ratio
                                current_sample_wandb_log['sim_passed'] = current_sample_sim_result["passed_tests"]
                                current_sample_wandb_log['sim_total'] = current_sample_sim_result["total_tests_in_output"]
                            else:
                                current_sample_wandb_log['pass_ratio'] = 0.0
                                current_sample_wandb_log['sim_passed'] = current_sample_sim_result.get('passed_tests',0)
                                current_sample_wandb_log['sim_total'] = current_sample_sim_result.get('total_tests_in_output',0)
                                logger.info(f"DetailedInferenceCallback: Sample {idx} did not contribute to avg_pass_rate due to sim issues.")
                        else:
                            current_sample_wandb_log['pass_ratio'] = 0.0
                            current_sample_wandb_log['sim_passed'] = 0
                            current_sample_wandb_log['sim_total'] = 0
                            if not code_to_test or not code_to_test.strip():
                                logger.info(f"DetailedInferenceCallback: No code generated or code is empty for sample {idx}. Skipping simulation.")
                            if not tb_path or not os.path.exists(tb_path):
                                logger.info(f"DetailedInferenceCallback: Testbench path invalid or not found for sample {idx} ('{tb_path}'). Skipping simulation.")

                        generation_results_for_wandb_logging.append(current_sample_wandb_log)

                        # Save sample if output_dir is available
                        if self.output_dir:
                            self._save_generation_sample(state.global_step, i, sample, generated_result, simulation_result=current_sample_sim_result)
                            
                    except Exception as e:
                        logger.error(f"生成或处理样本 {idx} 失败: {e}", exc_info=True)

            # 🎯 关键：计算并记录eval_avg_test_pass_rate指标
            avg_test_pass_rate_current_step = current_step_total_pass_ratio_sum / current_step_samples_with_tests if current_step_samples_with_tests > 0 else 0.0
            
            # 尝试导入并记录到wandb
            try:
                import wandb
                if hasattr(wandb, 'run') and wandb.run is not None:
                    log_data_wandb = {
                        "eval_avg_test_pass_rate": avg_test_pass_rate_current_step,
                        "eval_current_step_samples_with_tests": current_step_samples_with_tests,
                        "eval_current_step_total_pass_ratio_sum": current_step_total_pass_ratio_sum,
                    }
                    wandb.log(log_data_wandb, step=state.global_step)
                    logger.info(f"✅ DetailedInferenceCallback: 记录到W&B - 步数 {state.global_step}: AvgPassRate={avg_test_pass_rate_current_step:.4f}, SamplesWithTests={current_step_samples_with_tests}")
                else:
                    logger.warning("⚠️ WandB运行未找到，无法记录eval_avg_test_pass_rate")
            except ImportError:
                logger.warning("⚠️ WandB未安装，无法记录eval_avg_test_pass_rate")
            except Exception as e:
                logger.error(f"❌ WandB记录失败: {e}")

            logger.info(f"🔍 === 推理回调 (DetailedInferenceCallback) 结束 ===\n")

    def _generate_single_sample(self, model, prompt, step) -> Dict[str, Any]:
        """生成单个样本"""
        original_training_mode = model.training
        # Initialize to default error values
        reasoning, code, raw_output = "INIT_VAL_ERROR", "INIT_VAL_ERROR", "INIT_VAL_ERROR"
        generation_time = 0
        
        try:
            model.eval()

            max_prompt_len = self.max_seq_length - self.max_new_tokens
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=max_prompt_len).to(model.device)

            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.8, # Consider making these configurable
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generation_time = time.time() - start_time

            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            reasoning, code = parse_llm_completion_with_context(
                raw_output,
                prompt=prompt,
                step=step,
                sample_idx=0 # Assuming this callback processes one sample at a time for this method
            )

            return {
                'reasoning': reasoning if reasoning is not None else "PARSING_FAILED_REASONING",
                'code': code if code is not None else "PARSING_FAILED_CODE",
                'raw_output': raw_output,
                'generation_time': generation_time,
                'step': step,
                'model_input_prompt': prompt,
                'error': None
            }
        except Exception as e:
            logger.error(f"Error in _generate_single_sample for prompt starting with '{str(prompt)[:100]}...': {e}", exc_info=True)
            if 'start_time' in locals() and 'generation_time' not in locals(): # Check if start_time was defined
                 generation_time = time.time() - start_time
            else:
                 generation_time = 0 # If error before start_time

            return {
                'reasoning': "GENERATION_ERROR",
                'code': f"Error during generation: {str(e)}",
                'raw_output': f"Exception: {str(e)}",
                'generation_time': generation_time,
                'step': step,
                'model_input_prompt': prompt,
                'error': str(e)
            }
        finally:
            # Restore model's original training mode
            model.train(original_training_mode)

    def _save_generation_sample(self, step, sample_idx, original_sample, generated_result, simulation_result: Optional[Dict[str, Any]] = None):
        """保存生成样本到文件，包含模拟结果"""
        if not self.samples_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id_slug = str(original_sample.get('task_id', f"sample_{sample_idx}")).replace('/', '_')

        filename = f"step_{step}_{task_id_slug}_{timestamp}.json"
        filepath = os.path.join(self.samples_dir, filename)

        orig_prompt_for_log = original_sample.get('original_prompt_for_debug', original_sample.get('original_enhanced_prompt', 'N/A'))

        sample_data_to_save = {
            "step": step,
            "timestamp_iso": datetime.now().isoformat(),
            "dataset_original_sample_info": {
                "level": original_sample.get('level'),
                "complexity_score": original_sample.get('complexity_score'),
                "original_problem_desc_for_debug": orig_prompt_for_log[:300],
                "testbench_path": original_sample.get('testbench_path', ''),
                "task_id": original_sample.get('task_id')
            },
            "model_input_prompt_preview": generated_result.get('model_input_prompt', '')[:300] + "...",
            "generated_result": {
                'reasoning': generated_result.get('reasoning'),
                'code': generated_result.get('code'),
                'raw_output_preview': generated_result.get('raw_output', '')[:300] + "...",
                'generation_time_seconds': generated_result.get('generation_time')
            },
            "simulation_details": simulation_result if simulation_result else "Not run or not applicable"
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data_to_save, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved generation sample to {filepath}")
        except Exception as e_save:
            logger.error(f"Failed to save generation sample to {filepath}: {e_save}", exc_info=True)


class Qwen3InferenceCallback(BaseCallback):
    def __init__(self, tokenizer, eval_dataset=None, num_samples=2, eval_every_n_steps=25,
                 max_new_tokens=512, max_seq_length=4096, output_dir=None):
        super().__init__(output_dir=output_dir)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.eval_every_n_steps = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length

        if self.output_dir:
            self.samples_dir = os.path.join(self.output_dir, "qwen3_generated_samples")
            os.makedirs(self.samples_dir, exist_ok=True)
        else:
            self.samples_dir = None
            logger.warning("Qwen3InferenceCallback: output_dir not provided, cannot save samples.")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        # Content of on_step_end from utils.Qwen3InferenceCallback
        # Uses parse_llm_completion_qwen3
        pass

    def _generate_qwen3_sample(self, model, prompt, step) -> Dict[str, Any]:
        # Content of _generate_qwen3_sample from utils.Qwen3InferenceCallback
        # Uses parse_llm_completion_qwen3
        return {"error": "Not implemented in stub"} # Placeholder

    def _save_qwen3_sample(self, step, sample_idx, original_sample, generated_result):
        # Content of _save_qwen3_sample from utils.Qwen3InferenceCallback
        pass
