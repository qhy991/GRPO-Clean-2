"""
Hard-case监控回调类
在训练过程中定期运行Hard-case测试并记录结果
"""

import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from grpo_project.monitoring import HardCaseMonitor

logger = logging.getLogger(__name__)

class HardCaseMonitoringCallback(TrainerCallback):
    """Hard-case测试监控回调"""
    
    def __init__(self, 
                 hard_case_monitor: HardCaseMonitor,
                 reward_calculator,
                 tokenizer,
                 monitor_interval: int = 100,
                 reference_verilog_dir: Optional[str] = None):
        """
        初始化Hard-case监控回调
        
        Args:
            hard_case_monitor: Hard-case监控器实例
            reward_calculator: 奖励计算器实例
            tokenizer: tokenizer实例
            monitor_interval: 监控间隔（训练步数）
            reference_verilog_dir: 参考Verilog文件目录
        """
        self.hard_case_monitor = hard_case_monitor
        self.reward_calculator = reward_calculator
        self.tokenizer = tokenizer
        self.monitor_interval = monitor_interval
        self.reference_verilog_dir = reference_verilog_dir
        
        self.last_monitored_step = -1
        
        logger.info(f"HardCaseMonitoringCallback初始化完成，监控间隔: {monitor_interval} 步")
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在每个训练步骤结束时检查是否需要进行监控"""
        
        current_step = state.global_step
        
        # 检查是否需要进行监控
        if self._should_monitor(current_step):
            try:
                logger.info(f"🔍 开始Hard-case监控 - 训练步骤: {current_step}")
                
                # 运行监控
                results = self.hard_case_monitor.run_monitoring(
                    reward_calculator=self.reward_calculator,
                    tokenizer=self.tokenizer,
                    training_step=current_step,
                    reference_verilog_dir=self.reference_verilog_dir
                )
                
                # 记录结果到日志
                self._log_monitoring_results(results, current_step)
                
                # 记录到WandB（如果可用）
                self._log_to_wandb(results, current_step, kwargs.get('logs', {}))
                
                self.last_monitored_step = current_step
                
            except Exception as e:
                logger.error(f"❌ Hard-case监控失败 (步骤 {current_step}): {e}")
    
    def _should_monitor(self, current_step: int) -> bool:
        """判断是否应该在当前步骤进行监控"""
        
        # 避免重复监控同一步骤
        if current_step == self.last_monitored_step:
            return False
        
        # 按间隔监控 (确保step 0不触发)
        if current_step > 0 and current_step % self.monitor_interval == 0:
            return True
        
        return False
    
    def _log_monitoring_results(self, results: Dict[str, Any], current_step: int):
        """记录监控结果到日志"""
        if not results:
            logger.warning(f"步骤 {current_step}: 没有获得监控结果")
            return
        
        logger.info(f"📊 Hard-case监控结果 (步骤 {current_step}):")
        
        total_reward = 0
        total_tokens = 0
        success_count = 0
        
        for case_name, result in results.items():
            logger.info(f"  - {case_name}:")
            logger.info(f"    奖励: {result.final_reward:.3f}")
            logger.info(f"    令牌数: {result.token_count}")
            logger.info(f"    编译成功: {result.compilation_success}")
            logger.info(f"    仿真成功: {result.simulation_success}")
            logger.info(f"    测试通过率: {result.test_pass_rate:.1%}")
            
            total_reward += result.final_reward
            total_tokens += result.token_count
            
            if result.simulation_success and result.test_pass_rate > 0.5:
                success_count += 1
        
        # 汇总统计
        avg_reward = total_reward / len(results) if results else 0
        avg_tokens = total_tokens / len(results) if results else 0
        success_rate = success_count / len(results) if results else 0
        
        logger.info(f"📈 汇总统计:")
        logger.info(f"  - 平均奖励: {avg_reward:.3f}")
        logger.info(f"  - 平均令牌数: {avg_tokens:.0f}")
        logger.info(f"  - 成功率: {success_rate:.1%} ({success_count}/{len(results)})")
    
    def _log_to_wandb(self, results: Dict[str, Any], current_step: int, training_logs: Dict[str, Any]):
        """记录结果到WandB"""
        try:
            import wandb
            
            if not wandb.run:
                return
            
            # 准备WandB日志数据
            wandb_logs = {}
            
            if results:
                # 按案例记录详细指标
                for case_name, result in results.items():
                    prefix = f"hard_case/{case_name}"
                    wandb_logs.update({
                        f"{prefix}/reward": result.final_reward,
                        f"{prefix}/token_count": result.token_count,
                        f"{prefix}/compilation_success": 1.0 if result.compilation_success else 0.0,
                        f"{prefix}/simulation_success": 1.0 if result.simulation_success else 0.0,
                        f"{prefix}/test_pass_rate": result.test_pass_rate,
                        f"{prefix}/elapsed_time": result.elapsed_time
                    })
                
                # 汇总指标
                total_reward = sum(r.final_reward for r in results.values())
                total_tokens = sum(r.token_count for r in results.values())
                success_count = sum(1 for r in results.values() 
                                  if r.simulation_success and r.test_pass_rate > 0.5)
                
                wandb_logs.update({
                    "hard_case/avg_reward": total_reward / len(results),
                    "hard_case/avg_token_count": total_tokens / len(results),
                    "hard_case/success_rate": success_count / len(results),
                    "hard_case/total_cases": len(results)
                })
            
            # 记录到WandB
            wandb.log(wandb_logs)
            
        except Exception as e:
            logger.debug(f"WandB日志记录失败: {e}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时的最终监控"""
        try:
            logger.info("🏁 训练结束，进行最终Hard-case监控...")
            
            # 最终监控
            final_results = self.hard_case_monitor.run_monitoring(
                reward_calculator=self.reward_calculator,
                tokenizer=self.tokenizer,
                training_step=state.global_step,
                reference_verilog_dir=self.reference_verilog_dir
            )
            
            # 记录最终结果
            self._log_monitoring_results(final_results, state.global_step)
            
            # 导出监控结果
            try:
                export_file = self.hard_case_monitor.export_results()
                if export_file:
                    logger.info(f"📄 Hard-case监控结果已导出到: {export_file}")
            except Exception as e:
                logger.warning(f"导出监控结果失败: {e}")
            
            # 获取监控汇总
            summary = self.hard_case_monitor.get_monitoring_summary()
            if summary and 'latest_results' in summary:
                logger.info("📊 最终Hard-case监控汇总:")
                for case_name, case_summary in summary['latest_results'].items():
                    logger.info(f"  - {case_name}: 奖励={case_summary['final_reward']:.3f}, "
                              f"令牌={case_summary['token_count']}, "
                              f"通过率={case_summary['test_pass_rate']:.1%}")
                              
        except Exception as e:
            logger.error(f"最终Hard-case监控失败: {e}") 