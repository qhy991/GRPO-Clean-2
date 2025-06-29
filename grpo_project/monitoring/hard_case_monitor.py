#!/usr/bin/env python3
"""
Hard-case测试案例监控系统
用于在训练过程中跟踪模型在特定困难案例上的表现
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class HardCaseResult:
    """单个Hard-case测试结果"""
    case_name: str
    training_step: int
    timestamp: float
    prompt: str
    completion: str
    rewards: Dict[str, float]  # 各组件奖励分数
    final_reward: float
    token_count: int
    simulation_success: bool
    compilation_success: bool
    test_pass_rate: float
    elapsed_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式便于序列化"""
        return asdict(self)

class HardCaseMonitor:
    """Hard-case测试案例监控器"""
    
    def __init__(self, hard_case_dir: str, output_dir: str = "hard_case_logs"):
        """
        初始化监控器
        
        Args:
            hard_case_dir: Hard-case测试案例目录
            output_dir: 监控结果输出目录
        """
        self.hard_case_dir = Path(hard_case_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载测试案例
        self.test_cases = self._load_test_cases()
        self.results_history: List[HardCaseResult] = []
        
        # 创建日志文件
        self.log_file = self.output_dir / "hard_case_monitor.jsonl"
        self.summary_file = self.output_dir / "hard_case_summary.json"
        
        logger.info(f"HardCaseMonitor initialized with {len(self.test_cases)} test cases")
        logger.info(f"Test cases: {list(self.test_cases.keys())}")
    
    def _load_test_cases(self) -> Dict[str, Dict[str, str]]:
        """加载Hard-case测试案例"""
        test_cases = {}
        
        if not self.hard_case_dir.exists():
            logger.warning(f"Hard-case directory not found: {self.hard_case_dir}")
            return test_cases
        
        for case_dir in self.hard_case_dir.iterdir():
            if case_dir.is_dir():
                case_name = case_dir.name
                prompt_file = case_dir / f"{case_name}.txt"
                testbench_file = case_dir / "testbench.v"
                
                if prompt_file.exists() and testbench_file.exists():
                    try:
                        with open(prompt_file, 'r', encoding='utf-8') as f:
                            prompt = f.read().strip()
                        
                        test_cases[case_name] = {
                            'prompt': prompt,
                            'testbench_path': str(testbench_file),
                            'case_dir': str(case_dir)
                        }
                        
                        logger.info(f"Loaded test case: {case_name}")
                    except Exception as e:
                        logger.error(f"Failed to load test case {case_name}: {e}")
                else:
                    logger.warning(f"Missing files for test case {case_name}")
        
        return test_cases
    
    def should_monitor(self, training_step: int, monitor_interval: int = 100) -> bool:
        """
        判断是否应该在当前步骤进行监控
        
        Args:
            training_step: 当前训练步骤
            monitor_interval: 监控间隔
            
        Returns:
            是否应该监控
        """
        return training_step % monitor_interval == 0 or training_step <= 10
    
    def run_monitoring(self, 
                      reward_calculator,
                      tokenizer,
                      training_step: int,
                      reference_verilog_dir: Optional[str] = None) -> Dict[str, HardCaseResult]:
        """
        运行Hard-case监控
        
        Args:
            reward_calculator: 奖励计算器实例
            tokenizer: tokenizer实例
            training_step: 当前训练步骤
            reference_verilog_dir: 参考Verilog文件目录
            
        Returns:
            测试结果字典
        """
        if not self.test_cases:
            logger.warning("No test cases available for monitoring")
            return {}
        
        logger.info(f"Running Hard-case monitoring at step {training_step}")
        results = {}
        
        for case_name, case_info in self.test_cases.items():
            try:
                start_time = time.time()
                result = self._run_single_case(
                    case_name=case_name,
                    case_info=case_info,
                    reward_calculator=reward_calculator,
                    tokenizer=tokenizer,
                    training_step=training_step,
                    reference_verilog_dir=reference_verilog_dir
                )
                
                result.elapsed_time = time.time() - start_time
                results[case_name] = result
                
                # 记录结果
                self._log_result(result)
                logger.info(f"Completed monitoring for {case_name}: reward={result.final_reward:.3f}, tokens={result.token_count}")
                
            except Exception as e:
                logger.error(f"Failed to monitor case {case_name}: {e}")
                continue
        
        # 更新汇总统计
        self._update_summary(results, training_step)
        
        return results
    
    def _run_single_case(self,
                        case_name: str,
                        case_info: Dict[str, str],
                        reward_calculator,
                        tokenizer,
                        training_step: int,
                        reference_verilog_dir: Optional[str] = None) -> HardCaseResult:
        """运行单个测试案例"""
        
        prompt = case_info['prompt']
        testbench_path = case_info['testbench_path']
        
        # 使用奖励计算器生成模型输出（这里需要模型生成，暂时用空字符串模拟）
        # 在实际集成时，这里应该调用模型生成代码
        completion = ""  # 这里应该是模型的实际输出
        
        # 查找参考Verilog文件
        reference_verilog_path = None
        if reference_verilog_dir:
            ref_file = Path(reference_verilog_dir) / f"{case_name}.v"
            if ref_file.exists():
                reference_verilog_path = str(ref_file)
        
        # 计算奖励和指标
        try:
            # 假设有3个测试用例（根据testbench分析得出）
            expected_total_tests = self._count_tests_in_testbench(testbench_path)
            
            reward_result = reward_calculator._calculate_single_reward(
                prompt_str=prompt,
                completion_str=completion,
                testbench_path=testbench_path,
                expected_total_tests=expected_total_tests,
                reference_verilog_path=reference_verilog_path,
                training_step=training_step,
                completion_idx=0,
                tokenizer=tokenizer
            )
            
            # 解析奖励结果
            rewards = reward_result['unscaled_components']
            final_reward = reward_result['final_reward']
            token_count = reward_result.get('funnel_metrics', {}).get('output_token_count', 0)
            
            # 解析仿真结果
            metrics = reward_result.get('funnel_metrics', {})
            simulation_success = metrics.get('simulation_successful', False)
            compilation_success = metrics.get('compilation_successful', False)
            test_pass_rate = metrics.get('test_pass_rate', 0.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate reward for {case_name}: {e}")
            # 使用默认值
            rewards = {}
            final_reward = -10.0
            token_count = 0
            simulation_success = False
            compilation_success = False
            test_pass_rate = 0.0
        
        return HardCaseResult(
            case_name=case_name,
            training_step=training_step,
            timestamp=time.time(),
            prompt=prompt,
            completion=completion,
            rewards=rewards,
            final_reward=final_reward,
            token_count=token_count,
            simulation_success=simulation_success,
            compilation_success=compilation_success,
            test_pass_rate=test_pass_rate,
            elapsed_time=0.0  # 会在外部设置
        )
    
    def _count_tests_in_testbench(self, testbench_path: str) -> int:
        """分析testbench文件中的测试用例数量"""
        try:
            with open(testbench_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的测试用例计数（基于注释或assert语句）
            test_count = 0
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip().lower()
                if ('test' in line and ('case' in line or 'assert' in line)) or \
                   line.startswith('assert') or \
                   ('$display' in line and 'test' in line):
                    test_count += 1
            
            return max(1, test_count)  # 至少1个测试
            
        except Exception as e:
            logger.warning(f"Failed to count tests in {testbench_path}: {e}")
            return 3  # 默认值
    
    def _log_result(self, result: HardCaseResult):
        """记录单个测试结果到日志文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
    
    def _update_summary(self, results: Dict[str, HardCaseResult], training_step: int):
        """更新汇总统计信息"""
        try:
            # 计算汇总统计
            summary = {
                'last_update_step': training_step,
                'last_update_time': time.time(),
                'total_cases': len(self.test_cases),
                'case_names': list(self.test_cases.keys()),
                'latest_results': {}
            }
            
            for case_name, result in results.items():
                summary['latest_results'][case_name] = {
                    'final_reward': result.final_reward,
                    'token_count': result.token_count,
                    'simulation_success': result.simulation_success,
                    'compilation_success': result.compilation_success,
                    'test_pass_rate': result.test_pass_rate,
                    'timestamp': result.timestamp
                }
            
            # 计算历史趋势（如果有历史数据）
            if self.results_history:
                summary['trends'] = self._calculate_trends()
            
            # 保存汇总文件
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            # 更新历史记录
            self.results_history.extend(results.values())
            
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """计算性能趋势"""
        trends = {}
        
        # 按案例分组计算趋势
        for case_name in self.test_cases.keys():
            case_results = [r for r in self.results_history if r.case_name == case_name]
            
            if len(case_results) >= 2:
                # 按训练步骤排序
                case_results.sort(key=lambda x: x.training_step)
                
                # 计算奖励趋势
                rewards = [r.final_reward for r in case_results[-10:]]  # 最近10次
                if len(rewards) >= 2:
                    trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]
                    trends[case_name] = {
                        'reward_trend': float(trend_slope),
                        'recent_avg_reward': float(np.mean(rewards)),
                        'best_reward': float(max(r.final_reward for r in case_results)),
                        'latest_reward': case_results[-1].final_reward,
                        'total_evaluations': len(case_results)
                    }
        
        return trends
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控汇总信息"""
        try:
            if self.summary_file.exists():
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {'message': 'No monitoring data available yet'}
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {'error': str(e)}
    
    def export_results(self, output_file: str = None) -> str:
        """导出监控结果为CSV格式"""
        import pandas as pd
        
        if not self.results_history:
            logger.warning("No results to export")
            return ""
        
        # 转换为DataFrame
        data = []
        for result in self.results_history:
            row = {
                'case_name': result.case_name,
                'training_step': result.training_step,
                'final_reward': result.final_reward,
                'token_count': result.token_count,
                'simulation_success': result.simulation_success,
                'compilation_success': result.compilation_success,
                'test_pass_rate': result.test_pass_rate,
                'elapsed_time': result.elapsed_time,
                'timestamp': result.timestamp
            }
            
            # 添加各组件奖励
            for component, value in result.rewards.items():
                row[f'reward_{component}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if output_file is None:
            output_file = str(self.output_dir / "hard_case_results.csv")
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results exported to {output_file}")
        
        return output_file 