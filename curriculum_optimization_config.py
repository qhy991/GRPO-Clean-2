#!/usr/bin/env python3
"""
课程学习优化配置
解决性能波动大、进阶不稳定的问题
"""

import os
import logging

logger = logging.getLogger(__name__)

class CurriculumOptimizationConfig:
    """课程学习优化配置类"""
    
    # 🔧 优化后的阈值策略：递增式阈值
    PROGRESSIVE_THRESHOLDS = {
        # 策略1: 温和递增 (推荐用于稳定训练)
        "gentle": [0.60, 0.65, 0.70, 0.75, 0.80],
        
        # 策略2: 标准递增 (平衡效果和稳定性)  
        "standard": [0.65, 0.70, 0.75, 0.80, 0.85],
        
        # 策略3: 激进递增 (追求高性能)
        "aggressive": [0.70, 0.75, 0.80, 0.85, 0.90],
        
        # 策略4: 自适应 (根据当前性能动态调整)
        "adaptive": [0.60, 0.67, 0.74, 0.81, 0.88]
    }
    
    # 🔧 优化的最小评估次数设置
    MIN_EVALUATIONS_CONFIG = {
        # 每个阶段的最小评估次数
        "conservative": [15, 12, 10, 8, 6],  # 确保充分验证
        "balanced": [10, 10, 8, 6, 5],       # 平衡效率和稳定性
        "aggressive": [8, 6, 5, 4, 3],      # 快速进阶
    }
    
    # 🔧 循环训练的阈值递增策略
    ROUND_INCREMENT_STRATEGIES = {
        "linear": 0.05,      # 每轮+0.05 (线性增长)
        "exponential": 0.02, # 每轮*1.02 (指数增长) 
        "adaptive": 0.03,    # 根据历史性能调整
    }

    @staticmethod
    def apply_optimization_strategy(strategy_name: str = "standard"):
        """应用优化策略到环境变量"""
        
        if strategy_name not in CurriculumOptimizationConfig.PROGRESSIVE_THRESHOLDS:
            logger.warning(f"未知策略 {strategy_name}，使用默认 'standard'")
            strategy_name = "standard"
        
        thresholds = CurriculumOptimizationConfig.PROGRESSIVE_THRESHOLDS[strategy_name]
        min_evals = CurriculumOptimizationConfig.MIN_EVALUATIONS_CONFIG["balanced"]
        
        # 设置环境变量
        for i, threshold in enumerate(thresholds, 1):
            os.environ[f"CURRICULUM_PERFORMANCE_THRESHOLD_{i}"] = str(threshold)
            
        for i, min_eval in enumerate(min_evals):
            os.environ[f"CURRICULUM_MIN_EVALUATIONS_{i}"] = str(min_eval)
            
        # 设置循环训练参数
        os.environ["CURRICULUM_MAX_ROUNDS"] = "3"
        os.environ["CURRICULUM_THRESHOLD_INCREMENT"] = "0.03"
        
        logger.info(f"✅ 应用课程优化策略: {strategy_name}")
        logger.info(f"📈 性能阈值: {thresholds}")
        logger.info(f"📊 最小评估: {min_evals}")
        
        return thresholds, min_evals

    @staticmethod 
    def create_adaptive_thresholds(baseline_performance: float = 0.7):
        """根据基线性能创建自适应阈值"""
        # 基于当前模型性能水平设置合理的递增阈值
        if baseline_performance < 0.5:
            return [0.45, 0.50, 0.55, 0.60, 0.65]
        elif baseline_performance < 0.7:
            return [0.60, 0.65, 0.70, 0.75, 0.80] 
        else:
            return [0.70, 0.75, 0.80, 0.85, 0.90]

def apply_quick_fix():
    """快速修复当前训练的阈值问题"""
    print("🔧 正在应用课程学习优化配置...")
    
    # 应用标准策略
    thresholds, min_evals = CurriculumOptimizationConfig.apply_optimization_strategy("standard")
    
    print("📊 优化配置已应用:")
    print(f"   阈值策略: 递增式 {thresholds}")
    print(f"   评估次数: {min_evals}")
    print(f"   循环轮次: 3轮")
    print(f"   阈值递增: 每轮+0.03")
    
    # 创建配置文件供脚本使用
    with open("curriculum_override.env", "w") as f:
        f.write("# 课程学习优化配置\n")
        for i, threshold in enumerate(thresholds, 1):
            f.write(f"export CURRICULUM_PERFORMANCE_THRESHOLD_{i}={threshold}\n")
        for i, min_eval in enumerate(min_evals):
            f.write(f"export CURRICULUM_MIN_EVALUATIONS_{i}={min_eval}\n")
        f.write("export CURRICULUM_MAX_ROUNDS=3\n")
        f.write("export CURRICULUM_THRESHOLD_INCREMENT=0.03\n")
    
    print("📝 配置文件已保存到: curriculum_override.env")
    print("📌 使用方法: source curriculum_override.env && python main.py")

if __name__ == "__main__":
    apply_quick_fix() 