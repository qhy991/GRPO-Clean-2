#!/usr/bin/env python3
"""
quick_resume_check.py - 快速断续训练参数检查工具
在开始断续训练前进行快速验证
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def check_project_structure() -> Tuple[bool, List[str]]:
    """检查项目结构"""
    issues = []
    
    # 检查关键文件
    required_files = [
        "main.py",
        "run_enhanced_grpo_training.sh",
        "grpo_project/__init__.py",
        "grpo_project/configs/__init__.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"❌ 缺少关键文件: {file_path}")
    
    return len(issues) == 0, issues

def check_python_imports() -> Tuple[bool, List[str]]:
    """检查Python模块导入"""
    issues = []
    
    # 检查必要的包
    required_packages = [
        "torch", "transformers", "trl", "datasets", 
        "wandb", "numpy", "peft"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"❌ 缺少Python包: {package}")
    
    # 检查项目模块
    try:
        sys.path.insert(0, ".")
        from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
        from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
    except ImportError as e:
        issues.append(f"❌ 项目模块导入失败: {e}")
    
    return len(issues) == 0, issues

def check_checkpoint_state(checkpoint_path: str) -> Tuple[bool, List[str]]:
    """检查checkpoint状态"""
    issues = []
    
    if not checkpoint_path:
        return True, ["ℹ️ 新训练，无需检查checkpoint"]
    
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        issues.append(f"❌ Checkpoint目录不存在: {checkpoint_path}")
        return False, issues
    
    # 检查必要文件
    required_files = {
        "trainer_state.json": "训练状态",
        "config.json": "模型配置"
    }
    
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_file = any((checkpoint_dir / f).exists() for f in model_files)
    
    if not has_model_file:
        issues.append(f"❌ 未找到模型权重文件: {model_files}")
    
    for filename, description in required_files.items():
        filepath = checkpoint_dir / filename
        if not filepath.exists():
            issues.append(f"❌ 缺少{description}文件: {filename}")
            continue
        
        # 验证JSON格式
        if filename.endswith('.json'):
            try:
                with open(filepath, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                issues.append(f"❌ {description}文件损坏: {filename}")
    
    return len(issues) == 0, issues

def check_config_consistency() -> Tuple[bool, List[str]]:
    """检查配置一致性"""
    issues = []
    
    try:
        sys.path.insert(0, ".")
        from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
        
        # 创建默认配置
        script_cfg = ScriptConfig(model_name_or_path="dummy")
        
        # 检查长度配置
        total_length = script_cfg.script_max_prompt_length + script_cfg.script_max_completion_length
        if total_length > script_cfg.max_seq_length:
            issues.append(
                f"⚠️ 长度配置不匹配: "
                f"prompt({script_cfg.script_max_prompt_length}) + "
                f"completion({script_cfg.script_max_completion_length}) = "
                f"{total_length} > max_seq_length({script_cfg.max_seq_length})"
            )
        
    except Exception as e:
        issues.append(f"❌ 配置检查失败: {e}")
    
    return len(issues) == 0, issues

def check_wandb_setup() -> Tuple[bool, List[str]]:
    """检查WandB设置"""
    issues = []
    
    try:
        import wandb
    except ImportError:
        issues.append("❌ WandB未安装")
        return False, issues
    
    # 检查认证
    netrc_file = Path.home() / ".netrc"
    if netrc_file.exists():
        try:
            with open(netrc_file, 'r') as f:
                content = f.read()
                if "api.wandb.ai" not in content:
                    issues.append("⚠️ WandB可能未认证")
        except:
            pass
    else:
        issues.append("⚠️ 未找到WandB认证信息")
    
    return len(issues) == 0, issues

def extract_checkpoint_from_script() -> Optional[str]:
    """从训练脚本提取checkpoint路径"""
    script_path = Path("run_enhanced_grpo_training.sh")
    if not script_path.exists():
        return None
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 查找RESUME_FROM_CHECKPOINT_DIR
        for line in content.split('\n'):
            if 'RESUME_FROM_CHECKPOINT_DIR=' in line and not line.strip().startswith('#'):
                # 提取路径
                parts = line.split('=', 1)
                if len(parts) == 2:
                    path = parts[1].strip().strip('"').strip("'")
                    if path and path != '""' and path != "''":
                        return path
        
        return None
    except:
        return None

def print_summary(all_results: Dict[str, Tuple[bool, List[str]]]):
    """打印检查结果摘要"""
    print("\n" + "="*60)
    print("🔍 GRPO断续训练快速检查结果")
    print("="*60)
    
    total_issues = 0
    critical_issues = 0
    
    for check_name, (passed, issues) in all_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"\n📋 {check_name}: {status}")
        
        if issues:
            for issue in issues:
                print(f"   {issue}")
                total_issues += 1
                if issue.startswith("❌"):
                    critical_issues += 1
    
    print(f"\n📊 检查统计:")
    print(f"   总检查项目: {len(all_results)}")
    print(f"   通过项目: {sum(1 for passed, _ in all_results.values() if passed)}")
    print(f"   发现问题: {total_issues}")
    print(f"   严重问题: {critical_issues}")
    
    if critical_issues == 0:
        print(f"\n🎉 所有关键检查通过！可以安全地进行断续训练。")
    elif critical_issues <= 2:
        print(f"\n⚠️ 发现少量严重问题，建议修复后再继续训练。")
    else:
        print(f"\n🚨 发现多个严重问题，强烈建议修复后再继续训练。")
    
    print("\n💡 修复建议:")
    if critical_issues > 0:
        print("   1. 运行完整诊断: ./cleanup_before_training.sh")
        print("   2. 检查缺失的文件和包")
        print("   3. 验证checkpoint完整性")
    else:
        print("   1. 检查警告信息并根据需要调整")
        print("   2. 确认训练配置参数")
    
    print("   3. 开始训练: ./run_enhanced_grpo_training.sh")

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 开始GRPO断续训练快速检查...")
    
    # 获取checkpoint路径
    checkpoint_path = extract_checkpoint_from_script()
    if checkpoint_path:
        print(f"📂 检测到checkpoint: {checkpoint_path}")
    else:
        print("🆕 未设置checkpoint，将开始新训练")
    
    # 执行各项检查
    all_results = {}
    
    print("\n🔍 检查项目结构...")
    all_results["项目结构"] = check_project_structure()
    
    print("🐍 检查Python环境...")
    all_results["Python环境"] = check_python_imports()
    
    print("⚙️ 检查配置一致性...")
    all_results["配置一致性"] = check_config_consistency()
    
    if checkpoint_path:
        print("📂 检查Checkpoint状态...")
        all_results["Checkpoint状态"] = check_checkpoint_state(checkpoint_path)
    
    print("📊 检查WandB设置...")
    all_results["WandB设置"] = check_wandb_setup()
    
    # 打印结果
    print_summary(all_results)

if __name__ == "__main__":
    main() 