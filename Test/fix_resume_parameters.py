#!/usr/bin/env python3
"""
fix_resume_parameters.py - 断续训练参数传递修复工具
解决GRPO训练中的参数传递断层问题
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
import traceback
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resume_parameter_fix.log')
    ]
)
logger = logging.getLogger(__name__)

class ResumeParameterFixer:
    """断续训练参数传递修复器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_found = []
        self.fixes_applied = []
        
    def diagnose_resume_issues(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """诊断断续训练中的参数传递问题"""
        logger.info("🔍 开始诊断断续训练参数传递问题...")
        
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "checkpoint_path": checkpoint_path,
            "issues": [],
            "recommendations": [],
            "config_mismatches": [],
            "state_inconsistencies": []
        }
        
        # 1. 检查配置文件一致性
        config_issues = self._check_config_consistency()
        diagnosis["config_mismatches"].extend(config_issues)
        
        # 2. 检查checkpoint状态
        if checkpoint_path and Path(checkpoint_path).exists():
            state_issues = self._check_checkpoint_state(checkpoint_path)
            diagnosis["state_inconsistencies"].extend(state_issues)
        
        # 3. 检查WandB状态同步
        wandb_issues = self._check_wandb_sync_issues()
        diagnosis["issues"].extend(wandb_issues)
        
        # 4. 检查课程学习状态
        curriculum_issues = self._check_curriculum_continuity(checkpoint_path)
        diagnosis["issues"].extend(curriculum_issues)
        
        # 5. 生成修复建议
        recommendations = self._generate_fix_recommendations(diagnosis)
        diagnosis["recommendations"].extend(recommendations)
        
        # 保存诊断结果
        self._save_diagnosis_report(diagnosis)
        
        return diagnosis
    
    def _check_config_consistency(self) -> List[Dict[str, Any]]:
        """检查配置文件一致性"""
        logger.info("⚙️ 检查配置文件一致性...")
        
        issues = []
        
        try:
            # 检查配置模块是否可导入
            sys.path.insert(0, str(self.project_root))
            
            try:
                from grpo_project.configs import EnvConfig, ScriptConfig, EnhancedRewardConfig
                logger.info("✅ 配置模块导入成功")
            except ImportError as e:
                issues.append({
                    "type": "config_import_error",
                    "severity": "high",
                    "message": f"配置模块导入失败: {e}",
                    "fix_suggestion": "检查grpo_project/configs目录和__init__.py文件"
                })
                return issues
            
            # 检查配置类的字段一致性
            try:
                # 创建默认配置实例
                env_cfg = EnvConfig()
                script_cfg = ScriptConfig(model_name_or_path="dummy")
                reward_cfg = EnhancedRewardConfig()
                
                # 检查长度配置一致性
                if hasattr(script_cfg, 'script_max_prompt_length') and hasattr(script_cfg, 'script_max_completion_length'):
                    total_length = script_cfg.script_max_prompt_length + script_cfg.script_max_completion_length
                    if total_length > script_cfg.max_seq_length:
                        issues.append({
                            "type": "length_config_mismatch",
                            "severity": "medium",
                            "message": f"prompt长度({script_cfg.script_max_prompt_length}) + completion长度({script_cfg.script_max_completion_length}) = {total_length} > 最大序列长度({script_cfg.max_seq_length})",
                            "fix_suggestion": "调整长度配置或使用自动长度分配"
                        })
                
                logger.info("✅ 配置类字段检查完成")
                
            except Exception as e:
                issues.append({
                    "type": "config_instantiation_error",
                    "severity": "high",
                    "message": f"配置类实例化失败: {e}",
                    "fix_suggestion": "检查配置类定义和默认值"
                })
        
        except Exception as e:
            issues.append({
                "type": "config_check_error",
                "severity": "high",
                "message": f"配置检查过程出错: {e}",
                "fix_suggestion": "检查项目目录结构和Python环境"
            })
        
        return issues
    
    def _check_checkpoint_state(self, checkpoint_path: str) -> List[Dict[str, Any]]:
        """检查checkpoint状态一致性"""
        logger.info(f"📂 检查checkpoint状态: {checkpoint_path}")
        
        issues = []
        checkpoint_dir = Path(checkpoint_path)
        
        # 检查必要的文件
        required_files = {
            "trainer_state.json": "训练器状态文件",
            "config.json": "模型配置文件",
        }
        
        model_files = ["pytorch_model.bin", "model.safetensors"]
        has_model_file = any((checkpoint_dir / f).exists() for f in model_files)
        
        if not has_model_file:
            issues.append({
                "type": "missing_model_weights",
                "severity": "high",
                "message": "未找到模型权重文件",
                "fix_suggestion": f"确保checkpoint目录包含 {' 或 '.join(model_files)} 中的一个"
            })
        
        for filename, description in required_files.items():
            filepath = checkpoint_dir / filename
            if not filepath.exists():
                issues.append({
                    "type": "missing_checkpoint_file",
                    "severity": "high",
                    "message": f"缺少{description}: {filename}",
                    "fix_suggestion": f"确保checkpoint目录包含完整的{description}"
                })
                continue
            
            # 验证JSON文件格式
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    issues.append({
                        "type": "corrupted_checkpoint_file",
                        "severity": "high",
                        "message": f"{description}损坏: {e}",
                        "fix_suggestion": f"检查{filename}文件格式，可能需要从备份恢复"
                    })
        
        # 检查trainer_state.json中的关键信息
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                
                # 检查必要的状态信息
                required_keys = ["global_step", "epoch", "best_metric", "best_model_checkpoint"]
                for key in required_keys:
                    if key not in trainer_state:
                        issues.append({
                            "type": "incomplete_trainer_state",
                            "severity": "medium",
                            "message": f"trainer_state.json缺少关键字段: {key}",
                            "fix_suggestion": "可能需要重新开始训练或从更早的checkpoint恢复"
                        })
                
                # 检查步数合理性
                global_step = trainer_state.get("global_step", 0)
                if global_step <= 0:
                    issues.append({
                        "type": "invalid_global_step",
                        "severity": "medium",
                        "message": f"无效的global_step: {global_step}",
                        "fix_suggestion": "检查训练是否正常进行"
                    })
                
            except Exception as e:
                issues.append({
                    "type": "trainer_state_read_error",
                    "severity": "high",
                    "message": f"读取trainer_state.json失败: {e}",
                    "fix_suggestion": "检查文件权限和格式"
                })
        
        return issues
    
    def _check_wandb_sync_issues(self) -> List[Dict[str, Any]]:
        """检查WandB同步问题"""
        logger.info("📊 检查WandB同步问题...")
        
        issues = []
        
        # 检查WandB安装
        try:
            import wandb
        except ImportError:
            issues.append({
                "type": "wandb_not_installed",
                "severity": "medium",
                "message": "WandB未安装或导入失败",
                "fix_suggestion": "运行 pip install wandb"
            })
            return issues
        
        # 检查WandB认证
        try:
            api_key_file = Path.home() / ".netrc"
            if api_key_file.exists():
                with open(api_key_file, 'r') as f:
                    content = f.read()
                    if "api.wandb.ai" not in content:
                        issues.append({
                            "type": "wandb_not_authenticated",
                            "severity": "medium",
                            "message": "WandB可能未认证",
                            "fix_suggestion": "运行 wandb login"
                        })
            else:
                issues.append({
                    "type": "wandb_no_credentials",
                    "severity": "medium",
                    "message": "未找到WandB认证信息",
                    "fix_suggestion": "运行 wandb login"
                })
        except Exception as e:
            logger.warning(f"检查WandB认证时出错: {e}")
        
        # 检查wandb目录状态
        wandb_dir = self.project_root / "wandb"
        if wandb_dir.exists():
            # 检查是否有run目录
            run_dirs = list(wandb_dir.glob("run-*"))
            if not run_dirs:
                issues.append({
                    "type": "no_wandb_runs",
                    "severity": "low",
                    "message": "wandb目录中未找到run记录",
                    "fix_suggestion": "这可能是新项目，无需修复"
                })
            else:
                logger.info(f"找到 {len(run_dirs)} 个WandB run目录")
        
        return issues
    
    def _check_curriculum_continuity(self, checkpoint_path: Optional[str]) -> List[Dict[str, Any]]:
        """检查课程学习连续性"""
        logger.info("📚 检查课程学习状态连续性...")
        
        issues = []
        
        # 检查课程管理器模块
        try:
            sys.path.insert(0, str(self.project_root))
            from grpo_project.curriculum.manager import setup_fixed_curriculum_manager
            logger.info("✅ 课程管理器模块导入成功")
        except ImportError as e:
            issues.append({
                "type": "curriculum_module_missing",
                "severity": "high",
                "message": f"课程管理器模块导入失败: {e}",
                "fix_suggestion": "检查grpo_project/curriculum/manager.py文件"
            })
            return issues
        
        # 检查课程状态文件
        curriculum_state_file = self.project_root / "curriculum_state.json"
        if curriculum_state_file.exists():
            try:
                with open(curriculum_state_file, 'r') as f:
                    json.load(f)
                logger.info("✅ 课程状态文件格式正确")
            except json.JSONDecodeError as e:
                issues.append({
                    "type": "corrupted_curriculum_state",
                    "severity": "medium",
                    "message": f"课程状态文件损坏: {e}",
                    "fix_suggestion": "删除损坏的课程状态文件，将从头开始课程学习"
                })
        
        return issues
    
    def _generate_fix_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 统计问题严重程度
        high_severity_count = sum(1 for issue in diagnosis["issues"] + diagnosis["config_mismatches"] + diagnosis["state_inconsistencies"] if issue.get("severity") == "high")
        medium_severity_count = sum(1 for issue in diagnosis["issues"] + diagnosis["config_mismatches"] + diagnosis["state_inconsistencies"] if issue.get("severity") == "medium")
        
        if high_severity_count > 0:
            recommendations.append(f"⚠️ 发现 {high_severity_count} 个高严重性问题，建议修复后再继续训练")
        
        if medium_severity_count > 0:
            recommendations.append(f"⚠️ 发现 {medium_severity_count} 个中等严重性问题，建议检查并修复")
        
        # 基于问题类型生成具体建议
        issue_types = set()
        for issue_list in [diagnosis["issues"], diagnosis["config_mismatches"], diagnosis["state_inconsistencies"]]:
            for issue in issue_list:
                issue_types.add(issue.get("type", "unknown"))
        
        if "config_import_error" in issue_types:
            recommendations.append("📁 检查项目目录结构，确保grpo_project包完整")
        
        if "missing_model_weights" in issue_types:
            recommendations.append("📦 验证checkpoint目录包含完整的模型权重文件")
        
        if "wandb_not_authenticated" in issue_types:
            recommendations.append("🔐 配置WandB认证: wandb login")
        
        if "length_config_mismatch" in issue_types:
            recommendations.append("📏 调整长度配置，确保prompt+completion不超过最大序列长度")
        
        if not recommendations:
            recommendations.append("✅ 未发现严重问题，可以安全地进行断续训练")
        
        return recommendations
    
    def _save_diagnosis_report(self, diagnosis: Dict[str, Any]):
        """保存诊断报告"""
        report_file = self.project_root / f"resume_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(diagnosis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 诊断报告已保存: {report_file}")
            
            # 同时生成易读的文本报告
            text_report_file = report_file.with_suffix('.txt')
            self._generate_text_report(diagnosis, text_report_file)
            
        except Exception as e:
            logger.error(f"保存诊断报告失败: {e}")
    
    def _generate_text_report(self, diagnosis: Dict[str, Any], output_file: Path):
        """生成易读的文本报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== GRPO断续训练参数传递诊断报告 ===\n")
                f.write(f"生成时间: {diagnosis['timestamp']}\n")
                f.write(f"项目根目录: {diagnosis['project_root']}\n")
                f.write(f"Checkpoint路径: {diagnosis['checkpoint_path'] or '未指定'}\n\n")
                
                # 问题统计
                all_issues = diagnosis["issues"] + diagnosis["config_mismatches"] + diagnosis["state_inconsistencies"]
                if all_issues:
                    f.write(f"发现问题总数: {len(all_issues)}\n")
                    
                    # 按严重程度分类
                    high_issues = [i for i in all_issues if i.get("severity") == "high"]
                    medium_issues = [i for i in all_issues if i.get("severity") == "medium"]
                    low_issues = [i for i in all_issues if i.get("severity") == "low"]
                    
                    f.write(f"  - 高严重性: {len(high_issues)}\n")
                    f.write(f"  - 中等严重性: {len(medium_issues)}\n")
                    f.write(f"  - 低严重性: {len(low_issues)}\n\n")
                    
                    # 详细问题列表
                    if high_issues:
                        f.write("🚨 高严重性问题:\n")
                        for i, issue in enumerate(high_issues, 1):
                            f.write(f"  {i}. {issue['message']}\n")
                            f.write(f"     修复建议: {issue['fix_suggestion']}\n")
                        f.write("\n")
                    
                    if medium_issues:
                        f.write("⚠️ 中等严重性问题:\n")
                        for i, issue in enumerate(medium_issues, 1):
                            f.write(f"  {i}. {issue['message']}\n")
                            f.write(f"     修复建议: {issue['fix_suggestion']}\n")
                        f.write("\n")
                else:
                    f.write("✅ 未发现严重问题\n\n")
                
                # 修复建议
                if diagnosis["recommendations"]:
                    f.write("💡 修复建议:\n")
                    for i, rec in enumerate(diagnosis["recommendations"], 1):
                        f.write(f"  {i}. {rec}\n")
                    f.write("\n")
                
                f.write("=== 报告结束 ===\n")
            
            logger.info(f"📋 文本报告已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"生成文本报告失败: {e}")
    
    def apply_automatic_fixes(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """应用自动修复"""
        logger.info("🔧 开始应用自动修复...")
        
        fix_results = {
            "applied_fixes": [],
            "failed_fixes": [],
            "manual_fixes_needed": []
        }
        
        all_issues = diagnosis["issues"] + diagnosis["config_mismatches"] + diagnosis["state_inconsistencies"]
        
        for issue in all_issues:
            issue_type = issue.get("type", "unknown")
            
            try:
                if issue_type == "corrupted_curriculum_state":
                    # 删除损坏的课程状态文件
                    curriculum_state_file = self.project_root / "curriculum_state.json"
                    if curriculum_state_file.exists():
                        backup_name = f"curriculum_state.json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        curriculum_state_file.rename(self.project_root / backup_name)
                        fix_results["applied_fixes"].append(f"已备份并删除损坏的课程状态文件: {backup_name}")
                
                elif issue_type == "length_config_mismatch":
                    # 这个需要手动修复，因为涉及配置策略选择
                    fix_results["manual_fixes_needed"].append("需要手动调整长度配置策略")
                
                elif issue_type in ["wandb_not_authenticated", "wandb_no_credentials"]:
                    fix_results["manual_fixes_needed"].append("需要手动运行: wandb login")
                
                elif issue_type in ["missing_model_weights", "missing_checkpoint_file"]:
                    fix_results["manual_fixes_needed"].append("需要手动验证checkpoint文件完整性")
                
                else:
                    fix_results["manual_fixes_needed"].append(f"需要手动处理: {issue['message']}")
                    
            except Exception as e:
                fix_results["failed_fixes"].append(f"修复失败 ({issue_type}): {e}")
        
        return fix_results


def main():
    parser = argparse.ArgumentParser(description="断续训练参数传递修复工具")
    parser.add_argument("--project-root", default=".", help="项目根目录路径")
    parser.add_argument("--checkpoint", help="要检查的checkpoint路径")
    parser.add_argument("--auto-fix", action="store_true", help="自动应用可修复的问题")
    parser.add_argument("--report-only", action="store_true", help="仅生成诊断报告")
    
    args = parser.parse_args()
    
    try:
        fixer = ResumeParameterFixer(args.project_root)
        
        # 执行诊断
        diagnosis = fixer.diagnose_resume_issues(args.checkpoint)
        
        # 显示诊断结果摘要
        all_issues = diagnosis["issues"] + diagnosis["config_mismatches"] + diagnosis["state_inconsistencies"]
        if all_issues:
            print(f"\n🔍 诊断完成，发现 {len(all_issues)} 个问题:")
            for issue in all_issues:
                severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(issue.get("severity"), "❓")
                print(f"  {severity_icon} {issue['message']}")
        else:
            print("\n✅ 诊断完成，未发现问题")
        
        # 显示建议
        if diagnosis["recommendations"]:
            print("\n💡 修复建议:")
            for rec in diagnosis["recommendations"]:
                print(f"  {rec}")
        
        # 应用自动修复
        if args.auto_fix and not args.report_only:
            fix_results = fixer.apply_automatic_fixes(diagnosis)
            
            if fix_results["applied_fixes"]:
                print("\n🔧 已应用的自动修复:")
                for fix in fix_results["applied_fixes"]:
                    print(f"  ✅ {fix}")
            
            if fix_results["manual_fixes_needed"]:
                print("\n📝 需要手动处理的问题:")
                for fix in fix_results["manual_fixes_needed"]:
                    print(f"  📌 {fix}")
            
            if fix_results["failed_fixes"]:
                print("\n❌ 修复失败的问题:")
                for fix in fix_results["failed_fixes"]:
                    print(f"  ❌ {fix}")
        
        print(f"\n📋 详细报告已保存到项目目录")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 