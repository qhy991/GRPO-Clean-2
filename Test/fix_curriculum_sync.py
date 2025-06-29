#!/usr/bin/env python3
"""
fix_curriculum_sync.py - 课程学习状态同步修复工具
专门解决断续训练时课程学习状态传递断层问题
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CurriculumStateFixer:
    """课程学习状态修复器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.curriculum_state_file = self.project_root / "curriculum_state.json"
        
    def diagnose_curriculum_issues(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """诊断课程学习状态问题"""
        logger.info("📚 开始诊断课程学习状态问题...")
        
        issues = {
            "state_file_issues": [],
            "manager_issues": [],
            "checkpoint_sync_issues": [],
            "callback_issues": []
        }
        
        # 1. 检查状态文件
        self._check_state_file(issues)
        
        # 2. 检查管理器模块
        self._check_manager_module(issues)
        
        # 3. 检查与checkpoint的同步
        if checkpoint_path:
            self._check_checkpoint_sync(checkpoint_path, issues)
        
        # 4. 检查回调模块
        self._check_callback_modules(issues)
        
        return issues
    
    def _check_state_file(self, issues: Dict[str, List]):
        """检查课程状态文件"""
        logger.info("📄 检查课程状态文件...")
        
        if not self.curriculum_state_file.exists():
            logger.info("课程状态文件不存在（正常，可能是新训练）")
            return
        
        try:
            with open(self.curriculum_state_file, 'r') as f:
                state_data = json.load(f)
            
            # 检查必要的字段
            required_fields = ["current_stage", "stage_count", "stages", "performance_history"]
            for field in required_fields:
                if field not in state_data:
                    issues["state_file_issues"].append({
                        "type": "missing_field",
                        "message": f"状态文件缺少字段: {field}",
                        "severity": "medium"
                    })
            
            # 检查阶段信息完整性
            if "stages" in state_data and isinstance(state_data["stages"], list):
                for i, stage in enumerate(state_data["stages"]):
                    if not isinstance(stage, dict):
                        issues["state_file_issues"].append({
                            "type": "invalid_stage_format",
                            "message": f"阶段{i}格式无效",
                            "severity": "high"
                        })
                        continue
                    
                    stage_required_fields = ["name", "dataset_levels", "complexity_range", "performance_threshold"]
                    for field in stage_required_fields:
                        if field not in stage:
                            issues["state_file_issues"].append({
                                "type": "incomplete_stage_config",
                                "message": f"阶段{i}缺少配置字段: {field}",
                                "severity": "medium"
                            })
            
            logger.info("✅ 课程状态文件格式检查完成")
            
        except json.JSONDecodeError as e:
            issues["state_file_issues"].append({
                "type": "corrupted_state_file",
                "message": f"状态文件损坏: {e}",
                "severity": "high"
            })
        except Exception as e:
            issues["state_file_issues"].append({
                "type": "state_file_read_error",
                "message": f"读取状态文件失败: {e}",
                "severity": "high"
            })
    
    def _check_manager_module(self, issues: Dict[str, List]):
        """检查课程管理器模块"""
        logger.info("🎯 检查课程管理器模块...")
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            # 检查管理器导入
            try:
                from grpo_project.curriculum.manager import setup_fixed_curriculum_manager, FixedEnhancedCurriculumManager
                logger.info("✅ 课程管理器模块导入成功")
            except ImportError as e:
                issues["manager_issues"].append({
                    "type": "manager_import_error",
                    "message": f"课程管理器导入失败: {e}",
                    "severity": "high"
                })
                return
            
            # 检查管理器类的方法
            manager_methods = [
                "get_current_stage_dataset",
                "should_advance_stage", 
                "advance_to_next_stage",
                "save_state",
                "load_state"
            ]
            
            for method in manager_methods:
                if not hasattr(FixedEnhancedCurriculumManager, method):
                    issues["manager_issues"].append({
                        "type": "missing_manager_method",
                        "message": f"课程管理器缺少方法: {method}",
                        "severity": "medium"
                    })
            
        except Exception as e:
            issues["manager_issues"].append({
                "type": "manager_check_error",
                "message": f"检查课程管理器时出错: {e}",
                "severity": "high"
            })
    
    def _check_checkpoint_sync(self, checkpoint_path: str, issues: Dict[str, List]):
        """检查与checkpoint的同步问题"""
        logger.info(f"🔄 检查与checkpoint的同步: {checkpoint_path}")
        
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            issues["checkpoint_sync_issues"].append({
                "type": "checkpoint_not_found",
                "message": f"指定的checkpoint不存在: {checkpoint_path}",
                "severity": "high"
            })
            return
        
        # 检查trainer_state.json中的课程相关信息
        trainer_state_file = checkpoint_dir / "trainer_state.json"
        if trainer_state_file.exists():
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                
                # 检查是否有课程学习相关的日志
                log_history = trainer_state.get("log_history", [])
                has_curriculum_logs = any(
                    any("curriculum" in str(key).lower() or "stage" in str(key).lower() 
                        for key in entry.keys() if isinstance(entry, dict))
                    for entry in log_history
                )
                
                if not has_curriculum_logs:
                    issues["checkpoint_sync_issues"].append({
                        "type": "no_curriculum_logs",
                        "message": "trainer_state.json中未找到课程学习日志",
                        "severity": "low"
                    })
                
                # 检查global_step与课程状态的一致性
                global_step = trainer_state.get("global_step", 0)
                if self.curriculum_state_file.exists():
                    try:
                        with open(self.curriculum_state_file, 'r') as f:
                            curriculum_state = json.load(f)
                        
                        # 简单的一致性检查
                        curriculum_step = curriculum_state.get("last_update_step", 0)
                        if abs(global_step - curriculum_step) > 100:  # 允许一定误差
                            issues["checkpoint_sync_issues"].append({
                                "type": "step_mismatch",
                                "message": f"课程状态步数({curriculum_step})与训练步数({global_step})相差较大",
                                "severity": "medium"
                            })
                    except:
                        pass  # 课程状态文件问题已在其他地方检查
                
            except Exception as e:
                issues["checkpoint_sync_issues"].append({
                    "type": "trainer_state_read_error",
                    "message": f"读取trainer_state.json失败: {e}",
                    "severity": "medium"
                })
    
    def _check_callback_modules(self, issues: Dict[str, List]):
        """检查课程学习回调模块"""
        logger.info("📞 检查课程学习回调模块...")
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            # 检查回调导入
            try:
                from grpo_project.curriculum.callbacks import (
                    CurriculumProgressCallback,
                    EnhancedCurriculumDebugCallback,
                    OptimizedCurriculumCallback
                )
                logger.info("✅ 课程学习回调模块导入成功")
            except ImportError as e:
                issues["callback_issues"].append({
                    "type": "callback_import_error",
                    "message": f"课程回调导入失败: {e}",
                    "severity": "high"
                })
                return
            
            # 检查回调类的必要方法
            callback_methods = [
                "on_train_begin",
                "on_step_end", 
                "on_evaluate",
                "on_log"
            ]
            
            for callback_class in [CurriculumProgressCallback, EnhancedCurriculumDebugCallback]:
                for method in callback_methods:
                    if not hasattr(callback_class, method):
                        issues["callback_issues"].append({
                            "type": "missing_callback_method",
                            "message": f"回调类{callback_class.__name__}缺少方法: {method}",
                            "severity": "medium"
                        })
            
        except Exception as e:
            issues["callback_issues"].append({
                "type": "callback_check_error",
                "message": f"检查回调模块时出错: {e}",
                "severity": "high"
            })
    
    def fix_curriculum_state(self, issues: Dict[str, Any], auto_fix: bool = False) -> Dict[str, Any]:
        """修复课程学习状态问题"""
        logger.info("🔧 开始修复课程学习状态问题...")
        
        fix_results = {
            "applied_fixes": [],
            "failed_fixes": [],
            "manual_fixes_needed": []
        }
        
        # 处理状态文件问题
        for issue in issues.get("state_file_issues", []):
            issue_type = issue.get("type")
            
            if issue_type == "corrupted_state_file" and auto_fix:
                try:
                    # 备份损坏的文件
                    backup_name = f"curriculum_state.json.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.curriculum_state_file.rename(self.project_root / backup_name)
                    fix_results["applied_fixes"].append(f"已备份损坏的状态文件: {backup_name}")
                except Exception as e:
                    fix_results["failed_fixes"].append(f"备份损坏状态文件失败: {e}")
            
            elif issue_type in ["missing_field", "incomplete_stage_config"]:
                fix_results["manual_fixes_needed"].append(f"需要手动修复状态文件: {issue['message']}")
        
        # 处理管理器问题
        for issue in issues.get("manager_issues", []):
            fix_results["manual_fixes_needed"].append(f"需要手动检查课程管理器: {issue['message']}")
        
        # 处理同步问题
        for issue in issues.get("checkpoint_sync_issues", []):
            if issue.get("type") == "step_mismatch":
                fix_results["manual_fixes_needed"].append("建议重新初始化课程状态以匹配checkpoint步数")
            else:
                fix_results["manual_fixes_needed"].append(f"同步问题需手动处理: {issue['message']}")
        
        return fix_results
    
    def create_fresh_curriculum_state(self, checkpoint_path: Optional[str] = None) -> bool:
        """创建新的课程状态文件"""
        logger.info("🆕 创建新的课程状态文件...")
        
        try:
            # 从checkpoint获取当前步数
            current_step = 0
            if checkpoint_path:
                checkpoint_dir = Path(checkpoint_path)
                trainer_state_file = checkpoint_dir / "trainer_state.json"
                if trainer_state_file.exists():
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)
                        current_step = trainer_state.get("global_step", 0)
            
            # 创建默认的课程状态
            fresh_state = {
                "current_stage": 0,
                "stage_count": 4,  # 默认4个阶段
                "last_update_step": current_step,
                "performance_history": [],
                "stage_advancement_log": [],
                "stages": [
                    {
                        "name": "Basic",
                        "dataset_levels": ["basic"],
                        "complexity_range": [1, 3],
                        "performance_threshold": 0.5,
                        "min_evaluations": 5
                    },
                    {
                        "name": "Intermediate", 
                        "dataset_levels": ["basic", "intermediate"],
                        "complexity_range": [2, 4],
                        "performance_threshold": 0.6,
                        "min_evaluations": 5
                    },
                    {
                        "name": "Advanced",
                        "dataset_levels": ["intermediate", "advanced"],
                        "complexity_range": [3, 5],
                        "performance_threshold": 0.65,
                        "min_evaluations": 5
                    },
                    {
                        "name": "Expert",
                        "dataset_levels": ["advanced", "expert"],
                        "complexity_range": [4, 6],
                        "performance_threshold": 0.7,
                        "min_evaluations": 5
                    }
                ],
                "created_at": datetime.now().isoformat(),
                "sync_info": {
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_step": current_step
                }
            }
            
            # 备份现有文件（如果存在）
            if self.curriculum_state_file.exists():
                backup_name = f"curriculum_state.json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.curriculum_state_file.rename(self.project_root / backup_name)
                logger.info(f"已备份现有状态文件: {backup_name}")
            
            # 写入新状态
            with open(self.curriculum_state_file, 'w', encoding='utf-8') as f:
                json.dump(fresh_state, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 新的课程状态文件已创建: {self.curriculum_state_file}")
            logger.info(f"   - 起始步数: {current_step}")
            logger.info(f"   - 课程阶段数: {fresh_state['stage_count']}")
            
            return True
            
        except Exception as e:
            logger.error(f"创建新状态文件失败: {e}")
            return False
    
    def generate_sync_report(self, issues: Dict[str, Any]) -> str:
        """生成同步问题报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.project_root / f"curriculum_sync_report_{timestamp}.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== 课程学习状态同步诊断报告 ===\n")
                f.write(f"生成时间: {datetime.now()}\n")
                f.write(f"项目根目录: {self.project_root}\n\n")
                
                # 统计问题
                total_issues = sum(len(issue_list) for issue_list in issues.values())
                f.write(f"发现问题总数: {total_issues}\n\n")
                
                # 各类问题详情
                for category, issue_list in issues.items():
                    if issue_list:
                        f.write(f"=== {category.replace('_', ' ').title()} ===\n")
                        for i, issue in enumerate(issue_list, 1):
                            severity = issue.get("severity", "unknown")
                            severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(severity, "❓")
                            f.write(f"{i}. {severity_icon} {issue['message']}\n")
                        f.write("\n")
                
                if total_issues == 0:
                    f.write("✅ 未发现课程学习状态同步问题\n")
                
                f.write("=== 报告结束 ===\n")
            
            logger.info(f"📋 同步报告已保存: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return ""


def main():
    parser = argparse.ArgumentParser(description="课程学习状态同步修复工具")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--checkpoint", help="checkpoint路径")
    parser.add_argument("--diagnose", action="store_true", help="仅诊断问题")
    parser.add_argument("--auto-fix", action="store_true", help="自动修复问题")
    parser.add_argument("--create-fresh", action="store_true", help="创建新的状态文件")
    
    args = parser.parse_args()
    
    try:
        fixer = CurriculumStateFixer(args.project_root)
        
        if args.create_fresh:
            # 创建新的状态文件
            success = fixer.create_fresh_curriculum_state(args.checkpoint)
            if success:
                print("✅ 新的课程状态文件创建成功")
            else:
                print("❌ 创建新状态文件失败")
                sys.exit(1)
        else:
            # 诊断问题
            issues = fixer.diagnose_curriculum_issues(args.checkpoint)
            
            # 显示问题统计
            total_issues = sum(len(issue_list) for issue_list in issues.values())
            if total_issues > 0:
                print(f"\n🔍 发现 {total_issues} 个课程学习状态问题:")
                
                for category, issue_list in issues.items():
                    if issue_list:
                        print(f"\n{category.replace('_', ' ').title()}:")
                        for issue in issue_list:
                            severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(issue.get("severity"), "❓")
                            print(f"  {severity_icon} {issue['message']}")
            else:
                print("\n✅ 未发现课程学习状态问题")
            
            # 应用修复
            if args.auto_fix and total_issues > 0:
                fix_results = fixer.fix_curriculum_state(issues, auto_fix=True)
                
                if fix_results["applied_fixes"]:
                    print("\n🔧 已应用的修复:")
                    for fix in fix_results["applied_fixes"]:
                        print(f"  ✅ {fix}")
                
                if fix_results["manual_fixes_needed"]:
                    print("\n📝 需要手动处理的问题:")
                    for fix in fix_results["manual_fixes_needed"]:
                        print(f"  📌 {fix}")
            
            # 生成报告
            report_file = fixer.generate_sync_report(issues)
            if report_file:
                print(f"\n📋 详细报告已保存: {report_file}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 