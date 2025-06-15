"""
WandB同步管理器
解决断续训练时的步数同步问题
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings

logger = logging.getLogger(__name__)

class WandBSyncManager:
    """WandB步数同步管理器"""
    
    def __init__(self, output_dir: str, project_name: str, run_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.run_name = run_name
        self.wandb_run = None
        self.wandb_initialized = False
        self.local_backup_file = self.output_dir / "wandb_local_backup.jsonl"
        self.step_offset = 0  # 用于修正步数偏移
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_wandb_run(self, resume_from_checkpoint: Optional[str] = None, 
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """设置WandB运行，处理断续训练"""
        try:
            # 🔧 修复WandB导入问题
            try:
                import wandb
                # 检查WandB是否正确安装
                if not hasattr(wandb, 'init'):
                    raise ImportError("WandB安装不完整，缺少init函数")
            except ImportError as e:
                logger.warning(f"⚠️ WandB导入失败: {e}")
                logger.info("💡 请安装或更新WandB: pip install wandb")
                self.wandb_initialized = False
                return False
            
            is_resuming = resume_from_checkpoint and os.path.exists(resume_from_checkpoint)
            
            # 准备WandB配置
            wandb_config = config or {}
            wandb_config.update({
                "output_dir": str(self.output_dir),
                "is_resuming": is_resuming,
                "checkpoint_path": resume_from_checkpoint if is_resuming else None,
            })
            
            # 设置初始化参数
            init_kwargs = {
                "project": self.project_name,
                "config": wandb_config,
                "save_code": True,
                "tags": ["grpo", "verilog", "sync_fixed"],
            }
            
            if self.run_name:
                init_kwargs["name"] = self.run_name
            
            # 处理断续训练
            if is_resuming:
                # 尝试从环境变量获取run ID
                env_run_id = os.environ.get("WANDB_RUN_ID")
                env_resume = os.environ.get("WANDB_RESUME", "allow")
                
                if env_run_id:
                    init_kwargs["id"] = env_run_id
                    init_kwargs["resume"] = env_resume
                    logger.info(f"🔄 使用环境变量恢复WandB run: {env_run_id}, 模式: {env_resume}")
                else:
                    # 尝试从checkpoint提取
                    extracted_run_id = self._extract_run_id_from_checkpoint(resume_from_checkpoint)
                    if extracted_run_id:
                        init_kwargs["id"] = extracted_run_id
                        init_kwargs["resume"] = "must"
                        logger.info(f"🔄 从checkpoint提取WandB run ID: {extracted_run_id}")
                    else:
                        # 创建新的run，但标记为继续训练
                        init_kwargs["resume"] = "allow"
                        logger.warning("⚠️ 无法找到原始run ID，创建新的WandB run")
            
            # 初始化WandB
            self.wandb_run = wandb.init(**init_kwargs)
            self.wandb_initialized = True
            
            # 检测步数偏移
            if is_resuming:
                self._detect_step_offset()
            
            logger.info(f"✅ WandB初始化成功: {self.wandb_run.url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ WandB初始化失败: {e}", exc_info=True)
            self.wandb_initialized = False
            return False
    
    def _extract_run_id_from_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """从checkpoint目录提取WandB run ID"""
        try:
            checkpoint_dir = Path(checkpoint_path)
            
            # 方法1: 查找wandb目录
            possible_wandb_dirs = [
                checkpoint_dir.parent / "wandb",
                checkpoint_dir / "wandb"
            ]
            
            for wandb_dir in possible_wandb_dirs:
                if wandb_dir.exists():
                    run_dirs = list(wandb_dir.glob("run-*"))
                    if run_dirs:
                        latest_run = sorted(run_dirs)[-1]
                        run_name = latest_run.name
                        if "-" in run_name:
                            parts = run_name.split("-")
                            if len(parts) >= 3:
                                return parts[-1]
            
            # 方法2: 从trainer_state.json查找
            trainer_state = checkpoint_dir / "trainer_state.json"
            if trainer_state.exists():
                with open(trainer_state, 'r') as f:
                    state_data = json.load(f)
                    # 可能在state中有wandb信息
                    if "wandb" in state_data:
                        return state_data["wandb"].get("run_id")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ 从checkpoint提取run ID失败: {e}")
            return None
    
    def _detect_step_offset(self):
        """检测步数偏移"""
        try:
            if not self.wandb_run:
                return
                
            # 获取WandB当前步数（通过查看run的历史记录）
            wandb_step = 0
            try:
                # 尝试从WandB run获取最新的step
                if hasattr(self.wandb_run, 'summary') and self.wandb_run.summary:
                    wandb_step = self.wandb_run.summary.get('_step', 0)
                elif hasattr(self.wandb_run, 'step'):
                    wandb_step = self.wandb_run.step
            except Exception:
                wandb_step = 0
            
            # 尝试从checkpoint获取trainer步数
            trainer_step = 0
            try:
                checkpoint_dirs = list(self.output_dir.glob("checkpoint-*"))
                if checkpoint_dirs:
                    # 获取最新的checkpoint
                    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.name.split('-')[1]))[-1]
                    trainer_state_file = latest_checkpoint / "trainer_state.json"
                    if trainer_state_file.exists():
                        with open(trainer_state_file, 'r') as f:
                            state_data = json.load(f)
                            trainer_step = state_data.get('global_step', 0)
            except Exception as e:
                logger.debug(f"从checkpoint获取trainer步数失败: {e}")
            
            # 计算偏移
            self.step_offset = wandb_step - trainer_step
            
            if self.step_offset > 0:
                logger.warning(f"⚠️ 检测到步数偏移: WandB={wandb_step}, Trainer={trainer_step}, 偏移={self.step_offset}")
                logger.warning(f"⚠️ 这通常发生在断续训练时，WandB将自动修正步数")
            else:
                logger.info(f"✅ 步数同步正常: WandB={wandb_step}, Trainer={trainer_step}")
                
        except Exception as e:
            logger.warning(f"⚠️ 步数偏移检测失败: {e}")
            self.step_offset = 0
    
    def update_step_offset(self, trainer_step: int):
        """动态更新步数偏移（当已知trainer的实际步数时）"""
        try:
            if not self.wandb_run:
                return
                
            # 获取WandB当前步数
            wandb_step = 0
            try:
                if hasattr(self.wandb_run, 'summary') and self.wandb_run.summary:
                    wandb_step = self.wandb_run.summary.get('_step', 0)
                elif hasattr(self.wandb_run, 'step'):
                    wandb_step = self.wandb_run.step
            except Exception:
                wandb_step = 0
            
            old_offset = self.step_offset
            self.step_offset = wandb_step - trainer_step
            
            if self.step_offset != old_offset:
                logger.info(f"🔄 步数偏移已更新: {old_offset} -> {self.step_offset} (WandB={wandb_step}, Trainer={trainer_step})")
                
        except Exception as e:
            logger.warning(f"⚠️ 步数偏移更新失败: {e}")

    def safe_log(self, data: Dict[str, Any], step: Optional[int] = None, 
                commit: bool = True) -> bool:
        """安全的日志记录，处理步数同步"""
        try:
            # 本地备份
            self._backup_locally(data, step)
            
            if not self.wandb_initialized or not self.wandb_run:
                logger.debug("WandB未初始化，仅保存本地备份")
                return False
            
            # 修正步数（但不允许步数倒退）
            corrected_step = step
            if step is not None:
                if self.step_offset > 0:
                    # 向前修正
                    corrected_step = step + self.step_offset
                    logger.debug(f"步数修正: {step} -> {corrected_step}")
                elif self.step_offset < 0:
                    # 不允许倒退，使用原始步数
                    corrected_step = step
                    logger.debug(f"步数保持: {step} (偏移={self.step_offset}，但不倒退)")
            
            # 记录到WandB
            self.wandb_run.log(data, step=corrected_step, commit=commit)
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ WandB记录失败: {e}")
            return False
    
    def _backup_locally(self, data: Dict[str, Any], step: Optional[int] = None):
        """本地备份日志数据"""
        try:
            backup_entry = {
                "timestamp": str(Path(__file__).stat().st_mtime),  # 简单时间戳
                "step": step,
                "data": data
            }
            
            with open(self.local_backup_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(backup_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.debug(f"本地备份失败: {e}")
    
    def finish(self):
        """结束WandB运行"""
        if self.wandb_initialized and self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("✅ WandB运行已结束")
            except Exception as e:
                logger.warning(f"⚠️ WandB结束失败: {e}")


# 全局实例
_global_sync_manager: Optional[WandBSyncManager] = None

def initialize_wandb_sync_manager(output_dir: str, project_name: str, 
                                run_name: Optional[str] = None) -> WandBSyncManager:
    """初始化全局WandB同步管理器"""
    global _global_sync_manager
    _global_sync_manager = WandBSyncManager(output_dir, project_name, run_name)
    return _global_sync_manager

def get_wandb_sync_manager() -> Optional[WandBSyncManager]:
    """获取全局WandB同步管理器"""
    return _global_sync_manager

def safe_wandb_log(data: Dict[str, Any], step: Optional[int] = None, 
                  commit: bool = True) -> bool:
    """全局安全WandB记录函数"""
    sync_manager = get_wandb_sync_manager()
    if sync_manager:
        return sync_manager.safe_log(data, step, commit)
    else:
        logger.warning("⚠️ WandB同步管理器未初始化")
        return False

def update_wandb_step_offset(trainer_step: int):
    """全局步数偏移更新函数"""
    sync_manager = get_wandb_sync_manager()
    if sync_manager:
        sync_manager.update_step_offset(trainer_step)
    else:
        logger.debug("WandB同步管理器未初始化，跳过步数偏移更新") 