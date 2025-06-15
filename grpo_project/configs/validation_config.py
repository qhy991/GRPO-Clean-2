"""
验证配置文件
控制Verilog代码验证的各种参数
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationConfig:
    """Verilog代码验证配置"""
    
    # 模块名验证设置
    strict_module_name: bool = False  # 默认使用灵活模式
    allow_module_name_variations: bool = True  # 允许模块名变体
    
    # 端口验证设置
    strict_port_order: bool = False  # 是否严格要求端口顺序
    allow_extra_ports: bool = True   # 是否允许额外的端口
    
    # 代码结构验证设置
    require_endmodule: bool = True   # 是否要求endmodule
    allow_comments: bool = True      # 是否允许注释
    
    # 质量评估设置
    enable_quality_assessment: bool = True  # 是否启用代码质量评估
    quality_threshold: float = 0.3   # 质量评估的最低阈值
    
    # 调试设置
    save_validation_errors: bool = True    # 是否保存验证错误
    verbose_validation: bool = False       # 是否启用详细验证日志
    
    def __post_init__(self):
        """配置验证"""
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")
    
    @classmethod
    def create_strict_config(cls) -> 'ValidationConfig':
        """创建严格验证配置"""
        return cls(
            strict_module_name=True,
            allow_module_name_variations=False,
            strict_port_order=True,
            allow_extra_ports=False,
            verbose_validation=True
        )
    
    @classmethod
    def create_flexible_config(cls) -> 'ValidationConfig':
        """创建灵活验证配置（推荐用于训练）"""
        return cls(
            strict_module_name=False,
            allow_module_name_variations=True,
            strict_port_order=False,
            allow_extra_ports=True,
            verbose_validation=False
        )
    
    @classmethod
    def create_production_config(cls) -> 'ValidationConfig':
        """创建生产环境配置（平衡严格性和灵活性）"""
        return cls(
            strict_module_name=False,
            allow_module_name_variations=True,
            strict_port_order=False,
            allow_extra_ports=False,
            quality_threshold=0.5,
            verbose_validation=True
        )

# 默认配置实例
DEFAULT_VALIDATION_CONFIG = ValidationConfig.create_flexible_config()
STRICT_VALIDATION_CONFIG = ValidationConfig.create_strict_config()
PRODUCTION_VALIDATION_CONFIG = ValidationConfig.create_production_config() 