
# 临时课程配置检查代码
# 添加到main.py中的课程管理器初始化之后

if hasattr(curriculum_manager, 'curriculum_stages') and curriculum_manager.curriculum_stages:
    foundation_stage = curriculum_manager.curriculum_stages[0]
    print(f"🔍 当前foundation阶段配置检查:")
    print(f"  - 性能阈值: {foundation_stage.performance_threshold}")
    print(f"  - 最小评估次数: {foundation_stage.min_evaluations}")
    
    # 强制修改foundation阶段配置
    if foundation_stage.performance_threshold > 0.65:
        print(f"⚠️ 强制修改foundation阶段阈值: {foundation_stage.performance_threshold} -> 0.65")
        foundation_stage.performance_threshold = 0.65
    
    if foundation_stage.min_evaluations > 5:
        print(f"⚠️ 强制修改foundation阶段最小评估: {foundation_stage.min_evaluations} -> 5")
        foundation_stage.min_evaluations = 5
