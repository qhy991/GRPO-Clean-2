# test_curriculum_debug.py - 测试课程学习调试日志
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('.')

# 设置基础日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_curriculum_debug():
    """测试课程学习调试功能"""
    try:
        from grpo_project.curriculum.manager import FixedEnhancedCurriculumManager
        from grpo_project.curriculum.stages import CurriculumStageConfig
        from grpo_project.curriculum.callbacks import CurriculumProgressCallback, EnhancedCurriculumDebugCallback
        from datasets import Dataset
        
        print("✅ 成功导入课程学习模块")
        
        # 创建测试数据集
        test_data = []
        for i in range(1000):
            level_choice = ['basic', 'intermediate', 'advanced', 'expert'][i % 4]
            complexity = (i % 10) + 1  # 1-10的复杂度
            
            test_data.append({
                'prompt': f'Test prompt {i}',
                'level': level_choice,
                'complexity_score': complexity,
                'testbench_path': f'/path/to/tb_{i}.v',
                'expected_total_tests': 5,
                'reference_verilog_path': f'/path/to/ref_{i}.v'
            })
        
        dataset = Dataset.from_list(test_data)
        print(f"✅ 创建测试数据集: {len(dataset)} 样本")
        
        # 创建课程阶段
        stages = [
            CurriculumStageConfig(
                name="foundation",
                dataset_levels=["basic"],
                complexity_range=(0.0, 3.5),
                epochs_ratio=0.25,
                performance_threshold=0.65,
                min_evaluations=3,
                description="基础阶段测试"
            ),
            CurriculumStageConfig(
                name="elementary", 
                dataset_levels=["basic", "intermediate"],
                complexity_range=(0.0, 5.5),
                epochs_ratio=0.25,
                performance_threshold=0.6,
                min_evaluations=3,
                description="初级阶段测试"
            ),
            CurriculumStageConfig(
                name="intermediate",
                dataset_levels=["intermediate"],
                complexity_range=(2.0, 7.5),
                epochs_ratio=0.25,
                performance_threshold=0.55,
                min_evaluations=4,
                description="中级阶段测试"
            )
        ]
        
        print(f"✅ 创建课程阶段: {len(stages)} 个阶段")
        
        # 创建课程管理器
        curriculum_manager = FixedEnhancedCurriculumManager(stages, dataset)
        print("✅ 创建课程管理器")
        
        # 测试输出目录
        test_output_dir = "./test_curriculum_debug_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # 创建回调
        progress_callback = CurriculumProgressCallback(
            curriculum_manager=curriculum_manager,
            trainer_ref=None,
            output_dir=test_output_dir
        )
        
        debug_callback = EnhancedCurriculumDebugCallback(
            curriculum_manager=curriculum_manager,
            trainer_ref=None,
            output_dir=test_output_dir
        )
        
        print("✅ 创建课程学习回调")
        
        # 模拟训练过程
        print("\n🚀 开始模拟训练过程...")
        
        # 模拟初始状态
        curriculum_manager.force_debug_output()
        
        # 模拟几次评估和可能的进阶
        for step in range(1, 101):
            # 模拟性能提升
            performance = 0.3 + (step * 0.005) + (0.1 * (step // 20))
            
            # 每10步检查一次进阶
            if step % 10 == 0:
                print(f"\n--- 步数 {step}: 性能检查 ---")
                print(f"当前性能: {performance:.4f}")
                
                should_advance = curriculum_manager.should_advance_stage(performance)
                print(f"是否应该进阶: {should_advance}")
                
                if should_advance:
                    success = curriculum_manager.advance_stage()
                    print(f"进阶结果: {'成功' if success else '失败'}")
                    
                    if success:
                        current_info = curriculum_manager.get_current_stage_info()
                        print(f"新阶段: {current_info['stage_name']}")
                        print(f"新数据集大小: {current_info['dataset_size']}")
            
            # 每20步记录定期状态
            if step % 20 == 0:
                curriculum_manager.log_periodic_status(step)
                
            # 每50步保存详细日志
            if step % 50 == 0:
                curriculum_manager.save_detailed_log(test_output_dir)
        
        # 最终状态
        print("\n📊 最终状态:")
        final_info = curriculum_manager.get_current_stage_info()
        print(f"最终阶段: {final_info['stage_index']} ({final_info['stage_name']})")
        print(f"进阶统计: {final_info['advancement_stats']}")
        print(f"调试日志条数: {len(curriculum_manager.debug_log)}")
        
        # 保存最终日志
        curriculum_manager.save_detailed_log(test_output_dir)
        
        # 检查日志文件
        log_files = [
            "curriculum_progress_debug.txt",
            "enhanced_curriculum_debug_log.txt", 
            "curriculum_detailed_debug.json",
            "curriculum_debug_text.log"
        ]
        
        print(f"\n📁 检查输出文件 (目录: {test_output_dir}):")
        for log_file in log_files:
            file_path = os.path.join(test_output_dir, log_file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {log_file}: {size} bytes")
            else:
                print(f"❌ {log_file}: 不存在")
        
        print(f"\n🎉 测试完成! 检查输出目录: {test_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 开始课程学习调试日志测试...")
    success = test_curriculum_debug()
    
    if success:
        print("✅ 所有测试通过!")
        print("\n📝 运行建议:")
        print("1. 检查生成的日志文件内容")
        print("2. 在实际训练中使用相同的回调设置")
        print("3. 确保在 main.py 中正确导入和使用这些回调")
    else:
        print("❌ 测试失败，请检查错误信息")
    
    sys.exit(0 if success else 1)