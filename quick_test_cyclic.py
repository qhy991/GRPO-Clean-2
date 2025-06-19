#!/usr/bin/env python3
"""
快速测试循环课程学习的核心功能
"""

class MockStage:
    def __init__(self, name, threshold, min_evals=2):
        self.name = name
        self.performance_threshold = threshold
        self.min_evaluations = min_evals
        self.dataset_levels = ['test']
        self.complexity_range = (0, 10)
        self.epochs_ratio = 0.3

class MockCurriculumManager:
    def __init__(self):
        # 模拟3个阶段
        self.curriculum_stages = [
            MockStage("stage_0", 0.6),
            MockStage("stage_1", 0.7), 
            MockStage("stage_2", 0.8)
        ]
        
        # 循环训练相关变量
        self.current_stage = 0
        self.current_round = 1
        self.max_rounds = 3
        self.threshold_increment = 0.1
        self.stage_performance_history = []
        self.completed_rounds = 0
        self.round_history = []
        self.all_stage_history = []
        
    def get_current_threshold(self, stage_index=None):
        """获取当前轮次的有效性能阈值"""
        if stage_index is None:
            stage_index = self.current_stage
            
        if stage_index >= len(self.curriculum_stages):
            return 0.9
            
        base_threshold = self.curriculum_stages[stage_index].performance_threshold
        current_threshold = base_threshold + (self.current_round - 1) * self.threshold_increment
        return min(current_threshold, 0.95)
    
    def should_continue_curriculum(self):
        """判断是否应该继续课程学习"""
        return self.current_round <= self.max_rounds
    
    def start_new_round(self):
        """开始新一轮训练"""
        self.completed_rounds += 1
        
        # 记录轮次历史
        round_summary = {
            'round_number': self.current_round,
            'completed_stages': len(self.all_stage_history)
        }
        self.round_history.append(round_summary)
        
        # 开始新轮次
        self.current_round += 1
        self.current_stage = 0
        self.stage_performance_history = []
        
        print(f"🔄 完成第{self.completed_rounds}轮，开始第{self.current_round}轮")
        print(f"📈 新轮次阈值提升: +{(self.current_round - 1) * self.threshold_increment:.2f}")
    
    def should_advance_stage(self, performance):
        """判断是否应该进入下一阶段"""
        self.stage_performance_history.append(performance)
        
        if len(self.stage_performance_history) < self.curriculum_stages[self.current_stage].min_evaluations:
            return False
            
        current_threshold = self.get_current_threshold()
        recent_avg = sum(self.stage_performance_history[-2:]) / min(2, len(self.stage_performance_history))
        
        return recent_avg >= current_threshold
    
    def advance_stage(self):
        """进入下一阶段或新轮次"""
        old_stage = self.current_stage
        
        # 记录阶段历史
        final_stats = {
            'stage': old_stage,
            'round': self.current_round,
            'final_performance': self.stage_performance_history[-1] if self.stage_performance_history else 0
        }
        self.all_stage_history.append(final_stats)
        
        # 检查是否在最后阶段
        if self.current_stage >= len(self.curriculum_stages) - 1:
            if self.should_continue_curriculum():
                # 开始新轮次
                print(f"🔄 完成轮次{self.current_round}最后阶段，开始新轮次")
                self.start_new_round()
                return True
            else:
                print(f"🏁 所有{self.max_rounds}轮训练完成")
                return False
        else:
            # 正常阶段进阶
            self.current_stage += 1
            self.stage_performance_history = []
            print(f"🎉 轮次{self.current_round}: 阶段{old_stage} -> 阶段{self.current_stage}")
            return True

def test_cyclic_logic():
    """测试循环逻辑"""
    print("🧪 测试循环课程学习逻辑")
    print("=" * 50)
    
    manager = MockCurriculumManager()
    
    # 显示初始状态
    print(f"📊 初始状态: 轮次{manager.current_round}, 阶段{manager.current_stage}")
    print(f"🔄 最大轮次: {manager.max_rounds}, 阈值递增: {manager.threshold_increment}")
    print()
    
    # 模拟训练过程
    performances = [0.5, 0.65, 0.75, 0.85] * 10  # 重复性能模式
    step = 0
    
    for performance in performances:
        if not manager.should_continue_curriculum():
            break
            
        step += 1
        current_threshold = manager.get_current_threshold()
        
        print(f"步骤{step}: 轮次{manager.current_round}, 阶段{manager.current_stage}")
        print(f"  性能: {performance:.2f}, 阈值: {current_threshold:.2f}")
        
        should_advance = manager.should_advance_stage(performance)
        print(f"  进阶检查: {'✅' if should_advance else '❌'}")
        
        if should_advance:
            success = manager.advance_stage()
            if not success:
                break
        
        print()
        
        if step > 20:  # 防止无限循环
            break
    
    # 最终统计
    print("=" * 50)
    print("📊 最终统计:")
    print(f"  完成轮次: {manager.completed_rounds}")
    print(f"  当前轮次: {manager.current_round}")
    print(f"  总阶段完成: {len(manager.all_stage_history)}")
    
    # 显示阈值变化
    print("\n📈 各轮次阈值变化:")
    for round_num in range(1, manager.current_round + 1):
        print(f"  轮次{round_num}:")
        for stage_idx in range(len(manager.curriculum_stages)):
            base = manager.curriculum_stages[stage_idx].performance_threshold
            current = base + (round_num - 1) * manager.threshold_increment
            print(f"    阶段{stage_idx}: {base:.2f} -> {current:.2f} (+{current-base:.2f})")

if __name__ == "__main__":
    test_cyclic_logic() 