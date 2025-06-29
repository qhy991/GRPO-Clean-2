#!/usr/bin/env python3
"""
数据集分层抽样预览脚本
用于分析数据集的类别分布，并预览分层抽样效果
"""

import json
import logging
from collections import defaultdict, Counter
from datasets import load_dataset
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_distribution(dataset_path: str):
    """分析数据集的类别分布"""
    logger.info(f"📊 分析数据集: {dataset_path}")
    
    # 加载数据集
    try:
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        logger.info(f"✅ 数据集加载成功: {len(dataset)} 条记录")
    except Exception as e:
        logger.error(f"❌ 数据集加载失败: {e}")
        return
    
    # 分析数据集字段
    logger.info(f"📋 数据集字段: {dataset.column_names}")
    
    # 分析各个潜在分类字段的分布
    classification_fields = ['level', 'category', 'difficulty', 'complexity_score']
    field_distributions = {}
    
    for field in classification_fields:
        if field in dataset.column_names:
            field_values = dataset[field]
            
            # 对于数值字段，进行分桶
            if field == 'complexity_score':
                bucketed_values = []
                for value in field_values:
                    if value is None:
                        bucketed_values.append('unknown')
                    elif isinstance(value, (int, float)):
                        if value <= 3:
                            bucketed_values.append('simple')
                        elif value <= 7:
                            bucketed_values.append('medium')
                        else:
                            bucketed_values.append('complex')
                    else:
                        bucketed_values.append('unknown')
                field_values = bucketed_values
            
            # 统计分布
            distribution = Counter(field_values)
            field_distributions[field] = distribution
            
            logger.info(f"📈 {field} 分布:")
            total = len(field_values)
            for value, count in distribution.most_common():
                percentage = (count / total) * 100
                logger.info(f"  {value}: {count} ({percentage:.1f}%)")
        else:
            logger.warning(f"⚠️ 字段 '{field}' 不存在于数据集中")
    
    # 分析组合类别分布
    if len(field_distributions) > 1:
        logger.info("🔍 分析组合类别分布...")
        combined_categories = defaultdict(int)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            category_parts = []
            
            for field in classification_fields:
                if field in example and field in field_distributions:
                    value = example[field]
                    
                    # 对complexity_score进行分桶
                    if field == 'complexity_score' and isinstance(value, (int, float)):
                        if value <= 3:
                            value = 'simple'
                        elif value <= 7:
                            value = 'medium'
                        else:
                            value = 'complex'
                    
                    category_parts.append(f"{field}:{value}")
            
            combined_key = '|'.join(category_parts) if category_parts else 'unknown'
            combined_categories[combined_key] += 1
        
        logger.info("📊 组合类别分布 (前10个):")
        total = len(dataset)
        for category, count in sorted(combined_categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        logger.info(f"📋 总共有 {len(combined_categories)} 个不同的组合类别")
    
    return field_distributions, dataset

def preview_stratified_sampling(dataset, sample_ratio=0.1, stratify_columns=['level', 'category']):
    """预览分层抽样效果"""
    logger.info(f"🎯 预览分层抽样效果 (采样比例: {sample_ratio*100:.1f}%)")
    
    # 导入分层抽样函数
    from grpo_project.data.dataset import stratified_sample_dataset
    
    # 执行分层抽样
    sampled_dataset = stratified_sample_dataset(
        dataset=dataset,
        sample_ratio=sample_ratio,
        stratify_columns=stratify_columns,
        min_samples_per_category=1,
        random_seed=42
    )
    
    return sampled_dataset

def main():
    parser = argparse.ArgumentParser(description='数据集分层抽样预览工具')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/qhy/Research/LLM/GRPO-RV/dataset/all-with-module-2.jsonl',
                       help='数据集路径')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                       help='采样比例 (默认: 0.1 即 10%)')
    parser.add_argument('--stratify_columns', type=str, default='level,category',
                       help='分层字段，用逗号分隔 (默认: level,category)')
    
    args = parser.parse_args()
    
    # 解析分层字段
    stratify_columns = [col.strip() for col in args.stratify_columns.split(',') if col.strip()]
    
    logger.info("=" * 60)
    logger.info("📊 数据集分层抽样预览工具")
    logger.info("=" * 60)
    
    # 分析原始数据集分布
    field_distributions, dataset = analyze_dataset_distribution(args.dataset_path)
    
    if dataset is None:
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 分层抽样预览")
    logger.info("=" * 60)
    
    # 预览分层抽样效果
    try:
        sampled_dataset = preview_stratified_sampling(
            dataset=dataset,
            sample_ratio=args.sample_ratio,
            stratify_columns=stratify_columns
        )
        
        logger.info(f"\n✅ 分层抽样预览完成！")
        logger.info(f"原始数据: {len(dataset)} 条")
        logger.info(f"采样后: {len(sampled_dataset)} 条")
        logger.info(f"实际采样比例: {len(sampled_dataset)/len(dataset)*100:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ 分层抽样预览失败: {e}")
        logger.error("请确保已正确安装 grpo_project 包")

if __name__ == "__main__":
    main() 