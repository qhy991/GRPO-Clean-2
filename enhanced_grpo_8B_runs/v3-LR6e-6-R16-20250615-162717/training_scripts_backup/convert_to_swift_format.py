#!/usr/bin/env python3
"""
convert_to_swift_format.py - 将GRPO数据集转换为Swift格式

Swift支持的标准对话格式：
{
    "conversations": [
        {"from": "user", "value": "问题内容"},
        {"from": "assistant", "value": "回答内容"}
    ]
}
"""

import json
import argparse
import os
from typing import Dict, Any, List
from pathlib import Path

def convert_grpo_to_swift(grpo_data: Dict[str, Any]) -> Dict[str, Any]:
    """将单个GRPO数据样本转换为Swift格式"""
    
    # 提取问题和答案
    problem = grpo_data.get('problem', '')
    reference_solution = grpo_data.get('reference_solution', '')
    
    # 如果有testbench信息，加入到问题中
    testbench_info = ""
    if 'testbench_path' in grpo_data:
        testbench_info = f"\n\n测试平台要求请参考相关文件。"
    
    # 构造完整的问题
    full_problem = f"{problem}{testbench_info}"
    
    # Swift格式转换
    swift_data = {
        "conversations": [
            {
                "from": "user",
                "value": full_problem
            },
            {
                "from": "assistant", 
                "value": reference_solution
            }
        ]
    }
    
    # 保留原始元数据（可选）
    if 'level' in grpo_data:
        swift_data['level'] = grpo_data['level']
    if 'complexity_score' in grpo_data:
        swift_data['complexity_score'] = grpo_data['complexity_score']
    if 'category' in grpo_data:
        swift_data['category'] = grpo_data['category']
    
    return swift_data

def main():
    parser = argparse.ArgumentParser(description='转换GRPO数据集为Swift格式')
    parser.add_argument('input_file', help='输入的GRPO数据集文件(.jsonl)')
    parser.add_argument('--output_file', help='输出的Swift格式文件(.jsonl)', default=None)
    parser.add_argument('--sample_limit', type=int, help='限制转换的样本数量（用于测试）', default=None)
    parser.add_argument('--preview', action='store_true', help='预览转换结果而不保存')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1
    
    # 确定输出文件名
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"swift_{input_path.stem}.jsonl"
    
    print(f"🔄 转换数据集格式...")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    
    converted_count = 0
    error_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            if args.preview:
                print(f"\n📝 预览转换结果:")
                print("=" * 50)
            else:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for line_num, line in enumerate(infile, 1):
                        if args.sample_limit and converted_count >= args.sample_limit:
                            break
                            
                        try:
                            grpo_data = json.loads(line.strip())
                            swift_data = convert_grpo_to_swift(grpo_data)
                            
                            if args.preview:
                                if converted_count < 3:  # 只预览前3个
                                    print(f"\n样本 {converted_count + 1}:")
                                    print(f"原始问题长度: {len(grpo_data.get('problem', ''))}")
                                    print(f"原始答案长度: {len(grpo_data.get('reference_solution', ''))}")
                                    print(f"Swift格式:")
                                    print(json.dumps(swift_data, ensure_ascii=False, indent=2))
                                    print("-" * 30)
                            else:
                                outfile.write(json.dumps(swift_data, ensure_ascii=False) + '\n')
                            
                            converted_count += 1
                            
                            if converted_count % 1000 == 0:
                                print(f"已转换 {converted_count} 个样本...")
                                
                        except json.JSONDecodeError as e:
                            print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                            error_count += 1
                        except Exception as e:
                            print(f"⚠️ 第{line_num}行处理错误: {e}")
                            error_count += 1
                            
    except Exception as e:
        print(f"❌ 文件处理错误: {e}")
        return 1
    
    print(f"\n✅ 转换完成!")
    print(f"成功转换: {converted_count} 个样本")
    if error_count > 0:
        print(f"错误样本: {error_count} 个")
    
    if not args.preview:
        print(f"输出文件: {output_path}")
        
        # 验证输出文件
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"文件大小: {file_size:.1f} MB")
            
            # 检查第一行格式
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    first_data = json.loads(first_line)
                    if 'conversations' in first_data:
                        print("✅ Swift格式验证通过")
                    else:
                        print("⚠️ Swift格式验证失败：缺少conversations字段")
            except Exception as e:
                print(f"⚠️ 格式验证失败: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 