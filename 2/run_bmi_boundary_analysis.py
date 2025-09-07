#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题二：使用BMI分界值方法的检测误差分析
展示如何使用新的分组策略进行检测误差分析
"""

import os
import sys

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from detection_error_analysis import BMIBoundaryStrategy, run_error_analysis_with_strategy, compare_strategies

def main():
    """
    运行BMI分界值方法的检测误差分析
    """
    print("=== 问题二：BMI分界值分组的检测误差分析 ===")
    
    # 方法1：单独运行BMI分界值策略
    print("\n1. 运行BMI分界值分组策略:")
    strategy = BMIBoundaryStrategy()
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    try:
        result = run_error_analysis_with_strategy(
            strategy=strategy,
            data_file=data_file,
            output_suffix='bmi_boundary'
        )
        
        if result:
            print("BMI分界值分组分析完成！")
            print(f"策略: {result['strategy']}")
            if 'sim_summary' in result:
                print(f"平均时间偏差: {result['sim_summary']['总体偏差']['平均偏差']:.2f}周")
                print(f"平均RMSE: {result['sim_summary']['总体偏差']['平均RMSE']:.2f}周")
        
    except Exception as e:
        print(f"BMI分界值分组分析失败: {e}")
        print("这可能是由于缺少NIPTTimingOptimizer依赖或数据文件问题")
    
    # 方法2：比较两种策略（如果都可用）
    print("\n2. 比较不同分组策略:")
    try:
        results = compare_strategies()
        
        if len(results) > 1:
            print("策略比较完成！可以查看不同输出目录中的结果进行对比。")
        elif len(results) == 1:
            print("只有一种策略成功运行。")
        else:
            print("所有策略都运行失败。")
            
    except Exception as e:
        print(f"策略比较失败: {e}")

if __name__ == '__main__':
    main()
