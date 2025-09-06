#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径修复后的代码是否正常工作
"""

import os
import sys

def test_imports():
    """测试所有模块的导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from gamm_y_chromosome_prediction import GAMMYChromosomePredictor
        print("✓ gamm_y_chromosome_prediction 导入成功")
    except Exception as e:
        print(f"✗ gamm_y_chromosome_prediction 导入失败: {e}")
    
    try:
        from kmeans_bmi_segmentation import BMISegmentationAnalyzer
        print("✓ kmeans_bmi_segmentation 导入成功")
    except Exception as e:
        print(f"✗ kmeans_bmi_segmentation 导入失败: {e}")
    
    try:
        from nipt_timing_optimization import NIPTTimingOptimizer
        print("✓ nipt_timing_optimization 导入成功")
    except Exception as e:
        print(f"✗ nipt_timing_optimization 导入失败: {e}")
    
    try:
        from gamm_detection_error_analysis import main
        print("✓ gamm_detection_error_analysis 导入成功")
    except Exception as e:
        print(f"✗ gamm_detection_error_analysis 导入失败: {e}")

def test_data_paths():
    """测试数据文件路径"""
    print("\n=== 测试数据文件路径 ===")
    
    data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    if os.path.exists(data_path):
        print(f"✓ 数据文件存在: {data_path}")
    else:
        print(f"✗ 数据文件不存在: {data_path}")
    
    # 检查问题2的数据文件（用于detection_error_analysis）
    q2_data_path = '/Users/Mac/Downloads/mm/2/processed_data.csv'
    if os.path.exists(q2_data_path):
        print(f"✓ 问题2数据文件存在: {q2_data_path}")
    else:
        print(f"✗ 问题2数据文件不存在: {q2_data_path}")

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 测试GAMM预测器初始化
        from gamm_y_chromosome_prediction import GAMMYChromosomePredictor
        predictor = GAMMYChromosomePredictor(use_r_gamm=True)
        print("✓ GAMM预测器初始化成功")
    except Exception as e:
        print(f"✗ GAMM预测器初始化失败: {e}")
    
    try:
        # 测试BMI分段分析器初始化
        from kmeans_bmi_segmentation import BMISegmentationAnalyzer
        analyzer = BMISegmentationAnalyzer()
        print("✓ BMI分段分析器初始化成功")
    except Exception as e:
        print(f"✗ BMI分段分析器初始化失败: {e}")
    
    try:
        # 测试NIPT时点优化器初始化
        from nipt_timing_optimization import NIPTTimingOptimizer
        data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
        optimizer = NIPTTimingOptimizer(data_path)
        print("✓ NIPT时点优化器初始化成功")
    except Exception as e:
        print(f"✗ NIPT时点优化器初始化失败: {e}")

def main():
    """主测试函数"""
    print("开始测试路径修复后的代码...")
    
    test_imports()
    test_data_paths()
    test_basic_functionality()
    
    print("\n=== 测试完成 ===")
    print("如果所有项目都显示 ✓，说明路径修复成功")
    print("如果有 ✗ 项目，请检查相应的错误信息")

if __name__ == "__main__":
    main()
