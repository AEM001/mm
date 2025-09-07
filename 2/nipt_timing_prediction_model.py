#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT时点预测模型
基于区间删失生存模型和指定BMI区间端点的NIPT最佳时点预测

区间端点：-3.9, -1.4, -0.2, 1.5, 1.6, 4.8
实现步骤：
1. 加载预训练的生存模型
2. 根据指定区间端点进行BMI分组
3. 为每组预测最佳NIPT时点
4. 生成时点建议报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTTimingPredictionModel:
    """
    NIPT时点预测模型类
    基于预训练的区间删失生存模型进行时点预测
    """
    
    def __init__(self):
        self.data = None
        self.model_params = None
        self.bmi_boundaries = [-3.9, -1.4, -0.2, 1.5, 1.6, 4.8]  # 指定的区间端点
        self.bmi_groups = None
        self.optimal_timings = None
        # 时间标准化参数（与原模型保持一致）
        self.time_mean = 16.846
        self.time_std = 4.076
        # BMI标准化参数（估算值）
        self.bmi_mean = 24.5
        self.bmi_std = 4.2
    
    def standardized_to_original_time(self, standardized_time):
        """将标准化时间转换为原始孕周"""
        return standardized_time * self.time_std + self.time_mean
    
    def original_to_standardized_time(self, original_time):
        """将原始孕周转换为标准化时间"""
        return (original_time - self.time_mean) / self.time_std
    
    def standardized_to_original_bmi(self, standardized_bmi):
        """将标准化BMI转换为原始BMI"""
        return standardized_bmi * self.bmi_std + self.bmi_mean
    
    def original_to_standardized_bmi(self, original_bmi):
        """将原始BMI转换为标准化BMI"""
        return (original_bmi - self.bmi_mean) / self.bmi_std
        
    def load_data_and_model(self, data_path):
        """
        加载数据并设置预训练模型参数
        
        Args:
            data_path: 预处理数据文件路径
        """
        print("=== 步骤1: 数据加载和模型初始化 ===")
        
        # 读取预处理数据
        self.data = pd.read_csv(data_path)
        print(f"数据加载成功，共{len(self.data)}条记录")
        print(f"可用的列: {list(self.data.columns)}")
        
        # 检查必要的列
        required_cols = ['孕妇代码', 'Y染色体浓度', 'BMI_标准化']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"警告: 缺少必要列: {missing_cols}")
            return False
        
        # 数据清洗
        self.data = self.data.dropna(subset=required_cols)
        print(f"清洗后数据量: {len(self.data)}")
        
        # 设置预训练模型参数（基于interval_censored_survival_model.py的实际训练结果）
        # 这些参数来自已训练好的AFT模型
        self.model_params = {
            'distribution': 'lognormal',
            'beta0': -2.1300,   # 截距
            'beta_bmi': 0.0797, # BMI系数（正值表示BMI越高，达标时间越晚）
            'beta_age': 0.0274, # 年龄系数
            'sigma': 0.5770     # 尺度参数
        }
        
        print(f"\n使用预训练模型参数:")
        for key, value in self.model_params.items():
            print(f"  {key}: {value}")
        
        return True
    
    def assign_bmi_groups_by_boundaries(self):
        """
        根据指定的BMI区间端点分配BMI组别
        """
        print(f"\n=== 步骤2: 根据指定区间端点进行BMI分组 ===")
        print(f"区间端点: {self.bmi_boundaries}")
        
        # 创建BMI组别（左闭右开区间）
        bmi_groups = []
        boundaries = sorted(self.bmi_boundaries)
        
        for _, row in self.data.iterrows():
            bmi_value = row['BMI_标准化']
            
            # 找到BMI值所属的区间（左闭右开）
            group_idx = 0
            for i, boundary in enumerate(boundaries):
                if bmi_value < boundary:
                    group_idx = i
                    break
            else:
                group_idx = len(boundaries)
            
            bmi_groups.append(group_idx)
        
        self.data['BMI组别'] = bmi_groups
        
        # 添加原始BMI值列
        self.data['BMI_原始'] = self.standardized_to_original_bmi(self.data['BMI_标准化'])
        
        # 统计各组别信息
        group_stats = []
        for group_id in sorted(self.data['BMI组别'].unique()):
            group_data = self.data[self.data['BMI组别'] == group_id]
            
            if len(group_data) == 0:
                continue
            
            # 计算区间范围
            if group_id == 0:
                left_bound = "-∞"
                right_bound = boundaries[0]
            elif group_id == len(boundaries):
                left_bound = boundaries[-1]
                right_bound = "+∞"
            else:
                left_bound = boundaries[group_id - 1]
                right_bound = boundaries[group_id]
            
            # 统计信息
            bmi_std_min = group_data['BMI_标准化'].min()
            bmi_std_max = group_data['BMI_标准化'].max()
            bmi_std_mean = group_data['BMI_标准化'].mean()
            
            bmi_orig_min = group_data['BMI_原始'].min()
            bmi_orig_max = group_data['BMI_原始'].max()
            bmi_orig_mean = group_data['BMI_原始'].mean()
            
            group_stats.append({
                '组别': group_id,
                '区间_标准化': f"[{left_bound}, {right_bound})",
                '实际BMI范围_标准化': f"[{bmi_std_min:.3f}, {bmi_std_max:.3f}]",
                '实际BMI范围_原始': f"[{bmi_orig_min:.1f}, {bmi_orig_max:.1f}]",
                'BMI均值_标准化': bmi_std_mean,
                'BMI均值_原始': bmi_orig_mean,
                '样本数': len(group_data),
                'Y染色体浓度均值': group_data['Y染色体浓度'].mean()
            })
        
        self.bmi_groups = pd.DataFrame(group_stats)
        
        print(f"\nBMI分组结果:")
        print(self.bmi_groups[['组别', '区间_标准化', '实际BMI范围_原始', 'BMI均值_原始', '样本数']])
        
        return self.bmi_groups
    
    def predict_survival_function(self, bmi_values, age_values=None, time_points=None):
        """
        基于AFT模型预测生存函数
        
        Args:
            bmi_values: BMI值数组（标准化）
            age_values: 年龄值数组（标准化）
            time_points: 时间点数组（标准化）
            
        Returns:
            survival_probs: 生存概率矩阵
            time_points: 时间点数组
        """
        if self.model_params is None:
            raise ValueError("模型参数尚未设置")
        
        if age_values is None:
            # 使用数据中年龄的均值（如果有年龄列）
            if '年龄_标准化' in self.data.columns:
                age_values = np.full_like(bmi_values, self.data['年龄_标准化'].mean())
            else:
                age_values = np.zeros_like(bmi_values)  # 如果没有年龄数据，使用0
            
        if time_points is None:
            # 设置时间点范围（标准化时间）
            time_points = np.linspace(-2, 3, 100)
        
        # 计算线性预测器
        linear_pred = (self.model_params['beta0'] + 
                      self.model_params['beta_bmi'] * bmi_values + 
                      self.model_params['beta_age'] * age_values)
        
        # 计算生存函数
        survival_probs = np.zeros((len(bmi_values), len(time_points)))
        
        distribution = self.model_params['distribution']
        sigma = self.model_params['sigma']
        
        for i, t in enumerate(time_points):
            if distribution == 'weibull':
                # Weibull生存函数
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            elif distribution == 'lognormal':
                # 对数正态生存函数
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = 1 - stats.norm.cdf(standardized_time)
            else:
                # 默认使用Weibull
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
        
        return survival_probs, time_points
    
    def determine_optimal_timing_for_groups(self, success_rate=0.9):
        """
        为每个BMI组确定最佳NIPT时点
        
        Args:
            success_rate: 目标成功率 (默认90%)
        """
        print(f"\n=== 步骤3: 最佳NIPT时点预测 (目标成功率: {success_rate*100}%) ===")
        
        optimal_timings = []
        
        for _, group in self.bmi_groups.iterrows():
            group_id = group['组别']
            group_data = self.data[self.data['BMI组别'] == group_id]
            
            if len(group_data) == 0:
                continue
            
            bmi_mean = group['BMI均值_标准化']
            
            # 获取年龄均值（如果有）
            if '年龄_标准化' in group_data.columns:
                age_mean = group_data['年龄_标准化'].mean()
            else:
                age_mean = 0.0
            
            print(f"\n第{group_id}组: BMI均值(标准化)={bmi_mean:.3f}, BMI均值(原始)={group['BMI均值_原始']:.1f}")
            
            # 预测该组的生存函数
            standardized_time_points = np.linspace(-2, 3, 200)
            survival_probs, time_points = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([age_mean]),
                time_points=standardized_time_points
            )
            
            # 计算达标概率 F(t) = 1 - S(t)
            success_probs = 1 - survival_probs[0]
            
            # 显示关键时间点的成功率
            key_times_std = [-1, 0, 1, 2]  # 标准化时间点
            print(f"  关键时间点成功率:")
            for kt in key_times_std:
                if kt >= time_points.min() and kt <= time_points.max():
                    idx = np.argmin(np.abs(time_points - kt))
                    original_week = self.standardized_to_original_time(kt)
                    print(f"    {original_week:.1f}周: {success_probs[idx]*100:.1f}%")
            
            # 找到首次达到目标成功率的时间点
            target_indices = np.where(success_probs >= success_rate)[0]
            
            if len(target_indices) > 0:
                optimal_time_std = time_points[target_indices[0]]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
                actual_success_rate = success_probs[target_indices[0]]
                print(f"  达到{success_rate*100}%成功率的时间点: {optimal_time_original:.1f}周")
            else:
                # 如果无法达到目标成功率，选择最高成功率对应的时间
                max_idx = np.argmax(success_probs)
                optimal_time_std = time_points[max_idx]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
                actual_success_rate = np.max(success_probs)
                print(f"  警告: 第{group_id}组无法达到{success_rate*100}%成功率")
                print(f"  实际最高成功率: {actual_success_rate*100:.1f}% (在{optimal_time_original:.1f}周)")
            
            # 风险等级评估
            if optimal_time_original <= 12:
                risk_level = "低风险"
            elif optimal_time_original <= 20:
                risk_level = "中风险"
            else:
                risk_level = "高风险"
            
            optimal_timings.append({
                '组别': group_id,
                'BMI区间_标准化': group['区间_标准化'],
                'BMI区间_原始': group['实际BMI范围_原始'],
                'BMI均值_原始': group['BMI均值_原始'],
                '样本数': group['样本数'],
                '最佳时点_周': optimal_time_original,
                '预期成功率': actual_success_rate,
                '风险等级': risk_level,
                '建议': self._generate_timing_recommendation(optimal_time_original, actual_success_rate, success_rate)
            })
        
        self.optimal_timings = pd.DataFrame(optimal_timings)
        
        print(f"\n=== 最佳NIPT时点预测结果 ===")
        for _, timing in self.optimal_timings.iterrows():
            print(f"\n第{timing['组别']}组 (BMI: {timing['BMI区间_原始']}, n={timing['样本数']})")
            print(f"  最佳时点: {timing['最佳时点_周']:.1f}周")
            print(f"  预期成功率: {timing['预期成功率']*100:.1f}%")
            print(f"  风险等级: {timing['风险等级']}")
            print(f"  建议: {timing['建议']}")
        
        return self.optimal_timings
    
    def _generate_timing_recommendation(self, optimal_time, actual_success_rate, target_success_rate):
        """
        生成时点建议
        """
        if actual_success_rate >= target_success_rate:
            if optimal_time <= 12:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，成功率高且风险低"
            elif optimal_time <= 20:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，成功率达标"
            else:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，但需注意较晚检测的风险"
        else:
            return f"该组难以达到{target_success_rate*100}%成功率，建议在{optimal_time:.1f}周检测并考虑其他检测方法"
    
    def visualize_results(self):
        """
        可视化分析结果
        """
        if self.optimal_timings is None:
            print("请先运行时点预测")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('基于指定区间端点的NIPT时点预测结果', fontsize=16, fontweight='bold')
        
        # 1. BMI分组分布
        ax1 = axes[0, 0]
        group_counts = self.bmi_groups['样本数'].values
        group_labels = [f"组{i}" for i in self.bmi_groups['组别']]
        bars1 = ax1.bar(group_labels, group_counts, color='lightblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('BMI组别')
        ax1.set_ylabel('样本数')
        ax1.set_title('各BMI组别样本分布')
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars1, group_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # 2. 最佳时点分布
        ax2 = axes[0, 1]
        optimal_times = self.optimal_timings['最佳时点_周'].values
        bars2 = ax2.bar(group_labels[:len(optimal_times)], optimal_times, 
                        color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('BMI组别')
        ax2.set_ylabel('最佳时点 (周)')
        ax2.set_title('各组最佳NIPT时点')
        ax2.grid(True, alpha=0.3)
        
        # 添加12周和20周参考线
        ax2.axhline(12, color='orange', linestyle='--', alpha=0.7, label='12周(低风险阈值)')
        ax2.axhline(20, color='red', linestyle='--', alpha=0.7, label='20周(中风险阈值)')
        ax2.legend()
        
        # 在柱状图上添加数值标签
        for bar, time in zip(bars2, optimal_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{time:.1f}', ha='center', va='bottom')
        
        # 3. 预期成功率
        ax3 = axes[1, 0]
        success_rates = self.optimal_timings['预期成功率'].values * 100
        bars3 = ax3.bar(group_labels[:len(success_rates)], success_rates, 
                        color='gold', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('BMI组别')
        ax3.set_ylabel('预期成功率 (%)')
        ax3.set_title('各组预期成功率')
        ax3.axhline(90, color='red', linestyle='--', alpha=0.7, label='90%目标线')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, rate in zip(bars3, success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 4. BMI均值 vs 最佳时点散点图
        ax4 = axes[1, 1]
        bmi_means = self.optimal_timings['BMI均值_原始'].values
        scatter = ax4.scatter(bmi_means, optimal_times, 
                             c=success_rates, cmap='RdYlGn', 
                             s=100, alpha=0.7, edgecolors='black')
        ax4.set_xlabel('BMI均值 (kg/m²)')
        ax4.set_ylabel('最佳时点 (周)')
        ax4.set_title('BMI vs 最佳时点关系')
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('预期成功率 (%)')
        
        # 添加组别标签
        for i, (bmi, time) in enumerate(zip(bmi_means, optimal_times)):
            ax4.annotate(f'组{self.optimal_timings.iloc[i]["组别"]}', 
                        (bmi, time), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('nipt_timing_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """
        生成预测报告
        """
        if self.optimal_timings is None:
            print("请先运行时点预测")
            return
        
        report = []
        report.append("# NIPT时点预测模型分析报告\n")
        report.append(f"## 分析概述\n")
        report.append(f"- 数据样本数: {len(self.data)}")
        report.append(f"- BMI区间端点: {self.bmi_boundaries}")
        report.append(f"- 分组数量: {len(self.bmi_groups)}")
        report.append(f"- 预测模型: {self.model_params['distribution']}分布AFT模型\n")
        
        report.append("## BMI分组信息\n")
        report.append("| 组别 | BMI区间(标准化) | BMI区间(原始) | BMI均值 | 样本数 |")
        report.append("|------|-----------------|---------------|---------|--------|")
        
        for _, group in self.bmi_groups.iterrows():
            report.append(f"| {group['组别']} | {group['区间_标准化']} | {group['实际BMI范围_原始']} | {group['BMI均值_原始']:.1f} | {group['样本数']} |")
        
        report.append("\n## NIPT时点预测结果\n")
        report.append("| 组别 | BMI区间 | 最佳时点(周) | 预期成功率 | 风险等级 | 建议 |")
        report.append("|------|---------|--------------|------------|----------|------|")
        
        for _, timing in self.optimal_timings.iterrows():
            report.append(f"| {timing['组别']} | {timing['BMI区间_原始']} | {timing['最佳时点_周']:.1f} | {timing['预期成功率']:.1%} | {timing['风险等级']} | {timing['建议']} |")
        
        report.append("\n## 总体建议\n")
        
        # 统计分析
        low_risk_count = len(self.optimal_timings[self.optimal_timings['风险等级'] == '低风险'])
        medium_risk_count = len(self.optimal_timings[self.optimal_timings['风险等级'] == '中风险'])
        high_risk_count = len(self.optimal_timings[self.optimal_timings['风险等级'] == '高风险'])
        
        avg_timing = self.optimal_timings['最佳时点_周'].mean()
        avg_success_rate = self.optimal_timings['预期成功率'].mean()
        
        report.append(f"- 平均最佳时点: {avg_timing:.1f}周")
        report.append(f"- 平均预期成功率: {avg_success_rate:.1%}")
        report.append(f"- 风险等级分布: 低风险{low_risk_count}组, 中风险{medium_risk_count}组, 高风险{high_risk_count}组")
        
        if avg_success_rate >= 0.9:
            report.append("- ✓ 整体预期成功率达到90%目标")
        else:
            report.append("- ⚠ 整体预期成功率未达到90%目标，建议优化检测方案")
        
        # 保存报告
        report_text = "\n".join(report)
        with open('nipt_timing_prediction_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n=== 预测报告 ===")
        print(report_text)
        
        # 保存详细结果
        self.optimal_timings.to_csv('nipt_timing_prediction_results.csv', index=False, encoding='utf-8')
        self.optimal_timings.to_excel('nipt_timing_prediction_results.xlsx', index=False)
        
        print("\n结果已保存:")
        print("- nipt_timing_prediction_report.md")
        print("- nipt_timing_prediction_results.csv")
        print("- nipt_timing_prediction_results.xlsx")
        print("- nipt_timing_prediction_results.png")
    
    def run_prediction(self, data_path):
        """
        运行完整的预测流程
        
        Args:
            data_path: 数据文件路径
        """
        print("开始NIPT时点预测分析...")
        
        # 1. 加载数据和模型
        if not self.load_data_and_model(data_path):
            return None
        
        # 2. 根据指定区间端点分组
        self.assign_bmi_groups_by_boundaries()
        
        # 3. 预测最佳时点
        results = self.determine_optimal_timing_for_groups()
        
        # 4. 可视化结果
        self.visualize_results()
        
        # 5. 生成报告
        self.generate_report()
        
        print("\n预测分析完成！")
        
        return results

def main():
    """
    主函数
    """
    # 数据文件路径
    data_path = r"c:\Users\Lu\Desktop\问题2&3代码\问题二\processed_data.csv"
    
    # 创建预测模型实例
    predictor = NIPTTimingPredictionModel()
    
    # 运行预测
    results = predictor.run_prediction(data_path)
    
    return results

if __name__ == "__main__":
    results = main()