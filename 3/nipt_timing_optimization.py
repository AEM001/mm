#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from set_chinese_font import set_chinese_font
    set_chinese_font()
except:
    # 如果中文字体设置失败，使用默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

class NIPTTimingOptimizer:
    """
    NIPT时点优化器
    基于BMI标准化分界值和Y染色体浓度进行检测时点优化
    """
    
    def __init__(self, data_path):
        """
        初始化NIPT时点优化器
        
        参数:
        data_path: str, 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.bmi_boundaries = [-3.857, -1.410, -1.374, -1.343, -1.130, 
                              0.360, 0.635, 1.085, 1.210, 2.073, 2.274, 2.935, 4.845]
        self.success_threshold = 0.9  # 90%成功率阈值
        self.detection_threshold = None  # Y染色体浓度检测阈值
        
        # 标准化参数（来自interval_censored_survival_model.py和kmeans_bmi_segmentation.py）
        self.time_mean = 16.846  # 原始孕周均值
        self.time_std = 4.076    # 原始孕周标准差
        # BMI标准化参数（根据数据范围估算）
        self.bmi_mean = 24.5     # 估算的BMI均值
        self.bmi_std = 4.2       # 估算的BMI标准差
        
    def load_data(self):
        """
        加载数据
        """
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
            print(f"数据加载成功，共{len(self.data)}条记录")
            print(f"数据列名: {list(self.data.columns)}")
            
            # 检查必要的列
            required_cols = ['Y染色体浓度', 'BMI_标准化', '孕周_标准化']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"警告: 缺少必要列: {missing_cols}")
            
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def convert_standardized_to_original(self, std_weeks=None, std_bmi=None):
        """
        将标准化值转换为原始值
        
        参数:
        std_weeks: float or array, 标准化孕周值
        std_bmi: float or array, 标准化BMI值
        
        返回:
        tuple: (原始孕周, 原始BMI)
        """
        original_weeks = None
        original_bmi = None
        
        if std_weeks is not None:
            original_weeks = std_weeks * self.time_std + self.time_mean
        
        if std_bmi is not None:
            original_bmi = std_bmi * self.bmi_std + self.bmi_mean
        
        return original_weeks, original_bmi
    
    def assign_bmi_groups(self):
        """
        根据BMI标准化分界值分配BMI组别
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 创建BMI组别（左闭右开区间），过滤极端组别
        bmi_groups = []
        boundaries = sorted(self.bmi_boundaries)
        
        for _, row in self.data.iterrows():
            bmi_value = row['BMI_标准化']
            
            # 找到BMI值所属的区间（左闭右开）
            group_idx = 0
            for i, boundary in enumerate(boundaries):
                if bmi_value < boundary:  # 改为严格小于，实现左闭右开
                    group_idx = i
                    break
            else:
                group_idx = len(boundaries)
            
            # 过滤极端组别：组别0 ([-∞, -3.857)) 和最后一个组别 ([4.845, +∞))
            if group_idx == 0 or group_idx == len(boundaries):
                group_idx = None  # 标记为无效组别
            
            bmi_groups.append(group_idx)
        
        self.data['BMI组别'] = bmi_groups
        
        # 过滤掉极端组别的数据
        valid_data = self.data[self.data['BMI组别'].notna()].copy()
        print(f"\n过滤前样本数: {len(self.data)}")
        print(f"过滤后样本数: {len(valid_data)}")
        print(f"过滤掉的极端BMI样本数: {len(self.data) - len(valid_data)}")
        
        # 更新数据为有效数据
        self.data = valid_data
        
        # 添加原始值列
        original_weeks, original_bmi = self.convert_standardized_to_original(
            std_weeks=self.data['孕周_标准化'], 
            std_bmi=self.data['BMI_标准化']
        )
        self.data['孕周_原始'] = original_weeks
        self.data['BMI_原始'] = original_bmi
        
        # 统计各组别信息
        group_stats = self.data.groupby('BMI组别').agg({
            'Y染色体浓度': ['count', 'mean', 'std', 'min', 'max'],
            'BMI_标准化': ['mean', 'min', 'max'],
            'BMI_原始': ['mean', 'min', 'max'],
            '孕周_标准化': ['mean', 'std'],
            '孕周_原始': ['mean', 'std']
        }).round(4)
        
        print("\n=== BMI组别统计 ===")
        print(group_stats)
        
        return group_stats
    
    def calculate_detection_threshold(self):
        """
        计算Y染色体浓度检测阈值
        确保90%以上的孕妇能成功检测
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        # 计算整体的Y染色体浓度分布
        y_concentrations = self.data['Y染色体浓度'].values
        
        # 计算10%分位数作为检测阈值（确保90%以上能检测到）
        threshold_10 = np.percentile(y_concentrations, 10)
        threshold_5 = np.percentile(y_concentrations, 5)
        threshold_1 = np.percentile(y_concentrations, 1)
        
        print("\n=== Y染色体浓度阈值分析 ===")
        print(f"1%分位数阈值: {threshold_1:.4f} (99%成功率)")
        print(f"5%分位数阈值: {threshold_5:.4f} (95%成功率)")
        print(f"10%分位数阈值: {threshold_10:.4f} (90%成功率)")
        
        # 选择10%分位数作为检测阈值
        self.detection_threshold = threshold_10
        
        # 计算实际成功率
        success_rate = (y_concentrations >= self.detection_threshold).mean()
        print(f"\n选择阈值: {self.detection_threshold:.4f}")
        print(f"实际成功率: {success_rate:.1%}")
        
        return self.detection_threshold
    
    def analyze_timing_by_bmi_groups(self):
        """
        按BMI组别分析最佳检测时点
        """
        if self.data is None or 'BMI组别' not in self.data.columns:
            print("请先加载数据并分配BMI组别")
            return
        
        if self.detection_threshold is None:
            print("请先计算检测阈值")
            return
        
        timing_recommendations = []
        
        print("\n=== 各BMI组别最佳检测时点分析 ===")
        
        for group in sorted(self.data['BMI组别'].unique()):
            group_data = self.data[self.data['BMI组别'] == group].copy()
            
            if len(group_data) == 0:
                continue
            
            # 筛选能够成功检测的样本
            successful_samples = group_data[group_data['Y染色体浓度'] >= self.detection_threshold]
            
            if len(successful_samples) == 0:
                print(f"\nBMI组别 {group}: 无法达到检测阈值的样本")
                continue
            
            # 计算该组的成功率
            success_rate = len(successful_samples) / len(group_data)
            
            # 计算最佳检测时点（成功样本的孕周分布）
            optimal_timing = successful_samples['孕周_标准化'].quantile(0.1)  # 10%分位数，确保早期检测
            mean_timing = successful_samples['孕周_标准化'].mean()
            
            # 转换为原始值
            optimal_timing_original, _ = self.convert_standardized_to_original(std_weeks=optimal_timing)
            mean_timing_original, _ = self.convert_standardized_to_original(std_weeks=mean_timing)
            
            # BMI范围（标准化和原始值）
            bmi_min_std = group_data['BMI_标准化'].min()
            bmi_max_std = group_data['BMI_标准化'].max()
            bmi_mean_std = group_data['BMI_标准化'].mean()
            
            _, bmi_min_orig = self.convert_standardized_to_original(std_bmi=bmi_min_std)
            _, bmi_max_orig = self.convert_standardized_to_original(std_bmi=bmi_max_std)
            _, bmi_mean_orig = self.convert_standardized_to_original(std_bmi=bmi_mean_std)
            
            timing_recommendations.append({
                'BMI组别': group,
                'BMI范围_标准化': f"[{bmi_min_std:.3f}, {bmi_max_std:.3f}]",
                'BMI范围_原始': f"[{bmi_min_orig:.1f}, {bmi_max_orig:.1f}]",
                'BMI均值_标准化': bmi_mean_std,
                'BMI均值_原始': bmi_mean_orig,
                '样本数': len(group_data),
                '成功率': success_rate,
                '建议检测时点_标准化': optimal_timing,
                '建议检测时点_原始周数': optimal_timing_original,
                '平均检测时点_标准化': mean_timing,
                '平均检测时点_原始周数': mean_timing_original,
                'Y染色体浓度均值': group_data['Y染色体浓度'].mean()
            })
            
            print(f"\nBMI组别 {group}:")
            print(f"  BMI范围(标准化): [{bmi_min_std:.3f}, {bmi_max_std:.3f}]")
            print(f"  BMI范围(原始): [{bmi_min_orig:.1f}, {bmi_max_orig:.1f}]")
            print(f"  样本数: {len(group_data)}")
            print(f"  成功率: {success_rate:.1%}")
            print(f"  建议检测时点(标准化): {optimal_timing:.3f}")
            print(f"  建议检测时点(原始): {optimal_timing_original:.1f}周")
            print(f"  平均检测时点(标准化): {mean_timing:.3f}")
            print(f"  平均检测时点(原始): {mean_timing_original:.1f}周")
        
        # 转换为DataFrame
        recommendations_df = pd.DataFrame(timing_recommendations)
        
        return recommendations_df
    
    def visualize_results(self):
        """
        可视化分析结果
        """
        if self.data is None:
            print("请先加载数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NIPT时点优化分析结果', fontsize=16, fontweight='bold')
        
        # 1. Y染色体浓度分布
        ax1 = axes[0, 0]
        ax1.hist(self.data['Y染色体浓度'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if self.detection_threshold is not None:
            ax1.axvline(self.detection_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'检测阈值: {self.detection_threshold:.3f}')
        ax1.set_xlabel('Y染色体浓度(标准化)')
        ax1.set_ylabel('频数')
        ax1.set_title('Y染色体浓度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. BMI组别vs Y染色体浓度
        ax2 = axes[0, 1]
        if 'BMI组别' in self.data.columns:
            box_data = [self.data[self.data['BMI组别'] == group]['Y染色体浓度'].values 
                       for group in sorted(self.data['BMI组别'].unique())]
            ax2.boxplot(box_data, labels=sorted(self.data['BMI组别'].unique()))
            if self.detection_threshold is not None:
                ax2.axhline(self.detection_threshold, color='red', linestyle='--', linewidth=2,
                           label=f'检测阈值: {self.detection_threshold:.3f}')
        ax2.set_xlabel('BMI组别')
        ax2.set_ylabel('Y染色体浓度(标准化)')
        ax2.set_title('各BMI组别Y染色体浓度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 孕周vs Y染色体浓度散点图（使用原始孕周）
        ax3 = axes[1, 0]
        if '孕周_原始' in self.data.columns:
            scatter = ax3.scatter(self.data['孕周_原始'], self.data['Y染色体浓度'], 
                                 c=self.data.get('BMI组别', 'blue'), alpha=0.6, cmap='viridis')
            ax3.set_xlabel('孕周(周)')
        else:
            scatter = ax3.scatter(self.data['孕周_标准化'], self.data['Y染色体浓度'], 
                                 c=self.data.get('BMI组别', 'blue'), alpha=0.6, cmap='viridis')
            ax3.set_xlabel('孕周(标准化)')
        
        if self.detection_threshold is not None:
            ax3.axhline(self.detection_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'检测阈值: {self.detection_threshold:.3f}')
        ax3.set_ylabel('Y染色体浓度(标准化)')
        ax3.set_title('孕周vs Y染色体浓度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        if 'BMI组别' in self.data.columns:
            plt.colorbar(scatter, ax=ax3, label='BMI组别')
        
        # 4. 成功率分析
        ax4 = axes[1, 1]
        if 'BMI组别' in self.data.columns and self.detection_threshold is not None:
            success_rates = []
            group_labels = []
            for group in sorted(self.data['BMI组别'].unique()):
                group_data = self.data[self.data['BMI组别'] == group]
                success_rate = (group_data['Y染色体浓度'] >= self.detection_threshold).mean()
                success_rates.append(success_rate)
                group_labels.append(f'组{group}')
            
            bars = ax4.bar(group_labels, success_rates, color='lightgreen', alpha=0.7, edgecolor='black')
            ax4.axhline(0.9, color='red', linestyle='--', linewidth=2, label='90%目标线')
            ax4.set_ylabel('成功率')
            ax4.set_title('各BMI组别检测成功率')
            ax4.set_ylim(0, 1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('nipt_timing_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, recommendations_df):
        """
        生成分析报告
        """
        if recommendations_df is None or len(recommendations_df) == 0:
            print("无有效的推荐结果")
            return
        
        report = []
        report.append("# NIPT时点优化分析报告\n")
        report.append(f"## 分析概述\n")
        report.append(f"- 数据样本数: {len(self.data)}")
        report.append(f"- BMI分界值: {self.bmi_boundaries}")
        report.append(f"- Y染色体浓度检测阈值: {self.detection_threshold:.4f}")
        report.append(f"- 目标成功率: {self.success_threshold:.0%}\n")
        
        report.append("## 各BMI组别检测建议\n")
        report.append("| BMI组别 | BMI范围(原始) | BMI范围(标准化) | 样本数 | 成功率 | 建议检测时点(周) | 建议检测时点(标准化) | 备注 |")
        report.append("|---------|---------------|-----------------|--------|--------|------------------|---------------------|------|")
        
        for _, row in recommendations_df.iterrows():
            status = "✓ 达标" if row['成功率'] >= 0.9 else "⚠ 需优化"
            report.append(f"| {row['BMI组别']} | {row['BMI范围_原始']} | {row['BMI范围_标准化']} | {row['样本数']} | {row['成功率']:.1%} | {row['建议检测时点_原始周数']:.1f} | {row['建议检测时点_标准化']:.3f} | {status} |")
        
        report.append("\n## 总体建议\n")
        overall_success = (self.data['Y染色体浓度'] >= self.detection_threshold).mean()
        report.append(f"- 整体成功率: {overall_success:.1%}")
        
        if overall_success >= 0.9:
            report.append("- ✓ 当前检测阈值能够满足90%成功率要求")
        else:
            report.append("- ⚠ 需要调整检测阈值或优化检测方案")
        
        # 保存报告
        report_text = "\n".join(report)
        with open('nipt_timing_optimization_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n=== 分析报告 ===")
        print(report_text)
        
        # 保存详细结果
        recommendations_df.to_csv('nipt_timing_recommendations.csv', index=False, encoding='utf-8')
        recommendations_df.to_excel('nipt_timing_recommendations.xlsx', index=False)
        
        print("\n结果已保存:")
        print("- nipt_timing_optimization_report.md")
        print("- nipt_timing_recommendations.csv")
        print("- nipt_timing_recommendations.xlsx")
        print("- nipt_timing_optimization_results.png")
    
    def run_analysis(self):
        """
        运行完整分析流程
        """
        print("开始NIPT时点优化分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 分配BMI组别
        self.assign_bmi_groups()
        
        # 3. 计算检测阈值
        self.calculate_detection_threshold()
        
        # 4. 分析各组别最佳时点
        recommendations_df = self.analyze_timing_by_bmi_groups()
        
        # 5. 可视化结果
        self.visualize_results()
        
        # 6. 生成报告
        self.generate_report(recommendations_df)
        
        print("\n分析完成！")
        
        return recommendations_df

def main():
    """
    主函数
    """
    # 数据文件路径
    data_path = "/Users/Mac/Downloads/mm/3/processed_data.csv"
    
    # 创建优化器实例
    optimizer = NIPTTimingOptimizer(data_path)
    
    # 运行分析
    results = optimizer.run_analysis()
    
    return results

if __name__ == "__main__":
    results = main()