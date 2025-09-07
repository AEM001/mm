#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_chinese_font import set_chinese_font
set_chinese_font()

class NIPTTimingOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.bmi_boundaries = [-3.857, -1.410, -1.374, -1.343, -1.130,
                              0.360, 0.635, 1.085, 1.210, 2.073, 2.274, 2.935, 4.845]
        self.success_threshold = 0.9
        self.detection_threshold = None

        self.time_mean = 16.846
        self.time_std = 4.076
        self.bmi_mean = 24.5
        self.bmi_std = 4.2
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        return True
    
    def convert_standardized_to_original(self, std_weeks=None, std_bmi=None):
        original_weeks = None
        original_bmi = None
        
        if std_weeks is not None:
            original_weeks = std_weeks * self.time_std + self.time_mean
        
        if std_bmi is not None:
            original_bmi = std_bmi * self.bmi_std + self.bmi_mean
        
        return original_weeks, original_bmi
    
    def assign_bmi_groups(self):
        if self.data is None:
            return

        bmi_groups = []
        boundaries = sorted(self.bmi_boundaries)

        for _, row in self.data.iterrows():
            bmi_value = row['BMI_标准化']

            group_idx = 0
            for i, boundary in enumerate(boundaries):
                if bmi_value < boundary:
                    group_idx = i
                    break
            else:
                group_idx = len(boundaries)

            if group_idx == 0 or group_idx == len(boundaries):
                group_idx = None

            bmi_groups.append(group_idx)

        self.data['BMI组别'] = bmi_groups

        valid_data = self.data[self.data['BMI组别'].notna()].copy()
        self.data = valid_data

        original_weeks, original_bmi = self.convert_standardized_to_original(
            std_weeks=self.data['孕周_标准化'],
            std_bmi=self.data['BMI_标准化']
        )
        self.data['孕周_原始'] = original_weeks
        self.data['BMI_原始'] = original_bmi

        group_stats = self.data.groupby('BMI组别').agg({
            'Y染色体浓度': ['count', 'mean', 'std', 'min', 'max'],
            'BMI_标准化': ['mean', 'min', 'max'],
            'BMI_原始': ['mean', 'min', 'max'],
            '孕周_标准化': ['mean', 'std'],
            '孕周_原始': ['mean', 'std']
        }).round(4)

        return group_stats
    
    def calculate_detection_threshold(self):
        if self.data is None:
            return

        y_concentrations = self.data['Y染色体浓度'].values

        threshold_10 = np.percentile(y_concentrations, 10)
        threshold_5 = np.percentile(y_concentrations, 5)
        threshold_1 = np.percentile(y_concentrations, 1)

        self.detection_threshold = threshold_10

        success_rate = (y_concentrations >= self.detection_threshold).mean()

        return self.detection_threshold
    
    def analyze_timing_by_bmi_groups(self):
        if self.data is None or 'BMI组别' not in self.data.columns:
            return

        if self.detection_threshold is None:
            return

        timing_recommendations = []

        for group in sorted(self.data['BMI组别'].unique()):
            group_data = self.data[self.data['BMI组别'] == group].copy()

            if len(group_data) == 0:
                continue

            successful_samples = group_data[group_data['Y染色体浓度'] >= self.detection_threshold]

            if len(successful_samples) == 0:
                continue

            success_rate = len(successful_samples) / len(group_data)

            optimal_timing = successful_samples['孕周_标准化'].quantile(0.1)
            mean_timing = successful_samples['孕周_标准化'].mean()

            optimal_timing_original, _ = self.convert_standardized_to_original(std_weeks=optimal_timing)
            mean_timing_original, _ = self.convert_standardized_to_original(std_weeks=mean_timing)

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

        recommendations_df = pd.DataFrame(timing_recommendations)

        return recommendations_df
    
    def visualize_results(self):
        if self.data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NIPT时点优化分析结果', fontsize=16, fontweight='bold')

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

            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('nipt_timing_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_analysis(self):
        self.load_data()

        self.assign_bmi_groups()

        self.calculate_detection_threshold()

        recommendations_df = self.analyze_timing_by_bmi_groups()

        self.visualize_results()

        print("\n分析完成！")

        return recommendations_df

def main():
    data_path = "/Users/Mac/Downloads/mm/3/processed_data.csv"

    optimizer = NIPTTimingOptimizer(data_path)

    results = optimizer.run_analysis()

    return results

if __name__ == "__main__":
    results = main()