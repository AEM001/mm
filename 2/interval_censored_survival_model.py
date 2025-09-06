#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区间删失生存模型在NIPT最佳时点选择中的应用
问题2：基于BMI的区间删失生存分析模型

实现步骤：
1. 数据准备：构建区间删失数据
2. 模型拟合：使用AFT模型拟合区间删失数据
3. BMI分组：基于模型结果进行BMI分组
4. 最佳时点确定：为每组确定最佳NIPT时点
5. 结果可视化和分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import os
import sys
# 将项目根目录加入模块搜索路径，确保可以导入 set_chinese_font.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from set_chinese_font import set_chinese_font
try:
    from lifelines import LogLogisticAFTFitter
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False

# 设置中文字体（跨平台）
set_chinese_font()

class IntervalCensoredSurvivalModel:
    """
    区间删失生存模型类
    """
    
    def __init__(self):
        self.data = None
        self.model_params = None
        self.bmi_groups = None
        self.optimal_timings = None
        # 时间标准化参数（假设：均值=15周，标准差=3周）
        self.time_mean = 16.846
        self.time_std = 4.076
        # 基于当前脚本所在目录的输出路径设置，确保跨设备可用
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, 'loglogistic_aft_results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def standardized_to_original_time(self, standardized_time):
        """将标准化时间转换为原始孕周"""
        return standardized_time * self.time_std + self.time_mean
    
    def original_to_standardized_time(self, original_time):
        """将原始孕周转换为标准化时间"""
        return (original_time - self.time_mean) / self.time_std
        
    def load_and_prepare_data(self, file_path):
        """
        加载并准备区间删失数据
        
        Args:
            file_path: 预处理数据文件路径 (processed_data.csv)
        """
        print("=== 步骤1: 数据准备 ===")
        
        # 读取预处理数据
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")
        print(f"可用的列: {list(df.columns)}")
        
        # 检查关键列是否存在
        required_cols = ['孕妇代码', 'Y染色体浓度']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"缺失的列: {missing_cols}")
            return None
            
        print(f"数据清洗前各列缺失值情况:")
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} 个缺失值")
        
        # 数据清洗 - 移除关键列的缺失值
        df_clean = df.dropna(subset=required_cols)
        print(f"移除关键列缺失值后数据形状: {df_clean.shape}")
        
        # 从标准化列名推断原始数据
        # 由于processed_data.csv中的数据已经标准化，我们需要反标准化或使用标准化数据
        # 这里我们使用标准化数据进行分析
        
        # 检查数据范围
        if len(df_clean) > 0:
            print(f"\n数据范围检查:")
            print(f"  Y染色体浓度范围: {df_clean['Y染色体浓度'].min():.6f} - {df_clean['Y染色体浓度'].max():.6f}")
            if '孕周_标准化' in df_clean.columns:
                print(f"  孕周(标准化)范围: {df_clean['孕周_标准化'].min():.3f} - {df_clean['孕周_标准化'].max():.3f}")
            if 'BMI_标准化' in df_clean.columns:
                print(f"  BMI(标准化)范围: {df_clean['BMI_标准化'].min():.3f} - {df_clean['BMI_标准化'].max():.3f}")
        
        df = df_clean
        
        # 构建区间删失数据（原始孕周尺度）
        interval_data = []
        
        # 由于processed_data.csv中数据已经标准化，我们需要适应新的列名
        # 检查可用的列名
        bmi_col = 'BMI_标准化' if 'BMI_标准化' in df.columns else None
        age_col = '年龄_标准化' if '年龄_标准化' in df.columns else None
        week_col = '孕周_标准化' if '孕周_标准化' in df.columns else None
        
        print(f"使用的列名: BMI={bmi_col}, 年龄={age_col}, 孕周={week_col}")
        
        eps = 1e-3  # 避免左删失下界为0
        
        for woman_code, woman_data in df.groupby('孕妇代码'):
            # 如果有孕周信息，按孕周排序，并去除缺失
            if week_col and week_col in woman_data.columns:
                woman_data = woman_data.dropna(subset=[week_col])
                if len(woman_data) == 0:
                    continue
                woman_data = woman_data.sort_values(week_col).reset_index(drop=True)
            else:
                # 无孕周信息则无法按原始尺度建模，跳过
                continue
            
            # 获取基本信息
            bmi = woman_data[bmi_col].iloc[0] if bmi_col else 0.0
            age = woman_data[age_col].iloc[0] if age_col and age_col in woman_data.columns else 0.0
            
            # Y 染色体浓度（logit 尺度）
            y_concentrations = woman_data['Y染色体浓度'].values
            threshold_percentage = 0.04
            threshold = np.log(threshold_percentage / (1 - threshold_percentage))
            qualified_indices = np.where(y_concentrations >= threshold)[0]
            
            # 将标准化孕周转换为原始孕周
            week_std = woman_data[week_col].values
            time_points = self.standardized_to_original_time(week_std)
            
            # 判别删失类型并生成区间（原始孕周）
            if len(qualified_indices) == 0:
                censoring_type = 'right'
                left_time = float(time_points[-1])
                right_time = np.inf
                event_time = left_time
            elif qualified_indices[0] == 0:
                censoring_type = 'left'
                left_time = eps
                right_time = float(time_points[0])
                event_time = right_time
            else:
                censoring_type = 'interval'
                first_qualified_idx = qualified_indices[0]
                left_time = float(time_points[first_qualified_idx - 1])
                right_time = float(time_points[first_qualified_idx])
                event_time = (left_time + right_time) / 2.0
            
            interval_data.append({
                '孕妇代码': woman_code,
                'BMI': bmi,
                '年龄': age,
                '删失类型': censoring_type,
                '左端点': left_time,
                '右端点': right_time,
                'lower_bound': left_time,
                'upper_bound': right_time,
                '事件时间估计': event_time,
                '检测次数': len(woman_data),
                'Y浓度均值': float(np.mean(y_concentrations)),
                'Y浓度最大值': float(np.max(y_concentrations)),
                # lifelines 使用 ASCII 列名
                'bmi': bmi,
                'age': age,
            })
        
        self.data = pd.DataFrame(interval_data)
        
        print(f"\n构建的区间删失数据:")
        print(f"总样本数: {len(self.data)}")
        print(f"删失类型分布:")
        print(self.data['删失类型'].value_counts())
        print(f"\nBMI范围: {self.data['BMI'].min():.1f} - {self.data['BMI'].max():.1f}")
        print(f"事件时间范围: {self.data['事件时间估计'].min():.1f} - {self.data['事件时间估计'].max():.1f}周")
        
        # 保存构建的数据集到子目录
        try:
            out_csv = os.path.join(self.output_dir, 'interval_censored_dataset.csv')
            self.data.to_csv(out_csv, index=False)
            print(f"已保存区间删失数据集: {os.path.basename(out_csv)}")
        except Exception:
            pass
        return self.data
    
    def fit_aft_model(self, distribution='loglogistic'):
        """
        拟合加速失效时间(AFT)模型（仅支持 Log-Logistic / 区间删失）
        
        Args:
            distribution: 目前仅支持 'loglogistic'
        """
        print(f"\n=== 步骤2: AFT模型拟合 ({distribution}分布) ===")
        # 仅支持 Log-Logistic AFT（区间删失）
        if distribution.lower() != 'loglogistic':
            raise ValueError("当前实现仅支持 'loglogistic' 分布。")
        if not HAVE_LIFELINES:
            raise ImportError("未检测到 lifelines，请先安装：pip install lifelines 或 conda install -c conda-forge lifelines")
        required_cols = ['lower_bound', 'upper_bound', 'bmi']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"缺少必要列: {col}，请先运行 load_and_prepare_data() 并确保包含孕周信息。")
        df_model = self.data[['lower_bound', 'upper_bound']].copy()
        df_model['bmi'] = self.data['bmi'].fillna(0.0).values
        df_model['age'] = self.data['age'].fillna(0.0).values if 'age' in self.data.columns else 0.0
        aft = LogLogisticAFTFitter()
        aft.fit_interval_censoring(df_model, lower_bound_col='lower_bound', upper_bound_col='upper_bound', formula="bmi + age")
        self.model = aft
        self.model_params = {
            'distribution': 'loglogistic_aft',
            'log_likelihood': float(getattr(aft, 'log_likelihood_', np.nan)),
        }
        print("模型拟合成功! (Log-Logistic AFT / 区间删失)")
        try:
            print(aft.summary)
        except Exception:
            pass
        return self.model_params
    
    def predict_survival_function(self, bmi_values, age_values=None, time_points=None):
        """
        预测生存函数 S(t|BMI, Age)（原始孕周尺度）
        
        Args:
            bmi_values: BMI值数组
            age_values: 年龄值数组
            time_points: 时间点数组
            
        Returns:
            survival_probs: 生存概率矩阵
        """
        if self.model_params is None:
            raise ValueError("模型尚未拟合，请先调用fit_aft_model()")
        
        if age_values is None:
            age_values = np.full_like(bmi_values, self.data['年龄'].mean())
        
        # 默认时间范围：基于区间上下界的有限值
        if time_points is None:
            upper_finite = self.data['upper_bound'].replace(np.inf, np.nan)
            finite_times = np.concatenate([
                self.data['lower_bound'].values,
                upper_finite.values
            ])
            finite_times = finite_times[np.isfinite(finite_times)]
            if finite_times.size > 0:
                min_time = max(0.01, np.nanmin(finite_times))
                max_time = np.nanmax(finite_times)
            else:
                min_time, max_time = 8.0, 30.0
            time_points = np.linspace(min_time, max_time, 200)
        
        # Log-Logistic AFT 预测路径（唯一支持）
        if str(self.model_params.get('distribution', '')).startswith('loglogistic') and hasattr(self, 'model'):
            X = pd.DataFrame({'bmi': bmi_values, 'age': age_values})
            sf_df = self.model.predict_survival_function(X, times=time_points)
            survival_probs = sf_df.values.T  # n x len(times)
            print(f"    调试信息: BMI={bmi_values[0]:.2f}, 年龄={age_values[0]:.2f}")
            print(f"    使用 Log-Logistic AFT 模型进行预测，时间范围: [{time_points[0]:.1f}, {time_points[-1]:.1f}] 周")
            return survival_probs, time_points
        raise ValueError("当前预测仅支持已拟合的 Log-Logistic AFT 模型。")
    
    def perform_bmi_clustering(self, n_clusters=None):
        """
        基于BMI进行聚类分组
        
        Args:
            n_clusters: 聚类数量，如果为None则自动确定
        """
        print(f"\n=== 步骤3: BMI聚类分组 ===")
        
        bmi_values = self.data['BMI'].values.reshape(-1, 1)
        
        if n_clusters is None:
            # 使用肘部法则确定最优聚类数
            inertias = []
            K_range = range(2, 8)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(bmi_values)
                inertias.append(kmeans.inertia_)
            
            # 简单的肘部检测
            diffs = np.diff(inertias)
            diff_ratios = diffs[:-1] / diffs[1:]
            optimal_k = K_range[np.argmax(diff_ratios) + 1]
            
            print(f"自动确定最优聚类数: {optimal_k}")
        else:
            optimal_k = n_clusters
            
        # 执行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(bmi_values)
        
        # 按聚类中心排序
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        
        # 重新标记聚类
        new_labels = np.zeros_like(cluster_labels)
        for new_idx, old_idx in enumerate(sorted_indices):
            new_labels[cluster_labels == old_idx] = new_idx
            
        self.data['BMI聚类'] = new_labels
        
        # 计算BMI区间
        bmi_groups = []
        for i in range(optimal_k):
            group_data = self.data[self.data['BMI聚类'] == i]
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            
            bmi_groups.append({
                '组别': i + 1,
                'BMI最小值': bmi_min,
                'BMI最大值': bmi_max,
                'BMI均值': bmi_mean,
                '样本数': len(group_data),
                'BMI区间': f'[{bmi_min:.1f}, {bmi_max:.1f}]'
            })
            
        self.bmi_groups = pd.DataFrame(bmi_groups)
        
        print(f"\nBMI分组结果:")
        print(self.bmi_groups)
        
        return self.bmi_groups
    
    def determine_optimal_timing(self, success_rate=0.9):
        """
        为每个BMI组确定最佳NIPT时点
        
        Args:
            success_rate: 目标成功率 (默认90%)
        """
        print(f"\n=== 步骤4: 最佳NIPT时点确定 (目标成功率: {success_rate*100}%) ===")
        
        optimal_timings = []
        
        for _, group in self.bmi_groups.iterrows():
            group_id = group['组别'] - 1
            group_data = self.data[self.data['BMI聚类'] == group_id]
            bmi_mean = group['BMI均值']
            age_mean = group_data['年龄'].mean()  # 计算该组的年龄均值
            
            print(f"第{group['组别']}组: BMI均值={bmi_mean:.2f}, 年龄均值={age_mean:.2f}")
            
            # 预测该组的生存函数（原始孕周尺度）
            survival_probs, time_points = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([age_mean])
            )
            
            # 计算达标概率 F(t) = 1 - S(t)
            success_probs = 1 - survival_probs[0]
            
            # 添加调试信息：显示关键时间点的成功率
            key_times = [12, 16, 20, 24]  # 原始孕周
            print(f"  关键时间点成功率:")
            for kt in key_times:
                if kt >= time_points.min() and kt <= time_points.max():
                    idx = np.argmin(np.abs(time_points - kt))
                    print(f"    {kt:.1f}周: {success_probs[idx]*100:.1f}%")
            
            # 找到首次达到目标成功率的时间点
            target_indices = np.where(success_probs >= success_rate)[0]
            
            if len(target_indices) > 0:
                optimal_time_original = time_points[target_indices[0]]
                print(f"  达到{success_rate*100}%成功率的时间点: {optimal_time_original:.1f}周")
            else:
                # 如果无法达到目标成功率，选择最高成功率对应的时间
                optimal_time_original = time_points[np.argmax(success_probs)]
                actual_success_rate = np.max(success_probs)
                print(f"  警告: 第{group['组别']}组无法达到{success_rate*100}%成功率，实际最高成功率: {actual_success_rate*100:.1f}%")
            
            # 应用12周风险阈值逻辑（使用原始时间）
            risk_level = "低风险" if optimal_time_original <= 12 else "中风险" if optimal_time_original <= 27 else "高风险"
            
            optimal_timings.append({
                '组别': group['组别'],
                'BMI区间': group['BMI区间'],
                'BMI均值': bmi_mean,
                '样本数': group['样本数'],
                '最佳时点': optimal_time_original,  # 使用转换后的原始时间
                '预期成功率': success_probs[target_indices[0]] if len(target_indices) > 0 else np.max(success_probs),
                '风险等级': risk_level
            })
            
        self.optimal_timings = pd.DataFrame(optimal_timings)
        
        print(f"\n最佳NIPT时点结果:")
        for _, timing in self.optimal_timings.iterrows():
            print(f"第{timing['组别']}组 (BMI: {timing['BMI区间']}, n={timing['样本数']})")
            print(f"  最佳时点: {timing['最佳时点']:.1f}周")
            print(f"  预期成功率: {timing['预期成功率']*100:.1f}%")
            print(f"  风险等级: {timing['风险等级']}")
            
            # 添加调试信息
            if timing['最佳时点'] <= 8.5:  # 如果最佳时点接近最小值
                print(f"  注意: 该组在{timing['最佳时点']:.1f}周就达到目标成功率，可能需要更早的时间点")
            print()
            
        return self.optimal_timings
    
    def visualize_results(self):
        """
        可视化分析结果
        """
        print(f"\n=== 步骤5: 结果可视化 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. BMI分组散点图
        ax1 = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.bmi_groups)))
        
        for i, (_, group) in enumerate(self.bmi_groups.iterrows()):
            group_data = self.data[self.data['BMI聚类'] == i]
            ax1.scatter(group_data['BMI'], group_data['事件时间估计'], 
                       c=[colors[i]], label=f'第{i+1}组', alpha=0.7, s=50)
            
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('Y染色体浓度达标时间 (周)')
        ax1.set_title('BMI分组与达标时间分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 生存曲线
        ax2 = axes[0, 1]
        # 使用原始孕周范围
        upper_finite = self.data['upper_bound'].replace(np.inf, np.nan)
        finite_times = np.concatenate([
            self.data['lower_bound'].values,
            upper_finite.values
        ])
        finite_times = finite_times[np.isfinite(finite_times)]
        if finite_times.size > 0:
            tmin = max(0.01, np.nanmin(finite_times))
            tmax = np.nanmax(finite_times)
        else:
            tmin, tmax = 8.0, 30.0
        original_time_points = np.linspace(tmin, tmax, 200)
        
        for i, (_, group) in enumerate(self.bmi_groups.iterrows()):
            bmi_mean = group['BMI均值']
            survival_probs, _ = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                time_points=original_time_points
            )
            
            ax2.plot(original_time_points, survival_probs[0], 
                    color=colors[i], linewidth=2, 
                    label=f'第{i+1}组 (BMI={bmi_mean:.1f})')
            
        ax2.set_xlabel('孕周')
        ax2.set_ylabel('生存概率 S(t)')
        ax2.set_title('各BMI组生存曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最佳时点柱状图
        ax3 = axes[1, 0]
        groups = [f"第{row['组别']}组\n{row['BMI区间']}" for _, row in self.optimal_timings.iterrows()]
        timings = self.optimal_timings['最佳时点'].values
        
        bars = ax3.bar(groups, timings, color=colors[:len(groups)], alpha=0.7)
        ax3.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周风险阈值')
        ax3.set_ylabel('最佳NIPT时点 (周)')
        ax3.set_title('各BMI组最佳NIPT时点')
        ax3.legend()
        
        # 在柱子上标注数值
        for bar, timing in zip(bars, timings):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{timing:.1f}', ha='center', va='bottom')
        
        # 4. 成功率对比
        ax4 = axes[1, 1]
        success_rates = self.optimal_timings['预期成功率'].values * 100
        
        bars = ax4.bar(groups, success_rates, color=colors[:len(groups)], alpha=0.7)
        ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90%目标成功率')
        ax4.set_ylabel('预期成功率 (%)')
        ax4.set_title('各BMI组预期成功率')
        ax4.legend()
        
        # 在柱子上标注数值
        for bar, rate in zip(bars, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'interval_censored_survival_analysis_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图表已保存为:", os.path.basename(save_path))
    
    def generate_report(self):
        """
        生成分析报告
        """
        print(f"\n=== 分析报告 ===")
        
        report = f"""
# 区间删失生存模型分析报告

## 1. 数据概况
- 总样本数: {len(self.data)}
- BMI范围: {self.data['BMI'].min():.1f} - {self.data['BMI'].max():.1f}
- 达标时间范围: {self.data['事件时间估计'].min():.1f} - {self.data['事件时间估计'].max():.1f}周

## 2. 删失类型分布
"""
        
        for censoring_type, count in self.data['删失类型'].value_counts().items():
            report += f"- {censoring_type}: {count}例 ({count/len(self.data)*100:.1f}%)\n"
            
        # 3. 模型参数
        report += "\n## 3. 模型参数\n"
        report += "- 分布类型: Log-Logistic AFT (区间删失)\n"
        try:
            ll = float(getattr(self.model, 'log_likelihood_', np.nan))
            report += f"- 对数似然: {ll:.2f}\n"
        except Exception:
            pass
        try:
            summary_str = self.model.summary.to_string()
            report += "\n### 参数估计摘要\n\n````\n" + summary_str + "\n````\n"
        except Exception:
            pass

        report += "\n## 4. BMI分组结果\n"
        
        for _, group in self.bmi_groups.iterrows():
            report += f"- 第{group['组别']}组: BMI {group['BMI区间']}, 样本数 {group['样本数']}\n"
            
        report += f"""

## 5. 最佳NIPT时点建议
"""
        
        for _, timing in self.optimal_timings.iterrows():
            report += f"""
### 第{timing['组别']}组 (BMI: {timing['BMI区间']})
- 最佳检测时点: {timing['最佳时点']:.1f}周
- 预期成功率: {timing['预期成功率']*100:.1f}%
- 风险等级: {timing['风险等级']}

"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'interval_censored_survival_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print("分析报告已保存为:", os.path.basename(report_path))
        print(report)
        
        return report


def main():
    """
    主函数：执行完整的区间删失生存分析流程
    """
    print("区间删失生存模型在NIPT最佳时点选择中的应用")
    print("使用预处理数据 (processed_data.csv)")
    print("=" * 50)
    
    # 创建模型实例
    model = IntervalCensoredSurvivalModel()
    
    try:
        # 步骤1: 数据准备
        # 使用与脚本同目录下的 processed_data.csv，避免跨设备路径问题
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data.csv')
        model.load_and_prepare_data(data_file)
        
        # 步骤2: 模型拟合（使用 Log-Logistic AFT 区间删失）
        model.fit_aft_model(distribution='loglogistic')
        
        # 步骤3: BMI分组
        model.perform_bmi_clustering(n_clusters=4)  # 可以调整聚类数
        
        # 步骤4: 最佳时点确定
        model.determine_optimal_timing(success_rate=0.9)
        
        # 步骤5: 结果可视化
        model.visualize_results()
        
        # 步骤6: 生成报告
        model.generate_report()
        
        print("\n=== 分析完成 ===")
        print("所有结果文件已保存到子目录: loglogistic_aft_results/")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()