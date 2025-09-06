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
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

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
        
        # 构建区间删失数据
        interval_data = []
        
        # 由于processed_data.csv中数据已经标准化，我们需要适应新的列名
        # 检查可用的列名
        bmi_col = 'BMI_标准化' if 'BMI_标准化' in df.columns else None
        age_col = '年龄_标准化' if '年龄_标准化' in df.columns else None
        week_col = '孕周_标准化' if '孕周_标准化' in df.columns else None
        
        print(f"使用的列名: BMI={bmi_col}, 年龄={age_col}, 孕周={week_col}")
        
        for woman_code, woman_data in df.groupby('孕妇代码'):
            # 如果有孕周信息，按孕周排序
            if week_col and week_col in woman_data.columns:
                woman_data = woman_data.sort_values(week_col).reset_index(drop=True)
            else:
                woman_data = woman_data.reset_index(drop=True)
            
            # 获取基本信息
            bmi = woman_data[bmi_col].iloc[0] if bmi_col else 0.0
            age = woman_data[age_col].iloc[0] if age_col and age_col in woman_data.columns else 0.0
            
            # 寻找Y染色体浓度达标的时间点
            # 注意：processed_data.csv中的Y染色体浓度可能已经是对数变换后的值
            y_concentrations = woman_data['Y染色体浓度'].values
            
            # 如果有孕周信息，使用孕周；否则使用索引作为时间代理
            if week_col and week_col in woman_data.columns:
                time_points = woman_data[week_col].values
            else:
                # 使用检测顺序作为时间代理
                time_points = np.arange(len(woman_data))
            
            # 判断删失类型 - 使用4%的Y染色体浓度作为达标标准（题目要求）
            # 4%的Logit变换: logit(0.04) = log(0.04/(1-0.04)) = log(0.04/0.96)
            threshold_percentage = 0.04
            threshold = np.log(threshold_percentage / (1 - threshold_percentage))  # Logit变换后的阈值
            qualified_indices = np.where(y_concentrations >= threshold)[0]
            
            if len(qualified_indices) == 0:
                # 右删失：所有检测都未达标
                censoring_type = 'right'
                left_time = time_points[-1]  # 最后一次检测时间
                right_time = np.inf
                event_time = time_points[-1]
                
            elif qualified_indices[0] == 0:
                # 左删失：第一次检测就达标
                censoring_type = 'left'
                left_time = 0
                right_time = time_points[0]
                event_time = time_points[0]
                
            else:
                # 区间删失：在某两次检测之间达标
                censoring_type = 'interval'
                first_qualified_idx = qualified_indices[0]
                left_time = time_points[first_qualified_idx - 1]
                right_time = time_points[first_qualified_idx]
                event_time = (left_time + right_time) / 2  # 区间中点作为估计
                
            interval_data.append({
                '孕妇代码': woman_code,
                'BMI': bmi,
                '年龄': age,
                '删失类型': censoring_type,
                '左端点': left_time,
                '右端点': right_time,
                '事件时间估计': event_time,
                '检测次数': len(woman_data),
                'Y浓度均值': np.mean(y_concentrations),
                'Y浓度最大值': np.max(y_concentrations)
            })
        
        self.data = pd.DataFrame(interval_data)
        
        print(f"\n构建的区间删失数据:")
        print(f"总样本数: {len(self.data)}")
        print(f"删失类型分布:")
        print(self.data['删失类型'].value_counts())
        print(f"\nBMI范围: {self.data['BMI'].min():.1f} - {self.data['BMI'].max():.1f}")
        print(f"事件时间范围: {self.data['事件时间估计'].min():.1f} - {self.data['事件时间估计'].max():.1f}周")
        
        return self.data
    
    def fit_aft_model(self, distribution='weibull'):
        """
        拟合加速失效时间(AFT)模型
        
        Args:
            distribution: 生存时间分布类型 ('weibull', 'lognormal', 'exponential')
        """
        print(f"\n=== 步骤2: AFT模型拟合 ({distribution}分布) ===")
        
        # 准备建模数据
        # 数据已经标准化，直接使用
        X = self.data[['BMI', '年龄']].fillna(0.0)  # 标准化数据的缺失值用0填充
        y = self.data['事件时间估计'].values
        
        # 确保时间值为正数（对于对数变换）
        y = np.maximum(y, 0.1)  # 避免零值或负值
        
        # 数据已经是标准化后的，直接使用
        X_scaled = X.values
        
        if distribution == 'weibull':
            # Weibull AFT模型的改进实现
            # log(T) = β0 + β1*BMI + β2*Age + σ*ε
            # 其中ε服从极值分布
            
            def weibull_log_likelihood(params):
                beta0, beta1, beta2, sigma = params
                
                # 确保sigma为正数
                if sigma <= 0:
                    return 1e10
                
                try:
                    # 线性预测器
                    linear_pred = beta0 + beta1 * X_scaled[:, 0] + beta2 * X_scaled[:, 1]
                    
                    # 标准化残差
                    log_y = np.log(np.maximum(y, 1e-10))  # 避免log(0)
                    standardized_residuals = (log_y - linear_pred) / sigma
                    
                    # 检查数值稳定性
                    if np.any(np.isnan(standardized_residuals)) or np.any(np.isinf(standardized_residuals)):
                        return 1e10
                    
                    # Weibull分布的对数似然（数值稳定版本）
                    exp_neg_resid = np.exp(-standardized_residuals)
                    # 避免exp溢出
                    exp_neg_resid = np.minimum(exp_neg_resid, 1e10)
                    
                    log_likelihood = np.sum(
                        -standardized_residuals - exp_neg_resid - np.log(sigma)
                    )
                    
                    # 检查结果的有效性
                    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                        return 1e10
                    
                    return -log_likelihood  # 返回负对数似然用于最小化
                    
                except Exception as e:
                    return 1e10
            
            # 改进的初始参数估计
            from scipy.optimize import minimize
            
            # 更稳定的初始值
            y_mean = np.mean(y)
            y_std = np.std(y)
            initial_params = [np.log(y_mean), 0.0, 0.0, max(0.5, y_std/y_mean)]
            
            # 参数边界约束
            bounds = [(-10, 10), (-5, 5), (-5, 5), (0.1, 10)]
            
            # 多种优化方法尝试
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            best_result = None
            best_likelihood = 1e10
            
            for method in methods:
                try:
                    result = minimize(weibull_log_likelihood, initial_params, 
                                    method=method, bounds=bounds,
                                    options={'maxiter': 2000, 'ftol': 1e-9})
                    
                    if result.success and result.fun < best_likelihood:
                        best_result = result
                        best_likelihood = result.fun
                        
                except Exception as e:
                    continue
            
            result = best_result
            
        elif distribution == 'lognormal':
            # 对数正态AFT模型实现
            # log(T) = β0 + β1*BMI + β2*Age + σ*ε
            # 其中ε服从标准正态分布
            
            def lognormal_log_likelihood(params):
                beta0, beta1, beta2, sigma = params
                
                # 确保sigma为正数
                if sigma <= 0:
                    return 1e10
                
                try:
                    # 线性预测器
                    linear_pred = beta0 + beta1 * X_scaled[:, 0] + beta2 * X_scaled[:, 1]
                    
                    # 标准化残差
                    log_y = np.log(np.maximum(y, 1e-10))  # 避免log(0)
                    standardized_residuals = (log_y - linear_pred) / sigma
                    
                    # 检查数值稳定性
                    if np.any(np.isnan(standardized_residuals)) or np.any(np.isinf(standardized_residuals)):
                        return 1e10
                    
                    # 对数正态分布的对数似然
                    log_likelihood = np.sum(
                        -0.5 * standardized_residuals**2 - 0.5 * np.log(2 * np.pi) - np.log(sigma)
                    )
                    
                    # 检查结果的有效性
                    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                        return 1e10
                    
                    return -log_likelihood  # 返回负对数似然用于最小化
                    
                except Exception as e:
                    return 1e10
            
            # 改进的初始参数估计
            from scipy.optimize import minimize
            
            # 更稳定的初始值
            y_mean = np.mean(y)
            y_std = np.std(y)
            initial_params = [np.log(y_mean), 0.0, 0.0, max(0.5, y_std/y_mean)]
            
            # 参数边界约束
            bounds = [(-10, 10), (-5, 5), (-5, 5), (0.1, 10)]
            
            # 多种优化方法尝试
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            best_result = None
            best_likelihood = 1e10
            
            for method in methods:
                try:
                    result = minimize(lognormal_log_likelihood, initial_params, 
                                    method=method, bounds=bounds,
                                    options={'maxiter': 2000, 'ftol': 1e-9})
                    
                    if result.success and result.fun < best_likelihood:
                        best_result = result
                        best_likelihood = result.fun
                        
                except Exception as e:
                    continue
            
            result = best_result
            
        else:
            raise ValueError(f"不支持的分布类型: {distribution}")
            
        # 处理优化结果
        if result is not None and result.success:
            beta0, beta1, beta2, sigma = result.x
            self.model_params = {
                'distribution': distribution,
                'beta0': beta0,
                'beta_bmi': beta1,
                'beta_age': beta2,
                'sigma': sigma,
                'log_likelihood': -result.fun
            }
            
            print(f"模型拟合成功!")
            print(f"参数估计:")
            print(f"  截距 (β0): {beta0:.4f}")
            print(f"  BMI系数 (β1): {beta1:.4f}")
            print(f"  年龄系数 (β2): {beta2:.4f}")
            print(f"  尺度参数 (σ): {sigma:.4f}")
            print(f"  对数似然: {-result.fun:.2f}")
            
            # 解释系数
            print(f"\n系数解释:")
            if beta1 > 0:
                print(f"  BMI每增加1个标准差，达标时间延长 {np.exp(beta1):.3f} 倍")
            else:
                print(f"  BMI每增加1个标准差，达标时间缩短为原来的 {np.exp(beta1):.3f} 倍")
                
        else:
            # 优化失败，使用简化模型
            print("标准优化失败，尝试简化模型...")
            
            # 使用简单的线性回归作为备用方案
            from sklearn.linear_model import LinearRegression
            
            try:
                # 对数变换的线性回归
                log_y = np.log(np.maximum(y, 1e-10))
                reg = LinearRegression().fit(X_scaled, log_y)
                
                beta0 = reg.intercept_
                beta1, beta2 = reg.coef_
                sigma = np.std(log_y - reg.predict(X_scaled))
                
                self.model_params = {
                    'distribution': 'linear_fallback',
                    'beta0': beta0,
                    'beta_bmi': beta1,
                    'beta_age': beta2,
                    'sigma': sigma,
                    'log_likelihood': None
                }
                
                print(f"使用线性回归备用模型成功!")
                print(f"参数估计:")
                print(f"  截距 (β0): {beta0:.4f}")
                print(f"  BMI系数 (β1): {beta1:.4f}")
                print(f"  年龄系数 (β2): {beta2:.4f}")
                print(f"  残差标准差 (σ): {sigma:.4f}")
                
            except Exception as e:
                print(f"所有模型拟合方法都失败了: {str(e)}")
                print(f"数据统计信息:")
                print(f"  样本数量: {len(y)}")
                print(f"  Y值范围: [{np.min(y):.3f}, {np.max(y):.3f}]")
                print(f"  X值范围: BMI[{np.min(X_scaled[:, 0]):.3f}, {np.max(X_scaled[:, 0]):.3f}], 年龄[{np.min(X_scaled[:, 1]):.3f}, {np.max(X_scaled[:, 1]):.3f}]")
                self.model_params = None
            return None
                
        return self.model_params
    
    def predict_survival_function(self, bmi_values, age_values=None, time_points=None):
        """
        预测生存函数 S(t|BMI, Age)
        
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
            
        if time_points is None:
            # 根据实际数据范围调整时间点
            min_time = self.data['事件时间估计'].min()
            max_time = self.data['事件时间估计'].max()
            time_points = np.linspace(min_time, max_time, 100)
            
        # 数据已经是标准化后的，直接使用
        X = np.column_stack([bmi_values, age_values])
        X_scaled = X
        
        # 计算线性预测器
        linear_pred = (self.model_params['beta0'] + 
                      self.model_params['beta_bmi'] * X_scaled[:, 0] + 
                      self.model_params['beta_age'] * X_scaled[:, 1])
        
        # 添加调试信息
        print(f"    调试信息: BMI={bmi_values[0]:.2f}, 年龄={age_values[0]:.2f}")
        print(f"    标准化后: BMI={X_scaled[0,0]:.2f}, 年龄={X_scaled[0,1]:.2f}")
        print(f"    线性预测器: {linear_pred[0]:.4f}")
        print(f"    模型参数: β0={self.model_params['beta0']:.4f}, β_bmi={self.model_params['beta_bmi']:.4f}, β_age={self.model_params['beta_age']:.4f}, σ={self.model_params['sigma']:.4f}")
        
        # 计算生存函数
        survival_probs = np.zeros((len(bmi_values), len(time_points)))
        
        distribution = self.model_params['distribution']
        
        for i, t in enumerate(time_points):
            # 注意：现在t是标准化时间，不需要取对数
            # 在AFT模型中，标准化时间直接用于计算
            if distribution == 'weibull':
                # Weibull生存函数（使用标准化时间）
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            elif distribution == 'lognormal':
                # 对数正态生存函数（使用标准化时间）
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = 1 - stats.norm.cdf(standardized_time)
            else:
                # 备用线性模型
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            
        return survival_probs, time_points
    
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
            
            # 预测该组的生存函数（使用标准化的时间尺度）
            # 注意：数据中的时间已经标准化，范围约为-1.4到2.0
            # 我们需要使用相同的标准化时间尺度进行预测
            standardized_time_points = np.linspace(-2, 3, 100)  # 扩展标准化时间范围
            survival_probs, time_points = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([age_mean]),  # 使用该组的年龄均值
                time_points=standardized_time_points
            )
            
            # 计算达标概率 F(t) = 1 - S(t)
            success_probs = 1 - survival_probs[0]
            
            # 添加调试信息：显示关键时间点的成功率
            key_times = [-1, 0, 1, 2]  # 标准化时间点
            print(f"  关键时间点成功率:")
            for kt in key_times:
                if kt >= time_points.min() and kt <= time_points.max():
                    idx = np.argmin(np.abs(time_points - kt))
                    original_week = self.standardized_to_original_time(kt)
                    print(f"    {original_week:.1f}周: {success_probs[idx]*100:.1f}%")
            
            # 找到首次达到目标成功率的时间点
            target_indices = np.where(success_probs >= success_rate)[0]
            
            if len(target_indices) > 0:
                optimal_time_std = time_points[target_indices[0]]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
                print(f"  达到{success_rate*100}%成功率的时间点: {optimal_time_original:.1f}周")
            else:
                # 如果无法达到目标成功率，选择最高成功率对应的时间
                optimal_time_std = time_points[np.argmax(success_probs)]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
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
            
            # 检查最佳时点是否接近最小值
            if optimal_time_original <= self.standardized_to_original_time(time_points[5]):  # 如果在前5个时间点内
                print(f"  警告: 第{group['组别']}组的最佳时点({optimal_time_original:.1f}周)接近最小时间点，可能需要扩展时间范围")
            
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
        # 使用标准化时间范围，与模型训练一致
        standardized_time_points = np.linspace(-2, 3, 100)
        # 转换为原始孕周显示
        original_time_points = self.standardized_to_original_time(standardized_time_points)
        
        for i, (_, group) in enumerate(self.bmi_groups.iterrows()):
            bmi_mean = group['BMI均值']
            survival_probs, _ = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                time_points=standardized_time_points
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
        plt.savefig('./interval_censored_survival_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图表已保存为: interval_censored_survival_analysis_results.png")
    
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
            
        report += f"""

## 3. 模型参数
- 分布类型: {self.model_params['distribution']}
- BMI系数: {self.model_params['beta_bmi']:.4f}
- 年龄系数: {self.model_params['beta_age']:.4f}
- 对数似然: {self.model_params['log_likelihood']:.2f}

## 4. BMI分组结果
"""
        
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
        with open('./interval_censored_survival_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
            
        print("分析报告已保存为: interval_censored_survival_analysis_report.md")
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
        data_file = r'c:\Users\Lu\Desktop\最终版本代码\问题二\processed_data.csv'
        model.load_and_prepare_data(data_file)
        
        # 步骤2: 模型拟合（使用对数正态分布）
        model.fit_aft_model(distribution='lognormal')
        
        # 步骤3: BMI分组
        model.perform_bmi_clustering(n_clusters=4)  # 可以调整聚类数
        
        # 步骤4: 最佳时点确定
        model.determine_optimal_timing(success_rate=0.9)
        
        # 步骤5: 结果可视化
        model.visualize_results()
        
        # 步骤6: 生成报告
        model.generate_report()
        
        print("\n=== 分析完成 ===")
        print("所有结果文件已保存到当前目录")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()