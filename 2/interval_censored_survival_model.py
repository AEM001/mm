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

import platform

def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['STSong', 'Songti SC', 'STHeiti']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

class IntervalCensoredSurvivalModel:
    def __init__(self):
        self.data = None
        self.model_params = None
        self.bmi_groups = None
        self.optimal_timings = None
        self.time_mean = 16.846
        self.time_std = 4.076
    
    def standardized_to_original_time(self, standardized_time):
        return standardized_time * self.time_std + self.time_mean
    
    def original_to_standardized_time(self, original_time):
        return (original_time - self.time_mean) / self.time_std
        
    def load_and_prepare_data(self, file_path):
        df = pd.read_csv(file_path)
        required_cols = ['孕妇代码', 'Y染色体浓度']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None
            
        df_clean = df.dropna(subset=required_cols)
        
        df = df_clean
        
        interval_data = []
        
        bmi_col = 'BMI_标准化' if 'BMI_标准化' in df.columns else None
        age_col = '年龄_标准化' if '年龄_标准化' in df.columns else None
        week_col = '孕周_标准化' if '孕周_标准化' in df.columns else None
        
        for woman_code, woman_data in df.groupby('孕妇代码'):
            if week_col and week_col in woman_data.columns:
                woman_data = woman_data.sort_values(week_col).reset_index(drop=True)
            else:
                woman_data = woman_data.reset_index(drop=True)
            
            bmi = woman_data[bmi_col].iloc[0] if bmi_col else 0.0
            age = woman_data[age_col].iloc[0] if age_col and age_col in woman_data.columns else 0.0
            
            y_concentrations = woman_data['Y染色体浓度'].values
            
            if week_col and week_col in woman_data.columns:
                time_points = woman_data[week_col].values
            else:
                time_points = np.arange(len(woman_data))
            
            threshold_percentage = 0.04
            threshold = np.log(threshold_percentage / (1 - threshold_percentage))
            qualified_indices = np.where(y_concentrations >= threshold)[0]
            
            if len(qualified_indices) == 0:
                censoring_type = 'right'
                left_time = time_points[-1]
                right_time = np.inf
                event_time = time_points[-1]
                
            elif qualified_indices[0] == 0:
                censoring_type = 'left'
                left_time = 0
                right_time = time_points[0]
                event_time = time_points[0]
                
            else:
                censoring_type = 'interval'
                first_qualified_idx = qualified_indices[0]
                left_time = time_points[first_qualified_idx - 1]
                right_time = time_points[first_qualified_idx]
                event_time = (left_time + right_time) / 2
                
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
        
        return self.data
    
    def fit_aft_model(self, distribution='weibull'):
        X = self.data[['BMI', '年龄']].fillna(0.0)
        y = self.data['事件时间估计'].values
        
        y = np.maximum(y, 0.1)
        
        X_scaled = X.values
        
        if distribution == 'weibull':
            def weibull_log_likelihood(params):
                beta0, beta1, beta2, sigma = params
                
                if sigma <= 0:
                    return 1e10
                
                try:
                    linear_pred = beta0 + beta1 * X_scaled[:, 0] + beta2 * X_scaled[:, 1]
                    
                    log_y = np.log(np.maximum(y, 1e-10))
                    standardized_residuals = (log_y - linear_pred) / sigma
                    
                    if np.any(np.isnan(standardized_residuals)) or np.any(np.isinf(standardized_residuals)):
                        return 1e10
                    
                    exp_neg_resid = np.exp(-standardized_residuals)
                    exp_neg_resid = np.minimum(exp_neg_resid, 1e10)
                    
                    log_likelihood = np.sum(
                        -standardized_residuals - exp_neg_resid - np.log(sigma)
                    )
                    
                    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                        return 1e10
                    
                    return -log_likelihood
                    
                except Exception as e:
                    return 1e10
            
            from scipy.optimize import minimize
            
            y_mean = np.mean(y)
            y_std = np.std(y)
            initial_params = [np.log(y_mean), 0.0, 0.0, max(0.5, y_std/y_mean)]
            
            bounds = [(-10, 10), (-5, 5), (-5, 5), (0.1, 10)]
            
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
            def lognormal_log_likelihood(params):
                beta0, beta1, beta2, sigma = params
                
                if sigma <= 0:
                    return 1e10
                
                try:
                    linear_pred = beta0 + beta1 * X_scaled[:, 0] + beta2 * X_scaled[:, 1]
                    
                    log_y = np.log(np.maximum(y, 1e-10))
                    standardized_residuals = (log_y - linear_pred) / sigma
                    
                    if np.any(np.isnan(standardized_residuals)) or np.any(np.isinf(standardized_residuals)):
                        return 1e10
                    
                    log_likelihood = np.sum(
                        -0.5 * standardized_residuals**2 - 0.5 * np.log(2 * np.pi) - np.log(sigma)
                    )
                    
                    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                        return 1e10
                    
                    return -log_likelihood
                    
                except Exception as e:
                    return 1e10
            
            from scipy.optimize import minimize
            
            y_mean = np.mean(y)
            y_std = np.std(y)
            initial_params = [np.log(y_mean), 0.0, 0.0, max(0.5, y_std/y_mean)]
            
            bounds = [(-10, 10), (-5, 5), (-5, 5), (0.1, 10)]
            
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
            
            if beta1 > 0:
                pass
            
        else:
            from sklearn.linear_model import LinearRegression
            
            try:
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
                
            except Exception as e:
                self.model_params = None
            return None
                
        return self.model_params
    
    def predict_survival_function(self, bmi_values, age_values=None, time_points=None):
        if self.model_params is None:
            raise ValueError("模型尚未拟合，请先调用fit_aft_model()")
        
        if age_values is None:
            age_values = np.full_like(bmi_values, self.data['年龄'].mean())
            
        if time_points is None:
            min_time = self.data['事件时间估计'].min()
            max_time = self.data['事件时间估计'].max()
            time_points = np.linspace(min_time, max_time, 100)
            
        X = np.column_stack([bmi_values, age_values])
        X_scaled = X
        
        linear_pred = (self.model_params['beta0'] + 
                      self.model_params['beta_bmi'] * X_scaled[:, 0] + 
                      self.model_params['beta_age'] * X_scaled[:, 1])
        
        survival_probs = np.zeros((len(bmi_values), len(time_points)))
        
        distribution = self.model_params['distribution']
        
        for i, t in enumerate(time_points):
            if distribution == 'weibull':
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            elif distribution == 'lognormal':
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = 1 - stats.norm.cdf(standardized_time)
            else:
                standardized_time = (t - linear_pred) / self.model_params['sigma']
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            
        return survival_probs, time_points
    
    def perform_bmi_clustering(self, n_clusters=None):
        features = np.column_stack([
            self.data['BMI'].values,
            self.data['事件时间估计'].values
        ])
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if n_clusters is None:
            inertias = []
            K_range = range(2, 8)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features_scaled)
                inertias.append(kmeans.inertia_)
            
            diffs = np.diff(inertias)
            diff_ratios = diffs[:-1] / diffs[1:]
            optimal_k = K_range[np.argmax(diff_ratios) + 1]
            
        else:
            optimal_k = n_clusters
            
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        
        bmi_centers = centers_original[:, 0]
        sorted_indices = np.argsort(bmi_centers)
        
        new_labels = np.zeros_like(cluster_labels)
        for new_idx, old_idx in enumerate(sorted_indices):
            new_labels[cluster_labels == old_idx] = new_idx
            
        self.data['BMI聚类'] = new_labels
        
        bmi_groups = []
        for i in range(optimal_k):
            group_data = self.data[self.data['BMI聚类'] == i]
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            time_min = group_data['事件时间估计'].min()
            time_max = group_data['事件时间估计'].max()
            time_mean = group_data['事件时间估计'].mean()
            
            time_min_weeks = self.standardized_to_original_time(time_min)
            time_max_weeks = self.standardized_to_original_time(time_max)
            time_mean_weeks = self.standardized_to_original_time(time_mean)
            
            bmi_groups.append({
                '组别': i + 1,
                'BMI最小值': bmi_min,
                'BMI最大值': bmi_max,
                'BMI均值': bmi_mean,
                '达标时刻最小值(周)': time_min_weeks,
                '达标时刻最大值(周)': time_max_weeks,
                '达标时刻均值(周)': time_mean_weeks,
                '样本数': len(group_data),
                'BMI区间': f'[{bmi_min:.1f}, {bmi_max:.1f}]',
                '达标时刻区间': f'[{time_min_weeks:.1f}, {time_max_weeks:.1f}]周'
            })
            
        self.bmi_groups = pd.DataFrame(bmi_groups)
        
        return self.bmi_groups
    
    def determine_optimal_timing(self, success_rate=0.9):
        
        optimal_timings = []
        
        for _, group in self.bmi_groups.iterrows():
            group_id = group['组别'] - 1
            group_data = self.data[self.data['BMI聚类'] == group_id]
            bmi_mean = group['BMI均值']
            age_mean = group_data['年龄'].mean()
            
            
            standardized_time_points = np.linspace(-2, 3, 100)
            survival_probs, time_points = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([age_mean]),

            )
            
            success_probs = 1 - survival_probs[0]
            
            target_indices = np.where(success_probs >= success_rate)[0]
            
            if len(target_indices) > 0:
                optimal_time_std = time_points[target_indices[0]]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
            else:
                optimal_time_std = time_points[np.argmax(success_probs)]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
            
            risk_level = "低风险" if optimal_time_original <= 12 else "中风险" if optimal_time_original <= 27 else "高风险"
            
            optimal_timings.append({
                '组别': group['组别'],
                'BMI区间': group['BMI区间'],
                'BMI均值': bmi_mean,
                '样本数': group['样本数'],
                '最佳时点': optimal_time_original,
                '预期成功率': success_probs[target_indices[0]] if len(target_indices) > 0 else np.max(success_probs),
                '风险等级': risk_level
            })
            
        self.optimal_timings = pd.DataFrame(optimal_timings)
        
        return self.optimal_timings
    
    def visualize_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1 = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.bmi_groups)))
        
        for i, (_, group) in enumerate(self.bmi_groups.iterrows()):
            group_data = self.data[self.data['BMI聚类'] == i]
            time_weeks = self.standardized_to_original_time(group_data['事件时间估计'])
            ax1.scatter(group_data['BMI'], time_weeks, 
                       c=[colors[i]], label=f'第{i+1}组 (n={len(group_data)})', alpha=0.7, s=50)
            
        ax1.set_xlabel('BMI (标准化值)')
        ax1.set_ylabel('Y染色体浓度达标时间 (周)')
        ax1.set_title('BMI和达标时间二维聚类结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        standardized_time_points = np.linspace(-2, 3, 100)
        original_time_points = self.standardized_to_original_time(standardized_time_points)
        
        for i, (_, group) in enumerate(self.bmi_groups.iterrows()):
            bmi_mean = group['BMI均值']
            survival_probs, _ = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([group_data['年龄'].mean()]),
                time_points=standardized_time_points
            )
            
            ax2.plot(original_time_points, survival_probs[0], 
                    color=colors[i], linewidth=2, 
                    label=f'第{i+1}组 (BMI={bmi_mean:.1f})')
            
        ax2.set_ylabel('生存概率 S(t)')
        ax2.set_title('各BMI组生存曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        groups = [f"第{row['组别']}组\n{row['BMI区间']}" for _, row in self.optimal_timings.iterrows()]
        timings = self.optimal_timings['最佳时点'].values
        
        bars = ax3.bar(groups, timings, color=colors[:len(groups)], alpha=0.7)
        ax3.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周风险阈值')
        ax3.set_ylabel('最佳NIPT时点 (周)')
        ax3.set_title('各BMI组最佳NIPT时点')
        ax3.legend()
        
        for bar, timing in zip(bars, timings):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{timing:.1f}', ha='center', va='bottom')
        
        ax4 = axes[1, 1]
        success_rates = self.optimal_timings['预期成功率'].values * 100
        
        bars2 = ax4.bar(groups, success_rates, color=colors[:len(groups)], alpha=0.7)
        ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='目标成功率90%')
        ax4.set_ylabel('预期成功率 (%)')
        ax4.set_title('各BMI组预期成功率')
        ax4.legend()
        
        for bar, rate in zip(bars2, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('interval_censored_survival_analysis.png', dpi=300, bbox_inches='tight')    

def main():
    model = IntervalCensoredSurvivalModel()
    
    try:
        data_file = r'c:\Users\Lu\Desktop\问题2&3代码\问题二\processed_data.csv'
        model.load_and_prepare_data(data_file)
        
        model.fit_aft_model(distribution='lognormal')
        
        model.perform_bmi_clustering(n_clusters=4)
        
        model.determine_optimal_timing(success_rate=0.9)
        
        model.visualize_results()
                
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()