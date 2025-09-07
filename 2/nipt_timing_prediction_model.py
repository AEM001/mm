import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class NIPTTimingPredictionModel:
    def __init__(self):
        self.data = None
        self.model_params = None
        self.bmi_boundaries = [-3.9, -1.4, -0.2, 1.5, 1.6, 4.8]  # 指定的区间端点
        self.bmi_groups = None
        self.optimal_timings = None
        self.time_mean = 16.846
        self.time_std = 4.076
        self.bmi_mean = 24.5
        self.bmi_std = 4.2
    
    def standardized_to_original_time(self, standardized_time):
        return standardized_time * self.time_std + self.time_mean
    
    def original_to_standardized_time(self, original_time):
        return (original_time - self.time_mean) / self.time_std
    
    def standardized_to_original_bmi(self, standardized_bmi):
        return standardized_bmi * self.bmi_std + self.bmi_mean
    
    def original_to_standardized_bmi(self, original_bmi):
        return (original_bmi - self.bmi_mean) / self.bmi_std
        
    def load_data_and_model(self, data_path):
        self.data = pd.read_csv(data_path)
        
        required_cols = ['孕妇代码', 'Y染色体浓度', 'BMI_标准化']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            return False
        
        self.data = self.data.dropna(subset=required_cols)
        
        self.model_params = {
            'distribution': 'lognormal',
            'beta0': -2.1300,   
            'beta_bmi': 0.0797, 
            'beta_age': 0.0274, 
            'sigma': 0.5770     
        }
        
        return True
    
    def assign_bmi_groups_by_boundaries(self):
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
            
            bmi_groups.append(group_idx)
        
        self.data['BMI组别'] = bmi_groups
        
        self.data['BMI_原始'] = self.standardized_to_original_bmi(self.data['BMI_标准化'])
        
        group_stats = []
        for group_id in sorted(self.data['BMI组别'].unique()):
            group_data = self.data[self.data['BMI组别'] == group_id]
            
            if len(group_data) == 0:
                continue
            
            if group_id == 0:
                left_bound = "-∞"
                right_bound = boundaries[0]
            elif group_id == len(boundaries):
                left_bound = boundaries[-1]
                right_bound = "+∞"
            else:
                left_bound = boundaries[group_id - 1]
                right_bound = boundaries[group_id]
            
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
        
        return self.bmi_groups
    
    def predict_survival_function(self, bmi_values, age_values=None, time_points=None):
        if self.model_params is None:
            raise ValueError("模型参数尚未设置")
        
        if age_values is None:
            if '年龄_标准化' in self.data.columns:
                age_values = np.full_like(bmi_values, self.data['年龄_标准化'].mean())
            else:
                age_values = np.zeros_like(bmi_values)
            
        if time_points is None:
            time_points = np.linspace(-2, 3, 100)
        
        linear_pred = (self.model_params['beta0'] + 
                      self.model_params['beta_bmi'] * bmi_values + 
                      self.model_params['beta_age'] * age_values)
        
        survival_probs = np.zeros((len(bmi_values), len(time_points)))
        
        distribution = self.model_params['distribution']
        sigma = self.model_params['sigma']
        
        for i, t in enumerate(time_points):
            if distribution == 'weibull':
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
            elif distribution == 'lognormal':
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = 1 - stats.norm.cdf(standardized_time)
            else:
                standardized_time = (t - linear_pred) / sigma
                survival_probs[:, i] = np.exp(-np.exp(-standardized_time))
        
        return survival_probs, time_points
    
    def determine_optimal_timing_for_groups(self, success_rate=0.9):
        optimal_timings = []
        
        for _, group in self.bmi_groups.iterrows():
            group_id = group['组别']
            group_data = self.data[self.data['BMI组别'] == group_id]
            if len(group_data) == 0:
                continue
            
            bmi_mean = group['BMI均值_标准化']
            
            if '年龄_标准化' in group_data.columns:
                age_mean = group_data['年龄_标准化'].mean()
            else:
                age_mean = 0.0
            
            standardized_time_points = np.linspace(-2, 3, 200)
            survival_probs, time_points = self.predict_survival_function(
                bmi_values=np.array([bmi_mean]),
                age_values=np.array([age_mean]),
                time_points=standardized_time_points
            )
            
            success_probs = 1 - survival_probs[0]
            
            key_times_std = [-1, 0, 1, 2]
            for kt in key_times_std:
                if kt >= time_points.min() and kt <= time_points.max():
                    idx = np.argmin(np.abs(time_points - kt))
                    original_week = self.standardized_to_original_time(kt)
            
            target_indices = np.where(success_probs >= success_rate)[0]
            
            if len(target_indices) > 0:
                optimal_time_std = time_points[target_indices[0]]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
                actual_success_rate = success_probs[target_indices[0]]
            else:
                max_idx = np.argmax(success_probs)
                optimal_time_std = time_points[max_idx]
                optimal_time_original = self.standardized_to_original_time(optimal_time_std)
                actual_success_rate = np.max(success_probs)
            
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
        
        return self.optimal_timings
    
    def _generate_timing_recommendation(self, optimal_time, actual_success_rate, target_success_rate):
        if actual_success_rate >= target_success_rate:
            if optimal_time <= 12:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，成功率高且风险低"
            elif optimal_time <= 20:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，成功率达标"
            else:
                return f"建议在{optimal_time:.1f}周进行NIPT检测，但需注意较晚检测的风险"
        else:
            return f"该组难以达到{target_success_rate*100}%成功率，建议在{optimal_time:.1f}周检测并考虑其他检测方法"
    

def main():
    data_path = r"c:\Users\Lu\Desktop\问题2&3代码\问题二\processed_data.csv"
    
    predictor = NIPTTimingPredictionModel()
    
    results = predictor.run_prediction(data_path)
    
    return results

if __name__ == "__main__":
    results = main()