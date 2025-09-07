#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_chinese_font import set_chinese_font
set_chinese_font()
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion

with pandas2ri.converter.context():
    pass

mgcv = importr('mgcv')
base = importr('base')
stats_r = importr('stats')
R_AVAILABLE = True


class GAMMYChromosomePredictor:
    
    def __init__(self, use_r_gamm=True):
        if not R_AVAILABLE:
            raise RuntimeError("R/mgcv (via rpy2) 不可用：本项目强制使用 R GAMM")
        self.use_r_gamm = True
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化',
                              'Y染色体Z值_重新标准化', '过滤读段比例_标准化', 'GC含量_标准化',
                              '唯一比对读段比例_标准化', '重复读段比例_标准化', '比对比例_标准化', '总读段数_标准化']
        self.significant_features = None
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path, encoding='utf-8')
        
        return df
    
    def extract_target_variable(self, df):
        df_copy = df.copy()

        target_data = []
        达标_count = 0
        未达标_count = 0

        for patient_id in df_copy['孕妇代码'].unique():
            patient_data = df_copy[df_copy['孕妇代码'] == patient_id].copy()
            patient_data = patient_data.sort_values('孕周_标准化')

            y_std = patient_data['Y染色体浓度']
            patient_data['Y_concentration_estimated'] = ((y_std + 3.6) / 2.0) * 8.0

            达标记录 = patient_data[patient_data['Y_concentration_estimated'] >= 4.0]

            if len(达标记录) > 0:
                target_week = 达标记录.iloc[0]['孕周_标准化']
                is_censored = False
                达标_count += 1
            else:
                target_week = patient_data.iloc[-1]['孕周_标准化']
                is_censored = True
                未达标_count += 1

            first_record = patient_data.iloc[0]
            
            target_data.append({
                '孕妇代码': patient_id,
                '达标孕周': target_week,
                'BMI_标准化': first_record['BMI_标准化'],
                '年龄_标准化': first_record['年龄_标准化'],
                '怀孕次数_标准化': first_record['怀孕次数_标准化'],
                '生产次数_标准化': first_record['生产次数_标准化'],
                'Y染色体Z值_重新标准化': first_record['Y染色体Z值_重新标准化'],
                '过滤读段比例_标准化': first_record['过滤读段比例_标准化'],
                'GC含量_标准化': first_record['GC含量_标准化'],
                '唯一比对读段比例_标准化': first_record['唯一比对读段比例_标准化'],
                '重复读段比例_标准化': first_record['重复读段比例_标准化'],
                '比对比例_标准化': first_record['比对比例_标准化'],
                '总读段数_标准化': first_record['总读段数_标准化'],
                'is_censored': is_censored,
                '检测次数': len(patient_data),
                '最大Y浓度': patient_data['Y_concentration_estimated'].max()
            })
        
        result_df = pd.DataFrame(target_data)
        
        
        return result_df
    
    def test_feature_significance(self, X, y, alpha=0.05):
        
        if self.use_r_gamm and R_AVAILABLE:
            return self._test_significance_r_gamm(X, y, alpha)
        else:
            raise RuntimeError("Python替代显著性检验已禁用；请确保R/mgcv可用")
    
    def _test_significance_r_gamm(self, X, y, alpha=0.05):
        n_features = X.shape[1]
        py_df = pd.DataFrame({'y': pd.Series(y).reset_index(drop=True)})
        for i in range(n_features):
            xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            py_df[f'x{i+1}'] = pd.Series(xi).reset_index(drop=True)
        with pandas2ri.converter.context():
            r_data = pandas2ri.py2rpy(py_df)

        n_features = X.shape[1]
        smooth_terms = []
        linear_terms = []
        for i in range(n_features):
            xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            n_unique = len(pd.Series(xi).dropna().unique())
            if n_unique >= 4:
                k_i = min(5, max(2, int(n_unique) - 1))
                smooth_terms.append(f"s(x{i+1}, k={k_i})")
            else:
                linear_terms.append(f"x{i+1}")
        rhs_terms = []
        if smooth_terms:
            rhs_terms.append(' + '.join(smooth_terms))
        if linear_terms:
            rhs_terms.append(' + '.join(linear_terms))
        rhs = ' + '.join([t for t in rhs_terms if t]) if rhs_terms else '1'
        formula = f"y ~ {rhs}"

        full_model = mgcv.gam(robjects.Formula(formula), data=r_data, method="REML")

        summary_result = base.summary(full_model)

        significant_features = []

        for i, feature_name in enumerate(self.feature_names[:n_features]):
            xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            n_unique = len(pd.Series(xi).dropna().unique())
            if n_unique >= 4:
                k_i = min(5, max(2, int(n_unique) - 1))
                single_formula = f"y ~ s(x{i+1}, k={k_i})"
            else:
                single_formula = f"y ~ x{i+1}"
            single_model = mgcv.gam(robjects.Formula(single_formula), data=r_data, method="REML")
            single_summary = base.summary(single_model)

            dev_expl = single_summary.rx2('dev.expl')[0]
            if dev_expl > 0.05:
                significant_features.append(feature_name)

        return significant_features
    
    def exploratory_data_analysis(self, df):
        达标数据 = df[~df['is_censored']]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Y染色体达标孕周探索性数据分析', fontsize=16)

        if len(达标数据) > 0:
            axes[0, 0].hist(达标数据['达标孕周'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('达标孕周分布')
            axes[0, 0].set_xlabel('标准化孕周')
            axes[0, 0].set_ylabel('频数')

        if len(达标数据) > 0:
            axes[0, 1].scatter(达标数据['BMI_标准化'], 达标数据['达标孕周'], alpha=0.6, color='coral')
            axes[0, 1].set_title('BMI vs 达标孕周')
            axes[0, 1].set_xlabel('BMI (标准化)')
            axes[0, 1].set_ylabel('达标孕周 (标准化)')

        if len(达标数据) > 0:
            axes[0, 2].scatter(达标数据['年龄_标准化'], 达标数据['达标孕周'], alpha=0.6, color='lightgreen')
            axes[0, 2].set_title('年龄 vs 达标孕周')
            axes[0, 2].set_xlabel('年龄 (标准化)')
            axes[0, 2].set_ylabel('达标孕周 (标准化)')

        numeric_cols = ['达标孕周', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']
        if len(达标数据) > 0:
            corr_matrix = 达标数据[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('特征相关性矩阵')

        axes[1, 1].hist(df['检测次数'], bins=range(1, df['检测次数'].max()+2), alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('每个孕妇检测次数分布')
        axes[1, 1].set_xlabel('检测次数')
        axes[1, 1].set_ylabel('孕妇数量')

        达标状态 = df['is_censored'].value_counts()
        labels = ['达标', '未达标']
        colors = ['lightblue', 'lightcoral']
        axes[1, 2].pie([len(df[~df['is_censored']]), len(df[df['is_censored']])],
                      labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('达标状态分布')

        plt.tight_layout()
        plt.savefig('./gamm_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.results['eda_stats'] = {
            '总样本数': len(df),
            '达标样本数': len(达标数据),
            '达标比例': len(达标数据) / len(df) * 100,
            '平均检测次数': df['检测次数'].mean(),
            '达标孕周统计': 达标数据['达标孕周'].describe().to_dict() if len(达标数据) > 0 else None
        }
    
    def fit_gamm_model(self, X, y, patient_ids=None, test_significance=True):
        if test_significance:
            self.significant_features = self.test_feature_significance(X, y)

            if len(self.significant_features) == 0:
                self.significant_features = self.feature_names[:X.shape[1]]
            else:
                significant_indices = [i for i, name in enumerate(self.feature_names) if name in self.significant_features]
                X = X.iloc[:, significant_indices] if hasattr(X, 'iloc') else X[:, significant_indices]
        else:
            self.significant_features = self.feature_names[:X.shape[1]]
        
        if self.use_r_gamm:
            self._fit_r_gamm(X, y, patient_ids)
        else:
            raise RuntimeError("Python替代建模已禁用；请确保R/mgcv可用")
    
    def _fit_r_gamm(self, X, y, patient_ids=None):
        n_features = X.shape[1]
        y_series = pd.Series(y).reset_index(drop=True)
        col_map = {
            'y': robjects.FloatVector(pd.to_numeric(y_series, errors='coerce').astype(float).tolist())
        }
        for i in range(n_features):
            xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            xi_series = pd.Series(xi).reset_index(drop=True)
            col_map[f'x{i+1}'] = robjects.FloatVector(pd.to_numeric(xi_series, errors='coerce').astype(float).tolist())
        r_data = robjects.DataFrame(col_map)

        smooth_terms = []
        linear_terms = []
        for i in range(n_features):
            xi = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            n_unique = len(pd.Series(xi).dropna().unique())
            if n_unique >= 4:
                k_i = min(5, max(2, int(n_unique) - 1))
                smooth_terms.append(f"s(x{i+1}, k={k_i})")
            else:
                linear_terms.append(f"x{i+1}")

        rhs_terms = []
        if smooth_terms:
            rhs_terms.append(' + '.join(smooth_terms))
        if linear_terms:
            rhs_terms.append(' + '.join(linear_terms))
        rhs = ' + '.join([t for t in rhs_terms if t]) if rhs_terms else '1'
        formula = f"y ~ {rhs}"

        self.model = mgcv.gam(robjects.Formula(formula), data=r_data, method="REML")

        summary_result = base.summary(self.model)

        self.results['model_type'] = 'R_GAMM'
        self.results['r_squared'] = summary_result.rx2('r.sq')[0]
        self.results['deviance_explained'] = summary_result.rx2('dev.expl')[0]
        self.results['significant_features'] = self.significant_features
        self.results['n_features'] = n_features
    
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.use_r_gamm:
            return self._predict_r_gamm(X_new)
        else:
            raise RuntimeError("Python替代预测已禁用；请确保R/mgcv可用")
    
    def _predict_r_gamm(self, X_new):
        data_dict = {}
        for i, col in enumerate(self.feature_names):
            data_dict[f'x{i+1}'] = X_new.iloc[:, i] if hasattr(X_new, 'iloc') else X_new[:, i]

        with pandas2ri.converter.context():
            r_newdata = robjects.DataFrame(data_dict)

        predictions = stats_r.predict(self.model, newdata=r_newdata)

        return np.array(predictions)
    
    def _predict_python_alternative(self, X_new):
        rf_pred = self.model['rf'].predict(X_new)

        X_new_poly = self.model['poly_features'].transform(X_new)
        ridge_pred = self.model['ridge'].predict(X_new_poly)

        ensemble_pred = 0.7 * rf_pred + 0.3 * ridge_pred

        return ensemble_pred
    
    def cross_validate(self, X, y, cv=5):
        if not self.use_r_gamm:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestRegressor

            rf_scores = cross_val_score(
                RandomForestRegressor(n_estimators=100, random_state=42),
                X, y, cv=cv, scoring='r2'
            )

            cv_results = {
                'mean_r2': rf_scores.mean(),
                'std_r2': rf_scores.std(),
                'scores': rf_scores
            }

            self.results['cv_results'] = cv_results
            return cv_results

        else:
            return None
    
    def plot_model_diagnostics(self, X, y):
        y_pred = self.predict(X)
        residuals = y - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAMM模型诊断图', fontsize=16)

        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('拟合值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')

        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q图 (正态性检验)')

        axes[1, 0].scatter(y, y_pred, alpha=0.6, color='green')
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 0].set_xlabel('实际值')
        axes[1, 0].set_ylabel('预测值')
        axes[1, 0].set_title('预测值 vs 实际值')

        axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('残差分布')

        plt.tight_layout()
        plt.savefig('./gamm_model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        
        self.results['diagnostics'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        }
    
    def generate_predictions_for_new_patients(self, patient_profiles):
        if self.significant_features is not None:
            available_features = [f for f in self.significant_features if f in patient_profiles.columns]
            if len(available_features) == 0:
                available_features = [f for f in self.feature_names if f in patient_profiles.columns]
        else:
            available_features = [f for f in self.feature_names if f in patient_profiles.columns]

        predictions = self.predict(patient_profiles[available_features])

        results_df = patient_profiles.copy()
        results_df['预测达标孕周'] = predictions

        def risk_stratification(week):
            if week < -0.5:
                return '低风险'
            elif week < 0.5:
                return '中风险'
            else:
                return '高风险'

        results_df['风险等级'] = results_df['预测达标孕周'].apply(risk_stratification)

        prediction_std = self.results.get('rmse', 0.5)
        results_df['预测下界'] = predictions - 1.96 * prediction_std
        results_df['预测上界'] = predictions + 1.96 * prediction_std

        return results_df
    
    def plot_partial_effects(self, X, y, output_dir='./'):
        features_to_plot = self.significant_features if hasattr(self, 'significant_features') and self.significant_features else self.feature_names[:X.shape[1]]
        n_features = len(features_to_plot)

        if n_features == 0:
            return

        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('各自变量偏效应图', fontsize=16, y=0.98)
        
        for idx, feature_name in enumerate(features_to_plot):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            feature_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else idx

            if feature_idx < X.shape[1]:
                feature_values = X.iloc[:, feature_idx] if hasattr(X, 'iloc') else X[:, feature_idx]

                feature_range = np.linspace(feature_values.min(), feature_values.max(), 100)

                if self.use_r_gamm and self.model is not None:
                    partial_effects = self._compute_partial_effects_r(feature_idx, feature_range, X)
                else:
                    partial_effects = self._compute_partial_effects_python(feature_idx, feature_range, X, y)
                
                ax.plot(feature_range, partial_effects, 'b-', linewidth=2, label='偏效应')

                ax.scatter(feature_values, y, alpha=0.3, s=20, color='gray', label='观测值')

                from scipy.interpolate import UnivariateSpline
                sorted_indices = np.argsort(feature_values)
                sorted_x = feature_values.iloc[sorted_indices] if hasattr(feature_values, 'iloc') else feature_values[sorted_indices]
                sorted_y = y.iloc[sorted_indices] if hasattr(y, 'iloc') else y[sorted_indices]

                spline = UnivariateSpline(sorted_x, sorted_y, s=len(sorted_x)*0.1)
                smooth_y = spline(feature_range)
                ax.plot(feature_range, smooth_y, 'r--', alpha=0.7, linewidth=1.5, label='平滑趋势')
                
                ax.set_xlabel(feature_name)
                ax.set_ylabel('达标孕周 (标准化)')
                ax.set_title(f'{feature_name}的偏效应')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                if hasattr(self, 'significance_results'):
                    sig_row = self.significance_results[self.significance_results['特征名称'] == feature_name]
                    if not sig_row.empty:
                        corr = sig_row.iloc[0]['相关系数']
                        p_val = sig_row.iloc[0]['p值']
                        ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}',
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)

        plt.tight_layout()

        output_path = f"{output_dir}gamm_partial_effects.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"偏效应图像已保存到: {output_path}")
        plt.show()

        return output_path
    
    def _compute_partial_effects_r(self, feature_idx, feature_range, X):
        n_points = len(feature_range)
        pred_data = {}

        for i in range(X.shape[1]):
            if i == feature_idx:
                pred_data[f'x{i+1}'] = feature_range
            else:
                mean_val = X.iloc[:, i].mean() if hasattr(X, 'iloc') else X[:, i].mean()
                pred_data[f'x{i+1}'] = [mean_val] * n_points

        with pandas2ri.converter.context():
            r_pred_data = robjects.DataFrame(pred_data)

        predictions = mgcv.predict_gam(self.model, r_pred_data)

        return np.array(predictions)
    
    def _compute_partial_effects_python(self, feature_idx, feature_range, X, y):
        if self.model is None:
            feature_values = X.iloc[:, feature_idx] if hasattr(X, 'iloc') else X[:, feature_idx]

            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(y, feature_values, frac=0.3)

            from scipy.interpolate import interp1d
            interp_func = interp1d(smoothed[:, 0], smoothed[:, 1],
                                 kind='linear', fill_value='extrapolate')
            return interp_func(feature_range)

        elif isinstance(self.model, dict):
            n_points = len(feature_range)
            pred_X = np.zeros((n_points, X.shape[1]))

            for i in range(X.shape[1]):
                if i == feature_idx:
                    pred_X[:, i] = feature_range
                else:
                    pred_X[:, i] = X.iloc[:, i].mean() if hasattr(X, 'iloc') else X[:, i].mean()

            rf_pred = self.model['rf'].predict(pred_X)

            pred_X_poly = self.model['poly_features'].transform(pred_X)
            ridge_pred = self.model['ridge'].predict(pred_X_poly)

            return 0.7 * rf_pred + 0.3 * ridge_pred

        else:
            return np.zeros_like(feature_range)
    
    def save_results(self, output_path):
        print(f"\n保存结果到: {output_path}")

        if hasattr(self, 'significance_results'):
            significance_table_path = output_path.replace('.md', '_significance_table.csv')
            self.significance_results.to_csv(significance_table_path, index=False, encoding='utf-8-sig')
            print(f"显著性检验结果表格已保存到: {significance_table_path}")

            excel_path = significance_table_path.replace('.csv', '.xlsx')
            self.significance_results.to_excel(excel_path, index=False)
            print(f"显著性检验结果表格(Excel)已保存到: {excel_path}")


        print("结果保存完成")

def main():
    predictor = GAMMYChromosomePredictor(use_r_gamm=True)

    data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    df = predictor.load_and_preprocess_data(data_path)

    target_df = predictor.extract_target_variable(df)

    predictor.exploratory_data_analysis(target_df)

    modeling_data = target_df[~target_df['is_censored']].copy()

    if len(modeling_data) < 10:
        return

    X = modeling_data[predictor.feature_names]
    y = modeling_data['达标孕周']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictor.fit_gamm_model(X_train, y_train)

    predictor.cross_validate(X_train, y_train, cv=5)

    predictor.plot_model_diagnostics(X_train, y_train)

    predictor.plot_partial_effects(X_train, y_train)

    if len(X_test) > 0:
        y_pred_test = predictor.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    example_patients = pd.DataFrame({
        'BMI_标准化': [-1.0, 0.0, 1.0],
        '年龄_标准化': [0.0, 0.0, 0.0],
        '怀孕次数_标准化': [0.0, 0.0, 0.0],
        '生产次数_标准化': [0.0, 0.0, 0.0],
        'Y染色体Z值_重新标准化': [0.0, 0.0, 0.0],
        '过滤读段比例_标准化': [0.0, 0.0, 0.0],
        'GC含量_标准化': [0.0, 0.0, 0.0],
        '唯一比对读段比例_标准化': [0.0, 0.0, 0.0],
        '重复读段比例_标准化': [0.0, 0.0, 0.0],
        '比对比例_标准化': [0.0, 0.0, 0.0],
        '总读段数_标准化': [0.0, 0.0, 0.0]
    })

    predictions_df = predictor.generate_predictions_for_new_patients(example_patients)

    predictor.save_results('./gamm_analysis_report.md')


if __name__ == "__main__":
    main()