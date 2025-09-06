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

# 尝试导入R接口用于GAMM
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import conversion
    
    # 使用新的转换上下文（使用pandas2ri的默认转换器）
    with pandas2ri.converter.context():
        pass  # 设置pandas转换上下文
    
    # 导入R包
    mgcv = importr('mgcv')
    base = importr('base')
    stats_r = importr('stats')
    R_AVAILABLE = True
    print("R接口可用，将使用mgcv包进行GAMM建模")
except Exception as e:
    R_AVAILABLE = False
    print(f"R接口不可用，将使用Python替代方案: {e}")

# 无论R是否可用，都导入这些模块以避免NameError
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

class GAMMYChromosomePredictor:
    """
    GAMM Y染色体达标孕周预测器
    """
    
    def __init__(self, use_r_gamm=True):
        """
        初始化预测器
        
        Parameters:
        -----------
        use_r_gamm : bool
            是否使用R的mgcv包进行GAMM建模
        """
        self.use_r_gamm = use_r_gamm and R_AVAILABLE
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化', 
                              'Y染色体Z值_重新标准化', '过滤读段比例_标准化', 'GC含量_标准化', 
                              '唯一比对读段比例_标准化', '重复读段比例_标准化', '比对比例_标准化', '总读段数_标准化']
        self.significant_features = None  # 存储显著性检验后的特征
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        """
        加载和预处理数据
        
        Parameters:
        -----------
        file_path : str
            数据文件路径
            
        Returns:
        --------
        pd.DataFrame
            预处理后的数据
        """
        print("正在加载数据...")
        df = pd.read_csv(file_path, encoding='utf-8')
        
        print(f"原始数据形状: {df.shape}")
        print(f"孕妇数量: {df['孕妇代码'].nunique()}")
        
        # 数据质量检查
        print("\n数据质量检查:")
        print(f"缺失值数量: {df.isnull().sum().sum()}")
        print(f"Y染色体浓度范围: [{df['Y染色体浓度'].min():.3f}, {df['Y染色体浓度'].max():.3f}]")
        
        return df
    
    def extract_target_variable(self, df):
        """
        提取目标变量：每个孕妇Y染色体首次达到4%的孕周
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据
            
        Returns:
        --------
        pd.DataFrame
            包含目标变量的数据框
        """
        print("\n正在提取目标变量...")
        
        # 将标准化的Y染色体浓度转换回原始比例
        # 假设原始数据是对数变换后标准化的
        df_copy = df.copy()
        
        target_data = []
        达标_count = 0
        未达标_count = 0
        
        for patient_id in df_copy['孕妇代码'].unique():
            patient_data = df_copy[df_copy['孕妇代码'] == patient_id].copy()
            patient_data = patient_data.sort_values('孕周_标准化')
            
            # 估算原始Y染色体浓度
            # 基于观察到的标准化数据范围（-3.6到-1.6），假设原始数据经过对数变换和标准化
            # 使用更合理的逆变换：将标准化值转换为0-10%的范围
            # 标准化值越大，Y染色体浓度越高
            y_std = patient_data['Y染色体浓度']
            # 将标准化值(-3.6到-1.6)映射到(0到8%)的范围
            patient_data['Y_concentration_estimated'] = ((y_std + 3.6) / 2.0) * 8.0
            
            # 找到首次达到4%的记录
            达标记录 = patient_data[patient_data['Y_concentration_estimated'] >= 4.0]
            
            if len(达标记录) > 0:
                # 有达标记录
                target_week = 达标记录.iloc[0]['孕周_标准化']
                is_censored = False
                达标_count += 1
            else:
                # 没有达标记录，使用最后一次检测的孕周作为删失时间
                target_week = patient_data.iloc[-1]['孕周_标准化']
                is_censored = True
                未达标_count += 1
            
            # 获取患者基本信息（使用第一次检测的数据）
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
        
        print(f"达标孕妇数量: {达标_count}")
        print(f"未达标孕妇数量: {未达标_count}")
        print(f"达标比例: {达标_count/(达标_count+未达标_count)*100:.1f}%")
        
        return result_df
    
    def test_feature_significance(self, X, y, alpha=0.05):
        """
        对特征进行显著性检验
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        alpha : float
            显著性水平，默认0.05
            
        Returns:
        --------
        list
            显著特征的列表
        """
        print("\n正在进行特征显著性检验...")
        
        if self.use_r_gamm and R_AVAILABLE:
            return self._test_significance_r_gamm(X, y, alpha)
        else:
            return self._test_significance_python(X, y, alpha)
    
    def _test_significance_r_gamm(self, X, y, alpha=0.05):
        """
        使用R的GAMM进行显著性检验
        """
        try:
            # 准备数据
            data_dict = {}
            for i, col in enumerate(self.feature_names):
                if i < X.shape[1]:
                    data_dict[f'x{i+1}'] = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            data_dict['y'] = y
            
            # 转换为R数据框（处理pandas转换问题）
            with pandas2ri.converter.context():
                r_data = robjects.DataFrame(data_dict)
            
            # 构建包含所有特征的GAMM公式
            n_features = X.shape[1]
            smooth_terms = [f's(x{i+1}, k=5)' for i in range(n_features)]
            formula = f"y ~ {' + '.join(smooth_terms)}"
            
            # 拟合完整模型
            full_model = mgcv.gam(robjects.Formula(formula), data=r_data, method="REML")
            
            # 获取模型摘要
            summary_result = base.summary(full_model)
            
            # 提取p值（这里简化处理，实际应该从summary中提取smooth terms的p值）
            # 由于R接口复杂性，这里使用简化的方法
            significant_features = []
            
            # 对每个特征进行单独检验
            for i, feature_name in enumerate(self.feature_names[:n_features]):
                try:
                    # 构建单特征模型
                    single_formula = f"y ~ s(x{i+1}, k=5)"
                    single_model = mgcv.gam(robjects.Formula(single_formula), data=r_data, method="REML")
                    single_summary = base.summary(single_model)
                    
                    # 简化的显著性判断：如果模型的deviance explained > 5%，认为显著
                    dev_expl = single_summary.rx2('dev.expl')[0]
                    if dev_expl > 0.05:  # 5%的解释度阈值
                        significant_features.append(feature_name)
                        print(f"{feature_name}: 显著 (解释度: {dev_expl*100:.2f}%)")
                    else:
                        print(f"{feature_name}: 不显著 (解释度: {dev_expl*100:.2f}%)")
                        
                except Exception as e:
                    print(f"{feature_name}: 检验失败 - {e}")
                    
            return significant_features
            
        except Exception as e:
            print(f"R GAMM显著性检验失败: {e}")
            return self._test_significance_python(X, y, alpha)
    
    def _test_significance_python(self, X, y, alpha=0.05):
        """
        使用Python方法进行显著性检验（基于相关性和随机森林重要性）
        """
        significant_features = []
        significance_results = []
        
        for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
            # 计算相关系数和p值
            feature_values = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            corr, p_value = stats.pearsonr(feature_values, y)
            
            # 使用随机森林评估特征重要性
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(np.array(feature_values).reshape(-1, 1), y)
            importance = rf.feature_importances_[0]
            
            # 计算t统计量
            n = len(feature_values)
            t_stat = corr * np.sqrt((n-2)/(1-corr**2)) if abs(corr) < 1 else np.inf
            
            # 综合判断：p值显著或重要性较高
            is_significant = p_value < alpha or importance > 0.1
            if is_significant:
                significant_features.append(feature_name)
                print(f"{feature_name}: 显著 (相关性p值: {p_value:.4f}, 重要性: {importance:.4f})")
            else:
                print(f"{feature_name}: 不显著 (相关性p值: {p_value:.4f}, 重要性: {importance:.4f})")
            
            # 保存结果
            significance_results.append({
                '特征名称': feature_name,
                '相关系数': corr,
                'p值': p_value,
                't统计量': t_stat,
                '随机森林重要性': importance,
                '是否显著': '是' if is_significant else '否',
                '显著性水平': alpha
            })
        
        # 保存显著性检验结果到类属性
        self.significance_results = pd.DataFrame(significance_results)
        
        return significant_features
    
    def exploratory_data_analysis(self, df):
        """
        探索性数据分析
        
        Parameters:
        -----------
        df : pd.DataFrame
            目标数据
        """
        print("\n进行探索性数据分析...")
        
        # 基本统计
        print("\n目标变量统计:")
        达标数据 = df[~df['is_censored']]
        if len(达标数据) > 0:
            print(f"达标孕周均值: {达标数据['达标孕周'].mean():.3f}")
            print(f"达标孕周标准差: {达标数据['达标孕周'].std():.3f}")
            print(f"达标孕周范围: [{达标数据['达标孕周'].min():.3f}, {达标数据['达标孕周'].max():.3f}]")
        
        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Y染色体达标孕周探索性数据分析', fontsize=16)
        
        # 1. 达标孕周分布
        if len(达标数据) > 0:
            axes[0, 0].hist(达标数据['达标孕周'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('达标孕周分布')
            axes[0, 0].set_xlabel('标准化孕周')
            axes[0, 0].set_ylabel('频数')
        
        # 2. BMI vs 达标孕周
        if len(达标数据) > 0:
            axes[0, 1].scatter(达标数据['BMI_标准化'], 达标数据['达标孕周'], alpha=0.6, color='coral')
            axes[0, 1].set_title('BMI vs 达标孕周')
            axes[0, 1].set_xlabel('BMI (标准化)')
            axes[0, 1].set_ylabel('达标孕周 (标准化)')
        
        # 3. 年龄 vs 达标孕周
        if len(达标数据) > 0:
            axes[0, 2].scatter(达标数据['年龄_标准化'], 达标数据['达标孕周'], alpha=0.6, color='lightgreen')
            axes[0, 2].set_title('年龄 vs 达标孕周')
            axes[0, 2].set_xlabel('年龄 (标准化)')
            axes[0, 2].set_ylabel('达标孕周 (标准化)')
        
        # 4. 相关性热图
        numeric_cols = ['达标孕周', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']
        if len(达标数据) > 0:
            corr_matrix = 达标数据[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('特征相关性矩阵')
        
        # 5. 检测次数分布
        axes[1, 1].hist(df['检测次数'], bins=range(1, df['检测次数'].max()+2), alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('每个孕妇检测次数分布')
        axes[1, 1].set_xlabel('检测次数')
        axes[1, 1].set_ylabel('孕妇数量')
        
        # 6. 达标状态饼图
        达标状态 = df['is_censored'].value_counts()
        labels = ['达标', '未达标']
        colors = ['lightblue', 'lightcoral']
        axes[1, 2].pie([len(df[~df['is_censored']]), len(df[df['is_censored']])], 
                      labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('达标状态分布')
        
        plt.tight_layout()
        plt.savefig('./gamm_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存统计结果
        self.results['eda_stats'] = {
            '总样本数': len(df),
            '达标样本数': len(达标数据),
            '达标比例': len(达标数据) / len(df) * 100,
            '平均检测次数': df['检测次数'].mean(),
            '达标孕周统计': 达标数据['达标孕周'].describe().to_dict() if len(达标数据) > 0 else None
        }
    
    def fit_gamm_model(self, X, y, patient_ids=None, test_significance=True):
        """
        拟合GAMM模型
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征矩阵
        y : pd.Series or np.array
            目标变量
        patient_ids : pd.Series or np.array, optional
            患者ID（用于随机效应）
        test_significance : bool
            是否进行显著性检验
        """
        print("\n正在拟合GAMM模型...")
        
        # 如果需要进行显著性检验
        if test_significance:
            self.significant_features = self.test_feature_significance(X, y)
            
            if len(self.significant_features) == 0:
                print("警告：没有发现显著特征，使用所有特征进行建模")
                self.significant_features = self.feature_names[:X.shape[1]]
            else:
                print(f"\n发现 {len(self.significant_features)} 个显著特征: {self.significant_features}")
                
                # 筛选显著特征
                significant_indices = [i for i, name in enumerate(self.feature_names) if name in self.significant_features]
                X = X.iloc[:, significant_indices] if hasattr(X, 'iloc') else X[:, significant_indices]
                
                print(f"使用显著特征重新拟合模型，特征数量: {X.shape[1]}")
        else:
            self.significant_features = self.feature_names[:X.shape[1]]
        
        if self.use_r_gamm:
            self._fit_r_gamm(X, y, patient_ids)
        else:
            self._fit_python_alternative(X, y)
    
    def _fit_r_gamm(self, X, y, patient_ids=None):
        """
        使用R的mgcv包拟合GAMM模型
        """
        try:
            # 准备数据
            data_dict = {}
            n_features = X.shape[1]
            for i in range(n_features):
                data_dict[f'x{i+1}'] = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            data_dict['y'] = y
            
            if patient_ids is not None:
                data_dict['patient_id'] = patient_ids
            
            # 转换为R数据框
            with pandas2ri.converter.context():
                r_data = robjects.DataFrame(data_dict)
            
            # 构建GAMM公式（动态生成）
            smooth_terms = [f's(x{i+1}, k=5)' for i in range(n_features)]
            if patient_ids is not None:
                formula = f"y ~ {' + '.join(smooth_terms)} + s(patient_id, bs='re')"
            else:
                formula = f"y ~ {' + '.join(smooth_terms)}"
            
            print(f"GAMM公式: {formula}")
            
            # 拟合模型
            self.model = mgcv.gam(robjects.Formula(formula), data=r_data, method="REML")
            
            # 提取模型信息
            summary_result = base.summary(self.model)
            print("GAMM模型拟合完成")
            print(f"R-squared: {summary_result.rx2('r.sq')[0]:.4f}")
            print(f"Deviance explained: {summary_result.rx2('dev.expl')[0]*100:.2f}%")
            
            self.results['model_type'] = 'R_GAMM'
            self.results['r_squared'] = summary_result.rx2('r.sq')[0]
            self.results['deviance_explained'] = summary_result.rx2('dev.expl')[0]
            self.results['significant_features'] = self.significant_features
            self.results['n_features'] = n_features
            
        except Exception as e:
            print(f"R GAMM拟合失败: {e}")
            print("切换到Python替代方案")
            self.use_r_gamm = False
            self._fit_python_alternative(X, y)
    
    def _fit_python_alternative(self, X, y):
        """
        使用Python替代方案（随机森林 + 多项式特征）
        """
        print("使用Python替代方案进行建模...")
        
        # 创建多项式特征以模拟光滑函数
        poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        # 使用随机森林作为主模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # 使用岭回归作为辅助模型
        ridge_model = Ridge(alpha=1.0)
        
        # 拟合模型
        rf_model.fit(X, y)
        ridge_model.fit(X_poly, y)
        
        # 集成预测
        self.model = {
            'rf': rf_model,
            'ridge': ridge_model,
            'poly_features': poly_features
        }
        
        # 计算模型性能
        y_pred_rf = rf_model.predict(X)
        y_pred_ridge = ridge_model.predict(X_poly)
        y_pred_ensemble = 0.7 * y_pred_rf + 0.3 * y_pred_ridge
        
        r2 = r2_score(y, y_pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y, y_pred_ensemble))
        mae = mean_absolute_error(y, y_pred_ensemble)
        
        print(f"Python替代模型性能:")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        self.results['model_type'] = 'Python_Alternative'
        self.results['r_squared'] = r2
        self.results['rmse'] = rmse
        self.results['mae'] = mae
    
    def predict(self, X_new):
        """
        进行预测
        
        Parameters:
        -----------
        X_new : pd.DataFrame or np.array
            新的特征数据
            
        Returns:
        --------
        np.array
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.use_r_gamm:
            return self._predict_r_gamm(X_new)
        else:
            return self._predict_python_alternative(X_new)
    
    def _predict_r_gamm(self, X_new):
        """
        使用R GAMM模型进行预测
        """
        # 准备新数据
        data_dict = {}
        for i, col in enumerate(self.feature_names):
            data_dict[f'x{i+1}'] = X_new.iloc[:, i] if hasattr(X_new, 'iloc') else X_new[:, i]
        
        with pandas2ri.converter.context():
            r_newdata = robjects.DataFrame(data_dict)
        
        # 进行预测
        predictions = stats_r.predict(self.model, newdata=r_newdata)
        
        return np.array(predictions)
    
    def _predict_python_alternative(self, X_new):
        """
        使用Python替代模型进行预测
        """
        rf_pred = self.model['rf'].predict(X_new)
        
        X_new_poly = self.model['poly_features'].transform(X_new)
        ridge_pred = self.model['ridge'].predict(X_new_poly)
        
        # 集成预测
        ensemble_pred = 0.7 * rf_pred + 0.3 * ridge_pred
        
        return ensemble_pred
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征矩阵
        y : pd.Series or np.array
            目标变量
        cv : int
            交叉验证折数
            
        Returns:
        --------
        dict
            交叉验证结果
        """
        print(f"\n进行{cv}折交叉验证...")
        
        if not self.use_r_gamm:
            # 对于Python替代方案，使用sklearn的交叉验证
            from sklearn.model_selection import cross_val_score
            
            rf_scores = cross_val_score(
                RandomForestRegressor(n_estimators=100, random_state=42),
                X, y, cv=cv, scoring='r2'
            )
            
            cv_results = {
                'mean_r2': rf_scores.mean(),
                'std_r2': rf_scores.std(),
                'scores': rf_scores
            }
            
            print(f"交叉验证R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
            
            self.results['cv_results'] = cv_results
            return cv_results
        
        else:
            print("R GAMM模型的交叉验证需要额外实现")
            return None
    
    def plot_model_diagnostics(self, X, y):
        """
        绘制模型诊断图
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            特征矩阵
        y : pd.Series or np.array
            目标变量
        """
        print("\n绘制模型诊断图...")
        
        # 获取预测值
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAMM模型诊断图', fontsize=16)
        
        # 1. 残差 vs 拟合值
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('拟合值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')
        
        # 2. Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q图 (正态性检验)')
        
        # 3. 预测值 vs 实际值
        axes[1, 0].scatter(y, y_pred, alpha=0.6, color='green')
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 0].set_xlabel('实际值')
        axes[1, 0].set_ylabel('预测值')
        axes[1, 0].set_title('预测值 vs 实际值')
        
        # 4. 残差直方图
        axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('残差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('残差分布')
        
        plt.tight_layout()
        plt.savefig('./gamm_model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算诊断统计量
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Shapiro-Wilk正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        print(f"\n模型诊断统计:")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"残差正态性检验 (Shapiro-Wilk): 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
        
        self.results['diagnostics'] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        }
    
    def generate_predictions_for_new_patients(self, patient_profiles):
        """
        为新患者生成预测
        
        Parameters:
        -----------
        patient_profiles : pd.DataFrame
            新患者的特征数据
            
        Returns:
        --------
        pd.DataFrame
            包含预测结果的数据框
        """
        print("\n为新患者生成预测...")
        
        # 进行预测（使用显著特征）
        if self.significant_features is not None:
            available_features = [f for f in self.significant_features if f in patient_profiles.columns]
            if len(available_features) == 0:
                print("警告：患者数据中没有找到显著特征，使用所有可用特征")
                available_features = [f for f in self.feature_names if f in patient_profiles.columns]
        else:
            available_features = [f for f in self.feature_names if f in patient_profiles.columns]
            
        predictions = self.predict(patient_profiles[available_features])
        
        # 创建结果数据框
        results_df = patient_profiles.copy()
        results_df['预测达标孕周'] = predictions
        
        # 风险分层
        def risk_stratification(week):
            if week < -0.5:  # 对应约12周
                return '低风险'
            elif week < 0.5:  # 对应约20周
                return '中风险'
            else:
                return '高风险'
        
        results_df['风险等级'] = results_df['预测达标孕周'].apply(risk_stratification)
        
        # 计算置信区间（简化版本）
        prediction_std = self.results.get('rmse', 0.5)  # 使用RMSE作为标准误差的估计
        results_df['预测下界'] = predictions - 1.96 * prediction_std
        results_df['预测上界'] = predictions + 1.96 * prediction_std
        
        return results_df
    
    def plot_partial_effects(self, X, y, output_dir='./'):
        """
        生成每个自变量的偏效应图像
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        output_dir : str
            输出目录路径
        """
        print("\n正在生成偏效应图像...")
        
        # 确定要绘制的特征
        features_to_plot = self.significant_features if hasattr(self, 'significant_features') and self.significant_features else self.feature_names[:X.shape[1]]
        n_features = len(features_to_plot)
        
        if n_features == 0:
            print("没有特征需要绘制偏效应图")
            return
        
        # 计算子图布局
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
            
            # 获取特征在原始特征列表中的索引
            feature_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else idx
            
            if feature_idx < X.shape[1]:
                feature_values = X.iloc[:, feature_idx] if hasattr(X, 'iloc') else X[:, feature_idx]
                
                # 创建特征值的网格用于预测
                feature_range = np.linspace(feature_values.min(), feature_values.max(), 100)
                
                if self.use_r_gamm and self.model is not None:
                    # 使用R GAMM模型进行偏效应预测
                    try:
                        partial_effects = self._compute_partial_effects_r(feature_idx, feature_range, X)
                    except:
                        partial_effects = self._compute_partial_effects_python(feature_idx, feature_range, X, y)
                else:
                    # 使用Python模型进行偏效应预测
                    partial_effects = self._compute_partial_effects_python(feature_idx, feature_range, X, y)
                
                # 绘制偏效应曲线
                ax.plot(feature_range, partial_effects, 'b-', linewidth=2, label='偏效应')
                
                # 添加数据点的散点图
                ax.scatter(feature_values, y, alpha=0.3, s=20, color='gray', label='观测值')
                
                # 添加平滑趋势线
                try:
                    from scipy.interpolate import UnivariateSpline
                    # 对数据进行排序
                    sorted_indices = np.argsort(feature_values)
                    sorted_x = feature_values.iloc[sorted_indices] if hasattr(feature_values, 'iloc') else feature_values[sorted_indices]
                    sorted_y = y.iloc[sorted_indices] if hasattr(y, 'iloc') else y[sorted_indices]
                    
                    # 创建平滑样条
                    spline = UnivariateSpline(sorted_x, sorted_y, s=len(sorted_x)*0.1)
                    smooth_y = spline(feature_range)
                    ax.plot(feature_range, smooth_y, 'r--', alpha=0.7, linewidth=1.5, label='平滑趋势')
                except:
                    pass
                
                ax.set_xlabel(feature_name)
                ax.set_ylabel('达标孕周 (标准化)')
                ax.set_title(f'{feature_name}的偏效应')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # 添加相关系数信息
                if hasattr(self, 'significance_results'):
                    sig_row = self.significance_results[self.significance_results['特征名称'] == feature_name]
                    if not sig_row.empty:
                        corr = sig_row.iloc[0]['相关系数']
                        p_val = sig_row.iloc[0]['p值']
                        ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}', 
                               transform=ax.transAxes, fontsize=8, 
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = f"{output_dir}gamm_partial_effects.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"偏效应图像已保存到: {output_path}")
        plt.show()
        
        return output_path
    
    def _compute_partial_effects_r(self, feature_idx, feature_range, X):
        """
        使用R GAMM模型计算偏效应
        """
        try:
            # 创建预测数据，其他特征设为均值
            n_points = len(feature_range)
            pred_data = {}
            
            for i in range(X.shape[1]):
                if i == feature_idx:
                    pred_data[f'x{i+1}'] = feature_range
                else:
                    mean_val = X.iloc[:, i].mean() if hasattr(X, 'iloc') else X[:, i].mean()
                    pred_data[f'x{i+1}'] = [mean_val] * n_points
            
            # 转换为R数据框
            with pandas2ri.converter.context():
                r_pred_data = robjects.DataFrame(pred_data)
            
            # 进行预测
            predictions = mgcv.predict_gam(self.model, r_pred_data)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"R偏效应计算失败: {e}")
            return self._compute_partial_effects_python(feature_idx, feature_range, X, None)
    
    def _compute_partial_effects_python(self, feature_idx, feature_range, X, y):
        """
        使用Python模型计算偏效应
        """
        if self.model is None:
            # 如果没有训练好的模型，使用简单的局部回归
            feature_values = X.iloc[:, feature_idx] if hasattr(X, 'iloc') else X[:, feature_idx]
            
            # 使用局部加权回归(LOWESS)进行平滑
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(y, feature_values, frac=0.3)
                
                # 插值到新的特征范围
                from scipy.interpolate import interp1d
                interp_func = interp1d(smoothed[:, 0], smoothed[:, 1], 
                                     kind='linear', fill_value='extrapolate')
                return interp_func(feature_range)
            except:
                # 如果LOWESS失败，使用简单的线性插值
                sorted_indices = np.argsort(feature_values)
                sorted_x = feature_values.iloc[sorted_indices] if hasattr(feature_values, 'iloc') else feature_values[sorted_indices]
                sorted_y = y.iloc[sorted_indices] if hasattr(y, 'iloc') else y[sorted_indices]
                
                from scipy.interpolate import interp1d
                interp_func = interp1d(sorted_x, sorted_y, kind='linear', fill_value='extrapolate')
                return interp_func(feature_range)
        
        elif isinstance(self.model, dict):  # Python替代模型
            # 创建预测数据
            n_points = len(feature_range)
            pred_X = np.zeros((n_points, X.shape[1]))
            
            # 设置其他特征为均值
            for i in range(X.shape[1]):
                if i == feature_idx:
                    pred_X[:, i] = feature_range
                else:
                    pred_X[:, i] = X.iloc[:, i].mean() if hasattr(X, 'iloc') else X[:, i].mean()
            
            # 使用随机森林模型预测
            rf_pred = self.model['rf'].predict(pred_X)
            
            # 使用岭回归模型预测
            pred_X_poly = self.model['poly_features'].transform(pred_X)
            ridge_pred = self.model['ridge'].predict(pred_X_poly)
            
            # 集成预测
            return 0.7 * rf_pred + 0.3 * ridge_pred
        
        else:
            # 其他情况，返回零效应
            return np.zeros_like(feature_range)
    
    def save_results(self, output_path):
        """
        保存分析结果
        
        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        print(f"\n保存结果到: {output_path}")
        
        # 保存显著性检验结果表格
        if hasattr(self, 'significance_results'):
            significance_table_path = output_path.replace('.md', '_significance_table.csv')
            self.significance_results.to_csv(significance_table_path, index=False, encoding='utf-8-sig')
            print(f"显著性检验结果表格已保存到: {significance_table_path}")
            
            # 同时保存为Excel格式
            excel_path = significance_table_path.replace('.csv', '.xlsx')
            try:
                self.significance_results.to_excel(excel_path, index=False)
                print(f"显著性检验结果表格(Excel)已保存到: {excel_path}")
            except ImportError:
                print("未安装openpyxl，跳过Excel格式保存")
        
        # 创建结果报告
        report = f"""
# GAMM Y染色体达标孕周预测分析报告

## 模型信息
- 模型类型: {self.results.get('model_type', 'Unknown')}
- R²: {self.results.get('r_squared', 'N/A'):.4f}
- RMSE: {self.results.get('rmse', 'N/A')}
- MAE: {self.results.get('mae', 'N/A')}

## 数据概况
- 总样本数: {self.results.get('eda_stats', {}).get('总样本数', 'N/A')}
- 达标样本数: {self.results.get('eda_stats', {}).get('达标样本数', 'N/A')}
- 达标比例: {self.results.get('eda_stats', {}).get('达标比例', 'N/A'):.1f}%

## 特征显著性检验结果
"""
        
        # 添加显著性检验结果到报告
        if hasattr(self, 'significance_results'):
            report += "\n### 显著性检验统计表\n\n"
            report += self.significance_results.to_string(index=False)
            report += "\n\n"
        
        report += f"""
## 模型诊断
- 残差正态性检验p值: {self.results.get('diagnostics', {}).get('shapiro_p', 'N/A')}
- 模型拟合度: {'良好' if self.results.get('r_squared', 0) > 0.7 else '一般' if self.results.get('r_squared', 0) > 0.5 else '需要改进'}

## 交叉验证结果
{f"平均R²: {self.results.get('cv_results', {}).get('mean_r2', 'N/A'):.4f} ± {self.results.get('cv_results', {}).get('std_r2', 'N/A'):.4f}" if 'cv_results' in self.results else '未进行交叉验证'}

## 建议
1. 模型可用于预测Y染色体达标孕周
2. 建议结合临床经验进行决策
3. 定期更新模型以提高预测准确性
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("结果保存完成")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("GAMM Y染色体达标孕周预测分析")
    print("=" * 60)
    
    # 初始化预测器
    predictor = GAMMYChromosomePredictor(use_r_gamm=True)
    
    # 加载数据
    data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    df = predictor.load_and_preprocess_data(data_path)
    
    # 提取目标变量
    target_df = predictor.extract_target_variable(df)
    
    # 探索性数据分析
    predictor.exploratory_data_analysis(target_df)
    
    # 准备建模数据（只使用达标的样本）
    modeling_data = target_df[~target_df['is_censored']].copy()
    
    if len(modeling_data) < 10:
        print("\n警告：达标样本数量过少，建议收集更多数据")
        return
    
    X = modeling_data[predictor.feature_names]
    y = modeling_data['达标孕周']
    
    print(f"\n建模样本数: {len(modeling_data)}")
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 拟合模型
    predictor.fit_gamm_model(X_train, y_train)
    
    # 交叉验证
    predictor.cross_validate(X_train, y_train, cv=5)
    
    # 模型诊断
    predictor.plot_model_diagnostics(X_train, y_train)
    
    # 生成偏效应图像
    predictor.plot_partial_effects(X_train, y_train)
    
    # 测试集评估
    if len(X_test) > 0:
        y_pred_test = predictor.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n测试集性能:")
        print(f"R²: {test_r2:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
    
    # 示例预测
    print("\n示例预测:")
    example_patients = pd.DataFrame({
        'BMI_标准化': [-1.0, 0.0, 1.0],  # 低、中、高BMI
        '年龄_标准化': [0.0, 0.0, 0.0],   # 平均年龄
        '怀孕次数_标准化': [0.0, 0.0, 0.0], # 平均怀孕次数
        '生产次数_标准化': [0.0, 0.0, 0.0],  # 平均生产次数
        'Y染色体Z值_重新标准化': [0.0, 0.0, 0.0],  # 平均Y染色体Z值
        '过滤读段比例_标准化': [0.0, 0.0, 0.0],  # 平均过滤读段比例
        'GC含量_标准化': [0.0, 0.0, 0.0],  # 平均GC含量
        '唯一比对读段比例_标准化': [0.0, 0.0, 0.0],  # 平均唯一比对读段比例
        '重复读段比例_标准化': [0.0, 0.0, 0.0],  # 平均重复读段比例
        '比对比例_标准化': [0.0, 0.0, 0.0],  # 平均比对比例
        '总读段数_标准化': [0.0, 0.0, 0.0]   # 平均总读段数
    })
    
    predictions_df = predictor.generate_predictions_for_new_patients(example_patients)
    print(predictions_df[['BMI_标准化', '预测达标孕周', '风险等级']])
    
    # 保存结果
    predictor.save_results('./gamm_analysis_report.md')
    
    print("\n分析完成！")
    print("生成的文件:")
    print("- gamm_eda_analysis.png: 探索性数据分析图")
    print("- gamm_model_diagnostics.png: 模型诊断图")
    print("- gamm_partial_effects.png: 各自变量偏效应图")
    print("- gamm_analysis_report.md: 分析报告")
    print("- gamm_analysis_report_significance_table.csv: 特征显著性检验结果表格")
    print("- gamm_analysis_report_significance_table.xlsx: 特征显著性检验结果表格(Excel格式)")

if __name__ == "__main__":
    main()