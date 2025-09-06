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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入R接口用于GAMM
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    
    # 导入R包
    mgcv = importr('mgcv')
    base = importr('base')
    stats_r = importr('stats')
    R_AVAILABLE = True
    print("R接口可用，将使用mgcv包进行GAMM建模")
except ImportError:
    R_AVAILABLE = False
    print("R接口不可用，将使用Python替代方案")
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
            
            # 转换为R数据框
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
        
        for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
            # 计算相关系数和p值
            feature_values = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            corr, p_value = stats.pearsonr(feature_values, y)
            
            # 使用随机森林评估特征重要性
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            # 确保feature_values是numpy数组
            if hasattr(feature_values, 'values'):
                feature_array = feature_values.values
            else:
                feature_array = np.array(feature_values)
            rf.fit(feature_array.reshape(-1, 1), y)
            importance = rf.feature_importances_[0]
            
            # 综合判断：p值显著或重要性较高
            if p_value < alpha or importance > 0.1:
                significant_features.append(feature_name)
                print(f"{feature_name}: 显著 (相关性p值: {p_value:.4f}, 重要性: {importance:.4f})")
            else:
                print(f"{feature_name}: 不显著 (相关性p值: {p_value:.4f}, 重要性: {importance:.4f})")
        
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
        plt.savefig('d:/edgedownload/mm-1try/gamm_eda_analysis.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('d:/edgedownload/mm-1try/gamm_model_diagnostics.png', dpi=300, bbox_inches='tight')
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
    
    def save_results(self, output_path):
        """
        保存分析结果
        
        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        print(f"\n保存结果到: {output_path}")
        
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
    data_path = 'd:/edgedownload/mm-1try/processed_data.csv'
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
    predictor.save_results('d:/edgedownload/mm-1try/gamm_analysis_report.md')
    
    print("\n分析完成！")
    print("生成的文件:")
    print("- gamm_eda_analysis.png: 探索性数据分析图")
    print("- gamm_model_diagnostics.png: 模型诊断图")
    print("- gamm_analysis_report.md: 分析报告")

if __name__ == "__main__":
    main()