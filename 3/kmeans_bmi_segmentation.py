#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Means++聚类分析用于BMI分段和NIPT时点设计
基于GAMM Y染色体达标孕周预测结果进行聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# 导入GAMM预测器
from gamm_y_chromosome_prediction import GAMMYChromosomePredictor

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

class BMISegmentationAnalyzer:
    """
    基于GAMM预测结果的一维K-Means无重叠BMI分段分析器
    
    该分析器使用GAMM预测的Y染色体浓度达标孕周作为唯一聚类依据，
    通过一维K-Means算法实现无重叠的BMI分组，确保分组有序且边界清晰。
    """
    
    def __init__(self, output_dir='./output/'):
        """
        初始化分析器
        
        Parameters:
        -----------
        output_dir : str
            输出目录路径
        """
        self.output_dir = output_dir
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        self.gamm_predictor = GAMMYChromosomePredictor()
        self.scaler = StandardScaler()  # 保留用于其他可能的标准化需求
        
        # 一维聚类相关属性
        self.cluster_centers = None  # 一维聚类中心
        self.split_points = None     # 聚类分割点
        self.cluster_results = None
        self.optimal_k = None
        self.bmi_segments = None
        
    def load_and_prepare_data(self, data_path):
        """
        加载数据并准备GAMM预测
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
            
        Returns:
        --------
        tuple
            (原始数据, 目标变量数据, 特征矩阵, 目标向量)
        """
        print("=== 数据加载和预处理 ===")
        
        # 加载原始数据
        df = self.gamm_predictor.load_and_preprocess_data(data_path)
        
        # 提取目标变量
        target_df = self.gamm_predictor.extract_target_variable(df)
        
        # 数据清洗：处理NaN值
        print(f"\n数据清洗前形状: {target_df.shape}")
        print(f"目标变量NaN数量: {target_df['达标孕周'].isna().sum()}")
        
        # 移除目标变量为NaN的记录
        target_df = target_df.dropna(subset=['达标孕周'])
        
        # 检查特征矩阵中的NaN值
        feature_cols = self.gamm_predictor.feature_names
        feature_nan_counts = target_df[feature_cols].isna().sum()
        if feature_nan_counts.sum() > 0:
            print(f"特征矩阵NaN数量: {feature_nan_counts[feature_nan_counts > 0]}")
            # 移除特征中有NaN的记录
            target_df = target_df.dropna(subset=feature_cols)
        
        # 准备特征矩阵
        X = target_df[feature_cols]
        y = target_df['达标孕周']
        
        print(f"\n数据清洗后形状: {target_df.shape}")
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        print(f"目标变量范围: [{y.min():.3f}, {y.max():.3f}]")
        
        return df, target_df, X, y
    
    def train_gamm_and_predict(self, X, y, target_df):
        """
        训练GAMM模型并生成预测
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        target_df : pd.DataFrame
            目标变量数据框
            
        Returns:
        --------
        pd.DataFrame
            包含预测结果的数据框
        """
        print("\n=== GAMM模型训练 ===")
        
        # 训练GAMM模型
        self.gamm_predictor.fit_gamm_model(X, y, patient_ids=target_df['孕妇代码'])
        
        # 生成预测
        predictions = self.gamm_predictor.predict(X)
        
        # 创建包含预测结果的数据框
        result_df = target_df.copy()
        result_df['预测达标孕周'] = predictions
        result_df['预测误差'] = result_df['预测达标孕周'] - result_df['达标孕周']
        
        print(f"预测完成，RMSE: {np.sqrt(np.mean(result_df['预测误差']**2)):.3f}")
        print(f"预测R²: {1 - np.var(result_df['预测误差']) / np.var(result_df['达标孕周']):.3f}")
        
        return result_df
    
    def determine_optimal_clusters(self, data_for_clustering, max_k=10):
        """
        确定最优聚类数量
        
        Parameters:
        -----------
        data_for_clustering : np.ndarray
            用于聚类的数据
        max_k : int
            最大聚类数量
            
        Returns:
        --------
        int
            最优聚类数量
        """
        print("\n=== 确定最优聚类数量 ===")
        
        k_range = range(2, max_k + 1)
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data_for_clustering)
            
            silhouette_scores.append(silhouette_score(data_for_clustering, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(data_for_clustering, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(data_for_clustering, cluster_labels))
            inertias.append(kmeans.inertia_)
        
        # 绘制评估指标
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚类数量选择评估指标', fontsize=16, fontweight='bold')
        
        # 轮廓系数
        axes[0, 0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('轮廓系数 (越高越好)', fontweight='bold')
        axes[0, 0].set_xlabel('聚类数量 (k)')
        axes[0, 0].set_ylabel('轮廓系数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz指数
        axes[0, 1].plot(k_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Calinski-Harabasz指数 (越高越好)', fontweight='bold')
        axes[0, 1].set_xlabel('聚类数量 (k)')
        axes[0, 1].set_ylabel('CH指数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Davies-Bouldin指数
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Davies-Bouldin指数 (越低越好)', fontweight='bold')
        axes[1, 0].set_xlabel('聚类数量 (k)')
        axes[1, 0].set_ylabel('DB指数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 肘部法则
        axes[1, 1].plot(k_range, inertias, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('肘部法则 (寻找拐点)', fontweight='bold')
        axes[1, 1].set_xlabel('聚类数量 (k)')
        axes[1, 1].set_ylabel('簇内平方和')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}kmeans_cluster_selection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 综合评估选择最优k
        # 标准化各指标
        silhouette_norm = np.array(silhouette_scores) / max(silhouette_scores)
        calinski_norm = np.array(calinski_scores) / max(calinski_scores)
        davies_bouldin_norm = 1 - (np.array(davies_bouldin_scores) / max(davies_bouldin_scores))  # 越小越好，所以取反
        
        # 计算综合得分
        composite_scores = silhouette_norm + calinski_norm + davies_bouldin_norm
        optimal_k = k_range[np.argmax(composite_scores)]
        
        print(f"各聚类数量的评估结果:")
        for i, k in enumerate(k_range):
            print(f"k={k}: 轮廓系数={silhouette_scores[i]:.3f}, CH指数={calinski_scores[i]:.1f}, DB指数={davies_bouldin_scores[i]:.3f}")
        
        print(f"\n推荐的最优聚类数量: {optimal_k}")
        
        return optimal_k
    
    def perform_one_dimensional_clustering(self, prediction_df, optimal_k=None):
        """
        执行基于BMI和预测达标孕周的二维K-Means聚类
        
        Parameters:
        -----------
        prediction_df : pd.DataFrame
            包含预测结果的数据框
        optimal_k : int, optional
            聚类数量，如果为None则自动确定
            
        Returns:
        --------
        pd.DataFrame
            包含聚类结果的数据框
        """
        print("\n=== 基于BMI和预测达标孕周的二维K-Means聚类分析 ===")
        
        # 准备聚类特征：使用BMI标准化值和预测达标孕周
        clustering_features = prediction_df[['BMI_标准化', '预测达标孕周']].values
        
        # 确定最优聚类数量
        if optimal_k is None:
            optimal_k = self.determine_optimal_clusters(clustering_features)
        
        self.optimal_k = optimal_k
        
        # 执行二维K-Means聚类
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_features)
        
        # 保存聚类结果到类属性
        self.cluster_centers = kmeans.cluster_centers_
        
        # 添加聚类结果到数据框
        result_df = prediction_df.copy()
        result_df['聚类标签'] = cluster_labels
        
        # 计算聚类质量指标
        silhouette_avg = silhouette_score(clustering_features, cluster_labels)
        calinski_score = calinski_harabasz_score(clustering_features, cluster_labels)
        davies_bouldin = davies_bouldin_score(clustering_features, cluster_labels)
        
        print(f"\n二维聚类质量评估:")
        print(f"聚类数量: {optimal_k}")
        print(f"轮廓系数: {silhouette_avg:.3f}")
        print(f"Calinski-Harabasz指数: {calinski_score:.1f}")
        print(f"Davies-Bouldin指数: {davies_bouldin:.3f}")
        print(f"聚类中心 (BMI, 预测达标孕周):")
        for i, center in enumerate(self.cluster_centers):
            print(f"  群组{i+1}: ({center[0]:.3f}, {center[1]:.3f})")
        
        # 输出各群组的特征范围
        print("\n各群组特征范围:")
        for i in range(optimal_k):
            cluster_data = result_df[result_df['聚类标签'] == i]
            bmi_range = f"[{cluster_data['BMI_标准化'].min():.3f}, {cluster_data['BMI_标准化'].max():.3f}]"
            pred_range = f"[{cluster_data['预测达标孕周'].min():.3f}, {cluster_data['预测达标孕周'].max():.3f}]"
            print(f"  群组{i+1}: BMI {bmi_range}, 预测达标孕周 {pred_range}, 样本数: {len(cluster_data)}")
        
        self.cluster_results = result_df
        
        return result_df
    
    def kmeans_1d(self, data, k, max_iter=100, tol=1e-4):
        """
        一维K-Means聚类算法实现
        
        Parameters:
        -----------
        data : np.ndarray
            一维数据数组
        k : int
            聚类数量
        max_iter : int
            最大迭代次数
        tol : float
            收敛容忍度
            
        Returns:
        --------
        tuple
            (聚类标签, 聚类中心, 分割点)
        """
        n = len(data)
        
        # 初始化聚类中心：使用K-Means++策略
        centers = self.initialize_centers_1d(data, k)
        
        for iteration in range(max_iter):
            # E步：分配数据点到最近的聚类中心
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                distances = [abs(data[i] - center) for center in centers]
                labels[i] = np.argmin(distances)
            
            # M步：更新聚类中心
            new_centers = []
            for j in range(k):
                cluster_points = data[labels == j]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points))
                else:
                    # 如果某个聚类为空，保持原中心
                    new_centers.append(centers[j])
            
            # 检查收敛
            center_shift = sum(abs(new_centers[j] - centers[j]) for j in range(k))
            if center_shift < tol:
                break
                
            centers = new_centers
        
        # 计算分割点
        sorted_centers = sorted(centers)
        split_points = []
        for i in range(len(sorted_centers) - 1):
            split_point = (sorted_centers[i] + sorted_centers[i + 1]) / 2
            split_points.append(split_point)
        
        return labels, centers, split_points
    
    def initialize_centers_1d(self, data, k):
        """
        使用K-Means++策略初始化一维聚类中心
        
        Parameters:
        -----------
        data : np.ndarray
            一维数据数组
        k : int
            聚类数量
            
        Returns:
        --------
        list
            初始聚类中心列表
        """
        np.random.seed(42)  # 确保可重复性
        centers = []
        
        # 随机选择第一个中心
        centers.append(np.random.choice(data))
        
        # 选择剩余的k-1个中心
        for _ in range(k - 1):
            distances = []
            for point in data:
                min_dist = min(abs(point - center) for center in centers)
                distances.append(min_dist ** 2)
            
            # 按距离平方的概率选择下一个中心
            distances = np.array(distances)
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            
            r = np.random.random()
            for i, prob in enumerate(cumulative_probs):
                if r <= prob:
                    centers.append(data[i])
                    break
        
        return centers
    
    def calculate_wcss_1d(self, data, labels, centers):
        """
        计算一维聚类的类内平方和(WCSS)
        
        Parameters:
        -----------
        data : np.ndarray
            一维数据数组
        labels : np.ndarray
            聚类标签
        centers : list
            聚类中心列表
            
        Returns:
        --------
        float
            类内平方和
        """
        wcss = 0
        for i, center in enumerate(centers):
            cluster_points = data[labels == i]
            wcss += np.sum((cluster_points - center) ** 2)
        return wcss
    
    def determine_optimal_clusters_1d(self, data, max_k=10):
        """
        确定一维聚类的最优聚类数量
        
        Parameters:
        -----------
        data : np.ndarray
            一维数据数组
        max_k : int
            最大聚类数量
            
        Returns:
        --------
        int
            最优聚类数量
        """
        print("\n=== 确定一维聚类最优数量 ===")
        
        k_range = range(2, min(max_k + 1, len(np.unique(data)) + 1))
        wcss_scores = []
        
        for k in k_range:
            labels, centers, _ = self.kmeans_1d(data, k)
            wcss = self.calculate_wcss_1d(data, labels, centers)
            wcss_scores.append(wcss)
        
        # 使用肘部法则选择最优k
        # 计算二阶差分来找到肘部
        if len(wcss_scores) >= 3:
            second_diffs = []
            for i in range(1, len(wcss_scores) - 1):
                second_diff = wcss_scores[i-1] - 2*wcss_scores[i] + wcss_scores[i+1]
                second_diffs.append(second_diff)
            
            # 选择二阶差分最大的点作为肘部
            elbow_idx = np.argmax(second_diffs) + 1  # +1因为second_diffs从索引1开始
            optimal_k = k_range[elbow_idx]
        else:
            # 如果数据点太少，默认选择中间值
            optimal_k = k_range[len(k_range) // 2]
        
        print(f"各聚类数量的WCSS:")
        for i, k in enumerate(k_range):
            print(f"k={k}: WCSS={wcss_scores[i]:.3f}")
        
        print(f"\n推荐的最优聚类数量: {optimal_k}")
        
        return optimal_k
    
    def perform_clustering(self, prediction_df, optimal_k=None):
        """
        执行聚类分析（调用一维聚类方法）
        
        Parameters:
        -----------
        prediction_df : pd.DataFrame
            包含预测结果的数据框
        optimal_k : int, optional
            聚类数量，如果为None则自动确定
            
        Returns:
        --------
        pd.DataFrame
            包含聚类结果的数据框
        """
        return self.perform_one_dimensional_clustering(prediction_df, optimal_k)
    
    def analyze_bmi_segments(self, cluster_df):
        """
        分析基于BMI和预测达标孕周的二维K-Means聚类结果
        
        Parameters:
        -----------
        cluster_df : pd.DataFrame
            包含聚类结果的数据框
            
        Returns:
        --------
        dict
            聚类分段分析结果
        """
        print("\n=== 基于BMI和预测达标孕周的二维K-Means聚类分析 ===")
        
        segments = {}
        
        # 按聚类标签排序，确保分析顺序
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        
        print(f"\n共识别出 {len(cluster_ids)} 个群组，基于BMI标准化值和预测达标孕周进行二维聚类")
        print("群组基于两个维度的综合特征进行划分")
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            
            # BMI统计（标准化值）
            bmi_std = cluster_data['BMI_标准化']
            bmi_min = bmi_std.min()
            bmi_max = bmi_std.max()
            bmi_mean = bmi_std.mean()
            bmi_median = bmi_std.median()
            
            # 预测达标孕周统计
            pred_weeks = cluster_data['预测达标孕周']
            pred_min = pred_weeks.min()
            pred_max = pred_weeks.max()
            pred_mean = pred_weeks.mean()
            pred_median = pred_weeks.median()
            
            # 将标准化时间转换为原始孕周
            # 注意：GAMM预测的是标准化孕周，需要转换为原始孕周
            # 使用interval_censored_survival_model.py中的标准化参数
            time_mean = 16.846  # 原始孕周均值
            time_std = 4.076    # 原始孕周标准差
            
            # 转换所有预测孕周到原始孕周
            pred_weeks_original = pred_weeks * time_std + time_mean
            pred_mean_original = pred_mean * time_std + time_mean
            
            # 根据用户要求优化NIPT时点：优先12周以内，其次27周以内，然后考虑成功率
            # 计算满足12周以内的孕妇比例
            within_12_weeks = (pred_weeks_original <= 12).sum() / len(pred_weeks_original)
            # 计算满足27周以内的孕妇比例
            within_27_weeks = (pred_weeks_original <= 27).sum() / len(pred_weeks_original)
            
            # 动态选择最优分位数
            if within_12_weeks >= 0.8:  # 如果80%以上孕妇能在12周内达标
                # 选择12周作为目标，找到对应的分位数
                target_week = 12
                optimal_percentile = (pred_weeks_original <= target_week).mean()
                optimal_percentile = min(0.95, max(0.5, optimal_percentile))  # 限制在50%-95%之间
            elif within_27_weeks >= 0.7:  # 如果70%以上孕妇能在27周内达标
                # 选择27周作为目标，找到对应的分位数
                target_week = 27
                optimal_percentile = (pred_weeks_original <= target_week).mean()
                optimal_percentile = min(0.95, max(0.5, optimal_percentile))  # 限制在50%-95%之间
            else:
                # 如果都不满足，使用90%分位数确保较高的成功率
                optimal_percentile = 0.90
                target_week = pred_weeks_original.quantile(optimal_percentile)
            
            # 计算最优分位数对应的孕周
            pred_optimal_percentile = pred_weeks.quantile(optimal_percentile)
            pred_optimal_percentile_original = pred_optimal_percentile * time_std + time_mean
            
            # 计算检测成功率（非删失样本比例）
            success_rate = (~cluster_data['is_censored']).mean() if 'is_censored' in cluster_data.columns else 1.0
            
            # 计算风险等级和推荐NIPT时点
            if pred_mean <= -0.5:  # 较早达标
                risk_level = "低风险"
                recommended_nipt_week = "10-11周"
            elif pred_mean <= 0:
                risk_level = "中低风险"
                recommended_nipt_week = "11-12周"
            elif pred_mean <= 0.5:
                risk_level = "中风险"
                recommended_nipt_week = "12-13周"
            else:
                risk_level = "高风险"
                recommended_nipt_week = "13-14周"
            
            # 计算BMI分位数，用于确定区间边界
            bmi_q25 = bmi_std.quantile(0.25)
            bmi_q75 = bmi_std.quantile(0.75)
            
            segments[f'群组{cluster_id+1}'] = {
                '样本数量': len(cluster_data),
                '样本比例': f"{len(cluster_data)/len(cluster_df)*100:.1f}%",
                'BMI标准化范围': f"[{bmi_min:.3f}, {bmi_max:.3f}]",
                'BMI标准化均值': f"{bmi_mean:.3f}",
                'BMI标准化中位数': f"{bmi_median:.3f}",
                'BMI标准化四分位距': f"[{bmi_q25:.3f}, {bmi_q75:.3f}]",
                '预测达标孕周范围': f"[{pred_min:.3f}, {pred_max:.3f}]",
                '预测达标孕周均值': f"{pred_mean:.3f}",
                '预测达标孕周中位数': f"{pred_median:.3f}",
                '预测达标孕周标准差': f"{pred_weeks.std():.3f}",
                f'预测达标孕周{optimal_percentile*100:.0f}%分位数': f"{pred_optimal_percentile:.3f}",
                f'{optimal_percentile*100:.0f}%孕妇达标检测孕周': f"{pred_optimal_percentile:.1f}周",
                'NIPT时点(原始孕周)': f"{pred_mean_original:.1f}周",
                f'NIPT时点{optimal_percentile*100:.0f}%分位数(原始孕周)': f"{pred_optimal_percentile_original:.1f}周",
                '目标孕周约束': f"{target_week:.1f}周以内",
                '12周内达标比例': f"{within_12_weeks*100:.1f}%",
                '27周内达标比例': f"{within_27_weeks*100:.1f}%",
                '检测成功率': f"{success_rate*100:.1f}%",
                '风险等级': risk_level,
                '推荐NIPT时点': recommended_nipt_week,
                '达标比例': f"{(~cluster_data['is_censored']).mean()*100:.1f}%" if 'is_censored' in cluster_data.columns else "100.0%"
            }
        
        self.bmi_segments = segments
        
        # 打印分段结果
        print("\n各群组特征分析:")
        for segment_name, info in segments.items():
            print(f"\n{segment_name}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # 专门打印优化分位数信息
        print("\n=== 各群组优化NIPT时点汇总 ===")
        for segment_name, info in segments.items():
            # 找到分位数相关的键
            percentile_key = [k for k in info.keys() if '孕妇达标检测孕周' in k and '%' in k][0]
            print(f"{segment_name}: {info[percentile_key]} (约束: {info['目标孕周约束']})")
        
        # 专门打印NIPT时点信息（原始孕周）
        print("\n=== 各群组NIPT时点详细信息（原始孕周）===")
        for segment_name, info in segments.items():
            # 找到NIPT时点分位数相关的键
            nipt_percentile_key = [k for k in info.keys() if 'NIPT时点' in k and '%分位数' in k and '原始孕周' in k][0]
            print(f"{segment_name}:")
            print(f"  NIPT时点(均值): {info['NIPT时点(原始孕周)']}")
            print(f"  NIPT时点(优化分位数): {info[nipt_percentile_key]}")
            print(f"  目标约束: {info['目标孕周约束']}")
            print(f"  12周内达标: {info['12周内达标比例']}")
            print(f"  27周内达标: {info['27周内达标比例']}")
            print(f"  样本数: {info['样本数量']}")
            print(f"  检测成功率: {info['检测成功率']}")
            print()
        
        return segments
    
    def visualize_gestational_weeks_by_cluster(self, cluster_df):
        """
        可视化每个群组的Y染色体浓度达标孕周数
        纵坐标为去标准化后的原始孕周数据
        
        Parameters:
        -----------
        cluster_df : pd.DataFrame
            包含聚类结果的数据框
        """
        print("\n=== 生成各群组达标孕周可视化图表 ===")
        
        # 标准化参数（与analyze_bmi_segments中保持一致）
        time_mean = 16.846  # 原始孕周均值
        time_std = 4.076    # 原始孕周标准差
        
        # 将标准化孕周转换为原始孕周
        cluster_df = cluster_df.copy()
        cluster_df['预测达标孕周_原始'] = cluster_df['预测达标孕周'] * time_std + time_mean
        
        # 获取聚类标签
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        n_clusters = len(cluster_ids)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('各群组Y染色体浓度达标孕周分析', fontsize=16, fontweight='bold')
        
        # 1. 箱线图 - 显示每个群组的分布
        ax1 = axes[0, 0]
        data_for_boxplot = [cluster_df[cluster_df['聚类标签'] == cid]['预测达标孕周_原始'].values 
                           for cid in cluster_ids]
        labels = [f'群组{cid+1}' for cid in cluster_ids]
        
        box_plot = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('各群组达标孕周分布（箱线图）', fontweight='bold')
        ax1.set_xlabel('群组')
        ax1.set_ylabel('达标孕周（周）')
        ax1.grid(True, alpha=0.3)
        
        # 添加关键孕周参考线
        ax1.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax1.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        ax1.legend()
        
        # 2. 小提琴图 - 显示密度分布
        ax2 = axes[0, 1]
        violin_parts = ax2.violinplot(data_for_boxplot, positions=range(1, n_clusters+1), 
                                     showmeans=True, showmedians=True)
        
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax2.set_title('各群组达标孕周密度分布（小提琴图）', fontweight='bold')
        ax2.set_xlabel('群组')
        ax2.set_ylabel('达标孕周（周）')
        ax2.set_xticks(range(1, n_clusters+1))
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3)
        
        # 添加关键孕周参考线
        ax2.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax2.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        ax2.legend()
        
        # 3. 散点图 - BMI vs 达标孕周
        ax3 = axes[1, 0]
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            ax3.scatter(cluster_data['BMI_标准化'], cluster_data['预测达标孕周_原始'], 
                       c=[colors[i]], label=f'群组{cluster_id+1}', alpha=0.6, s=50)
        
        ax3.set_title('BMI标准化值 vs 达标孕周', fontweight='bold')
        ax3.set_xlabel('BMI标准化值')
        ax3.set_ylabel('达标孕周（周）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加关键孕周参考线
        ax3.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周参考线')
        ax3.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周参考线')
        
        # 4. 统计汇总条形图
        ax4 = axes[1, 1]
        
        # 计算各群组的统计信息
        stats_data = []
        for cluster_id in cluster_ids:
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            weeks_data = cluster_data['预测达标孕周_原始']
            
            stats_data.append({
                '群组': f'群组{cluster_id+1}',
                '均值': weeks_data.mean(),
                '中位数': weeks_data.median(),
                '90%分位数': weeks_data.quantile(0.9)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        x = np.arange(len(stats_df))
        width = 0.25
        
        bars1 = ax4.bar(x - width, stats_df['均值'], width, label='均值', color='skyblue', alpha=0.8)
        bars2 = ax4.bar(x, stats_df['中位数'], width, label='中位数', color='lightgreen', alpha=0.8)
        bars3 = ax4.bar(x + width, stats_df['90%分位数'], width, label='90%分位数', color='salmon', alpha=0.8)
        
        ax4.set_title('各群组达标孕周统计汇总', fontweight='bold')
        ax4.set_xlabel('群组')
        ax4.set_ylabel('达标孕周（周）')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stats_df['群组'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 在条形图上添加数值标签
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        # 添加关键孕周参考线
        ax4.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax4.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = f"{self.output_dir}gestational_weeks_by_cluster_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存至: {output_path}")
        
        # 显示图表
        plt.show()
        
        # 打印详细统计信息
        print("\n=== 各群组达标孕周统计详情 ===")
        for i, row in stats_df.iterrows():
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_ids[i]]
            weeks_data = cluster_data['预测达标孕周_原始']
            
            within_12_weeks = (weeks_data <= 12).sum() / len(weeks_data) * 100
            within_27_weeks = (weeks_data <= 27).sum() / len(weeks_data) * 100
            
            print(f"\n{row['群组']} (样本数: {len(cluster_data)}):")
            print(f"  均值: {row['均值']:.2f}周")
            print(f"  中位数: {row['中位数']:.2f}周")
            print(f"  90%分位数: {row['90%分位数']:.2f}周")
            print(f"  标准差: {weeks_data.std():.2f}周")
            print(f"  12周内达标比例: {within_12_weeks:.1f}%")
            print(f"  27周内达标比例: {within_27_weeks:.1f}%")
            print(f"  范围: [{weeks_data.min():.2f}, {weeks_data.max():.2f}]周")
    
    def visualize_clustering_results(self, cluster_df):
        """
        可视化二维聚类结果
        
        Parameters:
        -----------
        cluster_df : pd.DataFrame
            包含聚类结果的数据框
        """
        print("\n=== 二维聚类结果可视化 ===")
        
        # 创建综合可视化
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 主要聚类散点图 - 基于BMI和预测达标孕周的二维聚类
        ax1 = plt.subplot(3, 3, (1, 2))
        scatter = ax1.scatter(cluster_df['BMI_标准化'], cluster_df['预测达标孕周'], 
                             c=cluster_df['聚类标签'], cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('BMI (标准化)', fontweight='bold')
        ax1.set_ylabel('预测达标孕周 (标准化)', fontweight='bold')
        ax1.set_title('基于BMI和预测达标孕周的二维K-Means聚类结果', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加二维聚类中心
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            for i, center in enumerate(self.cluster_centers):
                ax1.scatter(center[0], center[1], c='red', marker='x', s=200, linewidths=3)
                ax1.text(center[0], center[1] + 0.1, f'中心{i+1}', ha='center', va='bottom', 
                        fontsize=10, color='red', fontweight='bold')
        
        ax1.scatter([], [], c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        ax1.legend()
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('聚类标签', fontweight='bold')
        
        # 2. BMI分布直方图
        ax2 = plt.subplot(3, 3, 3)
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            ax2.hist(cluster_data['BMI_标准化'], alpha=0.6, label=f'群组{cluster_id+1}', bins=15)
        ax2.set_xlabel('BMI (标准化)', fontweight='bold')
        ax2.set_ylabel('频数', fontweight='bold')
        ax2.set_title('各群组BMI分布', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 预测达标孕周分布
        ax3 = plt.subplot(3, 3, 4)
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            ax3.hist(cluster_data['预测达标孕周'], alpha=0.6, label=f'群组{cluster_id+1}', bins=15)
        ax3.set_xlabel('预测达标孕周 (标准化)', fontweight='bold')
        ax3.set_ylabel('频数', fontweight='bold')
        ax3.set_title('各群组预测达标孕周分布', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 箱线图比较
        ax4 = plt.subplot(3, 3, 5)
        cluster_labels = [f'群组{i+1}' for i in sorted(cluster_df['聚类标签'].unique())]
        bmi_by_cluster = [cluster_df[cluster_df['聚类标签'] == i]['BMI_标准化'] for i in sorted(cluster_df['聚类标签'].unique())]
        ax4.boxplot(bmi_by_cluster, labels=cluster_labels)
        ax4.set_ylabel('BMI (标准化)', fontweight='bold')
        ax4.set_title('各群组BMI箱线图', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 预测达标孕周箱线图
        ax5 = plt.subplot(3, 3, 6)
        pred_by_cluster = [cluster_df[cluster_df['聚类标签'] == i]['预测达标孕周'] for i in sorted(cluster_df['聚类标签'].unique())]
        ax5.boxplot(pred_by_cluster, labels=cluster_labels)
        ax5.set_ylabel('预测达标孕周 (标准化)', fontweight='bold')
        ax5.set_title('各群组预测达标孕周箱线图', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 达标比例条形图
        ax6 = plt.subplot(3, 3, 7)
        达标比例 = []
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            达标比例.append((~cluster_data['is_censored']).mean() * 100)
        
        bars = ax6.bar(cluster_labels, 达标比例, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(cluster_labels)])
        ax6.set_ylabel('达标比例 (%)', fontweight='bold')
        ax6.set_title('各群组Y染色体达标比例', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, ratio in zip(bars, 达标比例):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7. 样本数量分布
        ax7 = plt.subplot(3, 3, 8)
        sample_counts = [len(cluster_df[cluster_df['聚类标签'] == i]) for i in sorted(cluster_df['聚类标签'].unique())]
        bars = ax7.bar(cluster_labels, sample_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(cluster_labels)])
        ax7.set_ylabel('样本数量', fontweight='bold')
        ax7.set_title('各群组样本数量分布', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, sample_counts):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 8. PCA降维可视化
        ax8 = plt.subplot(3, 3, 9)
        # 使用更多特征进行PCA
        pca_features = ['BMI_标准化', '预测达标孕周', '年龄_标准化', '怀孕次数_标准化']
        pca_data = cluster_df[pca_features].values
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        
        scatter_pca = ax8.scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=cluster_df['聚类标签'], cmap='viridis', 
                                 s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax8.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontweight='bold')
        ax8.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontweight='bold')
        ax8.set_title('PCA降维聚类结果', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}kmeans_clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_nipt_recommendations(self, segments):
        """
        生成NIPT时点推荐报告
        
        Parameters:
        -----------
        segments : dict
            BMI分段分析结果
            
        Returns:
        --------
        str
            推荐报告文本
        """
        print("\n=== 生成NIPT时点推荐 ===")
        
        report = []
        report.append("# 基于GAMM预测和K-Means++聚类的NIPT时点推荐报告")
        report.append("\n## 执行摘要")
        report.append(f"本分析基于{len(self.cluster_results)}名孕妇的数据，使用GAMM模型预测Y染色体浓度达到4%的时间，")
        report.append(f"然后通过K-Means++聚类算法将孕妇分为{self.optimal_k}个群组，为每个群组制定个性化的NIPT检测时点。")
        
        report.append("\n## 分群结果与推荐")
        
        for segment_name, info in segments.items():
            report.append(f"\n### {segment_name}")
            report.append(f"- **样本特征**: {info['样本数量']}名孕妇 ({info['样本比例']})")
            report.append(f"- **BMI特征**: 标准化均值 {info['BMI标准化均值']}，范围 {info['BMI标准化范围']}")
            report.append(f"- **预测达标时间**: 平均 {info['预测达标孕周均值']} 标准化孕周")
            report.append(f"- **Y染色体达标率**: {info['达标比例']}")
            report.append(f"- **风险等级**: {info['风险等级']}")
            report.append(f"- **推荐NIPT时点**: {info['推荐NIPT时点']}")
            
            # 添加风险解释
            if info['风险等级'] == "低风险":
                report.append(f"  - 该群组孕妇Y染色体浓度达标较早，可以在较早时点进行NIPT检测")
            elif info['风险等级'] == "高风险":
                report.append(f"  - 该群组孕妇Y染色体浓度达标较晚，建议推迟NIPT检测时点以确保准确性")
            else:
                report.append(f"  - 该群组孕妇风险适中，按标准时点进行NIPT检测")
        
        report.append("\n## 临床应用建议")
        report.append("1. **个性化检测**: 根据孕妇BMI等特征将其分配到相应群组，按推荐时点进行NIPT")
        report.append("2. **风险管理**: 高风险群组需要更密切的监测和可能的多次检测")
        report.append("3. **成本效益**: 低风险群组可以较早检测，减少等待时间和焦虑")
        report.append("4. **质量控制**: 建议对每个群组的检测结果进行独立的质量评估")
        
        report.append("\n## 模型性能")
        if hasattr(self.gamm_predictor, 'results') and 'cross_validation' in self.gamm_predictor.results:
            cv_results = self.gamm_predictor.results['cross_validation']
            report.append(f"- GAMM模型交叉验证R²: {cv_results.get('mean_r2', 'N/A'):.3f}")
            report.append(f"- GAMM模型交叉验证RMSE: {cv_results.get('mean_rmse', 'N/A'):.3f}")
        
        # 聚类质量
        if self.cluster_results is not None:
            clustering_data = self.cluster_results[['预测达标孕周']].values
            # 对于一维聚类，直接使用原始数据计算轮廓系数
            silhouette_avg = silhouette_score(clustering_data, self.cluster_results['聚类标签'])
            report.append(f"- 聚类轮廓系数: {silhouette_avg:.3f}")
        
        report_text = "\n".join(report)
        
        # 保存报告
        with open(f'{self.output_dir}nipt_timing_recommendation_report.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("推荐报告已保存到 nipt_timing_recommendation_report.md")
        
        return report_text
    
    def save_results(self, cluster_df, segments):
        """
        保存分析结果
        
        Parameters:
        -----------
        cluster_df : pd.DataFrame
            包含聚类结果的数据框
        segments : dict
            BMI分段分析结果
        """
        print("\n=== 保存分析结果 ===")
        
        # 保存详细聚类结果
        cluster_df.to_csv(f'{self.output_dir}kmeans_clustering_detailed_results.csv', 
                          index=False, encoding='utf-8')
        
        # 保存分段摘要
        segments_df = pd.DataFrame(segments).T
        segments_df.to_csv(f'{self.output_dir}bmi_segments_summary.csv', 
                          encoding='utf-8')
        segments_df.to_excel(f'{self.output_dir}bmi_segments_summary.xlsx', 
                           index=False)
        
        print("结果文件已保存:")
        print("- kmeans_clustering_detailed_results.csv: 详细聚类结果")
        print("- bmi_segments_summary.csv/xlsx: BMI分段摘要")
        print("- kmeans_clustering_results.png: 聚类结果可视化")
        print("- kmeans_cluster_selection.png: 聚类数量选择过程")
        print("- nipt_timing_recommendation_report.md: NIPT时点推荐报告")

def main():
    """
    主函数
    """
    print("=== 基于GAMM预测的K-Means++聚类BMI分段分析 ===")
    print("目标: 对孕妇BMI进行分段并设计个性化NIPT时点")
    
    # 初始化分析器
    analyzer = BMISegmentationAnalyzer()
    
    # 数据路径
    data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    
    try:
        # 1. 加载和准备数据
        df, target_df, X, y = analyzer.load_and_prepare_data(data_path)
        
        # 2. 训练GAMM模型并生成预测
        prediction_df = analyzer.train_gamm_and_predict(X, y, target_df)
        
        # 3. 执行K-Means++聚类
        cluster_df = analyzer.perform_clustering(prediction_df)
        
        # 4. 分析BMI分段
        segments = analyzer.analyze_bmi_segments(cluster_df)
        
        # 5. 可视化结果
        analyzer.visualize_clustering_results(cluster_df)
        
        # 5.1 可视化各群组达标孕周分布（去标准化）
        analyzer.visualize_gestational_weeks_by_cluster(cluster_df)
        
        # 6. 生成推荐报告
        report = analyzer.generate_nipt_recommendations(segments)
        
        # 7. 保存结果
        analyzer.save_results(cluster_df, segments)
        
        print("\n=== 分析完成 ===")
        print("所有结果文件已保存到输出目录")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()