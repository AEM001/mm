import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

from gamm_y_chromosome_prediction import GAMMYChromosomePredictor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_chinese_font import set_chinese_font
set_chinese_font()

class BMISegmentationAnalyzer:
    def __init__(self, output_dir='./output/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.gamm_predictor = GAMMYChromosomePredictor()
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.split_points = None
        self.cluster_results = None
        self.optimal_k = None
        self.bmi_segments = None
        
    def load_and_prepare_data(self, data_path):
        df = self.gamm_predictor.load_and_preprocess_data(data_path)
        target_df = self.gamm_predictor.extract_target_variable(df)
        target_df = target_df.dropna(subset=['达标孕周'])
        feature_cols = self.gamm_predictor.feature_names
        feature_nan_counts = target_df[feature_cols].isna().sum()
        if feature_nan_counts.sum() > 0:
            target_df = target_df.dropna(subset=feature_cols)
        X = target_df[feature_cols]
        y = target_df['达标孕周']
        return df, target_df, X, y
    
    def train_gamm_and_predict(self, X, y, target_df):
        self.gamm_predictor.fit_gamm_model(X, y, patient_ids=target_df['孕妇代码'])
        predictions = self.gamm_predictor.predict(X)
        result_df = target_df.copy()
        result_df['预测达标孕周'] = predictions
        result_df['预测误差'] = result_df['预测达标孕周'] - result_df['达标孕周']
        return result_df
    
    def determine_optimal_clusters(self, data_for_clustering, max_k=10):
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

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚类数量选择评估指标', fontsize=16, fontweight='bold')

        axes[0, 0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('轮廓系数 (越高越好)', fontweight='bold')
        axes[0, 0].set_xlabel('聚类数量 (k)')
        axes[0, 0].set_ylabel('轮廓系数')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(k_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Calinski-Harabasz指数 (越高越好)', fontweight='bold')
        axes[0, 1].set_xlabel('聚类数量 (k)')
        axes[0, 1].set_ylabel('CH指数')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(k_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Davies-Bouldin指数 (越低越好)', fontweight='bold')
        axes[1, 0].set_xlabel('聚类数量 (k)')
        axes[1, 0].set_ylabel('DB指数')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(k_range, inertias, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('肘部法则 (寻找拐点)', fontweight='bold')
        axes[1, 1].set_xlabel('聚类数量 (k)')
        axes[1, 1].set_ylabel('簇内平方和')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}kmeans_cluster_selection.png', dpi=300, bbox_inches='tight')
        plt.show()

        silhouette_norm = np.array(silhouette_scores) / max(silhouette_scores)
        calinski_norm = np.array(calinski_scores) / max(calinski_scores)
        davies_bouldin_norm = 1 - (np.array(davies_bouldin_scores) / max(davies_bouldin_scores))
        composite_scores = silhouette_norm + calinski_norm + davies_bouldin_norm
        optimal_k = k_range[np.argmax(composite_scores)]
        return optimal_k
    
    def perform_one_dimensional_clustering(self, prediction_df, optimal_k=None):
        clustering_features = prediction_df[['BMI_标准化', '预测达标孕周']].values
        if optimal_k is None:
            optimal_k = self.determine_optimal_clusters(clustering_features)
        self.optimal_k = optimal_k
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_features)
        self.cluster_centers = kmeans.cluster_centers_
        result_df = prediction_df.copy()
        result_df['聚类标签'] = cluster_labels
        silhouette_avg = silhouette_score(clustering_features, cluster_labels)
        calinski_score = calinski_harabasz_score(clustering_features, cluster_labels)
        davies_bouldin = davies_bouldin_score(clustering_features, cluster_labels)
        self.cluster_results = result_df
        return result_df
    
    def kmeans_1d(self, data, k, max_iter=100, tol=1e-4):
        n = len(data)
        centers = self.initialize_centers_1d(data, k)
        for iteration in range(max_iter):
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                distances = [abs(data[i] - center) for center in centers]
                labels[i] = np.argmin(distances)
            new_centers = []
            for j in range(k):
                cluster_points = data[labels == j]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points))
                else:
                    new_centers.append(centers[j])
            center_shift = sum(abs(new_centers[j] - centers[j]) for j in range(k))
            if center_shift < tol:
                break
            centers = new_centers
        sorted_centers = sorted(centers)
        split_points = []
        for i in range(len(sorted_centers) - 1):
            split_point = (sorted_centers[i] + sorted_centers[i + 1]) / 2
            split_points.append(split_point)
        return labels, centers, split_points
    
    def initialize_centers_1d(self, data, k):
        np.random.seed(42)
        centers = []
        centers.append(np.random.choice(data))
        for _ in range(k - 1):
            distances = []
            for point in data:
                min_dist = min(abs(point - center) for center in centers)
                distances.append(min_dist ** 2)
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
        wcss = 0
        for i, center in enumerate(centers):
            cluster_points = data[labels == i]
            wcss += np.sum((cluster_points - center) ** 2)
        return wcss

    def determine_optimal_clusters_1d(self, data, max_k=10):
        k_range = range(2, min(max_k + 1, len(np.unique(data)) + 1))
        wcss_scores = []
        for k in k_range:
            labels, centers, _ = self.kmeans_1d(data, k)
            wcss = self.calculate_wcss_1d(data, labels, centers)
            wcss_scores.append(wcss)
        if len(wcss_scores) >= 3:
            second_diffs = []
            for i in range(1, len(wcss_scores) - 1):
                second_diff = wcss_scores[i-1] - 2*wcss_scores[i] + wcss_scores[i+1]
                second_diffs.append(second_diff)
            elbow_idx = np.argmax(second_diffs) + 1
            optimal_k = k_range[elbow_idx]
        else:
            optimal_k = k_range[len(k_range) // 2]
        return optimal_k

    def perform_clustering(self, prediction_df, optimal_k=None):
        return self.perform_one_dimensional_clustering(prediction_df, optimal_k)
    
    def analyze_bmi_segments(self, cluster_df):
        segments = {}
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            bmi_std = cluster_data['BMI_标准化']
            bmi_min = bmi_std.min()
            bmi_max = bmi_std.max()
            bmi_mean = bmi_std.mean()
            bmi_median = bmi_std.median()
            pred_weeks = cluster_data['预测达标孕周']
            pred_min = pred_weeks.min()
            pred_max = pred_weeks.max()
            pred_mean = pred_weeks.mean()
            pred_median = pred_weeks.median()
            time_mean = 16.846
            time_std = 4.076
            pred_weeks_original = pred_weeks * time_std + time_mean
            pred_mean_original = pred_mean * time_std + time_mean
            within_12_weeks = (pred_weeks_original <= 12).sum() / len(pred_weeks_original)
            within_27_weeks = (pred_weeks_original <= 27).sum() / len(pred_weeks_original)
            if within_12_weeks >= 0.8:
                target_week = 12
                optimal_percentile = (pred_weeks_original <= target_week).mean()
                optimal_percentile = min(0.95, max(0.5, optimal_percentile))
            elif within_27_weeks >= 0.7:
                target_week = 27
                optimal_percentile = (pred_weeks_original <= target_week).mean()
                optimal_percentile = min(0.95, max(0.5, optimal_percentile))
            else:
                optimal_percentile = 0.90
                target_week = pred_weeks_original.quantile(optimal_percentile)
            pred_optimal_percentile = pred_weeks.quantile(optimal_percentile)
            pred_optimal_percentile_original = pred_optimal_percentile * time_std + time_mean
            success_rate = (~cluster_data['is_censored']).mean() if 'is_censored' in cluster_data.columns else 1.0
            if pred_mean <= -0.5:
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
        return segments
    
    def visualize_gestational_weeks_by_cluster(self, cluster_df):
        time_mean = 16.846
        time_std = 4.076
        cluster_df = cluster_df.copy()
        cluster_df['预测达标孕周_原始'] = cluster_df['预测达标孕周'] * time_std + time_mean
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        n_clusters = len(cluster_ids)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('各群组Y染色体浓度达标孕周分析', fontsize=16, fontweight='bold')
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
        ax1.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax1.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        ax1.legend()
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
        ax2.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax2.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        ax2.legend()
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
        ax3.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周参考线')
        ax3.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周参考线')
        ax4 = axes[1, 1]
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

        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        ax4.axhline(y=12, color='red', linestyle='--', alpha=0.7, label='12周')
        ax4.axhline(y=27, color='orange', linestyle='--', alpha=0.7, label='27周')
        plt.tight_layout()
        output_path = f"{self.output_dir}gestational_weeks_by_cluster_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_clustering_results(self, cluster_df):
        fig = plt.figure(figsize=(20, 15))
        ax1 = plt.subplot(3, 3, (1, 2))
        scatter = ax1.scatter(cluster_df['BMI_标准化'], cluster_df['预测达标孕周'],
                             c=cluster_df['聚类标签'], cmap='viridis',
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('BMI (标准化)', fontweight='bold')
        ax1.set_ylabel('预测达标孕周 (标准化)', fontweight='bold')
        ax1.set_title('基于BMI和预测达标孕周的二维K-Means聚类结果', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        cluster_ids = sorted(cluster_df['聚类标签'].unique())
        if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
            for i, center in enumerate(self.cluster_centers):
                ax1.scatter(center[0], center[1], c='red', marker='x', s=200, linewidths=3)
                ax1.text(center[0], center[1] + 0.1, f'中心{i+1}', ha='center', va='bottom',
                        fontsize=10, color='red', fontweight='bold')
        ax1.scatter([], [], c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        ax1.legend()
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('聚类标签', fontweight='bold')
        ax2 = plt.subplot(3, 3, 3)
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            ax2.hist(cluster_data['BMI_标准化'], alpha=0.6, label=f'群组{cluster_id+1}', bins=15)
        ax2.set_xlabel('BMI (标准化)', fontweight='bold')
        ax2.set_ylabel('频数', fontweight='bold')
        ax2.set_title('各群组BMI分布', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = plt.subplot(3, 3, 4)
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            ax3.hist(cluster_data['预测达标孕周'], alpha=0.6, label=f'群组{cluster_id+1}', bins=15)
        ax3.set_xlabel('预测达标孕周 (标准化)', fontweight='bold')
        ax3.set_ylabel('频数', fontweight='bold')
        ax3.set_title('各群组预测达标孕周分布', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4 = plt.subplot(3, 3, 5)
        cluster_labels = [f'群组{i+1}' for i in sorted(cluster_df['聚类标签'].unique())]
        bmi_by_cluster = [cluster_df[cluster_df['聚类标签'] == i]['BMI_标准化'] for i in sorted(cluster_df['聚类标签'].unique())]
        ax4.boxplot(bmi_by_cluster, labels=cluster_labels)
        ax4.set_ylabel('BMI (标准化)', fontweight='bold')
        ax4.set_title('各群组BMI箱线图', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax5 = plt.subplot(3, 3, 6)
        pred_by_cluster = [cluster_df[cluster_df['聚类标签'] == i]['预测达标孕周'] for i in sorted(cluster_df['聚类标签'].unique())]
        ax5.boxplot(pred_by_cluster, labels=cluster_labels)
        ax5.set_ylabel('预测达标孕周 (标准化)', fontweight='bold')
        ax5.set_title('各群组预测达标孕周箱线图', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax6 = plt.subplot(3, 3, 7)
        达标比例 = []
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_data = cluster_df[cluster_df['聚类标签'] == cluster_id]
            达标比例.append((~cluster_data['is_censored']).mean() * 100)
        bars = ax6.bar(cluster_labels, 达标比例, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(cluster_labels)])
        ax6.set_ylabel('达标比例 (%)', fontweight='bold')
        ax6.set_title('各群组Y染色体达标比例', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        for bar, ratio in zip(bars, 达标比例):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax7 = plt.subplot(3, 3, 8)
        sample_counts = [len(cluster_df[cluster_df['聚类标签'] == i]) for i in sorted(cluster_df['聚类标签'].unique())]
        bars = ax7.bar(cluster_labels, sample_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(cluster_labels)])
        ax7.set_ylabel('样本数量', fontweight='bold')
        ax7.set_title('各群组样本数量分布', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        for bar, count in zip(bars, sample_counts):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        ax8 = plt.subplot(3, 3, 9)
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
    

    def save_results(self, cluster_df, segments):
        cluster_df.to_csv(f'{self.output_dir}kmeans_clustering_detailed_results.csv',
                          index=False, encoding='utf-8')
        segments_df = pd.DataFrame(segments).T
        segments_df.to_csv(f'{self.output_dir}bmi_segments_summary.csv',
                          encoding='utf-8')
        segments_df.to_excel(f'{self.output_dir}bmi_segments_summary.xlsx',
                           index=False)

def main():
    analyzer = BMISegmentationAnalyzer()
    data_path = '/Users/Mac/Downloads/mm/3/processed_data.csv'
    df, target_df, X, y = analyzer.load_and_prepare_data(data_path)
    prediction_df = analyzer.train_gamm_and_predict(X, y, target_df)
    cluster_df = analyzer.perform_clustering(prediction_df)
    segments = analyzer.analyze_bmi_segments(cluster_df)
    analyzer.visualize_clustering_results(cluster_df)
    analyzer.visualize_gestational_weeks_by_cluster(cluster_df)
    analyzer.save_results(cluster_df, segments)

if __name__ == "__main__":
    main()