import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from set_chinese_font import set_chinese_font
set_chinese_font()

from kmeans_bmi_segmentation import BMISegmentationAnalyzer


class Q3DetectionErrorAnalyzer:

    def __init__(self, output_dir='.', threshold_logit=-3.178054):
        self.output_dir = output_dir
        self.error_dir = os.path.join(output_dir, 'detection_error_analysis')
        os.makedirs(self.error_dir, exist_ok=True)

        self.threshold_logit = threshold_logit
        self.threshold_percentage = 0.04

        self.cluster_noise_metrics = {}
        self.group_comparison = {}
        self.gamm_error_analysis = {}
        self.stability_analysis = {}
        self.simulation_results = {}

    def estimate_cluster_specific_noise(self, raw_df, cluster_df, col_woman='孕妇代码',
                                        col_y='Y染色体浓度', col_week='孕周_标准化'):
        woman_to_cluster = dict(zip(cluster_df['孕妇代码'], cluster_df['聚类标签']))
        
        cluster_noise_data = []
        cluster_summaries = {}

        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_women = cluster_df[cluster_df['聚类标签'] == cluster_id]['孕妇代码'].tolist()
            cluster_raw_data = raw_df[raw_df[col_woman].isin(cluster_women)]
            cluster_noise_list = []
            for woman_code, woman_data in cluster_raw_data.groupby(col_woman):
                if len(woman_data) < 3:
                    continue
                if col_week in woman_data.columns:
                    woman_data = woman_data.sort_values(col_week).reset_index(drop=True)
                    
                y_values = woman_data[col_y].values
                weeks = woman_data[col_week].values
                noise_metrics = self._estimate_individual_noise(y_values, weeks, woman_code)
                noise_metrics['聚类标签'] = cluster_id
                noise_metrics['聚类组'] = f"第{cluster_id+1}组"
                cluster_noise_list.append(noise_metrics)
                cluster_noise_data.append(noise_metrics)
            if cluster_noise_list:
                cluster_df_temp = pd.DataFrame(cluster_noise_list)
                cluster_summaries[f"第{cluster_id+1}组"] = {
                    '组ID': cluster_id,
                    '样本数': len(cluster_df_temp),
                    '平均测量次数': cluster_df_temp['测量次数'].mean(),
                    'BMI范围': f"标准化[{cluster_df[cluster_df['聚类标签']==cluster_id]['BMI_标准化'].min():.3f}, {cluster_df[cluster_df['聚类标签']==cluster_id]['BMI_标准化'].max():.3f}]",
                    '噪声特征': {
                        'GAMM残差噪声': {
                            '均值': cluster_df_temp['噪声_GAMM残差'].mean(),
                            '中位数': cluster_df_temp['噪声_GAMM残差'].median(),
                            '标准差': cluster_df_temp['噪声_GAMM残差'].std(),
                        },
                        '时间相关噪声': {
                            '均值': cluster_df_temp['噪声_时间相关'].mean(),
                            '中位数': cluster_df_temp['噪声_时间相关'].median(),
                            '标准差': cluster_df_temp['噪声_时间相关'].std(),
                        },
                        '聚类内变异': {
                            '均值': cluster_df_temp['噪声_聚类内变异'].mean(),
                            '中位数': cluster_df_temp['噪声_聚类内变异'].median(),
                            '标准差': cluster_df_temp['噪声_聚类内变异'].std(),
                        }
                    },
                    '信噪比': {
                        'SNR_GAMM': cluster_df_temp['SNR_GAMM'].mean(),
                        'SNR_时间': cluster_df_temp['SNR_时间'].mean(),
                        'SNR_聚类': cluster_df_temp['SNR_聚类'].mean(),
                    }
                }

        cluster_noise_df = pd.DataFrame(cluster_noise_data)
        cluster_noise_df.to_csv(os.path.join(self.error_dir, 'cluster_specific_noise_details.csv'), 
                               index=False, encoding='utf-8')
        
        self.cluster_noise_metrics = cluster_summaries
        return cluster_summaries, cluster_noise_df
    
    def _estimate_individual_noise(self, y_values, weeks, woman_code):
        n_measurements = len(y_values)
        signal_std = np.std(y_values)
        if n_measurements >= 4:  # 需要至少4个点进行样条插值
            from scipy.interpolate import UnivariateSpline
            try:
                spline = UnivariateSpline(weeks, y_values, s=0.1, k=min(3, n_measurements-1))
                y_gamm_fitted = spline(weeks)
                residuals_gamm = y_values - y_gamm_fitted
                noise_gamm = np.std(residuals_gamm)
            except:
                noise_gamm = np.std(y_values)  # 备用方法
        else:
            noise_gamm = np.std(y_values) if n_measurements > 1 else 0.0
        if n_measurements >= 3:
            weights = 1.0 / (1.0 + 0.1 * np.abs(np.diff(weeks, prepend=weeks[0])))
            y_weighted = np.average(y_values, weights=weights)
            residuals_time = y_values - y_weighted
            noise_time = np.std(residuals_time)
        else:
            noise_time = np.nan
        individual_mean = np.mean(y_values)
        residuals_cluster = y_values - individual_mean
        noise_cluster = np.std(residuals_cluster)
        
        return {
            '孕妇代码': woman_code,
            '测量次数': n_measurements,
            '噪声_GAMM残差': noise_gamm,
            '噪声_时间相关': noise_time,
            '噪声_聚类内变异': noise_cluster,
            'SNR_GAMM': signal_std / noise_gamm if not np.isnan(noise_gamm) and noise_gamm > 0 else np.nan,
            'SNR_时间': signal_std / noise_time if not np.isnan(noise_time) and noise_time > 0 else np.nan,
            'SNR_聚类': signal_std / noise_cluster if noise_cluster > 0 else np.nan,
        }

    def analyze_group_differences(self, cluster_noise_df):
        from scipy.stats import f_oneway, kruskal
        groups = cluster_noise_df.groupby('聚类组')
        noise_gamm_groups = [group['噪声_GAMM残差'].dropna() for _, group in groups]
        noise_time_groups = [group['噪声_时间相关'].dropna() for _, group in groups]
        noise_cluster_groups = [group['噪声_聚类内变异'].dropna() for _, group in groups]
        comparison_results = {}
        if len(noise_gamm_groups) > 1 and all(len(g) > 0 for g in noise_gamm_groups):
            f_stat_gamm, p_val_gamm = f_oneway(*noise_gamm_groups)
            kruskal_stat_gamm, kruskal_p_gamm = kruskal(*noise_gamm_groups)
            comparison_results['GAMM残差噪声'] = {
                'ANOVA_F值': f_stat_gamm,
                'ANOVA_p值': p_val_gamm,
                'Kruskal_H值': kruskal_stat_gamm,
                'Kruskal_p值': kruskal_p_gamm,
                '显著性': 'significant' if p_val_gamm < 0.05 else 'not_significant'
            }
        
        if len(noise_time_groups) > 1 and all(len(g) > 0 for g in noise_time_groups):
            f_stat_time, p_val_time = f_oneway(*noise_time_groups)
            kruskal_stat_time, kruskal_p_time = kruskal(*noise_time_groups)
            comparison_results['时间相关噪声'] = {
                'ANOVA_F值': f_stat_time,
                'ANOVA_p值': p_val_time,
                'Kruskal_H值': kruskal_stat_time,
                'Kruskal_p值': kruskal_p_time,
                '显著性': 'significant' if p_val_time < 0.05 else 'not_significant'
            }
        if len(noise_cluster_groups) > 1 and all(len(g) > 0 for g in noise_cluster_groups):
            f_stat_cluster, p_val_cluster = f_oneway(*noise_cluster_groups)
            kruskal_stat_cluster, kruskal_p_cluster = kruskal(*noise_cluster_groups)
            comparison_results['聚类内变异'] = {
                'ANOVA_F值': f_stat_cluster,
                'ANOVA_p值': p_val_cluster,
                'Kruskal_H值': kruskal_stat_cluster,
                'Kruskal_p值': kruskal_p_cluster,
                '显著性': 'significant' if p_val_cluster < 0.05 else 'not_significant'
            }
        pairwise_comparisons = {}
        group_names = list(groups.groups.keys())
        
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                group1_name, group2_name = group_names[i], group_names[j]
                group1_data = groups.get_group(group1_name)
                group2_data = groups.get_group(group2_name)
                
                pair_key = f"{group1_name} vs {group2_name}"
                pairwise_comparisons[pair_key] = {}
                if not group1_data['噪声_GAMM残差'].isna().all() and not group2_data['噪声_GAMM残差'].isna().all():
                    g1_gamm = group1_data['噪声_GAMM残差'].dropna()
                    g2_gamm = group2_data['噪声_GAMM残差'].dropna()
                    cohens_d_gamm = self._calculate_cohens_d(g1_gamm, g2_gamm)
                    pairwise_comparisons[pair_key]['GAMM残差_Cohens_d'] = cohens_d_gamm
                if not group1_data['噪声_时间相关'].isna().all() and not group2_data['噪声_时间相关'].isna().all():
                    g1_time = group1_data['噪声_时间相关'].dropna()
                    g2_time = group2_data['噪声_时间相关'].dropna()
                    cohens_d_time = self._calculate_cohens_d(g1_time, g2_time)
                    pairwise_comparisons[pair_key]['时间相关_Cohens_d'] = cohens_d_time
        group_comparison = {
            '统计检验结果': comparison_results,
            '配对比较': pairwise_comparisons,
            '组间差异摘要': self._summarize_group_differences(cluster_noise_df)
        }
        comparison_df = self._create_comparison_dataframe(comparison_results, pairwise_comparisons)
        comparison_df.to_csv(os.path.join(self.error_dir, 'group_noise_comparison.csv'),
                           index=False, encoding='utf-8')
        self.group_comparison = group_comparison
        return group_comparison
    
    def _calculate_cohens_d(self, group1, group2):
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return np.nan
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        if pooled_std == 0:
            return np.nan
        
        return (mean1 - mean2) / pooled_std
    
    def _summarize_group_differences(self, cluster_noise_df):
        summary = {}
        
        for group_name in cluster_noise_df['聚类组'].unique():
            group_data = cluster_noise_df[cluster_noise_df['聚类组'] == group_name]
            summary[group_name] = {
                '样本数': len(group_data),
                'GAMM残差噪声': {
                    '均值': group_data['噪声_GAMM残差'].mean(),
                    '标准差': group_data['噪声_GAMM残差'].std(),
                    '中位数': group_data['噪声_GAMM残差'].median(),
                },
                '时间相关噪声': {
                    '均值': group_data['噪声_时间相关'].mean(),
                    '标准差': group_data['噪声_时间相关'].std(),
                    '中位数': group_data['噪声_时间相关'].median(),
                },
                '聚类内变异': {
                    '均值': group_data['噪声_聚类内变异'].mean(),
                    '标准差': group_data['噪声_聚类内变异'].std(),
                    '中位数': group_data['噪声_聚类内变异'].median(),
                }
            }
        
        return summary
    
    def _create_comparison_dataframe(self, comparison_results, pairwise_comparisons):
        rows = []
        
        for noise_type, result in comparison_results.items():
            rows.append({
                '比较类型': '总体检验',
                '噪声类型': noise_type,
                '统计量': f"F={result['ANOVA_F值']:.4f}",
                'p值': result['ANOVA_p值'],
                '显著性': result['显著性'],
                '效应大小': None
            })
        
        for pair, comparisons in pairwise_comparisons.items():
            for noise_type, cohens_d in comparisons.items():
                rows.append({
                    '比较类型': '配对比较',
                    '组别对比': pair,
                    '噪声类型': noise_type.replace('_Cohens_d', ''),
                    '统计量': f"Cohen's d={cohens_d:.4f}",
                    'p值': None,
                    '显著性': None,
                    '效应大小': cohens_d
                })
        
        return pd.DataFrame(rows)

    def gamm_driven_error_simulation(self, cluster_df, prediction_df, n_simulations=1000):
        results = []
        sim_details = []
        for cluster_id in sorted(cluster_df['聚类标签'].unique()):
            cluster_group = cluster_df[cluster_df['聚类标签'] == cluster_id]
            cluster_predictions = prediction_df[prediction_df['孕妇代码'].isin(cluster_group['孕妇代码'])]
            cluster_key = f"第{cluster_id+1}组"
            if cluster_key in self.cluster_noise_metrics:
                noise_gamm = self.cluster_noise_metrics[cluster_key]['噪声特征']['GAMM残差噪声']['均值']
                noise_time = self.cluster_noise_metrics[cluster_key]['噪声特征']['时间相关噪声']['均值']
                noise_cluster = self.cluster_noise_metrics[cluster_key]['噪声特征']['聚类内变异']['均值']
            else:
                noise_gamm = 0.1
                noise_time = 0.08
                noise_cluster = 0.05
            true_target_time = cluster_predictions['预测达标孕周'].median()
            bmi_range = f"[{cluster_group['BMI_标准化'].min():.3f}, {cluster_group['BMI_标准化'].max():.3f}]"
            detected_times = []
            gamm_errors = []
            noise_contributions = []
            
            for sim_i in range(n_simulations):
                gamm_error = np.random.normal(0, noise_gamm)
                time_noise = np.random.normal(0, noise_time)
                cluster_noise = np.random.normal(0, noise_cluster)
                total_noise = np.sqrt(gamm_error**2 + time_noise**2 + cluster_noise**2)
                simulated_target_time = true_target_time + total_noise
                time_mean = 16.846
                time_std = 4.076
                detected_time_original = simulated_target_time * time_std + time_mean
                detected_time_original = np.clip(detected_time_original, 10, 30)
                
                detected_times.append(detected_time_original)
                gamm_errors.append(gamm_error)
                noise_contributions.append({
                    'GAMM误差': gamm_error,
                    '时间噪声': time_noise, 
                    '聚类噪声': cluster_noise,
                    '总噪声': total_noise
                })
            
            detected_times = np.array(detected_times)
            true_time_original = true_target_time * time_std + time_mean
            bias = np.mean(detected_times) - true_time_original
            rmse = np.sqrt(np.mean((detected_times - true_time_original) ** 2))
            mae = np.mean(np.abs(detected_times - true_time_original))
            group_result = {
                '聚类组': cluster_key,
                '组ID': cluster_id,
                'BMI范围(标准化)': bmi_range,
                '样本数': len(cluster_group),
                '真实达标时间(周)': true_time_original,
                '预测均值(周)': np.mean(detected_times),
                '预测中位数(周)': np.median(detected_times),
                '时间偏差(周)': bias,
                'RMSE(周)': rmse,
                'MAE(周)': mae,
                '预测标准差(周)': np.std(detected_times),
                'CI_2.5%(周)': np.percentile(detected_times, 2.5),
                'CI_97.5%(周)': np.percentile(detected_times, 97.5),
                '噪声特征': {
                    'GAMM残差噪声': noise_gamm,
                    '时间相关噪声': noise_time,
                    '聚类内变异噪声': noise_cluster,
                    '平均总噪声': np.mean([nc['总噪声'] for nc in noise_contributions])
                },
                '检测性能': {
                    '10周内检出率': np.mean(detected_times <= 10),
                    '15周内检出率': np.mean(detected_times <= 15),
                    '20周内检出率': np.mean(detected_times <= 20),
                    '有效检测率': np.mean(detected_times <= 28)
                }
            }
            
            results.append(group_result)
            
            for i, (dt, gc, nc) in enumerate(zip(detected_times, gamm_errors, noise_contributions)):
                sim_details.append({
                    '聚类组': cluster_key,
                    '模拟次数': i+1,
                    '检测时间(周)': dt,
                    '时间偏差(周)': dt - true_time_original,
                    'GAMM误差': gc,
                    '时间噪声': nc['时间噪声'],
                    '聚类噪声': nc['聚类噪声'],
                    '总噪声': nc['总噪声']
                })
        sim_df = pd.DataFrame(results)
        sim_detail_df = pd.DataFrame(sim_details)
        
        sim_df.to_csv(os.path.join(self.error_dir, 'gamm_cluster_simulation_summary.csv'), 
                     index=False, encoding='utf-8')
        sim_detail_df.to_csv(os.path.join(self.error_dir, 'gamm_cluster_simulation_details.csv'),
                           index=False, encoding='utf-8')
        overall_analysis = {
            '聚类间差异': {
                '时间偏差范围': f"[{sim_df['时间偏差(周)'].min():+.1f}, {sim_df['时间偏差(周)'].max():+.1f}]周",
                '平均RMSE': sim_df['RMSE(周)'].mean(),
                '最大RMSE': sim_df['RMSE(周)'].max(),
                '偏差标准差': sim_df['时间偏差(周)'].std()
            },
            '噪声贡献分析': {
                'GAMM误差占比': np.mean([r['噪声特征']['GAMM残差噪声'] for r in results]),
                '时间噪声占比': np.mean([r['噪声特征']['时间相关噪声'] for r in results]),
                '聚类噪声占比': np.mean([r['噪声特征']['聚类内变异噪声'] for r in results])
            },
            '各组结果': results
        }
        self.gamm_error_analysis = overall_analysis
        self.simulation_results = overall_analysis
        return overall_analysis, sim_detail_df
    
    def analyze_cluster_stability(self, raw_df, cluster_df, bmi_analyzer, n_bootstrap=100):
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        original_labels = cluster_df['聚类标签'].values
        n_clusters = len(np.unique(original_labels))
        stability_metrics = {
            'ARI_scores': [],
            'NMI_scores': [],
            'cluster_consistency': [],
            'noise_levels': [],
            'perturbed_clusters': []
        }
        if self.cluster_noise_metrics:
            avg_noise = np.mean([
                group['噪声特征']['GAMM残差噪声']['均值']
                for group in self.cluster_noise_metrics.values()
            ])
        else:
            avg_noise = 0.1
        noise_levels = np.linspace(0.5, 2.0, 5) * avg_noise
        
        for noise_level in noise_levels:
            level_ari_scores = []
            level_nmi_scores = []
            level_consistency = []
            
            for bootstrap_i in range(n_bootstrap):
                perturbed_df = raw_df.copy()
                noise = np.random.normal(0, noise_level, len(perturbed_df))
                perturbed_df['Y染色体浓度'] = perturbed_df['Y染色体浓度'] + noise

                target_data = []
                for woman_code, woman_data in perturbed_df.groupby('孕妇代码'):
                    qualified = woman_data[woman_data['Y染色体浓度'] >= np.log(0.04/(1-0.04))]
                    if len(qualified) > 0:
                        first_qualified_week = qualified['孕周_标准化'].min()
                    else:
                        first_qualified_week = woman_data['孕周_标准化'].max() + 1
                    bmi_info = cluster_df[cluster_df['孕妇代码'] == woman_code]
                    if len(bmi_info) > 0:
                        target_data.append({
                            '孕妇代码': woman_code,
                            'BMI_标准化': bmi_info['BMI_标准化'].iloc[0],
                            '预测达标孕周': first_qualified_week,
                            '年龄_标准化': bmi_info.get('年龄_标准化', [0]).iloc[0]
                        })

                if len(target_data) < 10:
                    continue

                perturbed_target_df = pd.DataFrame(target_data)

                perturbed_cluster_df = bmi_analyzer.perform_clustering(
                    perturbed_target_df,
                    optimal_k=n_clusters
                )

                common_women = set(cluster_df['孕妇代码']) & set(perturbed_cluster_df['孕妇代码'])
                if len(common_women) < 10:
                    continue

                original_matched = cluster_df[cluster_df['孕妇代码'].isin(common_women)].sort_values('孕妇代码')
                perturbed_matched = perturbed_cluster_df[perturbed_cluster_df['孕妇代码'].isin(common_women)].sort_values('孕妇代码')

                orig_labels = original_matched['聚类标签'].values
                pert_labels = perturbed_matched['聚类标签'].values

                ari = adjusted_rand_score(orig_labels, pert_labels)
                nmi = normalized_mutual_info_score(orig_labels, pert_labels)

                consistency = np.mean(orig_labels == pert_labels)

                level_ari_scores.append(ari)
                level_nmi_scores.append(nmi)
                level_consistency.append(consistency)

            if level_ari_scores:
                stability_metrics['ARI_scores'].append(np.mean(level_ari_scores))
                stability_metrics['NMI_scores'].append(np.mean(level_nmi_scores))
                stability_metrics['cluster_consistency'].append(np.mean(level_consistency))
                stability_metrics['noise_levels'].append(noise_level)
        stability_analysis = {
            '稳定性指标': {
                '平均ARI': np.mean(stability_metrics['ARI_scores']) if stability_metrics['ARI_scores'] else 0,
                '平均NMI': np.mean(stability_metrics['NMI_scores']) if stability_metrics['NMI_scores'] else 0,
                '平均一致性': np.mean(stability_metrics['cluster_consistency']) if stability_metrics['cluster_consistency'] else 0,
                'ARI标准差': np.std(stability_metrics['ARI_scores']) if stability_metrics['ARI_scores'] else 0,
                'NMI标准差': np.std(stability_metrics['NMI_scores']) if stability_metrics['NMI_scores'] else 0,
            },
            '噪声敏感性': {
                '噪声水平范围': f"[{min(stability_metrics['noise_levels']):.6f}, {max(stability_metrics['noise_levels']):.6f}]" if stability_metrics['noise_levels'] else "无数据",
                'ARI下降趋势': self._calculate_trend(stability_metrics['noise_levels'], stability_metrics['ARI_scores']),
                'NMI下降趋势': self._calculate_trend(stability_metrics['noise_levels'], stability_metrics['NMI_scores']),
                '稳定性阈值': self._find_stability_threshold(stability_metrics)
            },
            '详细结果': stability_metrics,
            '稳定性评估': self._assess_stability(stability_metrics)
        }
        
        # 保存稳定性分析结果
        stability_df = pd.DataFrame({
            '噪声水平': stability_metrics['noise_levels'],
            'ARI分数': stability_metrics['ARI_scores'],
            'NMI分数': stability_metrics['NMI_scores'],
            '聚类一致性': stability_metrics['cluster_consistency']
        })
        stability_df.to_csv(os.path.join(self.error_dir, 'cluster_stability_analysis.csv'), 
                          index=False, encoding='utf-8')
        
        self.stability_analysis = stability_analysis
        
        return stability_analysis
    
    def _calculate_trend(self, x_values, y_values):
        if len(x_values) < 2 or len(y_values) < 2:
            return 0
        slope = np.polyfit(x_values, y_values, 1)[0]
        return slope
    
    def _find_stability_threshold(self, metrics):
        threshold = 0.7
        for i, (ari, nmi) in enumerate(zip(metrics['ARI_scores'], metrics['NMI_scores'])):
            if ari < threshold or nmi < threshold:
                return metrics['noise_levels'][i] if metrics['noise_levels'] else None
        return None
    
    def _assess_stability(self, metrics):
        if not metrics['ARI_scores']:
            return "无法评估"
        
        avg_ari = np.mean(metrics['ARI_scores'])
        avg_nmi = np.mean(metrics['NMI_scores'])
        
        if avg_ari >= 0.8 and avg_nmi >= 0.8:
            return "高度稳定"
        elif avg_ari >= 0.6 and avg_nmi >= 0.6:
            return "中等稳定"
        elif avg_ari >= 0.4 and avg_nmi >= 0.4:
            return "低稳定性"
        else:
            return "不稳定"


def create_q3_specialized_visualizations(analyzer, cluster_noise_df, sim_detail_df):
    from set_chinese_font import set_chinese_font
    set_chinese_font()

    fig, axes = plt.subplots(3, 3, figsize=(24, 20))

    ax1 = axes[0, 0]
    groups = sorted(cluster_noise_df['聚类组'].unique())
    noise_types = ['噪声_GAMM残差', '噪声_时间相关', '噪声_聚类内变异']
    noise_type_labels = ['GAMM残差', '时间相关', '聚类内变异']

    positions = []
    noise_data = []
    box_labels = []
    pos = 1
    
    for i, noise_type in enumerate(noise_types):
        type_data = []
        for group in groups:
            group_data = cluster_noise_df[cluster_noise_df['聚类组'] == group][noise_type].dropna()
            if len(group_data) > 0:
                type_data.extend(group_data.values)
        
        if type_data:
            noise_data.append(type_data)
            positions.append(pos)
            box_labels.append(noise_type_labels[i])
            pos += 1
    
    if noise_data:
        bp = ax1.boxplot(noise_data, positions=positions, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax1.set_ylabel('噪声水平')
    ax1.set_title('聚类特异性噪声分布对比')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(box_labels)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if analyzer.group_comparison and '统计检验结果' in analyzer.group_comparison:
        comparison_results = analyzer.group_comparison['统计检验结果']
        noise_types_short = ['GAMM残差', '时间相关', '聚类内变异']
        p_values = []
        
        for nt in ['GAMM残差噪声', '时间相关噪声', '聚类内变异']:
            if nt in comparison_results:
                p_values.append(comparison_results[nt]['ANOVA_p值'])
            else:
                p_values.append(1.0)

        significance_matrix = np.array(p_values).reshape(1, -1)
        im = ax2.imshow(significance_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        ax2.set_xticks(range(len(noise_types_short)))
        ax2.set_xticklabels(noise_types_short)
        ax2.set_yticks([0])
        ax2.set_yticklabels(['p值'])
        ax2.set_title('组间差异显著性')
        ax2.tick_params(axis='x', labelsize=9)

        for i, p_val in enumerate(p_values):
            ax2.text(i, 0, f'{p_val:.3f}', ha='center', va='center', 
                    color='white' if p_val < 0.05 else 'black', fontweight='bold')

    ax3 = axes[0, 2]
    if len(sim_detail_df) > 0:
        for group in sim_detail_df['聚类组'].unique():
            group_data = sim_detail_df[sim_detail_df['聚类组'] == group]
            ax3.scatter(group_data['总噪声'], group_data['时间偏差(周)'], 
                       label=group, alpha=0.6, s=30)
        ax3.set_xlabel('总噪声水平')
        ax3.set_ylabel('时间偏差 (周)')
        ax3.set_title('噪声水平 vs 时间偏差')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    if analyzer.simulation_results and '各组结果' in analyzer.simulation_results:
        groups_results = analyzer.simulation_results['各组结果']
        group_names = [r['聚类组'] for r in groups_results]
        rmse_values = [r['RMSE(周)'] for r in groups_results]
        mae_values = [r['MAE(周)'] for r in groups_results]
        
        x_pos = np.arange(len(group_names))
        width = 0.35
        ax4.bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.7, color='skyblue')
        ax4.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.7, color='lightcoral')
        ax4.set_xlabel('聚类组')
        ax4.set_ylabel('误差 (周)')
        ax4.set_title('各聚类组预测误差对比')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(group_names)
        ax4.tick_params(axis='x', labelsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    if analyzer.simulation_results and '各组结果' in analyzer.simulation_results:
        groups_results = analyzer.simulation_results['各组结果']
        group_names = [r['聚类组'] for r in groups_results]
        biases = [r['时间偏差(周)'] for r in groups_results]
        
        colors = ['red' if b > 0 else 'blue' for b in biases]
        bars = ax5.bar(group_names, biases, color=colors, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.set_ylabel('时间偏差 (周)')
        ax5.set_title('各聚类组达标时间偏差')
        ax5.tick_params(axis='x', labelsize=8, rotation=45)
        ax5.grid(True, alpha=0.3)

        for bar, bias in zip(bars, biases):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, 
                    height + 0.1 if height > 0 else height - 0.1,
                    f'{bias:+.1f}', ha='center', 
                    va='bottom' if height > 0 else 'top')

    ax6 = axes[1, 2]
    if analyzer.simulation_results and '噪声贡献分析' in analyzer.simulation_results:
        contributions = analyzer.simulation_results['噪声贡献分析']
        labels = ['GAMM误差', '时间噪声', '聚类噪声']
        sizes = [contributions['GAMM误差占比'], contributions['时间噪声占比'], contributions['聚类噪声占比']]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax6.set_title('噪声贡献组成')

    ax7 = axes[2, 0]
    if analyzer.stability_analysis and '详细结果' in analyzer.stability_analysis:
        stability_metrics = analyzer.stability_analysis['详细结果']
        if stability_metrics['noise_levels']:
            ax7.plot(stability_metrics['noise_levels'], stability_metrics['ARI_scores'], 
                    'o-', label='ARI', linewidth=2, markersize=6)
            ax7.plot(stability_metrics['noise_levels'], stability_metrics['NMI_scores'], 
                    's-', label='NMI', linewidth=2, markersize=6)
            ax7.plot(stability_metrics['noise_levels'], stability_metrics['cluster_consistency'], 
                    '^-', label='一致性', linewidth=2, markersize=6)
            ax7.set_xlabel('噪声水平')
            ax7.set_ylabel('稳定性指标')
            ax7.set_title('聚类稳定性 vs 噪声水平')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='稳定性阈值')

    ax8 = axes[2, 1]
    if analyzer.simulation_results and '各组结果' in analyzer.simulation_results:
        groups_results = analyzer.simulation_results['各组结果']
        group_names = [r['聚类组'] for r in groups_results]
        
        performance_metrics = ['10周内检出率', '15周内检出率', '20周内检出率', '有效检测率']
        x_pos = np.arange(len(group_names))
        width = 0.2
        
        for i, metric in enumerate(performance_metrics):
            values = [r['检测性能'][metric] * 100 for r in groups_results]
            ax8.bar(x_pos + i*width, values, width, label=metric, alpha=0.7)
        
        ax8.set_xlabel('聚类组')
        ax8.set_ylabel('检出率 (%)')
        ax8.set_title('各聚类组检测性能对比')
        ax8.set_xticks(x_pos + width * 1.5)
        ax8.set_xticklabels(group_names)
        ax8.tick_params(axis='x', labelsize=7, rotation=30)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    ax9 = axes[2, 2]
    if analyzer.simulation_results and '各组结果' in analyzer.simulation_results:
        groups_results = analyzer.simulation_results['各组结果']
        group_names = [r['聚类组'] for r in groups_results]
        
        for i, result in enumerate(groups_results):
            true_time = result['真实达标时间(周)']
            pred_mean = result['预测均值(周)']
            ci_lower = result['CI_2.5%(周)']
            ci_upper = result['CI_97.5%(周)']

            ax9.errorbar(i, pred_mean, 
                        yerr=[[pred_mean - ci_lower], [ci_upper - pred_mean]], 
                        fmt='o', capsize=5, label='预测' if i == 0 else "", 
                        color='blue', alpha=0.7)

            ax9.scatter(i, true_time, color='red', s=50, marker='x', 
                       label='真实' if i == 0 else "")
        
        ax9.set_xlabel('聚类组')
        ax9.set_ylabel('达标时间 (周)')
        ax9.set_title('达标时间预测精度与置信区间')
        ax9.set_xticks(range(len(group_names)))
        ax9.set_xticklabels(group_names)
        ax9.tick_params(axis='x', labelsize=8, rotation=30)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = os.path.join(analyzer.error_dir, 'q3_gamm_kmeans_error_analysis.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    



def _interpret_cohens_d(cohens_d):
    if np.isnan(cohens_d):
        return "无效"
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "微小效应"
    elif abs_d < 0.5:
        return "小效应"
    elif abs_d < 0.8:
        return "中等效应"
    else:
        return "大效应"


def build_cluster_groups(cluster_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    groups = []
    for cid in sorted(cluster_df['聚类标签'].unique()):
        data = cluster_df[cluster_df['聚类标签'] == cid]
        bmi_min = float(data['BMI_标准化'].min()) if len(data) else np.nan
        bmi_max = float(data['BMI_标准化'].max()) if len(data) else np.nan
        groups.append({
            '组别': int(cid + 1),
            'BMI区间': f'BMI(标准化)∈[{bmi_min:.3f}, {bmi_max:.3f}]'
        })
    groups_df = pd.DataFrame(groups).sort_values('组别').reset_index(drop=True)
    return groups_df


def extract_true_times_by_cluster(cluster_df: pd.DataFrame) -> np.ndarray:
    time_mean = 16.846
    time_std = 4.076

    true_times = []
    for cid in sorted(cluster_df['聚类标签'].unique()):
        data = cluster_df[cluster_df['聚类标签'] == cid]
        if len(data) == 0:
            true_times.append(np.nan)
            continue
        median_std = float(data['预测达标孕周'].median())
        median_original = median_std * time_std + time_mean
        true_times.append(median_original)
    return np.array(true_times)


def main():
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    output_dir = os.path.join(SCRIPT_DIR, 'gamm_detection_error_analysis')
    os.makedirs(output_dir, exist_ok=True)
    bmi_analyzer = BMISegmentationAnalyzer(output_dir=output_dir + os.sep)
    if not getattr(bmi_analyzer.gamm_predictor, 'use_r_gamm', False):
        raise RuntimeError("R/mgcv GAMM 不可用。本分析严格依赖 R GAMM，请安装并配置 rpy2 与 R 包 mgcv。")

    raw_df, target_df, X, y = bmi_analyzer.load_and_prepare_data(data_file)
    
    prediction_df = bmi_analyzer.train_gamm_and_predict(X, y, target_df)
    
    cluster_df = bmi_analyzer.perform_clustering(prediction_df, optimal_k=None)
    
    groups_df = build_cluster_groups(cluster_df, n_clusters=len(cluster_df['聚类标签'].unique()))
    true_times = extract_true_times_by_cluster(cluster_df)
    analyzer = Q3DetectionErrorAnalyzer(output_dir=output_dir)
    cluster_noise_summary, cluster_noise_df = analyzer.estimate_cluster_specific_noise(
        raw_df=raw_df,
        cluster_df=cluster_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        col_week='孕周_标准化'
    )

    
    group_comparison = analyzer.analyze_group_differences(cluster_noise_df)

    
    gamm_simulation, sim_detail_df = analyzer.gamm_driven_error_simulation(
        cluster_df=cluster_df,
        prediction_df=prediction_df,
        n_simulations=1000
    )

    
    stability_analysis = analyzer.analyze_cluster_stability(
        raw_df=raw_df,
        cluster_df=cluster_df,
        bmi_analyzer=bmi_analyzer,
        n_bootstrap=50
    )
    
    create_q3_specialized_visualizations(analyzer, cluster_noise_df, sim_detail_df)

if __name__ == '__main__':
    main()
