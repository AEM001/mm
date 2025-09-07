import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['STSong', 'Songti SC', 'STHeiti']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'STSong']
    plt.rcParams['axes.unicode_minus'] = False

import platform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from set_chinese_font import set_chinese_font
set_chinese_font()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import sys

class GroupingStrategy:
    def get_groups_and_true_times(self, data_file):
        raise NotImplementedError("子类必须实现此方法")

    def get_strategy_name(self):
        raise NotImplementedError("子类必须实现此方法")

class IntervalCensoredStrategy(GroupingStrategy):
    def get_groups_and_true_times(self, data_file):
        from interval_censored_survival_model import IntervalCensoredSurvivalModel

        model = IntervalCensoredSurvivalModel()
        model.load_and_prepare_data(data_file)
        model.fit_aft_model(distribution='lognormal')
        model.perform_bmi_clustering(n_clusters=4)
        model.determine_optimal_timing(success_rate=0.9)

        bmi_groups = model.bmi_groups.copy()
        optimal_timings = model.optimal_timings.copy()
        true_times = optimal_timings.sort_values('组别')['最佳时点'].values

        return bmi_groups.sort_values('组别'), true_times

    def get_strategy_name(self):
        return "区间删失生存模型"

class BMIBoundaryStrategy(GroupingStrategy):
    def get_groups_and_true_times(self, data_file):
        from bmi_boundary_optimizer import NIPTTimingOptimizer

        optimizer = NIPTTimingOptimizer(data_file)
        recommendations_df = optimizer.run_analysis()

        bmi_groups = []
        true_times = []

        for _, row in recommendations_df.iterrows():
            bmi_groups.append({
                '组别': int(row['BMI组别']),
                'BMI区间': row['BMI范围_标准化'],
                'BMI均值': row['BMI均值_标准化'],
                '样本数': row['样本数']
            })
            true_times.append(row['建议检测时点_原始周数'])

        bmi_groups_df = pd.DataFrame(bmi_groups)
        true_times_array = np.array(true_times)

        return bmi_groups_df, true_times_array

    def get_strategy_name(self):
        return "BMI分界值分组"

class DetectionErrorAnalyzer:
    def __init__(self, output_dir='.', threshold_logit=-3.178054):
        self.output_dir = output_dir
        self.error_dir = os.path.join(output_dir, 'detection_error_analysis')
        self.threshold_logit = threshold_logit
        self.threshold_percentage = 0.04

        os.makedirs(self.error_dir, exist_ok=True)

        self.noise_metrics = {}
        self.error_rates = {}
        self.simulation_results = {}
        self.report_context = {}

    def estimate_measurement_noise(self, df, col_woman='孕妇代码', col_y='Y染色体浓度',
                                 col_week=None, smoothing_window=3):
        noise_data = []

        for woman_code, woman_data in df.groupby(col_woman):
            if len(woman_data) < 3:
                continue

            if col_week and col_week in woman_data.columns:
                woman_data = woman_data.sort_values(col_week).reset_index(drop=True)
            else:
                woman_data = woman_data.reset_index(drop=True)

            y_values = woman_data[col_y].values

            if len(y_values) >= smoothing_window:
                y_smooth = gaussian_filter1d(y_values, sigma=1.0)
                residuals_smooth = y_values - y_smooth
                noise_smooth = np.std(residuals_smooth)
            else:
                noise_smooth = np.nan

            if len(y_values) >= 2:
                diffs = np.diff(y_values)
                noise_diff = np.median(np.abs(diffs - np.median(diffs))) / 0.6745
            else:
                noise_diff = np.nan

            if len(y_values) >= 3:
                x_indices = np.arange(len(y_values))
                coeffs = np.polyfit(x_indices, y_values, 1)
                y_fit = np.polyval(coeffs, x_indices)
                residuals_linear = y_values - y_fit
                noise_linear = np.std(residuals_linear)
            else:
                noise_linear = np.nan

            signal_mean = np.mean(y_values)
            signal_std = np.std(y_values)

            noise_data.append({
                '孕妇代码': woman_code,
                '测量次数': len(y_values),
                '信号均值': signal_mean,
                '信号标准差': signal_std,
                '噪声_平滑': noise_smooth,
                '噪声_差分': noise_diff,
                '噪声_线性': noise_linear,
                'SNR_平滑': signal_std / noise_smooth if not np.isnan(noise_smooth) and noise_smooth > 0 else np.nan,
                'SNR_差分': signal_std / noise_diff if not np.isnan(noise_diff) and noise_diff > 0 else np.nan,
                'SNR_线性': signal_std / noise_linear if not np.isnan(noise_linear) and noise_linear > 0 else np.nan
            })

        noise_df = pd.DataFrame(noise_data)

        noise_summary = {
            '样本数': len(noise_df),
            '平均测量次数': noise_df['测量次数'].mean(),
            '噪声水平': {
                '平滑法': {
                    '均值': noise_df['噪声_平滑'].mean(),
                    '中位数': noise_df['噪声_平滑'].median(),
                    '标准差': noise_df['噪声_平滑'].std()
                },
                '差分法': {
                    '均值': noise_df['噪声_差分'].mean(),
                    '中位数': noise_df['噪声_差分'].median(),
                    '标准差': noise_df['噪声_差分'].std()
                },
                '线性法': {
                    '均值': noise_df['噪声_线性'].mean(),
                    '中位数': noise_df['噪声_线性'].median(),
                    '标准差': noise_df['噪声_线性'].std()
                }
            },
            '信噪比': {
                'SNR_平滑': noise_df['SNR_平滑'].mean(),
                'SNR_差分': noise_df['SNR_差分'].mean(),
                'SNR_线性': noise_df['SNR_线性'].mean()
            }
        }

        noise_df.to_csv(os.path.join(self.error_dir, 'measurement_noise_details.csv'),
                       index=False, encoding='utf-8')

        self.noise_metrics = noise_summary

        return noise_summary, noise_df

    def analyze_detection_errors(self, df, col_woman='孕妇代码', col_y='Y染色体浓度',
                               noise_level=None):
        if noise_level is None:
            if self.noise_metrics:
                noise_level = self.noise_metrics['噪声水平']['平滑法']['均值']
            else:
                noise_level = 0.1

        y_range = np.linspace(-5, 0, 100)

        error_analysis = []

        for true_conc in y_range:
            if true_conc < self.threshold_logit:
                false_positive_prob = 1 - stats.norm.cdf(
                    (self.threshold_logit - true_conc) / noise_level
                )
            else:
                false_positive_prob = 0.0

            if true_conc >= self.threshold_logit:
                false_negative_prob = stats.norm.cdf(
                    (self.threshold_logit - true_conc) / noise_level
                )
            else:
                false_negative_prob = 0.0

            correct_prob = 1 - false_positive_prob - false_negative_prob

            error_analysis.append({
                '真实浓度_logit': true_conc,
                '真实浓度_百分比': 1 / (1 + np.exp(-true_conc)),
                '假阳性概率': false_positive_prob,
                '假阴性概率': false_negative_prob,
                '正确检测概率': correct_prob,
                '距离阈值': abs(true_conc - self.threshold_logit)
            })

        error_df = pd.DataFrame(error_analysis)

        near_threshold = error_df[
            error_df['距离阈值'] <= 2 * noise_level
        ]

        error_summary = {
            '噪声水平': noise_level,
            '阈值附近误差': {
                '范围': f"±{2*noise_level:.4f} (logit单位)",
                '平均假阳性率': near_threshold['假阳性概率'].mean(),
                '平均假阴性率': near_threshold['假阴性概率'].mean(),
                '最大假阳性率': near_threshold['假阳性概率'].max(),
                '最大假阴性率': near_threshold['假阴性概率'].max()
            },
            '4%阈值处误差': {
                '假阳性率': error_df.loc[
                    np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)),
                    '假阳性概率'
                ],
                '假阴性率': error_df.loc[
                    np.argmin(np.abs(error_df['真实浓度_logit'] - self.threshold_logit)),
                    '假阴性概率'
                ]
            }
        }

        error_df.to_csv(os.path.join(self.error_dir, 'detection_error_analysis.csv'),
                       index=False, encoding='utf-8')

        self.error_rates = error_summary

        return error_summary, error_df

    def monte_carlo_simulation(self, true_times, bmi_groups, n_simulations=1000,
                             noise_level=None):
        if noise_level is None:
            if self.noise_metrics:
                noise_level = self.noise_metrics['噪声水平']['平滑法']['均值']
            else:
                noise_level = 0.1

        simulation_results = []

        for i, (_, group) in enumerate(bmi_groups.iterrows()):
            true_time = true_times[i] if i < len(true_times) else 15.0
            group_name = f"第{group['组别']}组"

            detected_times = []

            for sim in range(n_simulations):
                start_week = 10
                max_week = 30
                weeks = np.arange(start_week, max_week + 1)

                if true_time <= start_week:
                    true_concentrations = np.full(len(weeks), self.threshold_logit + 0.5)
                else:
                    slope = (self.threshold_logit - (self.threshold_logit - 1)) / (true_time - start_week)
                    true_concentrations = (self.threshold_logit - 1) + slope * (weeks - start_week)
                    true_concentrations = np.maximum(true_concentrations, self.threshold_logit - 1)

                measured_concentrations = true_concentrations + np.random.normal(0, noise_level, len(weeks))

                qualified_indices = np.where(measured_concentrations >= self.threshold_logit)[0]

                if len(qualified_indices) > 0:
                    detected_time = weeks[qualified_indices[0]]
                else:
                    detected_time = max_week

                detected_times.append(detected_time)

            detected_times = np.array(detected_times)

            bias = np.mean(detected_times) - true_time
            rmse = np.sqrt(np.mean((detected_times - true_time)**2))
            mae = np.mean(np.abs(detected_times - true_time))

            ci_lower = np.percentile(detected_times, 2.5)
            ci_upper = np.percentile(detected_times, 97.5)

            simulation_results.append({
                '组别': group['组别'],
                'BMI区间': group['BMI区间'],
                '真实达标时间': true_time,
                '预测均值': np.mean(detected_times),
                '预测中位数': np.median(detected_times),
                '偏差': bias,
                'RMSE': rmse,
                'MAE': mae,
                '标准差': np.std(detected_times),
                'CI_2.5%': ci_lower,
                'CI_97.5%': ci_upper,
                '检测成功率': np.mean(detected_times < max_week)
            })

        sim_df = pd.DataFrame(simulation_results)

        sim_df.to_csv(os.path.join(self.error_dir, 'monte_carlo_simulation.csv'),
                     index=False, encoding='utf-8')

        summary = {
            '总体偏差': {
                '平均偏差': sim_df['偏差'].mean(),
                '平均RMSE': sim_df['RMSE'].mean(),
                '平均MAE': sim_df['MAE'].mean()
            },
            '各组结果': sim_df.to_dict('records')
        }

        self.simulation_results = summary

        return summary, sim_df

def create_error_visualizations(analyzer, noise_df, error_df, sim_df):
    set_chinese_font()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax1 = axes[0, 0]
    noise_methods = ['噪声_平滑', '噪声_差分', '噪声_线性']
    noise_values = [noise_df[col].dropna() for col in noise_methods]

    ax1.boxplot(noise_values, labels=['平滑法', '差分法', '线性法'])
    ax1.set_ylabel('噪声水平')
    ax1.set_title('不同方法估计的噪声水平分布')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    snr_methods = ['SNR_平滑', 'SNR_差分', 'SNR_线性']
    snr_values = [noise_df[col].dropna() for col in snr_methods]

    ax2.boxplot(snr_values, labels=['平滑法', '差分法', '线性法'])
    ax2.set_ylabel('信噪比')
    ax2.set_title('不同方法计算的信噪比分布')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    conc_percent = error_df['真实浓度_百分比'] * 100

    ax3.plot(conc_percent, error_df['假阳性概率'], 'r-', label='假阳性率', linewidth=2)
    ax3.plot(conc_percent, error_df['假阴性概率'], 'b-', label='假阴性率', linewidth=2)
    ax3.axvline(x=4, color='black', linestyle='--', alpha=0.7, label='4%阈值')
    ax3.set_xlabel('真实Y染色体浓度 (%)')
    ax3.set_ylabel('误判概率')
    ax3.set_title('检测误差概率曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)

    ax4 = axes[1, 0]
    groups = [f"第{row['组别']}组" for _, row in sim_df.iterrows()]
    biases = sim_df['偏差'].values

    colors = ['red' if b > 0 else 'blue' for b in biases]
    bars = ax4.bar(groups, biases, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('时间偏差 (周)')
    ax4.set_title('各BMI组达标时间估计偏差')
    ax4.grid(True, alpha=0.3)

    for bar, bias in zip(bars, biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1 if height > 0 else height - 0.1,
                f'{bias:.1f}', ha='center', va='bottom' if height > 0 else 'top')

    ax5 = axes[1, 1]
    x_pos = np.arange(len(groups))
    width = 0.35

    ax5.bar(x_pos - width/2, sim_df['RMSE'], width, label='RMSE', alpha=0.7)
    ax5.bar(x_pos + width/2, sim_df['MAE'], width, label='MAE', alpha=0.7)
    ax5.set_xlabel('BMI组')
    ax5.set_ylabel('误差 (周)')
    ax5.set_title('各组预测误差对比')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(groups)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    true_times = sim_df['真实达标时间']
    pred_means = sim_df['预测均值']
    ci_lower = sim_df['CI_2.5%']
    ci_upper = sim_df['CI_97.5%']

    for i, group in enumerate(groups):
        lower_err = max(0, pred_means.iloc[i] - ci_lower.iloc[i])
        upper_err = max(0, ci_upper.iloc[i] - pred_means.iloc[i])

        ax6.errorbar(i, pred_means.iloc[i],
                    yerr=[[lower_err], [upper_err]],
                    fmt='o', capsize=5, label='预测' if i == 0 else "")
        ax6.scatter(i, true_times.iloc[i], color='red', marker='x', s=100,
                   label='真实' if i == 0 else "")

    ax6.set_xlabel('BMI组')
    ax6.set_ylabel('达标时间 (周)')
    ax6.set_title('达标时间预测置信区间')
    ax6.set_xticks(range(len(groups)))
    ax6.set_xticklabels(groups)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(analyzer.error_dir, 'detection_error_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_error_report(analyzer):
    if analyzer.noise_metrics:
        noise = analyzer.noise_metrics
        print(f"噪声水平: 平滑法均值={noise['噪声水平']['平滑法']['均值']:.6f}, SNR={noise['信噪比']['SNR_平滑']:.2f}")
    if analyzer.error_rates:
        error = analyzer.error_rates
        print(f"4%阈值误差: 假阳性={error['4%阈值处误差']['假阳性率']:.4f}, 假阴性={error['4%阈值处误差']['假阴性率']:.4f}")
    if analyzer.simulation_results:
        sim = analyzer.simulation_results
        print(f"模拟偏差: 平均偏差={sim['总体偏差']['平均偏差']:.2f}周, RMSE={sim['总体偏差']['平均RMSE']:.2f}周")
    return None

def run_error_analysis_with_strategy(strategy, data_file=None, output_suffix=''):
    if output_suffix:
        output_dir = os.path.join(SCRIPT_DIR, f'detection_error_results_{output_suffix}')
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'detection_error_results')
    os.makedirs(output_dir, exist_ok=True)

    if data_file is None:
        data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')

    raw_df = pd.read_csv(data_file)

    analyzer = DetectionErrorAnalyzer(output_dir=output_dir)
    analyzer.report_context['analysis_basis'] = strategy.get_strategy_name()
    noise_summary, noise_df = analyzer.estimate_measurement_noise(
        raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        col_week='孕周_标准化',
        smoothing_window=3,
    )

    error_summary, error_df = analyzer.analyze_detection_errors(
        raw_df,
        col_woman='孕妇代码',
        col_y='Y染色体浓度',
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )

    bmi_groups, true_times = strategy.get_groups_and_true_times(data_file)

    sim_summary, sim_df = analyzer.monte_carlo_simulation(
        true_times=true_times,
        bmi_groups=bmi_groups,
        n_simulations=1000,
        noise_level=noise_summary['噪声水平']['平滑法']['均值'] if '噪声水平' in noise_summary else None,
    )

    create_error_visualizations(analyzer, noise_df, error_df, sim_df)
    generate_error_report(analyzer)

    return {
        'strategy': strategy.get_strategy_name(),
        'analyzer': analyzer,
        'noise_summary': noise_summary,
        'error_summary': error_summary,
        'sim_summary': sim_summary
    }


def main(strategy_type='interval_censored'):
    """
    主函数：运行检测误差分析
    
    Args:
        strategy_type: 分组策略类型
            - 'interval_censored': 区间删失生存模型（原有方法）
            - 'bmi_boundary': BMI分界值分组（新方法）
    """
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    if strategy_type == 'interval_censored':
        strategy = IntervalCensoredStrategy()
        suffix = 'interval_censored'
    elif strategy_type == 'bmi_boundary':
        strategy = BMIBoundaryStrategy()
        suffix = 'bmi_boundary'
    else:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    return run_error_analysis_with_strategy(strategy, data_file, suffix)


def compare_strategies():
    """
    比较不同分组策略的检测误差分析结果
    """
    print("=== 比较不同分组策略 ===")
    
    strategies = [
        ('interval_censored', IntervalCensoredStrategy()),
        ('bmi_boundary', BMIBoundaryStrategy())
    ]
    
    results = {}
    data_file = os.path.join(SCRIPT_DIR, 'processed_data.csv')
    
    for strategy_name, strategy in strategies:
        try:
            print(f"\n--- 运行{strategy.get_strategy_name()}策略 ---")
            result = run_error_analysis_with_strategy(strategy, data_file, strategy_name)
            if result:
                results[strategy_name] = result
        except Exception as e:
            print(f"策略{strategy.get_strategy_name()}执行失败: {e}")
            continue
    
    # 生成比较报告
    if len(results) > 1:
        print("\n=== 策略比较总结 ===")
        for strategy_name, result in results.items():
            noise_level = result['noise_summary']['噪声水平']['平滑法']['均值'] if 'noise_summary' in result else 'N/A'
            avg_bias = result['sim_summary']['总体偏差']['平均偏差'] if 'sim_summary' in result else 'N/A'
            print(f"{result['strategy']}:")
            print(f"  平均噪声水平: {noise_level:.6f}" if isinstance(noise_level, float) else f"  平均噪声水平: {noise_level}")
            print(f"  平均时间偏差: {avg_bias:.2f}周" if isinstance(avg_bias, float) else f"  平均时间偏差: {avg_bias}")
            print()
    
    return results


if __name__ == '__main__':
    main()
