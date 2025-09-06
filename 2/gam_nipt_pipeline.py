#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题二：基于GAM的达标时间预测 → 聚类 → 推荐NIPT时点（Y已为logit变换）

说明
- 使用 2/processed_data.csv 作为训练与预测数据，Y=Y染色体浓度 已为 logit 变换。
- 特征：孕周_标准化, BMI_标准化, 年龄_标准化。
- 训练 LinearGAM（与问题一最优规格一致：s(孕周)+s(BMI)+s(年龄)+te(孕周,BMI)）。
- 对每位孕妇，遍历孕周区间，找到首次使预测logit(Y)≥logit(4%)的最早孕周（原始周）。
- 对预测的最早达标孕周做KMeans聚类，基于90%分位给出各簇推荐NIPT时点。
- 生成结果CSV、可视化图与报告，输出目录：2/gam_results/

注意
- 孕周标准化使用的均值/标准差需与生成“孕周_标准化”的处理保持一致。此处使用默认值：
  TIME_MEAN=16.846, TIME_STD=4.076（来自现有interval模型设置）。如与实际不符，请在此处修改。
- 如需中文字体，请确保  set_chinese_font.py  可被导入。
"""

import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pygam import LinearGAM, s, te
from scipy.special import expit

# 允许导入项目根目录以使用中文字体设置
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from set_chinese_font import set_chinese_font
    set_chinese_font()
except Exception:
    pass


# ---------------------------- 配置区 ---------------------------- #
TIME_MEAN = 16.846
TIME_STD = 4.076
LOGIT_THRESHOLD = float(np.log(0.04 / (1 - 0.04)))  # logit(4%)
WEEK_MIN = 10.0
WEEK_MAX = 25.0
WEEK_STEP = 0.1
DEFAULT_K_RANGE = range(2, 7)  # 2..6
SUCCESS_QUANTILE = 0.90

FEATURES = ['孕周_标准化', 'BMI_标准化', '年龄_标准化']
TARGET = 'Y染色体浓度'  # 已经是logit(Y) 形式


# ---------------------------- 工具函数 ---------------------------- #
def standardize_week(original_week: np.ndarray) -> np.ndarray:
    return (original_week - TIME_MEAN) / TIME_STD


def destandardize_week(standardized_week: np.ndarray) -> np.ndarray:
    return standardized_week * TIME_STD + TIME_MEAN


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------- 主类实现 ---------------------------- #
@dataclass
class GAMNIPTPipelineConfig:
    data_path: str
    output_dir: str
    
    time_mean: float = TIME_MEAN
    time_std: float = TIME_STD
    logit_threshold: float = LOGIT_THRESHOLD
    week_min: float = WEEK_MIN
    week_max: float = WEEK_MAX
    week_step: float = WEEK_STEP
    k_range: range = DEFAULT_K_RANGE
    success_quantile: float = SUCCESS_QUANTILE


class GAMNIPTPipeline:
    def __init__(self, config: GAMNIPTPipelineConfig):
        self.cfg = config
        ensure_dir(self.cfg.output_dir)
        self.df = None
        self.model = None
        self.pred_times_df = None
        self.cluster_summary_df = None
        self.recommendations_df = None

    # ---------- 数据加载 ---------- #
    def load_data(self):
        print(f"读取训练/预测数据: {self.cfg.data_path}")
        df = pd.read_csv(self.cfg.data_path)
        print(f"数据形状: {df.shape}")
        print(f"可用列: {list(df.columns)}")

        missing = [col for col in FEATURES + [TARGET, '孕妇代码'] if col not in df.columns]
        if missing:
            raise ValueError(f"数据缺失必要列: {missing}")

        # 基本清洗：关键列缺失的记录剔除
        before = len(df)
        df = df.dropna(subset=FEATURES + [TARGET, '孕妇代码']).copy()
        print(f"移除关键列缺失记录: {before - len(df)} 条，剩余: {len(df)} 条")

        # 只保留合理范围的标准化孕周（可选）
        # df = df[(df['孕周_标准化'] >= -3) & (df['孕周_标准化'] <= 3)].copy()

        self.df = df
        
        # 动态设置扫描下限：以训练数据中最小标准化孕周为基准，转换回原始孕周
        try:
            min_week_std = float(df['孕周_标准化'].min())
            min_week_orig = destandardize_week(min_week_std)
            old_min = self.cfg.week_min
            # 取更合理的下界（不低于10周）
            self.cfg.week_min = max(10.0, min_week_orig)
            if abs(self.cfg.week_min - old_min) > 1e-6:
                print(f"自动调整扫描下限: {old_min:.2f} → {self.cfg.week_min:.2f} 周 (依据最小观测孕周)")
        except Exception:
            pass
        return df

    # ---------- 模型训练 ---------- #
    def train_gam(self):
        if self.df is None:
            raise RuntimeError("请先调用 load_data()")

        X = self.df[FEATURES].values
        y = self.df[TARGET].values  # 已为logit(Y)

        print("\n训练 LinearGAM (s(W)+s(B)+s(A)+te(W,B)) ...")
        gam = LinearGAM(
            s(0, n_splines=20) + s(1, n_splines=20) + s(2, n_splines=20) + te(0, 1, n_splines=10)
        )
        gam.gridsearch(X, y)
        self.model = gam

        # 简要报告
        print("GAM 训练完成")
        try:
            print(gam.summary())
        except Exception:
            pass
        return gam

    # ---------- 个体最早达标孕周预测 ---------- #
    def _predict_logit_y(self, week_std: np.ndarray, bmi_std: float, age_std: float) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型未训练")
        # 组合输入：长度=len(week_std) 的 [week_std, bmi_std, age_std]
        X = np.column_stack([
            week_std,
            np.full_like(week_std, bmi_std, dtype=float),
            np.full_like(week_std, age_std, dtype=float)
        ])
        return self.model.predict(X)

    def find_earliest_attainment_week(self, bmi_std: float, age_std: float) -> float:
        # 先用粗网格搜索
        weeks = np.arange(self.cfg.week_min, self.cfg.week_max + 1e-6, self.cfg.week_step)
        weeks_std = standardize_week(weeks)
        preds = self._predict_logit_y(weeks_std, bmi_std, age_std)

        # 找到首次 ≥ 阈值 的位置
        meets = np.where(preds >= self.cfg.logit_threshold)[0]
        if len(meets) == 0:
            return float('inf')  # 未达标（右删失），后续报告中标注
        return float(weeks[meets[0]])

    def predict_earliest_times_per_subject(self):
        if self.df is None or self.model is None:
            raise RuntimeError("请先加载数据并训练模型")

        print("\n按孕妇代码预测最早达标孕周（基于 GAM）...")
        results = []
        for woman_code, g in self.df.groupby('孕妇代码'):
            # BMI/年龄（标准化）——对该孕妇取首行（常量）
            bmi_std = float(g['BMI_标准化'].iloc[0]) if 'BMI_标准化' in g.columns else 0.0
            age_std = float(g['年龄_标准化'].iloc[0]) if '年龄_标准化' in g.columns else 0.0
            week_star = self.find_earliest_attainment_week(bmi_std, age_std)
            results.append({
                '孕妇代码': woman_code,
                'BMI_标准化': bmi_std,
                '年龄_标准化': age_std,
                '预测最早达标孕周': week_star,
                '达标阈值_logit': self.cfg.logit_threshold,
                '是否达标_区间内': np.isfinite(week_star)
            })

        pred_df = pd.DataFrame(results).sort_values('预测最早达标孕周').reset_index(drop=True)
        self.pred_times_df = pred_df

        # 保存
        out_csv = os.path.join(self.cfg.output_dir, 'gam_predicted_attainment_times.csv')
        pred_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"预测结果已保存: {out_csv}")
        return pred_df

    # ---------- 与观测最早达标时间对比评估 ---------- #
    def evaluate_vs_observed(self):
        if self.df is None or self.pred_times_df is None:
            print("评估跳过：缺少训练数据或预测结果")
            return None
        print("\n评估：与观测最早达标孕周对比（按孕妇代码）...")
        # 从训练数据中计算：每名孕妇观测到的最早达标孕周（将logit(Y) -> 比例，阈值0.04）
        obs_rows = []
        for woman_code, g in self.df.groupby('孕妇代码'):
            # 找到满足 expit(logitY) >= 0.04 的最早记录
            prop = expit(pd.to_numeric(g[TARGET], errors='coerce').values)
            meets = np.where(prop >= 0.04)[0]
            if len(meets) == 0:
                obs_week = np.inf
            else:
                # 将标准化孕周还原
                w_std = pd.to_numeric(g['孕周_标准化'].iloc[meets[0]], errors='coerce')
                obs_week = float(destandardize_week(w_std))
            obs_rows.append({'孕妇代码': woman_code, '观测最早达标孕周': obs_week, '是否观测达标': np.isfinite(obs_week)})
        obs_df = pd.DataFrame(obs_rows)

        merged = pd.merge(self.pred_times_df, obs_df, on='孕妇代码', how='left')
        # 仅对双方均为有限值的样本计算误差
        mask = np.isfinite(merged['预测最早达标孕周']) & np.isfinite(merged['观测最早达标孕周'])
        eval_df = merged[mask].copy()
        if len(eval_df) > 0:
            diff = eval_df['预测最早达标孕周'] - eval_df['观测最早达标孕周']
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            bias = float(np.mean(diff))
            print(f"评估样本数: {len(eval_df)} | MAE={mae:.2f}周, RMSE={rmse:.2f}周, 偏差={bias:.2f}周")
        else:
            print("评估样本数为0（可能因为观测未达标或预测未达标）")

        # 保存评估数据
        out_csv = os.path.join(self.cfg.output_dir, 'gam_eval_pred_vs_observed.csv')
        merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"预测与观测对比已保存: {out_csv}")

        # 可视化散点图
        if len(eval_df) > 0:
            plt.figure(figsize=(7, 6))
            plt.scatter(eval_df['观测最早达标孕周'], eval_df['预测最早达标孕周'], alpha=0.6)
            mn = min(eval_df['观测最早达标孕周'].min(), eval_df['预测最早达标孕周'].min())
            mx = max(eval_df['观测最早达标孕周'].max(), eval_df['预测最早达标孕周'].max())
            plt.plot([mn, mx], [mn, mx], 'r--', label='y=x')
            plt.xlabel('观测最早达标孕周')
            plt.ylabel('预测最早达标孕周')
            plt.title('预测 vs 观测 最早达标孕周')
            plt.legend()
            out_png = os.path.join(self.cfg.output_dir, 'gam_pred_vs_obs_scatter.png')
            plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()
            print(f"保存图: {out_png}")
        
        return merged

    # ---------- 聚类与推荐 ---------- #
    def auto_select_k(self, times: np.ndarray) -> int:
        # 仅使用有限值
        valid = np.isfinite(times)
        X = times[valid].reshape(-1, 1)
        inertias, silhouettes = [], []
        best_k, best_score = None, -np.inf
        for k in self.cfg.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia__ if hasattr(km, 'inertia__') else km.inertia_)
            # silhouette 需要 k>=2 且样本数>k
            if len(X) > k:
                try:
                    sscore = silhouette_score(X, labels)
                except Exception:
                    sscore = np.nan
            else:
                sscore = np.nan
            silhouettes.append(sscore)
            score = sscore if not np.isnan(sscore) else -km.inertia_
            if score > best_score:
                best_score, best_k = score, k
        print(f"自动选择的聚类数: {best_k} (基于 silhouette/inertia)")
        return best_k if best_k is not None else 4

    def cluster_predicted_times(self, n_clusters: int | None = None):
        if self.pred_times_df is None:
            raise RuntimeError("请先预测最早达标孕周")

        times = self.pred_times_df['预测最早达标孕周'].values
        finite_mask = np.isfinite(times)
        X = times[finite_mask].reshape(-1, 1)

        if n_clusters is None:
            n_clusters = self.auto_select_k(times)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # 将标签放回完整DataFrame（未达标的记为 -1）
        cluster_labels = np.full_like(times, fill_value=-1, dtype=int)
        cluster_labels[finite_mask] = labels
        self.pred_times_df['KMeans簇'] = cluster_labels

        # 统计每簇信息
        clusters = []
        for k in range(n_clusters):
            group = self.pred_times_df[self.pred_times_df['KMeans簇'] == k]
            weeks = group['预测最早达标孕周'].values
            if len(weeks) == 0:
                continue
            clusters.append({
                '簇': k + 1,
                '样本数': len(weeks),
                '最小孕周': float(np.min(weeks)),
                '最大孕周': float(np.max(weeks)),
                '均值孕周': float(np.mean(weeks)),
                '中位数孕周': float(np.median(weeks)),
            })

        summary_df = pd.DataFrame(clusters)
        self.cluster_summary_df = summary_df

        # 保存
        out_csv = os.path.join(self.cfg.output_dir, 'gam_kmeans_summary.csv')
        summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"聚类摘要已保存: {out_csv}")
        return summary_df

    def recommend_timing_per_cluster(self, success_quantile: float | None = None):
        if self.pred_times_df is None or self.cluster_summary_df is None:
            raise RuntimeError("请先完成预测与聚类")
        if success_quantile is None:
            success_quantile = self.cfg.success_quantile

        print(f"\n按簇给出推荐NIPT时点（{int(success_quantile*100)}%分位）...")
        recs = []
        for k in sorted(self.pred_times_df['KMeans簇'].unique()):
            if k < 0:
                continue  # 跳过未达标组
            group = self.pred_times_df[self.pred_times_df['KMeans簇'] == k]
            q_time = float(np.quantile(group['预测最早达标孕周'].values, success_quantile))
            risk = '低风险' if q_time <= 12 else '中风险' if q_time <= 27 else '高风险'
            recs.append({
                '簇': k + 1,
                '样本数': len(group),
                '推荐NIPT时点_周': q_time,
                '风险等级': risk
            })

        recs_df = pd.DataFrame(recs)
        self.recommendations_df = recs_df

        # 保存
        out_csv = os.path.join(self.cfg.output_dir, 'gam_recommendations.csv')
        recs_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"推荐结果已保存: {out_csv}")
        return recs_df

    # ---------- 可视化与报告 ---------- #
    def visualize(self):
        if self.pred_times_df is None or self.recommendations_df is None:
            print("可视化跳过：缺少预测或推荐结果")
            return

        ensure_dir(self.cfg.output_dir)
        colors = plt.cm.Set1(np.linspace(0, 1, 8))

        # 1) 预测最早达标孕周的直方图（按簇着色）
        plt.figure(figsize=(10, 6))
        for k in sorted(self.pred_times_df['KMeans簇'].unique()):
            group = self.pred_times_df[self.pred_times_df['KMeans簇'] == k]
            if k < 0:
                # 未达标
                continue
            plt.hist(group['预测最早达标孕周'], bins=20, alpha=0.6, color=colors[k % len(colors)], label=f'第{k+1}簇')
        plt.axvline(12, color='red', linestyle='--', label='12周风险阈值')
        plt.xlabel('预测最早达标孕周')
        plt.ylabel('频数')
        plt.title('预测最早达标孕周分布（按KMeans簇）')
        plt.legend()
        out_png = os.path.join(self.cfg.output_dir, 'gam_predicted_attainment_hist.png')
        plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()
        print(f"保存图: {out_png}")

        # 2) 各簇推荐NIPT时点柱状图
        plt.figure(figsize=(10, 6))
        xlabels = [f"第{row['簇']}簇\n(n={int(row['样本数'])})" for _, row in self.recommendations_df.iterrows()]
        yvals = self.recommendations_df['推荐NIPT时点_周'].values
        bars = plt.bar(xlabels, yvals, color=colors[:len(xlabels)], alpha=0.8)
        for bar, y in zip(bars, yvals):
            plt.text(bar.get_x()+bar.get_width()/2, y+0.05, f"{y:.1f}", ha='center', va='bottom')
        plt.axhline(12, color='red', linestyle='--', label='12周风险阈值')
        plt.ylabel('推荐NIPT时点 (周)')
        plt.title(f'各簇推荐NIPT时点（{int(self.cfg.success_quantile*100)}%达标）')
        plt.legend()
        out_png = os.path.join(self.cfg.output_dir, 'gam_cluster_recommendations.png')
        plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()
        print(f"保存图: {out_png}")

    def generate_report(self):
        if self.pred_times_df is None or self.recommendations_df is None:
            print("报告跳过：缺少预测或推荐结果")
            return

        report_lines = []
        report_lines.append('# 基于GAM的NIPT时点建议（问题二）\n')
        report_lines.append('## 概述\n')
        report_lines.append('- 目标：预测Y染色体浓度达到4%的最早孕周，并按预测时间聚类，给出各群体的NIPT推荐时点。\n')
        report_lines.append('- 方法：LinearGAM，特征=孕周_标准化、BMI_标准化、年龄_标准化；阈值=logit(4%)。\n')
        report_lines.append(f'- 时间标准化还原：mean={self.cfg.time_mean:.3f}，std={self.cfg.time_std:.3f}。\n')
        report_lines.append('\n')

        # 聚类摘要
        report_lines.append('## 聚类摘要\n')
        if self.cluster_summary_df is not None and not self.cluster_summary_df.empty:
            for _, row in self.cluster_summary_df.iterrows():
                report_lines.append(f"- 第{int(row['簇'])}簇: n={int(row['样本数'])}, "
                                    f"范围[{row['最小孕周']:.1f}, {row['最大孕周']:.1f}], "
                                    f"均值={row['均值孕周']:.1f}, 中位数={row['中位数孕周']:.1f}\n")
        else:
            report_lines.append('- 无有效聚类摘要\n')
        report_lines.append('\n')

        # 推荐结果
        report_lines.append('## 推荐NIPT时点 (以簇内90%覆盖为目标)\n')
        for _, row in self.recommendations_df.iterrows():
            report_lines.append(f"- 第{int(row['簇'])}簇: 推荐时点={row['推荐NIPT时点_周']:.1f}周, 风险={row['风险等级']}\n")
        report_lines.append('\n')

        out_md = os.path.join(self.cfg.output_dir, 'gam_recommendations.md')
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"报告已保存: {out_md}")

    # ---------- 主流程 ---------- #
    def run(self, n_clusters: int | None = None):
        self.load_data()
        self.train_gam()
        self.predict_earliest_times_per_subject()
        self.evaluate_vs_observed()
        self.cluster_predicted_times(n_clusters=n_clusters)
        self.recommend_timing_per_cluster()
        self.visualize()
        self.generate_report()
        print("\n=== GAM流程完成 ===")
        print(f"结果输出目录: {self.cfg.output_dir}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'processed_data.csv')
    out_dir = os.path.join(script_dir, 'gam_results')

    cfg = GAMNIPTPipelineConfig(
        data_path=data_file,
        output_dir=out_dir,
    )
    pipeline = GAMNIPTPipeline(cfg)
    pipeline.run()
