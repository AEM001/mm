#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析与可视化工具函数：
- 模型项摘要表 create_detailed_summary
- 偏依赖图 create_partial_dependence_plots
- 模型诊断 perform_model_diagnostics（残差图、QQ、正态性、异方差检验）
- 重复K折交叉验证 cross_validate_all_models（可选保存每折预测与残差）
- 整体模型近似F检验 compute_overall_f_test
- 捕获GAM summary 文本 get_model_summary_text
"""

import os
import warnings
from io import StringIO
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def get_model_summary_text(model) -> str:
    """捕获 pygam 的 summary 文本输出"""
    _buf = StringIO()
    with contextlib.redirect_stdout(_buf):
        model.summary()
    return _buf.getvalue()


def create_detailed_summary(model, feature_names):
    """创建详细的模型摘要表格"""
    rows = []
    for i, term in enumerate(model.terms):
        term_type = term.__class__.__name__
        if term_type == 'SplineTerm':
            term_idx = term.feature
            term_name = f"s({feature_names[term_idx]})"
            term_type_cn = "平滑项 (Smooth)"
        elif term_type == 'LinearTerm':
            term_idx = term.feature
            term_name = f"l({feature_names[term_idx]})"
            term_type_cn = "线性项 (Linear)"
        elif term_type == 'TensorTerm':
            try:
                terms_info = term.info['terms']
                feat1 = int(terms_info[0]['feature'])
                feat2 = int(terms_info[1]['feature'])
            except (KeyError, IndexError, ValueError):
                feat1, feat2 = 0, 1
            feat_name1 = feature_names[feat1] if feat1 < len(feature_names) else f'feature_{feat1}'
            feat_name2 = feature_names[feat2] if feat2 < len(feature_names) else f'feature_{feat2}'
            term_name = f"te({feat_name1}, {feat_name2})"
            term_type_cn = "交互项 (Tensor)"
        else:
            term_name = "截距 (Intercept)"
            term_type_cn = "常数项 (Constant)"

        p_value = model.statistics_['p_values'][i] if i < len(model.statistics_['p_values']) else np.nan
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        edof = np.nan

        lam_display = ""
        try:
            if hasattr(model, 'lam') and i < len(model.lam):
                lam_val = model.lam[i]
                if isinstance(lam_val, (list, tuple, np.ndarray)):
                    lam_display = np.array2string(np.array(lam_val), precision=1, separator=' ')
                elif isinstance(lam_val, (int, float, np.floating)):
                    lam_display = f"{lam_val:.6f}"
                else:
                    lam_display = str(lam_val)
        except Exception:
            lam_display = ""

        rows.append({
            'Term (模型项)': term_name,
            'Term_Type (项类型)': term_type_cn,
            'EDoF (有效自由度)': f"{edof:.3f}" if not np.isnan(edof) else "",
            'Lambda (平滑惩罚系数)': lam_display,
            'P-value (P值)': f"{p_value:.4f}" if not np.isnan(p_value) else "",
            'Significance (显著性水平)': significance,
            'Interpretation (业务解读)': ""
        })
    return pd.DataFrame(rows)


def create_partial_dependence_plots(model, feature_names, output_dir):
    """为模型中的每个项创建偏依赖图"""
    plot_indices = [i for i, t in enumerate(model.terms) if not getattr(t, 'isintercept', False)]
    num_plots = len(plot_indices)
    rows = int(np.ceil(num_plots / 2))
    cols = 2 if num_plots > 1 else 1
    fig = plt.figure(figsize=(16, rows * 6))

    for plot_pos, i in enumerate(plot_indices, start=1):
        term = model.terms[i]
        term_type = term.__class__.__name__
        ax = fig.add_subplot(rows, cols, plot_pos)

        if term_type == 'SplineTerm':
            feature_idx = term.feature
            feature_name = feature_names[feature_idx]
            XX = model.generate_X_grid(term=i)
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            ax.plot(XX[:, feature_idx], pdep)
            ax.fill_between(XX[:, feature_idx], confi[:, 0], confi[:, 1], alpha=0.3)
            ax.set_title(f'{feature_name}的非线性影响', fontsize=14)
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel('Y染色体浓度的偏效应', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

        elif term_type == 'LinearTerm':
            feature_idx = term.feature
            feature_name = feature_names[feature_idx]
            XX = model.generate_X_grid(term=i)
            pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
            ax.plot(XX[:, feature_idx], pdep)
            ax.fill_between(XX[:, feature_idx], confi[:, 0], confi[:, 1], alpha=0.3)
            ax.set_title(f'{feature_name}的线性影响', fontsize=14)
            ax.set_xlabel(feature_name, fontsize=12)
            ax.set_ylabel('Y染色体浓度的偏效应', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

        elif term_type == 'TensorTerm':
            try:
                terms_info = term.info['terms']
                feat_idx1 = int(terms_info[0]['feature'])
                feat_idx2 = int(terms_info[1]['feature'])
            except (KeyError, IndexError, ValueError):
                feat_idx1, feat_idx2 = 0, 1
            feat_name1 = feature_names[feat_idx1] if feat_idx1 < len(feature_names) else f'feature_{feat_idx1}'
            feat_name2 = feature_names[feat_idx2] if feat_idx2 < len(feature_names) else f'feature_{feat_idx2}'
            XX = model.generate_X_grid(term=i, n=50)
            Z = model.partial_dependence(term=i, X=XX)
            x = XX[:, feat_idx1]
            y = XX[:, feat_idx2]
            n_unique_x = len(np.unique(x))
            n_unique_y = len(np.unique(y))
            X_mesh = x.reshape(n_unique_x, n_unique_y)
            Y_mesh = y.reshape(n_unique_x, n_unique_y)
            Z_mesh = Z.reshape(n_unique_x, n_unique_y)
            colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
            cmap_name = 'blue_white_red'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
            c = ax.pcolormesh(X_mesh, Y_mesh, Z_mesh, cmap=cm, shading='auto')
            contour = ax.contour(X_mesh, Y_mesh, Z_mesh, colors='k', alpha=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
            plt.colorbar(c, ax=ax, label='Y染色体浓度的偏效应')
            ax.set_title(f'{feat_name1}与{feat_name2}的交互效应', fontsize=14)
            ax.set_xlabel(feat_name1, fontsize=12)
            ax.set_ylabel(feat_name2, fontsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/partial_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存偏依赖图到 {output_dir}/partial_dependence_plots.png")


def perform_model_diagnostics(model, X, y, feature_names, output_dir):
    """进行模型诊断：残差分析、正态性与异方差检验"""
    if len(feature_names) != X.shape[1]:
        indices = [list(X.columns).index(feat) for feat in feature_names]
        X_for_use = X.iloc[:, indices]
    else:
        X_for_use = X
    y_pred = model.predict(X_for_use)
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='-')
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(residuals, y_pred, frac=0.3)
        axes[0].plot(smooth[:, 0], smooth[:, 1], color='red', lw=2)
    except Exception:
        pass
    axes[0].set_title('残差 vs. 拟合值图', fontsize=14)
    axes[0].set_xlabel('拟合值', fontsize=12)
    axes[0].set_ylabel('残差', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('残差正态Q-Q图', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/model_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存模型诊断图到 {output_dir}/model_diagnostics.png")

    try:
        shapiro_w, shapiro_p = stats.shapiro(residuals)
    except Exception:
        shapiro_w, shapiro_p = np.nan, np.nan
    try:
        jb_stat, jb_p = stats.jarque_bera(residuals)
    except Exception:
        jb_stat, jb_p = np.nan, np.nan
    bp_lm, bp_p = np.nan, np.nan
    if HAS_STATSMODELS:
        try:
            exog = np.column_stack([np.ones(len(X_for_use)), X_for_use.values])
            bp_lm, bp_lm_pvalue, _, _ = het_breuschpagan(residuals, exog)
            bp_p = bp_lm_pvalue
        except Exception:
            bp_lm, bp_p = np.nan, np.nan
    diag_stats = pd.DataFrame({
        '指标': [
            '均方误差(MSE)', '平均绝对误差(MAE)', '残差均值', '残差标准差', '残差中位数',
            'Shapiro-W (正态性)', 'Shapiro-p', 'Jarque-Bera', 'Jarque-Bera p', 'BP LM (异方差)', 'BP p'
        ],
        '值': [
            np.mean(residuals**2), np.mean(np.abs(residuals)), np.mean(residuals), np.std(residuals), np.median(residuals),
            shapiro_w, shapiro_p, jb_stat, jb_p, bp_lm, bp_p
        ]
    })
    diag_stats.to_csv(f'{output_dir}/diagnostic_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"已保存诊断统计量到 {output_dir}/diagnostic_statistics.csv")


def cross_validate_all_models(X, y, specs, k=5, repeats=3, random_state=42, lam_grid=None,
                              save_fold_predictions=False, output_dir=None, sample_ids=None):
    """重复K折交叉验证，返回各模型平均指标（含调参）；可保存每折预测与残差。"""
    n = len(y)
    rng_global = np.random.RandomState(random_state)
    results = {name: {'MAE': [], 'MedAE': [], 'RMSE': [], 'R2': []} for name in specs}
    pred_rows = [] if save_fold_predictions else None

    for rep in range(repeats):
        idx = np.arange(n)
        rng = np.random.RandomState(rng_global.randint(0, 10**6))
        rng.shuffle(idx)
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[: n % k] += 1
        current = 0

        for fold in range(k):
            start, stop = current, current + fold_sizes[fold]
            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            current = stop

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            for name, spec in specs.items():
                feats = spec['features']
                X_train = X[feats].iloc[train_idx]
                X_test = X[feats].iloc[test_idx]
                gam = spec['builder']()
                preds = None
                try:
                    if lam_grid is not None:
                        try:
                            gam = gam.gridsearch(X_train, y_train, lam=lam_grid, progress=False)
                        except Exception:
                            gam.fit(X_train, y_train)
                    else:
                        gam.fit(X_train, y_train)
                    preds = gam.predict(X_test)
                    mae = float(np.mean(np.abs(y_test - preds)))
                    medae = float(np.median(np.abs(y_test - preds)))
                    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                    ss_res = float(np.sum((y_test - preds) ** 2))
                    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
                    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    preds = np.full(shape=len(y_test), fill_value=np.nan, dtype=float)
                    mae, medae, rmse, r2 = np.nan, np.nan, np.nan, np.nan
                results[name]['MAE'].append(mae)
                results[name]['MedAE'].append(medae)
                results[name]['RMSE'].append(rmse)
                results[name]['R2'].append(r2)
                if save_fold_predictions:
                    try:
                        ids = sample_ids.iloc[test_idx].values if sample_ids is not None else test_idx
                    except Exception:
                        ids = test_idx
                    pred_rows.append(pd.DataFrame({
                        'repeat': rep + 1,
                        'fold': fold + 1,
                        '模型': name,
                        'sample_id': ids,
                        'y_true': y_test.values,
                        'y_pred': preds,
                        'residual': y_test.values - preds
                    }))

    rows = []
    for name, mets in results.items():
        row = {
            '模型': name,
            'MAE_mean': np.nanmean(mets['MAE']),
            'MAE_std': np.nanstd(mets['MAE']),
            'MedAE_mean': np.nanmean(mets['MedAE']),
            'MedAE_std': np.nanstd(mets['MedAE']),
            'RMSE_mean': np.nanmean(mets['RMSE']),
            'RMSE_std': np.nanstd(mets['RMSE']),
            'R2_mean': np.nanmean(mets['R2']),
            'R2_std': np.nanstd(mets['R2'])
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['RMSE_mean', 'MAE_mean', 'R2_mean'], ascending=[True, True, False])
    for col in ['MAE_mean', 'MAE_std', 'MedAE_mean', 'MedAE_std', 'RMSE_mean', 'RMSE_std', 'R2_mean', 'R2_std']:
        df[col] = df[col].map(lambda v: f"{v:.6f}" if pd.notnull(v) else "")

    if save_fold_predictions and output_dir is not None and pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        pred_df.to_csv(f'{output_dir}/cv_predictions.csv', index=False, encoding='utf-8-sig')
    return df


def compute_overall_f_test(model, X_used, y):
    """计算整体模型的近似 F 检验统计量（基于EDoF）"""
    y_pred = model.predict(X_used)
    n = len(y)
    SSE = float(np.sum((y - y_pred) ** 2))
    SSR = float(np.sum((y_pred - y.mean()) ** 2))
    edof = float(model.statistics_['edof'])
    df1 = max(1, int(np.ceil(edof)))
    df2 = max(1, int(n - df1))
    F_stat = (SSR / df1) / (SSE / df2)
    F_p_value = float(stats.f.sf(F_stat, df1, df2))
    return {
        'df1': df1,
        'df2': df2,
        'F_stat': F_stat,
        'F_p_value': F_p_value
    }
