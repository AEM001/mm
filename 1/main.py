#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：管理数据加载、建模、调参、可视化与评估输出。
将检验、可视化、结果分析集中在 analysis_utils.py 中，模型规格在 gam_enhanced_modeling.py 中。
"""

import os
import sys
import argparse
import warnings
from io import StringIO
import contextlib

import numpy as np
import pandas as pd
from pygam import LinearGAM

# 项目路径与中文字体
sys.path.append('/Users/Mac/Downloads/mm')
from set_chinese_font import set_chinese_font
set_chinese_font()

# 本项目模块
from model_specs import build_model_specs
from analysis_utils import (
    get_model_summary_text,
    create_detailed_summary,
    create_partial_dependence_plots,
    perform_model_diagnostics,
    cross_validate_all_models,
    compute_overall_f_test,
)


def run(processed_data_path='processed_data.csv', output_dir='gam_enhanced_results',
        significance_output_dir='gam_overall_significance_results',
        n_splines_s=20, n_splines_te=10, k=5, repeats=3,
        save_fold_predictions=True, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print('加载已处理的数据...')
    try:
        data = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"错误: 未找到 '{processed_data_path}'。")
        return 1

    print(f"加载的数据样本数: {len(data)}")

    # 2. 组装建模数据
    y = data['Y染色体浓度'].copy()
    X = data[['孕周_标准化', 'BMI_标准化', '年龄_标准化', '怀孕次数_标准化', '生产次数_标准化']].copy()

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[mask]
    X = X[mask]
    feature_names = X.columns.tolist()
    if '孕妇代码' in data.columns:
        sample_ids = data['孕妇代码'].astype(str)
    else:
        sample_ids = pd.Series(np.arange(len(data)), name='row_index').astype(str)
    sample_ids = sample_ids[mask]

    print(f"清理后的建模数据样本数: {len(y)}")

    # 3. 构建候选模型 + λ 调参
    print('\n开始构建和比较多个GAM模型...')
    lam_grid = np.logspace(-3, 3, 7)
    specs = build_model_specs(feature_names, n_splines_s=n_splines_s, n_splines_te=n_splines_te)
    print(f"候选模型数量: {len(specs)}")

    models = {}
    model_features = {}
    for i, (name, spec) in enumerate(specs.items(), start=1):
        feats = spec['features']
        X_used = X[feats]
        print(f"  - 正在训练模型{i}/{len(specs)}: {name} ...")
        gam = spec['builder']()
        try:
            gam = gam.gridsearch(X_used, y, lam=lam_grid, progress=False)
        except Exception as e:
            warnings.warn(f"模型{name} 使用gridsearch失败，改用直接拟合: {e}")
            try:
                gam.fit(X_used, y)
            except Exception as ee:
                warnings.warn(f"模型{name} 拟合失败：{ee}")
                continue
        models[name] = gam
        model_features[name] = feats

    # 4. 模型比较（AICc/伪R²/显著项）
    print('\n模型性能比较:')
    rows = []
    for name, model in models.items():
        p_values = model.statistics_['p_values']
        pvals_non_intercept = [p for t, p in zip(model.terms, p_values) if not t.isintercept]
        significant_terms = sum(p < 0.05 for p in pvals_non_intercept)
        rows.append({
            '模型': name,
            '伪R²': f"{model.statistics_['pseudo_r2']['explained_deviance']:.4f}",
            'AICc': f"{model.statistics_['AICc']:.2f}",
            '显著项数': f"{significant_terms}/{len(pvals_non_intercept)}"
        })
    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    # 4.5. 整体显著性检验（近似F检验）- 为所有候选模型单独输出到文件夹
    print('\n为所有候选模型计算整体显著性近似F检验，并单独导出结果...')
    os.makedirs(significance_output_dir, exist_ok=True)
    sig_rows = []
    for name, model in models.items():
        feats = model_features.get(name, feature_names)
        try:
            X_used_sig = X[feats]
        except Exception:
            # 回退：特征名顺序变化时尝试用索引匹配
            idxs = [list(X.columns).index(f) for f in feats if f in X.columns]
            X_used_sig = X.iloc[:, idxs]
        try:
            sig = compute_overall_f_test(model, X_used_sig, y)
            sig_rows.append({
                '模型': name,
                'df1': sig['df1'],
                'df2': sig['df2'],
                'F_stat': sig['F_stat'],
                'F_p_value': sig['F_p_value']
            })
        except Exception as e:
            sig_rows.append({
                '模型': name,
                'df1': '',
                'df2': '',
                'F_stat': '',
                'F_p_value': '',
                '错误': str(e)
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_csv_path = f"{significance_output_dir}/overall_significance_all_models.csv"
    sig_df.to_csv(sig_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存整体显著性检验汇总至 {sig_csv_path}")

    # 5. 选择AICc最优
    best_model_name = min(models, key=lambda k: models[k].statistics_['AICc'])
    best_model = models[best_model_name]
    print(f"\n最优模型 (基于AICc): {best_model_name}")

    # 6. 总结、可视化与诊断
    print(f"\n最优模型 ({best_model_name}) 摘要:")
    summary_str = get_model_summary_text(best_model)
    print(summary_str)

    best_model_features = model_features.get(best_model_name, feature_names)

    model_details = create_detailed_summary(best_model, best_model_features)
    print("\n详细模型摘要表格:")
    print(model_details.to_string())
    model_details.to_csv(f'{output_dir}/detailed_model_summary.csv', index=False, encoding='utf-8-sig')

    print('\n生成偏依赖图...')
    create_partial_dependence_plots(best_model, best_model_features, output_dir)

    print('\n进行模型诊断...')
    perform_model_diagnostics(best_model, X, y, best_model_features, output_dir)

    # 7. 整体模型的近似 F 检验
    if len(best_model_features) != X.shape[1]:
        _idx = [list(X.columns).index(feat) for feat in best_model_features]
        X_used_for_f = X.iloc[:, _idx]
    else:
        X_used_for_f = X
    ftest = compute_overall_f_test(best_model, X_used_for_f, y)
    print(f"\n整体模型显著性近似F检验: F({ftest['df1']}, {ftest['df2']}) = {ftest['F_stat']:.3f}, p = {ftest['F_p_value']:.4g}")

    # 将最优模型的整体显著性结果单独写入显著性输出目录
    with open(f"{significance_output_dir}/best_model_overall_significance.txt", 'w', encoding='utf-8') as f:
        f.write(f"最优模型: {best_model_name}\n")
        f.write(f"F({ftest['df1']}, {ftest['df2']}) = {ftest['F_stat']:.3f}, p = {ftest['F_p_value']:.4g}\n")
    print(f"已保存最优模型的整体显著性结果至 {significance_output_dir}/best_model_overall_significance.txt")

    with open(f'{output_dir}/best_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"最优GAM模型分析结果: {best_model_name}\n")
        f.write("="*70 + "\n")
        f.write("模型比较:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"最优模型 ({best_model_name}) 摘要:\n")
        f.write(summary_str + "\n")
        f.write("整体模型显著性近似F检验:\n")
        f.write(f"F({ftest['df1']}, {ftest['df2']}) = {ftest['F_stat']:.3f}, p = {ftest['F_p_value']:.4g}\n\n")
        f.write("注: GAM 平滑项的 p 值与上述 F 检验均为近似量，且 pyGAM 在估计平滑参数时 p 值可能偏小，请谨慎解读。\n")

    print(f"\n分析完成。所有结果已保存至 '{output_dir}' 目录。")

    # 8. 重复K折交叉验证
    print(f"\n开始进行重复{k}折交叉验证评估所有候选模型 (MAE/RMSE/MedAE/R²)...")
    cv_df = cross_validate_all_models(
        X, y, specs,
        k=k, repeats=repeats, random_state=42, lam_grid=lam_grid,
        save_fold_predictions=save_fold_predictions, output_dir=output_dir, sample_ids=sample_ids
    )
    print("\n交叉验证汇总结果:")
    print(cv_df.to_string(index=False))
    cv_df.to_csv(f'{output_dir}/cv_results.csv', index=False, encoding='utf-8-sig')
    print(f"已保存交叉验证结果至 {output_dir}/cv_results.csv")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAM 主运行入口')
    parser.add_argument('--processed_data', type=str, default='processed_data.csv')
    parser.add_argument('--output_dir', type=str, default='gam_enhanced_results')
    parser.add_argument('--sig_output_dir', type=str, default='gam_overall_significance_results')
    parser.add_argument('--n_splines_s', type=int, default=20)
    parser.add_argument('--n_splines_te', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no_save_fold_predictions', action='store_true')
    args = parser.parse_args()

    exit_code = run(
        processed_data_path=args.processed_data,
        output_dir=args.output_dir,
        significance_output_dir=args.sig_output_dir,
        n_splines_s=args.n_splines_s,
        n_splines_te=args.n_splines_te,
        k=args.k,
        repeats=args.repeats,
        save_fold_predictions=not args.no_save_fold_predictions,
    )
    raise SystemExit(exit_code)
