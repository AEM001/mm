#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ANOVA的特征选择，用于female_cleaned.csv
- 读取由q4_female_processing.py生成的清洗后数据
- 对每个数值特征与二元目标变量is_abnormal运行单因素方差分析(ANOVA)
- 输出ANOVA报告和包含选定特征的过滤数据集

设计选择:
- 当--always_keep_z为True时，即使p值>阈值，也保留原始Z分数特征(13/18/21, X)
- 默认情况下，如果SciPy可用，则通过p值进行阈值筛选；否则回退到使用eta平方阈值
- 不覆盖原始清洗数据；使用不同名称写入新文件

使用方法:
  python q4_anova_feature_select.py \
    --input_csv 4/female_cleaned.csv \
    --out_dir 4 \
    --p_threshold 0.05 \
    --eta_threshold 0.01 \
    --always_keep_z
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import f as f_dist
    _HAVE_SCIPY = True
except Exception:
    f_dist = None  # type: ignore
    _HAVE_SCIPY = False


Z_COLS = [
    "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"
]


def anova_two_groups(x: pd.Series, y: pd.Series) -> Tuple[float, float, int, int, Optional[float]]:
    """Compute one-way ANOVA (two groups) F-statistic and p-value if SciPy available.
    Returns: (F, eta_sq, df_between, df_within, p_value)
    """
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return (np.nan, np.nan, 1, 0, None)

    g0 = x[y == 0]
    g1 = x[y == 1]
    n0 = g0.shape[0]
    n1 = g1.shape[0]
    k = 2
    N = n0 + n1
    if n0 < 2 or n1 < 2:
        return (np.nan, np.nan, k - 1, max(N - k, 0), None)

    m = x.mean()
    m0 = g0.mean()
    m1 = g1.mean()

    ssb = n0 * (m0 - m) ** 2 + n1 * (m1 - m) ** 2
    ssw = ((g0 - m0) ** 2).sum() + ((g1 - m1) ** 2).sum()
    dfb = k - 1
    dfw = N - k

    if dfw <= 0 or ssw <= 0:
        return (np.nan, np.nan, dfb, dfw, None)

    msb = ssb / dfb
    msw = ssw / dfw
    F = float(msb / msw)
    eta_sq = float(ssb / (ssb + ssw)) if (ssb + ssw) > 0 else np.nan

    p_val: Optional[float] = None
    if _HAVE_SCIPY:
        try:
            p_val = float(f_dist.sf(F, dfb, dfw))
        except Exception:
            p_val = None

    return (F, eta_sq, dfb, dfw, p_val)


def main():
    parser = argparse.ArgumentParser(description="ANOVA feature selection for female_cleaned.csv")
    parser.add_argument("--input_csv", type=Path, default=Path(__file__).resolve().parent / "female_cleaned.csv")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", type=str, default="is_abnormal")
    parser.add_argument("--p_threshold", type=float, default=0.10)
    parser.add_argument("--eta_threshold", type=float, default=0.008)
    parser.add_argument("--always_keep_z", action="store_true", default=True)

    args = parser.parse_args()
    assert args.input_csv.exists(), f"Input CSV not found: {args.input_csv}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    assert args.target in df.columns, f"Target column not found: {args.target}"

    # Determine candidate numeric features (exclude target and label indicators)
    exclude_cols = set([args.target, "ab_T13", "ab_T18", "ab_T21"])
    candidates = [
        c for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    y = df[args.target]
    report_rows = []
    selected = []

    for c in candidates:
        F, eta, dfb, dfw, p = anova_two_groups(pd.to_numeric(df[c], errors="coerce"), y)
        row = {
            "feature": c,
            "F": F,
            "eta_sq": eta,
            "df_between": dfb,
            "df_within": dfw,
            "p_value": p,
            "n_total": int(y.notna().sum()),
            "n_class0": int(((y == 0) & df[c].notna()).sum()),
            "n_class1": int(((y == 1) & df[c].notna()).sum()),
            "mean_class0": float(df.loc[(y == 0) & df[c].notna(), c].mean()) if (((y == 0) & df[c].notna()).any()) else np.nan,
            "mean_class1": float(df.loc[(y == 1) & df[c].notna(), c].mean()) if (((y == 1) & df[c].notna()).any()) else np.nan,
            "is_z": c in Z_COLS,
        }
        report_rows.append(row)

        keep = False
        if row["is_z"] and args.always_keep_z:
            keep = True
        else:
            if p is not None:
                keep = (p <= args.p_threshold)
            else:
                keep = (eta is not None and not np.isnan(eta) and eta >= args.eta_threshold)
        if keep:
            selected.append(c)

    report_df = pd.DataFrame(report_rows).sort_values(by=["p_value", "F"], ascending=[True, False], na_position="last")
    report_csv = args.out_dir / "female_anova_report.csv"
    report_df.to_csv(report_csv, index=False)

    # Build filtered dataset: keep ID, target, selected features
    id_cols = [c for c in ["孕妇代码", "检测抽血次数"] if c in df.columns]
    out_cols = id_cols + [args.target] + selected
    out_cols = [c for c in out_cols if c in df.columns]

    filtered = df[out_cols].copy()
    out_csv = args.out_dir / "female_cleaned_anova_selected.csv"
    filtered.to_csv(out_csv, index=False)

    print({
        "report": str(report_csv),
        "selected_csv": str(out_csv),
        "n_features_in": len(candidates),
        "n_features_selected": len(selected),
        "have_scipy": _HAVE_SCIPY,
    })


if __name__ == "__main__":
    main()
