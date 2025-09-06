#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4 Female QC script (独立质控)
- 从 附件.xlsx -> 女胎检测数据 读取原始质量相关字段
- 计算 qc_pass 与 qc_fail_reasons（失败原因串）
- 输出 4/female_qc.csv（可通过 --out_dir 自定义）

阈值（可按需修改）：
- 唯一有效读段数 = 唯一比对的读段数 × (1 - 重复读段的比例) ≥ 3,000,000
- 在参考基因组上比对的比例 ≥ 0.80
- 重复读段的比例 ≤ 0.30
- 被过滤掉读段数的比例 ≤ 0.10
- GC含量 ∈ [0.35, 0.45]

用法：
  python q4_female_qc.py \
    --input ../附件.xlsx \
    --sheet 女胎检测数据 \
    --out_dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import re


def _norm_col_name(s: str) -> str:
    if s is None:
        return s
    s = str(s)
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_qc(row: pd.Series,
               min_unique_effective_reads: float = 3_000_000,
               min_map_rate: float = 0.80,
               max_dup_rate: float = 0.30,
               max_filter_ratio: float = 0.10,
               gc_low: float = 0.35,
               gc_high: float = 0.45) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    uniq = row.get("唯一比对的读段数", np.nan)
    dup = row.get("重复读段的比例", np.nan)
    mapr = row.get("在参考基因组上比对的比例", np.nan)
    filt = row.get("被过滤掉读段数的比例", np.nan)
    gc = row.get("GC含量", np.nan)

    uniq_eff = np.nan
    if pd.notna(uniq) and pd.notna(dup):
        uniq_eff = uniq * (1 - dup)
        if uniq_eff < min_unique_effective_reads:
            reasons.append(f"唯一有效读段数<{min_unique_effective_reads:.0f}")
    else:
        reasons.append("唯一有效读段数缺失")

    if pd.notna(mapr) and mapr < min_map_rate:
        reasons.append(f"比对比例<{min_map_rate}")
    if pd.notna(dup) and dup > max_dup_rate:
        reasons.append(f"重复比例>{max_dup_rate}")
    if pd.notna(filt) and filt > max_filter_ratio:
        reasons.append(f"过滤比例>{max_filter_ratio}")
    if pd.notna(gc) and (gc < gc_low or gc > gc_high):
        reasons.append(f"GC含量不在[{gc_low},{gc_high}]")

    return (len(reasons) == 0), reasons


def main():
    parser = argparse.ArgumentParser(description="Q4 Female independent QC")
    default_input = Path(__file__).resolve().parent.parent / "附件.xlsx"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to 附件.xlsx")
    parser.add_argument("--sheet", type=str, default="女胎检测数据", help="Sheet name for female data")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent, help="Output directory")

    parser.add_argument("--min_unique_effective_reads", type=float, default=3_000_000)
    parser.add_argument("--min_map_rate", type=float, default=0.80)
    parser.add_argument("--max_dup_rate", type=float, default=0.30)
    parser.add_argument("--max_filter_ratio", type=float, default=0.10)
    parser.add_argument("--gc_low", type=float, default=0.35)
    parser.add_argument("--gc_high", type=float, default=0.45)

    args = parser.parse_args()

    assert args.input.exists(), f"Input Excel not found: {args.input}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(args.input)
    assert args.sheet in xls.sheet_names, f"Sheet not found: {args.sheet}; sheets={xls.sheet_names}"

    df = pd.read_excel(xls, sheet_name=args.sheet)
    df = df.rename(columns={c: _norm_col_name(c) for c in df.columns})
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    # numeric coercions for QC related columns
    for c in [
        "唯一比对的读段数", "重复读段的比例", "在参考基因组上比对的比例",
        "被过滤掉读段数的比例", "GC含量"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    qc_pass_list = []
    qc_reason_list = []
    for _, r in df.iterrows():
        ok, reasons = compute_qc(
            r,
            min_unique_effective_reads=args.min_unique_effective_reads,
            min_map_rate=args.min_map_rate,
            max_dup_rate=args.max_dup_rate,
            max_filter_ratio=args.max_filter_ratio,
            gc_low=args.gc_low,
            gc_high=args.gc_high,
        )
        qc_pass_list.append(ok)
        qc_reason_list.append(";".join(reasons))

    # 输出包含标识与关键QC字段
    out_cols = [
        c for c in [
            "序号", "孕妇代码", "检测抽血次数", "检测日期", "检测孕周", "孕妇BMI",
            "唯一比对的读段数", "重复读段的比例", "在参考基因组上比对的比例",
            "被过滤掉读段数的比例", "GC含量"
        ] if c in df.columns
    ]
    qc_df = df[out_cols].copy()
    qc_df["qc_pass"] = qc_pass_list
    qc_df["qc_fail_reasons"] = qc_reason_list

    out_csv = args.out_dir / "female_qc.csv"
    qc_df.to_csv(out_csv, index=False)

    # 控制台简要输出
    print({
        'output': str(out_csv),
        'rows': len(qc_df),
        'qc_pass_count': int(qc_df['qc_pass'].sum()),
    })


if __name__ == "__main__":
    main()
