from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CHROMS = ["13", "18", "21"]


def _norm_col_name(s: str) -> str:
    """Normalize column names: strip whitespace, collapse spaces, remove BOMs."""
    if s is None:
        return s
    s = str(s)
    # replace special spaces/newlines/tabs
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # strip and collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_gestational_weeks(v: object) -> Optional[float]:
    """Parse strings like '13w+5' or '23w' into float weeks.
    Returns None if cannot parse.
    """
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    # common patterns: 13w+5, 17w+1, 23w, 12w + 5
    m = re.match(r"^(\d{1,2})\s*w(?:\s*\+\s*(\d{1,2}))?$", s)
    if not m:
        # sometimes format like '13+5' or '13周+5'
        m2 = re.match(r"^(\d{1,2})(?:周|w)?\s*\+\s*(\d{1,2})$", s)
        if m2:
            w = int(m2.group(1))
            d = int(m2.group(2))
            return w + d / 7.0
        m3 = re.match(r"^(\d{1,2})(?:周|w)$", s)
        if m3:
            return float(m3.group(1))
        return None
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) else 0
    return w + d / 7.0


def to_datetime_safe(v: object, fmt: Optional[str] = None) -> Optional[pd.Timestamp]:
    if pd.isna(v):
        return None
    try:
        if fmt:
            return pd.to_datetime(str(v), format=fmt, errors="coerce")
        # try excel serial or standard parsing
        return pd.to_datetime(v, errors="coerce")
    except Exception:
        return None


 


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    col = "染色体的非整倍体"
    if col not in df.columns:
        raise KeyError("缺少标注列：染色体的非整倍体")
    lab = df[col].astype(str).str.upper().str.replace(" ", "", regex=False)
    lab = lab.replace({"NAN": np.nan})
    df["ab_label_raw"] = lab
    df["is_abnormal"] = lab.notna().astype(int)
    for c in CHROMS:
        df[f"ab_T{c}"] = df["ab_label_raw"].fillna("").str.contains(f"T{c}").astype(int)
    # multi-class type
    df["ab_type"] = df["ab_label_raw"].fillna("NORMAL")
    return df


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    z_cols = [f"{c}号染色体的Z值" for c in CHROMS]
    for zc in z_cols + ["X染色体的Z值"]:
        if zc in df.columns:
            df[zc] = pd.to_numeric(df[zc], errors="coerce")
    z_abs = df[[c for c in z_cols if c in df.columns]].abs()
    df["rule_z_any_ge_3"] = (z_abs >= 3).any(axis=1)
    df["rule_z_any_ge_2p5"] = (z_abs >= 2.5).any(axis=1)
    return df


def clean_female_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    df = df.rename(columns={c: _norm_col_name(c) for c in df.columns})
    # drop unnamed columns
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    # parse weeks
    if "检测孕周" in df.columns:
        df["检测孕周_周"] = df["检测孕周"].apply(parse_gestational_weeks)

    # dates
    if "末次月经" in df.columns:
        df["末次月经_dt"] = pd.to_datetime(df["末次月经"], errors="coerce")
    if "检测日期" in df.columns:
        # 支持整数 yyyymmdd 或 日期
        def _parse_detect_date(v):
            if pd.isna(v):
                return pd.NaT
            try:
                if isinstance(v, (int, np.integer)):
                    return pd.to_datetime(str(int(v)), format="%Y%m%d", errors="coerce")
                if isinstance(v, float) and v.is_integer():
                    return pd.to_datetime(str(int(v)), format="%Y%m%d", errors="coerce")
            except Exception:
                pass
            return pd.to_datetime(v, errors="coerce")
        df["检测日期_dt"] = df["检测日期"].apply(_parse_detect_date)

    # gestational weeks from dates
    if "末次月经_dt" in df.columns and "检测日期_dt" in df.columns:
        delta_days = (df["检测日期_dt"] - df["末次月经_dt"]).dt.days
        df["孕周_按日期_周"] = (delta_days / 7.0).round(2)
        # prefer string weeks; fallback to date-based
        df["孕周_周"] = df["检测孕周_周"].fillna(df["孕周_按日期_周"]) if "检测孕周_周" in df.columns else df["孕周_按日期_周"]
    elif "检测孕周_周" in df.columns:
        df["孕周_周"] = df["检测孕周_周"]

    # numeric coercions（扩展：包含过滤比例、怀孕/生产次数等）
    num_cols = [
        "孕妇BMI", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数",
        "被过滤掉读段数的比例", "GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "X染色体浓度",
        "怀孕次数", "生产次数",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derived metrics
    if {"唯一比对的读段数", "重复读段的比例"}.issubset(df.columns):
        df["唯一有效读段数"] = df["唯一比对的读段数"] * (1 - df["重复读段的比例"])

    # labels and rule flags
    df = build_labels(df)
    df = rule_flags(df)

    # IVF妊娠二值编码，自然受孕=0，IVF=1（未知保持NaN）
    if "IVF妊娠" in df.columns:
        def _encode_ivf(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip().lower()
            if "自然" in s:
                return 0
            if "ivf" in s or "试管" in s or "体外" in s:
                return 1
            return np.nan
        df["IVF妊娠_编码"] = df["IVF妊娠"].apply(_encode_ivf)

    # 高龄产妇指示：年龄≥35 -> 1，否则0
    if "年龄" in df.columns:
        df["高龄"] = (pd.to_numeric(df["年龄"], errors="coerce") >= 35).astype("Int64")

    # 标准化连续变量（不对Z值进行标准化；其余连续变量均标准化）
    cont_cols = [
        # 时点与母体因素
        "孕周_周", "年龄", "孕妇BMI", "怀孕次数", "生产次数",
        # 测序质量
        "原始读段数", "被过滤掉读段数的比例", "唯一比对的读段数", "在参考基因组上比对的比例", "GC含量",
        # 染色体GC与其他连续变量
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "X染色体浓度",
    ]
    cont_cols = [c for c in cont_cols if c in df.columns]
    for c in cont_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mean = s.mean(skipna=True)
        std = s.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            df[f"{c}_std"] = 0.0
        else:
            df[f"{c}_std"] = (s - mean) / std

    # 精简特征选择：保留指导中涉及的特征（连续变量保留标准化版，Z值保留原值）
    keep_cols = [
        # 标识与基本信息
        "孕妇代码", "检测抽血次数", "IVF妊娠_编码", "高龄",

        # 连续变量（标准化）
        "孕周_周_std", "年龄_std", "孕妇BMI_std", "怀孕次数_std", "生产次数_std",
        "原始读段数_std", "被过滤掉读段数的比例_std", "唯一比对的读段数_std", "在参考基因组上比对的比例_std", "GC含量_std",
        "13号染色体的GC含量_std", "18号染色体的GC含量_std", "21号染色体的GC含量_std", "X染色体浓度_std",

        # Z值（原值）
        "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",

        # 标签
        "is_abnormal", "ab_T13", "ab_T18", "ab_T21"
    ]
    
    # 确保所有列都存在，过滤掉不存在的列
    keep_cols = [c for c in keep_cols if c in df.columns]

    cleaned = df[keep_cols].copy()
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Q4 Female data processing")
    default_input = Path(__file__).resolve().parent.parent / "附件.xlsx"
    parser.add_argument("--input", type=Path, default=default_input, help="Path to 附件.xlsx")
    parser.add_argument("--sheet", type=str, default="女胎检测数据", help="Sheet name for female data")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent, help="Output directory")
    args = parser.parse_args()

    assert args.input.exists(), f"Input Excel not found: {args.input}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(args.input)
    assert args.sheet in xls.sheet_names, f"Sheet not found: {args.sheet}; sheets={xls.sheet_names}"

    df = pd.read_excel(xls, sheet_name=args.sheet)
    cleaned = clean_female_sheet(df)

    out_csv = args.out_dir / "female_cleaned.csv"
    cleaned.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
