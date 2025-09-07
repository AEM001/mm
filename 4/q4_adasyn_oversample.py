#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADASYN oversampling for Q4 female dataset
- Split data 7:3 (stratified by is_abnormal)
- Apply ADASYN on training set to reach minority ratio ≈ 0.4 (异常:正常 ~ 4:6)
- Keep code concise; default input is female_cleaned_anova_selected.csv, fallback to female_cleaned.csv
- Outputs do NOT overwrite originals; writes:
  - 4/female_train_original.csv
  - 4/female_test_original.csv
  - 4/female_train_adasyn.csv (oversampled)

Note: imbalanced-learn is not required; a lightweight ADASYN is implemented with scikit-learn NearestNeighbors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer


DEF_ID_COLS = ["孕妇代码", "检测抽血次数"]


def _pick_input(default_dir: Path, user_path: Path | None) -> Path:
    if user_path is not None:
        return user_path
    p1 = default_dir / "female_cleaned_anova_selected.csv"
    if p1.exists():
        return p1
    p2 = default_dir / "female_cleaned.csv"
    assert p2.exists(), f"Neither {p1} nor {p2} found"
    return p2


def _adasyn_generate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_neighbors_density: int = 5,
    n_neighbors_minority: int = 3,
    target_minority_ratio: float = 0.4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(random_state)
    y = y.astype(int)
    minority_label = 1
    idx_min = np.where(y == minority_label)[0]
    idx_maj = np.where(y != minority_label)[0]

    n_min = idx_min.size
    n_maj = idx_maj.size
    if n_min == 0:
        return pd.DataFrame(columns=feature_names), 0

    # Target count for minority to reach desired ratio: m*/(m*+maj) = r => m* = (r/(1-r))*maj
    target_min_count = int(np.ceil((target_minority_ratio / (1 - target_minority_ratio)) * n_maj))
    n_to_generate = max(0, target_min_count - n_min)
    if n_to_generate <= 0:
        return pd.DataFrame(columns=feature_names), 0

    # Density difficulty via ρ_i = (# minority within k density neighbors) / k; difficulty = 1-ρ_i
    k_d = min(n_neighbors_density, X.shape[0])
    nbrs_all = NearestNeighbors(n_neighbors=min(k_d + 1, X.shape[0])).fit(X)
    neigh_ind_all = nbrs_all.kneighbors(X[idx_min], return_distance=False)
    # exclude self (first index is itself)
    neigh_ind_all = neigh_ind_all[:, 1:]
    rho = []
    for row in neigh_ind_all:
        labels = (y[row] == minority_label).astype(int)
        rho.append(labels.mean() if len(labels) > 0 else 0.0)
    rho = np.asarray(rho)
    difficulty = 1.0 - rho  # higher difficulty => more synthetic samples
    if np.all(difficulty <= 0):
        weights = np.ones_like(difficulty) / len(difficulty)
    else:
        s = difficulty.sum()
        weights = difficulty / s

    # Minority neighbor graph for synthesis (exclude self)
    if n_min >= 2:
        k_m = min(n_neighbors_minority + 1, n_min)
        nbrs_min = NearestNeighbors(n_neighbors=k_m).fit(X[idx_min])
        neigh_ind_min = nbrs_min.kneighbors(X[idx_min], return_distance=False)
        neigh_ind_min = neigh_ind_min[:, 1:]
    else:
        neigh_ind_min = None

    # Allocate counts per minority sample
    alloc = np.floor(weights * n_to_generate).astype(int)
    remainder = n_to_generate - alloc.sum()
    if remainder > 0:
        # distribute by top weights
        order = np.argsort(-weights)
        alloc[order[:remainder]] += 1

    synth_rows = []
    for i_local, count in enumerate(alloc):
        if count <= 0:
            continue
        i_global = idx_min[i_local]
        xi = X[i_global]

        if n_min >= 2 and neigh_ind_min is not None and neigh_ind_min.shape[1] > 0:
            # choose neighbors among nearest minority
            choices_local = neigh_ind_min[i_local]
            for _ in range(count):
                j_local = int(rng.choice(choices_local))
                xj = X[idx_min[j_local]]
                alpha = float(rng.uniform(0.0, 1.0))
                x_new = xi + alpha * (xj - xi)
                synth_rows.append(x_new)
        else:
            # fallback: jittering around xi
            for _ in range(count):
                noise = rng.normal(loc=0.0, scale=1e-3, size=xi.shape)
                x_new = xi + noise
                synth_rows.append(x_new)

    synth_arr = np.vstack(synth_rows) if synth_rows else np.empty((0, X.shape[1]))
    synth_df = pd.DataFrame(synth_arr, columns=feature_names)
    return synth_df, n_to_generate


def main():
    parser = argparse.ArgumentParser(description="ADASYN oversampling for Q4 female dataset")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--input_csv", type=Path, default=None, help="Path to input CSV; default: selected then cleaned in same directory")
    parser.add_argument("--out_dir", type=Path, default=default_dir)
    parser.add_argument("--target", type=str, default="is_abnormal")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--target_minority_ratio", type=float, default=0.4)
    parser.add_argument("--k_density", type=int, default=5)
    parser.add_argument("--k_synth", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    in_path = _pick_input(default_dir, args.input_csv)
    assert in_path.exists(), f"Input CSV not found: {in_path}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    assert args.target in df.columns, f"Target column not found: {args.target}"

    # Identify ID columns present
    id_cols = [c for c in DEF_ID_COLS if c in df.columns]

    # Candidate feature columns: numeric and not target and not ids
    feat_cols = [
        c for c in df.columns
        if c not in id_cols + [args.target] and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=int)

    # Stratified split 7:3
    X_train, X_test, y_train, y_test, df_train_idx, df_test_idx = train_test_split(
        X, y, df.index.to_numpy(), test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Median imputation to remove NaNs for neighbor-based synthesis and for saved splits
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Build original train/test DataFrames (keep IDs if present)
    train_orig = pd.DataFrame(X_train_imp, columns=feat_cols)
    test_orig = pd.DataFrame(X_test_imp, columns=feat_cols)
    train_orig[args.target] = y_train
    test_orig[args.target] = y_test
    # Attach IDs if available
    if id_cols:
        train_orig[id_cols] = df.loc[df_train_idx, id_cols].to_numpy()
        test_orig[id_cols] = df.loc[df_test_idx, id_cols].to_numpy()
        # reorder columns: ids, target, features
        train_orig = train_orig[id_cols + [args.target] + feat_cols]
        test_orig = test_orig[id_cols + [args.target] + feat_cols]
    else:
        train_orig = train_orig[[args.target] + feat_cols]
        test_orig = test_orig[[args.target] + feat_cols]

    # ADASYN on training set
    synth_df, n_gen = _adasyn_generate(
        X_train_imp, y_train, feat_cols,
        n_neighbors_density=args.k_density,
        n_neighbors_minority=args.k_synth,
        target_minority_ratio=args.target_minority_ratio,
        random_state=args.random_state,
    )

    if n_gen > 0:
        synth_df[args.target] = 1
        if id_cols:
            # Fill synthetic IDs deterministically
            synth_df[id_cols[0]] = [f"SYN_{i}" for i in range(n_gen)]
            if len(id_cols) > 1:
                synth_df[id_cols[1]] = 0
            synth_df = synth_df[id_cols + [args.target] + feat_cols]
        else:
            synth_df = synth_df[[args.target] + feat_cols]
        train_adasyn = pd.concat([train_orig, synth_df], ignore_index=True)
    else:
        train_adasyn = train_orig.copy()

    # Save outputs
    out_dir = args.out_dir
    train_orig_path = out_dir / "female_train_original.csv"
    test_orig_path = out_dir / "female_test_original.csv"
    train_adasyn_path = out_dir / "female_train_adasyn.csv"

    train_orig.to_csv(train_orig_path, index=False)
    test_orig.to_csv(test_orig_path, index=False)
    train_adasyn.to_csv(train_adasyn_path, index=False)

    # Console summary
    n_min_before = int((y_train == 1).sum())
    n_maj_before = int((y_train == 0).sum())
    n_min_after = int((train_adasyn[args.target] == 1).sum())
    n_maj_after = int((train_adasyn[args.target] == 0).sum())
    ratio_after = n_min_after / (n_min_after + n_maj_after)
    print({
        "input": str(in_path),
        "train_original": str(train_orig_path),
        "test_original": str(test_orig_path),
        "train_adasyn": str(train_adasyn_path),
        "train_minority_before": n_min_before,
        "train_majority_before": n_maj_before,
        "generated": n_gen,
        "train_minority_after": n_min_after,
        "train_majority_after": n_maj_after,
        "train_minority_ratio_after": round(float(ratio_after), 4),
    })


if __name__ == "__main__":
    main()
