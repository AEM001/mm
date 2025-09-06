#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train multiple models for Q4 (female) classification.
- Prefers train/test from ADASYN outputs (female_train_adasyn.csv, female_test_original.csv)
- Falls back to splitting female_cleaned_anova_selected.csv (7:3 stratified)
- Trains:
  * RandomForest (5-fold CV on F1, class_weight=balanced)
  * XGBoost (if available; logistic with scale_pos_weight)
  * LightGBM (if available; is_unbalance/scale_pos_weight)
- Threshold for risk reporting: 0.7 on predicted probability
- Saves concise metrics, predictions, and feature importances under the script directory
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Optional dependencies
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False

ID_COLS = ["孕妇代码", "检测抽血次数"]
TARGET = "is_abnormal"


def pick_datasets(base_dir: Path, input_train: Path | None, input_test: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if input_train is not None and input_test is not None:
        df_tr = pd.read_csv(input_train)
        df_te = pd.read_csv(input_test)
        return df_tr, df_te
    # Prefer ADASYN outputs
    tr = base_dir / "female_train_adasyn.csv"
    te = base_dir / "female_test_original.csv"
    if tr.exists() and te.exists():
        return pd.read_csv(tr), pd.read_csv(te)
    # Fallback: split selected or cleaned
    cand = base_dir / "female_cleaned_anova_selected.csv"
    if not cand.exists():
        cand = base_dir / "female_cleaned.csv"
    df = pd.read_csv(cand)
    assert TARGET in df.columns, f"Target column not found in {cand}: {TARGET}"
    # Identify features
    feat_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X = df[feat_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.3, stratify=y, random_state=42
    )
    train_df = X_train.copy()
    test_df = X_test.copy()
    train_df[TARGET] = y_train.values
    test_df[TARGET] = y_test.values
    # Add IDs if available
    for c in ID_COLS:
        if c in df.columns:
            train_df[c] = df.loc[idx_tr, c].values
            test_df[c] = df.loc[idx_te, c].values
    # Reorder
    cols = [c for c in ID_COLS if c in train_df.columns] + [TARGET] + [c for c in feat_cols]
    train_df = train_df[cols]
    test_df = test_df[cols]
    return train_df, test_df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    id_present = [c for c in ID_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in id_present + [TARGET]]
    X = df[feat_cols]
    y = df[TARGET]
    return X, y, feat_cols


def scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0


def fit_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    numeric_features = list(X.columns)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ], remainder="drop")
    clf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 12],
        "clf__min_samples_split": [2, 10],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_


def fit_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not _HAVE_XGB:
        return None
    spw = scale_pos_weight(y)
    numeric_features = list(X.columns)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ], remainder="drop")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=42,
        tree_method="auto",
    )
    pipe = Pipeline([("pre", pre), ("clf", xgb)])
    pipe.fit(X, y)
    return pipe


def fit_lightgbm(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not _HAVE_LGBM:
        return None
    spw = scale_pos_weight(y)
    numeric_features = list(X.columns)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
    ], remainder="drop")
    lgbm = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=600,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=spw,
    )
    pipe = Pipeline([("pre", pre), ("clf", lgbm)])
    pipe.fit(X, y)
    return pipe


def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipe.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    # Some classifiers may output 1D prob
    return np.squeeze(proba)


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.7) -> Dict:
    p = predict_proba(pipe, X)
    y_hat = (p >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_hat)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "positive_rate_pred": float(np.mean(y_hat)),
    }
    return metrics


def save_predictions(out_dir: Path, name: str, df_test: pd.DataFrame, proba: np.ndarray, threshold: float = 0.7) -> None:
    id_cols = [c for c in ID_COLS if c in df_test.columns]
    out = pd.DataFrame({
        **({id_cols[0]: df_test[id_cols[0]].values} if len(id_cols) >= 1 else {}),
        **({id_cols[1]: df_test[id_cols[1]].values} if len(id_cols) >= 2 else {}),
        TARGET: df_test[TARGET].values,
        f"{name}_proba": proba,
        f"{name}_risk_flag": (proba >= threshold).astype(int),
    })
    out.to_csv(out_dir / f"predictions_{name}.csv", index=False)


def save_feature_importance(out_dir: Path, name: str, pipe: Pipeline, feature_names: List[str]) -> None:
    clf = pipe.named_steps.get("clf")
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = getattr(clf, "feature_importances_")
    elif hasattr(clf, "booster_") and hasattr(clf.booster_, "feature_importances_"):
        importances = clf.booster_.feature_importances_
    if importances is not None:
        imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp.sort_values("importance", ascending=False).to_csv(out_dir / f"feature_importance_{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train multiple models for Q4 female dataset")
    base_dir = Path(__file__).resolve().parent.parent
    out_dir_default = Path(__file__).resolve().parent
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--test_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=out_dir_default)
    parser.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    df_train, df_test = pick_datasets(base_dir, args.train_csv, args.test_csv)
    X_tr, y_tr, feat_cols = split_xy(df_train)
    X_te, y_te, _ = split_xy(df_test)

    summary: Dict[str, Dict] = {}

    # RandomForest with CV
    rf = fit_random_forest(X_tr, y_tr)
    rf_metrics = evaluate(rf, X_te, y_te, threshold=args.threshold)
    summary["random_forest"] = rf_metrics
    save_predictions(args.out_dir, "rf", df_test, predict_proba(rf, X_te), threshold=args.threshold)
    save_feature_importance(args.out_dir, "rf", rf, feat_cols)
    with open(args.out_dir / "rf_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(rf_metrics, f, ensure_ascii=False, indent=2)

    # XGBoost (if available)
    if _HAVE_XGB:
        xgb_pipe = fit_xgboost(X_tr, y_tr)
        xgb_metrics = evaluate(xgb_pipe, X_te, y_te, threshold=args.threshold)
        summary["xgboost"] = xgb_metrics
        save_predictions(args.out_dir, "xgb", df_test, predict_proba(xgb_pipe, X_te), threshold=args.threshold)
        save_feature_importance(args.out_dir, "xgb", xgb_pipe, feat_cols)
        with open(args.out_dir / "xgb_metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(xgb_metrics, f, ensure_ascii=False, indent=2)
    else:
        summary["xgboost"] = {"skipped": True, "reason": "xgboost not installed"}

    # LightGBM (if available)
    if _HAVE_LGBM:
        lgbm_pipe = fit_lightgbm(X_tr, y_tr)
        lgbm_metrics = evaluate(lgbm_pipe, X_te, y_te, threshold=args.threshold)
        summary["lightgbm"] = lgbm_metrics
        save_predictions(args.out_dir, "lgbm", df_test, predict_proba(lgbm_pipe, X_te), threshold=args.threshold)
        save_feature_importance(args.out_dir, "lgbm", lgbm_pipe, feat_cols)
        with open(args.out_dir / "lgbm_metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(lgbm_metrics, f, ensure_ascii=False, indent=2)
    else:
        summary["lightgbm"] = {"skipped": True, "reason": "lightgbm not installed"}

    # Save summary
    with open(args.out_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
