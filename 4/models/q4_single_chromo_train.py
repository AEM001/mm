#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single chromosome classification for Q4 female dataset.
- Trains separate binary classifiers for ab_T13, ab_T18, ab_T21
- Uses same preprocessing pipeline as multi-class version
- Handles severe class imbalance with ADASYN oversampling per chromosome
- Outputs results organized by chromosome type under 4/models/single_chromo/

Key differences from combined approach:
- Three independent binary classification tasks
- Chromosome-specific ADASYN oversampling (target ratio 0.4)
- Separate model evaluation and feature importance per chromosome
- Threshold optimization per chromosome type
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
from sklearn.neighbors import NearestNeighbors

# Optional dependencies
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False

ID_COLS = ["孕妇代码", "检测抽血次数"]
CHROMO_TARGETS = ["ab_T13", "ab_T18", "ab_T21"]


def adasyn_oversample(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.4, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified ADASYN for single chromosome oversampling."""
    rng = np.random.default_rng(random_state)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    
    n_min = len(minority_idx)
    n_maj = len(majority_idx)
    
    if n_min == 0:
        return X, y
    
    target_min_count = int(np.ceil((target_ratio / (1 - target_ratio)) * n_maj))
    n_to_generate = max(0, target_min_count - n_min)
    
    if n_to_generate <= 0:
        return X, y
    
    # Simple density-based weight calculation
    k = min(5, len(X))
    if k > 1:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        _, indices = nbrs.kneighbors(X[minority_idx])
        
        weights = []
        for i, neighbors in enumerate(indices):
            minority_neighbors = sum(1 for idx in neighbors if y[idx] == 1)
            difficulty = 1.0 - (minority_neighbors / k)  # Higher difficulty = more synthesis
            weights.append(max(difficulty, 0.1))  # Minimum weight
        
        weights = np.array(weights)
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_min) / n_min
    
    # Allocate synthesis counts
    alloc = np.floor(weights * n_to_generate).astype(int)
    remainder = n_to_generate - alloc.sum()
    if remainder > 0:
        top_indices = np.argsort(-weights)[:remainder]
        alloc[top_indices] += 1
    
    # Generate synthetic samples
    synthetic_X = []
    synthetic_y = []
    
    for i, count in enumerate(alloc):
        if count <= 0:
            continue
            
        xi = X[minority_idx[i]]
        
        # Find nearest minority neighbors for interpolation
        if n_min >= 2:
            minority_X = X[minority_idx]
            distances = np.linalg.norm(minority_X - xi, axis=1)
            nearest_indices = np.argsort(distances)[1:min(4, n_min)]  # Exclude self
            
            for _ in range(count):
                if len(nearest_indices) > 0:
                    j = rng.choice(nearest_indices)
                    xj = minority_X[j]
                    alpha = rng.uniform(0.0, 1.0)
                    x_new = xi + alpha * (xj - xi)
                else:
                    # Fallback: add small noise
                    x_new = xi + rng.normal(0, 0.01, xi.shape)
                
                synthetic_X.append(x_new)
                synthetic_y.append(1)
    
    if synthetic_X:
        X_new = np.vstack([X, np.array(synthetic_X)])
        y_new = np.hstack([y, np.array(synthetic_y)])
        return X_new, y_new
    
    return X, y


def prepare_data_for_chromosome(df: pd.DataFrame, target_col: str, test_size: float = 0.3, 
                               target_ratio: float = 0.4, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train/test splits with ADASYN oversampling for specific chromosome."""
    # Identify feature columns
    id_cols = [c for c in ID_COLS if c in df.columns]
    exclude_cols = set(id_cols + CHROMO_TARGETS)
    feat_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    X = df[feat_cols].values
    y = df[target_col].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    
    # Stratified split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Apply ADASYN to training set
    X_train_over, y_train_over = adasyn_oversample(X_train, y_train, target_ratio, random_state)
    
    # Convert back to DataFrames
    train_df = pd.DataFrame(X_train_over, columns=feat_cols)
    train_df[target_col] = y_train_over
    
    test_df = pd.DataFrame(X_test, columns=feat_cols)
    test_df[target_col] = y_test
    
    # Add IDs to test set
    for col in id_cols:
        if col in df.columns:
            test_df[col] = df.loc[idx_test, col].values
    
    # Add synthetic IDs to oversampled training set
    n_original = len(y_train)
    n_synthetic = len(y_train_over) - n_original
    
    for col in id_cols:
        if col in df.columns:
            original_ids = df.loc[idx_train, col].values
            if col == ID_COLS[0]:  # Primary ID
                synthetic_ids = [f"SYN_{target_col}_{i}" for i in range(n_synthetic)]
                train_df[col] = list(original_ids) + synthetic_ids
            else:  # Secondary ID
                train_df[col] = list(original_ids) + [0] * n_synthetic
    
    return train_df, test_df, pd.DataFrame(X_train, columns=feat_cols), pd.DataFrame(X_test, columns=feat_cols)


def train_models_for_chromosome(X_train: pd.DataFrame, y_train: pd.Series, target_col: str) -> Dict[str, Pipeline]:
    """Train all available models for a specific chromosome."""
    models = {}
    
    # Random Forest
    numeric_features = list(X_train.columns)
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric_features)], remainder="drop")
    
    rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    rf_pipe = Pipeline([("pre", pre), ("clf", rf)])
    
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 12],
        "clf__min_samples_split": [2, 10],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(rf_pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    models["random_forest"] = gs.best_estimator_
    
    # XGBoost
    if _HAVE_XGB:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.05,
            n_estimators=500,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        )
        xgb_pipe = Pipeline([("pre", pre), ("clf", xgb)])
        xgb_pipe.fit(X_train, y_train)
        models["xgboost"] = xgb_pipe
    
    # LightGBM
    if _HAVE_LGBM:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        lgbm = LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=600,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        )
        lgbm_pipe = Pipeline([("pre", pre), ("clf", lgbm)])
        lgbm_pipe.fit(X_train, y_train)
        models["lightgbm"] = lgbm_pipe
    
    return models


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
    """Evaluate model performance."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Single chromosome classification for Q4 female dataset")
    parser.add_argument("--input_csv", type=Path, default=Path("4/female_cleaned_anova_selected.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("4/models/single_chromo"))
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--target_ratio", type=float, default=0.4)
    parser.add_argument("--ratio_T13", type=float, default=0.4)
    parser.add_argument("--ratio_T18", type=float, default=0.4)
    parser.add_argument("--ratio_T21", type=float, default=0.4)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    
    # Check which chromosome targets are available
    available_targets = [col for col in CHROMO_TARGETS if col in df.columns]
    if not available_targets:
        raise ValueError(f"No chromosome targets found in input data. Expected: {CHROMO_TARGETS}")
    
    results = {}
    
    ratio_map = {
        "ab_T13": args.ratio_T13,
        "ab_T18": args.ratio_T18,
        "ab_T21": args.ratio_T21,
    }

    for target_col in available_targets:
        print(f"\nProcessing {target_col}...")
        
        # Check class distribution
        class_counts = df[target_col].value_counts()
        if len(class_counts) < 2 or class_counts.min() < 5:
            print(f"Skipping {target_col}: insufficient positive samples ({class_counts.get(1, 0)})")
            continue
        
        # Prepare data
        tr_ratio = float(ratio_map.get(target_col, args.target_ratio))
        train_df, test_df, train_orig, test_orig = prepare_data_for_chromosome(
            df, target_col, args.test_size, tr_ratio, args.random_state
        )
        
        # Extract features and targets
        feat_cols = [c for c in train_df.columns if c not in [target_col] + ID_COLS]
        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test = test_df[feat_cols]
        y_test = test_df[target_col]
        
        # Train models
        models = train_models_for_chromosome(X_train, y_train, target_col)
        
        # Evaluate at different thresholds
        chromo_results = {}
        chromo_dir = args.out_dir / target_col
        chromo_dir.mkdir(exist_ok=True)
        
        best_overall = {"model": None, "threshold": None, "metrics": None, "f1": -1.0}
        for model_name, model in models.items():
            model_results = {}

            for threshold in args.thresholds:
                metrics = evaluate_model(model, X_test, y_test, threshold)
                model_results[f"threshold_{threshold}"] = metrics

                # Save predictions for this threshold
                y_proba = model.predict_proba(X_test)[:, 1]
                pred_df = test_df[ID_COLS + [target_col]].copy() if any(c in test_df.columns for c in ID_COLS) else test_df[[target_col]].copy()
                pred_df[f"{model_name}_proba"] = y_proba
                pred_df[f"{model_name}_pred"] = (y_proba >= threshold).astype(int)
                pred_df.to_csv(chromo_dir / f"predictions_{model_name}_t{threshold}.csv", index=False)

            chromo_results[model_name] = model_results

            # Save feature importance if available
            clf = model.named_steps.get("clf")
            if hasattr(clf, "feature_importances_"):
                imp_df = pd.DataFrame({
                    "feature": feat_cols,
                    "importance": clf.feature_importances_
                }).sort_values("importance", ascending=False)
                imp_df.to_csv(chromo_dir / f"feature_importance_{model_name}.csv", index=False)

            # Track best by F1 across thresholds for this model
            for threshold, metrics in model_results.items():
                f1 = metrics.get("f1", 0.0) or 0.0
                t_val = float(threshold.split("_")[-1])
                if f1 > best_overall["f1"]:
                    best_overall = {"model": model_name, "threshold": t_val, "metrics": metrics, "f1": f1}

        results[target_col] = chromo_results

        # Save chromosome-specific summary
        with open(chromo_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(chromo_results, f, ensure_ascii=False, indent=2)

        # Save best choice and final predictions
        if best_overall["model"] is not None:
            with open(chromo_dir / "best_choice.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in best_overall.items() if k != "f1"}, f, ensure_ascii=False, indent=2)
            # Produce final predictions for the best setup
            best_model = models[best_overall["model"]]
            y_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= float(best_overall["threshold"])) .astype(int)
            final_df = test_df[ID_COLS + [target_col]].copy() if any(c in test_df.columns for c in ID_COLS) else test_df[[target_col]].copy()
            final_df[f"{best_overall['model']}_proba"] = y_proba
            final_df[f"{best_overall['model']}_pred"] = y_pred
            final_df.to_csv(chromo_dir / "final_predictions.csv", index=False)
        
        print(f"Completed {target_col}. Results saved to {chromo_dir}")
    
    # Save overall summary
    with open(args.out_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAll results saved to {args.out_dir}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
