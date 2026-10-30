#!/usr/bin/env python3
"""Building damage grade (1/2/3) on DrivenData Richter's Predictor (Nepal 2015).

Reproduces the mid-review Categorical Naive Bayes baseline, then tests the
improvements proposed in the review (feature selection, discretisation,
class balancing) and compares against the base paper P4 (Chen & Zhang, 2022).

Protocol (no test-set peeking):
  * stratified 80/20 split (same ratio as the mid-review: 52,121 test rows of 260,601)
  * every choice (feature subset, bins, alpha, balancing) is picked with 5-fold
    stratified CV on the TRAINING split only
  * the test split is scored once per model at the end

Inputs (download from DrivenData, you must be logged in):
  train_values.csv   260,601 rows x (building_id + 38 features)
  train_labels.csv   building_id, damage_grade (1, 2, 3)

Usage:
  python damage_nb_compare.py --values data/train_values.csv --labels data/train_labels.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import warnings
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore", message="Bins whose width are too small")

SEED = 42
NUMERIC = ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]
# geographic IDs are categorical (as in the mid-review); remaining binary/cat columns are categorical too
P4_REFERENCE = {"model": "SMOTE-XGBoost (Chen & Zhang 2022, P4)", "macro_f1": 0.729}


# ----------------------------------------------------------------- data
def load(values_path: str, labels_path: str):
    X = pd.read_csv(values_path)
    y = pd.read_csv(labels_path)
    df = X.merge(y, on="building_id", how="inner", validate="one_to_one")
    assert len(df) == len(X) == len(y), "building_id join failed"
    target = df.pop("damage_grade").astype(int).values
    ids = df.pop("building_id")
    return df, target, ids


# ----------------------------------------------------------------- encoding
class NBEncoder:
    """Discretise numerics into quantile bins and map categoricals to ints.
    Unseen categories at predict time get their own code instead of crashing NB."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, df: pd.DataFrame):
        self.num_cols = [c for c in df.columns if c in NUMERIC]
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.kbins = None
        if self.num_cols:
            self.kbins = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile",
                                          subsample=None)
            self.kbins.fit(df[self.num_cols].astype(float))
        self.cat_maps = {}
        for c in self.cat_cols:
            cats = pd.Index(pd.unique(df[c].astype(str)))
            self.cat_maps[c] = cats
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts, names = [], []
        if self.kbins is not None:
            parts.append(self.kbins.transform(df[self.num_cols].astype(float)).astype(int))
            names += self.num_cols
        for c in self.cat_cols:
            cats = self.cat_maps[c]
            codes = cats.get_indexer(df[c].astype(str))
            codes = np.where(codes < 0, len(cats), codes)  # unseen -> extra code
            parts.append(codes.reshape(-1, 1))
            names.append(c)
        return np.hstack(parts)

    def n_categories(self, arr: np.ndarray) -> np.ndarray:
        # CategoricalNB needs min_categories >= max code + 1 per column
        return arr.max(axis=0) + 1


def encode_split(train_df, test_df, n_bins):
    enc = NBEncoder(n_bins).fit(train_df)
    A = enc.transform(train_df)
    B = enc.transform(test_df)
    cols = list(train_df.columns)
    n_cat = np.maximum(A.max(axis=0), B.max(axis=0)) + 1
    return A, B, cols, n_cat


# ----------------------------------------------------------------- model
def fit_predict(A, y, B, n_cat, alpha=1.0, balance=None, seed=SEED, class_prior=None):
    """balance: None | 'ros' (random oversampling) | 'uniform_prior' (equal class priors)."""
    Atr, ytr = A, y
    if balance == "ros":
        # random oversampling of minority grades: keeps category codes intact (no interpolation)
        Atr, ytr = RandomOverSampler(random_state=seed).fit_resample(A, y)
    elif balance == "uniform_prior":
        class_prior = np.full(3, 1 / 3)
    nb = CategoricalNB(alpha=alpha, min_categories=n_cat, fit_prior=class_prior is None,
                       class_prior=class_prior)
    nb.fit(Atr, ytr)
    return nb.predict(B), nb.predict_proba(B)


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1, 2, 3], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),          # = DrivenData micro-F1
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, c in enumerate([1, 2, 3])},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3]).tolist(),
    }


# ----------------------------------------------------------------- CV selection (train only)
def cv_score(df, y, cfg, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    f1s = []
    for tr, va in skf.split(df, y):
        dtr, dva = df.iloc[tr], df.iloc[va]
        cols = cfg["cols"]
        A, B, _, n_cat = encode_split(dtr[cols], dva[cols], cfg["n_bins"])
        pred, _ = fit_predict(A, y[tr], B, n_cat, alpha=cfg["alpha"], balance=cfg["balance"])
        f1s.append(f1_score(y[va], pred, average="macro"))
    return float(np.mean(f1s))


def select_features_mi(df, y, k):
    """Keep top-k features by mutual information computed on training data only."""
    enc = NBEncoder(5).fit(df)
    A = enc.transform(df)
    mi = mutual_info_classif(A, y, discrete_features=True, random_state=SEED)
    order = np.argsort(-mi)
    return [df.columns[i] for i in order[:k]], dict(zip(df.columns, mi.round(5).tolist()))


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="subsample 50k rows for a fast check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    df, y, _ = load(args.values, args.labels)
    if args.quick:
        keep = np.random.default_rng(SEED).choice(len(df), min(50000, len(df)), replace=False)
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    print(f"rows={len(df)} features={df.shape[1]} class dist={np.bincount(y)[1:] / len(y)}")

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    ytr, yte = y[tr_idx], y[te_idx]
    all_cols = list(df.columns)
    results = {}

    # 1) BASELINE: mid-review setup (all features, 5 quantile bins, alpha=1, no balancing)
    cfg = dict(cols=all_cols, n_bins=5, alpha=1.0, balance=None)
    A, B, _, n_cat = encode_split(Xtr[all_cols], Xte[all_cols], cfg["n_bins"])
    pred, _ = fit_predict(A, ytr, B, n_cat, alpha=1.0)
    results["baseline_naive_bayes"] = {"config": {k: v for k, v in cfg.items() if k != "cols"} | {"n_features": len(all_cols)},
                                       "test": scores(yte, pred)}

    # 2) FEATURE SELECTION by mutual information (k chosen by CV on train)
    mi_order, mi_scores = select_features_mi(Xtr, ytr, len(all_cols))
    best_k, best_cv = len(all_cols), -1
    for k in [10, 15, 20, 25, 30, len(all_cols)]:
        cv = cv_score(Xtr, ytr, dict(cols=mi_order[:k], n_bins=5, alpha=1.0, balance=None))
        print(f"  feature k={k:>2}  CV macro-F1={cv:.4f}")
        if cv > best_cv:
            best_k, best_cv = k, cv
    fs_cols = mi_order[:best_k]
    A, B, _, n_cat = encode_split(Xtr[fs_cols], Xte[fs_cols], 5)
    pred, _ = fit_predict(A, ytr, B, n_cat, alpha=1.0)
    results["plus_feature_selection"] = {"config": {"n_features": best_k, "selected": fs_cols},
                                         "test": scores(yte, pred)}

    # 3) DISCRETISATION + SMOOTHING + BALANCING grid (CV on train)
    grid = []
    for n_bins, alpha, balance in itertools.product([3, 5, 8], [0.1, 1.0], [None, "ros", "uniform_prior"]):
        cfg = dict(cols=fs_cols, n_bins=n_bins, alpha=alpha, balance=balance)
        cv = cv_score(Xtr, ytr, cfg, folds=3)
        grid.append({"n_bins": n_bins, "alpha": alpha, "balance": balance or "none", "cv_macro_f1": cv})
        print(f"  bins={n_bins} alpha={alpha} balance={balance or 'none':8s} CV macro-F1={cv:.4f}")
    best = max(grid, key=lambda g: g["cv_macro_f1"])
    cfg = dict(cols=fs_cols, n_bins=best["n_bins"], alpha=best["alpha"],
               balance=None if best["balance"] == "none" else best["balance"])
    A, B, _, n_cat = encode_split(Xtr[fs_cols], Xte[fs_cols], cfg["n_bins"])
    pred, _ = fit_predict(A, ytr, B, n_cat, alpha=cfg["alpha"], balance=cfg["balance"])
    results["improved_naive_bayes"] = {"config": {"n_features": len(fs_cols), "n_bins": cfg["n_bins"],
                                                  "alpha": cfg["alpha"], "balance": best["balance"],
                                                  "grid": grid},
                                       "test": scores(yte, pred)}

    base = results["baseline_naive_bayes"]["test"]
    imp = results["improved_naive_bayes"]["test"]
    summary = {
        "protocol": "stratified 80/20 split, seed 42; all choices by CV on train only",
        "rows_total": int(len(df)), "rows_train": int(len(tr_idx)), "rows_test": int(len(te_idx)),
        "class_distribution": {str(c): int((y == c).sum()) for c in [1, 2, 3]},
        "results": results,
        "comparison": {
            "baseline_accuracy": base["accuracy"], "improved_accuracy": imp["accuracy"],
            "baseline_macro_f1": base["macro_f1"], "improved_macro_f1": imp["macro_f1"],
            "delta_macro_f1": imp["macro_f1"] - base["macro_f1"],
            "base_paper_P4": P4_REFERENCE,
            "note": "P4 used XGBoost with its own split/preprocessing; compare as a reference, not a controlled test.",
        },
        "mutual_information": mi_scores,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["comparison"], indent=2))
    print(f"saved {args.out}/comparison.json")


if __name__ == "__main__":
    main()
