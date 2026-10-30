#!/usr/bin/env python3
"""Gradient-boosting models for building damage grade (1/2/3) on Richter's Predictor.

Same split as damage_nb_compare.py (stratified 80/20, seed 42) so results are comparable.
Early stopping uses a stratified 10% validation slice carved from the TRAIN split only.
The test split is scored once per model.

Usage:
  python damage_boost.py --values ... --labels ... --out results_boost
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
SEED = 42
GEO = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
SUP_PREFIX = "has_superstructure_"
SEC_PREFIX = "has_secondary_use"


def load(values_path, labels_path):
    X = pd.read_csv(values_path)
    y = pd.read_csv(labels_path)
    df = X.merge(y, on="building_id", how="inner", validate="one_to_one")
    assert len(df) == len(X) == len(y)
    target = df.pop("damage_grade").astype(int).values - 1  # 0,1,2 for xgboost/lightgbm
    ids = df.pop("building_id")
    return df, target, ids


def engineer(df: pd.DataFrame, geo_freq: dict[str, pd.Series]) -> pd.DataFrame:
    """Feature engineering. geo_freq maps geo column -> frequency table fitted on TRAIN only."""
    out = df.copy()
    sup = [c for c in df.columns if c.startswith(SUP_PREFIX)]
    sec = [c for c in df.columns if c.startswith(SEC_PREFIX)]
    out["n_superstructures"] = df[sup].sum(axis=1)
    out["n_secondary_uses"] = df[sec].sum(axis=1)
    f = df["count_floors_pre_eq"].clip(lower=1)
    a = df["area_percentage"].clip(lower=1)
    h = df["height_percentage"].clip(lower=1)
    out["height_per_floor"] = h / f
    out["area_per_floor"] = a / f
    out["volume_proxy"] = a * h
    out["families_per_area"] = df["count_families"] / a
    out["age_x_floors"] = df["age"] * f
    for g in GEO:
        out[f"{g}_freq"] = df[g].map(geo_freq[g]).fillna(0).astype(float)
    return out


def to_model_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {str(i + 1): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i in range(3)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="results_boost")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    df, y, _ = load(args.values, args.labels)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr_raw, Xte_raw = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    ytr, yte = y[tr_idx], y[te_idx]

    # validation slice for early stopping (from TRAIN only)
    fit_idx, val_idx = train_test_split(np.arange(len(Xtr_raw)), test_size=0.1, stratify=ytr,
                                        random_state=SEED)

    # geo frequency tables fitted on the fit slice only
    geo_freq = {g: Xtr_raw.iloc[fit_idx][g].value_counts(normalize=True) for g in GEO}

    Xfit = engineer(Xtr_raw.iloc[fit_idx].reset_index(drop=True), geo_freq)
    Xval = engineer(Xtr_raw.iloc[val_idx].reset_index(drop=True), geo_freq)
    Xte = engineer(Xte_raw, geo_freq)
    yfit, yval = ytr[fit_idx], ytr[val_idx]

    str_cols = [c for c in Xfit.columns if Xfit[c].dtype == object]
    cat_cols = str_cols + GEO
    Xfit_m, Xval_m, Xte_m = (to_model_frame(d, cat_cols) for d in (Xfit, Xval, Xte))
    for c in cat_cols:  # shared categories across splits
        allc = pd.Categorical(pd.concat([Xfit[c], Xval[c], Xte[c]]).astype(str)).categories
        for d in (Xfit_m, Xval_m, Xte_m):
            d[c] = pd.Categorical(d[c].astype(str), categories=allc)
    print(f"features={Xfit_m.shape[1]} categorical={cat_cols} fit={len(Xfit_m)} val={len(Xval_m)} test={len(Xte_m)}",
          flush=True)

    results = {}

    # ---- LightGBM -------------------------------------------------------
    lgb_params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=127,
                      min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, cat_smooth=20, max_cat_to_onehot=8, verbose=-1, seed=SEED,
                      num_threads=2)
    dtr = lgb.Dataset(Xfit_m, yfit, categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(Xval_m, yval, categorical_feature=cat_cols, reference=dtr)
    booster = lgb.train(lgb_params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p_lgb = booster.predict(Xte_m, num_iteration=booster.best_iteration)
    results["lightgbm"] = {"best_iteration": int(booster.best_iteration),
                           "test": metrics(yte, p_lgb.argmax(1))}
    print(f"LightGBM best_iter={booster.best_iteration} test acc={results['lightgbm']['test']['accuracy']:.4f} "
          f"macroF1={results['lightgbm']['test']['macro_f1']:.4f}", flush=True)

    # ---- XGBoost (hist, native categorical support: no one-hot blow-up) ----
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=3000,
                                  learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=2, reg_lambda=1.0, tree_method="hist", n_jobs=2,
                                  enable_categorical=True, max_cat_to_onehot=1, max_bin=128, random_state=SEED,
                                  early_stopping_rounds=100, eval_metric="mlogloss")
    # geo IDs as plain integers for XGBoost (their frequency features carry the signal; avoids
    # a memory blow-up from ~12k categories)
    def xgb_frame(d):
        d = d.copy()
        for g in GEO:
            d[g] = d[g].cat.codes.astype(np.int32)
        return d
    Xfit_x, Xval_x, Xte_x = xgb_frame(Xfit_m), xgb_frame(Xval_m), xgb_frame(Xte_m)
    xgb_model.fit(Xfit_x, yfit, eval_set=[(Xval_x, yval)], verbose=False)
    p_xgb = xgb_model.predict_proba(Xte_x)
    results["xgboost"] = {"best_iteration": int(xgb_model.best_iteration),
                          "test": metrics(yte, p_xgb.argmax(1))}
    print(f"XGBoost best_iter={xgb_model.best_iteration} test acc={results['xgboost']['test']['accuracy']:.4f} "
          f"macroF1={results['xgboost']['test']['macro_f1']:.4f}", flush=True)

    # ---- simple ensemble (average probabilities) -------------------------
    p_ens = (p_lgb + p_xgb) / 2
    results["lightgbm_xgboost_average"] = {"test": metrics(yte, p_ens.argmax(1))}
    print(f"Average ensemble acc={results['lightgbm_xgboost_average']['test']['accuracy']:.4f} "
          f"macroF1={results['lightgbm_xgboost_average']['test']['macro_f1']:.4f}", flush=True)

    summary = {
        "protocol": "same stratified 80/20 split (seed 42) as damage_nb_compare.py; early stopping on 10% of TRAIN",
        "rows_train": int(len(tr_idx)), "rows_test": int(len(te_idx)),
        "features_engineered": list(Xfit.columns),
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "boost_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.out}/boost_results.json  ({summary['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
