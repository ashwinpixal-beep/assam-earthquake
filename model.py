#!/usr/bin/env python3
"""Building damage grade model (DrivenData Richter's Predictor, Nepal 2015 earthquake).

Final model = average of class probabilities from LightGBM and XGBoost, trained on
engineered building features. Grades: 1 (low), 2 (medium), 3 (near-complete destruction).

Protocol used for the reported results
--------------------------------------
* Stratified 80/20 split of train_values/train_labels (seed 42). The 20% test split is
  only used for the final evaluation.
* Inside the training 80%, a stratified 10% slice is held out for early stopping.
  Geographic frequency tables are computed from the 90% fit slice only.

Usage
-----
Train and evaluate (writes models/damage_model.joblib and models/metrics.json):
    python model.py train --values train_values.csv --labels train_labels.csv --out models

Predict on new building rows (e.g. test_values.csv) and write a submission-style CSV:
    python model.py predict --bundle models/damage_model.joblib \
        --values test_values.csv --out predictions.csv

Requirements: numpy, pandas, scikit-learn, lightgbm, xgboost, joblib
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 42
ID_COL = "building_id"
TARGET = "damage_grade"
GEO = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
SUP_PREFIX = "has_superstructure_"
SEC_PREFIX = "has_secondary_use"

LGB_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=127,
    min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, cat_smooth=20, max_cat_to_onehot=8, verbose=-1, seed=SEED, num_threads=2,
)
XGB_PARAMS = dict(
    objective="multi:softprob", num_class=3, n_estimators=3000, learning_rate=0.05,
    max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
    tree_method="hist", n_jobs=2, enable_categorical=True, max_cat_to_onehot=1, max_bin=128,
    random_state=SEED, early_stopping_rounds=100, eval_metric="mlogloss",
)


# --------------------------------------------------------------------------- data
def load_train(values_path: str, labels_path: str):
    """Join features and labels on building_id. Returns (raw features, y in {0,1,2})."""
    X = pd.read_csv(values_path)
    y = pd.read_csv(labels_path)
    df = X.merge(y, on=ID_COL, how="inner", validate="one_to_one")
    if len(df) != len(X) or len(df) != len(y):
        raise ValueError("building_id values do not match between values and labels files")
    labels = df[TARGET].astype(int).values - 1
    return df.drop(columns=[TARGET]), labels


def load_features(values_path: str) -> pd.DataFrame:
    return pd.read_csv(values_path)


# --------------------------------------------------------------------------- features
def engineer(raw: pd.DataFrame, geo_freq: dict[str, pd.Series]) -> pd.DataFrame:
    """Build model features from raw survey columns. geo_freq comes from the fit slice only."""
    df = raw.drop(columns=[c for c in (ID_COL, TARGET) if c in raw.columns]).copy()
    sup = [c for c in df.columns if c.startswith(SUP_PREFIX)]
    sec = [c for c in df.columns if c.startswith(SEC_PREFIX)]

    out = df.copy()
    out["n_superstructures"] = df[sup].sum(axis=1)
    out["n_secondary_uses"] = df[sec].sum(axis=1)

    floors = df["count_floors_pre_eq"].clip(lower=1)
    area = df["area_percentage"].clip(lower=1)
    height = df["height_percentage"].clip(lower=1)
    out["height_per_floor"] = height / floors
    out["area_per_floor"] = area / floors
    out["volume_proxy"] = area * height
    out["families_per_area"] = df["count_families"] / area
    out["age_x_floors"] = df["age"] * floors

    for g in GEO:
        out[f"{g}_freq"] = df[g].map(geo_freq[g]).fillna(0.0).astype(float)
    return out


# --------------------------------------------------------------------------- model
class DamageModel:
    """LightGBM + XGBoost probability-average classifier for building damage grade."""

    def __init__(self):
        self.geo_freq: dict[str, pd.Series] | None = None
        self.cat_cols: list[str] | None = None
        self.categories: dict[str, pd.Index] | None = None
        self.lgb_model: lgb.Booster | None = None
        self.xgb_model: xgb.XGBClassifier | None = None
        self.best_iterations: dict[str, int] = {}

    # -- internal helpers ---------------------------------------------------
    def _lgb_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.cat_cols:
            out[c] = pd.Categorical(out[c].astype(str), categories=self.categories[c])
        return out

    @staticmethod
    def _xgb_frame(L: pd.DataFrame) -> pd.DataFrame:
        out = L.copy()
        for g in GEO:  # high-cardinality geo IDs as plain codes (avoids memory blow-up)
            out[g] = out[g].cat.codes.astype(np.int32)
        return out

    # -- training -----------------------------------------------------------
    def fit(self, raw: pd.DataFrame, y: np.ndarray, val_frac: float = 0.1) -> "DamageModel":
        fit_i, val_i = train_test_split(np.arange(len(raw)), test_size=val_frac,
                                        stratify=y, random_state=SEED)
        fit_raw = raw.iloc[fit_i].reset_index(drop=True)
        val_raw = raw.iloc[val_i].reset_index(drop=True)
        y_fit, y_val = y[fit_i], y[val_i]

        self.geo_freq = {g: fit_raw[g].value_counts(normalize=True) for g in GEO}
        X_fit = engineer(fit_raw, self.geo_freq)
        X_val = engineer(val_raw, self.geo_freq)

        self.cat_cols = [c for c in X_fit.columns if X_fit[c].dtype == object] + GEO
        self.categories = {c: pd.Categorical(pd.concat([X_fit[c], X_val[c]]).astype(str)).categories
                           for c in self.cat_cols}

        L_fit, L_val = self._lgb_frame(X_fit), self._lgb_frame(X_val)

        # LightGBM with native categorical splits
        d_fit = lgb.Dataset(L_fit, y_fit, categorical_feature=self.cat_cols, free_raw_data=False)
        d_val = lgb.Dataset(L_val, y_val, categorical_feature=self.cat_cols, reference=d_fit)
        self.lgb_model = lgb.train(
            LGB_PARAMS, d_fit, num_boost_round=3000, valid_sets=[d_val],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        self.best_iterations["lightgbm"] = int(self.lgb_model.best_iteration)

        # XGBoost (hist, native categoricals for text columns, codes for geo IDs)
        self.xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        self.xgb_model.fit(self._xgb_frame(L_fit), y_fit,
                           eval_set=[(self._xgb_frame(L_val), y_val)], verbose=False)
        self.best_iterations["xgboost"] = int(self.xgb_model.best_iteration)
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Averaged probabilities, shape (n, 3) for grades 1, 2, 3."""
        L = self._lgb_frame(engineer(raw, self.geo_freq))
        p_lgb = self.lgb_model.predict(L, num_iteration=self.best_iterations["lightgbm"])
        p_xgb = self.xgb_model.predict_proba(self._xgb_frame(L))
        return (p_lgb + p_xgb) / 2.0

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        """Predicted damage grade in {1, 2, 3}."""
        return self.predict_proba(raw).argmax(axis=1) + 1

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # record the importable module name, so other programs (e.g. the web app) can load the bundle
        DamageModel.__module__ = "model"
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "DamageModel":
        return joblib.load(path)


# --------------------------------------------------------------------------- evaluation
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics on grades 0..2 (internal encoding)."""
    labels = [0, 1, 2]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {str(i + 1): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i in range(3)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------- CLI
def cmd_train(args) -> None:
    t0 = time.time()
    raw, y = load_train(args.values, args.labels)
    idx = np.arange(len(raw))
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    raw_tr, raw_te = raw.iloc[tr_i].reset_index(drop=True), raw.iloc[te_i].reset_index(drop=True)
    y_tr, y_te = y[tr_i], y[te_i]

    model = DamageModel().fit(raw_tr, y_tr)
    proba = model.predict_proba(raw_te)
    metrics = {
        "ensemble": evaluate(y_te, proba.argmax(axis=1)),
        "lightgbm_only": evaluate(y_te, model.lgb_model.predict(
            model._lgb_frame(engineer(raw_te, model.geo_freq)),
            num_iteration=model.best_iterations["lightgbm"]).argmax(axis=1)),
        "best_iterations": model.best_iterations,
        "rows_train": int(len(tr_i)), "rows_test": int(len(te_i)),
        "protocol": "stratified 80/20 split (seed 42); early stopping on 10% of the training split",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "damage_model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    e = metrics["ensemble"]
    print(f"Ensemble  accuracy={e['accuracy']:.4f}  macro_f1={e['macro_f1']:.4f}  "
          f"weighted_f1={e['weighted_f1']:.4f}")
    print(f"LightGBM  accuracy={metrics['lightgbm_only']['accuracy']:.4f}")
    print(f"saved {args.out}/damage_model.joblib and metrics.json ({metrics['elapsed_sec']}s)")


def cmd_predict(args) -> None:
    model = DamageModel.load(args.bundle)
    raw = load_features(args.values)
    grades = model.predict(raw)
    out = pd.DataFrame({ID_COL: raw[ID_COL], TARGET: grades})
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions to {args.out}")
    print(out[TARGET].value_counts(normalize=True).sort_index().round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train, evaluate and save the model")
    t.add_argument("--values", required=True, help="train_values.csv")
    t.add_argument("--labels", required=True, help="train_labels.csv")
    t.add_argument("--out", default="models", help="output folder")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict damage grades for new rows")
    p.add_argument("--bundle", required=True, help="path to damage_model.joblib")
    p.add_argument("--values", required=True, help="CSV with building_id and the 38 features")
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
