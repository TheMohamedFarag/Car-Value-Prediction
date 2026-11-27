"""Stage 2: model families, experiment matrix, K-fold cross-validation.

Fixes vs. the original notebook:
  * ToCategoryDtype learns category levels from TRAIN folds only (unseen levels
    become NaN at transform time) - previously each frame re-coded its own
    categories, which can silently misalign codes for XGBoost / LightGBM.
  * Outlier filtering happens INSIDE each training fold (via
    fe.TargetAwareRegressor.drop_outliers). Previously it removed rows before
    CV, so 'filtered' validation folds lacked the hard rows and were not
    comparable with 'keep_all'.
"""
from __future__ import annotations

import importlib
import itertools
import json
import os

import lightgbm as lgbm
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

fe = importlib.import_module("05_feature_engineering")

# ---------------------------------------------------------------- config
MODEL_FAMILIES = ["linear", "ridge", "lasso", "random_forest",
                  "xgboost", "lightgbm", "catboost"]
TARGET_MODES = ["log1p", "raw"]
FEATURE_MODES = ["core", "full"]
OUTLIER_MODES = ["keep_all", "filtered"]


def rmse_metric(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


RMSE_SCORER = make_scorer(rmse_metric, greater_is_better=False)
MAE_SCORER = make_scorer(mean_absolute_error, greater_is_better=False)
SCORERS = {"mae": MAE_SCORER, "rmse": RMSE_SCORER, "r2": "r2"}


# ---------------------------------------------------------------- dtype transformer
class ToCategoryDtype(BaseEstimator, TransformerMixin):
    """Cast categorical columns to pandas 'category' with TRAIN-learned levels."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.categories_ = {c: pd.Index(sorted(X[c].dropna().unique()))
                            for c in self.cols if c in X.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X


def _numeric_selector(df):
    return [c for c in df.columns if c not in fe.CAT_COLS]


# ---------------------------------------------------------------- pipelines
def build_pipeline(model_name: str, use_agg_features: bool = True,
                   random_state: int = fe.RANDOM_STATE) -> Pipeline:
    """Leakage-safe pipeline for one estimator (no target transform here)."""
    prep = fe.CarPreprocessor(use_agg_features=use_agg_features)

    if model_name in ("linear", "ridge", "lasso"):
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scale", RobustScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                             ("ohe", fe.make_ohe(min_frequency=20,
                                                 handle_unknown="infrequent_if_exist"))])
        ct = ColumnTransformer([("num", num_pipe, _numeric_selector),
                                ("cat", cat_pipe, fe.CAT_COLS)], remainder="drop")
        reg = {"linear": LinearRegression(),
               "ridge": Ridge(alpha=1.0, max_iter=5000),
               "lasso": Lasso(alpha=0.001, max_iter=20000)}[model_name]
        return Pipeline([("prep", prep), ("feats", ct), ("model", reg)])

    if model_name == "random_forest":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                             ("ohe", fe.make_ohe(min_frequency=50,
                                                 handle_unknown="ignore",
                                                 sparse_output=False))])
        ct = ColumnTransformer([("num", num_pipe, _numeric_selector),
                                ("cat", cat_pipe, fe.CAT_COLS)], remainder="drop")
        return Pipeline([("prep", prep), ("feats", ct),
                         ("model", RandomForestRegressor(n_estimators=300,
                                                         min_samples_leaf=2,
                                                         max_features=0.5,
                                                         random_state=random_state,
                                                         n_jobs=-1))])

    if model_name == "xgboost":
        return Pipeline([
            ("prep", prep), ("dtypes", ToCategoryDtype(fe.CAT_COLS)),
            ("model", xgb.XGBRegressor(n_estimators=600, max_depth=7,
                                       learning_rate=0.05, subsample=0.9,
                                       colsample_bytree=0.8, reg_lambda=1.0,
                                       tree_method="hist", enable_categorical=True,
                                       random_state=random_state, n_jobs=-1))])

    if model_name == "lightgbm":
        return Pipeline([
            ("prep", prep), ("dtypes", ToCategoryDtype(fe.CAT_COLS)),
            ("model", lgbm.LGBMRegressor(n_estimators=600, num_leaves=127,
                                         learning_rate=0.05, subsample=0.9,
                                         subsample_freq=1, colsample_bytree=0.8,
                                         random_state=random_state, n_jobs=-1,
                                         verbose=-1))])

    if model_name == "catboost":
        return Pipeline([
            ("prep", prep),
            ("model", CatBoostRegressor(iterations=800, depth=8, learning_rate=0.06,
                                        cat_features=tuple(fe.CAT_COLS),  # tuple: survives sklearn clone()
                                        random_seed=random_state, verbose=0,
                                        thread_count=-1))])
    raise ValueError(f"unknown model: {model_name}")


def build_estimator(cfg: dict, use_agg_features: bool | None = None):
    """Full estimator for one config: pipeline + target/outlier options."""
    if use_agg_features is None:
        use_agg_features = cfg["features"] == "full"
    return fe.TargetAwareRegressor(
        base=build_pipeline(cfg["model"], use_agg_features=use_agg_features),
        log_target=(cfg["target"] == "log1p"),
        drop_outliers=(cfg["outliers"] == "filtered"))


# ---------------------------------------------------------------- experiment grid
def build_experiment_grid(models=MODEL_FAMILIES, targets=TARGET_MODES,
                          features=FEATURE_MODES, outliers=OUTLIER_MODES):
    return [dict(model=m, target=t, features=f, outliers=o)
            for m, t, f, o in itertools.product(models, targets, features, outliers)]


def evaluate_experiment(cfg, X_train, y_train, n_folds=5,
                        subsample_n=None, random_state=fe.RANDOM_STATE):
    Xtr, ytr = X_train, y_train
    # same subsample for every config so scores are comparable
    if subsample_n and len(Xtr) > subsample_n:
        idx = np.random.RandomState(random_state).choice(len(Xtr), subsample_n,
                                                         replace=False)
        Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]

    est = build_estimator(cfg)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    res = cross_validate(est, Xtr, ytr, cv=cv, scoring=SCORERS, n_jobs=1)
    return {**cfg,
            "n_train_rows": int(len(Xtr)),
            "cv_mae": float(-res["test_mae"].mean()),
            "cv_mae_std": float(res["test_mae"].std()),
            "cv_rmse": float(-res["test_rmse"].mean()),
            "cv_rmse_std": float(res["test_rmse"].std()),
            "cv_r2": float(res["test_r2"].mean()),
            "cv_r2_std": float(res["test_r2"].std()),
            "fit_time_s": float(res["fit_time"].mean())}


def run_all_experiments(X_train, y_train, models=None, targets=None, features=None,
                        outliers=None, n_folds=5, subsample_n=None,
                        random_state=fe.RANDOM_STATE,
                        out_csv="artifacts/metrics/all_experiments_comparison.csv",
                        best_json="artifacts/metrics/best_experiment_config.json"):
    grid = build_experiment_grid(models or MODEL_FAMILIES, targets or TARGET_MODES,
                                 features or FEATURE_MODES, outliers or OUTLIER_MODES)
    rows = []
    for i, cfg in enumerate(grid, 1):
        try:
            row = evaluate_experiment(cfg, X_train, y_train, n_folds=n_folds,
                                      subsample_n=subsample_n, random_state=random_state)
            msg = f"cv_rmse={row['cv_rmse']:,.0f}$  cv_r2={row['cv_r2']:.4f}  ({row['fit_time_s']:.0f}s)"
        except Exception as e:  # keep matrix running, record the failure
            row = {**cfg, "error": repr(e)}
            msg = f"ERROR: {e}"
        rows.append(row)
        print(f"[{i:02d}/{len(grid)}] {cfg['model']:<13s} {cfg['target']:<5s} "
              f"{cfg['features']:<5s} {cfg['outliers']:<9s} -> {msg}", flush=True)

    results = pd.DataFrame(rows)
    if "error" in results.columns:
        print(f"\n{results['error'].notna().sum()} configs failed; see 'error' column.")
    results = results.sort_values("cv_rmse", ignore_index=True, na_position="last")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"\nSaved comparison table -> {out_csv}")

    best = results.iloc[0]
    best_cfg = {"model": str(best["model"]), "target": str(best["target"]),
                "features": str(best["features"]), "outliers": str(best["outliers"]),
                "cv_mae": float(best["cv_mae"]), "cv_rmse": float(best["cv_rmse"]),
                "cv_r2": float(best["cv_r2"]), "fit_time_s": float(best["fit_time_s"]),
                "subsample_n": subsample_n}
    os.makedirs(os.path.dirname(best_json), exist_ok=True)
    with open(best_json, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"Saved best config    -> {best_json}")
    return results, best_cfg
