"""Stage 3: hyperparameter tuning of the Stage-2 winner, test evaluation, export.

Fixes vs. the original notebook:
  * Parameter keys are always prefixed with `base__` (log/outlier options now
    live in TargetAwareRegressor, so there is one wrapper, not two).
  * Exported metadata lists the modules needed to unpickle the model.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

fe = importlib.import_module("05_feature_engineering")
ml = importlib.import_module("06_machine_learning")

PARAM_SPACES = {
    "linear": {},
    "ridge": {"model__alpha": loguniform(1e-2, 1e4)},
    "lasso": {"model__alpha": loguniform(1e-4, 1e1)},
    "random_forest": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [0.3, 0.5, 0.8]},
    "xgboost": {
        "model__n_estimators": [400, 700, 1000],
        "model__max_depth": [5, 7, 9],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.8, 0.9],
        "model__colsample_bytree": [0.6, 0.8],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [1.0, 3.0]},
    "lightgbm": {
        "model__n_estimators": [400, 700, 1000],
        "model__num_leaves": [63, 127, 255],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8],
        "model__min_child_samples": [20, 50],
        "model__reg_lambda": [0.0, 1.0]},
    "catboost": {
        "model__iterations": [600, 900, 1200],
        "model__depth": [6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__l2_leaf_reg": [1.0, 3.0, 5.0]},
}

MODEL_PATH = "artifacts/models/best_car_price_model.pkl"
PRED_PATH = "artifacts/predictions/test_predictions.csv"
META_PATH = "artifacts/metrics/final_model_metadata.json"


def load_best_config(path="artifacts/metrics/best_experiment_config.json"):
    with open(path) as f:
        return json.load(f)


def tune_winner(cfg, X_train, y_train, n_iter=25, n_folds=5,
                random_state=fe.RANDOM_STATE):
    """RandomizedSearchCV on the winning model + formula. Outlier filtering
    happens inside each CV fold through the estimator itself."""
    est = ml.build_estimator(cfg)
    space = PARAM_SPACES.get(cfg["model"], {})
    space = {"base__" + k: v for k, v in space.items()}

    if not space:  # plain LinearRegression: nothing to tune
        est.fit(X_train, y_train)
        return est, {}, None

    search = RandomizedSearchCV(est, param_distributions=space, n_iter=n_iter,
                                scoring=ml.RMSE_SCORER,
                                cv=KFold(n_splits=n_folds, shuffle=True,
                                         random_state=random_state),
                                random_state=random_state, n_jobs=1, refit=True,
                                error_score="raise")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search


def final_evaluate(est, X_test, y_test):
    pred = est.predict(X_test)
    metrics = {"test_mae": float(mean_absolute_error(y_test, pred)),
               "test_rmse": float(ml.rmse_metric(y_test, pred)),
               "test_r2": float(r2_score(y_test, pred))}
    preds = pd.DataFrame({"y_true": np.asarray(y_test), "y_pred": np.asarray(pred)},
                         index=getattr(y_test, "index", None))
    preds["abs_error"] = (preds["y_pred"] - preds["y_true"]).abs()
    return metrics, preds


def export_artifacts(est, cfg, best_params, cv_rmse, test_metrics, preds,
                     model_path=MODEL_PATH, pred_path=PRED_PATH, meta_path=META_PATH):
    for p in (model_path, pred_path, meta_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    joblib.dump(est, model_path)
    preds.to_csv(pred_path, index_label="row_index")
    input_cols = ["make", "model", "year", "mileage", "condition", "cylinders",
                  "fuel", "title_status", "transmission", "drive", "type"]
    usage = (
        "# Requires 05_feature_engineering.py and 06_machine_learning.py on sys.path "
        "(the pickle references their classes).\n"
        "import sys, joblib, pandas as pd\n"
        "sys.path.append('<folder containing the .py modules>')\n"
        "model = joblib.load('artifacts/models/best_car_price_model.pkl')\n"
        f"X = pd.DataFrame([...])  # columns: {input_cols}\n"
        "prices = model.predict(X)  # returns price in dollars")
    meta = {"exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "winning_config": cfg,
            "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                            for k, v in (best_params or {}).items()},
            "cv_rmse_dollars_stage2": None if cv_rmse is None else float(cv_rmse),
            "test_metrics": test_metrics,
            "input_columns": input_cols,
            "required_modules": ["05_feature_engineering.py", "06_machine_learning.py"],
            "usage": usage}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Model       -> {model_path}")
    print(f"Predictions -> {pred_path}")
    print(f"Metadata    -> {meta_path}")
    return {"model": model_path, "predictions": pred_path, "metadata": meta_path}


def main(n_iter=25, n_folds=5):
    data = fe.load_split()
    cfg = load_best_config()
    print("winning config:", cfg, flush=True)
    est, params, search = tune_winner(cfg, data["X_train"], data["y_train"],
                                      n_iter=n_iter, n_folds=n_folds)
    if search is not None:
        print(f"best CV RMSE (tuned): ${-search.best_score_:,.0f}", flush=True)
    metrics, preds = final_evaluate(est, data["X_test"], data["y_test"])
    print("TEST:", metrics, flush=True)
    export_artifacts(est, cfg, params, cfg.get("cv_rmse"), metrics, preds)


if __name__ == "__main__":
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(n_iter=n_iter, n_folds=n_folds)
