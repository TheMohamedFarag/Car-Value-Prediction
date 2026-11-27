# Stage 2: Model Families, Experiment Matrix & 5-Fold Cross-Validation

## Overview
Evaluates seven estimators (three linear variants, Random Forest, XGBoost, LightGBM, CatBoost) over an experiment matrix with 5-fold `KFold` (`shuffle=True`, `random_state=42`). Every preprocessing step, target transform and outlier filter is fitted inside the training folds. Metrics (MAE, RMSE, R²) are reported in real dollars.

## Experiment matrix
- **Models:** `linear`, `ridge`, `lasso`, `random_forest`, `xgboost`, `lightgbm`, `catboost`
- **Target:** `log1p` (fit on log, predict with clipped `expm1`) vs `raw`
- **Features:** `core` (no aggregates) vs `full` (with aggregates)
- **Outliers:** `keep_all` vs `filtered` (year == 1900, or year ≥ 2024 with price < $2,000, removed from each **training fold only**)

## Key fixes in this version
- Outlier filtering moved into the training fold; previously it was applied before CV, which made `filtered` and `keep_all` scores non-comparable.
- `ToCategoryDtype` learns category levels from training folds, so XGBoost/LightGBM codes cannot drift between train and validation.
- CatBoost `cat_features` is passed as a tuple so that sklearn `clone()` works inside `cross_validate`.

## Screening run
- Screening used a fixed random 20,000-row subsample of the training set (the same rows for every configuration) because the full matrix is too slow on a single CPU.
- Winner: **XGBoost, raw target, full features, keep_all outliers**, CV RMSE ≈ $10,211, CV R² ≈ 0.637, CV MAE ≈ $4,143.
- Runners-up: LightGBM raw/full/keep_all ≈ $10,264; CatBoost raw/full/keep_all ≈ $10,269; Random Forest best ≈ $10,883; linear models ≈ $12,500 or worse (full-feature linear fits are unstable).

Results are saved to `artifacts/metrics/stage2_*.csv` and the winner to `artifacts/metrics/best_experiment_config.json`.
