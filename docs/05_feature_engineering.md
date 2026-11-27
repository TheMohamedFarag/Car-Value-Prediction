# Stage 1: Cleaning, Cross-Column Imputation, Feature Engineering & Ridge Baseline

## Overview
Prepares the car-price dataset (`cleaned_cars.csv`, 182,801 rows, 13 columns; `data_source` is dropped, leaving 12) for modeling. The dataset is split into training (80%) and test (20%) sets with `random_state=42` **before** any stateful transformer or encoder is fitted, which is the anti-leakage rule for the whole project.

## What is implemented
1. **Discovery & basic cleaning** (`locate_dataset`, `basic_clean`)
   - Looks for `cleaned_cars_2.csv` or `cleaned_cars.csv` under `data/`, and extracts from a `.zip` if needed.
   - Drops the constant `data_source` column; lowercases and strips text; merges `land rover` / `rover` into `land_rover`; removes hyphens from `model` and collapses repeated spaces.
2. **Split & persistence** (`split_first`, `save_split`, `load_split`)
   - Saves `X_train`, `X_test`, `y_train`, `y_test` to `artifacts/splits/split_data.pkl`.
3. **Rule-based cross-column imputation & feature engineering** (`CarPreprocessor`)
   - Flags placeholder odometer readings (`mileage <= 10`).
   - Fills `'unknown'` in `drive`, `fuel`, `transmission`, `type`, `cylinders` from regex rules on `model`.
   - Learns a `make` map from the first token of `model` on training data only.
   - Features: `car_age`, `is_classic`, `miles_per_year`, `log_mileage`, `cylinders_num`, `condition_score`, `is_clean_title`, `model_word_count`, `base_model`, and three regex flags.
   - `BinaryEncoder` on `model` (fitted on training data only) gives `model_0` … `model_14`.
   - Training-fitted aggregates (`mileage_ratio_to_year_avg`, `make_freq`, `base_model_freq`, `make_type_freq`) are included when `use_agg_features=True`.
4. **Target-aware wrapper & Ridge baseline** (`TargetAwareRegressor`, `run_ridge_baseline`)
   - `TargetAwareRegressor` has two training-only options: `log_target` (fit on `log1p(y)`, predict with `expm1`, with predictions clipped to the training log-range so that `expm1` cannot explode) and `drop_outliers` (remove gross outliers from training rows only).
   - Ridge baseline (`alpha=1.0`) on `log1p(price)`. Test results: MAE ≈ $4,937, RMSE ≈ $11,241, R² ≈ 0.587.
