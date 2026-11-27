# Stage 3: Hyperparameter Tuning, Final Test Evaluation & Model Export

## Overview
Loads the Stage-2 winner (`best_experiment_config.json`), tunes it with `RandomizedSearchCV` on the **full training set**, evaluates the refitted pipeline once on the untouched 20% test split, and exports the model, test predictions and metadata.

Run: `python 07_hyperparameter_tuning_and_export.py <n_iter> <n_folds>`. This run used `8 3`.

## Search space
`PARAM_SPACES` holds distributions for every family. For XGBoost the grid covers `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`. All keys are prefixed with `base__` because the target and outlier options live in `TargetAwareRegressor`.

## Tuning and deviation from the original plan
- Scoring: CV RMSE in dollars.
- This run used **8 iterations and 3 folds** instead of the planned 25 and 5. The reason is compute: the environment has a single CPU, and each full-data XGBoost fit takes about a minute. Re-run with `python 07_hyperparameter_tuning_and_export.py 25 5` if more time is available.
- Tuned CV RMSE on the full training set: ≈ $8,394. This is **not comparable** with the Stage-2 figure of ≈ $10,211, which was measured on a 20k subsample.

## Final results (untouched test set, 20%)
| Metric | Ridge baseline | Final model (tuned XGBoost) |
|---|---|---|
| MAE | $4,937 | **$3,448** |
| RMSE | $11,241 | **$8,749** |
| R² | 0.587 | **0.750** |

Best hyperparameters: `n_estimators=400`, `max_depth=9`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.6`, `min_child_weight=1`, `reg_lambda=1.0`.

## Artifacts
- `artifacts/models/best_car_price_model.pkl`: the complete pipeline (`CarPreprocessor` → features → tuned XGBoost in a `TargetAwareRegressor`).
- `artifacts/predictions/test_predictions.csv`: test `y_true`, `y_pred`, `abs_error`.
- `artifacts/metrics/final_model_metadata.json`: export time, winning config, best params, CV and test metrics, input columns, and the required modules.

## Using the model
The `.py` modules `05_feature_engineering.py` and `06_machine_learning.py` must be on `sys.path` before `joblib.load`, because the pickle references their classes. Input is a DataFrame with the 11 raw columns listed in the metadata (`make, model, year, mileage, condition, cylinders, fuel, title_status, transmission, drive, type`); `predict` returns prices in dollars.
