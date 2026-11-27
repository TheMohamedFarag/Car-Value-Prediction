"""Stage 1: cleaning, cross-column imputation, feature engineering, Ridge baseline.

Fixes vs. the original notebook:
  * locate_dataset accepts either cleaned_cars_2.csv or cleaned_cars.csv.
  * Target-side logic (log1p / outlier filter) lives in TargetAwareRegressor
    (see 06_machine_learning) so that outlier filtering happens inside CV folds.
"""
from __future__ import annotations

import glob
import inspect
import os
import re
import zipfile

import category_encoders as ce
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ---------------------------------------------------------------- config
RANDOM_STATE = 42
TEST_SIZE = 0.2
REF_YEAR = 2026
TARGET = "price"
DATASET_NAMES = ("cleaned_cars_2.csv", "cleaned_cars.csv")

CAT_COLS = ["make", "condition", "cylinders", "fuel", "title_status",
            "transmission", "drive", "type", "base_model"]

CONDITION_SCORE = {"new": 5, "excellent": 4, "like new": 3, "good": 2,
                   "fair": 1, "salvage": 0, "unknown": -1}

_DRIVE_RULES = [(r"4wd|4x4|awd|four wheel", "4wd"),
                (r"\bfwd\b|front wheel", "fwd"),
                (r"\brwd\b|rear wheel", "rwd")]
_FUEL_RULES = [(r"\belectric\b|\bev\b", "electric"),
               (r"hybrid", "hybrid"),
               (r"diesel", "diesel")]
_TRANS_RULES = [(r"manual|stick shift", "manual"),
                (r"automatic|\bauto\b", "automatic")]
_TYPE_RULES = [(r"minivan|mini van", "mini-van"),
               (r"\bsuv\b|sport utility", "suv"),
               (r"pickup|pick up", "pickup"),
               (r"\btruck\b", "truck"),
               (r"sedan", "sedan"),
               (r"coupe", "coupe"),
               (r"hatchback", "hatchback"),
               (r"convertible|cabriolet|roadster", "convertible"),
               (r"\bvan\b", "van"),
               (r"wagon", "wagon"),
               (r"\bbus\b", "bus")]
_CYL_RULES = [(r"\bv?12\b|12\s*cyl", "12 cylinders"),
              (r"\bv?10\b|10\s*cyl", "10 cylinders"),
              (r"\bv8\b|8\s*cyl", "8 cylinders"),
              (r"\bv6\b|6\s*cyl", "6 cylinders"),
              (r"5\s*cyl", "5 cylinders"),
              (r"\bv?4\b|4\s*cyl|\bi4\b|\bl4\b", "4 cylinders"),
              (r"3\s*cyl", "3 cylinders")]

LUXURY_SPORT_RE = re.compile(
    r"\b(?:limited|platinum|premium|luxury|prestige|signature|touring|sport|"
    r"denali|laramie|autobiography|hse|quattro|type r|s line|n line|xse|"
    r"gt|gts|amg|rs)\b")
WORK_HEAVY_RE = re.compile(
    r"\b(?:2500|3500|4500|5500|hd|super duty|superduty|cargo|work van|dualie|"
    r"f250|f350|f450|f550|heavy duty)\b")
TURBO_PERF_RE = re.compile(
    r"\b(?:turbo|twin turbo|ecoboost|supercharged|shelby|hellcat|gt500|gt350|"
    r"z06|zr1|nismo|gtr|gti|srt|r/t|rt|type r|trd pro)\b")


# ---------------------------------------------------------------- I/O & split
def locate_dataset(data_dir: str = "data", names: tuple = DATASET_NAMES) -> str:
    """Find the CSV (any of `names`); extract from a .zip if needed."""
    for fname in names:
        hits = glob.glob(os.path.join(data_dir, "**", fname), recursive=True)
        if hits:
            return hits[0]
    for zpath in glob.glob(os.path.join(data_dir, "**", "*.zip"), recursive=True):
        with zipfile.ZipFile(zpath) as z:
            for fname in names:
                member = next((n for n in z.namelist() if n.endswith(fname)), None)
                if member:
                    z.extract(member, data_dir)
                    print(f"[locate_dataset] extracted '{member}' from '{zpath}'")
                    return os.path.join(data_dir, member)
    raise FileNotFoundError(f"None of {names} found under '{data_dir}'.")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, deterministic cleaning (safe to apply before the split)."""
    df = df.copy()
    if "data_source" in df.columns:
        df = df.drop(columns=["data_source"])
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["make"] = df["make"].replace({"land rover": "land_rover", "rover": "land_rover"})
    df["model"] = (df["model"].str.replace("-", "", regex=False)
                              .str.replace(r"\s+", " ", regex=True))
    return df


def unknown_report(df: pd.DataFrame) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object", "string"])
    n = (obj == "unknown").sum()
    return pd.DataFrame({"unknown_count": n,
                         "unknown_pct": (100 * n / len(df)).round(2)}
                        ).sort_values("unknown_count", ascending=False)


def split_first(df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split BEFORE any fitting - the core anti-leakage rule."""
    y = df[target]
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_split(X_train, X_test, y_train, y_test,
               path="artifacts/splits/split_data.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"X_train": X_train, "X_test": X_test,
                 "y_train": y_train, "y_test": y_test}, path)
    print(f"Split saved -> {path}")


def load_split(path="artifacts/splits/split_data.pkl"):
    return joblib.load(path)


def make_ohe(min_frequency=None, handle_unknown="ignore", sparse_output=True):
    """OneHotEncoder compatible with old/new sklearn (sparse vs sparse_output)."""
    param = ("sparse_output"
             if "sparse_output" in inspect.signature(OneHotEncoder).parameters
             else "sparse")
    return OneHotEncoder(min_frequency=min_frequency, handle_unknown=handle_unknown,
                         **{param: sparse_output})


def _fill_by_rules(df, col, rules):
    for pat, val in rules:
        m = (df[col] == "unknown") & df["model"].str.contains(pat, regex=True, na=False)
        df.loc[m, col] = val
    return df


def _one_base_model(s: str) -> str:
    t = s.split()
    if not t:
        return "unknown"
    if len(t) > 1 and t[1].isdigit():
        return f"{t[0]} {t[1]}"
    return t[0]


# ---------------------------------------------------------------- preprocessor
class CarPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe feature pipeline.

    fit():       learns make<-model map, per-year mileage medians, frequency
                 tables and the BinaryEncoder using TRAIN data only.
    transform(): cleaning -> cross-column imputation -> feature engineering
                 -> binary columns for `model` (model_0 ... model_14).
    """
    AGG_COLS = ["mileage_ratio_to_year_avg", "make_freq", "base_model_freq",
                "make_type_freq"]

    def __init__(self, use_agg_features=True):
        self.use_agg_features = use_agg_features

    def fit(self, X, y=None):
        df = self._prepare_text(X, apply_make_map=False)
        known = df[df["make"] != "unknown"]
        self.make_from_model_ = (known.assign(_tok=known["model"].str.split().str[0])
                                      .groupby("_tok")["make"]
                                      .agg(lambda s: s.mode().iat[0]).to_dict())
        self.year_mileage_median_ = df.groupby("year")["mileage"].median()
        self.global_mileage_median_ = float(df["mileage"].median())
        self.make_freq_ = np.log1p(df["make"].value_counts())
        self.base_model_freq_ = np.log1p(df["base_model"].value_counts())
        self.make_type_freq_ = np.log1p((df["make"] + "_" + df["type"]).value_counts())
        self.binary_encoder_ = ce.BinaryEncoder(cols=["model"], return_df=True,
                                                handle_unknown="value",
                                                handle_missing="value")
        self.binary_encoder_.fit(df[["model"]])
        sample = self.transform(X.head(64))
        self.feature_names_out_ = list(sample.columns)
        self.cat_cols_ = [c for c in sample.columns if c in CAT_COLS]
        self.numeric_cols_ = [c for c in sample.columns if c not in CAT_COLS]
        return self

    def transform(self, X):
        df = self._prepare_text(X, apply_make_map=True)
        med = df["year"].map(self.year_mileage_median_).fillna(self.global_mileage_median_)
        df["mileage_ratio_to_year_avg"] = df["mileage"] / (med + 1.0)
        df["make_freq"] = df["make"].map(self.make_freq_).fillna(0.0)
        df["base_model_freq"] = df["base_model"].map(self.base_model_freq_).fillna(0.0)
        df["make_type_freq"] = (df["make"] + "_" + df["type"]).map(
            self.make_type_freq_).fillna(0.0)
        bin_df = self.binary_encoder_.transform(df[["model"]]).fillna(0.0)
        bin_df.index = df.index
        df = pd.concat([df.drop(columns=["model"]), bin_df], axis=1)
        if not self.use_agg_features:
            df = df.drop(columns=[c for c in self.AGG_COLS if c in df.columns])
        return df

    def _prepare_text(self, X, apply_make_map=True):
        df = basic_clean(X)
        df["is_mileage_placeholder"] = (df["mileage"] <= 10).astype(int)
        # cross-column imputation from `model`
        df = _fill_by_rules(df, "drive", _DRIVE_RULES)
        df = _fill_by_rules(df, "fuel", _FUEL_RULES)
        df = _fill_by_rules(df, "transmission", _TRANS_RULES)
        df = _fill_by_rules(df, "type", _TYPE_RULES)
        m_el = (df["cylinders"] == "unknown") & (df["fuel"] == "electric")
        df.loc[m_el, "cylinders"] = "0 cylinders"
        df = _fill_by_rules(df, "cylinders", _CYL_RULES)
        if apply_make_map:
            m = df["make"] == "unknown"
            first_tok = df.loc[m, "model"].str.split().str[0]
            filled = first_tok.map(getattr(self, "make_from_model_", {}))
            df.loc[m, "make"] = filled.where(filled.notna(), "unknown")
        # deterministic feature engineering
        df["car_age"] = (REF_YEAR - df["year"]).clip(lower=0)
        df["is_classic"] = (df["year"] < 1995).astype(int)
        df["miles_per_year"] = df["mileage"] / df["car_age"].clip(lower=1)
        df["log_mileage"] = np.log1p(df["mileage"])
        df["cylinders_num"] = df["cylinders"].str.extract(r"(\d+)")[0].astype("float64")
        df["condition_score"] = df["condition"].map(CONDITION_SCORE).fillna(-1).astype(int)
        df["is_clean_title"] = (df["title_status"] == "clean").astype(int)
        df["model_word_count"] = df["model"].str.split().str.len()
        df["is_luxury_or_sport_trim"] = df["model"].str.contains(LUXURY_SPORT_RE, na=False).astype(int)
        df["is_work_or_heavy_duty"] = df["model"].str.contains(WORK_HEAVY_RE, na=False).astype(int)
        df["is_turbo_or_performance"] = df["model"].str.contains(TURBO_PERF_RE, na=False).astype(int)
        df["base_model"] = df["model"].map(_one_base_model)
        return df


# ---------------------------------------------------------------- target-aware wrapper
def filter_outlier_mask(X: pd.DataFrame, y) -> np.ndarray:
    """True = keep. Gross outliers: year == 1900, or year >= 2024 with price < $2,000."""
    year = X["year"].to_numpy()
    y = np.asarray(y)
    bad = (year == 1900) | ((year >= 2024) & (y < 2000))
    return ~bad


class TargetAwareRegressor(BaseEstimator, RegressorMixin):
    """Wraps any regressor with two training-only options:

    log_target   : fit on log1p(y), predict with expm1 (metrics stay in dollars).
    drop_outliers: drop gross outliers from the TRAINING rows only, so CV
                   validation folds always keep the full distribution.
    """

    def __init__(self, base=None, log_target=False, drop_outliers=False):
        self.base = base
        self.log_target = log_target
        self.drop_outliers = drop_outliers

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        if self.drop_outliers:
            keep = filter_outlier_mask(X, y_arr)
            X = X.loc[keep] if hasattr(X, "loc") else X[keep]
            y_arr = y_arr[keep]
        y_fit = np.log1p(y_arr) if self.log_target else y_arr
        if self.log_target:
            # bound predictions to the training log-range so expm1 cannot explode
            self.log_min_, self.log_max_ = float(y_fit.min()), float(y_fit.max())
        self.base_ = clone(self.base)
        self.base_.fit(X, y_fit)
        self.n_train_used_ = len(y_arr)
        return self

    def predict(self, X):
        pred = self.base_.predict(X)
        if not self.log_target:
            return pred
        pred = np.clip(pred, self.log_min_, self.log_max_)
        return np.expm1(pred)


# ---------------------------------------------------------------- ridge baseline
def run_ridge_baseline(X_train, y_train, X_test, y_test):
    """Leakage-safe Ridge baseline on log1p(price); metrics in real dollars."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scale", RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                         ("ohe", make_ohe(min_frequency=20,
                                          handle_unknown="infrequent_if_exist"))])
    ct = ColumnTransformer(
        [("num", num_pipe, lambda d: [c for c in d.columns if c not in CAT_COLS]),
         ("cat", cat_pipe, CAT_COLS)], remainder="drop")
    pipe = Pipeline([("prep", CarPreprocessor(use_agg_features=True)),
                     ("feats", ct),
                     ("model", Ridge(alpha=1.0, max_iter=5000))])
    est = TargetAwareRegressor(base=pipe, log_target=True)
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    m = {"baseline_mae": float(mean_absolute_error(y_test, pred)),
         "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
         "baseline_r2": float(r2_score(y_test, pred))}
    print("Ridge baseline (test, $ scale):")
    for k, v in m.items():
        print(f"  {k}: {v:,.3f}")
    return est, m


def main():
    path = locate_dataset()
    df = basic_clean(pd.read_csv(path))
    print("dataset:", df.shape)
    X_train, X_test, y_train, y_test = split_first(df)
    save_split(X_train, X_test, y_train, y_test)
    run_ridge_baseline(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
