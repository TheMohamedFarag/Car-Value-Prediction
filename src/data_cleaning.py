from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "integrated_cars_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_cars.csv"

df = pd.read_csv(INPUT_PATH, low_memory=False)

if "size" in df.columns:
    df = df.drop(columns=["size"]).copy()

df = df.drop_duplicates().copy()

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

df = df.dropna(subset=["price", "year", "mileage"]).copy()

df = df[df["year"].between(1886, 2026)].copy()
df = df[df["mileage"].between(1, 500_000)].copy()
df = df[df["price"].between(500, 500_000)].copy()

df["price"] = df["price"].astype("int64")
df["year"] = df["year"].astype("int64")
df["mileage"] = df["mileage"].astype("int64")

categorical_cols = [
    "make",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "data_source",
]

for column in categorical_cols:
    if column in df.columns:
        df[column] = (
            df[column]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        )

df = df.drop_duplicates().copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Cleaning completed successfully.")
print(f"Saved file: {OUTPUT_PATH}")
print(f"Final dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Price range: {df['price'].min():,.0f} to {df['price'].max():,.0f}")