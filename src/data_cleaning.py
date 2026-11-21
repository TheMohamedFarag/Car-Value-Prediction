from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "integrated_cars_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_cars.csv"

# Load integrated dataset
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Convert required numeric columns
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Remove rows with missing target or year
df = df.dropna(subset=["price", "year"]).copy()

# Keep realistic car prices and valid model years
df = df[df["price"].between(500, 500_000)].copy()
df = df[df["year"].between(1886, 2030)].copy()

# Convert year to integer
df["year"] = df["year"].astype(int)

# Standardize categorical text columns
for column in ["make", "model"]:
    df[column] = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "unknown")
    )

# Remove duplicates revealed after text standardization
df = df.drop_duplicates().copy()

# Save cleaned dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# Final validation summary
print("Cleaning completed successfully.")
print(f"Saved file: {OUTPUT_PATH}")
print(f"Final dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Price range: {df['price'].min():,.0f} to {df['price'].max():,.0f}")