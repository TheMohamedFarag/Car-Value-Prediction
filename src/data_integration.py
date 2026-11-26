import os
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw")
INTERIM_DATA_PATH = os.path.join("data", "interim")

KAGGLE_CSV_PATH = os.path.join(RAW_DATA_PATH, "kaggle_cars_raw.csv")
SCRAPED_CSV_PATH = os.path.join(RAW_DATA_PATH, "scraped_cars_raw.csv")
INTEGRATED_CSV_PATH = os.path.join(INTERIM_DATA_PATH, "integrated_cars_raw.csv")

os.makedirs(INTERIM_DATA_PATH, exist_ok=True)


def integrate_and_standardize_sources():
    print("=== Phase 3: Data Integration & Schema Standardization ===")

    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"[-] Missing file: {KAGGLE_CSV_PATH}")
        return None
    kaggle_df = pd.read_csv(KAGGLE_CSV_PATH, low_memory=False)
    kaggle_df["data_source"] = "kaggle"

    kaggle_df = kaggle_df.rename(columns={
        "manufacturer": "make",
        "odometer": "mileage",
    })

    if os.path.exists(SCRAPED_CSV_PATH):
        scraped_df = pd.read_csv(SCRAPED_CSV_PATH, low_memory=False)
        scraped_df["data_source"] = "web_scraped"
        scraped_df = scraped_df.rename(columns={
            "manufacturer": "make",
            "odometer": "mileage",
        })
    else:
        scraped_df = pd.DataFrame()

    target_schema = [
        "make",
        "model",
        "year",
        "mileage",
        "condition",
        "cylinders",
        "fuel",
        "title_status",
        "transmission",
        "drive",
        "size",
        "type",
        "price",
        "data_source",
    ]

    kaggle_sub = kaggle_df.reindex(columns=target_schema)

    if not scraped_df.empty:
        scraped_sub = scraped_df.reindex(columns=target_schema)
        integrated_df = pd.concat([kaggle_sub, scraped_sub], ignore_index=True)
    else:
        integrated_df = kaggle_sub

    integrated_df.to_csv(INTEGRATED_CSV_PATH, index=False)
    print("[+] Multi-source Integration Complete!")
    print(f"    - Total Combined Rows: {len(integrated_df)}")
    print(f"    - Total Columns: {integrated_df.shape[1]}")
    print(f"    - Saved Integrated Dataset to: '{INTEGRATED_CSV_PATH}'")

    return integrated_df


if __name__ == "__main__":
    integrate_and_standardize_sources()