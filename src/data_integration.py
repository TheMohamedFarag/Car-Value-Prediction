import os
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw")
INTERIM_DATA_PATH = os.path.join("data", "interim")

KAGGLE_CSV_PATH = os.path.join(RAW_DATA_PATH, "kaggle_cars_raw.csv")
SCRAPED_CSV_PATH = os.path.join(RAW_DATA_PATH, "scraped_cars_raw.csv")
INTEGRATED_CSV_PATH = os.path.join(INTERIM_DATA_PATH, "integrated_cars_raw.csv")

os.makedirs(INTERIM_DATA_PATH, exist_ok=True)

# ---------------------------------------------------------
# 2. Integration & Schema Standardization Pipeline
# ---------------------------------------------------------
def integrate_and_standardize_sources():
    print("=== Phase 3: Data Integration & Schema Standardization ===")
    
    # Load Kaggle Data
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"[-] Missing file: {KAGGLE_CSV_PATH}")
        return None
    kaggle_df = pd.read_csv(KAGGLE_CSV_PATH, low_memory=False)
    kaggle_df['data_source'] = 'kaggle'
    
    # Standardize Kaggle Column Names
    kaggle_df = kaggle_df.rename(columns={
        'manufacturer': 'make',
        'odometer': 'mileage'
    })
    
    # Load Web Scraped Data
    if os.path.exists(SCRAPED_CSV_PATH):
        scraped_df = pd.read_csv(SCRAPED_CSV_PATH)
        scraped_df['data_source'] = 'web_scraped'
        scraped_df = scraped_df.rename(columns={'manufacturer': 'make'})
    else:
        scraped_df = pd.DataFrame()

    # Combine Raw Sources (Vertical Concatenation)
    common_cols = ['make', 'model', 'year', 'price', 'data_source']
    kaggle_sub = kaggle_df[[col for col in common_cols if col in kaggle_df.columns]]
    
    if not scraped_df.empty:
        scraped_sub = scraped_df[[col for col in common_cols if col in scraped_df.columns]]
        integrated_df = pd.concat([kaggle_sub, scraped_sub], ignore_index=True)
    else:
        integrated_df = kaggle_sub

    # Save Integrated Dataset to data/interim/
    integrated_df.to_csv(INTEGRATED_CSV_PATH, index=False)
    print(f"[+] Multi-source Integration Complete!")
    print(f"    - Total Combined Rows: {len(integrated_df)}")
    print(f"    - Saved Integrated Dataset to: '{INTEGRATED_CSV_PATH}'")
    
    return integrated_df

if __name__ == "__main__":
    integrate_and_standardize_sources()