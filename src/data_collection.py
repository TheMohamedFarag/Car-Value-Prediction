import os
import requests
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw")
KAGGLE_CSV_PATH = os.path.join(RAW_DATA_PATH, "kaggle_cars_raw.csv")

os.makedirs(RAW_DATA_PATH, exist_ok=True)

# ---------------------------------------------------------
# 2. NHTSA VPIC API Fetcher (Composite Key Integration)
# ---------------------------------------------------------
def fetch_nhtsa_specs(make: str, model: str, year: int) -> dict:
    """
    Fetch technical specs from NHTSA VPIC API using composite key (Make + Model + Year).
    """
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformakeyear/make/{make}/modelyear/{year}?format=json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json().get("Results", [])
            for item in results:
                if str(item.get("Model_Name", "")).lower() == str(model).lower():
                    return {
                        "make": make,
                        "model": model,
                        "year": year,
                        "vpic_matched": True,
                        "vehicle_type": item.get("VehicleTypeName", "N/A"),
                    }
    except Exception as e:
        print(f"Error querying VPIC API for {make} {model} {year}: {e}")
    
    return {"make": make, "model": model, "year": year, "vpic_matched": False}

# ---------------------------------------------------------
# 3. Main Data Collection & Validation Pipeline
# ---------------------------------------------------------
def load_and_inspect_raw_data():
    print("=== Phase 2: Loading & Validating Raw Kaggle Dataset ===")
    
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"[-] File not found at {KAGGLE_CSV_PATH}. Please place 'kaggle_cars_raw.csv' in 'data/raw/'.")
        return None

    # Load Kaggle raw dataset
    df = pd.read_csv(KAGGLE_CSV_PATH)
    print(f"[+] Kaggle Raw Dataset Loaded Successfully!")
    print(f"    - Total Rows: {len(df)}")
    print(f"    - Total Columns: {len(df.columns)}")
    print(f"    - Columns: {list(df.columns[:8])}... (and more)")
    
    # Verify Composite Key Columns (manufacturer/make, model, year)
    required_keys = ['manufacturer', 'model', 'year', 'price']
    found_keys = [col for col in required_keys if col in df.columns]
    print(f"[+] Found Target & Composite Key Columns: {found_keys}")

    # Test NHTSA VPIC API on first valid row
    sample_row = df.dropna(subset=['manufacturer', 'model', 'year']).iloc[0]
    make = sample_row['manufacturer']
    model = sample_row['model']
    year = int(sample_row['year'])
    
    print(f"\n[+] Testing VPIC API Enrichment on Sample Listing: {make} {model} ({year})...")
    api_res = fetch_nhtsa_specs(make, model, year)
    print(f"    - API Response: {api_res}")
    
    return df

if __name__ == "__main__":
    load_and_inspect_raw_data()