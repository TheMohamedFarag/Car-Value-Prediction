import os
import requests
import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw")
KAGGLE_CSV_PATH = os.path.join(RAW_DATA_PATH, "kaggle_cars_raw.csv")
VPIC_API_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/"

# Ensure output raw directory exists
os.makedirs(RAW_DATA_PATH, exist_ok=True)


# ---------------------------------------------------------
# 2. NHTSA VPIC API Fetcher (Composite Key Integration)
# ---------------------------------------------------------
def fetch_nhtsa_specs(make: str, model: str, year: int) -> dict:
    """
    Fetch technical specs (Engine Cylinders, Displacement, Vehicle Type)
    from NHTSA VPIC API using composite key (Make + Model + Year).
    """
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformakeyear/make/{make}/modelyear/{year}?format=json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json().get("Results", [])
            for item in results:
                if item.get("Model_Name", "").lower() == model.lower():
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
# 3. Main Collection Pipeline Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== Phase 2: Data Collection Pipeline Started ===")
    
    # Example test call to VPIC API
    print("Testing NHTSA VPIC API connection...")
    sample_spec = fetch_nhtsa_specs(make="Toyota", model="Camry", year=2020)
    print("VPIC Sample Result:", sample_spec)
    
    print("\nData collection module ready. Add Kaggle dataset download step to raw/ directory.")