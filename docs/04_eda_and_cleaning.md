# Phase 4: Exploratory Data Analysis and Data Cleaning

## Input
- **Input file:** `data/interim/integrated_cars_raw.csv`
- **Initial dataset shape:** `(295,454, 14)`

## Cleaning Steps
- Removed exact duplicate records, including duplicates revealed after standardizing text values.
- Dropped the completely empty `size` column (retaining 13 clean columns).
- Converted `price`, `year`, and `mileage` to numeric integer types (`int64`).
- Removed rows with missing or invalid `year` values outside `1886`–`2026`.
- Removed rows with missing or invalid `mileage` values outside `1`–`500,000` miles.
- Replaced missing categorical values across all text columns with `unknown` and standardized formatting (`lowercase`).
- Removed invalid target prices outside the accepted range (`$500` to `$500,000`).

## Final Validation
- Zero missing values (`0` NaNs).
- Zero duplicate rows.
- All prices within `$500` to `$500,000`.
- Valid `year` (`1886`–`2026`) and `mileage` (`1`–`500,000`) values only.
- **Final dataset shape:** `(182,799, 13)` (all remaining rows originate from the `kaggle` data source after filtering out sample `web_scraped` records missing `mileage`).

## EDA Findings
- **Price Distribution:** Strongly right-skewed (`median` ≈ `$11,500`), indicating that a log transformation (`log1p`) will benefit regression modeling.
- **Vehicle Year:** Most vehicles were manufactured between 2008 and 2019, with newer model years showing a clear upward price trend.
- **Mileage Impact:** Strong negative relationship with price; higher mileage listings cluster tightly in lower price brackets.
- **Condition & Make:** `new` and `like new` vehicles command noticeably higher median prices than `fair` or `salvage` listings. Ford, Chevrolet, and Toyota are the most frequent makes, while Ram and GMC achieve the highest median prices among the top 10 common makes.

## Outputs
- **Notebook:** `notebooks/04_eda_and_cleaning.ipynb`
- **Cleaning script:** `src/data_cleaning.py`
- **Cleaned dataset:** `data/processed/cleaned_cars.csv`