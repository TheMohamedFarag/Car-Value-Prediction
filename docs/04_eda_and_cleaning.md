# Phase 4: Exploratory Data Analysis and Data Cleaning

## Input
- Input file: `data/interim/integrated_cars_raw.csv`
- Initial dataset shape: `(295,454, 5)`

## Cleaning Steps
- Removed exact duplicate records.
- Standardized `make` and `model` values by removing spaces and converting text to lowercase.
- Replaced missing `make` and `model` values with `unknown`.
- Removed rows with missing `year`.
- Converted `price` to numeric.
- Removed invalid prices outside the range `$500` to `$500,000`.

## Final Validation
- No missing values.
- No duplicate rows.
- Valid price range: `$500` to `$500,000`.
- Final dataset shape: `(171,923, 5)`.

## EDA Findings
- The price distribution is right-skewed.
- The median price is approximately `$11,500`.
- Most vehicles were manufactured between 2008 and 2019.
- Newer vehicles generally have higher prices.
- Ford, Chevrolet, and Toyota are the most common known makes.
- Median car prices vary by make; Ram and GMC are among the highest-priced common makes.

## Outputs
- Notebook: `notebooks/04_eda_and_cleaning.ipynb`
- Cleaning script: `src/data_cleaning.py`
- Cleaned dataset: `data/processed/cleaned_cars.csv`
