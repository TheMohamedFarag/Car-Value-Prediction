# Phase 1: Business Understanding & Project Objectives

## 1. Problem Statement
The used car market experiences high price volatility and asymmetry of information between buyers and sellers. Without a standardized valuation benchmark, individual buyers risk overpaying, while sellers and dealerships struggle to price inventory competitively and efficiently.

## 2. Project Goals & Success Criteria
- **Machine Learning Metric:** Build a regression model that achieves an R2 score >= 0.85 and minimizes Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on unseen test data.
- **Business Metric (Deal Intelligence):** Implement a valuation engine that compares actual listed prices against fair-value model predictions (\Delta = Actual Price - Predicted Price) to label listings as **Good Deal**, **Fair Price**, or **Overpriced**.

## 3. Financial & ROI Framework
- **Buyer Cost Savings:** Estimated average savings of **$500 - $1,500** per transaction for buyers utilizing "Good Deal" indicators to avoid paying above fair market values.
- **Evaluation Time Efficiency:** Reduces average listing evaluation time from several days of manual cross-referencing down to **less than 3 seconds** via an automated Streamlit dashboard.

## 4. Target & Currency
- **Target Variable:** `price` (Continuous variable representing vehicle fair market value).
- **Currency:** USD ($).
- **Scope:** Global / US Used Car Market.

## 5. Data Strategy & Composite Key Integration
Data will be fetched and combined from three sources:
1. **Kaggle (CSV):** Core historical dataset containing base listing attributes.
2. **NHTSA VPIC API:** Feature enrichment source providing technical engine/vehicle specifications. Joined via composite key: `(Make + Model + Year)`.
3. **Web Scraping (Craigslist Live Listings):** Real-time market listings stacked vertically via concatenation.
