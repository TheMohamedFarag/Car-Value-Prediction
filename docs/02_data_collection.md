# Phase 2 Documentation: Multi-Source Data Collection Strategy

## Executive Summary
This document details the ingestion pipelines designed and implemented for the **Car Price Prediction System**. To ensure maximum accuracy, standard coverage, and current market alignment, data is collected from three distinct data tiers:
1. Historical used car market listings.
2. Official manufacturer specs (Federal API).
3. Live market listings (Web Scraping).

---

## 1. Data Sources & Architecture

### Source A: Kaggle US Used Cars Dataset
* **Type:** Static Bulk CSV Ingestion (`data/raw/kaggle_cars_raw.csv`).
* **Volume:** ~295,451 records.
* **Role:** Serves as the primary volumetric base layer. Provides historical price behavior, mileage (odometer), vehicle condition, location, and regional distributions.

### Source B: NHTSA VPIC API (National Highway Traffic Safety Administration)
* **Type:** REST API / JSON Lookup Service.
* **Endpoint:** `https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformakeyear/`
* **Composite Key:** `(Make, Model, Year)`
* **Role:** Feature Enrichment Tier. Verifies vehicle legitimacy and attaches official manufacturing specs (Body Type, Standardized Vehicle Category, Displacement) to resolve missing structural information.

### Source C: Web Scraping Pipeline
* **Type:** Dynamic HTML Scraper / Micro-Service (`src/data_collection.py`).
* **Role:** Recency Tier. Captures real-time market price adjustments to protect the Machine Learning model from economic inflation and outdated pricing patterns.

---

## 2. Ingestion Execution Script
* **Script Location:** `src/data_collection.py`
* **Validation Notebook:** `notebooks/02_data_collection.ipynb`