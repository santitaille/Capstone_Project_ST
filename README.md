# Capstone_Project_ST

# EA FC 26 Player Price Prediction  
**Advanced Programming & Data Science – Final Project (DSAP 2025)**

This repository contains the full implementation of a data science and machine learning pipeline designed to **predict, explain, and evaluate player card prices** in the *EA FC 26 Ultimate Team* transfer market.

The project combines **web scraping**, **feature engineering**, **predictive modeling**, **model interpretability**, and **market efficiency analysis** to study one of the largest virtual peer-to-peer economies in gaming.

---

## Research Questions

This project aims to answer the following questions:

1. **What card features influence card prices the most?**  
2. **Which machine learning model can most accurately predict card prices?**  
3. **Can prediction models identify market inefficiencies?**

---

## Project Overview

- **Dataset:** ~800 tradeable, non-goalkeeper player cards (rating ≥ 83)  
- **Data source:** FUTBIN (scraped)  
- **Validation scheme:** Temporal validation (Week 1 → Week 2)  
- **Models implemented:**
  - Median-based baselines (2)
  - Linear Regression
  - Random Forest
  - XGBoost
  - Neural Network (MLP – TensorFlow)
- **Evaluation metrics:**
  - R²
  - MAE (credits)
  - RMSE (credits)
- **Interpretability methods:**
  - OLS coefficients
  - XGBoost feature importance
  - SHAP values
- **Market efficiency analysis:** Comparison of predicted “fair values” vs observed market prices

**Best performing model:** XGBoost (R² ≈ 0.956)

---


Capstone_Project_ST/
│
├── data/
│   ├── raw/                # Raw scraped data (if scraping is run)
│   └── processed/          # Cleaned & engineered datasets
│
├── results/
│   ├── figures/            # Plots (EDA, SHAP, performance, etc.)
│   ├── tables/             # Model metrics and outputs
│
├── src/
│   ├── scrapers/           # FUTBIN scraping scripts
│   ├── preprocessing/     # Feature engineering & encoding
│   ├── models/             # Model training implementations
│   ├── analysis/           # Evaluation & interpretability
│
├── tests/                  # Unit tests (data, features, models)
│
├── main.py                 # Main entry point (reproduces all results)
├── requirements.txt        # Python dependencies (un-pinned)
├── requirements-pinned.txt # Fully pinned environment (optional)
├── PROPOSAL.md
├── Final_Report.pdf
└── README.md
