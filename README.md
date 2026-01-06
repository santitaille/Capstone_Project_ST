# EA FC 26 Player Price Prediction

**Course:** Data Science and Advanced Programming 2025  
**Student:** Santiago Tailleferd
**Student ID:** 20557377

This repository contains the full implementation of a data science and machine learning pipeline designed to **predict**, **explain** and **evaluate player card prices** in the *EA FC 26 Ultimate Team* transfer market. The project combines **web scraping**, **extensive feature engineering**, **two baseline models**, **four machine learning models** and **three interpretability analyses**

## Research Questions
This project addresses the following research questions:
1. **What card features influence card prices the most?**
2. **Which machine learning model can most accurately predict card prices?**
3. **Can prediction models identify market inefficiencies?**

---

## Project Overview

- **Dataset:** ~801 player cards scraped from FUTBIN
- **Validation scheme:** Temporal validation (Trained in Week 1 → Tested in Week 2)
- **Models implemented:** Two median-based baselines, linear regression, Random Forest, XGBoost, neural network (MLP)
- **Evaluation metrics:** R², MAE (credits) and RMSE (credits)
- **Interpretability methods:** OLS coefficients, XGBoost feature importance, SHAP values
- **Market inefficiency analysis:** Comparison of predicted “fair values” vs observed market prices

---

## Repository Structure

```text
Capstone_Project_ST/
│
├── data/
│   ├── raw/                 # Raw scraped data (if scraping is run)
│   └── processed/           # Cleaned & engineered datasets
│
├── results/
│   ├── figures/             # Plots (EDA, SHAP, performance, etc.)
│   └── tables/              # Model metrics and outputs
│
├── src/
│   ├── scrapers/            # FUTBIN scraping scripts
│   ├── preprocessing/       # Feature engineering & encoding
│   ├── models/              # Model training implementations
│   └── analysis/            # Evaluation & interpretability
│
├── tests/                   # 5 Unit tests
│
├── main.py                  # Main entry point (reproduces all results)
├── requirements.txt         # Python dependencies
├── requirements-pinned.txt  # Fully pinned environment
├── PROPOSAL.md
├── Final_Report.pdf
└── README.md


**Best performing model:** XGBoost (R² ≈ 0.956)
