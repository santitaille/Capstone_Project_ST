# EA FC 26 Player Card Price Prediction  

**Course:** Data Science and Advanced Programming  
**Student Name:** Santiago Tailleferd  
**Student ID:** 20557377  

This repository contains the full implementation of a data science and machine learning pipeline designed to **predict**, **explain** and **evaluate player card prices** in the EA FC 26 Ultimate Team transfer market. The project combines **web scraping**, **extensive feature engineering**, **two baseline models**, **four machine learning models** and **three interpretability analyses**

## Research Questions
This project addresses the following research questions:  

**1. What card features influence card prices the most?**  
**2. Which machine learning model can most accurately predict card prices?**  
**3. Can prediction models identify market inefficiencies?**  

---

## Project Overview

- **Dataset:** 801 player cards scraped from FUTBIN
- **Validation scheme:** Temporal validation (Trained in Week 1 → Tested in Week 2)
- **Models implemented:** Two median-based baselines, linear regression, Random Forest, XGBoost and neural network (MLP)
- **Evaluation metrics:** R², MAE (credits) and RMSE (credits)
- **Interpretability methods:** OLS coefficients, XGBoost feature importance, SHAP values
- **Market inefficiency analysis:** Comparison of predicted “fair values” vs observed market prices

---

## Repository Structure

```text
Capstone_Project_ST/
│
├── data/                           # Raw, intermediate and fully merged datasets
│
├── results/
│   ├── figures/                    # 16 figures
│   ├── predictions/                # 4 prediction files (one per model)
│   └── tables/                     # 5 tables (evaluation metrics and interpretability results)
│
├── src/
│   ├── scrapers/                   # Data collection from FUTBIN
│   ├── preprocessing/              # Data merging and feature engineering
│   ├── models/                     # Model implementations
│   └── analysis/                   # Evaluation and interpretability
│
├── tests/                          # 5 automated unit tests
│
├── main.py                         # Single entry point reproducing all results
├── ST_project_report.pdf           # Final project report
├── PROPOSAL.md                     # Approved project proposal
├── AI_USAGE.md                     # AI usage declaration
├── README.md                       # Setup instructions and project overview
├── requirements.txt                # Main Python dependencies (unpinned)
└── requirements-pinned.txt         # Fully pinned environment
```

---

## Setup, Environment and Reproducibility
- Python version: 3.10
- Developed and tested using VS Code and Nuvolos
- Fixed random seed (42)

Note: Data scrapers are included as they are part of the project but do not need to be executed to reproduce the results. Otherwise, temporal validation would not be possible. Executing main.py generates all figures, tables and predictions presented in the report.

### 1. Clone the repository
```text
git clone https://github.com/santitaille/Capstone_Project_ST.git
cd Capstone_Project_ST
```

### 2. Create a Conda environment with Python 3.10
```text
conda create -n eafc_dsap python=3.10 -y
conda activate eafc_dsap
```

### 3. Install dependencies

#### 3.1. Unpinned environment
```text
pip install -r requirements.txt
```

#### 3.2. Fully pinned environment
```text
pip install -r requirements-pinned.txt
```
### 4. Run the full pipeline
```text
python main.py
```

**Note:** TensorFlow is very heavy when downloaded via environment.yml, so creating a Conda 3.10 environment and installing requirements is faster.

**Note:** The project was tested using both `requirements.txt` and `requirements-pinned.txt`.

**Note:** When testing the code on other machines in VS Code, sometimes all figures appeared as **modified (M)**. However, the figures are visually identical, contain the same results and in any case results are impacted.

---

## Results, Key Findings and Limitations
Table 1 – Model Performance Metrics (ranked by R²)
| Model                    | R²    | RMSE    | MAE    | R² improvement over benchmark |
|--------------------------|-------|---------|--------|-------------------------------|
| XGBoost                  | 0.956 | 137,993 | 42,997 | 32.3%                         |
| Random Forest            | 0.860 | 245,831 | 63,961 | 22.7%                         |
| Neural Network (MLP)     | 0.669 | 377,422 | 92,889 | 3.6%                          |
| Baseline 2 (Benchmark)   | 0.633 | 397,507 | 129,361| 0.0%                          |
| Linear Regression        | 0.631 | 398,273 | 116,050| (0.2%)                        |
| Baseline 1               | 0.608 | 410,874 | 142,364| (2.5%)                        |

XGBoost clearly outperformed all models (R² = 0.956), followed by Random Forest, neural network and linear regression. It improved the benchmark by **32.3 percentage points**, indicating that the relationship between card features and prices is **highly nonlinear**. Among all interpretability analyses, **overall rating**, **card category** and the **number of Playstyles+** were detected as the most influential drivers of price. Taking XGBoost predictions as **fair value estimates** and using a conservative **±20% threshold**, **35.1% of player cards** were classified as mispriced, suggesting the presence of market inefficiencies.

This project focuses on **two discrete weekly price snapshots** rather than continuous price dynamics. Additionally, factors such as SBC and evolution requirements, sentiment-driven demand and pack availability are not modeled and may affect predicted prices.

---

## Video Presentation URL
https://youtu.be/hzh6kNWK304
