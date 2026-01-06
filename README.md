# EA FC 26 Player Price Prediction

**Course:** Data Science and Advanced Programming 
**Student Name:** Santiago Tailleferd
**Student ID:** 20557377

This repository contains the full implementation of a data science and machine learning pipeline designed to **predict**, **explain** and **evaluate player card prices** in the *EA FC 26 Ultimate Team* transfer market. The project combines **web scraping**, **extensive feature engineering**, **two baseline models**, **four machine learning models** and **three interpretability analyses**

## Research Questions
This project addresses the following research questions:
**1. What card features influence card prices the most?**
**2. Which machine learning model can most accurately predict card prices?**
**3. Can prediction models identify market inefficiencies?**

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
├── data/                           # Raw, intermediary and completely merged data
│
├── results/
│   ├── figures/                    # 16 generated figures
│   ├── predictions/                # 4 prediction files (one per model, Week 2)
│   └── tables/                     # 5 result tables (metrics, coefficients, importances)
│
├── src/
│   ├── scrapers/                   # Data collection from FUTBIN
│   ├── preprocessing/              # Data merging & feature engineering
│   ├── models/                     # Models implementation
│   └── analysis/                   # Evaluation & interpretability
│
├── tests/                          # 5 unit tests
│
├── main.py                         # Single entry point (reproduces all results)
├── project_report.pdf              # Final project report
├── PROPOSAL.md                     # Approved project proposal
├── AI_USAGE.md                     # AI usage declaration
├── README.md                       # Setup, usage instructions and overview
├── requirements.txt                # Main Python dependencies (unversioned)
└── requirements-pinned.txt         # Fully pinned environment
```
## Setup, Environment and Reproducibility
- Python version: 3.10
- Developed and tested using VSCode + Nuvulous
- Fixed random seed (42)

Note: Scraping scripts are included for completeness but do not need to be executed to reproduce the results.
Running main.py reproduces all figures and tables used in the report.

### 1.Clone the repository
```text
git clone https://github.com/santitaille/Capstone_Project_ST.git
cd Capstone_Project_ST
```

### 2.Create a Conda environment with Python 3.10
```text
conda create -n eafc_dsap python=3.10 -y
conda activate eafc_dsap
```

### 3.Install dependencies

#### 3.1. Standard installation (recommended)
```text
pip install -r requirements.txt
```

#### 3.2. Fully pinned environment (optional):
```text
pip install -r requirements-pinned.txt
```
### 4. Run the full pipeline
```text
python main.py
```

Note: TensorFlow is very heavy when downloaded via environment.yml, so creating a Conda 3.10 environment and installing requirements is faster.

## Results and Key Finding
**Table 1: Model Performance Metrics (ranked by R²)**
| Model                    | R²    | RMSE    | MAE    | R² improvement over benchmark |
|--------------------------|-------|---------|--------|-------------------------------|
| XGBoost                  | 0.956 | 137,993 | 42,997 | 32.3%                         |
| Random Forest            | 0.860 | 245,831 | 63,961 | 22.7%                         |
| Neural Network (MLP)     | 0.669 | 377,422 | 92,889 | 3.6%                          |
| Baseline 2 (Benchmark)   | 0.633 | 397,507 | 129,361| 0.0%                          |
| Linear Regression        | 0.631 | 398,273 | 116,050| (0.2%)                        |
| Baseline 1               | 0.608 | 410,874 | 142,364| (2.5%)                        |

XGBoost clearly outperformed all models (R² = 0.956), followed by Random Forest, neural netowrk and linear regression. It improved the benchmark by 32.3 percentage points, showing that the relationship between card features and card prices is highly nonlinear. Overall rating, card category and the number of Playstyles+ were detected as the most impactful features. Finally, taking XGBoost's price predictions are fair value estimates and a ±20% conservative threshold 35.1% of player cards were classified as mispriced.
