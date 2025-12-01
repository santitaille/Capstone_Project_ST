"""
EA FC 26 Player Price Prediction - Main Pipeline
Santiago Tailleferd - Student #20557377

Runs the complete analysis pipeline in order:
1. EDA Analysis
2. OLS Coefficient Analysis  
3. OLS Visualizations
4. XGBoost Feature Importance
5. SHAP Analysis
6. Baseline Models
7. Linear Regression
8. Random Forest
9. XGBoost Model
10. Neural Network
11. Market Inefficiencies
12. Model Comparison
"""

import sys
from pathlib import Path

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def print_header(step, total, title):
    """Print formatted step header"""
    print("\n" + "=" * 80)
    print(f"[{step}/{total}] {title}")
    print("=" * 80)


def main():
    """Run complete analysis pipeline"""
    print("=" * 80)
    print("EA FC 26 PLAYER PRICE PREDICTION - FULL PIPELINE")
    print("Student: Santiago Tailleferd (#20557377)")
    print("=" * 80)
    
    total_steps = 12
    
    try:
        # 1. EDA Analysis
        print_header(1, total_steps, "EXPLORATORY DATA ANALYSIS")
        from analysis import eda_analysis
        eda_analysis.main()
        
        # 2. OLS Coefficient Analysis
        print_header(2, total_steps, "OLS COEFFICIENT ANALYSIS")
        from analysis import ols_coefficient_analysis
        ols_coefficient_analysis.main()
        
        # 3. OLS Visualizations
        print_header(3, total_steps, "OLS VISUALIZATIONS")
        from analysis import ols_visualizations
        ols_visualizations.main()
        
        # 4. XGBoost Feature Importance
        print_header(4, total_steps, "XGBOOST FEATURE IMPORTANCE")
        from analysis import xgboost_feature_importance
        xgboost_feature_importance.main()
        
        # 5. SHAP Analysis
        print_header(5, total_steps, "SHAP ANALYSIS")
        from analysis import shap_analysis
        shap_analysis.main()
        
        # 6. Baseline Models
        print_header(6, total_steps, "BASELINE MODELS")
        from models import baseline_models
        baseline_models.main()


        # 7. Linear Regression
        print_header(7, total_steps, "LINEAR REGRESSION MODEL")
        from models import linear_regression
        linear_regression.main()


        # 8. Random Forest
        print_header(8, total_steps, "RANDOM FOREST MODEL")
        from models import random_forest
        random_forest.main()


        # 9. XGBoost Model
        print_header(9, total_steps, "XGBOOST MODEL")
        from models import xgboost_model
        xgboost_model.main()


        # 10. Neural Network
        print_header(10, total_steps, "NEURAL NETWORK MODEL")
        from models import neural_network
        neural_network.main()


        # 11. Market Inefficiencies
        print_header(11, total_steps, "MARKET INEFFICIENCIES DETECTION")
        from analysis import market_inefficiencies
        market_inefficiencies.main()


        # 12. Model Comparison
        print_header(12, total_steps, "MODEL COMPARISON")
        from analysis import model_comparison
        model_comparison.main()


        # Final Summary
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)
        print("\nAll results saved to:")
        print(f"  - Figures: {PROJECT_ROOT / 'results' / 'figures'}")
        print(f"  - Tables: {PROJECT_ROOT / 'results' / 'tables'}")
        print(f"  - Predictions: {PROJECT_ROOT / 'results' / 'predictions'}")
        print("\nGenerated outputs:")
        print("  - 17 visualizations (PNG)")
        print("  - 4 model predictions (CSV)")
        print("  - 5 analysis tables (CSV)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed at current step")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()