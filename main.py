"""
EA FC 26 Player Price Prediction - DATA SCIENCE AND ADVANCED PROGRAMMING
Santiago Tailleferd (#20557377)

Runs the complete analysis pipeline in 12 steps:

PART I: DATA PREPARATION & EXPLORATION
    1. EDA Analysis
    2. Feature Engineering

PART II: BASELINE & STATISTICAL ANALYSIS
    3. OLS Coefficient Analysis & Visualizations
    4. Baseline Models

PART III: MACHINE LEARNING MODELS
    5. Linear Regression
    6. Random Forest
    7. XGBoost
    8. Neural Network

PART IV: MODEL EVALUATION & INTERPRETATION
    9. Model Comparison
    10. XGBoost Feature Importance
    11. SHAP Analysis
    12. Market Inefficiencies

Note: This project assumes the complete dataset (players_complete.csv) is already available locally.
All data was obtained by 3 scrapers (URL, price, attributes) and 1 merger.
These 4 files are available on this repository but won't be run in this file.

Type Ctrl+C on Mac to stop the project
"""

import sys
import logging
from pathlib import Path

# Setup logging (in this file, only for errors so terminal looks clean)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# False positive disabled: imports at each step for clarity and memory efficiency (+ clean terminal)
# pylint: disable=wrong-import-position,import-error,import-outside-toplevel,broad-exception-caught


def print_section_header(title: str) -> None:
    """Print major section divider (PART)."""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def print_step_header(title: str) -> None:
    """Print minor section divider (Step)."""
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def main() -> None:
    """Run the whole project step by step."""
    import time

    pipeline_start = time.time()

    print("=" * 120)
    print("EA FC 26 PLAYER PRICE PREDICTION | DATA SCIENCE AND ADVANCED PROGRAMMING")
    print("Made by Santiago Tailleferd (#20557377)")
    print("Random Seed: 42 (reproducible results - fixed for all models)")
    print("=" * 120)

    try:
        # PART I: DATA PREPARATION & EXPLORATION
        print_section_header("PART I: DATA PREPARATION & EXPLORATION")

        # Step 1: EDA Analysis
        print_step_header("Step 1: EDA Analysis")
        from analysis import eda_analysis

        eda_analysis.main()

        # Step 2: Feature Engineering
        print_step_header("Step 2: Feature Engineering")
        from preprocessing.feature_engineering import load_data, prepare_features

        df = load_data()
        _, _, _, _, _ = prepare_features(df, "price_w1", None, None, 10, None)
        logger.info("Note: All ML models will be trained on log-transformed prices")
        logger.info(
            "Note: RMSE and MAE are transformed back to original prices (credits)"
        )
        logger.info(
            "Note: R² is computed in log-price space (consistent with model training)"
        )

        # PART II: BASELINE & STATISTICAL ANALYSIS
        print_section_header("PART II: BASELINE & STATISTICAL ANALYSIS")

        # Step 3: OLS Coefficient Analysis & Visualizations
        print_step_header("Step 3: OLS Coefficient Analysis & Visualizations")
        from analysis import ols_coefficient_analysis, ols_visualizations

        ols_coefficient_analysis.main()
        ols_visualizations.main()

        # Step 4: Baseline Models
        print_step_header("Step 4: Baseline Models")
        from models import baseline_models

        baseline_models.main()

        # PART III: MACHINE LEARNING MODELS
        print_section_header("PART III: MACHINE LEARNING MODELS")

        # Step 5: Linear Regression
        print_step_header("Step 5: Linear Regression")
        from models import linear_regression

        linear_regression.main()

        # Step 6: Random Forest
        print_step_header("Step 6: Random Forest")
        from models import random_forest

        random_forest.main()

        # Step 7: XGBoost
        print_step_header("Step 7: XGBoost")
        from models import xgboost_model

        xgboost_model.main()

        # Step 8: Neural Network
        print_step_header("Step 8: Neural Network (MLP)")
        from models import neural_network

        neural_network.main()

        # PART IV: MODEL EVALUATION & INTERPRETATION
        print_section_header("PART IV: MODEL EVALUATION & INTERPRETATION")

        # Step 9: Model Comparison
        print_step_header("Step 9: Model Comparison")
        from analysis import model_comparison

        model_comparison.main()

        # Step 10: XGBoost Feature Importance
        print_step_header("Step 10: XGBoost Feature Importance")
        from analysis import xgboost_feature_importance

        xgboost_feature_importance.main()

        # Step 11: SHAP Analysis
        print_step_header("Step 11: SHAP Analysis")
        from analysis import shap_analysis

        shap_analysis.main()

        # Step 12: Market Inefficiencies
        print_step_header("Step 12: Market Inefficiencies")
        from analysis import market_inefficiencies

        market_inefficiencies.main()

        # FINAL SUMMARY
        print("\n" + "=" * 120)
        print("PROJECT EXECUTION COMPLETE")
        print("=" * 120)
        print("\nAll results saved to:")
        print(" - Figures: results/figures (16 PNG files)")
        print(" - Tables: results/tables (5 CSV files)")
        print(" - Predictions: results/predictions (4 CSV files)")
        print("\nLines of code:")
        print(" - Scrapers: 464 lines (not run on this file)")
        print(" - Preprocessing: 501 lines")
        print(" - Models: 858 lines")
        print(" - Analysis: 1,572 lines")
        print(" - Main: 207 lines")
        print("Total: 3,602 lines (+440 lines of tests)")
        print(
            "\nBest Model: XGBoost (R² = 0.956 | RMSE = 137,993 credits | MAE = 42,997 credits)"
        )
        print("+51% improvement over benchmark")
        print(f"\nTotal execution runtime: {time.time() - pipeline_start:.1f} seconds")
        print("=" * 120)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error("Pipeline failed at current step")
        logger.error("Error details: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
