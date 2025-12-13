"""
SHAP Analysis for EA FC 26 Player Price Prediction.

Re-trains XGBoost (best model) to calculate SHAP Values and creates figures.

Generates:
* Figure 11: SHAP summary plot (beeswarm) - top 20 features (PNG)
* Figure 12: SHAP bar plot - mean absolute impact (PNG)
* Figure 13: SHAP waterfall plot - individual prediction breakdown (PNG)
* Table: SHAP feature importance rankings (CSV)

First stretch goal from proposal
"""

import sys
from pathlib import Path
import logging

# To have a clean temminal output in main.py
import warnings

warnings.filterwarnings("ignore", message=".*The NumPy global RNG was seeded.*")

# False positive warnings, so to have no warnings
# pylint: disable=wrong-import-position,import-error,protected-access
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

# Setup paths for imports
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Style configurations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def main() -> None:
    """Run SHAP analysis and visualizations."""
    try:
        # Load data
        df = load_data()
        logger.info("Training XGBoost model for SHAP analysis...")
        print()

        # Prepare train features (Week 1)
        x_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )

        # Prepare test features (Week 2)
        x_test, y_test_log, _, _, _ = prepare_features(
            df,
            target_col="price_w2",
            scaler=scaler,
            club_encoding_map=club_map,
            smoothing=10,
            feature_names=feature_names,
        )

        # Train XGBoost model (same parameters as xgboost_model.py)
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.5,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(x_train, y_train_log)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(x_test)

        # Calculate feature importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": shap_importance}
        ).sort_values("mean_abs_shap", ascending=False)

        # Summary
        logger.info("Top 5 Features (by |SHAP|):")
        logger.info("-" * 80)
        for i, (_, row) in enumerate(shap_importance_df.head(5).iterrows(), 1):
            logger.info("  %-26s %5.3f", f"{i}. {row['feature']}", row["mean_abs_shap"])
        logger.info("-" * 80)
        logger.info("Base value (log-price): %.3f", explainer.expected_value)
        logger.info(
            "SHAP range (log-price): %.3f to %.3f", shap_values.min(), shap_values.max()
        )
        print()

        # Example with most expensive player in the game
        high_value_idx = y_test_log.values.argmax()
        high_value_player = df.iloc[high_value_idx]
        player_shap = shap_values[high_value_idx]

        # XGBoost prediction = Base Value + sum of all SHAP values
        predicted_log = explainer.expected_value + player_shap.sum()
        predicted_price = np.expm1(predicted_log)
        actual_price = high_value_player["price_w2"]
        diff = predicted_price - actual_price
        diff_pct = (diff / actual_price) * 100

        # Identify key drivers (top 3 will appear in terminal)
        drivers = list(zip(feature_names, player_shap))
        drivers.sort(key=lambda x: x[1], reverse=True)
        drivers_str = ", ".join([f"{name} ({val:+.2f})" for name, val in drivers[:3]])

        # Display
        logger.info(
            "Example Player: %s (Actual Price: %s)",
            high_value_player["player_name"],
            f"{actual_price:,.0f}",
        )
        logger.info("Key Drivers: %s", drivers_str)
        logger.info(
            "Prediction: %s (Difference: %s | %s)",
            f"{predicted_price:,.0f}",
            f"{diff:+,.0f}",
            f"{diff_pct:+.1f}%",
        )
        print()

        # Figure 1: SHAP summary plot (Beeswarm)
        plt.figure(figsize=(10, 10))
        shap.summary_plot(
            shap_values, x_test, feature_names=feature_names, show=False, max_display=20
        )
        plt.title(
            "SHAP Summary Plot - Feature Impact on Predictions",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "12_shap_summary_beeswarm.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Figure 2: SHAP bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            x_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title(
            "SHAP Feature Importance - Mean Absolute Impact",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Mean |SHAP Value| (Average Impact on Prediction)", fontsize=12)
        plt.tight_layout()

        output_path = f"{OUTPUT_DIR}/13_shap_bar_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Figure 3: SHAP waterfall plot for same players as in example
        plt.figure(figsize=(10, 8))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[high_value_idx],
            feature_names=feature_names,
            max_display=15,
            show=False,
        )
        plt.title(
            f'SHAP Waterfall - {high_value_player["player_name"]} Prediction Breakdown',
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = f"{OUTPUT_DIR}/14_shap_waterfall_example.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save importance table
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        shap_importance_path = TABLES_DIR / "shap_feature_importance.csv"
        shap_importance_df.to_csv(shap_importance_path, index=False)

        # Summary
        logger.info("  → Saved: results/tables/shap_feature_importance.csv")
        logger.info("  → Saved: results/figures/12_shap_summary_beeswarm.png")
        logger.info("  → Saved: results/figures/13_shap_bar_importance.png")
        logger.info("  → Saved: results/figures/14_shap_waterfall_example.png")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except ImportError as e:
        logger.error("SHAP library not installed: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during SHAP analysis: %s", e)
        raise


if __name__ == "__main__":
    main()
