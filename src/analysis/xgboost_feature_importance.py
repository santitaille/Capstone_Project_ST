"""
XGBoost Feature Importance for EA FC 26 Player Price Prediction.

Re-trains XGBoost (best model) to extract feature importances and creates figures.

Generates:
* Figure 10: XGBoost feature importance with cumulative percentage (PNG)
* Table: XGBoost feature importance rankings (CSV)
"""

import sys
from pathlib import Path

# Setup paths for imports
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# False positive warnings, so to have no warnings
# pylint: disable=wrong-import-position,import-error,no-name-in-module
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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
    """Run XGBoost feature importance visualization."""
    try:
        # Load full dataset (players_complete.csv)
        df = load_data()
        logger.info("Training XGBoost model to extract feature importance...")
        print()

        # Prepare features (Week 1 for training)
        x_train, y_train_log, _, _, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
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

        # Extract feature importances
        importance_gain = xgb_model.feature_importances_

        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_gain}
        ).sort_values("importance", ascending=False)

        # Calculate cumulative importance
        importance_df["cumulative"] = importance_df["importance"].cumsum()
        importance_df["cumulative_pct"] = (
            importance_df["cumulative"] / importance_df["importance"].sum() * 100
        )

        # Save table
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        importance_path = TABLES_DIR / "xgboost_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        # Summary
        logger.info("Top 5 Features (by gain):")
        logger.info("-" * 80)
        for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
            label = f"{i}. {row['feature']}"
            logger.info(
                "  %-26s %5.1f%%  (Cumulative: %5.1f%%)",
                label,
                row["importance"] * 100,
                row["cumulative_pct"],
            )
        logger.info("-" * 80)
        feats_for_80 = importance_df[importance_df["cumulative_pct"] <= 80].shape[0] + 1
        logger.info("Top %d features explain 80%% of importance", feats_for_80)
        print()

        # Figure 1: XGBoost feature importance with cumulative percentage
        top42 = importance_df.head(42)

        _, ax1 = plt.subplots(figsize=(12, 8))

        # Bar chart for importance
        x_pos = np.arange(len(top42))
        _ = ax1.bar(
            x_pos, top42["importance"], color="steelblue", alpha=0.7, edgecolor="black"
        )
        ax1.set_xlabel("Features (ranked by importance)", fontsize=12)
        ax1.set_ylabel("Feature Importance (Gain)", fontsize=12, color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(top42["feature"], rotation=45, ha="right", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_xlim(-1, len(top42))

        # Cumulative line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            x_pos,
            top42["cumulative_pct"],
            color="red",
            marker="o",
            linewidth=2,
            markersize=4,
            label="Cumulative %",
        )
        ax2.axhline(y=80, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax2.set_ylabel("Cumulative Importance (%)", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim([0, 105])

        plt.title(
            "XGBoost Feature Importance with Cumulative Percentage",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "11_xgboost_cumulative_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("  → Saved: results/tables/xgboost_feature_importance.csv")
        logger.info("  → Saved: results/figures/11_xgboost_cumulative_importance.png")
        plt.close()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error during XGBoost feature importance visualization: %s", e
        )
        raise


if __name__ == "__main__":
    main()
