"""
Final Model Comparison for EA FC 26 Player Price Prediction.

Compares all 6 methods on Week 2 test set:
- 2 Baselines (median-based)
- 4 Machine Learning models

Generates:
* Figure 0: Model comparison bar chart (R² performance) (PNG)
* Table: Model comparison results (R², RMSE, MAE) (CSV)
"""

import logging
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main() -> None:
    """Compare all models and generate leaderboard."""
    try:
        logger.info("Combining Results from 4 ML Models and 2 Baseline Models...")
        print()
        # Manual results from all models (Week 2 test set)
        results = {
            "Model": [
                "Baseline 1: Median by Rating",
                "Baseline 2: Median by Rating + Card",
                "Linear Regression",
                "Neural Network (MLP)",
                "Random Forest",
                "XGBoost",
            ],
            "R²": [
                0.608,  # Baseline 1
                0.633,  # Baseline 2
                0.631,  # Linear Regression
                0.669,  # Neural Network
                0.860,  # Random Forest
                0.956,
            ],  # XGBoost
            "RMSE": [
                410874,  # Baseline 1
                397507,  # Baseline 2
                398273,  # Linear Regression
                377422,  # Neural Network
                245831,  # Random Forest
                137993,
            ],  # XGBoost
            "MAE": [
                142364,  # Baseline 1
                129361,  # Baseline 2
                116050,  # Linear Regression
                92889,  # Neural Network
                63961,  # Random Forest
                42997,
            ],
        }  # XGBoost

        df = pd.DataFrame(results)

        # Add improvement over best baseline
        best_baseline_r2 = 0.633
        df["R² vs Benchmark"] = ((df["R²"] - best_baseline_r2) * 100).round(1)

        # Sort by R² descending
        df = df.sort_values("R²", ascending=False).reset_index(drop=True)

        # Save table
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "model_comparison.csv"
        df.to_csv(output_path, index=False)

        # Summary
        logger.info("Model Performance Comparison on Test Set (Sorted by R²):")
        logger.info("-" * 80)
        logger.info(
            "%-6s %-36s %8s %12s %12s %14s",
            "Rank",
            "Model",
            "R²",
            "RMSE",
            "MAE",
            "R² vs Benchmark",
        )
        logger.info("-" * 80)
        for idx, row in df.iterrows():
            r2_diff = (row["R²"] - best_baseline_r2) * 100

            logger.info(
                "%-6s %-36s %8s %12s %12s %14s",
                idx + 1,
                row["Model"],
                row["R²"],
                f"{row['RMSE']:,.0f}",
                f"{row['MAE']:,.0f}",
                f"{r2_diff:+.1f}%",
            )
        logger.info("-" * 80)
        winner = df.iloc[0]
        logger.info("Winner: %s (R² = %.3f)", winner["Model"], winner["R²"])
        print()

        # Figure 1: R² Comparison Bar Chart
        _, ax = plt.subplots(figsize=(12, 8))

        # Color code: baselines vs ML models
        colors = ["#2E7D32", "#4878A8", "#4878A8", "#808080", "#4878A8", "#FFA726"]
        bars = ax.barh(
            df["Model"], df["R²"], color=colors, alpha=0.8, edgecolor="black"
        )

        # Add R² values on bars
        for idx, (_, r2) in enumerate(zip(bars, df["R²"])):
            ax.text(
                r2 + 0.01, idx, f"{r2:.3f}", va="center", fontsize=10, fontweight="bold"
            )

        # Add baseline reference line
        ax.axvline(
            x=best_baseline_r2,
            color="#D32F2F",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Best Baseline",
        )

        ax.set_title(
            "Model Performance Comparison on Test Set (Sorted by R²)",
            fontsize=14,
            fontweight="bold",
            pad=25,
        )
        ax.set_xlim([0, 1.1])
        ax.legend(fontsize=10)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        # Invert y-axis so best is on top
        ax.invert_yaxis()

        # Save
        os.makedirs(FIGURES_DIR, exist_ok=True)
        viz_path = FIGURES_DIR / "10_model_comparison.png"
        plt.subplots_adjust(left=0.25, right=0.90, top=0.88, bottom=0.1)
        plt.savefig(viz_path, dpi=300)
        logger.info("  → Saved: results/tables/model_comparison.csv")
        logger.info("  → Saved: results/figures/10_model_comparison.png")
        plt.close()

    except Exception as e:
        logger.error("Error during model comparison: %s", e)
        raise


if __name__ == "__main__":
    main()
