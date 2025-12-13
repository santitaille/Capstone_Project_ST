"""
Underpriced and Overpriced Players Detector for EA FC 26 Player Price Prediction.

Uses XGBoost predictions (best model) to identify players whose actual market price
differs significantly (+/- 20%) from their predicted value, revealing potential
trading opportunities.

- Underpriced: Actual price < Predicted price (potential buys)
- Overpriced: Actual price > Predicted price (potential sells)

Generates:
* Figure 16: Market inefficiency scatter plot (actual vs predicted) (PNG)
* Figure 17: Trading opportunities bar charts (top 15 each) (PNG)
* Table: Market inefficiencies - all mispriced players (CSV)

Second stretch goal from proposal
"""

import logging
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
PREDICTIONS_FILE = (
    PROJECT_ROOT / "results" / "predictions" / "predictions_xgboost_w2.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main() -> None:
    """Detect market inefficiencies using XGBoost predictions."""
    try:
        # Load XGBoost predictions (best model)
        df = pd.read_csv(PREDICTIONS_FILE)
        logger.info("Analyzing XGBoost predictions for mispriced players...")
        print()

        # Calculate price difference
        df["price_diff"] = df["pred_price_w2"] - df["price_w2"]
        df["price_diff_pct"] = (df["price_diff"] / df["price_w2"]) * 100
        df["abs_price_diff"] = df["price_diff"].abs()

        # Classify players (threshold: >20% difference)
        threshold = 20
        df["status"] = "Fair Value"
        df.loc[df["price_diff_pct"] > threshold, "status"] = "Underpriced"
        df.loc[df["price_diff_pct"] < -threshold, "status"] = "Overpriced"

        n_under = len(df[df["status"] == "Underpriced"])
        n_over = len(df[df["status"] == "Overpriced"])
        n_fair = len(df[df["status"] == "Fair Value"])

        # Display summary
        logger.info("Market Inefficiency Summary (Threshold: ±20%):")
        logger.info("-" * 80)
        logger.info(
            "  %-36s %4d players", "Underpriced (>20% below predicted):", n_under
        )
        logger.info(
            "  %-36s %4d players", "Overpriced  (>20% above predicted):", n_over
        )
        logger.info("  %-36s %4d players", "Fair value  (±20%):", n_fair)
        logger.info("-" * 80)
        print()

        # Figure 1: Scatter plot
        _, ax = plt.subplots(figsize=(12, 8))

        colors = {"Underpriced": "green", "Fair Value": "gray", "Overpriced": "red"}
        for status in ["Fair Value", "Underpriced", "Overpriced"]:
            subset = df[df["status"] == status]
            ax.scatter(
                subset["price_w2"],
                subset["pred_price_w2"],
                c=colors[status],
                alpha=0.6,
                s=30,
                label=status,
            )

        # Perfect prediction line and bands (20%)
        max_price = max(df["price_w2"].max(), df["pred_price_w2"].max())
        ax.plot(
            [0, max_price],
            [0, max_price],
            "k--",
            linewidth=2,
            alpha=0.5,
            label="Perfect Prediction",
        )
        ax.fill_between(
            [0, max_price],
            [0, max_price * 0.8],
            [0, max_price * 1.2],
            alpha=0.1,
            color="gray",
            label="±20% Band",
        )

        ax.set_xlabel("Actual Price W2 (credits)", fontsize=12)
        ax.set_ylabel("Predicted Price W2 (credits)", fontsize=12)
        ax.set_title(
            "Market Inefficiency Detection: Actual vs Predicted Prices",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "15_market_inefficiency_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Figure 2: Top opportunities bar chart
        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Top 15 underpriced
        top_underpriced = df[df["status"] == "Underpriced"].nlargest(
            15, "price_diff_pct"
        )
        axes[0].barh(
            range(len(top_underpriced)),
            top_underpriced["price_diff_pct"],
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        axes[0].set_yticks(range(len(top_underpriced)))
        axes[0].set_yticklabels(top_underpriced["player_name"], fontsize=9)
        axes[0].set_xlabel("Underpriced % (Predicted - Actual)", fontsize=11)
        axes[0].set_title(
            "Top 15 Underpriced Players (Best Buys)", fontsize=12, fontweight="bold"
        )
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.3)

        # Add percentage labels
        for idx, val in enumerate(top_underpriced["price_diff_pct"]):
            axes[0].text(val + 2, idx, f"+{val:.1f}%", va="center", fontsize=8)

        # Top 15 overpriced
        top_overpriced = df[df["status"] == "Overpriced"].nsmallest(
            15, "price_diff_pct"
        )
        axes[1].barh(
            range(len(top_overpriced)),
            top_overpriced["price_diff_pct"].abs(),
            color="red",
            alpha=0.7,
            edgecolor="black",
        )
        axes[1].set_yticks(range(len(top_overpriced)))
        axes[1].set_yticklabels(top_overpriced["player_name"], fontsize=9)
        axes[1].set_xlabel("Overpriced % (Actual - Predicted)", fontsize=11)
        axes[1].set_title(
            "Top 15 Overpriced Players (Avoid/Sell)", fontsize=12, fontweight="bold"
        )
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.3)

        # Add percentage labels
        for idx, val in enumerate(top_overpriced["price_diff_pct"].abs()):
            axes[1].text(val + 2, idx, f"+{val:.1f}%", va="center", fontsize=8)

        plt.tight_layout()
        output_path = FIGURES_DIR / "16_trading_opportunities.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save full results
        df_output = df[
            [
                "player_name",
                "rating",
                "card_category",
                "price_w2",
                "pred_price_w2",
                "price_diff",
                "price_diff_pct",
                "status",
            ]
        ]
        df_output = df_output.sort_values("price_diff_pct", ascending=False)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "market_inefficiencies.csv"
        df_output.to_csv(output_path, index=False)

        # Display
        top_buy = df[df["status"] == "Underpriced"].nlargest(3, "price_diff")
        top_sell = df[df["status"] == "Overpriced"].nsmallest(3, "price_diff")
        logger.info("Top 3 Buying Opportunities (by absolute upside):")
        for i, (_, row) in enumerate(top_buy.iterrows()):
            logger.info(
                "  %d. %-36s Actual: %10s | Predicted: %7s (%+8s credits)",
                i + 1,
                row["player_name"],
                f"{row['price_w2']:,.0f}",
                f"{row['pred_price_w2']:,.0f}",
                f"{row['price_diff']:,.0f}",
            )
        print()
        logger.info("Top 3 Selling Opportunities (by absolute downside):")
        for i, (idx, row) in enumerate(top_sell.iterrows()):
            logger.info(
                "  %d. %-36s Actual: %10s | Predicted: %7s (%+8s credits)",
                i + 1,
                row["player_name"],
                f"{row['price_w2']:,.0f}",
                f"{row['pred_price_w2']:,.0f}",
                f"{row['price_diff']:,.0f}",
            )
        print()

        # Summary
        logger.info("  → Saved: results/tables/market_inefficiencies.csv")
        logger.info("  → Saved: results/figures/15_market_inefficiency_scatter.png")
        logger.info("  → Saved: results/figures/16_trading_opportunities.png")
        print()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error during market inefficiency detection: %s", e)
        raise


if __name__ == "__main__":
    main()
