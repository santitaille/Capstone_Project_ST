"""
Exploratory Data Analysis for EA FC 26 Player Price Prediction.

Generates:
* Figure 01: Price distribution (normal and log scale) (PNG)
* Figure 02: Rating vs Price scatter by card category (PNG)
* Figure 03: Price by card category boxplot (PNG)
* Figure 04: Price by position cluster bar chart (PNG)
* Figure 05: Correlation matrix heatmap (PNG)
* Figure 06: Top 4 attributes vs price scatter plots (PNG)
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "players_with_week1.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

# Style configurations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def main() -> None:
    """Run EDA analysis and generate figures."""
    try:
        # Load data
        logger.info("Loading data from %s", DATA_FILE.relative_to(PROJECT_ROOT))
        df = pd.read_csv(DATA_FILE)
        logger.info("Loaded %d players", len(df))

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # So in the Figures, we can show "Icons/Heroes" instead of "Icons_Heroes"
        df["card_category_display"] = df["card_category"].replace(
            "Icons_Heroes", "Icons/Heroes"
        )

        # Figure 1: Price distribution (normal and log scale)
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Normal scale
        axes[0].hist(
            df["price_w1"], bins=50, color="#4878A8", edgecolor="black", alpha=0.7
        )
        axes[0].set_xlabel("Price (Week 1)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12, labelpad=10)
        axes[0].set_title(
            "Price Distribution (Normal Scale)", fontsize=14, fontweight="bold"
        )
        axes[0].tick_params(labelsize=10)
        axes[0].grid(alpha=0.3)

        axes[1].hist(
            np.log1p(df["price_w1"]),
            bins=50,
            color="#4878A8",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_xlabel("Log Price (Week 1)", fontsize=12)
        axes[1].set_title(
            "Price Distribution (Log Scale)", fontsize=14, fontweight="bold"
        )
        axes[1].tick_params(labelsize=10)
        axes[1].grid(alpha=0.3)

        # Save
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
        plt.savefig(OUTPUT_DIR / "01_price_distribution.png", dpi=300)
        plt.close()

        # Figure 2: Rating vs Price by card category
        plt.figure(figsize=(12, 8))
        colors = {"Gold": "#D4AF37", "Special": "#2E7D32", "Icons/Heroes": "#1976D2"}

        for category in df["card_category_display"].unique():
            subset = df[df["card_category_display"] == category]
            plt.scatter(
                subset["rating"],
                subset["price_w1"],
                label=category,
                alpha=0.6,
                s=30,
                color=colors.get(category, "#808080"),
            )

        plt.xlabel("Overall Rating", fontsize=12, labelpad=10)
        plt.yscale("log")
        plt.ylabel("Price (Week 1, Log Scale)", fontsize=12, labelpad=10)
        plt.title(
            "Overall Rating vs Price by Card Category", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=10)
        plt.gca().tick_params(labelsize=10)
        plt.grid(alpha=0.3)

        # Save
        plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        plt.savefig(OUTPUT_DIR / "02_rating_vs_price.png", dpi=300)
        plt.close()

        # Figure 3: Price by card category
        plt.figure(figsize=(12, 8))
        order = ["Gold", "Special", "Icons/Heroes"]
        palette = {"Gold": "#D4AF37", "Special": "#2E7D32", "Icons/Heroes": "#1976D2"}

        sns.boxplot(
            data=df,
            x="card_category_display",
            y="price_w1",
            order=order,
            hue="card_category_display",
            palette=palette,
            legend=False,
            ax=plt.gca(),
        )

        plt.title("Price by Card Category", fontsize=14, fontweight="bold")
        plt.suptitle("")
        plt.yscale("log")
        plt.ylabel("Price (Week 1, Log Scale)", fontsize=12, labelpad=10)
        plt.xlabel("Card Category", fontsize=12, labelpad=10)
        plt.xticks(rotation=0)
        plt.gca().tick_params(labelsize=10)
        plt.grid(alpha=0.3)

        # Save
        plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        plt.savefig(OUTPUT_DIR / "03_price_by_category.png", dpi=300)
        plt.close()

        # Figure 4: Price by position cluster
        position_cols = [
            "cluster_cb",
            "cluster_fullbacks",
            "cluster_mid",
            "cluster_att_mid",
            "cluster_st",
        ]
        position_names = [
            "Center Backs",
            "Fullbacks",
            "Midfielders",
            "Attacking Midfielders",
            "Strikers",
        ]
        position_prices = []
        for col in position_cols:
            if col in df.columns:
                players_in_pos = df[df[col] == 1]
                position_prices.append(players_in_pos["price_w1"].mean())
            else:
                position_prices.append(0)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            position_names,
            position_prices,
            color="#4878A8",
            edgecolor="black",
            alpha=0.7,
        )
        plt.xlabel("Position Cluster", fontsize=12, labelpad=10)
        plt.ylabel("Average Price (Week 1)", fontsize=12, labelpad=10)
        plt.title("Average Price by Position Cluster", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.gca().tick_params(labelsize=10)
        plt.grid(axis="y", alpha=0.3)

        for plot_bar, price in zip(bars, position_prices):
            height = plot_bar.get_height()
            plt.text(
                plot_bar.get_x() + plot_bar.get_width() / 2.0,
                height,
                f"{price:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Save
        plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.20)
        plt.savefig(OUTPUT_DIR / "04_price_by_position.png", dpi=300)
        plt.close()

        # Figure 5: Correlation matrix
        numeric_cols = [
            "rating",
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physical",
            "skill_moves",
            "weak_foot",
            "num_playstyles",
            "num_playstyles_plus",
            "num_positions",
            "price_w1",
        ]

        numeric_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[numeric_cols].corr()
        price_corr = corr_matrix["price_w1"].sort_values(ascending=False)

        plt.figure(figsize=(12, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9},
        )
        plt.title(
            "Correlation Matrix: Player Attributes", fontsize=14, fontweight="bold"
        )

        # Save
        plt.xticks(rotation=45, ha="right")
        plt.gca().tick_params(labelsize=10)
        plt.subplots_adjust(left=0.15, right=1, top=1.05, bottom=0)
        plt.savefig(OUTPUT_DIR / "05_correlation_matrix.png", dpi=300)
        plt.close()

        # Figure 6: Top 4 attributes vs price
        _, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Pace
        axes[0, 0].scatter(df["pace"], df["price_w1"], alpha=0.5, s=30, color="#5BA85B")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlabel("Pace", fontsize=12)
        axes[0, 0].set_ylabel("Price (Week 1, Log Scale)", fontsize=12, labelpad=10)
        axes[0, 0].set_title("Pace vs Price", fontsize=14, fontweight="bold")

        # Dribbling
        axes[0, 1].scatter(
            df["dribbling"], df["price_w1"], alpha=0.5, s=30, color="#FF8C00"
        )
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_xlabel("Dribbling", fontsize=12)
        axes[0, 1].set_title("Dribbling vs Price", fontsize=14, fontweight="bold")

        # Skill Moves
        axes[1, 0].scatter(
            df["skill_moves"], df["price_w1"], alpha=0.5, s=30, color="#8E44AD"
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlabel("Skill Moves", fontsize=12)
        axes[1, 0].set_ylabel("Price (Week 1, Log Scale)", fontsize=12, labelpad=10)
        axes[1, 0].set_title("Skill Moves vs Price", fontsize=14, fontweight="bold")

        # Weak Foot
        axes[1, 1].scatter(
            df["weak_foot"], df["price_w1"], alpha=0.5, s=30, color="#D64545"
        )
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Weak Foot", fontsize=12)
        axes[1, 1].set_title("Weak Foot vs Price", fontsize=14, fontweight="bold")

        # Save
        plt.subplots_adjust(
            left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.3
        )
        plt.savefig(OUTPUT_DIR / "06_top4_attributes.png", dpi=300)
        plt.close()

        # Summary
        print()
        logger.info("EDA Summary Statistics:")
        logger.info("-" * 80)
        logger.info(
            "Strongest correlation with price: %s (r = %.3f)",
            price_corr.index[1],
            price_corr.iloc[1],
        )
        logger.info("")
        logger.info("Price Ranges by Category:")

        for category in ["Gold", "Special", "Icons/Heroes"]:
            cat_data = df[df["card_category"] == category]["price_w1"]
            if len(cat_data) > 0:
                logger.info(
                    "  %-13s: %7s - %9s credits (median: %7s credits)",
                    f"{category}",
                    f"{cat_data.min():,.0f}",
                    f"{cat_data.max():,.0f}",
                    f"{cat_data.median():,.0f}",
                )
        logger.info("-" * 80)
        print()

        logger.info("  → Saved: 01_price_distribution.png")
        logger.info("  → Saved: 02_rating_vs_price.png")
        logger.info("  → Saved: 03_price_by_category.png")
        logger.info("  → Saved: 04_price_by_position.png")
        logger.info("  → Saved: 05_correlation_matrix.png")
        logger.info("  → Saved: 06_top4_attributes.png")

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during EDA: %s", e)
        raise


if __name__ == "__main__":
    main()
