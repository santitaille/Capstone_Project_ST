"""
Baseline Models for EA FC 26 Player Price Prediction.

Baseline Model 1: Median Price by Rating
Baseline Model 2: Median Price by Rating + Card Category

Evaluates both baselines on:
- Week 1 (training set)
- Week 2 (test set, using W1 medians)

Comparison of both models using MAE, RMSE, and R².
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "players_complete.csv"


def main() -> None:
    """Evaluate baseline models on Week 2."""
    try:
        # Load full dataset (players_complete.csv)
        df = pd.read_csv(DATA_FILE)
        print()

        # Learn baseline models from Week 1
        rating_medians = df.groupby("rating")["price_w1"].median()
        rating_category_medians = df.groupby(["rating", "card_category"])[
            "price_w1"
        ].median()

        # Predict Week 2 prices
        df["baseline1_pred_w2"] = df["rating"].map(rating_medians)
        df["baseline2_pred_w2"] = df.apply(
            lambda row: rating_category_medians.get(
                (row["rating"], row["card_category"]), rating_medians[row["rating"]]
            ),
            axis=1,
        )

        # Evaluate basline models on Week 2
        mae1 = mean_absolute_error(df["price_w2"], df["baseline1_pred_w2"])
        rmse1 = np.sqrt(mean_squared_error(df["price_w2"], df["baseline1_pred_w2"]))
        r2_1 = r2_score(df["price_w2"], df["baseline1_pred_w2"])

        mae2 = mean_absolute_error(df["price_w2"], df["baseline2_pred_w2"])
        rmse2 = np.sqrt(mean_squared_error(df["price_w2"], df["baseline2_pred_w2"]))
        r2_2 = r2_score(df["price_w2"], df["baseline2_pred_w2"])

        # Summary
        logger.info("Baseline Models Results:")
        logger.info("-" * 80)
        logger.info("%-36s %6s %10s %10s", "Model", "R²", "RMSE", "MAE")
        logger.info("-" * 80)
        logger.info(
            "%-36s %6.3f %10s %10s",
            "Baseline 1: Median by Rating",
            r2_1,
            f"{rmse1:,.0f}",
            f"{mae1:,.0f}",
        )
        logger.info(
            "%-36s %6.3f %10s %10s",
            "Baseline 2: Median by Rating + Card",
            r2_2,
            f"{rmse2:,.0f}",
            f"{mae2:,.0f}",
        )
        logger.info("-" * 80)
        logger.info("Benchmark to beat: Baseline 2 (R² = %.3f)", r2_2)
        print()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Check that players_complete.csv exists in data/processed/")

    except KeyError as e:
        logger.error("Column not found: %s", e)
        logger.error("Check that price_w1 and price_w2 columns exist")

    except Exception as e:
        logger.error("Unexpected error during baseline evaluation: %s", e)
        raise


if __name__ == "__main__":
    main()
