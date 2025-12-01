"""
Baseline Models Evaluation for EA FC 26 Player Price Prediction

Baseline Model 1: Median Price per OVR Rating
Baseline Model 2: Median Price per (OVR Rating x Card Category)

Evaluates both baselines on:
- Week 1 (training set)
- Week 2 (test set, using W1 medians)

Comparison of both models using MAE, RMSE, and R².
"""

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:

        # load data (both W1 and W2 prices)
        df = pd.read_csv("/files/Capstone_Project_ST/data/processed/players_complete.csv")
        logger.info("Loaded %d players", len(df))
        logger.info("Evaluating baselines on Week 2 (test set)")
        print()

        # LEARN BASELINES FROM WEEK 1
        # Baseline model 1: Median Price per OVR rating (from W1)
        rating_medians = df.groupby('rating')['price_w1'].median()

        # Baseline model 2: Median Price per (OVR Rating x Card Category) (from W1)
        rating_category_medians = df.groupby(['rating', 'card_category'])['price_w1'].median()

        # PREDICT WEEK 2 PRICES USING WEEK 1 MEDIANS
        # Baseline model 1: simple rating mapping
        df['baseline1_pred_w2'] = df['rating'].map(rating_medians)

        # Baseline model 2: rating x category lookup (if missing, fallback to rating median)
        df['baseline2_pred_w2'] = df.apply(
            lambda row: rating_category_medians.get(
                (row['rating'], row['card_category']),
                rating_medians[row['rating']]
            ), axis=1
        )

        # EVALUATE BASELINES ON WEEK 2
        # Evaluate Baseline model 1 on W2
        mae1 = mean_absolute_error(df['price_w2'], df['baseline1_pred_w2'])
        rmse1 = np.sqrt(mean_squared_error(df['price_w2'], df['baseline1_pred_w2']))
        r2_1 = r2_score(df['price_w2'], df['baseline1_pred_w2'])

        logger.info("Baseline 1 (OVR only) - Week 2:")
        logger.info("  MAE:  %s credits", f"{mae1:,.0f}")
        logger.info("  RMSE: %s credits", f"{rmse1:,.0f}")
        logger.info("  R²:   %.4f", r2_1)
        print()

        # Evaluate Baseline model 2 on W2
        mae2 = mean_absolute_error(df['price_w2'], df['baseline2_pred_w2'])
        rmse2 = np.sqrt(mean_squared_error(df['price_w2'], df['baseline2_pred_w2']))
        r2_2 = r2_score(df['price_w2'], df['baseline2_pred_w2'])

        logger.info("Baseline 2 (OVR x Category) - Week 2:")
        logger.info("  MAE:  %s credits", f"{mae2:,.0f}")
        logger.info("  RMSE: %s credits", f"{rmse2:,.0f}")
        logger.info("  R²:   %.4f", r2_2)
        print()

        # SUMMARY
        logger.info("BENCHMARK FOR ML MODELS")
        logger.info("Best Baseline: Baseline 2 (OVR x Category)")
        logger.info("  R² = %.4f - ML models must beat this", r2_2)
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
