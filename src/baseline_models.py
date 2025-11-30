"""
Baseline Model 1: Median Price per OVR Rating
Baseline Model 2: Median Price per (OVR Rating x Card Category)

Evaluates both baselines on:
- Week 1 (training set)
- Week 2 (test set, using W1-learned medians)

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
        logger.info(f"Dataset: {len(df)} players")
        logger.info(f"Evaluating baselines on Week 2 (test set)\n")

        # LEARN BASELINES FROM WEEK 1
        # Baseline model 1: Median Price per OVR rating (from W1)
        rating_medians = df.groupby('rating')['price_w1'].median()

        # Baseline model 2: Median Price per (OVR Rating x Card Category) (from W1)
        rating_category_medians = df.groupby(['rating', 'card_category'])['price_w1'].median()

        # EVALUATE ON WEEK 2 (TEST SET - USING W1 MEDIANS)
        # Baseline 1 predictions on W2 (using W1 medians)
        df['baseline1_pred_w2'] = df['rating'].map(rating_medians)

        # Baseline 2 predictions on W2 (using W1 medians)
        df['baseline2_pred_w2'] = df.apply(
            lambda row: rating_category_medians.get(
                (row['rating'], row['card_category']), 
                rating_medians[row['rating']]
            ), axis=1
        )

        # Evaluate Baseline 1 on W2
        mae1_w2 = mean_absolute_error(df['price_w2'], df['baseline1_pred_w2'])
        rmse1_w2 = np.sqrt(mean_squared_error(df['price_w2'], df['baseline1_pred_w2']))
        r2_1_w2 = r2_score(df['price_w2'], df['baseline1_pred_w2'])

        logger.info(f"\nBaseline 1 (OVR only) - Week 2:")
        logger.info(f"  MAE:  {mae1_w2:,.0f} credits")
        logger.info(f"  RMSE: {rmse1_w2:,.0f} credits")
        logger.info(f"  R²:   {r2_1_w2:.4f}")

        # Evaluate Baseline 2 on W2
        mae2_w2 = mean_absolute_error(df['price_w2'], df['baseline2_pred_w2'])
        rmse2_w2 = np.sqrt(mean_squared_error(df['price_w2'], df['baseline2_pred_w2']))
        r2_2_w2 = r2_score(df['price_w2'], df['baseline2_pred_w2'])

        logger.info(f"\nBaseline 2 (OVR x Card Category) - Week 2:")
        logger.info(f"  MAE:  {mae2_w2:,.0f} credits")
        logger.info(f"  RMSE: {rmse2_w2:,.0f} credits")
        logger.info(f"  R²:   {r2_2_w2:.4f}")

        # SUMMARY
        logger.info("\nSUMMARY")
        logger.info(f"Best Baseline on Week 2 (Test Set):")
        logger.info(f"  Baseline 2 (OVR x Category): R² = {r2_2_w2:.4f} - this is the benchmark that ML models must beat!")
        
    except FileNotFoundError as e:  # in case file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
        logger.error("Make sure you're using players_complete.csv (with both W1 and W2)")
    
    except KeyError as e:  # in case column is missing
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist")
    
    except Exception as e:  # in case any other error happens
        logger.error(f"Unexpected errors during baseline evaluation: {e}")
        raise