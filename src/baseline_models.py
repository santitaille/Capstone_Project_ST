"""
Baseline Model 1: Median Price per OVR Rating
Baseline Model 2: Median Price per (OVR Rating × Card Category)
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

        # load data
        df = pd.read_csv("/files/Capstone_Project_ST/data/processed/players_with_week1.csv")
        logger.info(f"Dataset: {len(df)} players\n")

        #BASELINE MODEL 1: Median Price per OVR rating
        rating_medians = df.groupby('rating')['price_w1'].median()

        logger.info("Baseline model 1 : Median prices by rating:")
        print(rating_medians)
        print()

        # make predictions for baseline model 1
        df['baseline1_pred'] = df['rating'].map(rating_medians)

        # evaluate baseline model 1
        mae = mean_absolute_error(df['price_w1'], df['baseline1_pred']) # on average, the model is off by this many credits
        rmse = np.sqrt(mean_squared_error(df['price_w1'], df['baseline1_pred'])) # on average, the model is off by this many credits (penalizes larger errors more)
        r2 = r2_score(df['price_w1'], df['baseline1_pred']) # how much variance is explained by the model

        logger.info(f"Baseline model 1 (OVR only):")
        logger.info(f"  MAE:  {mae:,.0f} credits")
        logger.info(f"  RMSE: {rmse:,.0f} credits")
        logger.info(f"  R²:   {r2:.3f}")


        #BASELINE MODEL 2: Median Price per (OVR Rating × Card Category)
        rating_category_medians = df.groupby(['rating', 'card_category'])['price_w1'].median()

        logger.info("\nBaseline model 2: Median prices by rating and category:")
        print(rating_category_medians.head(30))
        print()

        # make predictions for baseline model 2 (fallback to rating median if combo not found)
        df['baseline2_pred'] = df.apply(
            lambda row: rating_category_medians.get((row['rating'], row['card_category']), rating_medians[row['rating']]), axis=1)

        # evaluate baseline model 2
        mae2 = mean_absolute_error(df['price_w1'], df['baseline2_pred']) # on average, the model is off by this many credits
        rmse2 = np.sqrt(mean_squared_error(df['price_w1'], df['baseline2_pred'])) # on average, the model is off by this many credits (penalizes larger errors more)
        r2_2 = r2_score(df['price_w1'], df['baseline2_pred']) # how much variance is explained by the model

        logger.info(f"Baseline model 2 (OVR x Card Category):")
        logger.info(f"  MAE:  {mae2:,.0f} credits")
        logger.info(f"  RMSE: {rmse2:,.0f} credits")
        logger.info(f"  R²:   {r2_2:.3f}")

        #COMPARISON OF BOTH MODELS
        print("\nComparison between both baseline models:")
        mae_improvement = (mae - mae2) / mae * 100
        rmse_improvement = (rmse - rmse2) / rmse * 100
        r2_improvement = r2_2 - r2
        print(f"  Baseline 2 is {mae_improvement:.1f}% better than Baseline 1 (MAE)")
        print(f"  Baseline 2 is {rmse_improvement:.1f}% better than Baseline 1 (RMSE)")
        print(f"  R² improvement: {r2_improvement:.3f}")
        
    except FileNotFoundError as e:  # in case file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
    
    except KeyError as e:  # in case column is missing
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist")
    
    except Exception as e:  # in case any other error happens
        logger.error(f"Unexpected errors during baseline evaluation: {e}")
        raise
