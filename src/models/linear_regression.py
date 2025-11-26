"""
Model 1: Multiple Linear Regression on log-transformed prices.

- Train on Week 1 prices (price_w1)
- Test on Week 2 prices (price_w2)
- Uses feature_engineering.prepare_features() to avoid data leakage
- Evaluates performance in both log-space and original price-space
"""

import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from feature_engineering import load_data, prepare_features
from evaluation_metrics import evaluate_predictions

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Load full dataset (players_complete.csv)
        df = load_data()
        logger.info("Starting Linear Regression model training and evaluation")

        # Ensure prices are positive
        if (df["price_w1"] <= 0).any() or (df["price_w2"] <= 0).any():
            raise ValueError("Detected non-positive prices. Clean the data before modeling")

        # Prepare TRAIN features (Week 1)
        logger.info("\n=== PREPARING TRAINING FEATURES (W1) ===")
        X_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train_log shape: {y_train_log.shape}")

        # Prepare TEST features (Week 2) using SAME scaler + club_map + feature_names
        logger.info("\n=== PREPARING TEST FEATURES (W2) ===")
        X_test, y_test_log, _, _, _ = prepare_features(
            df,
            target_col="price_w2",
            scaler=scaler,
            club_encoding_map=club_map,
            smoothing=10,
            feature_names=feature_names,
        )

        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test_log shape: {y_test_log.shape}")

        # Fit Linear Regression on W1
        logger.info("\n=== FITTING LINEAR REGRESSION MODEL ON W1 ===")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train_log)

        logger.info("Model fitted")
        logger.info(f"Number of coefficients: {len(lr_model.coef_)}")

        # Inspect top coefficients
        coef_series = pd.Series(lr_model.coef_, index=feature_names)
        coef_abs_sorted = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)

        logger.info("\nTop 15 features by absolute coefficient magnitude:")
        for name, value in coef_abs_sorted.head(15).items():
            logger.info(f"  {name:30s} : {value:+.4f}")

        # Evaluate on TRAIN (W1) and TEST (W2)
        logger.info("\n=== EVALUATING MODEL ===")

        # Train performance
        y_train_pred_log = lr_model.predict(X_train)
        _ = evaluate_predictions(y_train_log, y_train_pred_log, label="[TRAIN W1]")

        # Test performance
        y_test_pred_log = lr_model.predict(X_test)
        metrics_test = evaluate_predictions(y_test_log, y_test_pred_log, label="[TEST W2]")

        # Save predictions
        logger.info("\nSaving predictions to CSV")
        results_df = pd.DataFrame({
            "player_name": df["player_name"],
            "rating": df["rating"],
            "card_category": df["card_category"],
            "price_w1": df["price_w1"],
            "price_w2": df["price_w2"],
            "pred_price_w2": np.expm1(y_test_pred_log),
        })

        results_path = "/files/Capstone_Project_ST/data/processed/predictions_linear_regression_w2.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Predictions saved to: {results_path}")

        # Final summary
        logger.info("\n=== FINAL SUMMARY (LINEAR REGRESSION) ===")
        logger.info(f"Test R² (W2):   {metrics_test['r2']:.4f}")
        logger.info(f"Test RMSE (W2): {metrics_test['rmse']:,.0f} credits")
        logger.info(f"Test MAE (W2):  {metrics_test['mae']:,.0f} credits")
        logger.info(f"Test MAPE (W2): {metrics_test['mape']:.2f}%")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")

    except KeyError as e:
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")

    except Exception as e:
        logger.error(f"Unexpected error during linear regression modeling: {e}")
        raise