"""
Model 3: XGBoost Regressor on log-transformed prices.

- Train on Week 1 prices (price_w1)
- Test on Week 2 prices (price_w2)
- Uses feature_engineering.prepare_features() to avoid data leakage
- Evaluates performance in both log-space and original price-space
"""

import logging
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold

import sys
from pathlib import Path
# Add src to path
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.feature_engineering import load_data, prepare_features
from models.evaluation_metrics import evaluate_predictions

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Run XGBoost"""
    try:

        # Load full dataset (players_complete.csv)
        df = load_data()
        logger.info("Starting XGBoost model training and evaluation")

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

        # Fit XGBoost on W1
        logger.info("\n=== FITTING XGBOOST MODEL ON W1 ===")
        xgb_model = XGBRegressor(
            n_estimators=1000, # Number of boosting rounds, large enough to fully fit with low learning rate
            learning_rate=0.03, # Slow learning rate for better generalization
            max_depth=4, # Moderate depth to prevent memorization (good bias-variance tradeoff)
            min_child_weight=5, # Reasonable default 
            subsample=0.7, # Use 70% of data per tree to reduce overfitting
            colsample_bytree=0.7, # Use 70% of features per tree, also to reduce overfitting
            reg_lambda=1.5, # L2: Smooths weights slightly more than default
            reg_alpha=0.5, # L1: Prunes useless one-hot columns
            random_state=42, # Random seed for reproducibility (same results on each run): 42 is commonly used
            n_jobs=-1 # -1 means use all available cores for maximum speed
        )

        # CROSS-VALIDATION CHECK (before fitting on full data)
        logger.info("\n=== CROSS-VALIDATION (Overfitting Check) ===")
        logger.info("Running 5-fold cross-validation on training data...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, X_train, y_train_log, cv=cv, scoring='r2', n_jobs=-1)
        
        logger.info(f"CV R² scores (5 folds): {[f'{score:.4f}' for score in cv_scores]}")
        logger.info(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Fit the model on the full training set
        xgb_model.fit(X_train, y_train_log)

        # Get the training score (R²)
        train_r2 = xgb_model.score(X_train, y_train_log)

        # Compare with Cross-Validation Mean
        cv_mean = cv_scores.mean()
        overfitting_gap = train_r2 - cv_mean

        logger.info(f"\n=== OVERFITTING CHECK (Train vs CV) ===")
        logger.info(f"Train R²:    {train_r2:.4f}")
        logger.info(f"CV Mean R²:  {cv_mean:.4f}")
        logger.info(f"Gap:         {overfitting_gap:.4f}")

        if overfitting_gap < 0.05:
            logger.info("✅ Excellent generalization - minimal overfitting")
        elif overfitting_gap < 0.10:
            logger.info("⚠️  Slight overfitting - acceptable for this model complexity")
        else:
            logger.warning("❌ Significant overfitting detected - consider more regularization")

        logger.info("\nModel fitted")
        logger.info(f"Number of boosting rounds: {xgb_model.n_estimators}")
        logger.info(f"Learning rate: {xgb_model.learning_rate}")

        # Inspect feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 15 features by importance:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"  {row['feature']:30s} : {row['importance']:.4f}")

        # Evaluate on TRAIN (W1) and TEST (W2)
        logger.info("\n=== EVALUATING MODEL ===")

        # Train performance
        y_train_pred_log = xgb_model.predict(X_train)
        _ = evaluate_predictions(y_train_log, y_train_pred_log, label="[TRAIN W1]")

        # Test performance
        y_test_pred_log = xgb_model.predict(X_test)
        metrics_test = evaluate_predictions(y_test_log, y_test_pred_log, label="[TEST W2]")

        # Save predictions
        logger.info("\nSaving predictions to CSV")
        results_df = pd.DataFrame({
            "player_name": df["player_name"],
            "rating": df["rating"],
            "card_category": df["card_category"],
            "price_w1": df["price_w1"],
            "price_w2": df["price_w2"],
            "pred_price_w2": np.expm1(y_test_pred_log).round(0),
        })

        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        results_path = PROJECT_ROOT / "results" / "predictions" / "predictions_xgboost_w2.csv"
        # Make sure directory exists:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False, float_format='%.0f')
        logger.info(f"Predictions saved to: {results_path}")

        # Final summary
        logger.info("\n=== FINAL SUMMARY (XGBOOST) ===")
        logger.info(f"Test R² (W2):   {metrics_test['r2']:.4f}")
        logger.info(f"Test RMSE (W2): {metrics_test['rmse']:,.0f} credits")
        logger.info(f"Test MAE (W2):  {metrics_test['mae']:,.0f} credits")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")

    except KeyError as e:
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")

    except ImportError as e:
        logger.error(f"XGBoost not installed: {e}")
        logger.error("Please install with: pip install xgboost")

    except Exception as e:
        logger.error(f"Unexpected error during XGBoost modeling: {e}")
        raise

if __name__ == "__main__":
    main()