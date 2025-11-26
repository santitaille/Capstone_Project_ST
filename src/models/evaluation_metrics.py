"""
Evaluation metrics for EA FC 26 Player Price Prediction models.

This module provides shared evaluation functions used by all models.
Computes metrics in both log-space and original price-space.
"""

import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_predictions(y_true_log, y_pred_log, label=""):
    """
    Compute metrics in both log-space and original price-space.
    
    Args:
        y_true_log: True values in log1p(price) space
        y_pred_log: Predicted values in log1p(price) space
        label: Label for printing (e.g., "[TRAIN W1]" or "[TEST W2]")
        
    Returns:
        Dictionary with all metrics
    """
    try:
        # Validate inputs
        if len(y_true_log) == 0 or len(y_pred_log) == 0:
            raise ValueError("Empty prediction arrays")
        
        if len(y_true_log) != len(y_pred_log):
            raise ValueError(f"Array length mismatch: {len(y_true_log)} vs {len(y_pred_log)}")
        
        # Log-space metrics
        mae_log = mean_absolute_error(y_true_log, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        r2_log = r2_score(y_true_log, y_pred_log)

        # Back-transform to original price space
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Log results
        logger.info(f"\nPerformance {label}:")
        logger.info(f"  Log-space:")
        logger.info(f"    MAE:  {mae_log:.4f}")
        logger.info(f"    RMSE: {rmse_log:.4f}")
        logger.info(f"    R²:   {r2_log:.4f}")
        logger.info(f"  Price-space:")
        logger.info(f"    MAE:  {mae:,.0f} credits")
        logger.info(f"    RMSE: {rmse:,.0f} credits")
        logger.info(f"    R²:   {r2:.4f}")
        logger.info(f"    MAPE: {mape:.2f}%")

        return {
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "r2_log": r2_log,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }
    
    except ValueError as e:
        logger.error(f"Validation error in evaluate_predictions: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        raise


# TESTING
if __name__ == "__main__":
    try:
        # Simple test with dummy data
        logger.info("Testing evaluate_predictions function")
        
        # Create dummy log-space predictions
        y_true_log = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        y_pred_log = np.array([10.1, 10.9, 12.2, 12.8, 14.1])
        
        metrics = evaluate_predictions(y_true_log, y_pred_log, label="[DUMMY TEST]")
        
        logger.info(f"\nReturned metrics keys: {list(metrics.keys())}")
        
        # Test error handling
        logger.info("\nTesting error handling with mismatched arrays")
        try:
            y_short = np.array([10.0, 11.0])
            evaluate_predictions(y_true_log, y_short, label="[ERROR TEST]")
        except ValueError as e:
            logger.info(f"Correctly caught error: {e}")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise