"""
Evaluation metrics for EA FC 26 Player Price Prediction models.

This module provides shared evaluation functions used by all 4 models.
Computes metrics in both log-space and original price-space.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    # Log-space metrics
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)

    # Goes back to original price space
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Print results
    print(f"\nPerformance {label}:")
    print(f"  Log-space:")
    print(f"    MAE:  {mae_log:.4f}")
    print(f"    RMSE: {rmse_log:.4f}")
    print(f"    R²:   {r2_log:.4f}")
    print(f"  Price-space:")
    print(f"    MAE:  {mae:,.0f} credits")
    print(f"    RMSE: {rmse:,.0f} credits")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAPE: {mape:.2f}%")

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# TESTING
if __name__ == "__main__":
    # Simple test with dummy data
    print("Testing evaluate_predictions function")
    
    # Create dummy log-space predictions
    y_true_log = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    y_pred_log = np.array([10.1, 10.9, 12.2, 12.8, 14.1])
    
    metrics = evaluate_predictions(y_true_log, y_pred_log, label="[DUMMY TEST]")
    
    print(f"\nReturned metrics keys: {list(metrics.keys())}")