"""
Evaluation metrics for EA FC 26 Player Price Prediction models.

Shared evaluation function computing MAE, RMSE, and R².
"""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true_log: pd.Series, y_pred_log: pd.Series
) -> Dict[str, float]:
    """
    Compute metrics in original price-space:
    Inputs y_true_log and y_pred_log) are in log1p(price) space.
    Returns a dictionary with MAE, RMSE, and R² in price-space.
    """
    try:
        # Validate inputs
        if len(y_true_log) == 0 or len(y_pred_log) == 0:
            raise ValueError("Empty prediction arrays")

        if len(y_true_log) != len(y_pred_log):
            raise ValueError(
                f"Array length mismatch: {len(y_true_log)} vs {len(y_pred_log)}"
            )

        # Back to original price-space
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {"mae": mae, "rmse": rmse, "r2": r2}

    except ValueError as e:
        logger.error("Validation error: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during evaluation: %s", e)
        raise
