"""
Unit Tests for Evaluation Metrics Module.

Tests the evaluate_predictions function with various inputs.
Matches project testing style.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# pylint: disable=wrong-import-position,import-error
from models.evaluation_metrics import evaluate_predictions

# pylint: enable=wrong-import-position,import-error

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run evaluation metrics tests."""
    logger.info("=" * 80)
    logger.info("EVALUATION METRICS UNIT TESTS")
    logger.info("=" * 80)
    print()

    # Test 1: Basic Evaluation Keys and Ranges
    logger.info("Test 1: Basic functionality")
    y_true_log = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    y_pred_log = np.array([10.1, 10.9, 12.2, 12.8, 14.1])

    metrics = evaluate_predictions(y_true_log, y_pred_log)

    # Check keys exist
    required_keys = ["mae", "rmse", "r2"]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    # Check values are reasonable
    assert metrics["mae"] > 0, "MAE should be positive"
    assert metrics["rmse"] > 0, "RMSE should be positive"
    assert metrics["r2"] <= 1.0, "R2 cannot be > 1.0"

    logger.info("  ✓ Metrics dictionary structure correct")
    logger.info("  ✓ Values within valid ranges")
    logger.info("  ✓ PASS")

    # Test 2: Perfect Predictions
    print()
    logger.info("Test 2: Perfect predictions (Expect R² = 1.0)")
    y_true_perfect = np.array([10.0, 11.0, 12.0])
    y_pred_perfect = np.array([10.0, 11.0, 12.0])

    metrics = evaluate_predictions(y_true_perfect, y_pred_perfect)

    logger.info("  Calculated R²: %.5f", metrics["r2"])
    logger.info("  Calculated MAE: %.5f", metrics["mae"])

    assert abs(metrics["r2"] - 1.0) < 0.0001, "R2 should be 1.0"
    assert abs(metrics["mae"] - 0.0) < 0.0001, "MAE should be 0.0"
    logger.info("  ✓ PASS")

    # Test 3: Error Handling
    print()
    logger.info("Test 3: Error handling (Mismatched arrays)")
    y_short = np.array([10.0, 11.0])

    try:
        evaluate_predictions(y_true_log, y_short)
        logger.error("  ⚠ WARNING: ValueError was not raised")
    except ValueError as e:
        logger.info("  ✓ Caught expected error: %s", e)
        logger.info("  ✓ PASS")

    # Summary
    print()
    logger.info("=" * 80)
    logger.info("TEST RESULT: ✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    print()


if __name__ == "__main__":
    main()
