"""
Feature Engineering Validation Test.

Validates feature engineering correctness: shape, scaling, and train/test consistency.
"""

import sys
from pathlib import Path
import logging

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# pylint: disable=wrong-import-position,import-error
from preprocessing.feature_engineering import load_data, prepare_features

# pylint: enable=wrong-import-position,import-error

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Test feature engineering file."""
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING VALIDATION TEST")
    logger.info("=" * 80)
    print()

    # Load data
    df = load_data()
    logger.info("Loaded %d players", len(df))

    # Prepare features (Week 1)
    x_train, _, scaler, club_map, features = prepare_features(
        df, "price_w1", None, None, 10, None
    )

    # Test 1: Correct shape
    print()
    logger.info("Test 1: Feature matrix shape")
    logger.info("  Features: %d (expected 41)", x_train.shape[1])
    logger.info("  Samples: %d", x_train.shape[0])
    assert x_train.shape[1] == 41, "Wrong number of features"
    assert x_train.shape[0] == len(df), "Sample count mismatch"
    logger.info("  ✓ PASS")

    # Test 2: Features are scaled (mean = 0, std = 1)
    print()
    logger.info("Test 2: Feature scaling")
    rating_idx = features.index("rating")
    mean = x_train.iloc[:, rating_idx].mean()
    std = x_train.iloc[:, rating_idx].std()
    logger.info("  Rating: mean=%.3f, std=%.3f", mean, std)
    assert abs(mean) < 0.1, "Mean not close to 0"
    assert abs(std - 1.0) < 0.1, "Std not close to 1"
    logger.info("  ✓ PASS")

    # Test 3: Test set uses same scaler (no data leakage)
    print()
    logger.info("Test 3: Train/test consistency")
    x_test, _, _, _, _ = prepare_features(
        df, "price_w2", scaler, club_map, 10, features
    )
    logger.info("  Train features: %d", x_train.shape[1])
    logger.info("  Test features: %d", x_test.shape[1])
    assert x_test.shape[1] == x_train.shape[1], "Feature count mismatch"
    assert list(x_test.columns) == list(x_train.columns), "Feature names differ"
    logger.info("  ✓ PASS")

    # Summary
    print()
    logger.info("=" * 80)
    logger.info("TEST RESULT: ✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    print()


if __name__ == "__main__":
    main()
