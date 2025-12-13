"""
XGBoost Cross-Validation Test.

Tests model stability using 10-fold cross-validation.
"""

import sys
from pathlib import Path
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold

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
    """Run XGBoost cross-validation."""
    logger.info("=" * 80)
    logger.info("XGBOOST CROSS-VALIDATION TEST")
    logger.info("=" * 80)
    print()

    # Load data
    df = load_data()
    x_train, y_train_log, _, _, _ = prepare_features(
        df, "price_w1", None, None, 10, None
    )

    # Define model (same hyperparams as in ML model)
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # 10-fold CV
    logger.info("Running 10-fold cross-validation...")
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, x_train, y_train_log, cv=cv, scoring="r2")

    # Results
    print()
    logger.info("CV R² scores: %s", [f"{s:.3f}" for s in cv_scores])
    logger.info("Mean CV R²: %.3f ± %.3f", cv_scores.mean(), cv_scores.std() * 2)

    # Overfitting check
    xgb_model.fit(x_train, y_train_log)
    train_r2 = xgb_model.score(x_train, y_train_log)
    gap = train_r2 - cv_scores.mean()

    logger.info("Train R²: %.3f | Gap: %.3f", train_r2, gap)

    if gap < 0.05:
        logger.info("Status: ✓ Excellent generalization")
        test_result = "✓ Excellent generalization"
    elif gap < 0.10:
        logger.info("Status: ✓ Acceptable stability")
        test_result = "✓ Acceptable stability"
    else:
        logger.info("Status: ⚠ Significant overfitting")
        test_result = "⚠ WARNING: Significant overfitting"

    # Summary
    print()
    logger.info("=" * 80)
    logger.info("TEST RESULT: %s", test_result)
    logger.info("=" * 80)
    print()


if __name__ == "__main__":
    main()
