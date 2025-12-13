"""
Data Integrity Test.

Validates all CSV files exist and have correct structure.
"""

import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Test data file integrity."""
    logger.info("=" * 80)
    logger.info("DATA INTEGRITY TEST")
    logger.info("=" * 80)
    print()

    # Test 1: Raw data files exist
    logger.info("Test 1: Raw data files")
    raw_files = {
        "Player URLs": PROJECT_ROOT / "data" / "player_urls.csv",
        "Attributes": PROJECT_ROOT / "data" / "players_attributes.csv",
        "Week 1 Prices": PROJECT_ROOT / "data" / "week1" / "prices_week1.csv",
        "Week 2 Prices": PROJECT_ROOT / "data" / "week2" / "prices_week2.csv",
    }

    for name, path in raw_files.items():
        assert path.exists(), f"{name} missing: {path}"
        logger.info("  ✓ %s", name)
    logger.info("  ✓ PASS")

    # Test 2: Processed data files exist
    print()
    logger.info("Test 2: Processed data files")
    processed_files = {
        "Week 1 Merged": PROJECT_ROOT / "data" / "processed" / "players_with_week1.csv",
        "Complete": PROJECT_ROOT / "data" / "processed" / "players_complete.csv",
    }

    for name, path in processed_files.items():
        assert path.exists(), f"{name} missing: {path}"
        logger.info("  ✓ %s", name)
    logger.info("  ✓ PASS")

    # Test 3: Schema validation
    print()
    logger.info("Test 3: Data schema")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "players_complete.csv")

    required_cols = [
        "player_name",
        "rating",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physical",
        "price_w1",
        "price_w2",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    logger.info("  ✓ All required columns present")

    # Check no nulls in critical columns
    critical_nulls = (
        df[["player_name", "rating", "price_w1", "price_w2"]].isnull().sum()
    )
    assert (
        critical_nulls.sum() == 0
    ), f"Nulls found: {critical_nulls[critical_nulls > 0].to_dict()}"
    logger.info("  ✓ No nulls in critical columns")
    logger.info("  Dataset: %d players, %d columns", len(df), len(df.columns))
    logger.info("  ✓ PASS")

    # Summary
    print()
    logger.info("=" * 80)
    logger.info("TEST RESULT: ✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    print()


if __name__ == "__main__":
    main()
