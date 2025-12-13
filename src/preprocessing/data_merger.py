"""
Data Merger for EA FC 26 Player Price Prediction.

Merges player attributes with weekly price data in two stages:
1. Attributes + Week 1 prices → players_with_week1.csv
2. Previous result + Week 2 prices → players_complete.csv

Generates:
* Table: Merged player data with prices (CSV)

IMPORTANT:
For second merge: uncomment lines marked with #W2 and comment lines marked with #W1.
"""

import logging
import os
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
ATTRIBUTES_FILE = "/files/Capstone_Project_ST/data/players_attributes.csv"
WEEK1_FILE = "/files/Capstone_Project_ST/data/week1/prices_week1.csv"
WEEK2_FILE = "/files/Capstone_Project_ST/data/week2/prices_week2.csv"


if __name__ == "__main__":
    try:
        # load CSV files
        logger.info("Starting data merge")
        logger.info("Loading CSV files")

        # CAREFUL: check comments for W1 and W2
        df_attrs = pd.read_csv(ATTRIBUTES_FILE)
        df_w1 = pd.read_csv(WEEK1_FILE)  # ALWAYS UNCOMMENTED
        df_w2 = pd.read_csv(WEEK2_FILE)  # UNCOMMENT FOR W2

        logger.info("Players attributes: %d rows", len(df_attrs))
        # CAREFUL: only 1 line should be uncommented at a time
        # logger.info(f"Week 1 prices: {len(df_w1)} rows") # UNCOMMENT FOR W1
        logger.info("Week 2 prices: %d rows", len(df_w2))  # UNCOMMENT FOR W2

        # Merge datasets
        logger.info("Merging datasets")

        # CAREFUL: only 1 line should be uncommented at a time
        # pylint: disable=line-too-long
        # df_merged = df_attrs.merge(df_w1.rename(columns={'price': 'price_w1'}), on="url", how="inner") # UNCOMMENT FOR W1
        df_merged = df_attrs.merge(df_w1, on="url").merge(
            df_w2, on="url", how="inner", suffixes=("_w1", "_w2")
        )  # UNCOMMENT FOR W2

        logger.info(
            "Merged datasets: %d rows, %d columns",
            len(df_merged),
            len(df_merged.columns),
        )

        # in case there were any missing values after merge
        missing = df_merged.isnull().sum()
        missing_counts = missing[missing > 0]

        if len(missing_counts) > 0:
            logger.warning(
                "There are %d missing values after merge", missing_counts.sum()
            )
        else:
            logger.info("There are no missing values")

        logger.info(
            "Merged datasets: %s rows, %s columns",
            len(df_merged),
            len(df_merged.columns),
        )

        # remove players with price = 0
        logger.info("Removing players with invalid prices")
        initial_size = len(df_merged)
        # df_merged = df_merged[df_merged['price_w1'] > 0]  # UNCOMMENT FOR W1
        df_merged = df_merged[
            (df_merged["price_w1"] > 0) & (df_merged["price_w2"] > 0)
        ]  # UNCOMMENT FOR W2
        logger.info("Removed %d players with price = 0", initial_size - len(df_merged))

        # Save merged dataset
        # CAREFUL: only 1 line should be uncommented at a time
        # OUTPUT_PATH = "/files/Capstone_Project_ST/data/processed/players_with_week1.csv" # UNCOMMENT FOR W1
        OUTPUT_PATH = "/files/Capstone_Project_ST/data/processed/players_complete.csv"  # UNCOMMENT FOR W2

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df_merged.to_csv(OUTPUT_PATH, index=False)
        logger.info("Merged data saved to %s", OUTPUT_PATH)

    except FileNotFoundError as e:  # in case the file name is wrong
        logger.error("File not found: %s", e)
        logger.error("Please check that all required CSV files exist")

    except KeyError as e:  # in case the URL column is missing
        logger.error("Column not found during merge: %s", e)
        logger.error("Please check that URL column exists in all files")

    except Exception as e:  # in case any other error happens
        logger.error("Unexpected error during merge: %s", e)
        raise
