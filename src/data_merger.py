"""
Data Merger using Pandas. Will be used for 2 merges:
1. Merge player attributes with week 1 prices
2. Merge player attributes + week 1 prices with week 2 prices
3. Eliminate players with price = 0

IMPORTANT:
For second merge: uncomment lines marked with #W2 and comment lines marked with #W1.
"""

import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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

        #CAREFUL: check comments for W1 and W2
        df_attrs = pd.read_csv(ATTRIBUTES_FILE)
        df_w1 = pd.read_csv(WEEK1_FILE) # ALWAYS UNCOMMENTED
        #df_w2 = pd.read_csv(WEEK2_FILE) # UNCOMMENT FOR W2

        logger.info(f"Players attributes: {len(df_attrs)} rows")
        #CAREFUL: only 1 line should be uncommented at a time
        logger.info(f"Week 1 prices: {len(df_w1)} rows") # UNCOMMENT FOR W1
        #logger.info(f"Week 2 prices: {len(df_w2)} rows") # UNCOMMENT FOR W2

        # merge datasets
        logger.info("Merging datasets")

        #CAREFUL: only 1 line should be uncommented at a time
        df_merged = df_attrs.merge(df_w1.rename(columns={'price': 'price_w1'}), on="url", how="inner") # UNCOMMENT FOR W1
        #df_merged = df_attrs.merge(df_w1, on="url").merge(df_w2, on="url", how="inner", suffixes=('_w1', '_w2')) # UNCOMMENT FOR W2

        logger.info(f"Merged datasets: {len(df_merged)} rows, {len(df_merged.columns)} columns")


        # in case there were any missing values after merge
        missing = df_merged.isnull().sum()
        missing_counts = missing[missing > 0]

        if len(missing_counts) > 0:
            logger.warning(f"There are {missing_counts} missing values after merge")
        else:
            logger.info("There are no missing values")

        logger.info(f"Merged datasets: {len(df_merged)} rows, {len(df_merged.columns)} columns")

        # remove players with price = 0
        logger.info("Removing players with invalid prices")
        initial_size = len(df_merged)
        df_merged = df_merged[df_merged['price_w1'] > 0]  # UNCOMMENT FOR W1
        # df_merged = df_merged[(df_merged['price_w1'] > 0) & (df_merged['price_w2'] > 0)]  # UNCOMMENT FOR W2
        logger.info(f"Removed {initial_size - len(df_merged)} players with price = 0")

        # save merged dataset
        # CAREFUL: only 1 line should be uncommented at a time
        output_path = "/files/Capstone_Project_ST/data/processed/players_with_week1.csv" # UNCOMMENT FOR W1
        #output_path = "/files/Capstone_Project_ST/data/processed/players_complete.csv" # UNCOMMENT FOR W2

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_merged.to_csv(output_path, index=False)
        logger.info(f"Merged data to {output_path}")

    except FileNotFoundError as e: # in case the file name is wrong
        logger.error(f"File not found: {e}")
        logger.error("Please check that all required CSV files exist")
    
    except KeyError as e: # in case the URL column is missing
        logger.error(f"Column not found during merge: {e}")
        logger.error("Please check that URL column exists in all files")
    
    except Exception as e: # in case any other error happens
        logger.error(f"Unexpected error during merge: {e}")
        raise
