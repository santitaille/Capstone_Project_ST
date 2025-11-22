"""
Data Merger using Pandas. Will be used for 2 merges:
1. Merge player attributes with week 1 prices
2. Merge player attributes + week 1 prices with week 2 prices

IMPORTANT:
For second merge: uncomment lines marked with #W2 and comment lines marked with #W1.
"""

import pandas as pd

# Load CSV files
print("Loading CSV files")
df_attrs = pd.read_csv("/files/Capstone_Project_ST/data/players_attributes.csv")

#CAREFUL: only 1 line should be uncommented at a time
df_w1 = pd.read_csv("/files/Capstone_Project_ST/data/week1/prices_week1.csv") # ALWAYS UNCOMMENTED
df_w2 = pd.read_csv("/files/Capstone_Project_ST/data/week2/prices_week2.csv") # UNCOMMENT FOR W2

print(f"Players attributes: {len(df_attrs)} rows")
#print(f"Week 1 prices: {len(df_w1)} rows") # UNCOMMENT FOR W1
print(f"Week 2 prices: {len(df_w2)} rows") # UNCOMMENT FOR W2

# Merge datasets
print("\nMerging datasets")

#CAREFUL: only 1 line should be uncommented at a time
#df_merged = df_attrs.merge(df_w1.rename(columns={'price': 'price_w1'}), on="url", how="inner") # UNCOMMENT FOR W1
df_merged = df_attrs.merge(df_w1, on="url").merge(df_w2, on="url", how="inner", suffixes=('_w1', '_w2')) # UNCOMMENT FOR W2

print(f"Merged datasets: {len(df_merged)} rows, {len(df_merged.columns)} columns")


# In case there were any missing values after merge
print("\nMissing values:")
missing = df_merged.isnull().sum()
missing_counts = missing[missing > 0]
if len(missing_counts) > 0:
    print(missing_counts)
else:
    print("No missing values")

# Save merged dataset
#CAREFUL: only 1 line should be uncommented at a time
#output_path = "/files/Capstone_Project_ST/data/processed/players_with_week1.csv" # UNCOMMENT FOR W1
output_path = "/files/Capstone_Project_ST/data/processed/players_complete.csv" # UNCOMMENT FOR W2

df_merged.to_csv(output_path, index=False)
print(f"\nMerged data to {output_path}")