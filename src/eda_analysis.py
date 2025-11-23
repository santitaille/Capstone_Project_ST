"""
Exploratory Data Analysis using Pandas and Matplotlib. Using week1 prices merged dataset.

Will analyze:
1. Price distribution (normal and log scale)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# style configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# data loading
df = pd.read_csv("/files/Capstone_Project_ST/data/processed/players_with_week1.csv")

print(f"Dataset loaded: {df.shape[0]} players, {df.shape[1]} columns")

# basic price statistics
print("\nPrice Statistics (Week 1):")
print(df['price_w1'].describe())

# price distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# normal scale
axes[0].hist(df['price_w1'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Price (Week 1)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# log scale because many player prices are low
axes[1].hist(df['price_w1'], bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Price (Week 1)', fontsize=12)
axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
axes[1].set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

# save figues
plt.tight_layout()
plt.savefig('/files/Capstone_Project_ST/results/figures/01_price_distribution.png', dpi=300, bbox_inches='tight')
print("\nAnalysis visualization to results/figures/01_price_distribution.png")
plt.close()

print("Price distribution analysis completed")
