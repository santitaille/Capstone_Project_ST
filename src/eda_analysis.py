"""
Exploratory Data Analysis using Pandas and Matplotlib. Using week1 prices merged dataset.

Will analyze:
1. Price distribution (normal and log scale)
2. Rating vs Price
3. Price by Card Category
4. Price by Position Cluster
5. Correlation Matrix
6. Top 4 Attributes Combined
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
print("\nprice statistics (for week 1 prices):")
print(df['price_w1'].describe())

#PRICE DISTRIBUTION (NORMAL AND LOG SCALE)
print("\nGenerating price distribution visualization")
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

# save price distribution
plt.tight_layout()
plt.savefig('/files/Capstone_Project_ST/results/figures/01_price_distribution.png', dpi=300, bbox_inches='tight')
print("Saved to results/figures/01_price_distribution.png")
plt.close()


#RATING VS PRICE
print("\nGenerating rating vs price visualization")
plt.figure(figsize=(12, 6))
plt.scatter(df['rating'], df['price_w1'], alpha=0.5, s=30)
plt.xlabel('Overall Rating (OVR)', fontsize=12)
plt.ylabel('Price (Week 1)', fontsize=12)
plt.title('Rating vs Price', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()

# save rating vs price
plt.savefig('/files/Capstone_Project_ST/results/figures/02_rating_vs_price.png', dpi=300, bbox_inches='tight')
print("Saved to results/figures/02_rating_vs_price.png")
plt.close()

# correlation between rating and price
corr_rating = df[['rating', 'price_w1']].corr().iloc[0, 1]
print(f"Correlation between rating and price: {corr_rating:.3f}")


#PRICE BY CARD CATEGORY
print("\nGenerating price by card category visualization")
plt.figure(figsize=(10, 6))
df.boxplot(column='price_w1', by='card_category', ax=plt.gca())
plt.title('Price by Card Category', fontsize=14, fontweight='bold')
plt.suptitle('')  # remove default title
plt.ylabel('Price (Week 1)', fontsize=12)
plt.xlabel('Card Category', fontsize=12)
plt.xticks(rotation=0)
plt.grid(alpha=0.3)
plt.tight_layout()

# save price by card category
plt.savefig('/files/Capstone_Project_ST/results/figures/03_price_by_category.png', dpi=300, bbox_inches='tight')
print("Saved: results/figures/03_price_by_category.png")
plt.close()

# average price by category
print("Average price by card category:")
category_stats = df.groupby('card_category')['price_w1'].agg(['mean', 'median', 'count'])
print(category_stats)


#PRICE BY POSITION CLUSTER
print("\nGenerating price by position cluster visualization")
position_cols = ['cluster_cb', 'cluster_fullbacks', 'cluster_mid', 'cluster_att_mid', 'cluster_st']
position_names = ['Center Backs', 'Fullbacks', 'Midfielders', 'Attacking Midfielders', 'Strikers']

position_prices = []
for col in position_cols:
    if col in df.columns:
        players_in_pos = df[df[col] == 1]
        position_prices.append(players_in_pos['price_w1'].mean())
    else:
        position_prices.append(0)

plt.figure(figsize=(10, 6))
bars = plt.bar(position_names, position_prices, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Position Cluster', fontsize=12)
plt.ylabel('Average Price (Week 1)', fontsize=12)
plt.title('Average Price by Position Cluster', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# add value labels on bars
for bar, price in zip(bars, position_prices):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{price:,.0f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('/files/Capstone_Project_ST/results/figures/04_price_by_position.png', dpi=300, bbox_inches='tight')
print("Saved to results/figures/04_price_by_position.png")
plt.close()


#CORRELATION MATRIX
print("\nGenerating correlation matrix")
numeric_cols = ['rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical', 'skill_moves', 'weak_foot', 'num_playstyles','num_playstyles_plus', 'num_positions', 'price_w1']

# filter to only existing columns
numeric_cols = [col for col in numeric_cols if col in df.columns]

corr_matrix = df[numeric_cols].corr()
price_corr = corr_matrix['price_w1'].sort_values(ascending=False)

print(f'Correlation with price:')
print(price_corr)

# correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Player Attributes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/files/Capstone_Project_ST/results/figures/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved to results/figures/05_correlation_matrix.png")
plt.close()


#TOP 4 ATTRIBUTES COMBINED
print("\nGenerating top 4 attributes combined visualization")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pace vs Price
axes[0, 0].scatter(df['pace'], df['price_w1'], alpha=0.5, s=20, color='green')
axes[0, 0].set_xlabel('Pace', fontsize=11)
axes[0, 0].set_ylabel('Price (Week 1)', fontsize=11)
axes[0, 0].set_title('Pace vs Price', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Dribbling vs Price
axes[0, 1].scatter(df['dribbling'], df['price_w1'], alpha=0.5, s=20, color='orange')
axes[0, 1].set_xlabel('Dribbling', fontsize=11)
axes[0, 1].set_ylabel('Price (Week 1)', fontsize=11)
axes[0, 1].set_title('Dribbling vs Price', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Skill Moves vs Price
axes[1, 0].scatter(df['skill_moves'], df['price_w1'], alpha=0.5, s=20, color='purple')
axes[1, 0].set_xlabel('Skill Moves', fontsize=11)
axes[1, 0].set_ylabel('Price (Week 1)', fontsize=11)
axes[1, 0].set_title('Skill Moves vs Price', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Weak Foot vs Price
axes[1, 1].scatter(df['weak_foot'], df['price_w1'], alpha=0.5, s=20, color='red')
axes[1, 1].set_xlabel('Weak Foot', fontsize=11)
axes[1, 1].set_ylabel('Price (Week 1)', fontsize=11)
axes[1, 1].set_title('Weak Foot vs Price', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# save top 4 attributes
plt.tight_layout()
plt.savefig('/files/Capstone_Project_ST/results/figures/06_top4_attributes.png', dpi=300, bbox_inches='tight')
print("Saved to results/figures/06_top4_attributes.png")
plt.close()


# saved plots summary
print(f"\nGenerated 6 visualizations in results/figures/")

# key statistics summary
print(f"\nDataset: {len(df)} players")
print(f"Price range: {df['price_w1'].min():,.0f} - {df['price_w1'].max():,.0f} credits")
print(f"Strongest correlation with price: {price_corr.index[1]} ({price_corr.iloc[1]:.3f})")