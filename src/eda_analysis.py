"""
Exploratory Data Analysis for EA FC 26 Player Price Prediction

Generates 6 visualizations exploring relationships between player attributes and prices:
1. Price distribution (normal and log scale)
2. Rating vs Price scatter
3. Price by Card Category boxplot
4. Price by Position Cluster bar chart
5. Correlation Matrix heatmap
6. Top 4 Attributes vs Price (4-panel scatter)
"""
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Style configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# configuration
DATA_FILE = "/files/Capstone_Project_ST/data/processed/players_with_week1.csv"
OUTPUT_DIR = "/files/Capstone_Project_ST/results/figures"

if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv(DATA_FILE)
        logger.info("Loaded %d players", len(df))
        print()

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        logger.info("Generating 6 EDA visualizations...")
        print()

        # VISUZALIZATION 1: PRICE DISTRIBUTION (NORMAL AND LOG SCALE)
        # To see overall price distribution and log-scale due to skewness
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Normal scale
        axes[0].hist(df['price_w1'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Price (Week 1)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # Log scale as data is right-skewed
        axes[1].hist(df['price_w1'], bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Price (Week 1)', fontsize=12)
        axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
        axes[1].set_title('Price Distribution (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)

        # Save visualization
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/01_price_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("→ 01_price_distribution.png")
        plt.close()


        #VISUALIZATION 2: RATING VS PRICE
        # To see how player rating affects price
        plt.figure(figsize=(12, 6))
        for category in df['card_category'].unique():
            subset = df[df['card_category'] == category]
            plt.scatter(subset['rating'], subset['price_w1'],
                    label=category, alpha=0.6, s=30)
        plt.xlabel('Overall Rating (OVR)', fontsize=12)
        plt.ylabel('Price (Week 1)', fontsize=12)
        plt.title('Rating vs Price by Card Category', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save visualization
        plt.savefig(f'{OUTPUT_DIR}/02_rating_vs_price.png', dpi=300, bbox_inches='tight')
        logger.info("→ 02_rating_vs_price.png")
        plt.close()


        # VISUALIZATION 3: PRICE BY CARD CATEGORY
        # To see how different card categories affect price
        plt.figure(figsize=(10, 6))
        df.boxplot(column='price_w1', by='card_category', ax=plt.gca())
        plt.title('Price by Card Category', fontsize=14, fontweight='bold')
        plt.suptitle('')  # remove default title
        plt.ylabel('Price (Week 1)', fontsize=12)
        plt.xlabel('Card Category', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save visualization
        plt.savefig(f'{OUTPUT_DIR}/03_price_by_category.png', dpi=300, bbox_inches='tight')
        logger.info("→ 03_price_by_category.png")
        plt.close()


        # VISUALIZATION 4: PRICE BY POSITION CLUSTER
        # To see average price by position clusters with error bars for std deviation
        position_cols = ['cluster_cb', 'cluster_fullbacks', 'cluster_mid',
            'cluster_att_mid', 'cluster_st']
        position_names = ['Center Backs', 'Fullbacks', 'Midfielders',
            'Attacking Midfielders', 'Strikers']
        position_prices = []
        for col in position_cols:
            if col in df.columns:
                players_in_pos = df[df[col] == 1]
                position_prices.append(players_in_pos['price_w1'].mean())
            else:
                position_prices.append(0)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(position_names, position_prices, color='steelblue',
            edgecolor='black', alpha=0.7)
        plt.xlabel('Position Cluster', fontsize=12)
        plt.ylabel('Average Price (Week 1)', fontsize=12)
        plt.title('Average Price by Position Cluster', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars for better readability
        for bar, price in zip(bars, position_prices):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{price:,.0f}',
                     ha='center', va='bottom', fontsize=10)

        # Save visualization
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/04_price_by_position.png', dpi=300, bbox_inches='tight')
        logger.info("→ 04_price_by_position.png")
        plt.close()


        # VISUALIZATION 5: CORRELATION MATRIX
        # To see correlations between numeric attributes and price
        numeric_cols = ['rating', 'pace', 'shooting', 'passing', 'dribbling',
                       'defending', 'physical', 'skill_moves', 'weak_foot', 
                       'num_playstyles', 'num_playstyles_plus', 'num_positions', 'price_w1']   

        # Ensure columns exist
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[numeric_cols].corr()
        price_corr = corr_matrix['price_w1'].sort_values(ascending=False)

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Player Attributes', fontsize=14, fontweight='bold')

        # Save visualization
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("→ 05_correlation_matrix.png")
        plt.close()


        # VISUALIZATION 6: TOP 4 ATTRIBUTES COMBINED VS PRICE
        # To see relationships of top 4 correlated attributes with price
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

        # Save visualization
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/06_top4_attributes.png', dpi=300, bbox_inches='tight')
        logger.info("→ 06_top4_attributes.png")
        plt.close()


        # SUMMARY
        print()
        logger.info("EDA complete: 6 visualizations saved to results/figures/")
        logger.info("Price range: %s - %s credits",
                   f"{df['price_w1'].min():,.0f}",
                   f"{df['price_w1'].max():,.0f}")
        logger.info("Strongest correlation with price: %s (r = %.3f)",
                   price_corr.index[1], price_corr.iloc[1])
        print()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Check that players_with_week1.csv exists in data/processed/")

    except KeyError as e:
        logger.error("Column not found: %s", e)
        logger.error("Check that required columns exist in dataset")

    except Exception as e:
        logger.error("Unexpected error during EDA: %s", e)
        raise
