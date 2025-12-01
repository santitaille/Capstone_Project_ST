"""
Underpriced and Overpriced Players Detector f

Uses XGBoost predictions to identify players whose actual market price
differs significantly from their predicted value, revealing potential
trading opportunities.

- Underpriced: Actual price < Predicted price (potential buys)
- Overpriced: Actual price > Predicted price (potential sells)

This completes the second stretch goal from the project proposal.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")

# Configuration
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "predictions" / "predictions_xgboost_w2.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main():
    """Run Market Inefficiencies"""
    try:
        logger.info("Starting market inefficiency detection")
        
        # Load XGBoost predictions (best model)
        df = pd.read_csv(PREDICTIONS_FILE)
        logger.info(f"Loaded {len(df)} player predictions")
        
        # Calculate price difference
        df['price_diff'] = df['pred_price_w2'] - df['price_w2']
        df['price_diff_pct'] = (df['price_diff'] / df['price_w2']) * 100
        df['abs_price_diff'] = df['price_diff'].abs()
        
        # Classify players
        # Threshold: >20% difference = significant mispricing
        THRESHOLD = 20
        
        df['status'] = 'Fair Value'
        df.loc[df['price_diff_pct'] > THRESHOLD, 'status'] = 'Underpriced'
        df.loc[df['price_diff_pct'] < -THRESHOLD, 'status'] = 'Overpriced'
        
        # Summary statistics
        logger.info("\nMARKET INEFFICIENCY SUMMARY")
        logger.info(f"Total players analyzed: {len(df)}")
        logger.info(f"Underpriced (>20%): {len(df[df['status'] == 'Underpriced'])} players")
        logger.info(f"Overpriced (>20%): {len(df[df['status'] == 'Overpriced'])} players")
        logger.info(f"Fair value (±20%): {len(df[df['status'] == 'Fair Value'])} players")
        
        # Get top opportunities for summary
        underpriced = df[df['status'] == 'Underpriced'].nlargest(10, 'price_diff_pct')
        overpriced = df[df['status'] == 'Overpriced'].nsmallest(10, 'price_diff_pct')
        
        logger.info("\nGenerating visualizations")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot by status
        colors = {'Underpriced': 'green', 'Fair Value': 'gray', 'Overpriced': 'red'}
        for status in ['Fair Value', 'Underpriced', 'Overpriced']:
            subset = df[df['status'] == status]
            ax.scatter(subset['price_w2'], subset['pred_price_w2'], 
                      c=colors[status], alpha=0.6, s=30, label=status)
        
        # Perfect prediction line
        max_price = max(df['price_w2'].max(), df['pred_price_w2'].max())
        ax.plot([0, max_price], [0, max_price], 'k--', linewidth=2, alpha=0.5, label='Perfect Prediction')
        
        # 20% bands
        ax.fill_between([0, max_price], [0, max_price * 0.8], [0, max_price * 1.2], 
                       alpha=0.1, color='gray', label='±20% Band')
        
        ax.set_xlabel('Actual Price W2 (credits)', fontsize=12)
        ax.set_ylabel('Predicted Price W2 (credits)', fontsize=12)
        ax.set_title('Market Inefficiency Detection: Actual vs Predicted Prices', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Log scale for better visibility
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        os.makedirs(FIGURES_DIR, exist_ok=True)
        output_path = f'{FIGURES_DIR}/16_market_inefficiency_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # VISUALIZATION 2: Top Opportunities Bar Chart
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 15 underpriced
        top_underpriced = df[df['status'] == 'Underpriced'].nlargest(15, 'price_diff_pct')
        axes[0].barh(range(len(top_underpriced)), top_underpriced['price_diff_pct'], 
                    color='green', alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(top_underpriced)))
        axes[0].set_yticklabels(top_underpriced['player_name'], fontsize=9)
        axes[0].set_xlabel('Underpriced % (Predicted - Actual)', fontsize=11)
        axes[0].set_title('Top 15 Underpriced Players (Best Buys)', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for idx, val in enumerate(top_underpriced['price_diff_pct']):
            axes[0].text(val + 2, idx, f'+{val:.1f}%', va='center', fontsize=8)
        
        # Top 15 overpriced
        top_overpriced = df[df['status'] == 'Overpriced'].nsmallest(15, 'price_diff_pct')
        axes[1].barh(range(len(top_overpriced)), top_overpriced['price_diff_pct'].abs(), 
                    color='red', alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(top_overpriced)))
        axes[1].set_yticklabels(top_overpriced['player_name'], fontsize=9)
        axes[1].set_xlabel('Overpriced % (Actual - Predicted)', fontsize=11)
        axes[1].set_title('Top 15 Overpriced Players (Avoid/Sell)', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for idx, val in enumerate(top_overpriced['price_diff_pct'].abs()):
            axes[1].text(val + 2, idx, f'+{val:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        
        output_path = f'{FIGURES_DIR}/17_trading_opportunities.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # Save full results
        df_output = df[['player_name', 'rating', 'card_category', 'price_w2', 
                       'pred_price_w2', 'price_diff', 'price_diff_pct', 'status']]
        df_output = df_output.sort_values('price_diff_pct', ascending=False)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f'{OUTPUT_DIR}/market_inefficiencies.csv'
        df_output.to_csv(output_path, index=False)
        logger.info(f"\nFull results saved to: {output_path}")
        
        
        # FINAL SUMMARY
        logger.info("\nSTRETCH GOAL 2 COMPLETE")
        logger.info(f"Identified {len(df[df['status'] != 'Fair Value'])} mispriced players")
        logger.info(f"Best buying opportunity: {underpriced.iloc[0]['player_name']} "
                   f"({underpriced.iloc[0]['price_diff_pct']:.1f}% underpriced)")
        logger.info(f"Most overpriced: {overpriced.iloc[0]['player_name']} "
                   f"({abs(overpriced.iloc[0]['price_diff_pct']):.1f}% overpriced)")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure XGBoost predictions CSV exists")
        logger.error("Run xgboost_model.py first to generate predictions")
    
    except Exception as e:
        logger.error(f"Error during market inefficiency detection: {e}")
        raise

if __name__ == "__main__":
    main()