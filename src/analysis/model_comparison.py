"""
Final Model Comparison

Compares all 6 methods on Week 2 test set:
- 2 Baselines (median-based)
- 4 Machine Learning models

Creates:
1. Comparison table (CSV)
2. Comparison visualization (bar chart)
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

def main():
    """Run Model Comparison"""
    try:
        logger.info("Starting final model comparison")
        
        # Manual results from all models (Week 2 test set)
        results = {
            'Model': [
                'Baseline 1: Median by OVR',
                'Baseline 2: Median by OVR x Category',
                'Linear Regression',
                'Neural Network (MLP)',
                'Random Forest',
                'XGBoost'
            ],
            'R²': [
                0.6078,   # Baseline 1
                0.6329,   # Baseline 2
                0.6315,   # Linear Regression
                0.6690,   # Neural Network
                0.8596,   # Random Forest
                0.9558    # XGBoost
            ],
            'RMSE': [
                410874,   # Baseline 1
                397507,   # Baseline 2
                398273,   # Linear Regression
                377422,   # Neural Network
                245831,   # Random Forest
                137993    # XGBoost
            ],
            'MAE': [
                142364,   # Baseline 1
                129361,   # Baseline 2
                116050,   # Linear Regression
                92889,    # Neural Network
                63961,    # Random Forest
                42997     # XGBoost
            ]
        }
        
        df = pd.DataFrame(results)
        
        # Add improvement over best baseline
        best_baseline_r2 = 0.6329
        df['Improvement_over_Baseline'] = ((df['R²'] - best_baseline_r2) / best_baseline_r2 * 100).round(1)
        
        # Sort by R² descending 
        df = df.sort_values('R²', ascending=False).reset_index(drop=True)
        
        # Display table
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON - WEEK 2 TEST SET")
        logger.info("="*80)
        print(df.to_string(index=False))
        
        # Save table
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f"{OUTPUT_DIR}/model_comparison.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\nComparison table saved to: {output_path}")
        
        
        # VISUALIZATION: R² Comparison Bar Chart
        logger.info("\nGenerating comparison visualization")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code: baselines vs ML models
        colors = ['darkgreen', 'steelblue', 'steelblue', 'gray', 'steelblue', 'orange']
        
        bars = ax.barh(df['Model'], df['R²'], color=colors, alpha=0.8, edgecolor='black')
        
        # Add R² values on bars
        for idx, (bar, r2) in enumerate(zip(bars, df['R²'])):
            ax.text(r2 + 0.01, idx, f'{r2:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        # Add baseline reference line
        ax.axvline(x=best_baseline_r2, color='red', linestyle='--', 
                  linewidth=2, alpha=0.5, label='Best Baseline')
        
        ax.set_xlabel('R² (Coefficient of Determination)', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title('Model Comparison - Week 2 Test Set Performance', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        # Invert y-axis so best is on top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save
        os.makedirs(FIGURES_DIR, exist_ok=True)
        viz_path = f"{FIGURES_DIR}/15_model_comparison.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {viz_path}")
        plt.close()
        
        
        # SUMMARY
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Best Model: {df.iloc[0]['Model']}")
        logger.info(f"  R²: {df.iloc[0]['R²']:.4f}")
        logger.info(f"  RMSE: {df.iloc[0]['RMSE']:,.0f} credits")
        logger.info(f"  Improvement over baseline: +{df.iloc[0]['Improvement_over_Baseline']:.1f}%")
        logger.info(f"\nModels beating baseline: {len(df[df['R²'] > best_baseline_r2])} out of 6")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        raise

if __name__ == "__main__":
    main()