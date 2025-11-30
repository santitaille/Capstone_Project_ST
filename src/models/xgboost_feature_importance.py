"""
XGBoost Feature Importance Visualization

Extracts and visualizes feature importance from the trained XGBoost model.
Shows which features the model considers most important for predictions.

This complements the OLS coefficient analysis by showing nonlinear feature importance
from the best-performing model (XGBoost R² = 0.956).

Re-trains the XGBoost model to extract feature importances, then creates visualizations.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os

from feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Style configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
OUTPUT_DIR = "/files/Capstone_Project_ST/results/figures"

if __name__ == "__main__":
    try:
        # Load data
        logger.info("Starting XGBoost feature importance visualization")
        df = load_data()
        
        # Prepare features (Week 1 for training)
        logger.info("Preparing features (Week 1)")
        X_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )
        
        logger.info(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")
        
        # Train XGBoost model (same parameters as xgboost_model.py)
        logger.info("Training XGBoost model to extract feature importances")
        
        xgb_model = xgb.XGBRegressor(
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
            verbosity=0
        )
    
        xgb_model.fit(X_train, y_train_log)
        logger.info("XGBoost model trained")

        # Extract feature importances
        importance_gain = xgb_model.feature_importances_
        
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_gain
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Most Important Features (by gain):")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        # Save importance table
        os.makedirs(OUTPUT_DIR.replace('/figures', ''), exist_ok=True)
        importance_path = "/files/Capstone_Project_ST/results/xgboost_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"\nFeature importance table saved to: {importance_path}")
        
        
        # VISUALIZATION 1: Top 20 Features Bar Chart
        logger.info("\nGenerating feature importance bar chart (top 20)")
        
        top20 = importance_df.head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top20)), top20['importance'], 
                      color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for idx, (bar, importance) in enumerate(zip(bars, top20['importance'])):
            ax.text(importance + 0.002, idx, f'{importance:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Labels and formatting
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'])
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('XGBoost Feature Importance (Top 20 Features)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Invert y-axis so highest importance is at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f'{OUTPUT_DIR}/10_xgboost_feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # VISUALIZATION 2: Feature Importance with Cumulative Percentage
        logger.info("\nGenerating cumulative importance plot")
        
        # Calculate cumulative importance
        importance_df['cumulative'] = importance_df['importance'].cumsum()
        importance_df['cumulative_pct'] = (importance_df['cumulative'] / 
                                           importance_df['importance'].sum() * 100)
        
        # Get top features that explain 80% of importance
        top_features_80 = importance_df[importance_df['cumulative_pct'] <= 80]
        logger.info(f"\nTop {len(top_features_80)} features explain 80% of total importance")
        
        # Plot top 25 features with cumulative line
        top25 = importance_df.head(25)
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Bar chart for importance
        x_pos = np.arange(len(top25))
        bars = ax1.bar(x_pos, top25['importance'], 
                      color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Features (ranked by importance)', fontsize=12)
        ax1.set_ylabel('Feature Importance (Gain)', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(top25['feature'], rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Cumulative line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x_pos, top25['cumulative_pct'], 
                color='red', marker='o', linewidth=2, markersize=4, label='Cumulative %')
        ax2.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Cumulative Importance (%)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 105])
        
        # Add 80% reference text
        ax2.text(len(top25)-1, 82, '80%', fontsize=10, color='red', fontweight='bold')
        
        plt.title('XGBoost Feature Importance with Cumulative Percentage', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/10b_xgboost_cumulative_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("XGBOOST FEATURE IMPORTANCE VISUALIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Created 2 visualizations in {OUTPUT_DIR}/")
        logger.info(f"  - 10_xgboost_feature_importance.png (Top 20 features)")
        logger.info(f"  - 10b_xgboost_cumulative_importance.png (Top 25 with cumulative %)")
        logger.info(f"\nTop 3 most important features:")
        for idx, row in importance_df.head(3).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f} "
                       f"({row['cumulative_pct']:.1f}% cumulative)")
        
    except FileNotFoundError as e:
        # In case the data file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
    
    except KeyError as e:
        # In case required columns are missing
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")
    
    except Exception as e:
        # In case of any other unexpected errors
        logger.error(f"Unexpected error during XGBoost feature importance visualization: {e}")
        raise