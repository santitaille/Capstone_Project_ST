"""
SHAP (SHapley Additive exPlanations) Analysis for XGBoost Model

SHAP values provide advanced model interpretation by showing:
- How each feature contributes to individual predictions
- Feature importance with directionality (positive/negative impact)
- Feature interactions and dependencies

This is the most sophisticated interpretation method and complements:
- OLS coefficients (linear relationships)
- XGBoost feature importance (split-based importance)

SHAP reveals the true complexity of the XGBoost model's decision-making.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import os

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from preprocessing.feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Style configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
shap_importance_path = PROJECT_ROOT / "results" / "tables" / "shap_feature_importance.csv"

def main():
    """Run SHAP analysis"""
    try:
        # Load data
        logger.info("Starting SHAP analysis for XGBoost model")
        df = load_data()
        
        # Prepare TRAIN features (Week 1)
        logger.info("Preparing training features (Week 1)")
        X_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )
        
        # Prepare TEST features (Week 2) - for SHAP analysis
        logger.info("Preparing test features (Week 2)")
        X_test, y_test_log, _, _, _ = prepare_features(
            df,
            target_col="price_w2",
            scaler=scaler,
            club_encoding_map=club_map,
            smoothing=10,
            feature_names=feature_names,
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train XGBoost model (same parameters as xgboost_model.py)
        logger.info("\n=== TRAINING XGBOOST MODEL ===")
        
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
        
        # Calculate SHAP values
        # TreeExplainer is optimized for tree-based models like XGBoost
        logger.info("\n=== CALCULATING SHAP VALUES ===")
        logger.info("This may take 30-60 seconds...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values for test set
        # SHAP values show how each feature affects each prediction
        shap_values = explainer.shap_values(X_test)
        
        logger.info(f"SHAP values calculated: {shap_values.shape}")
        logger.info(f"Base value (average prediction): {explainer.expected_value:.4f}")
        
        
        # PLOT 1: SHAP Summary Plot (Beeswarm)
        # Most important visualization - shows:
        # - Feature importance (top to bottom)
        # - Impact direction (left=negative, right=positive)
        # - Feature values (color: red=high, blue=low)
        logger.info("\n=== CREATING SHAP VISUALIZATIONS ===")
        logger.info("Generating SHAP summary plot (beeswarm)")
        
        plt.figure(figsize=(10, 10))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                         show=False, max_display=20)
        plt.title('SHAP Summary Plot - Feature Impact on Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f'{OUTPUT_DIR}/11_shap_summary_beeswarm.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # PLOT 2: SHAP Bar Plot (Mean Absolute SHAP Values)
        # Shows average magnitude of impact (easier to read than beeswarm)
        logger.info("Generating SHAP bar plot (mean absolute impact)")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                         plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance - Mean Absolute Impact', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Mean |SHAP Value| (Average Impact on Prediction)', fontsize=12)
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/12_shap_bar_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # PLOT 3: SHAP Waterfall Plot for a Single High-Value Prediction
        # Shows how features combine to create one specific prediction
        logger.info("Generating SHAP waterfall plot (individual prediction)")
        
        # Find a high-value player for interesting example
        high_value_idx = y_test_log.values.argmax()
        high_value_player = df.iloc[high_value_idx]
        
        logger.info(f"\nExample player: {high_value_player['player_name']}")
        logger.info(f"  Rating: {high_value_player['rating']}")
        logger.info(f"  Card: {high_value_player['card_category']}")
        logger.info(f"  Actual price (W2): {high_value_player['price_w2']:,.0f} credits")
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[high_value_idx],
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.title(f'SHAP Waterfall - {high_value_player["player_name"]} Prediction Breakdown', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/13_shap_waterfall_example.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # PLOT 4: SHAP Dependence Plot for Rating
        # Shows how rating affects predictions (with interaction effects)
        logger.info("Generating SHAP dependence plot (rating)")
        
        # Get rating feature index
        rating_idx = list(feature_names).index('rating')
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            rating_idx, 
            shap_values, 
            X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title('SHAP Dependence Plot - Rating Impact on Price', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/14_shap_dependence_rating.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # Calculate and save SHAP-based feature importance
        logger.info("\n=== SHAP FEATURE IMPORTANCE RANKING ===")
        
        # Mean absolute SHAP value = feature importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': shap_importance
        }).sort_values('mean_abs_shap', ascending=False)
        
        logger.info("\nTop 15 Features by SHAP Importance:")
        for idx, row in shap_importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['mean_abs_shap']:.4f}")
        
        # Save SHAP importance
        shap_importance_path = "/files/Capstone_Project_ST/results/tables/shap_feature_importance.csv"
        shap_importance_df.to_csv(shap_importance_path, index=False)
        logger.info(f"\nSHAP importance table saved to: {shap_importance_path}")
        
        
        # Compare SHAP vs XGBoost built-in importance
        logger.info("\n=== COMPARISON: SHAP vs XGBoost Importance ===")
        
        xgb_importance = xgb_model.feature_importances_
        comparison_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance,
            'xgb_importance': xgb_importance
        }).sort_values('shap_importance', ascending=False)
        
        logger.info("\nTop 10 Features - SHAP vs XGBoost Importance:")
        logger.info(f"{'Feature':<30} {'SHAP':<10} {'XGBoost':<10}")
        logger.info("-" * 50)
        for idx, row in comparison_df.head(10).iterrows():
            logger.info(f"{row['feature']:<30} {row['shap_importance']:<10.4f} {row['xgb_importance']:<10.4f}")
        
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SHAP ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Created 4 visualizations in {OUTPUT_DIR}/")
        logger.info(f"  - 11_shap_summary_beeswarm.png (Feature impacts - top 20)")
        logger.info(f"  - 12_shap_bar_importance.png (Mean absolute SHAP values)")
        logger.info(f"  - 13_shap_waterfall_example.png (Individual prediction breakdown)")
        logger.info(f"  - 14_shap_dependence_rating.png (Rating effect with interactions)")
        logger.info(f"\nKey Insights:")
        logger.info(f"  - Most important feature: {shap_importance_df.iloc[0]['feature']}")
        logger.info(f"  - SHAP value range: {shap_values.min():.4f} to {shap_values.max():.4f}")
        logger.info(f"  - Model base value: {explainer.expected_value:.4f} (log-price)")
        
    except FileNotFoundError as e:
        # In case the data file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
    
    except KeyError as e:
        # In case required columns are missing
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")
    
    except ImportError as e:
        # In case SHAP is not installed
        logger.error(f"SHAP library not installed: {e}")
        logger.error("Please install with: pip install shap --break-system-packages")
    
    except Exception as e:
        # In case of any other unexpected errors
        logger.error(f"Unexpected error during SHAP analysis: {e}")
        raise


if __name__ == "__main__":
    main()
