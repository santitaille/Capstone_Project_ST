"""
OLS Coefficient Visualizations

Creates visualizations from the OLS coefficient analysis results:
1. Coefficient plot with confidence intervals (top 20 features)
2. Top features bar chart (top 15 positive + top 5 negative)
3. Residuals diagnostics (4-panel plot)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import os
import statsmodels.api as sm
from scipy import stats

from feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Style configurations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
INPUT_FILE = "/files/Capstone_Project_ST/results/ols_coefficients.csv"
OUTPUT_DIR = "/files/Capstone_Project_ST/results/figures"

if __name__ == "__main__":
    try:
        # Load OLS coefficients
        logger.info("Starting OLS visualizations")
        logger.info(f"Loading coefficients from {INPUT_FILE}")
        
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"Loaded {len(df)} coefficients")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Remove intercept for visualization (not meaningful to compare with features)
        df_features = df[df['feature'] != 'intercept'].copy()
        
        
        # PLOT 1: COEFFICIENT PLOT WITH CONFIDENCE INTERVALS (Top 20)
        logger.info("\nGenerating coefficient plot with confidence intervals")
        
        # Get top 20 features by absolute coefficient value
        top20 = df_features.nlargest(20, 'abs_coefficient').copy()
        # Sort by coefficient value (not absolute) for better visualization
        top20 = top20.sort_values('coefficient')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot confidence intervals as horizontal lines
        for idx, row in top20.iterrows():
            y_pos = range(len(top20)).__getitem__(top20.index.get_loc(idx))
            
            # Color based on significance
            if row['sig_level'] == '***':
                color = 'darkgreen'
                alpha = 1.0
            elif row['sig_level'] == '**':
                color = 'green'
                alpha = 0.8
            elif row['sig_level'] == '*':
                color = 'orange'
                alpha = 0.7
            else:
                color = 'gray'
                alpha = 0.5
            
            # Plot CI line
            ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], 
                   color=color, alpha=alpha, linewidth=2)
            
            # Plot coefficient point
            ax.scatter(row['coefficient'], y_pos, color=color, s=100, 
                      zorder=3, alpha=alpha)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        
        # Labels and formatting
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'])
        ax.set_xlabel('Coefficient (with 95% Confidence Interval)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('OLS Coefficients with Confidence Intervals (Top 20 Features)', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='darkgreen', lw=2, label='*** (p < 0.001)'),
            Line2D([0], [0], color='green', lw=2, label='** (p < 0.01)'),
            Line2D([0], [0], color='orange', lw=2, label='* (p < 0.05)'),
            Line2D([0], [0], color='gray', lw=2, label='ns (not significant)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/07_ols_coefficient_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # PLOT 2: TOP FEATURES BAR CHART (Top 15 positive + Top 5 negative)
        logger.info("\nGenerating top features bar chart")
        
        # Get top positive and negative features
        positive_features = df_features[df_features['coefficient'] > 0].nlargest(15, 'coefficient')
        negative_features = df_features[df_features['coefficient'] < 0].nsmallest(5, 'coefficient')
        
        # Combine
        combined = pd.concat([positive_features, negative_features])
        combined = combined.sort_values('coefficient')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create bar chart
        colors = ['red' if c < 0 else 'steelblue' for c in combined['coefficient']]
        bars = ax.barh(range(len(combined)), combined['coefficient'], 
                      color=colors, alpha=0.7, edgecolor='black')
        
        # Add significance stars
        for idx, (bar, row) in enumerate(zip(bars, combined.itertuples())):
            # Add value label
            x_pos = row.coefficient
            if x_pos > 0:
                ha = 'left'
                x_pos += 0.05
            else:
                ha = 'right'
                x_pos -= 0.05
            
            label = f"{row.coefficient:.2f} {row.sig_level}"
            ax.text(x_pos, idx, label, va='center', ha=ha, fontsize=9, fontweight='bold')
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Labels and formatting
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined['feature'])
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top Features Driving Player Prices (OLS Regression)', 
                    fontsize=14, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/08_ols_top_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # PLOT 3: RESIDUALS DIAGNOSTICS (4-panel plot)
        logger.info("\nGenerating residuals diagnostics (re-fitting model)")
        
        # Need to re-fit OLS model to get residuals
        # Load data and prepare features (same as ols_coefficient_analysis.py)
        df_data = load_data()
        X_train, y_train_log, _, _, feature_names_model = prepare_features(
            df_data,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )
        
        # Fit OLS model
        X_train_np = X_train.values
        y_train_log_np = y_train_log.values
        X_train_with_const = sm.add_constant(X_train_np)
        ols_model = sm.OLS(y_train_log_np, X_train_with_const).fit()
        
        # Get residuals and fitted values
        fitted_values = ols_model.fittedvalues
        residuals = ols_model.resid
        standardized_residuals = residuals / np.sqrt(np.var(residuals))
        
        logger.info("Model re-fitted, creating diagnostic plots")
        
        # Create 2x2 subplot for diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Residuals vs Fitted
        axes[0, 0].scatter(fitted_values, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values (Log-Price)', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Add lowess smoothing line to check for patterns
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted_values, frac=0.3)
        axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, label='LOWESS')
        axes[0, 0].legend(fontsize=9)
        
        # Plot 2: Q-Q Plot (Normal Probability Plot)
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Scale-Location (Spread-Location)
        axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=20)
        axes[1, 0].set_xlabel('Fitted Values (Log-Price)', fontsize=11)
        axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 0].set_title('Scale-Location', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Add lowess smoothing line
        smoothed_scale = lowess(np.sqrt(np.abs(standardized_residuals)), fitted_values, frac=0.3)
        axes[1, 0].plot(smoothed_scale[:, 0], smoothed_scale[:, 1], color='blue', linewidth=2)
        
        # Plot 4: Residuals Histogram
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Histogram of Residuals', fontsize=12, fontweight='bold')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].grid(alpha=0.3)
        
        # Add normal curve overlay
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 1].plot(x, len(residuals) * (residuals.max() - residuals.min()) / 50 * 
                       stats.norm.pdf(x, mu, sigma), 
                       'r-', linewidth=2, label='Normal Distribution')
        axes[1, 1].legend(fontsize=9)
        
        plt.suptitle('OLS Regression Diagnostics', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        output_path = f'{OUTPUT_DIR}/09_ols_residuals_diagnostics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved to {output_path}")
        plt.close()
        
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("OLS VISUALIZATIONS COMPLETE")
        logger.info("="*80)
        logger.info(f"Created 3 visualizations in {OUTPUT_DIR}/")
        logger.info(f"  - 07_ols_coefficient_plot.png (Top 20 with confidence intervals)")
        logger.info(f"  - 08_ols_top_features.png (Top 15 positive + 5 negative)")
        logger.info(f"  - 09_ols_residuals_diagnostics.png (4-panel diagnostics)")
        
    except FileNotFoundError as e:
        # In case the CSV file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please run ols_coefficient_analysis.py first to generate the CSV")
    
    except KeyError as e:
        # In case required columns are missing
        logger.error(f"Column not found: {e}")
        logger.error("Please check that the CSV has the correct columns")
    
    except Exception as e:
        # In case of any other unexpected errors
        logger.error(f"Unexpected error during visualization: {e}")
        raise