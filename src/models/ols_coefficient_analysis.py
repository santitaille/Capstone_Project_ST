"""
OLS Coefficient Analysis using statsmodels for detailed statistical inference.

This provides comprehensive statistical analysis of the relationship between
player attributes and prices, including:
- Coefficient estimates with standard errors
- Statistical significance (p-values)
- Confidence intervals
- Model diagnostics (R², adjusted R², F-statistic)

Complements linear_regression.py (which focuses on prediction) by providing
detailed interpretation and statistical inference.
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

from feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        # Load full dataset
        df = load_data()
        logger.info("Starting OLS coefficient analysis")
        
        # Ensure prices are positive
        if (df["price_w1"] <= 0).any() or (df["price_w2"] <= 0).any():
            raise ValueError("Detected non-positive prices. Clean the data before modeling")
        
        # Prepare features (Week 1 only - we're analyzing relationships, not predicting)
        logger.info("\n=== PREPARING FEATURES (W1) ===")
        X_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train_log shape: {y_train_log.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Convert to numpy arrays for statsmodels
        X_train_np = X_train.values
        y_train_log_np = y_train_log.values
        
        # Add constant (intercept) for OLS
        logger.info("\n=== FITTING OLS MODEL ===")
        X_train_with_const = sm.add_constant(X_train_np)
        
        # Fit OLS regression model
        ols_model = sm.OLS(y_train_log_np, X_train_with_const).fit()
        
        logger.info("OLS model fitted successfully")
        logger.info(f"R²: {ols_model.rsquared:.4f}")
        logger.info(f"Adjusted R²: {ols_model.rsquared_adj:.4f}")
        logger.info(f"F-statistic: {ols_model.fvalue:.2f} (p-value: {ols_model.f_pvalue:.2e})")
        
        # Display full regression summary
        logger.info("\n" + "="*80)
        logger.info("FULL OLS REGRESSION SUMMARY")
        logger.info("="*80)
        print(ols_model.summary())
        
        # Extract coefficient information into clean DataFrame
        logger.info("\n=== COEFFICIENT ANALYSIS ===")
        
        # Create coefficient DataFrame with all statistics
        conf_int = ols_model.conf_int()  

        coef_df = pd.DataFrame({
            'feature': ['intercept'] + list(feature_names),
            'coefficient': ols_model.params,
            'std_error': ols_model.bse,
            't_statistic': ols_model.tvalues,
            'p_value': ols_model.pvalues,
            'ci_lower': conf_int[:, 0],  # 95% confidence interval lower bound
            'ci_upper': conf_int[:, 1],  # 95% confidence interval upper bound
        })
        
        # Standard significance levels: * p<0.05, ** p<0.01, *** p<0.001
        coef_df['significant'] = coef_df['p_value'] < 0.05
        coef_df['sig_level'] = pd.cut(
            coef_df['p_value'],
            bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
            labels=['***', '**', '*', 'ns']
        )
        
        # Sort by absolute coefficient value
        coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
        coef_df_sorted = coef_df.sort_values('abs_coefficient', ascending=False)
        
        # Save full coefficient table
        output_path = "/files/Capstone_Project_ST/results/ols_coefficients.csv"
        coef_df_sorted.to_csv(output_path, index=False)
        logger.info(f"\nFull coefficient table saved to: {output_path}")
        
        # Display top features driving price up
        logger.info("\n" + "="*80)
        logger.info("TOP 10 FEATURES DRIVING PRICE UP (Positive Coefficients)")
        logger.info("="*80)
        positive_coef = coef_df_sorted[coef_df_sorted['coefficient'] > 0].head(15)
        for idx, row in positive_coef.iterrows():
            logger.info(f"  {row['feature']:30s}: {row['coefficient']:7.4f} {row['sig_level']:3s} "
                       f"(p={row['p_value']:.4f}, CI=[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])")
        
        # Display top features driving price down
        logger.info("\n" + "="*80)
        logger.info("TOP 10 FEATURES DRIVING PRICE DOWN (Negative Coefficients)")
        logger.info("="*80)
        negative_coef = coef_df_sorted[coef_df_sorted['coefficient'] < 0].head(15)
        for idx, row in negative_coef.iterrows():
            logger.info(f"  {row['feature']:30s}: {row['coefficient']:7.4f} {row['sig_level']:3s} "
                       f"(p={row['p_value']:.4f}, CI=[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])")

        # Summary statistics
        logger.info("\n=== SUMMARY STATISTICS ===")
        logger.info(f"Total features analyzed: {len(feature_names)}")
        logger.info(f"Significant features (p < 0.05): {coef_df['significant'].sum() - 1}")
        logger.info(f"Non-significant features: {(~coef_df['significant']).sum()}")
        logger.info(f"Model R²: {ols_model.rsquared:.4f}")
        logger.info(f"Adjusted R²: {ols_model.rsquared_adj:.4f}")
    
    except FileNotFoundError as e:
        # In case the data file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
    
    except KeyError as e:
        # In case required columns are missing from the dataset
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")
    
    except Exception as e:
        # In case of any other unexpected errors
        logger.error(f"Unexpected error during OLS analysis: {e}")
        raise