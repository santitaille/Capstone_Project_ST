"""
OLS Coefficient Analysis using statsmodels for detailed statistical inference.

Analyzes relationship between player attributes and prices using OLS regression.

- linear_regression.py focuses on prediction (train W1/test W2, MAE, RMSE, R²)
- this module focuses on interpretation (p-values, coefficients, inference)
"""
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Then import works:
from preprocessing.feature_engineering import load_data, prepare_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup paths (this file is at src/analysis/ols_coefficient_analysis.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "results" / "tables" / "ols_coefficients.csv"


def main():
    """Run OLS coefficient analysis."""
    try:
        # Load data
        df = load_data()
        logger.info("Starting OLS coefficient analysis")
        print()

        # Prepare features (Week 1 only, to analyze relationships)
        x_train, y_train_log, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None)

        # Convert to numpy arrays for statsmodels
        x_train_np = x_train.values
        y_train_log_np = y_train_log.values

        # Add constant (intercept) for OLS
        x_train_with_const = sm.add_constant(x_train_np)

        # Fit OLS regression model
        logger.info("Fitting OLS regression model...")
        ols_model = sm.OLS(y_train_log_np, x_train_with_const).fit()
        logger.info("Model R²: %.4f, Adjusted R²: %.4f", 
                   ols_model.rsquared, ols_model.rsquared_adj)
        print()

        # Extract coefficient information into clean DataFrame
        conf_int = ols_model.conf_int()

        coef_df = pd.DataFrame({
            'feature': ['intercept'] + list(feature_names),
            'coefficient': ols_model.params,
            'std_error': ols_model.bse,
            't_statistic': ols_model.tvalues,
            'p_value': ols_model.pvalues,
            'ci_lower': conf_int[:, 0],  # 95% confidence interval lower bound
            'ci_upper': conf_int[:, 1]})  # 95% confidence interval upper bound

        # Standard significance levels: * p<0.05, ** p<0.01, *** p<0.001
        coef_df['significant'] = coef_df['p_value'] < 0.05
        coef_df['sig_level'] = pd.cut(coef_df['p_value'],
            bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
            labels=['***', '**', '*', 'ns'])

        # Sort by absolute coefficient value
        coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
        coef_df_sorted = coef_df.sort_values('abs_coefficient', ascending=False)

        # Save coefficient table
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        coef_df_sorted.to_csv(OUTPUT_FILE, index=False)
        logger.info("Results saved to %s", OUTPUT_FILE)

        # Summary
        significant_count = (coef_df['p_value'] < 0.05).sum() - 1
        logger.info("Analysis complete: %d/%d features significant (p < 0.05)",
            significant_count, len(feature_names))
        print()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Check that players_complete.csv exists in data/processed/")
        raise

    except KeyError as e:
        logger.error("Column not found: %s", e)
        logger.error("Check that required columns exist in dataset")
        raise

    except Exception as e:
        logger.error("Unexpected error during OLS analysis: %s", e)
        raise


if __name__ == "__main__":
    main()
