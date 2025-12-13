"""
Model 4: Neural Network (MLP) on log-transformed prices.

Trains on Week 1 prices, tests on Week 2 prices.
Uses feature_engineering.prepare_features() to avoid data leakage.

Generates:
* Table: Neural Network predictions on Week 2 (CSV)
"""

import os
import random
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# False positive warnings, so to have no warnings
# pylint: disable=wrong-import-position,import-error,no-name-in-module
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix random seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.feature_engineering import load_data, prepare_features
from models.evaluation_metrics import evaluate_predictions

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_model(input_dim: int) -> models.Sequential:
    """Build MLP neural network architecture."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
    )
    return model


def main() -> None:
    """Train and evaluate Neural Network model."""
    try:
        # Load full dataset (players_complete.csv)
        df = load_data()

        # Ensure prices are positive
        if (df["price_w1"] <= 0).any() or (df["price_w2"] <= 0).any():
            raise ValueError("Detected non-positive prices")

        # Prepare train features (Week 1)
        x_train_all, y_train_log_all, scaler, club_map, feature_names = (
            prepare_features(
                df,
                target_col="price_w1",
                scaler=None,
                club_encoding_map=None,
                smoothing=10,
                feature_names=None,
            )
        )

        # Prepare test features (Week 2)
        x_test, y_test_log, _, _, _ = prepare_features(
            df,
            target_col="price_w2",
            scaler=scaler,
            club_encoding_map=club_map,
            smoothing=10,
            feature_names=feature_names,
        )

        # Scale club_encoded feature separately
        club_scaler = StandardScaler()
        x_train_all["club_encoded"] = club_scaler.fit_transform(
            x_train_all[["club_encoded"]]
        )
        x_test["club_encoded"] = club_scaler.transform(x_test[["club_encoded"]])

        # Scale target variable (y) for neural network stability
        y_scaler = StandardScaler()
        y_train_log_all_reshaped = y_train_log_all.values.reshape(-1, 1)
        y_train_scaled_all = y_scaler.fit_transform(y_train_log_all_reshaped).flatten()

        # Internal train/validation split (15% for validation)
        x_train, x_val, y_train_scaled, y_val_scaled = train_test_split(
            x_train_all, y_train_scaled_all, test_size=0.15, random_state=42
        )

        # Train model
        logger.info("Training Neural Network (MLP)...")

        input_dim = x_train.shape[1]
        model = build_model(input_dim)

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )

        history = model.fit(
            x_train,
            y_train_scaled,
            validation_data=(x_val, y_val_scaled),
            epochs=300,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0,
        )

        epochs_trained = len(history.history["loss"])
        logger.info(
            "Architecture: 41 → 128 → 64 → 32 → 1 (4 layers - dropout: 30%, 20%)"
        )
        logger.info(
            "Hyperparameters: epochs=300, batch_size=32, early_stop_patience=20"
        )
        logger.info(
            "Training stopped at epoch %d (validation loss plateaued)", epochs_trained
        )
        print()

        # Predict and calculate train metrics
        y_train_pred_scaled = model.predict(x_train_all, verbose=0).flatten()
        y_train_pred_log_all = y_scaler.inverse_transform(
            y_train_pred_scaled.reshape(-1, 1)
        ).flatten()
        metrics_train = evaluate_predictions(y_train_log_all, y_train_pred_log_all)

        # Predict and calculate test metrics
        y_test_pred_scaled = model.predict(x_test, verbose=0).flatten()
        y_test_pred_log = y_scaler.inverse_transform(
            y_test_pred_scaled.reshape(-1, 1)
        ).flatten()
        metrics_test = evaluate_predictions(y_test_log, y_test_pred_log)

        # Save predictions
        results_df = pd.DataFrame(
            {
                "player_name": df["player_name"],
                "rating": df["rating"],
                "card_category": df["card_category"],
                "price_w1": df["price_w1"],
                "price_w2": df["price_w2"],
                "pred_price_w2": np.expm1(y_test_pred_log).round(0),
            }
        )

        project_root = Path(__file__).resolve().parent.parent.parent
        results_path = (
            project_root
            / "results"
            / "predictions"
            / "predictions_neural_network_w2.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False, float_format="%.0f")

        # Calculate gaps
        gap_r2 = metrics_test["r2"] - metrics_train["r2"]
        gap_rmse = metrics_test["rmse"] - metrics_train["rmse"]
        gap_mae = metrics_test["mae"] - metrics_train["mae"]

        # Display results
        logger.info("Neural Network (MLP) Performance:")
        logger.info("-" * 80)
        logger.info("%-10s %15s %15s %12s", "Metric", "Train (W1)", "Test (W2)", "Gap")
        logger.info("-" * 80)
        logger.info(
            "%-10s %15.3f %15.3f %12.3f",
            "R²",
            metrics_train["r2"],
            metrics_test["r2"],
            gap_r2,
        )
        logger.info(
            "%-10s %15s %15s %12s",
            "RMSE",
            f"{metrics_train['rmse']:,.0f}",
            f"{metrics_test['rmse']:,.0f}",
            f"{gap_rmse:,.0f}",
        )
        logger.info(
            "%-10s %15s %15s %12s",
            "MAE",
            f"{metrics_train['mae']:,.0f}",
            f"{metrics_test['mae']:,.0f}",
            f"{gap_mae:,.0f}",
        )
        logger.info("-" * 80)
        logger.info("  → Saved: results/predictions/predictions_neural_network_w2.csv")
        print()

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during neural network: %s", e)
        raise


if __name__ == "__main__":
    main()
