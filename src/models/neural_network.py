"""
Model 4: Neural Network (TensorFlow/Keras) on log-transformed prices.

- Train on Week 1 prices (price_w1)
- Test on Week 2 prices (price_w2)
- Uses feature_engineering.prepare_features() to avoid data leakage
- Evaluates performance in both log-space and original price-space
"""

import os
import random
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages for cleaner terminal output

import numpy as np
import pandas as pd

# Tried with MLPRegressor but results were terrible (R² ~0.04)
# Switched to TensorFlow/Keras for more control over architecture and training)
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import load_data, prepare_features
from evaluation_metrics import evaluate_predictions

# Fix random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def build_model(input_dim):
    """Build MLP neural network architecture."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),        # Input layer: 41 features
        keras.layers.Dense(128, activation="relu"),     # First hidden layer: 128 neurons with ReLU activation
        keras.layers.Dropout(0.30),                     # Dropout 30%: randomly drop neurons during training (prevents overfitting)
        keras.layers.Dense(64, activation="relu"),      # Second hidden layer: 64 neurons (progressively compress information)
        keras.layers.Dropout(0.20),                     # Dropout 20%: lighter dropout in deeper layers
        keras.layers.Dense(32, activation="relu"),      # Third hidden layer: 32 neurons (final compression before output)
        keras.layers.Dense(1)                           # Output layer: 1 neuron (predicts scaled log-price)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Adam optimizer: 0.001 is conservative, prevents overshooting
        loss="mse",
        metrics=["mae"]
    )
    return model


if __name__ == "__main__":
    try:
        # Load full dataset (players_complete.csv)
        df = load_data()
        logger.info("Starting Neural Network model training and evaluation")
        
        # Ensure prices are positive (prices <= 0 would cause issues with log transformation)
        if (df["price_w1"] <= 0).any() or (df["price_w2"] <= 0).any():
            raise ValueError("Detected non-positive prices. Clean the data before modeling")
        
        # Prepare TRAIN features (Week 1)
        logger.info("\n=== PREPARING TRAINING FEATURES (W1) ===")
        X_train_all, y_train_log_all, scaler, club_map, feature_names = prepare_features(
            df,
            target_col="price_w1",
            scaler=None,
            club_encoding_map=None,
            smoothing=10,
            feature_names=None,
        )
        
        logger.info(f"X_train_all shape: {X_train_all.shape}")
        logger.info(f"y_train_log_all shape: {y_train_log_all.shape}")
        
        # Prepare TEST features (Week 2) using same scaler + club_map + feature_names
        logger.info("\n=== PREPARING TEST FEATURES (W2) ===")
        X_test, y_test_log, _, _, _ = prepare_features(
            df,
            target_col="price_w2",
            scaler=scaler,
            club_encoding_map=club_map,
            smoothing=10,
            feature_names=feature_names,
        )
        
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test_log shape: {y_test_log.shape}")
        
        # Scale club_encoded feature separately as they are not standardized
        logger.info("\n=== SCALING CLUB_ENCODED FEATURE ===")
        club_scaler = StandardScaler()
        X_train_all['club_encoded'] = club_scaler.fit_transform(X_train_all[['club_encoded']])
        X_test['club_encoded'] = club_scaler.transform(X_test[['club_encoded']])
        logger.info("Successfully scaled club_encoded feature")
        
        # Scale target variable (y) for neural network stability as well
        # We'll inverse transform predictions back to original log-price scale
        logger.info("\n=== SCALING TARGET VARIABLE (Y) ===")  # Log y-scaling operation
        y_scaler = StandardScaler()
        y_train_log_all_reshaped = y_train_log_all.values.reshape(-1, 1)
        y_train_scaled_all = y_scaler.fit_transform(y_train_log_all_reshaped).flatten()
        logger.info("Successfully scaled target variable")
        
        # Internal train/validation split (15% for validation)
        logger.info("\n=== CREATING VALIDATION SPLIT ===")
        X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
            X_train_all,
            y_train_scaled_all,
            test_size=0.15,
            random_state=42
        )
        
        logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")  # Show split sizes
        
        # Build and train Neural Network
        logger.info("\n=== FITTING NEURAL NETWORK MODEL ON W1 ===")
        input_dim = X_train.shape[1]
        model = build_model(input_dim)
        
        logger.info(f"Architecture: {input_dim} -> 128 -> 64 -> 32 -> 1")
        logger.info("Training neural network...")
        
        # Stop training when validation loss stops improving
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",          # Monitor validation loss
            patience=20,                 # Stop if no improvement for 20 epochs
            restore_best_weights=True    # Restore weights from best epoch
        )
        
        history = model.fit(
            X_train,
            y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=300,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        logger.info("Model fitted")  # Confirm training complete
        logger.info(f"Number of layers: 4")  # Report network depth
        logger.info(f"Architecture: 41 -> 128 -> 64 -> 32 -> 1")  # Report network structure
        logger.info(f"Training stopped at epoch: {len(history.history['loss'])}")  # Show when early stopping triggered
        logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")  # Report best performance
        
        # Evaluate on TRAIN (W1) and TEST (W2)
        logger.info("\n=== EVALUATING MODEL ===")  # Log evaluation section start
        
        # Train performance
        # Step 1: Predict scaled values
        # Step 2: Inverse transform to log-price scale
        y_train_pred_scaled = model.predict(X_train_all, verbose=0).flatten()
        y_train_pred_log_all = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        _ = evaluate_predictions(y_train_log_all, y_train_pred_log_all, label="[TRAIN W1]")
        
        # Test performance
        # Step 1: Predict scaled values
        # Step 2: Inverse transform to log-price scale
        y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_test_pred_log = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        metrics_test = evaluate_predictions(y_test_log, y_test_pred_log, label="[TEST W2]")
        
        # Save predictions
        logger.info("\nSaving predictions to CSV")
        results_df = pd.DataFrame({
            "player_name": df["player_name"],
            "rating": df["rating"],
            "card_category": df["card_category"],
            "price_w1": df["price_w1"],
            "price_w2": df["price_w2"],
            "pred_price_w2": np.expm1(y_test_pred_log).round(0),
        })
        
        results_path = "/files/Capstone_Project_ST/data/processed/predictions_neural_network_w2.csv"
        results_df.to_csv(results_path, index=False, float_format='%.0f')
        logger.info(f"Predictions saved to: {results_path}")  # CSV file saved successfully
        
        # Final summary
        logger.info("\n=== FINAL SUMMARY (NEURAL NETWORK) ===")
        logger.info(f"Test R² (W2):   {metrics_test['r2']:.4f}")
        logger.info(f"Test RMSE (W2): {metrics_test['rmse']:,.0f} credits")
        logger.info(f"Test MAE (W2):  {metrics_test['mae']:,.0f} credits")
        logger.info(f"Test MAPE (W2): {metrics_test['mape']:.2f}%")
        
    
    except FileNotFoundError as e:
        # In case the data file is missing
        logger.error(f"File not found: {e}")
        logger.error("Please check that the data file exists")
    
    except KeyError as e:
        # In case required columns are missing from the dataset
        logger.error(f"Column not found: {e}")
        logger.error("Please check that required columns exist in the dataset")
    
    except ImportError as e:
        # In case TensorFlow is not installed
        logger.error(f"TensorFlow not installed: {e}")
        logger.error("Please install with: pip install tensorflow")
    
    except Exception as e:
        # In case of any other unexpected errors
        logger.error(f"Unexpected error during neural network modeling: {e}")
        raise