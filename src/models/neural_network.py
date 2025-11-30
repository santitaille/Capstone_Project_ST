"""
Model 4: Neural Network (TensorFlow/Keras) on log-transformed prices.

- Train on Week 1 prices (price_w1)
- Test on Week 2 prices (price_w2)
- Uses feature_engineering.prepare_features() to avoid data leakage
- Evaluates performance in both log-space and original price-space
"""

import os
import random

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
    
    # Load full dataset (players_complete.csv)
    df = load_data()
    
    # Prepare TRAIN features (Week 1)
    X_train_all, y_train_log_all, scaler, club_map, feature_names = prepare_features(
        df,
        target_col="price_w1",
        scaler=None,
        club_encoding_map=None,
        smoothing=10,
        feature_names=None,
    )
    
    # Prepare TEST features (Week 2) using SAME scaler + club_map + feature_names
    X_test, y_test_log, _, _, _ = prepare_features(
        df,
        target_col="price_w2",
        scaler=scaler,
        club_encoding_map=club_map,
        smoothing=10,
        feature_names=feature_names,
    )
    
    # Scale club_encoded feature separately, otherwise gradients explode during training and model fails to learn
    # club_encoded contains large values (100k-800k) while other features are standardized (~0-1)
    club_scaler = StandardScaler()
    X_train_all['club_encoded'] = club_scaler.fit_transform(X_train_all[['club_encoded']])
    X_test['club_encoded'] = club_scaler.transform(X_test[['club_encoded']])
    
    # We'll inverse transform predictions back to original log-price scale later
    y_scaler = StandardScaler()
    y_train_log_all_reshaped = y_train_log_all.values.reshape(-1, 1)  # StandardScaler needs 2D array
    y_train_scaled_all = y_scaler.fit_transform(y_train_log_all_reshaped).flatten()  # Back to 1D
    
    # Early stopping uses validation loss to decide when to stop
    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_train_all,
        y_train_scaled_all,
        test_size=0.15,  # 15% = 121 players for validation, 680 for training
        random_state=42  # Same random split every time for reproducibility
    )
    
    # Build and train Neural Network
    input_dim = X_train.shape[1]  # 41 features
    model = build_model(input_dim)
    
    # Stop training when validation loss stops improving (prevents overfitting by not training too long)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",          # Watch validation loss (not training loss)
        patience=20,                 # Stop if no improvement for 20 consecutive epochs
        restore_best_weights=True    # Restore model weights from best epoch (not final epoch)
    )
    
    # Train the model
    history = model.fit(
        X_train,                              # Training features (680 players)
        y_train_scaled,                       # Training targets (scaled log-prices)
        validation_data=(X_val, y_val_scaled), # Validation data for early stopping
        epochs=300,                           # Maximum 300 passes through data with early stopping
        batch_size=32,                        # Update weights after every 32 samples (mini-batch gradient descent)
        callbacks=[early_stopping],           # Apply early stopping
        verbose=0                             # Silent training (no progress bar)
    )
    
    print(f"\nModel fitted")
    print(f"Number of layers: 4")
    print(f"Architecture: 41 -> 128 -> 64 -> 32 -> 1")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")
    
    # Evaluate on TRAIN (W1) and TEST (W2)
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Train performance (inverse transform predictions)
    # Step 1: Predict on FULL training set (all 801 players)
    y_train_pred_scaled = model.predict(X_train_all, verbose=0).flatten()  # Predictions in scaled space
    # Step 2: Inverse transform from scaled space back to log-price space
    y_train_pred_log_all = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    # Step 3: Evaluate
    _ = evaluate_predictions(y_train_log_all, y_train_pred_log_all, label="[TRAIN W1]")
    
    # Test performance (inverse transform predictions)
    # Step 1: Predict on test set (Week 2 prices)
    y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()  # Predictions in scaled space
    # Step 2: Inverse transform from scaled space back to log-price space
    y_test_pred_log = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    # Step 3: Evaluate
    metrics_test = evaluate_predictions(y_test_log, y_test_pred_log, label="[TEST W2]")
    
    # Save predictions
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
    print(f"\nPredictions saved to: {results_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY (NEURAL NETWORK)")
    print("="*60)
    print(f"Test R² (W2):   {metrics_test['r2']:.4f}")
    print(f"Test RMSE (W2): {metrics_test['rmse']:,.0f} credits")
    print(f"Test MAE (W2):  {metrics_test['mae']:,.0f} credits")
    print(f"Test MAPE (W2): {metrics_test['mape']:.2f}%")