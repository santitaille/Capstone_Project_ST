"""
Baseline Model 1: Median Price per OVR Rating
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load data
df = pd.read_csv("/files/Capstone_Project_ST/data/processed/players_with_week1.csv")
print(f"Dataset: {len(df)} players\n")

# calculate median price for each rating (83-95)
rating_medians = df.groupby('rating')['price_w1'].median()

print("Median prices by rating:")
print(rating_medians)
print()

# make predictions for baseline model 1
df['baseline1_pred'] = df['rating'].map(rating_medians)

# evaluate baseline model 1
mae = mean_absolute_error(df['price_w1'], df['baseline1_pred']) # on average, the model is off by this many credits
rmse = np.sqrt(mean_squared_error(df['price_w1'], df['baseline1_pred'])) # on average, the model is off by this many credits (penalizes larger errors more)
r2 = r2_score(df['price_w1'], df['baseline1_pred']) # how much variance is explained by the model

print(f"Baseline model 1 (OVR only):")
print(f"  MAE:  {mae:,.0f} credits")
print(f"  RMSE: {rmse:,.0f} credits")
print(f"  R²:   {r2:.3f}")