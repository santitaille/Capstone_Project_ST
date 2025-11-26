"""
Feature Engineering: Converts raw player data into numbers for ML models.

Standardized numerical features:
- rating, pace, shooting, passing, dribbling, defending, physical, skill_moves, weak_foot,
  num_playstyles, num_playstyles_plus, num_positions.

Binary position cluster features (as-is):
- cluster_cb, cluster_fullbacks, cluster_mid, cluster_att_mid, cluster_st.

One-hot encoded categorical features:
- Nationality: top 10 nationalities (~70% of players) + 'OTHER'
- League: top 8 leagues (~75% of players) + 'OTHER'
- Card category: Gold, Special, Icons_Heroes

Target encoded categorical feature:
- Club: smoothed target encoding based on mean price per club.

Prices are log-transformed for better distribution.

Gender was excluded as it is highly correlated with league.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CONFIGURATION
DATA_FILE = "/files/Capstone_Project_ST/data/processed/players_complete.csv"

# Nationality categories to keep (others grouped as 'OTHER')
TOP_NATIONALITIES = [
    'France', 'Spain', 'England', 'Germany', 'Netherlands',
    'Brazil', 'Italy', 'Portugal', 'Argentina', 'United States'
]

# League categories to keep (others grouped as 'OTHER')
TOP_LEAGUES = [
    'Premier League', 'Icons', 'LALIGA EA SPORTS', 'Serie A TIM',
    'Barclays WSL', 'Bundesliga', 'Ligue 1 McDonald\'s', 'Liga F'
]

# Card categories (for safe one-hot encoding)
CARD_CATEGORIES = ['Gold', 'Special', 'Icons_Heroes']

# Numerical features to use in models
NUMERICAL_FEATURES = [
    'rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical',
    'skill_moves', 'weak_foot', 'num_playstyles', 'num_playstyles_plus', 'num_positions'
]

# Position cluster features (binary)
POSITION_CLUSTERS = [
    'cluster_cb', 'cluster_fullbacks', 'cluster_mid', 'cluster_att_mid', 'cluster_st'
]

# Default target encoding smoothing parameter
DEFAULT_SMOOTHING = 10


# DATA LOADING
def load_data(file_path=DATA_FILE):
    """Load the dataset from CSV file."""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} players with {len(df.columns)} columns")
    return df


# FEATURE ENGINEERING FUNCTIONS
def create_nationality_features(df):
    """
    Creates one-hot encoded features for nationality.
    Top nationalities get their own column, others grouped as 'OTHER'.
    Top nationalities represent ~70% of players. 
    """
    print("Creating nationality features")
    df = df.copy()
    
    # Group rare nationalities as OTHER
    df['nationality_grouped'] = df['nationality'].apply(
        lambda x: x if x in TOP_NATIONALITIES else 'OTHER'
    )
    
    # One-hot encode
    nationality_dummies = pd.get_dummies(df['nationality_grouped'], prefix='nat', dtype=int)
    
    # Rename columns to lowercase with underscores
    nationality_dummies.columns = [col.lower().replace(' ', '_') for col in nationality_dummies.columns]
    
    # Ensure all expected columns exist (safety for train/test consistency)
    expected_cols = [f'nat_{c.lower().replace(" ", "_")}' for c in TOP_NATIONALITIES + ['OTHER']]
    for col in expected_cols:
        if col not in nationality_dummies.columns:
            nationality_dummies[col] = 0
    
    # Sort columns for consistency
    nationality_dummies = nationality_dummies.reindex(sorted(nationality_dummies.columns), axis=1)
    
    print(f"  Created {len(nationality_dummies.columns)} nationality features")
    return nationality_dummies


def create_league_features(df):
    """
    Creates one-hot encoded features for league.
    Top leagues get their own column, others grouped as 'OTHER'.
    Top leagues represent ~75% of players. 
    """
    print("Creating league features")
    df = df.copy()
    
    # Group rare leagues as OTHER
    df['league_grouped'] = df['league'].apply(
        lambda x: x if x in TOP_LEAGUES else 'OTHER'
    )
    
    # One-hot encode
    league_dummies = pd.get_dummies(df['league_grouped'], prefix='league', dtype=int)
    
    # Rename columns to lowercase with underscores
    league_dummies.columns = [
        col.lower().replace(' ', '_').replace("'", "").replace(".", "") 
        for col in league_dummies.columns
    ]
    
    # Ensure all expected columns exist (safety for train/test consistency)
    expected_cols = [f'league_{c.lower().replace(" ", "_").replace("\'", "").replace(".", "")}' for c in TOP_LEAGUES + ['OTHER']]
    for col in expected_cols:
        if col not in league_dummies.columns:
            league_dummies[col] = 0
    
    # Sort columns for consistency
    league_dummies = league_dummies.reindex(sorted(league_dummies.columns), axis=1)
    
    print(f"  Created {len(league_dummies.columns)} league features")
    return league_dummies


def create_card_category_features(df):
    """
    Creates one-hot encoded features for card category.
    Card categories: Gold, Special, Icons_Heroes.
    Unknown categories are grouped as 'OTHER' (safety measure).
    """
    print("Creating card category features")
    df = df.copy()
    
    # Group unknown categories as OTHER (safety measure)
    df['card_grouped'] = df['card_category'].apply(
        lambda x: x if x in CARD_CATEGORIES else 'OTHER'
    )
    
    # One-hot encode
    category_dummies = pd.get_dummies(df['card_grouped'], prefix='card', dtype=int)
    category_dummies.columns = [col.lower().replace(' ', '_') for col in category_dummies.columns]
    
    # Ensure all expected columns exist (only the 3 known categories, no OTHER)
    expected_cols = [f'card_{c.lower()}' for c in CARD_CATEGORIES]
    for col in expected_cols:
        if col not in category_dummies.columns:
            category_dummies[col] = 0
    
    # Keep only expected columns (remove OTHER if it was created)
    category_dummies = category_dummies[[col for col in category_dummies.columns if col in expected_cols]]
    
    # Sort columns for consistency
    category_dummies = category_dummies.reindex(sorted(category_dummies.columns), axis=1)
    
    print(f"  Created {len(category_dummies.columns)} card category features")
    return category_dummies


def create_club_encoding_map(df, target_col, smoothing=DEFAULT_SMOOTHING):
    """
    Creates the club target encoding map from TRAINING data only.
    Call this only on training data to avoid data leakage.
    
    Args:
        df: Training DataFrame
        target_col: Name of target column (price)
        smoothing: Smoothing factor (higher = more regularization toward global mean)
        
    Returns:
        club_encoding_map: Dictionary mapping club names to encoded values
    """
    print(f"Creating club encoding map (smoothing={smoothing})")
    
    global_mean = df[target_col].mean()
    
    # Calculate club statistics
    club_stats = df.groupby('club')[target_col].agg(['mean', 'count'])
    
    # Apply smoothing formula
    club_stats['smoothed_mean'] = (
        (club_stats['count'] * club_stats['mean'] + smoothing * global_mean) /
        (club_stats['count'] + smoothing)
    )
    
    # Create encoding map
    club_encoding_map = club_stats['smoothed_mean'].to_dict()
    club_encoding_map['__global_mean__'] = global_mean  # Fallback for unknown clubs
    
    print(f"  Created encoding map for {len(club_encoding_map) - 1} clubs")
    return club_encoding_map


def apply_club_encoding(df, club_encoding_map):
    """
    Applies pre-computed club encoding map to data.
    Unknown clubs get the global mean (no leakage).
    
    Args:
        df: DataFrame with 'club' column
        club_encoding_map: Pre-computed encoding map from create_club_encoding_map()
        
    Returns:
        Series with encoded club values
    """
    global_mean = club_encoding_map.get('__global_mean__', 0)
    club_encoded = df['club'].map(club_encoding_map).fillna(global_mean)
    return club_encoded


def transform_target(y, inverse=False):
    """Apply log transformation to target variable."""
    if inverse:
        return np.expm1(y)
    else:
        return np.log1p(y)


# MAIN PREPROCESSING PIPELINE
def prepare_features(df, target_col='price_w1', scaler=None, club_encoding_map=None, 
                     smoothing=DEFAULT_SMOOTHING, feature_names=None):
    """
    Main pipeline that combines all feature engineering steps.
    
    TRAINING MODE: scaler=None, club_encoding_map=None, feature_names=None
        - Fits new scaler and creates new encoding map
        - Use this for training data
    
    TEST MODE: scaler=fitted_scaler, club_encoding_map=fitted_map, feature_names=train_features
        - Uses pre-fitted scaler and encoding map
        - Reindexes columns to match training features exactly
        - No data leakage - test data never influences encoding
    
    Args:
        df: Raw DataFrame with player data
        target_col: Name of target column (price)
        scaler: Pre-fitted StandardScaler (None = fit new one)
        club_encoding_map: Pre-computed club encoding map (None = create new one)
        smoothing: Smoothing factor for club target encoding
        feature_names: Feature names from training (None for training, required for test)
        
    Returns:
        X: Feature matrix (DataFrame with 41 features)
        y: Target vector (log-transformed prices)
        scaler: Fitted StandardScaler (save for test data)
        club_encoding_map: Club encoding map (save for test data)
        feature_names: List of all feature names
    """
    print("\nPREPARING FEATURES")
    
    is_training = (scaler is None)
    mode = "TRAINING" if is_training else "TEST"
    print(f"Mode: {mode}")
    
    df = df.copy()
    
    # Target variable (log-transformed)
    y = transform_target(df[target_col])
    print(f"Target: log-transformed {target_col}")
    
    # Numerical features (standardized)
    print("Standardizing numerical features")
    if is_training:
        scaler = StandardScaler()
        X_numerical = pd.DataFrame(
            scaler.fit_transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index
        )
    else:
        X_numerical = pd.DataFrame(
            scaler.transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index
        )
    print(f"  Standardized {len(NUMERICAL_FEATURES)} numerical features")
    
    # Position clusters (binary, as-is)
    X_positions = df[POSITION_CLUSTERS].copy()
    print(f"Position clusters: {len(POSITION_CLUSTERS)} features")
    
    # Nationality features (one-hot)
    X_nationality = create_nationality_features(df)
    X_nationality.index = df.index
    
    # League features (one-hot)
    X_league = create_league_features(df)
    X_league.index = df.index
    
    # Card category features (one-hot)
    X_card = create_card_category_features(df)
    X_card.index = df.index
    
    # Club target encoding (FIT only on training data)
    if is_training:
        club_encoding_map = create_club_encoding_map(df, target_col, smoothing)
    else:
        print("Applying pre-computed club encoding")
    club_encoded = apply_club_encoding(df, club_encoding_map)
    X_club = pd.DataFrame({'club_encoded': club_encoded}, index=df.index)
    
    # Combine all features
    X = pd.concat([
        X_numerical,
        X_positions,
        X_nationality,
        X_league,
        X_card,
        X_club
    ], axis=1)
    
    # In test mode, reindex to match training features exactly
    if not is_training and feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
        print(f"Reindexed to match training features")
    
    # Get feature names (from training) or use current columns
    if is_training:
        feature_names = list(X.columns)
    
    print("\nFEATURE PREPARATION COMPLETE")
    print(f"Total features: {len(feature_names)}")
    print(f"Total samples: {len(X)}")
    
    return X, y, scaler, club_encoding_map, feature_names


# TESTING
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Show train/test split demonstration
    print("\n" + "=" * 60)
    print("DEMONSTRATING TRAIN/TEST WORKFLOW (NO DATA LEAKAGE)")
    print("=" * 60)
    
    # STEP 1: Prepare TRAINING features (fit scaler and encoding on W1)
    print("\n--- STEP 1: TRAINING DATA ---")
    X_train, y_train, scaler, club_encoding_map, feature_names = prepare_features(
        df, 
        target_col='price_w1',
        scaler=None,              # Fit new scaler
        club_encoding_map=None,   # Create new encoding
        feature_names=None        # Will be created
    )
    
    # STEP 2: Prepare TEST features (use pre-fitted scaler and encoding)
    print("\n--- STEP 2: TEST DATA (same players, different target) ---")
    X_test, y_test, _, _, _ = prepare_features(
        df,
        target_col='price_w2',              # Predict W2 prices
        scaler=scaler,                      # Use W1-fitted scaler
        club_encoding_map=club_encoding_map, # Use W1-based encoding (NO LEAKAGE)
        feature_names=feature_names         # Ensure column alignment
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTraining: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Features match: {list(X_train.columns) == list(X_test.columns)}")
    
    print(f"\nFeatures by category:")
    print(f"  Numerical (standardized): {len(NUMERICAL_FEATURES)}")
    print(f"  Position clusters: {len(POSITION_CLUSTERS)}")
    print(f"  Nationality: 11")
    print(f"  League: 9")
    print(f"  Card category: 3")
    print(f"  Club encoded: 1")
    print(f"  TOTAL: {len(feature_names)}")
    
    print(f"\nAll {len(feature_names)} features:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    print(f"\nTarget comparison:")
    print(f"  W1 (train): min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
    print(f"  W2 (test):  min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
    
    print(f"\nTop 5 clubs by encoded value (from W1 training data):")
    sorted_clubs = sorted(
        [(k, v) for k, v in club_encoding_map.items() if k != '__global_mean__'],
        key=lambda x: x[1], 
        reverse=True
    )
    for club, value in sorted_clubs[:5]:
        print(f"  {club}: {value:,.0f}")