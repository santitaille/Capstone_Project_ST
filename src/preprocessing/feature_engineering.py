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

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Setup paths (this file is at src/preprocessing/feature_engineering.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "players_complete.csv"

# Nationality categories to keep (others grouped as 'OTHER')
TOP_NATIONALITIES = [
    'France', 'Spain', 'England', 'Germany', 'Netherlands',
    'Brazil', 'Italy', 'Portugal', 'Argentina', 'United States']

# League categories to keep (others grouped as 'OTHER')
TOP_LEAGUES = [
    'Premier League', 'Icons', 'LALIGA EA SPORTS', 'Serie A TIM',
    'Barclays WSL', 'Bundesliga', 'Ligue 1 McDonald\'s', 'Liga F']

# Card categories (for safe one-hot encoding)
CARD_CATEGORIES = ['Gold', 'Special', 'Icons_Heroes']

# Numerical features to use in models
NUMERICAL_FEATURES = [
    'rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical',
    'skill_moves', 'weak_foot', 'num_playstyles', 'num_playstyles_plus', 'num_positions']

# Position cluster features (binary)
POSITION_CLUSTERS = [
    'cluster_cb', 'cluster_fullbacks', 'cluster_mid', 'cluster_att_mid', 'cluster_st']

# Default target encoding smoothing parameter
DEFAULT_SMOOTHING = 10


def load_data(file_path=None):
    """Load the dataset from CSV file."""
    if file_path is None:
        file_path = DATA_FILE

    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Loaded %d players with %d columns", len(df), len(df.columns))
    return df


def create_nationality_features(df):
    """
    Creates one-hot encoded features for nationality.
    Top nationalities get their own column, others grouped as 'OTHER'.
    Top nationalities represent ~70% of players.
    """
    logger.info("Creating nationality features")
    df = df.copy()

    # Group rare nationalities as OTHER
    df['nationality_grouped'] = df['nationality'].apply(
        lambda x: x if x in TOP_NATIONALITIES else 'OTHER')

    # One-hot encode
    nationality_dummies = pd.get_dummies(df['nationality_grouped'], prefix='nat', dtype=int)

    # Rename columns to lowercase with underscores
    nationality_dummies.columns = [col.lower().replace(' ', '_')
        for col in nationality_dummies.columns]

    # Ensure all expected columns exist (safety for train/test consistency)
    expected_cols = [f'nat_{c.lower().replace(" ", "_")}' for c in TOP_NATIONALITIES + ['OTHER']]
    for col in expected_cols:
        if col not in nationality_dummies.columns:
            nationality_dummies[col] = 0

    # Sort columns for consistency
    nationality_dummies = nationality_dummies.reindex(sorted(nationality_dummies.columns), axis=1)

    logger.info("  Created %d nationality features", len(nationality_dummies.columns))
    return nationality_dummies


def create_league_features(df):
    """
    Creates one-hot encoded features for league.
    Top leagues get their own column, others grouped as 'OTHER'.
    Top leagues represent ~75% of players.
    """
    logger.info("Creating league features")
    df = df.copy()

    # Group rare leagues as OTHER
    df['league_grouped'] = df['league'].apply(
        lambda x: x if x in TOP_LEAGUES else 'OTHER')

    # One-hot encode
    league_dummies = pd.get_dummies(df['league_grouped'], prefix='league', dtype=int)

    # Rename columns to lowercase with underscores
    league_dummies.columns = [col.lower().replace(' ', '_').replace("'", "").replace(".", "")
        for col in league_dummies.columns]

    # Ensure all expected columns exist (safety for train/test consistency)
    expected_cols = [f'league_{c.lower().replace(" ", "_").replace("\'", "").replace(".", "")}'
        for c in TOP_LEAGUES + ['OTHER']]
    for col in expected_cols:
        if col not in league_dummies.columns:
            league_dummies[col] = 0

    # Sort columns for consistency
    league_dummies = league_dummies.reindex(sorted(league_dummies.columns), axis=1)

    logger.info("  Created %d league features", len(league_dummies.columns))
    return league_dummies


def create_card_category_features(df):
    """
    Creates one-hot encoded features for card category.
    Card categories: Gold, Special, Icons_Heroes.
    Unknown categories are grouped as 'OTHER' (safety measure).
    """
    logger.info("Creating card category features")
    df = df.copy()

    # Group unknown categories as OTHER (safety measure)
    df['card_grouped'] = df['card_category'].apply(
        lambda x: x if x in CARD_CATEGORIES else 'OTHER')

    # One-hot encode
    category_dummies = pd.get_dummies(df['card_grouped'], prefix='card', dtype=int)
    category_dummies.columns = [col.lower().replace(' ', '_') for col in category_dummies.columns]

    # Ensure all expected columns exist (only the 3 known categories, no OTHER)
    expected_cols = [f'card_{c.lower()}' for c in CARD_CATEGORIES]
    for col in expected_cols:
        if col not in category_dummies.columns:
            category_dummies[col] = 0

    # Keep only expected columns (remove OTHER if it was created)
    category_dummies = category_dummies[
        [col for col in category_dummies.columns if col in expected_cols]]

    # Sort columns for consistency
    category_dummies = category_dummies.reindex(sorted(category_dummies.columns), axis=1)

    logger.info("  Created %d card category features", len(category_dummies.columns))
    return category_dummies


def create_club_encoding_map(df, target_col, smoothing=DEFAULT_SMOOTHING):
    """
    Create club target encoding map from training data only.
    Use smoothing to prevent overfitting on rare clubs.
    Return dictionary mapping club names to encoded values.
    """
    logger.info("Creating club encoding map (smoothing=%d)", smoothing)

    global_mean = df[target_col].mean()

    # Calculate club statistics
    club_stats = df.groupby('club')[target_col].agg(['mean', 'count'])

    # Apply smoothing formula to prevent overfitting on rare clubs
    club_stats['smoothed_mean'] = (
        (club_stats['count'] * club_stats['mean'] + smoothing * global_mean) /
        (club_stats['count'] + smoothing))

    # Create encoding map
    club_encoding_map = club_stats['smoothed_mean'].to_dict()
    club_encoding_map['__global_mean__'] = global_mean

    logger.info("  Created encoding map for %d clubs", len(club_encoding_map) - 1)
    return club_encoding_map


def apply_club_encoding(df, club_encoding_map):
    """
    Apply pre-computed club encoding to data.
    Unknown clubs get global mean (to avoid data leakage).
    Return series with encoded club values.
    """
    global_mean = club_encoding_map.get('__global_mean__', 0)
    club_encoded = df['club'].map(club_encoding_map).fillna(global_mean)
    return club_encoded


def transform_target(y, inverse=False):
    """Apply log transformation to target variable."""
    if inverse:
        return np.expm1(y)
    return np.log1p(y)


# MAIN PREPROCESSING PIPELINE
def prepare_features(df, target_col='price_w1', scaler=None, club_encoding_map=None,
    smoothing=DEFAULT_SMOOTHING, feature_names=None):
    """
    Transform raw player data into 41 ML-ready features.
    Returns feature matrix x, target y, and fitted objects for test data.
    """
    logger.info("Preparing features")

    is_training = scaler is None
    mode = "TRAINING" if is_training else "TEST"
    logger.info("Mode: %s", mode)

    df = df.copy()

    # Target variable (log-transformed)
    y = transform_target(df[target_col])
    logger.info("Target: log-transformed %s", target_col)

    # Numerical features (standardized)
    logger.info("Standardizing numerical features")
    if is_training:
        scaler = StandardScaler()
        x_numerical = pd.DataFrame(
            scaler.fit_transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index
        )
    else:
        x_numerical = pd.DataFrame(
            scaler.transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index
        )
    logger.info("  Standardized %d numerical features", len(NUMERICAL_FEATURES))

    # Position clusters (binary, as-is)
    x_positions = df[POSITION_CLUSTERS].copy()
    logger.info("Position clusters: %d features", len(POSITION_CLUSTERS))

    # Nationality features (one-hot)
    x_nationality = create_nationality_features(df)
    x_nationality.index = df.index

    # League features (one-hot)
    x_league = create_league_features(df)
    x_league.index = df.index

    # Card category features (one-hot)
    x_card = create_card_category_features(df)
    x_card.index = df.index

    # Club target encoding (fit only on training data)
    if is_training:
        club_encoding_map = create_club_encoding_map(df, target_col, smoothing)
    else:
        logger.info("Applying pre-computed club encoding")
    club_encoded = apply_club_encoding(df, club_encoding_map)
    x_club = pd.DataFrame({'club_encoded': club_encoded}, index=df.index)

    # Combine all features
    x = pd.concat([
        x_numerical,
        x_positions,
        x_nationality,
        x_league,
        x_card,
        x_club
    ], axis=1)

    # In test mode, reindex to match training features exactly
    if not is_training and feature_names is not None:
        x = x.reindex(columns=feature_names, fill_value=0)
        logger.info("Reindexed to match training features")

    # Get feature names (from training) or use current columns
    if is_training:
        feature_names = list(x.columns)

    logger.info("Feature preparation complete: %d features, %d samples", len(feature_names), len(x))
    print()

    return x, y, scaler, club_encoding_map, feature_names
