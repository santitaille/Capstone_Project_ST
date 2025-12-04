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

Gender was excluded as it is 100% correlated with league.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Checks if features have already been printed in main.py
class _FeatureConfig:
    logged = False

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


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    if file_path is None:
        file_path = DATA_FILE

    # Suppress logging after first call
    if _FeatureConfig.logged:
        logger.setLevel(logging.WARNING)

    logger.info("Loading data from %s", DATA_FILE.relative_to(PROJECT_ROOT))
    df = pd.read_csv(file_path)
    logger.info("Loaded %d players with %d features", len(df), len(df.columns))
    if not _FeatureConfig.logged: # So there is no empty lines in the main.py terminal when I import it
        print()
    return df


def create_nationality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one-hot encoded features for nationality.
    Top nationalities get their own column, others grouped as 'OTHER'.
    Top nationalities represent ~70% of players.
    """
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

    logger.info("  %-12s: %3d features (Top 10 + 'OTHER' - representing 70%%)", "Nationality", 
        len(nationality_dummies.columns))
    return nationality_dummies


def create_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one-hot encoded features for league.
    Top leagues get their own column, others grouped as 'OTHER'.
    Top leagues represent ~75% of players.
    """
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

    logger.info("  %-12s: %3d features (Top 8 + 'OTHER' - representing 75%%)", "Leagues", len(league_dummies.columns))
    return league_dummies


def create_card_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one-hot encoded features for card category.
    Card categories: Gold, Special, Icons_Heroes.
    Unknown categories are grouped as 'OTHER' (should not be the case).
    """
    df = df.copy()

    # Group unknown categories as OTHER (should not be the case)
    df['card_grouped'] = df['card_category'].apply(
        lambda x: x if x in CARD_CATEGORIES else 'OTHER')

    # One-hot encode
    category_dummies = pd.get_dummies(df['card_grouped'], prefix='card', dtype=int)
    category_dummies.columns = [col.lower().replace(' ', '_') for col in category_dummies.columns]

    # Ensure all expected columns exist (only the 3 known categories, no 'OTHER')
    expected_cols = [f'card_{c.lower()}' for c in CARD_CATEGORIES]
    for col in expected_cols:
        if col not in category_dummies.columns:
            category_dummies[col] = 0

    # Keep only expected columns (remove 'OTHER' if it was created)
    category_dummies = category_dummies[
        [col for col in category_dummies.columns if col in expected_cols]]

    # Sort columns for consistency
    category_dummies = category_dummies.reindex(sorted(category_dummies.columns), axis=1)

    logger.info("  %-12s: %3d features (Gold, Special, Icons_Heroes)", "Category", len(category_dummies.columns))
    return category_dummies


def create_club_encoding_map(df: pd.DataFrame, target_col: str,
    smoothing: int = DEFAULT_SMOOTHING) -> Dict[str, float]:
    """
    Create club target encoding map from training data only.
    Use smoothing (10) to prevent overfitting on rare clubs.
    """
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

    logger.info("  %-12s: %3d encoded (Target Encoding, smooth=10)", "Clubs", len(club_encoding_map) - 1)
    if not _FeatureConfig.logged: # So there is no empty lines in the main.py terminal when I import it
        print()
    return club_encoding_map


def apply_club_encoding(df: pd.DataFrame, club_encoding_map: Dict[str, float]) -> pd.Series:
    """
    Apply pre-computed club encoding to data.
    Unknown clubs get global mean (to avoid data leakage).
    """
    global_mean = club_encoding_map.get('__global_mean__', 0)
    club_encoded = df['club'].map(club_encoding_map).fillna(global_mean)
    return club_encoded


def transform_target(y: pd.Series, inverse: bool = False) -> pd.Series:
    """Apply log transformation to target variable."""
    if inverse:
        return np.expm1(y)
    return np.log1p(y)


def prepare_features(df: pd.DataFrame,
    target_col: str = 'price_w1',
    scaler: Optional[StandardScaler] = None,
    club_encoding_map: Optional[Dict[str, float]] = None,
    smoothing: int = DEFAULT_SMOOTHING,
    feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series,
        StandardScaler, Dict[str, float], List[str]]:
    """
    Transform raw player data into ML features.
    """
    # Suppress logger after first call (in main.py) to avoid repetition
    if _FeatureConfig.logged:
        logger.setLevel(logging.WARNING)
    else:
        _FeatureConfig.logged = True

    is_training = scaler is None
    df = df.copy()

    # Target variable (log-transformed)
    y = transform_target(df[target_col])

    # Numerical features (standardized)
    if is_training:
        scaler = StandardScaler()
        x_numerical = pd.DataFrame(
            scaler.fit_transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index)
    else:
        x_numerical = pd.DataFrame(
            scaler.transform(df[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=df.index)

    logger.info("Feature Engineering Summary:")
    logger.info("  %-12s: %3d features (Standardized)", "Numerical", len(NUMERICAL_FEATURES))

    # Position clusters (binary, as-is)
    x_positions = df[POSITION_CLUSTERS].copy()
    logger.info("  %-12s: %3d clusters (Center Backs, Fullbacks, Midfielders, Attacking Midfielders, Strikers)", "Position", len(POSITION_CLUSTERS))

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

    # In test mode, reindex to match training features
    if not is_training and feature_names is not None:
        x = x.reindex(columns=feature_names, fill_value=0)

    # Get feature names (from training) or use current columns
    if is_training:
        feature_names = list(x.columns)

    logger.info("Feature preparation complete: %d players with %d features", len(x), len(feature_names))
    if not _FeatureConfig.logged: # So there is no empty lines in the main.py terminal when I import it
        print()

    return x, y, scaler, club_encoding_map, feature_names
