"""
Feature Engineering: Converts raw player data into numbers for ML models.

Standardized numerical features:
- rating, pace, shooting, passing, dribbling, defending, physical, skill_moves, weak_foot,
  num_playstyles, num_playstyles_plus, num_positions.

Binary position cluster features (as-is):
- cluster_cb, cluster_fullbacks, cluster_mid, cluster_att_mid, cluster_st.

One-hot encoded categorical features:
- Nationality: top 10 nationalities + 'OTHER'
- League: top 8 leagues + 'OTHER'
- Card category: Gold, Special, Icons_Heroes

Target encoded categorical feature:
- Club: smoothed target encoding based on mean price per club.

Prices are log-transformed for better distribution.

Gender was excluded as it is 100% correlated with their league.
"""



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CONFIGURATION
DATA_FILE = "/files/Capstone_Project_ST/data/processed/players_with_week1.csv"

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

# Numerical features to use in models
NUMERICAL_FEATURES = [
    'rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical',
    'skill_moves', 'weak_foot', 'num_playstyles', 'num_playstyles_plus', 'num_positions'
]

# Position cluster features (binary)
POSITION_CLUSTERS = [
    'cluster_cb', 'cluster_fullbacks', 'cluster_mid', 'cluster_att_mid', 'cluster_st'
]

# Target encoding smoothing parameter
SMOOTHING_FACTOR = 10


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
    Top nationalities represents ~70% of players. 
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
    
    print(f"  Created {len(nationality_dummies.columns)} nationality features")
    return nationality_dummies


def create_league_features(df):
    """
    Creates one-hot encoded features for league.
    Top leagues get their own column, others grouped as 'OTHER'.
    Top leagues represents ~85% of players. 
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
    
    print(f"  Created {len(league_dummies.columns)} league features")
    return league_dummies


def create_club_target_encoding(df, target_col='price_w1', smoothing=SMOOTHING_FACTOR):
    """
    Creates target-encoded feature for club using mean price with smoothing.
    Smoothing formula: (count * club_mean + smoothing * global_mean) / (count + smoothing)
    """
    print(f"Creating club target encoding (smoothing={smoothing})")
    
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
    club_encoding_map['__global_mean__'] = global_mean
    
    # Map to dataframe
    club_encoded = df['club'].map(club_stats['smoothed_mean'])
    
    print(f"  Encoded {df['club'].nunique()} unique clubs")
    return club_encoded, club_encoding_map


def create_card_category_features(df):
    """Create one-hot encoded features for card category."""
    print("Creating card category features")
    
    category_dummies = pd.get_dummies(df['card_category'], prefix='card', dtype=int)
    category_dummies.columns = [col.lower().replace(' ', '_') for col in category_dummies.columns]
    
    print(f"  Created {len(category_dummies.columns)} card category features")
    return category_dummies


def transform_target(y, inverse=False):
    """Apply log transformation to target variable."""
    if inverse:
        return np.expm1(y)
    else:
        return np.log1p(y)

# TESTING THE FEATURE ENGINEERING FUNCTIONS
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Test each function
    print("TESTING FEATURE ENGINEERING FUNCTIONS")
    
    # Nationality features
    nat_features = create_nationality_features(df)
    print(f"  Columns: {list(nat_features.columns)}")
    
    # League features
    league_features = create_league_features(df)
    print(f"  Columns: {list(league_features.columns)}")
    
    # Club target encoding
    club_encoded, club_map = create_club_target_encoding(df)
    print(f"  Top 3 clubs: {sorted(club_map.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Card category features
    card_features = create_card_category_features(df)
    print(f"  Columns: {list(card_features.columns)}")
    
    # Target transformation
    y = transform_target(df['price_w1'])
    print(f"\nTarget transformation:")
    print(f"  Original: min={df['price_w1'].min()}, max={df['price_w1'].max()}")
    print(f"  Log-transformed: min={y.min():.2f}, max={y.max():.2f}")