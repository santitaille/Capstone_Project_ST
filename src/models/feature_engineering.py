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
    """Creates one-hot encoded features for card category.
    Card categories: Gold, Special, Icons_Heroes
    """
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

# MAIN PREPROCESSING PIPELINE
def prepare_features(df, target_col='price_w1'):
    """
    Main pipeline that combines all feature engineering steps.
    
    Args:
        df: Raw DataFrame with player data
        target_col: Name of target column (price)
        
    Returns:
        X: Feature matrix (dataframe with 41 features)
        y: Target vector (log-transformed prices)
        scaler: Fitted StandardScaler (to use on test data)
        club_encoding_map: Club encoding mapping (to use on test data)
        feature_names: List of all feature names
    """
    print("\nPREPARING FEATURES")
    
    df = df.copy()
    
    # Target variable (log-transformed)
    y = transform_target(df[target_col])
    print(f"Target: log-transformed {target_col}")
    
    # Numerical features (standardized)
    print("Standardizing numerical features")
    scaler = StandardScaler()
    X_numerical = pd.DataFrame(
        scaler.fit_transform(df[NUMERICAL_FEATURES]),
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
    
    # Club target encoding
    club_encoded, club_encoding_map = create_club_target_encoding(df, target_col)
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
    
    feature_names = list(X.columns)
    
    print("FEATURE PREPARATION COMPLETE")
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Total samples: {len(X)}")
    
    return X, y, scaler, club_encoding_map, feature_names


# TESTING THE FEATURE ENGINEERING FUNCTIONS
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Prepare features using main pipeline
    X, y, scaler, club_encoding_map, feature_names = prepare_features(df)
    
    #SUMMARY
    print("\nSUMMARY")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    print(f"\nFeatures by category:")
    print(f"  Numerical (standardized): {len(NUMERICAL_FEATURES)}")
    print(f"  Position clusters: {len(POSITION_CLUSTERS)}")
    print(f"  Nationality: 11")
    print(f"  League: 9")
    print(f"  Card category: 3")
    print(f"  Club encoded: 1")
    print(f"  TOTAL: {len(feature_names)}")
    
    # Detailed feature list of all 41 features
    print(f"\nAll features:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Show target stats for log-transformed prices
    print(f"\nTarget (log-transformed price):")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    print(f"  Mean: {y.mean():.2f}")
    
    # Show top 5 clubs by encoded value
    print(f"\nTop 5 clubs by encoded value:")
    sorted_clubs = sorted(
        [(k, v) for k, v in club_encoding_map.items() if k != '__global_mean__'],
        key=lambda x: x[1], 
        reverse=True
    )
    for club, value in sorted_clubs[:5]:
        print(f"  {club}: {value:,.0f}")
