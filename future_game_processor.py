"""
Process future NFL games data for ML model predictions.
Converts API data into the format expected by trained models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from api_fetchers import APIConfig, fetch_upcoming_nfl_games, fetch_odds_data, combine_game_data, fetch_all_upcoming_nfl_games_oddsapi
from feature_engineering import (
    compute_team_rolling_averages,
    compute_streaks_and_last5,
    calculate_head_to_head,
    add_rest_days_features,
    final_feature_engineering
)
from elo_system import calculate_elo_ratings
from data_loading import load_teams_data, load_and_clean_data

def extract_odds_features(odds_data: Dict) -> Dict:
    """
    Extract betting odds features from odds data.
    
    Args:
        odds_data: Dictionary containing odds information
    
    Returns:
        Dictionary with extracted odds features
    """
    features = {
        'spread_favorite': None,
        'over_under_line': None,
        'home_is_favorite': None
    }
    
    if not odds_data or 'bookmakers' not in odds_data:
        return features
    
    # Find the first bookmaker with spread and totals data
    for bookmaker in odds_data['bookmakers']:
        markets = bookmaker.get('markets', {})
        
        # Extract spread data
        if 'spreads' in markets:
            spreads = markets['spreads']
            if len(spreads) >= 2:
                # Find favorite and underdog
                favorite = None
                underdog = None
                for outcome in spreads:
                    if outcome.get('point', 0) < 0:  # Negative points = favorite
                        favorite = outcome
                    else:
                        underdog = outcome
                
                if favorite and underdog:
                    features['spread_favorite'] = abs(favorite['point'])
                    features['home_is_favorite'] = 1 if favorite['name'] == odds_data['home_team'] else 0
        
        # Extract over/under data
        if 'totals' in markets:
            totals = markets['totals']
            if totals:
                features['over_under_line'] = totals[0].get('point')
        
        # If we found both spread and totals, break
        if features['spread_favorite'] is not None and features['over_under_line'] is not None:
            break
    
    return features

def create_future_game_features(game_data: List[Dict], historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for future games using historical data for context.
    
    Args:
        game_data: List of future game dictionaries from APIs
        historical_data: Historical games DataFrame for feature engineering
    
    Returns:
        DataFrame with engineered features for future games
    """
    if not game_data:
        return pd.DataFrame()
    
    # Convert game data to DataFrame
    future_games = []
    
    for game in game_data:
        # Basic game info
        game_row = {
            'schedule_date': pd.to_datetime(game['date']),
            'schedule_season': int(game.get('season', datetime.now().year)),
            'schedule_week': None,  # Will be calculated
            'schedule_playoff': 0,  # Assume regular season for now
            'team_home': game['home_team'],
            'team_away': game['away_team'],
            'stadium_neutral': game.get('stadium_neutral', False),
            'stadium': game.get('venue', ''),
            'spread_favorite': None,
            'over_under_line': None,
            'home_is_favorite': None
        }
        
        # Extract odds features
        if game.get('odds'):
            odds_features = extract_odds_features(game['odds'])
            game_row.update(odds_features)
        
        future_games.append(game_row)
    
    # Create DataFrame
    future_df = pd.DataFrame(future_games)
    
    if future_df.empty:
        return future_df
    
    # Calculate schedule week
    future_df['schedule_week'] = future_df['schedule_date'].dt.isocalendar().week
    
    # Combine with historical data for feature engineering
    # We need historical data to calculate rolling averages, streaks, etc.
    combined_data = pd.concat([historical_data, future_df], ignore_index=True)
    combined_data['schedule_date'] = pd.to_datetime(combined_data['schedule_date'])
    combined_data = combined_data.sort_values('schedule_date')
    
    # Apply feature engineering (this will work on the combined dataset)
    print("🔧 Applying feature engineering to future games...")
    
    # Load teams data
    teams = load_teams_data()
    
    # Apply feature engineering steps
    combined_data = compute_team_rolling_averages(combined_data)
    combined_data = compute_streaks_and_last5(combined_data)
    combined_data['head_to_head'] = calculate_head_to_head(combined_data)
    combined_data = add_rest_days_features(combined_data)
    combined_data = calculate_elo_ratings(combined_data, teams)
    combined_data = final_feature_engineering(combined_data, teams)
    
    # Extract only the future games with engineered features
    future_games_with_features = combined_data.tail(len(future_games))
    
    print(f"✅ Created features for {len(future_games_with_features)} future games")
    return future_games_with_features

def prepare_future_games_for_prediction(future_games_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare future games data for model prediction.
    
    Args:
        future_games_df: DataFrame with engineered features for future games
    
    Returns:
        Dictionary with prepared datasets for classification and regression
    """
    if future_games_df.empty:
        return {'classification': pd.DataFrame(), 'regression': pd.DataFrame()}
    
    # Classification dataset (drop regression targets)
    classification_df = future_games_df.drop(
        columns=['spread_favorite', 'over_under_line', 'score_home', 'score_away'], 
        errors='ignore'
    )
    
    # Regression dataset (drop classification target)
    regression_df = future_games_df.drop(
        columns=['home_team_wins'], 
        errors='ignore'
    )
    
    # Remove any rows with missing critical features
    classification_df = classification_df.dropna()
    regression_df = regression_df.dropna()
    
    print(f"✅ Prepared {len(classification_df)} games for classification prediction")
    print(f"✅ Prepared {len(regression_df)} games for regression prediction")
    
    return {
        'classification': classification_df,
        'regression': regression_df
    }

def fetch_and_process_future_games(days_ahead: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Fetch and process future NFL games for prediction.
    
    Args:
        days_ahead: Number of days to look ahead for games
    
    Returns:
        Dictionary with processed datasets for prediction
    """
    print("🚀 Fetching and processing future NFL games...")
    
    # Load historical data for feature engineering context
    historical_data = load_and_clean_data()
    
    # Fetch future games from APIs
    config = APIConfig()
    games = fetch_upcoming_nfl_games(config, days_ahead)
    odds = fetch_odds_data(config)
    combined_games = combine_game_data(games, odds)
    
    if not combined_games:
        print("❌ No future games found")
        return {'classification': pd.DataFrame(), 'regression': pd.DataFrame()}
    
    # Create features for future games
    future_games_with_features = create_future_game_features(combined_games, historical_data)
    
    # Prepare for prediction
    prediction_datasets = prepare_future_games_for_prediction(future_games_with_features)
    
    return prediction_datasets

def extract_odds_features_from_oddsapi(game: Dict) -> Dict:
    """
    Extract favorite, underdog, spread, moneyline, and over/under from Odds API game dict.
    """
    home = game['home_team']
    away = game['away_team']
    spread = None
    spread_favorite = None
    moneyline_home = None
    moneyline_away = None
    over_under = None
    over_odds = None
    under_odds = None
    favorite_team = None
    underdog_team = None
    if game.get('bookmakers'):
        for bookmaker in game['bookmakers']:
            markets = {m['key']: m for m in bookmaker.get('markets', [])}
            if 'h2h' in markets:
                for outcome in markets['h2h']['outcomes']:
                    if outcome['name'] == home:
                        moneyline_home = outcome['price']
                    elif outcome['name'] == away:
                        moneyline_away = outcome['price']
            if 'spreads' in markets:
                for outcome in markets['spreads']['outcomes']:
                    if outcome['name'] == home:
                        home_spread = outcome.get('point')
                        home_spread_odds = outcome.get('price')
                    elif outcome['name'] == away:
                        away_spread = outcome.get('point')
                        away_spread_odds = outcome.get('price')
                if 'home_spread' in locals() and 'away_spread' in locals():
                    if home_spread < 0:
                        favorite_team = home
                        underdog_team = away
                        spread = abs(home_spread)
                        spread_favorite = home
                    elif away_spread < 0:
                        favorite_team = away
                        underdog_team = home
                        spread = abs(away_spread)
                        spread_favorite = away
            if 'totals' in markets:
                for outcome in markets['totals']['outcomes']:
                    if outcome['name'] == 'Over':
                        over_under = outcome.get('point')
                        over_odds = outcome.get('price')
                    elif outcome['name'] == 'Under':
                        under_odds = outcome.get('price')
            break
    return {
        'favorite_team': favorite_team,
        'underdog_team': underdog_team,
        'spread': spread,
        'spread_favorite': spread_favorite,
        'moneyline_home': moneyline_home,
        'moneyline_away': moneyline_away,
        'over_under': over_under,
        'over_odds': over_odds,
        'under_odds': under_odds
    }

def process_oddsapi_games_for_prediction(num_games: int = 5) -> pd.DataFrame:
    """
    Fetch and process Odds API games for ML prediction and betting logic.
    Returns a DataFrame with all required features for the strategy.
    """
    config = APIConfig()
    games = fetch_all_upcoming_nfl_games_oddsapi(config)
    if not games:
        print("No games found from Odds API.")
        return pd.DataFrame()
    # Limit to requested number of games
    games = games[:num_games]
    rows = []
    for game in games:
        odds_features = extract_odds_features_from_oddsapi(game)
        row = {
            'game_id': game['game_id'],
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            **odds_features
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def process_oddsapi_games_with_features(num_games: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch Odds API games, merge with historical data, and run full feature engineering pipeline.
    Returns dict with 'classification' and 'regression' DataFrames ready for model input.
    """
    # 1. Load historical data
    historical = load_and_clean_data()
    teams = load_teams_data()
    # 2. Fetch games from Odds API
    config = APIConfig()
    games = fetch_all_upcoming_nfl_games_oddsapi(config)
    if not games:
        print("No games found from Odds API.")
        return {'classification': pd.DataFrame(), 'regression': pd.DataFrame()}
    games = games[:num_games]
    # 3. Build DataFrame for new games
    future_games = []
    for game in games:
        odds_features = extract_odds_features_from_oddsapi(game)
        game_row = {
            'schedule_date': pd.to_datetime(game['date']),
            'schedule_season': pd.to_datetime(game['date']).year,
            'schedule_week': None,  # Will be calculated
            'schedule_playoff': 0,
            'team_home': game['home_team'],
            'team_away': game['away_team'],
            'stadium_neutral': 0,
            'stadium': '',
            'spread_favorite': odds_features['spread_favorite'],
            'over_under_line': odds_features['over_under'],
            'home_is_favorite': 1 if odds_features['favorite_team'] == game['home_team'] else 0
        }
        future_games.append(game_row)
    future_df = pd.DataFrame(future_games)
    if future_df.empty:
        return {'classification': pd.DataFrame(), 'regression': pd.DataFrame()}
    # 4. Calculate schedule week
    future_df['schedule_week'] = future_df['schedule_date'].dt.isocalendar().week
    # 5. Combine with historical for feature engineering
    combined = pd.concat([historical, future_df], ignore_index=True)
    combined['schedule_date'] = pd.to_datetime(combined['schedule_date'])
    combined = combined.sort_values('schedule_date')
    # 6. Feature engineering
    combined = compute_team_rolling_averages(combined)
    combined = compute_streaks_and_last5(combined)
    combined['head_to_head'] = calculate_head_to_head(combined)
    combined = add_rest_days_features(combined)
    combined = calculate_elo_ratings(combined, teams)
    combined = final_feature_engineering(combined, teams)
    # 7. Extract only the new games
    future_with_features = combined.tail(len(future_games))
    # 8. Prepare for model input
    classification_df = future_with_features.drop(
        columns=['spread_favorite', 'over_under_line', 'score_home', 'score_away', 'home_team_wins'], errors='ignore')
    regression_df = future_with_features.drop(
        columns=['home_team_wins'], errors='ignore')
    classification_df = classification_df.dropna()
    regression_df = regression_df.dropna()
    return {'classification': classification_df, 'regression': regression_df}

if __name__ == "__main__":
    # Test the future game processor
    prediction_datasets = fetch_and_process_future_games()
    
    print("\n📊 Future Games Summary:")
    print(f"Classification dataset shape: {prediction_datasets['classification'].shape}")
    print(f"Regression dataset shape: {prediction_datasets['regression'].shape}")
    
    if not prediction_datasets['classification'].empty:
        print("\n🔍 Sample classification features:")
        print(prediction_datasets['classification'].columns.tolist()[:10])
    
    if not prediction_datasets['regression'].empty:
        print("\n🔍 Sample regression features:")
        print(prediction_datasets['regression'].columns.tolist()[:10]) 