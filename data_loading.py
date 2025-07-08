#https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data/discussion/550839
import pandas as pd
import numpy as np

def load_and_clean_data():
    """
    Load and perform initial cleaning of the NFL scores dataset.
    Returns the cleaned DataFrame.
    """
    teams_scores = pd.read_csv("spreadspoke_scores.csv")
    
    # Filter data from 1979 onwards
    teams_scores = teams_scores[teams_scores['schedule_season'] >= 1979]
    
    # Fill missing weather data
    teams_scores['weather_detail'] = teams_scores['weather_detail'].fillna('none')
    teams_scores['weather_humidity'] = teams_scores['weather_humidity'].fillna(teams_scores['weather_humidity'].median())
    teams_scores['weather_wind_mph'] = teams_scores['weather_wind_mph'].fillna(teams_scores['weather_wind_mph'].median())
    teams_scores['weather_temperature'] = teams_scores['weather_temperature'].fillna(teams_scores['weather_temperature'].mean())

    # Clean over_under_line column
    teams_scores['over_under_line'] = teams_scores['over_under_line'].astype(str).str.strip()
    teams_scores['over_under_line'] = teams_scores['over_under_line'].replace('', np.nan)
    teams_scores['over_under_line'] = teams_scores['over_under_line'].astype(float)

    # Drop rows with missing values
    teams_scores = teams_scores.dropna()

    # Update franchise names to current names
    franchise_updates = {
        "Baltimore Colts": "Indianapolis Colts",
        "Boston Patriots": "New England Patriots",
        "Houston Oilers": "Tennessee Titans",
        "Los Angeles Raiders": "Las Vegas Raiders",
        "Oakland Raiders": "Las Vegas Raiders",
        "Pheonix Cardinals": "Arizona Cardinals",
        "San Diego Chargers": "Los Angeles Chargers",
        "St. Louis Cardinals": "Arizona Cardinals",
        "St. Louis Rams": "Los Angeles Rams",
        "Tennessee Oilers": "Tennessee Titans",
        "Washington Redskins": "Washington Commanders",
        "Washington Football Team": "Washington Commanders"
    }

    # Replace old franchise names with new ones
    teams_scores['team_home'] = teams_scores['team_home'].replace(franchise_updates)
    teams_scores['team_away'] = teams_scores['team_away'].replace(franchise_updates)

    # Create home team wins column
    teams_scores['home_team_wins'] = (teams_scores['score_home'] > teams_scores['score_away']).astype(int)
    
    return teams_scores

def load_teams_data():
    """
    Load and clean the teams data.
    Returns the cleaned teams DataFrame.
    """
    teams = pd.read_csv("nfl_teams.csv")
    teams = teams.drop([2, 4, 14, 21, 29, 31, 33, 36, 37, 39, 42, 43])
    return teams 