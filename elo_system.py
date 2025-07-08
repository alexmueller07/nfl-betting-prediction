import pandas as pd
import numpy as np

START_ELO = 1000

def margin_multiplier(margin):
    """
    Calculate margin multiplier for Elo adjustments.
    """
    return 1 + min(margin, 30) / 30

def calculate_elo_ratings(df, teams):
    """
    Calculate Elo ratings for all teams across different categories.
    """
    df = df.copy()
    
    # Initialize Elo structures
    elo_dict = {
        'general': {}, 'home': {}, 'away': {}, 'playoff': {},
        'recent': {}, 'favorite': {}, 'underdog': {}, 'bad_weather': {}
    }
    recent_games = {}

    # Add team IDs
    df['home_team_id'] = df['team_home'].map(
        teams.set_index('team_name')['team_id']
    )
    df['away_team_id'] = df['team_away'].map(
        teams.set_index('team_name')['team_id']
    )

    # Drop indoor elos (not using them anymore)
    elo_columns = [
        'home_elo', 'away_elo',
        'home_home_elo', 'home_away_elo', 'away_home_elo', 'away_away_elo',
        'home_playoff_elo', 'away_playoff_elo',
        'home_recent_elo', 'away_recent_elo',
        'home_favorite_elo', 'home_underdog_elo', 'away_favorite_elo', 'away_underdog_elo',
        'home_bad_weather_elo', 'away_bad_weather_elo'
    ]
    for col in elo_columns:
        df[col] = np.nan

    last_reset_year = None

    for idx, row in df.iterrows():
        year = row['schedule_season']
        home_team = row['home_team_id']
        away_team = row['away_team_id']
        home_score = row['score_home']
        away_score = row['score_away']
        favorite_id = row['team_favorite_id']
        week = row['schedule_week']
        weather = row['weather_detail']
        is_playoff = not str(week).isdigit()
        is_bad_weather = weather not in ['Indoor', 'None', np.nan, None]

        # Reset Elo every 3 years
        if last_reset_year is None or (year - last_reset_year >= 3):
            for d in elo_dict:
                for team in set(df['home_team_id']).union(set(df['away_team_id'])):
                    elo_dict[d][team] = START_ELO
            recent_games = {team: [] for team in elo_dict['general'].keys()}
            last_reset_year = year

        # Ensure teams are initialized
        for d in elo_dict:
            for team in [home_team, away_team]:
                if team not in elo_dict[d]:
                    elo_dict[d][team] = START_ELO
        for team in [home_team, away_team]:
            if team not in recent_games:
                recent_games[team] = []

        elos = {
            'home': {
                'general': elo_dict['general'][home_team],
                'home': elo_dict['home'][home_team],
                'away': elo_dict['away'][home_team],
                'playoff': elo_dict['playoff'][home_team],
                'recent': elo_dict['recent'][home_team],
                'favorite': elo_dict['favorite'][home_team],
                'underdog': elo_dict['underdog'][home_team],
                'bad_weather': elo_dict['bad_weather'][home_team],
            },
            'away': {
                'general': elo_dict['general'][away_team],
                'home': elo_dict['home'][away_team],
                'away': elo_dict['away'][away_team],
                'playoff': elo_dict['playoff'][away_team],
                'recent': elo_dict['recent'][away_team],
                'favorite': elo_dict['favorite'][away_team],
                'underdog': elo_dict['underdog'][away_team],
                'bad_weather': elo_dict['bad_weather'][away_team],
            }
        }

        # Assign Elos to DataFrame
        df.at[idx, 'home_elo'] = elos['home']['general']
        df.at[idx, 'away_elo'] = elos['away']['general']

        df.at[idx, 'home_home_elo'] = elos['home']['home']
        df.at[idx, 'home_away_elo'] = elos['home']['away']
        df.at[idx, 'away_home_elo'] = elos['away']['home']
        df.at[idx, 'away_away_elo'] = elos['away']['away']

        df.at[idx, 'home_playoff_elo'] = elos['home']['playoff']
        df.at[idx, 'away_playoff_elo'] = elos['away']['playoff']

        df.at[idx, 'home_recent_elo'] = elos['home']['recent']
        df.at[idx, 'away_recent_elo'] = elos['away']['recent']

        df.at[idx, 'home_favorite_elo'] = elos['home']['favorite']
        df.at[idx, 'home_underdog_elo'] = elos['home']['underdog']
        df.at[idx, 'away_favorite_elo'] = elos['away']['favorite']
        df.at[idx, 'away_underdog_elo'] = elos['away']['underdog']

        df.at[idx, 'home_bad_weather_elo'] = elos['home']['bad_weather']
        df.at[idx, 'away_bad_weather_elo'] = elos['away']['bad_weather']

        # Skip if no score
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        if home_score > away_score:
            winner, loser = home_team, away_team
            home_result, away_result = 1, 0
        elif home_score < away_score:
            winner, loser = away_team, home_team
            home_result, away_result = 0, 1
        else:
            continue  # skip ties

        margin = abs(home_score - away_score)
        multiplier = margin_multiplier(margin)
        base_change = 50

        home_elo = elos['home']['general']
        away_elo = elos['away']['general']

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        home_change = base_change * (home_result - expected_home) * multiplier
        away_change = base_change * (away_result - expected_away) * multiplier

        # Update Elo for each category
        elo_dict['general'][home_team] += home_change
        elo_dict['general'][away_team] += away_change

        elo_dict['home'][home_team] += home_change
        elo_dict['away'][away_team] += away_change

        if is_playoff:
            elo_dict['playoff'][home_team] += home_change
            elo_dict['playoff'][away_team] += away_change

        if is_bad_weather:
            elo_dict['bad_weather'][home_team] += home_change
            elo_dict['bad_weather'][away_team] += away_change

        if favorite_id == home_team:
            elo_dict['favorite'][home_team] += home_change
            elo_dict['underdog'][away_team] += away_change
        elif favorite_id == away_team:
            elo_dict['favorite'][away_team] += away_change
            elo_dict['underdog'][home_team] += home_change

        # Recent Elo (rolling 10 games)
        for team, change in [(home_team, home_change), (away_team, away_change)]:
            recent_games[team].append(change)
            if len(recent_games[team]) > 10:
                recent_games[team].pop(0)
            elo_dict['recent'][team] = START_ELO + sum(recent_games[team])

    return df 