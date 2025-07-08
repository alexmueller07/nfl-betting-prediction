import pandas as pd
import numpy as np

def compute_team_rolling_averages(df):
    """
    For each row, compute rolling averages (last up to 5 games) for both home and away teams:
    - Points scored
    - Points allowed
    Only includes games that occurred BEFORE the current game.
    """
    df = df.copy()
    df = df.sort_values("schedule_date").reset_index(drop=True)

    # Initialize output columns
    df['home_avg_scored'] = np.nan
    df['home_avg_allowed'] = np.nan
    df['away_avg_scored'] = np.nan
    df['away_avg_allowed'] = np.nan

    # Dict to store past games for each team
    team_history = {}

    for idx, row in df.iterrows():
        home = row['team_home']
        away = row['team_away']
        home_score = row['score_home']
        away_score = row['score_away']

        # Ensure team history entries exist
        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        # Look up past games for each team (exclude current game)
        home_past = team_history[home][-5:]
        away_past = team_history[away][-5:]

        # Compute home team rolling averages
        if home_past:
            scored = [g[0] for g in home_past]
            allowed = [g[1] for g in home_past]
            df.at[idx, 'home_avg_scored'] = np.mean(scored)
            df.at[idx, 'home_avg_allowed'] = np.mean(allowed)

        # Compute away team rolling averages
        if away_past:
            scored = [g[0] for g in away_past]
            allowed = [g[1] for g in away_past]
            df.at[idx, 'away_avg_scored'] = np.mean(scored)
            df.at[idx, 'away_avg_allowed'] = np.mean(allowed)

        # After computing, update each team's history
        team_history[home].append((home_score, away_score))
        team_history[away].append((away_score, home_score))

    return df

def compute_streaks_and_last5(df):
    """
    Compute win/loss streaks and last 5 games performance for each team.
    """
    # Ensure the DataFrame is sorted chronologically
    df = df.copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"])
    df = df.sort_values("schedule_date").reset_index(drop=True)

    # Create win/loss result for home and away teams
    df["home_result"] = (df["score_home"] > df["score_away"]).astype(int).replace({1: 1, 0: -1})
    df["away_result"] = -df["home_result"]

    # Containers to store the results
    home_streaks, away_streaks = [], []
    home_last5s, away_last5s = [], []

    # Dictionary to track each team's past game results
    team_results = {}

    # Iterate over rows in order
    for idx, row in df.iterrows():
        home_team = row["team_home"]
        away_team = row["team_away"]
        home_result = row["home_result"]
        away_result = row["away_result"]

        # Get past results for both teams
        home_history = team_results.get(home_team, [])
        away_history = team_results.get(away_team, [])

        # Compute streaks
        def compute_streak(history):
            if not history:
                return None
            streak = 0
            last = history[-1]
            for res in reversed(history):
                if res == last:
                    streak += res
                else:
                    break
            return streak

        # Compute last5s
        def compute_last5(history):
            if not history:
                return None
            return sum(history[-5:])

        # Append streaks and last5s for this game
        home_streaks.append(compute_streak(home_history))
        away_streaks.append(compute_streak(away_history))
        home_last5s.append(compute_last5(home_history))
        away_last5s.append(compute_last5(away_history))

        # Update team histories AFTER calculating
        team_results.setdefault(home_team, []).append(home_result)
        team_results.setdefault(away_team, []).append(away_result)

    # Assign to DataFrame
    df["home_streak"] = home_streaks
    df["away_streak"] = away_streaks
    df["home_last5"] = home_last5s
    df["away_last5"] = away_last5s

    df = df.drop(['home_result', 'away_result'], axis=1)
    return df

def calculate_head_to_head(df):
    """
    Calculate head-to-head record between teams based on their last 5 matchups.
    """
    head_to_head_results = []

    # Ensure proper sorting
    df = df.sort_values('schedule_date').reset_index(drop=True)

    for idx, row in df.iterrows():
        date = row['schedule_date']
        team_home = row['team_home']
        team_away = row['team_away']

        # Find previous matchups between the same two teams
        past_matchups = df[
            (((df['team_home'] == team_home) & (df['team_away'] == team_away)) |
             ((df['team_home'] == team_away) & (df['team_away'] == team_home))) &
            (df['schedule_date'] < date)
        ].sort_values('schedule_date')

        # Take up to the last 5 games
        past_matchups = past_matchups.tail(5)

        score = 0
        for _, match in past_matchups.iterrows():
            # Determine who won the past game
            if match['score_home'] > match['score_away']:
                winner = match['team_home']
            elif match['score_home'] < match['score_away']:
                winner = match['team_away']
            else:
                continue  # Tie or draw — skip, or you could count as 0

            # If the current home team was the winner, +1
            if winner == team_home:
                score += 1
            elif winner == team_away:
                score -= 1
            # else: one of the teams is different (edge case — skip)

        head_to_head_results.append(score)

    return head_to_head_results

def get_rest_days(df, team_col):
    """
    Calculate rest days between games for each team.
    """
    rest = []

    for team in df[team_col].unique():
        team_df = df[(df['team_home'] == team) | (df['team_away'] == team)].copy()
        team_df = team_df.sort_values('schedule_date')
        team_df['last_game'] = team_df['schedule_date'].shift()
        team_df['rest_days'] = (team_df['schedule_date'] - team_df['last_game']).dt.days
        team_df['team'] = team
        rest.append(team_df[['schedule_date', 'rest_days', 'team']])

    return pd.concat(rest)

def add_rest_days_features(df):
    """
    Add rest days features for both home and away teams.
    """
    df = df.copy()
    df['schedule_date'] = pd.to_datetime(df['schedule_date'])
    
    home_rest = get_rest_days(df, 'team_home')
    away_rest = get_rest_days(df, 'team_away')

    df = df.merge(
        home_rest.rename(columns={'rest_days': 'home_rest_days'}),
        left_on=['team_home', 'schedule_date'],
        right_on=['team', 'schedule_date'],
        how='left'
    )

    df = df.merge(
        away_rest.rename(columns={'rest_days': 'away_rest_days'}),
        left_on=['team_away', 'schedule_date'],
        right_on=['team', 'schedule_date'],
        how='left'
    )

    df.drop(columns=['team_x', 'team_y'], inplace=True)
    return df

def determine_home_favorite(row):
    """
    Determine if home team is favorite, underdog, or if it's a pick.
    """
    if row['team_favorite_id'] == 'PICK':
        return -1
    return 1 if row['team_favorite_id'] == row['home_team_id'] else 0

def convert_week_to_numeric(x):
    """
    Convert playoff week labels to numeric values.
    """
    playoff_week_mapping = {
        "wildcard": 19,
        "division": 20,
        "conference": 21,
        "superbowl": 22,
        "super bowl": 22
    }
    
    try:
        return int(x)
    except:
        key = str(x).strip().lower()
        return playoff_week_mapping.get(key, None)

def final_feature_engineering(df, teams):
    """
    Perform final feature engineering steps including encoding and cleaning.
    """
    df = df.copy()
    
    # Map full team names to team IDs using the 'teams' DataFrame
    team_name_to_id = teams.set_index('team_name')['team_id'].to_dict()

    # Create a new column that maps each home team to its team_id
    df['home_team_id'] = df['team_home'].map(team_name_to_id)

    # Generate 'home_is_favorite' with logic for PICK
    df['home_is_favorite'] = df.apply(determine_home_favorite, axis=1)

    # Replace 'team_favorite_id' with 'home_is_favorite'
    favorite_idx = df.columns.get_loc('team_favorite_id')
    df.drop(columns=['team_favorite_id'], inplace=True)
    cols = list(df.columns)
    # Move 'home_is_favorite' into the original 'team_favorite_id' position
    cols.insert(favorite_idx, cols.pop(cols.index('home_is_favorite')))
    df = df[cols]

    # Clean up
    df.drop(columns=['home_team_id'], inplace=True)

    # Ensure binary columns are integers
    df['schedule_playoff'] = df['schedule_playoff'].astype(int)
    df['stadium_neutral'] = df['stadium_neutral'].astype(int)

    # Clean and standardize playoff round labels
    df['schedule_week'] = df['schedule_week'].apply(convert_week_to_numeric)

    # Drop unused columns if they exist
    df.drop(columns=['away_team_id', 'stadium'], errors='ignore', inplace=True)

    # Add a column 'is_weekend': 1 if Saturday or Sunday, else 0
    df['is_weekend'] = df['schedule_date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Drop the original 'schedule_date' column
    df = df.drop(columns=['schedule_date'])

    # One-hot encode team_home, team_away, and weather_detail
    one_hot_encoded = pd.get_dummies(df[['team_home', 'team_away', 'weather_detail']],
                                      prefix=['home', 'away', 'weather'])

    # Drop original columns
    df = df.drop(columns=['team_home', 'team_away', 'weather_detail'])

    # Concatenate one-hot encoded columns
    df = pd.concat([df, one_hot_encoded], axis=1)

    # Convert all boolean columns (True/False) to integers (1/0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.dropna()
    return df 