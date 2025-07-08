import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def evaluate_regression(y_true, y_pred, label):
    """
    Evaluate regression model performance.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{label} Score Prediction:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

def train_regression_models(teams_scores):
    """
    Train XGBoost regression models for predicting home and away team scores.
    """
    print("=== Training Regression Models ===")
    
    # Prepare data for regression
    teams_scores_regression = teams_scores.drop(columns=['home_team_wins'])

    # Separate features and target
    target_columns = ['score_home', 'score_away']
    features = teams_scores_regression.drop(columns=target_columns)
    targets = teams_scores_regression[target_columns]

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Rebuild full scaled DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=teams_scores.index)

    # Add target columns back
    scaled_teams_scores = pd.concat([scaled_df, targets], axis=1)

    # Prepare Data
    y_home = scaled_teams_scores['score_home']
    y_away = scaled_teams_scores['score_away']

    # Drop the target columns from features
    X = scaled_teams_scores.drop(columns=['score_home', 'score_away'])

    # Single train-test split (for both targets at once)
    X_train, X_test, y_train_full, y_test_full = train_test_split(
        X, targets, test_size=0.2, random_state=42
    )

    # Separate individual target columns
    y_home_train = y_train_full['score_home']
    y_home_test = y_test_full['score_home']
    y_away_train = y_train_full['score_away']
    y_away_test = y_test_full['score_away']

    # Initialize Models
    home_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42
    )

    away_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=42
    )

    # Train Models
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    # Predict
    y_home_pred = home_model.predict(X_test)
    y_away_pred = away_model.predict(X_test)

    # Evaluate
    evaluate_regression(y_home_test, y_home_pred, "Home Team")
    evaluate_regression(y_away_test, y_away_pred, "Away Team")

    # Plot feature importances
    xgb.plot_importance(home_model, max_num_features=10, importance_type='gain', title='Feature Importance - Home')
    plt.show()

    xgb.plot_importance(away_model, max_num_features=10, importance_type='gain', title='Feature Importance - Away')
    plt.show()
    
    return home_model, away_model, scaled_teams_scores, X, scaler

def train_final_regression_models(scaled_teams_scores):
    """
    Train final regression models on the full dataset without test split.
    """
    print("=== Training Final Regression Models on Full Dataset ===")
    
    # Retrain Regression Models on All Data
    X_full_reg = scaled_teams_scores.drop(columns=['score_home', 'score_away'])
    y_home_full = scaled_teams_scores['score_home']
    y_away_full = scaled_teams_scores['score_away']

    # Re-initialize with best parameters
    final_home_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42
    )

    final_away_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=42
    )

    # Train on full dataset
    final_home_model.fit(X_full_reg, y_home_full)
    final_away_model.fit(X_full_reg, y_away_full)

    print("✅ Final regression models trained on full dataset.")
    
    return final_home_model, final_away_model 