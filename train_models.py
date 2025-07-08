"""
Train NFL ML models and save them to disk for future use.
"""

from data_loading import load_and_clean_data, load_teams_data
from feature_engineering import (
    compute_team_rolling_averages,
    compute_streaks_and_last5,
    calculate_head_to_head,
    add_rest_days_features,
    final_feature_engineering
)
from elo_system import calculate_elo_ratings
from classification_model import (
    train_classification_model,
    train_final_classification_model
)
from regression_model import (
    train_regression_models,
    train_final_regression_models
)
import joblib
import os
import pickle

def train_and_save_models():
    print("🚀 Starting NFL Model Training and Saving Script")
    print("=" * 50)
    # Step 1: Load and clean data
    print("📊 Loading and cleaning data...")
    teams_scores = load_and_clean_data()
    teams = load_teams_data()
    print(f"✅ Loaded {len(teams_scores)} games and {len(teams)} teams")

    # Step 2: Feature engineering
    print("\n🔧 Performing feature engineering...")
    teams_scores = compute_team_rolling_averages(teams_scores)
    teams_scores = compute_streaks_and_last5(teams_scores)
    teams_scores['head_to_head'] = calculate_head_to_head(teams_scores)
    teams_scores = add_rest_days_features(teams_scores)
    teams_scores = calculate_elo_ratings(teams_scores, teams)
    teams_scores = final_feature_engineering(teams_scores, teams)
    print(f"✅ Feature engineering complete. Final dataset has {teams_scores.shape[1]} features")

    # Step 3: Prepare data for models
    print("\n🤖 Preparing data for machine learning models...")
    teams_scores_classification = teams_scores.drop(columns=['spread_favorite', 'over_under_line', 'score_home', 'score_away'])
    teams_scores_regression = teams_scores.drop(columns=['home_team_wins'])
    print(f"✅ Classification dataset: {teams_scores_classification.shape}")
    print(f"✅ Regression dataset: {teams_scores_regression.shape}")

    # Step 4: Train and evaluate models
    print("\n🎯 Training and evaluating models...")
    class_model, X_class, y_class = train_classification_model(teams_scores_classification)
    home_model, away_model, scaled_teams_scores, X_reg, scaler = train_regression_models(teams_scores)

    # Step 5: Train final models on full dataset
    print("\n🏆 Training final models on full dataset...")
    final_class_model = train_final_classification_model(teams_scores_classification)
    final_home_model, final_away_model = train_final_regression_models(scaled_teams_scores)

    # Save models to disk
    print("\n💾 Saving models to disk...")
    os.makedirs("models", exist_ok=True)
    # Save classification model (XGBoost Booster)
    final_class_model.save_model("models/final_class_model.json")
    # Save regression models (XGBRegressor)
    joblib.dump(final_home_model, "models/final_home_model.joblib")
    joblib.dump(final_away_model, "models/final_away_model.joblib")
    # Save the scaler used for regression
    if scaler is not None:
        joblib.dump(scaler, "models/regression_scaler.joblib")
    # Save feature names for prediction compatibility
    with open("models/model_features_classification.pkl", "wb") as f:
        pickle.dump(list(teams_scores_classification.drop(columns=['home_team_wins']).columns), f)
    with open("models/model_features_regression.pkl", "wb") as f:
        pickle.dump(list(teams_scores_regression.columns), f)
    print("✅ Models saved in 'models/' directory:")
    print("  - final_class_model.json (XGBoost Booster)")
    print("  - final_home_model.joblib (XGBRegressor)")
    print("  - final_away_model.joblib (XGBRegressor)")
    print("  - regression_scaler.joblib (if available)")
    print("  - model_features_classification.pkl (feature list)")
    print("  - model_features_regression.pkl (feature list)")
    print("\n🎉 Training and saving complete!")

if __name__ == "__main__":
    train_and_save_models() 