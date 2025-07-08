"""
Prediction Engine for NFL Games
Loads trained models and makes predictions on future games.
"""

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from future_game_processor import fetch_and_process_future_games, process_oddsapi_games_for_prediction, process_oddsapi_games_with_features

class NFLPredictionEngine:
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the prediction engine.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.classification_model = None
        self.home_regression_model = None
        self.away_regression_model = None
        self.regression_scaler = None
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            print("📥 Loading trained models...")
            
            # Load classification model (XGBoost Booster)
            class_model_path = os.path.join(self.models_dir, "final_class_model.json")
            if os.path.exists(class_model_path):
                self.classification_model = xgb.Booster()
                self.classification_model.load_model(class_model_path)
                print("✅ Classification model loaded")
            else:
                print("❌ Classification model not found")
                return False
            
            # Load regression models (XGBRegressor)
            home_model_path = os.path.join(self.models_dir, "final_home_model.joblib")
            away_model_path = os.path.join(self.models_dir, "final_away_model.joblib")
            
            if os.path.exists(home_model_path) and os.path.exists(away_model_path):
                self.home_regression_model = joblib.load(home_model_path)
                self.away_regression_model = joblib.load(away_model_path)
                print("✅ Regression models loaded")
            else:
                print("❌ Regression models not found")
                return False
            
            # Load regression scaler
            scaler_path = os.path.join(self.models_dir, "regression_scaler.joblib")
            if os.path.exists(scaler_path):
                self.regression_scaler = joblib.load(scaler_path)
                print("✅ Regression scaler loaded")
            else:
                print("⚠️ Regression scaler not found (will use unscaled features)")
            
            self.models_loaded = True
            print("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_classification(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make classification predictions (home team wins).
        
        Args:
            features_df: DataFrame with features for prediction
        
        Returns:
            Tuple of (probabilities, predictions)
        """
        if not self.models_loaded or self.classification_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Prepare features (drop target column if present)
        if 'home_team_wins' in features_df.columns:
            X = features_df.drop(columns=['home_team_wins'])
        else:
            X = features_df
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(X)
        
        # Make predictions
        probabilities = self.classification_model.predict(dmatrix)
        predictions = (probabilities > 0.5).astype(int)
        
        return probabilities, predictions
    
    def predict_regression(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make regression predictions (home and away scores).
        
        Args:
            features_df: DataFrame with features for prediction
        
        Returns:
            Tuple of (home_scores, away_scores)
        """
        if not self.models_loaded or self.home_regression_model is None or self.away_regression_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Prepare features (drop target columns if present)
        target_cols = ['score_home', 'score_away']
        X = features_df.drop(columns=[col for col in target_cols if col in features_df.columns])
        
        # Scale features if scaler is available
        if self.regression_scaler is not None:
            X_scaled = self.regression_scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Make predictions
        home_scores = self.home_regression_model.predict(X)
        away_scores = self.away_regression_model.predict(X)
        
        return home_scores, away_scores
    
    def predict_future_games(self, days_ahead: int = 30) -> pd.DataFrame:
        """
        Predict outcomes for future NFL games.
        
        Args:
            days_ahead: Number of days to look ahead for games
        
        Returns:
            DataFrame with predictions for future games
        """
        if not self.models_loaded:
            if not self.load_models():
                raise ValueError("Failed to load models")
        
        print(f"🔮 Making predictions for games in the next {days_ahead} days...")
        
        # Fetch and process future games
        prediction_datasets = fetch_and_process_future_games(days_ahead)
        
        if prediction_datasets['classification'].empty:
            print("❌ No future games found for prediction")
            return pd.DataFrame()
        
        # Make predictions
        results = []
        
        for idx, game in prediction_datasets['classification'].iterrows():
            try:
                # Classification prediction
                game_features = game.drop('home_team_wins', errors='ignore')
                win_prob, win_pred = self.predict_classification(pd.DataFrame([game_features]))
                
                # Regression prediction
                if not prediction_datasets['regression'].empty:
                    reg_game = prediction_datasets['regression'].iloc[idx] if idx < len(prediction_datasets['regression']) else None
                    if reg_game is not None:
                        reg_features = reg_game.drop(['score_home', 'score_away'], errors='ignore')
                        home_score, away_score = self.predict_regression(pd.DataFrame([reg_features]))
                    else:
                        home_score, away_score = [None], [None]
                else:
                    home_score, away_score = [None], [None]
                
                # Create result row
                result = {
                    'date': game.get('schedule_date'),
                    'home_team': game.get('team_home'),
                    'away_team': game.get('team_away'),
                    'venue': game.get('stadium'),
                    'home_win_probability': float(win_prob[0]),
                    'home_win_prediction': int(win_pred[0]),
                    'predicted_home_score': float(home_score[0]) if home_score[0] is not None else None,
                    'predicted_away_score': float(away_score[0]) if away_score[0] is not None else None,
                    'predicted_total': float(home_score[0] + away_score[0]) if home_score[0] is not None and away_score[0] is not None else None,
                    'confidence': abs(win_prob[0] - 0.5) * 2  # Distance from 0.5
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error predicting game {game.get('team_home')} vs {game.get('team_away')}: {e}")
                continue
        
        if not results:
            print("❌ No predictions generated")
            return pd.DataFrame()
        
        # Create results DataFrame
        predictions_df = pd.DataFrame(results)
        predictions_df = predictions_df.sort_values('date')
        
        print(f"✅ Generated predictions for {len(predictions_df)} games")
        return predictions_df
    
    def format_predictions(self, predictions_df: pd.DataFrame) -> str:
        """
        Format predictions into a readable string.
        
        Args:
            predictions_df: DataFrame with predictions
        
        Returns:
            Formatted string with predictions
        """
        if predictions_df.empty:
            return "No predictions available."
        
        output = []
        output.append("🏈 NFL Game Predictions")
        output.append("=" * 50)
        
        for _, game in predictions_df.iterrows():
            date_str = game['date'].strftime('%Y-%m-%d') if pd.notna(game['date']) else 'Unknown'
            
            output.append(f"\n📅 {date_str}")
            output.append(f"🏠 {game['home_team']} vs 🚌 {game['away_team']}")
            output.append(f"📍 {game['venue']}")
            
            # Win prediction
            win_prob = game['home_win_probability']
            confidence = game['confidence']
            if game['home_win_prediction'] == 1:
                output.append(f"🎯 Prediction: {game['home_team']} wins ({win_prob:.1%} probability)")
            else:
                output.append(f"🎯 Prediction: {game['away_team']} wins ({(1-win_prob):.1%} probability)")
            
            output.append(f"📊 Confidence: {confidence:.1%}")
            
            # Score prediction
            if game['predicted_home_score'] is not None and game['predicted_away_score'] is not None:
                home_score = round(game['predicted_home_score'])
                away_score = round(game['predicted_away_score'])
                total = round(game['predicted_total'])
                output.append(f"📈 Predicted Score: {home_score}-{away_score} (Total: {total})")
            
            output.append("-" * 30)
        
        return "\n".join(output)

    def predict_for_games(self, games_df: pd.DataFrame, num_games: int = 5) -> pd.DataFrame:
        if not self.models_loaded:
            if not self.load_models():
                raise ValueError("Failed to load models")
        # Get engineered features for model input
        features = process_oddsapi_games_with_features(num_games)
        if features['classification'].empty or features['regression'].empty:
            print("No games with valid features for prediction.")
            return pd.DataFrame()
        # Classification prediction
        class_X = features['classification']
        dmatrix = xgb.DMatrix(class_X)
        win_probs = self.classification_model.predict(dmatrix)
        win_preds = (win_probs > 0.5).astype(int)
        # Regression prediction
        reg_X = features['regression']
        X_reg = reg_X.drop(columns=['score_home', 'score_away'], errors='ignore')
        if self.regression_scaler is not None:
            X_reg_scaled = self.regression_scaler.transform(X_reg)
            X_reg = pd.DataFrame(X_reg_scaled, columns=X_reg.columns, index=X_reg.index)
        home_scores = self.home_regression_model.predict(X_reg)
        away_scores = self.away_regression_model.predict(X_reg)
        # Merge predictions with original odds/game info
        results = []
        for i in range(len(class_X)):
            row = games_df.iloc[i] if i < len(games_df) else {}
            # Determine favorite/underdog
            fav = row.get('favorite_team')
            dog = row.get('underdog_team')
            spread = row.get('spread')
            ml_home = row.get('moneyline_home')
            ml_away = row.get('moneyline_away')
            ou = row.get('over_under')
            ou_over_odds = row.get('over_odds')
            ou_under_odds = row.get('under_odds')
            # Model outputs
            win_prob = win_probs[i]
            win_pred = win_preds[i]
            predicted_home_score = home_scores[i]
            predicted_away_score = away_scores[i]
            predicted_total = predicted_home_score + predicted_away_score
            # Margin: favorite - underdog
            if fav == row.get('home_team'):
                predicted_margin = predicted_home_score - predicted_away_score
            else:
                predicted_margin = predicted_away_score - predicted_home_score
            predicted_winner = 'favorite' if win_pred == 1 else 'underdog'
            results.append({
                **row,
                'predicted_winner': predicted_winner,
                'predicted_margin': predicted_margin,
                'predicted_home_score': predicted_home_score,
                'predicted_away_score': predicted_away_score,
                'confidence': win_prob
            })
        return pd.DataFrame(results)

def main():
    """
    Main function to run predictions on future games.
    """
    print("🚀 Starting NFL Prediction Engine")
    
    # Initialize prediction engine
    engine = NFLPredictionEngine()
    
    # Load models
    if not engine.load_models():
        print("❌ Failed to load models. Please run train_models.py first.")
        return
    
    # Make predictions
    predictions = engine.predict_future_games(days_ahead=30)
    
    if not predictions.empty:
        # Display formatted predictions
        formatted_output = engine.format_predictions(predictions)
        print(formatted_output)
        
        # Save predictions to CSV
        predictions.to_csv('future_game_predictions.csv', index=False)
        print("\n💾 Predictions saved to 'future_game_predictions.csv'")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    main() 