import pandas as pd
import numpy as np
from data_loading import load_and_clean_data, load_teams_data
from feature_engineering import (
    compute_team_rolling_averages,
    compute_streaks_and_last5,
    calculate_head_to_head,
    add_rest_days_features,
    final_feature_engineering
)
from elo_system import calculate_elo_ratings
from prediction_engine import NFLPredictionEngine
import argparse
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import random

# Realistic odds ranges (instead of fixed odds)
SPREAD_ODDS_RANGE = (-120, -110)  # -120 to -110
ML_FAV_RANGE = (-250, -150)       # -250 to -150 for favorites
ML_DOG_RANGE = (120, 300)         # +120 to +300 for underdogs
OU_ODDS_RANGE = (-120, -110)      # -120 to -110

# Strategy parameters - MONEYLINE FOCUS
MIN_EDGE_PCT = 0.015  # 1.5% minimum edge (very aggressive for more ML bets)
MAX_BETS_PER_WEEK = 7  # More bets per week
MAX_KELLY_PCT = 0.07  # Max 7% of bankroll per bet (aggressive)
MIN_CONFIDENCE = 0.51  # Lower confidence threshold for more ML bets

# Bet type filters - MONEYLINE FOCUS
ENABLE_SPREAD_BETS = False  # Disable spread bets
ENABLE_MONEYLINE_BETS = True
ENABLE_OU_BETS = True  # Keep O/U as secondary

# Aggressive thresholds for moneyline
ML_FAV_THRESHOLD = 0.54  # Lower threshold for favorites
ML_DOG_THRESHOLD = 0.46  # Higher threshold for underdogs
OU_THRESHOLD = 5  # O/U threshold remains

# Add at the top, after other parameters
MAX_BET_ABS = 1000  # Absolute max bet per game
MAX_BET_PCT = 0.02  # 2% of current bankroll

def calculate_implied_probability(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_edge(model_prob, odds):
    """Calculate the edge: model probability minus implied probability."""
    implied_prob = calculate_implied_probability(odds)
    return model_prob - implied_prob

def kelly_criterion(prob, odds):
    """Calculate Kelly Criterion bet size."""
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    q = 1 - prob
    k = (prob * b - q) / b
    return max(0, min(k, MAX_KELLY_PCT))  # Cap at MAX_KELLY_PCT

def get_realistic_odds(bet_type, is_favorite=True, spread=None):
    """Get realistic odds within ranges."""
    if bet_type == "spread":
        return random.uniform(*SPREAD_ODDS_RANGE)
    elif bet_type == "moneyline":
        if is_favorite:
            return random.uniform(*ML_FAV_RANGE)
        else:
            return random.uniform(*ML_DOG_RANGE)
    elif bet_type in ["over", "under"]:
        return random.uniform(*OU_ODDS_RANGE)
    return -110  # Default

def place_bet(bet_type, team, amount, odds, line=None, edge=None, model_prob=None):
    return {
        'bet_type': bet_type,
        'team': team,
        'amount': amount,
        'odds': odds,
        'line': line,
        'edge': edge,
        'model_probability': model_prob
    }

def calibrate_model_probability(raw_prob, model_type="classification"):
    """Calibrate model probability to be more realistic and accurate."""
    if model_type == "classification":
        # Apply sigmoid-like calibration to make probabilities more realistic
        # This compresses extreme probabilities and makes them more conservative
        if raw_prob > 0.5:
            # For probabilities > 50%, compress them down
            calibrated = 0.5 + (raw_prob - 0.5) * 0.6
        else:
            # For probabilities < 50%, compress them up
            calibrated = 0.5 - (0.5 - raw_prob) * 0.6
        return max(0.15, min(0.85, calibrated))
    else:
        return raw_prob

def calculate_spread_probability(win_prob, predicted_margin, spread_line):
    """Calculate realistic spread cover probability."""
    # Start with classification probability
    base_prob = calibrate_model_probability(win_prob, "classification")
    
    # Adjust based on regression margin vs spread
    margin_ratio = predicted_margin / spread_line if spread_line > 0 else 0
    margin_ratio = max(-1, min(1, margin_ratio))  # Clamp to [-1, 1]
    
    # Conservative adjustment based on margin
    adjustment = margin_ratio * 0.15  # Max 15% adjustment
    final_prob = base_prob + adjustment
    
    return max(0.2, min(0.8, final_prob))

def calculate_ou_probability(predicted_total, ou_line, distance_from_line):
    """Calculate realistic over/under probability based on distance from line."""
    # Base probability starts at 50%
    base_prob = 0.5
    
    # Adjust based on distance, but be very conservative
    if distance_from_line > 0:
        # Max adjustment of 25% based on distance
        adjustment = min(0.25, distance_from_line / 30)
        if predicted_total > ou_line:  # Over
            return base_prob + adjustment
        else:  # Under
            return base_prob + adjustment
    return base_prob

def calculate_aggressive_probability(raw_prob, bet_type="classification"):
    """Calculate more aggressive probability that accounts for model uncertainty."""
    if bet_type == "classification":
        # Add uncertainty buffer - assume model is slightly better than random
        uncertainty_factor = 0.1  # 10% uncertainty buffer
        if raw_prob > 0.5:
            # For favorites, be slightly more aggressive
            adjusted = raw_prob + uncertainty_factor
        else:
            # For underdogs, be slightly more aggressive
            adjusted = raw_prob - uncertainty_factor
        return max(0.1, min(0.9, adjusted))
    else:
        return raw_prob

def calculate_momentum_edge(home_score, away_score, predicted_total, ou_line):
    """Calculate edge based on scoring momentum and regression prediction."""
    # If both teams are high-scoring, favor over
    # If both teams are low-scoring, favor under
    avg_score = (home_score + away_score) / 2
    
    if avg_score > 25 and predicted_total > ou_line + 3:
        return 0.65  # 65% confidence for over
    elif avg_score < 20 and predicted_total < ou_line - 3:
        return 0.65  # 65% confidence for under
    else:
        return 0.5  # Neutral

def main():
    parser = argparse.ArgumentParser(description='NFL ML Betting Bot Backtest')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Starting bankroll')
    parser.add_argument('--season', type=int, default=None, help='Single season year to backtest (e.g., 2022)')
    parser.add_argument('--start-season', type=int, default=None, help='Start season for multi-season backtest (e.g., 2022)')
    parser.add_argument('--end-season', type=int, default=None, help='End season for multi-season backtest (e.g., 2024)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.season and (args.start_season or args.end_season):
        print("❌ Error: Use either --season OR --start-season/--end-season, not both")
        return
    
    if args.start_season and not args.end_season:
        print("❌ Error: --start-season requires --end-season")
        return
    
    if args.end_season and not args.start_season:
        print("❌ Error: --end-season requires --start-season")
        return
    
    bankroll = args.bankroll
    starting_bankroll = bankroll
    
    print(f"\n🏈 NFL Machine Learning Betting Bot Backtest (IMPROVED STRATEGY)\n{'='*65}")
    print(f"Starting bankroll: ${bankroll:.2f}")
    print(f"Strategy: Only bet with {MIN_EDGE_PCT*100:.0f}%+ edge, max {MAX_BETS_PER_WEEK} bets/week")
    print(f"Max Kelly: {MAX_KELLY_PCT*100:.0f}% of bankroll per bet")
    print(f"Bet types: Spread={ENABLE_SPREAD_BETS}, ML={ENABLE_MONEYLINE_BETS}, O/U={ENABLE_OU_BETS}")
    
    # Determine seasons to test
    if args.season:
        seasons = [args.season]
        print(f"Testing single season: {args.season}")
    elif args.start_season and args.end_season:
        seasons = list(range(args.start_season, args.end_season + 1))
        print(f"Testing seasons: {args.start_season} to {args.end_season}")
    else:
        print("❌ Error: Must specify either --season OR --start-season/--end-season")
        return
    
    print()

    # 1. Load and engineer historical data
    df = load_and_clean_data()
    teams = load_teams_data()
    
    # Filter by seasons
    if seasons:
        df = df[df['schedule_season'].isin(seasons)]
        print(f"📊 Loaded {len(df)} games across {len(seasons)} season(s)")

    # Save columns needed for logging before feature engineering
    logging_cols = ['schedule_date', 'schedule_season', 'team_home', 'team_away', 'score_home', 'score_away', 'spread_favorite', 'over_under_line', 'home_team_wins']
    df_logging = df[logging_cols].reset_index(drop=True)

    # Now do feature engineering on a copy
    df = compute_team_rolling_averages(df)
    df = compute_streaks_and_last5(df)
    df['head_to_head'] = calculate_head_to_head(df)
    df = add_rest_days_features(df)
    df = calculate_elo_ratings(df, teams)
    df = final_feature_engineering(df, teams)
    df = df.reset_index(drop=True)

    # 2. Prepare for model input
    classification_df = df.drop(columns=['spread_favorite', 'over_under_line', 'score_home', 'score_away'], errors='ignore')
    regression_df = df.drop(columns=['home_team_wins'], errors='ignore')
    # Remove any rows with missing critical features
    classification_df = classification_df.dropna()
    regression_df = regression_df.dropna()

    # Drop target if present (before reindexing)
    if 'home_team_wins' in classification_df.columns:
        classification_df = classification_df.drop(columns=['home_team_wins'])

    # Load feature names and reindex for model compatibility
    with open('models/model_features_classification.pkl', 'rb') as f:
        classification_features = pickle.load(f)
    with open('models/model_features_regression.pkl', 'rb') as f:
        regression_features = pickle.load(f)
    classification_df = classification_df.reindex(columns=classification_features, fill_value=0)
    regression_df = regression_df.reindex(columns=regression_features, fill_value=0)

    # 3. Load models
    engine = NFLPredictionEngine()
    engine.load_models()

    # 4. Predict for all games
    # Classification
    dmatrix = xgb.DMatrix(classification_df)
    win_probs = engine.classification_model.predict(dmatrix)
    win_preds = (win_probs > 0.5).astype(int)
    # Regression
    X_reg = regression_df.drop(columns=['score_home', 'score_away'], errors='ignore')
    if engine.regression_scaler is not None:
        X_reg_scaled = engine.regression_scaler.transform(X_reg)
        X_reg = pd.DataFrame(X_reg_scaled, columns=X_reg.columns, index=X_reg.index)
    home_scores = engine.home_regression_model.predict(X_reg)
    away_scores = engine.away_regression_model.predict(X_reg)

    # 5. Enhanced betting strategy with edge calculation
    logs = []
    weekly_bets = {}  # Track bets per week
    
    for i in range(len(classification_df)):
        row = df_logging.iloc[i]
        home = row['team_home']
        away = row['team_away']
        season = row['schedule_season']
        actual_spread = row['score_home'] - row['score_away']
        ou_line = row.get('over_under_line', None)
        actual_total = row['score_home'] + row['score_away']
        
        # Model outputs
        win_prob = win_probs[i]
        win_pred = win_preds[i]
        predicted_home_score = home_scores[i]
        predicted_away_score = away_scores[i]
        predicted_total = predicted_home_score + predicted_away_score
        predicted_margin = predicted_home_score - predicted_away_score
        
        # Determine favorite/underdog based on actual spread
        favorite = home if actual_spread < 0 else away
        underdog = away if favorite == home else home
        spread_line = abs(actual_spread)  # Use actual spread as line
        
        # Get week for bet limiting
        week = df.iloc[i]['schedule_week'] if 'schedule_week' in df.columns else 1
        week_key = f"{season}_{week}"
        if week_key not in weekly_bets:
            weekly_bets[week_key] = 0
            
        # Skip if we've already bet too much this week
        if weekly_bets[week_key] >= MAX_BETS_PER_WEEK:
            continue
            
        bet_list = []
        
        # MONEYLINE BETTING (AGGRESSIVE APPROACH)
        if ENABLE_MONEYLINE_BETS:
            ml_fav_odds = get_realistic_odds("moneyline", is_favorite=True)
            ml_dog_odds = get_realistic_odds("moneyline", is_favorite=False)
            
            # Use aggressive probability calculation
            aggressive_win_prob = calculate_aggressive_probability(win_prob, "classification")
            
            # Bet favorite if model favors (lower threshold)
            if aggressive_win_prob > ML_FAV_THRESHOLD and favorite == home:
                edge = calculate_edge(aggressive_win_prob, ml_fav_odds)
                if edge >= MIN_EDGE_PCT:
                    kelly_pct = kelly_criterion(aggressive_win_prob, ml_fav_odds)
                    bet_amount = kelly_pct * bankroll
                    bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                    if bet_amount > bet_cap:
                        print(f"[WARNING] Bet capped from ${bet_amount:.2f} to ${bet_cap:.2f}")
                        bet_amount = bet_cap
                    if bet_amount > 0:
                        bet_list.append(place_bet("moneyline", home, bet_amount, ml_fav_odds,
                                                edge=edge, model_prob=aggressive_win_prob))
            
            # Bet underdog if model strongly favors (higher threshold)
            if aggressive_win_prob < ML_DOG_THRESHOLD and favorite == home:
                underdog_prob = 1 - aggressive_win_prob
                edge = calculate_edge(underdog_prob, ml_dog_odds)
                if edge >= MIN_EDGE_PCT:
                    kelly_pct = kelly_criterion(underdog_prob, ml_dog_odds)
                    bet_amount = kelly_pct * bankroll
                    bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                    if bet_amount > bet_cap:
                        print(f"[WARNING] Bet capped from ${bet_amount:.2f} to ${bet_cap:.2f}")
                        bet_amount = bet_cap
                    if bet_amount > 0:
                        bet_list.append(place_bet("moneyline", away, bet_amount, ml_dog_odds,
                                                edge=edge, model_prob=underdog_prob))
        
        # OVER/UNDER BETTING (MOMENTUM-BASED APPROACH)
        if ENABLE_OU_BETS and ou_line is not None and not pd.isna(ou_line):
            ou_odds = get_realistic_odds("over")
            
            # Calculate distance from line
            distance_from_line = abs(predicted_total - ou_line)
            
            # Use momentum-based edge calculation
            momentum_prob = calculate_momentum_edge(predicted_home_score, predicted_away_score, 
                                                  predicted_total, ou_line)
            
            # Only bet if prediction is far enough from line
            if distance_from_line >= OU_THRESHOLD:
                if predicted_total > ou_line:  # Over prediction
                    # Combine regression prediction with momentum
                    final_prob = (momentum_prob + 0.6) / 2 if momentum_prob > 0.5 else 0.55
                    edge = calculate_edge(final_prob, ou_odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(final_prob, ou_odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        if bet_amount > bet_cap:
                            print(f"[WARNING] Bet capped from ${bet_amount:.2f} to ${bet_cap:.2f}")
                            bet_amount = bet_cap
                        if bet_amount > 0:
                            bet_list.append(place_bet("over", None, bet_amount, ou_odds,
                                                    line=ou_line, edge=edge, model_prob=final_prob))
                            
                elif predicted_total < ou_line:  # Under prediction
                    # Combine regression prediction with momentum
                    final_prob = (momentum_prob + 0.6) / 2 if momentum_prob > 0.5 else 0.55
                    edge = calculate_edge(final_prob, ou_odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(final_prob, ou_odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        if bet_amount > bet_cap:
                            print(f"[WARNING] Bet capped from ${bet_amount:.2f} to ${bet_cap:.2f}")
                            bet_amount = bet_cap
                        if bet_amount > 0:
                            bet_list.append(place_bet("under", None, bet_amount, ou_odds,
                                                    line=ou_line, edge=edge, model_prob=final_prob))
        
        # Sort bets by edge (highest first) and take only the best ones
        bet_list.sort(key=lambda x: x['edge'], reverse=True)
        bet_list = bet_list[:3]  # Max 3 bets per game (increased)
        
        # Simulate bet results and update bankroll
        for bet in bet_list:
            if weekly_bets[week_key] >= MAX_BETS_PER_WEEK:
                break
                
            result = None
            payout = 0
            
            if bet['bet_type'] == 'spread':
                # Win if bet on team covers the spread
                if bet['team'] == home:
                    covered = (row['score_home'] - row['score_away']) > -bet['line']
                else:
                    covered = (row['score_away'] - row['score_home']) > -bet['line']
                result = 'win' if covered else 'lose'
                
            elif bet['bet_type'] == 'moneyline':
                # Win if bet on team wins
                if bet['team'] == home:
                    won = row['score_home'] > row['score_away']
                else:
                    won = row['score_away'] > row['score_home']
                result = 'win' if won else 'lose'
                
            elif bet['bet_type'] in ['over', 'under']:
                if bet['bet_type'] == 'over':
                    won = actual_total > bet['line']
                else:
                    won = actual_total < bet['line']
                result = 'win' if won else 'lose'
            
            # Calculate payout
            if result == 'win':
                if bet['odds'] > 0:
                    payout = bet['amount'] * (bet['odds'] / 100)
                else:
                    payout = bet['amount'] * (100 / abs(bet['odds']))
                bankroll += payout
            else:
                bankroll -= bet['amount']
            
            # Bankroll protection - stop if we lose too much
            if bankroll < starting_bankroll * 0.7:  # Stop if we lose 30% of bankroll
                print(f"⚠️  Bankroll protection triggered at ${bankroll:.2f}")
                break
            
            weekly_bets[week_key] += 1
            
            logs.append({
                'date': row['schedule_date'],
                'season': season,
                'week': week,
                'home_team': home,
                'away_team': away,
                'bet_type': bet['bet_type'],
                'team': bet['team'],
                'amount': bet['amount'],
                'odds': bet['odds'],
                'line': bet['line'],
                'edge': bet['edge'],
                'model_probability': bet['model_probability'],
                'result': result,
                'payout': payout,
                'bankroll': bankroll
            })
        
        # Stop betting if bankroll is too low
        if bankroll < starting_bankroll * 0.7:
            break
    
    # --- Print results ---
    print(f"\nFinal bankroll: ${bankroll:.2f}")
    roi = (bankroll - starting_bankroll) / starting_bankroll * 100
    print(f"ROI: {roi:.2f}%")
    log_df = pd.DataFrame(logs)
    log_df.to_csv('backtest_bet_log.csv', index=False)
    print(f"All bets and results logged to backtest_bet_log.csv")

    # --- Enhanced stats ---
    if not log_df.empty:
        num_bets = len(log_df)
        num_wins = (log_df['result'] == 'win').sum()
        win_rate = num_wins / num_bets if num_bets > 0 else 0
        avg_bet = log_df['amount'].mean() if num_bets > 0 else 0
        avg_payout = log_df[log_df['result'] == 'win']['payout'].mean() if num_wins > 0 else 0
        avg_edge = log_df['edge'].mean() if num_bets > 0 else 0
        
        # Equity curve
        equity_curve = log_df['bankroll']
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.values, label='Equity Curve (Bankroll)', linewidth=2)
        plt.axhline(y=starting_bankroll, color='r', linestyle='--', alpha=0.7, label='Starting Bankroll')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.title('Backtest Equity Curve - Improved Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
        print("Equity curve saved to equity_curve.png")
        
        # Max drawdown
        roll_max = equity_curve.cummax()
        drawdown = (equity_curve - roll_max) / roll_max
        max_drawdown = drawdown.min() * 100
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"Number of bets: {num_bets}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Average bet size: ${avg_bet:.2f}")
        print(f"Average payout (win): ${avg_payout:.2f}")
        print(f"Average edge: {avg_edge:.2%}")
        print(f"Max drawdown: {max_drawdown:.2f}%")
        
        # Season-by-season breakdown
        if len(seasons) > 1:
            print(f"\n📅 SEASON-BY-SEASON BREAKDOWN:")
            for season in seasons:
                season_bets = log_df[log_df['season'] == season]
                if not season_bets.empty:
                    season_wins = (season_bets['result'] == 'win').sum()
                    season_rate = season_wins / len(season_bets)
                    season_roi = (season_bets['bankroll'].iloc[-1] - season_bets['bankroll'].iloc[0]) / season_bets['bankroll'].iloc[0] * 100
                    print(f"  {season}: {len(season_bets)} bets, {season_rate:.2%} win rate, {season_roi:.2f}% ROI")
        
        # Bet type breakdown
        print(f"\n🎯 BET TYPE BREAKDOWN:")
        for bet_type in log_df['bet_type'].unique():
            type_bets = log_df[log_df['bet_type'] == bet_type]
            type_wins = (type_bets['result'] == 'win').sum()
            type_rate = type_wins / len(type_bets) if len(type_bets) > 0 else 0
            type_roi = (type_bets['bankroll'].iloc[-1] - type_bets['bankroll'].iloc[0]) / type_bets['bankroll'].iloc[0] * 100 if len(type_bets) > 0 else 0
            print(f"  {bet_type.capitalize()}: {len(type_bets)} bets, {type_rate:.2%} win rate, {type_roi:.2f}% ROI")
            
    else:
        print("No bets were placed - no profitable opportunities found.")

if __name__ == "__main__":
    main() 