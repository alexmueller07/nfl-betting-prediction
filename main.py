"""
NFL Prediction Bot - Main Application
Makes predictions on upcoming NFL games using trained ML models.
"""

from prediction_engine import NFLPredictionEngine
import argparse
import sys
import pandas as pd
from future_game_processor import process_oddsapi_games_for_prediction
import random

# Strategy parameters - MATCH BACKTEST
MIN_EDGE_PCT = 0.015  # 1.5% minimum edge
MAX_KELLY_PCT = 0.05  # Max 5% of bankroll per bet
MIN_CONFIDENCE = 0.51  # Minimum model confidence
MAX_BET_ABS = 70  # Absolute max bet per game
MAX_BET_PCT = 0.07  # 7% of current bankroll
MIN_BET_PCT = 0.02  # 2% of current bankroll

# Bet type filters - MONEYLINE FOCUS
ENABLE_SPREAD_BETS = False  # Disable spread bets
ENABLE_MONEYLINE_BETS = True
ENABLE_OU_BETS = True

# Aggressive thresholds for moneyline
ML_FAV_THRESHOLD = 0.54  # Lower threshold for favorites
ML_DOG_THRESHOLD = 0.46  # Higher threshold for underdogs
OU_THRESHOLD = 5  # O/U threshold

# Realistic odds ranges
SPREAD_ODDS_RANGE = (-120, -110)
ML_FAV_RANGE = (-250, -150)
ML_DOG_RANGE = (120, 300)
OU_ODDS_RANGE = (-120, -110)

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
    """Calculate Kelly Criterion bet size with more conservative scaling."""
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    q = 1 - prob
    k = (prob * b - q) / b
    # Conservative scaling for more bets in 4-5% range
    edge = prob - (1 / (1 + b))
    if edge > 0.05:
        k = k * 0.8
    elif edge > 0.03:
        k = k * 0.6
    else:
        k = k * 0.4
    return max(0, min(k, MAX_KELLY_PCT))

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
    return -110

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

def main():
    """
    Main function to run NFL game predictions.
    """
    parser = argparse.ArgumentParser(description='NFL ML Betting Bot')
    parser.add_argument('--games', type=int, default=5, help='Number of upcoming games to process')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Starting bankroll')
    args = parser.parse_args()

    bankroll = args.bankroll
    print(f"\n🏈 NFL Machine Learning Betting Bot (MONEYLINE FOCUS)\n{'='*60}")
    print(f"Starting bankroll: ${bankroll:.2f}")
    print(f"Strategy: Only bet with {MIN_EDGE_PCT*100:.1f}%+ edge, max {MAX_KELLY_PCT*100:.0f}% Kelly")
    print(f"Bet types: Spread={ENABLE_SPREAD_BETS}, ML={ENABLE_MONEYLINE_BETS}, O/U={ENABLE_OU_BETS}")
    print(f"Bet range: ${MIN_BET_PCT*100:.0f}%-${MAX_BET_ABS} (${MIN_BET_PCT*100:.0f}%-{MAX_BET_PCT*100:.0f}% of bankroll)\n")
    
    # 1. Get games and odds
    games_df = process_oddsapi_games_for_prediction(args.games)
    if games_df.empty:
        print("No games available.")
        sys.exit(1)

    # 2. Get model predictions
    engine = NFLPredictionEngine()
    pred_df = engine.predict_for_games(games_df, num_games=args.games)

    # 3. Bet logic - MATCH BACKTEST STRATEGY
    bets = []
    logs = []
    
    for _, row in pred_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        fav = row['favorite_team']
        dog = row['underdog_team']
        spread = row['spread']
        ml_home = row['moneyline_home']
        ml_away = row['moneyline_away']
        ou = row['over_under']
        ou_over_odds = row['over_odds']
        ou_under_odds = row['under_odds']
        
        # Model outputs
        win_prob = row['confidence']  # Use confidence as win probability
        predicted_home_score = row['predicted_home_score']
        predicted_away_score = row['predicted_away_score']
        predicted_total = predicted_home_score + predicted_away_score
        predicted_margin = predicted_home_score - predicted_away_score
        
        bet_list = []
        
        # MONEYLINE BETTING (PRIMARY FOCUS)
        if ENABLE_MONEYLINE_BETS:
            # Use aggressive probability calculation
            aggressive_win_prob = calculate_aggressive_probability(win_prob, "classification")
            
            # Determine if home team is favorite
            home_is_favorite = (fav == home)
            
            if home_is_favorite:
                # Bet favorite if model favors
                if aggressive_win_prob > ML_FAV_THRESHOLD:
                    odds = ml_home if ml_home else get_realistic_odds("moneyline", is_favorite=True)
                    edge = calculate_edge(aggressive_win_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(aggressive_win_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("moneyline", home, bet_amount, odds,
                                                    edge=edge, model_prob=aggressive_win_prob))
                
                # Bet underdog if model strongly favors
                if aggressive_win_prob < ML_DOG_THRESHOLD:
                    underdog_prob = 1 - aggressive_win_prob
                    odds = ml_away if ml_away else get_realistic_odds("moneyline", is_favorite=False)
                    edge = calculate_edge(underdog_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(underdog_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("moneyline", away, bet_amount, odds,
                                                    edge=edge, model_prob=underdog_prob))
            else:
                # Away team is favorite
                if aggressive_win_prob < ML_DOG_THRESHOLD:
                    odds = ml_home if ml_home else get_realistic_odds("moneyline", is_favorite=False)
                    edge = calculate_edge(aggressive_win_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(aggressive_win_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("moneyline", home, bet_amount, odds,
                                                    edge=edge, model_prob=aggressive_win_prob))
                
                if aggressive_win_prob > ML_FAV_THRESHOLD:
                    underdog_prob = 1 - aggressive_win_prob
                    odds = ml_away if ml_away else get_realistic_odds("moneyline", is_favorite=True)
                    edge = calculate_edge(underdog_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(underdog_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("moneyline", away, bet_amount, odds,
                                                    edge=edge, model_prob=underdog_prob))
        
        # OVER/UNDER BETTING (SECONDARY)
        if ENABLE_OU_BETS and ou is not None and not pd.isna(ou):
            # Calculate distance from line
            distance_from_line = abs(predicted_total - ou)
            
            # Use momentum-based edge calculation
            momentum_prob = calculate_momentum_edge(predicted_home_score, predicted_away_score, 
                                                  predicted_total, ou)
            
            # Only bet if prediction is far enough from line
            if distance_from_line >= OU_THRESHOLD:
                if predicted_total > ou:  # Over prediction
                    # Combine regression prediction with momentum
                    final_prob = (momentum_prob + 0.6) / 2 if momentum_prob > 0.5 else 0.55
                    odds = ou_over_odds if ou_over_odds else get_realistic_odds("over")
                    edge = calculate_edge(final_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(final_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("over", None, bet_amount, odds,
                                                    line=ou, edge=edge, model_prob=final_prob))
                            
                elif predicted_total < ou:  # Under prediction
                    # Combine regression prediction with momentum
                    final_prob = (momentum_prob + 0.6) / 2 if momentum_prob > 0.5 else 0.55
                    odds = ou_under_odds if ou_under_odds else get_realistic_odds("under")
                    edge = calculate_edge(final_prob, odds)
                    if edge >= MIN_EDGE_PCT:
                        kelly_pct = kelly_criterion(final_prob, odds)
                        bet_amount = kelly_pct * bankroll
                        bet_cap = min(MAX_BET_ABS, MAX_BET_PCT * bankroll)
                        bet_amount = max(MIN_BET_PCT * bankroll, min(bet_amount, bet_cap))
                        if bet_amount >= MIN_BET_PCT * bankroll:
                            bet_list.append(place_bet("under", None, bet_amount, odds,
                                                    line=ou, edge=edge, model_prob=final_prob))
        
        # Sort bets by edge (highest first)
        bet_list.sort(key=lambda x: x['edge'], reverse=True)
        bet_list = bet_list[:2]  # Max 2 bets per game
        
        logs.append({
            'game_id': row['game_id'],
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'favorite_team': fav,
            'underdog_team': dog,
            'spread': spread,
            'moneyline_home': ml_home,
            'moneyline_away': ml_away,
            'over_under': ou,
            'predicted_winner': row['predicted_winner'],
            'predicted_margin': predicted_margin,
            'predicted_home_score': predicted_home_score,
            'predicted_away_score': predicted_away_score,
            'confidence': win_prob,
            'bets': bet_list
        })
        
        for bet in bet_list:
            bets.append((bet, win_prob, row))
    
    # --- Print Bets ---
    print(f"\nGame Summaries and Bet Recommendations:")
    bets_by_game = {}
    for bet, conf, row in bets:
        bets_by_game.setdefault(row['game_id'], []).append((bet, conf, row))
    
    for idx, row in pred_df.iterrows():
        print('='*80)
        print(f"{row['away_team']} at {row['home_team']} on {row['date']}")
        print(f"Favorite: {row['favorite_team']} | Spread: {row['spread']}")
        print(f"Classification Predicted Winner: {row['favorite_team'] if row['predicted_winner']=='favorite' else row['underdog_team']} | Confidence: {row['confidence']:.1%}")
        print(f"Regression Predicted Score: {row['home_team']}: {row['predicted_home_score']:.1f} | {row['away_team']}: {row['predicted_away_score']:.1f}")
        print(f"Predicted Total: {row['predicted_home_score'] + row['predicted_away_score']:.1f} | O/U Line: {row['over_under']}")
        
        game_bets = bets_by_game.get(row['game_id'], [])
        if not game_bets:
            print("Bet: NO BET")
        else:
            for bet, conf, _ in game_bets:
                if bet['bet_type'] == 'moneyline':
                    print(f"Bet: {bet['bet_type'].capitalize()} on {bet['team']} | Amount: ${bet['amount']:.2f} | Odds: {bet['odds']} | Edge: {bet['edge']:.1%}")
                elif bet['bet_type'] in ['over', 'under']:
                    print(f"Bet: {bet['bet_type'].capitalize()} | Amount: ${bet['amount']:.2f} | Odds: {bet['odds']} | Line: {bet['line']} | Edge: {bet['edge']:.1%}")
    
    # --- Log all bets ---
    log_df = pd.DataFrame(logs)
    log_df.to_csv('bet_log.csv', index=False)
    print(f"\nAll bets and predictions logged to bet_log.csv")

if __name__ == "__main__":
    main()