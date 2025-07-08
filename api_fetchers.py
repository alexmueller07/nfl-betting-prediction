"""
API Fetchers for NFL Game Data
Fetches upcoming games, odds, and weather data from various APIs.
"""

import requests
import json
import os
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd

# API Configuration
class APIConfig:
    def __init__(self):
        self.sportsdb_api_key = os.getenv('SPORTSDB_API_KEY', '123')  # TheSportsDB doesn't require API key for basic usage
        self.odds_api_key = os.getenv('ODDS_API_KEY', '8de0c28a77560b5a67a8b318f916fcbb')
        
        
        # Base URLs
        self.sportsdb_base = "https://www.thesportsdb.com/api/v1/json"
        self.odds_base = "https://api.the-odds-api.com/v4/sports"

def fetch_upcoming_nfl_games(api_config: APIConfig, days_ahead: int = 30) -> List[Dict]:
    """
    Fetch upcoming NFL games from TheSportsDB API.
    Only returns games where league is NFL.
    """
    try:
        url = f"{api_config.sportsdb_base}/3/eventsnextleague.php?id=4391"  # NFL league ID
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("Raw API data:", data)
        if 'events' not in data:
            print("No events found in API response")
            return []
        games = []
        current_date = datetime.now()
        end_date = current_date + timedelta(days=days_ahead)
        for event in data['events']:
            try:
                event_date = datetime.strptime(event['dateEvent'], '%Y-%m-%d')
                # Only include games within our date range and where league is NFL
                if (current_date <= event_date <= end_date) and (event.get('strLeague', '').lower() == 'nfl'):
                    game = {
                        'game_id': event.get('idEvent'),
                        'date': event['dateEvent'],
                        'time': event.get('strTime', ''),
                        'home_team': event.get('strHomeTeam', ''),
                        'away_team': event.get('strAwayTeam', ''),
                        'venue': event.get('strVenue', ''),
                        'venue_location': event.get('strVenueLocation', ''),
                        'league': event.get('strLeague', ''),
                        'season': event.get('strSeason', ''),
                        'status': event.get('strStatus', '')
                    }
                    games.append(game)
            except (KeyError, ValueError) as e:
                print(f"Error parsing event: {e}")
                continue
        print(f"✅ Fetched {len(games)} upcoming NFL games (league == 'NFL')")
        return games
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching upcoming games: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        return []

def fetch_odds_data(api_config: APIConfig, sport_key: str = 'americanfootball_nfl') -> List[Dict]:
    """
    Fetch odds data from The Odds API.
    
    Args:
        api_config: API configuration object
        sport_key: Sport key for the API (default: americanfootball_nfl)
    
    Returns:
        List of odds dictionaries
    """
    if not api_config.odds_api_key:
        print("❌ Odds API key not found. Set ODDS_API_KEY environment variable.")
        return []
    
    try:
        url = f"{api_config.odds_base}/{sport_key}/odds"
        params = {
            'apiKey': api_config.odds_api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        odds_list = []
        for game in data:
            try:
                # Extract basic game info
                odds_data = {
                    'game_id': game.get('id'),
                    'sport_key': game.get('sport_key'),
                    'sport_title': game.get('sport_title'),
                    'commence_time': game.get('commence_time'),
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'bookmakers': []
                }
                
                # Extract odds from different bookmakers
                for bookmaker in game.get('bookmakers', []):
                    bookmaker_data = {
                        'bookmaker': bookmaker.get('title'),
                        'markets': {}
                    }
                    
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        market_data = []
                        
                        for outcome in market.get('outcomes', []):
                            outcome_data = {
                                'name': outcome.get('name'),
                                'price': outcome.get('price'),
                                'point': outcome.get('point')
                            }
                            market_data.append(outcome_data)
                        
                        bookmaker_data['markets'][market_key] = market_data
                    
                    odds_data['bookmakers'].append(bookmaker_data)
                
                odds_list.append(odds_data)
                
            except (KeyError, ValueError) as e:
                print(f"Error parsing odds data: {e}")
                continue
        
        print(f"✅ Fetched odds data for {len(odds_list)} games")
        return odds_list
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds data: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing odds JSON response: {e}")
        return []

def determine_neutral_stadium(home_team: str, venue: str, venue_location: str) -> bool:
    """
    Determine if a stadium is neutral based on home team and venue.
    
    Args:
        home_team: Home team name
        venue: Venue name
        venue_location: Venue location
    
    Returns:
        True if neutral stadium, False otherwise
    """
    # Known neutral sites
    neutral_sites = [
        'london', 'england', 'uk',
        'mexico city', 'mexico',
        'toronto', 'canada',
        'munich', 'germany',
        'frankfurt', 'germany',
        'super bowl',
        'pro bowl'
    ]
    
    # Check if venue location contains neutral site keywords
    location_lower = venue_location.lower()
    for site in neutral_sites:
        if site in location_lower:
            return True
    
    # Check if venue name suggests neutral site
    venue_lower = venue.lower()
    for site in neutral_sites:
        if site in venue_lower:
            return True
    
    # TODO: Add more sophisticated logic to check if venue is team's home stadium
    # This would require a database of team home stadiums
    
    return False

def combine_game_data(games: List[Dict], odds_data: List[Dict]) -> List[Dict]:
    """
    Combine game data with odds data based on team names and dates.
    
    Args:
        games: List of game dictionaries from TheSportsDB
        odds_data: List of odds dictionaries from The Odds API
    
    Returns:
        Combined game data with odds
    """
    combined_games = []
    
    for game in games:
        combined_game = game.copy()
        combined_game['odds'] = None
        combined_game['stadium_neutral'] = determine_neutral_stadium(
            game['home_team'], game['venue'], game['venue_location']
        )
        
        # Try to match with odds data
        for odds in odds_data:
            if (odds['home_team'].lower() == game['home_team'].lower() and 
                odds['away_team'].lower() == game['away_team'].lower()):
                combined_game['odds'] = odds
                break
        
        combined_games.append(combined_game)
    
    return combined_games

def save_upcoming_games_to_csv(games: List[Dict], filename: str = 'upcoming_games.csv'):
    """
    Save upcoming NFL games (from Odds API) to CSV file.
    """
    if not games:
        print("No games to save")
        return
    flattened_games = []
    for game in games:
        flat_game = {
            'game_id': game.get('game_id'),
            'date': game.get('date'),
            'time': game.get('time'),
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'sport_title': game.get('sport_title'),
            'status': game.get('status'),
        }
        # Add odds data if available
        if game.get('bookmakers'):
            for bookmaker in game['bookmakers']:
                if 'h2h' in bookmaker.get('markets', {}):
                    h2h_markets = bookmaker['markets']['h2h']
                    for outcome in h2h_markets:
                        if outcome['name'] == game['home_team']:
                            flat_game[f"{bookmaker['bookmaker']}_home_odds"] = outcome['price']
                        elif outcome['name'] == game['away_team']:
                            flat_game[f"{bookmaker['bookmaker']}_away_odds"] = outcome['price']
        flattened_games.append(flat_game)
    df = pd.DataFrame(flattened_games)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(flattened_games)} NFL games to {filename}")

# --- MAIN FUNCTION: Fetch all upcoming NFL games from The Odds API ---
def fetch_all_upcoming_nfl_games_oddsapi(api_config: APIConfig) -> List[Dict]:
    """
    Fetch all upcoming NFL games from The Odds API.
    Returns a list of games with home/away, date, and odds info.
    """
    if not api_config.odds_api_key:
        print("❌ Odds API key not found. Set ODDS_API_KEY environment variable.")
        return []
    try:
        url = f"{api_config.odds_base}/americanfootball_nfl/odds"
        params = {
            'apiKey': api_config.odds_api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Odds API returned {len(data)} games.")
        games = []
        for game in data:
            try:
                # Parse date
                dt = game.get('commence_time')
                date = dt[:10] if dt else None
                time = dt[11:16] if dt else None
                games.append({
                    'game_id': game.get('id'),
                    'date': date,
                    'time': time,
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'bookmakers': game.get('bookmakers', []),
                    'sport_title': game.get('sport_title'),
                    'status': game.get('status', ''),
                })
            except Exception as e:
                print(f"Error parsing game: {e}")
                continue
        print(f"✅ Parsed {len(games)} NFL games from Odds API.")
        return games
    except Exception as e:
        print(f"❌ Error fetching from Odds API: {e}")
        return []

def print_human_readable_games(games: List[Dict]):
    """
    Print a human-readable summary for each game:
    {away_team} at {home_team} on {date} | {Home_team Odds ML}, {away_team_odds ML}. {Spread and spread odds}, {over_under line and over under odds}
    """
    for game in games:
        away = game.get('away_team')
        home = game.get('home_team')
        date = game.get('date')
        readable = f"{away} at {home} on {date} | "
        # Find first bookmaker with all markets
        ml_home = ml_away = spread = spread_odds = ou_line = ou_over_odds = ou_under_odds = None
        if game.get('bookmakers'):
            for bookmaker in game['bookmakers']:
                markets = {m['key']: m for m in bookmaker.get('markets', [])}
                # Moneyline
                if 'h2h' in markets:
                    for outcome in markets['h2h']['outcomes']:
                        if outcome['name'] == home:
                            ml_home = outcome['price']
                        elif outcome['name'] == away:
                            ml_away = outcome['price']
                # Spread
                if 'spreads' in markets:
                    for outcome in markets['spreads']['outcomes']:
                        if outcome['name'] == home:
                            spread = outcome.get('point')
                            spread_odds = outcome.get('price')
                            break
                # Over/Under
                if 'totals' in markets:
                    for outcome in markets['totals']['outcomes']:
                        if outcome['name'] == 'Over':
                            ou_line = outcome.get('point')
                            ou_over_odds = outcome.get('price')
                        elif outcome['name'] == 'Under':
                            ou_under_odds = outcome.get('price')
                # Only use the first bookmaker with all info
                break
        readable += f"{home} ML: {ml_home}, {away} ML: {ml_away}. "
        readable += f"Spread: {spread} ({spread_odds}), "
        readable += f"O/U: {ou_line} (Over: {ou_over_odds}, Under: {ou_under_odds})"
        print(readable)

if __name__ == "__main__":
    config = APIConfig()
    print("🔍 Fetching all upcoming NFL games from Odds API...")
    games = fetch_all_upcoming_nfl_games_oddsapi(config)
    print(f"Number of NFL games fetched: {len(games)}")
    if games:
        print("Sample NFL game:", games[0])
    print("💾 Saving all NFL games to CSV...")
    save_upcoming_games_to_csv(games)
    print("✅ API fetching complete!")
    print("\n--- Human Readable Game List ---")
    print_human_readable_games(games) 