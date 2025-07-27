# NFL Machine Learning Prediction Bot

A comprehensive machine learning system for predicting NFL game outcomes using historical data and real-time APIs.

## 🚀 Features

- **Historical Data Analysis**: Trains ML models on historical NFL game data
- **Real-time Predictions**: Fetches upcoming games and makes predictions
- **Multiple APIs Integration**:
  - TheSportsDB API for upcoming games
  - The Odds API for betting odds
  - OpenWeatherMap API for weather data
- **Dual Model Approach**:
  - Classification model for win/loss prediction
  - Regression models for score prediction
- **Feature Engineering**: Advanced feature creation including Elo ratings, rolling averages, and weather conditions

---

## Execution 

![Screen Recording 2025-07-27 110236 (online-video-cutter com)](https://github.com/user-attachments/assets/ce5df8da-7e88-41e6-a6dd-c985a8263b71)

Backtest for the 2024-2025 NFL season: 

<img width="374" height="326" alt="Screenshot 2025-07-27 110814" src="https://github.com/user-attachments/assets/cd5ebbe7-fd65-43fe-97b1-ebc071330710" />

---

## 📋 Prerequisites

### API Keys Required

You'll need to obtain free API keys from the following services:

1. **The Odds API** (https://the-odds-api.com/)

   - Sign up for a free account
   - Get your API key
   - Set environment variable: `ODDS_API_KEY`

2. **OpenWeatherMap API** (https://openweathermap.org/api)

   - Sign up for a free account
   - Get your API key
   - Set environment variable: `WEATHER_API_KEY`

3. **TheSportsDB API** (https://www.thesportsdb.com/)
   - No API key required for basic usage
   - Optional: Set `SPORTSDB_API_KEY` if you have one

### Setting Environment Variables

**Windows:**

```cmd
set ODDS_API_KEY=your_odds_api_key_here
set WEATHER_API_KEY=your_weather_api_key_here
```

**macOS/Linux:**

```bash
export ODDS_API_KEY=your_odds_api_key_here
export WEATHER_API_KEY=your_weather_api_key_here
```

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd nfl_ml
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up API keys** (see Prerequisites section above)

## 🎯 Usage

### Step 1: Train Models

First, train and save the ML models:

```bash
python train_models.py
```

This will:

- Load and clean historical NFL data
- Perform feature engineering
- Train classification and regression models
- Save models to the `models/` directory

### Step 2: Make Predictions

Predict upcoming NFL games:

```bash
python main.py
```

**Optional arguments:**

- `--days N`: Predict games for the next N days (default: 30)
- `--output filename.csv`: Save predictions to specific file (default: predictions.csv)

**Examples:**

```bash
# Predict next 7 days of games
python main.py --days 7

# Save predictions to custom file
python main.py --output my_predictions.csv

# Predict next 14 days and save to specific file
python main.py --days 14 --output week2_predictions.csv
```

### Step 3: View Results

The system will:

- Fetch upcoming games from APIs
- Apply the same feature engineering as training data
- Make predictions using trained models
- Display formatted results
- Save predictions to CSV file

## 📊 Model Performance

### Classification Model (Home Team Wins)

- **Test Accuracy**: ~65.61%
- **Training Accuracy**: ~67.09%

### Regression Models (Score Prediction)

- **Home Team**: RMSE: 10.05, MAE: 8.01
- **Away Team**: RMSE: 9.51, MAE: 7.60

## 🔧 Features Created

The system creates over 100 features including:

- Team rolling averages (points scored/allowed)
- Win/loss streaks and recent performance
- Head-to-head records
- Rest days between games
- Elo ratings (general, home/away, playoff, recent, favorite/underdog, weather)
- Weather conditions (temperature, wind, humidity)
- Game context (playoff, weekend, neutral stadium)
- One-hot encoded team and weather variables
- Betting odds (spreads, over/under lines)

## 📁 Project Structure

```
nfl_ml/
├── main.py                 # Main prediction application
├── train_models.py         # Model training and saving
├── api_fetchers.py         # API integration for real-time data
├── future_game_processor.py # Process future games for prediction
├── prediction_engine.py    # Load models and make predictions
├── data_loading.py         # Historical data loading
├── feature_engineering.py  # Feature creation
├── elo_system.py          # Elo rating calculations
├── classification_model.py # Classification model training
├── regression_model.py     # Regression model training
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # Saved trained models (created after training)
├── spreadspoke_scores.csv # Historical NFL data
└── nfl_teams.csv          # Team information
```

## 🎯 Model Details

- **Algorithm**: XGBoost
- **Classification**: Binary logistic regression for win/loss
- **Regression**: Separate models for home and away scores
- **Hyperparameters**: Optimized through grid search
- **Evaluation**: Train/test split with stratification

## 📈 Prediction Output

The system provides:

- **Win/Loss Prediction**: Which team will win
- **Win Probability**: Confidence level for the prediction
- **Score Prediction**: Predicted final scores for both teams
- **Total Points**: Predicted combined score
- **Confidence Score**: Overall prediction confidence

## 🔍 API Data Sources

### TheSportsDB API

- Upcoming NFL games
- Team names, dates, venues
- Game schedules and locations

### The Odds API

- Betting odds and spreads
- Over/under lines
- Favorite/underdog information
- Multiple bookmaker data

### OpenWeatherMap API

- Current weather conditions
- Temperature, wind speed, humidity
- Weather descriptions
- Location-based weather data

## 🚨 Important Notes

- **Data Filtering**: Historical data filtered from 1979 onwards for consistency
- **Franchise Updates**: Team name updates applied for continuity
- **Elo Ratings**: Reset every 3 years for accuracy
- **Missing Values**: Handled with appropriate imputation strategies
- **Rate Limiting**: Weather API calls are rate-limited to avoid hitting limits
- **Neutral Stadiums**: Automatically detected for international games and special events
- **Weather Features**: Weather features are not currently included in predictions. You may add them later if desired.

## 🛠️ Troubleshooting

### Common Issues

1. **"Models not found" error**

   - Run `python train_models.py` first

2. **"API key not found" error**

   - Set your environment variables correctly
   - Check API key validity

3. **"No future games found" error**

   - Check internet connection
   - Verify API keys are working
   - NFL season may be over

4. **"Feature mismatch" error**
   - Retrain models with `python train_models.py`
   - Ensure data format consistency

## 📝 License

This project is for educational and research purposes. Please ensure compliance with API terms of service.

## Author

Alexander Mueller

- GitHub: [alexmueller07](https://github.com/alexmueller07)
- LinkedIn: [Alexander Mueller](https://www.linkedin.com/in/alexander-mueller-021658307/)
- Email: amueller.code@gmail.com
