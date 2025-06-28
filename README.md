# 🏟 Premier League Match Score Predictor

This project builds an advanced machine learning pipeline to **predict football match outcomes and exact scorelines** for the English Premier League, using historical match data and engineered features derived from team form, match dynamics, and bookmaker odds.

## 📈 Objectives

- Predict the **match outcome** (Home Win / Draw / Away Win) using a classification model.
- Forecast the **exact match scoreline** (home and away goals) using regression.
- Evaluate performance using both traditional metrics (accuracy, RMSE) and probabilistic metrics (Brier score).

---

## 📂 Dataset

The data comprises historical Premier League match results from **2000–2025**, cleaned and aggregated from open source APIs. It includes:

- Match metadata (date, teams, season, home/away)
- Full-time scores and results
- Match statistics (shots, shots on target, corners)
- Bookmaker probabilties (Bet365)
- Engineered time-series and aggregate features (see below)

---

## 🛠 Features

Over **40 engineered features** were created to improve predictive power, grouped into several categories:

### 📊 Time-Series Performance Metrics
- `HomeGoalScoringForm`, `AwayGoalsConceded`, `HomeWinningForm`, etc.
- Derived from cumulative and rolling averages of team performance
- Captures medium-term form and defensive/offensive trends

### 🧠 Recent Form Indicators (Last N Matches)
- `HomeRecentGoalScoringForm`, `AwayRecentWinningForm`, etc.
- Short-term performance context using a moving window

### ⚔️ Head-to-Head Dynamics
- `H2H_HomeGoalsScored`, `H2H_HomeWinRate`, etc.
- Tracks matchup history between two specific teams

### ⚙️ Season-to-Date Aggregates
- `HomeSeasonToDatePoints`, `AwaySeasonToDateGoalDifference`, etc.
- Cumulative stats (goals, points, goal difference) as of matchday

### 📅 Scheduling & Fatigue
- `HomeDaysSinceLastMatch`, `AwayDaysSinceLastMatch`
- Captures rest period and fixture congestion

### 📈 Elo & Momentum
- `HomeEloBefore`, `HomeTiltBefore`, etc.
- Tracks team strength and emotional tilt through recent events

### 🎯 Market Expectations
- `B365HomeProbNorm`, `B365DrawProbNorm`, `B365AwayProbNorm`
- Normalized implied probabilities from bookmaker odds

### 🎯 Target Encoded Matchups
- `MatchupAvgPoints`: Historical average points gained in this matchup
- Built using target encoding to represent historical matchup advantage

---

## 🤖 Models

### 1. Match Outcome Prediction (Classification)
- **Model**: `XGBoostClassifier`
- **Target**: `MatchOutcome` (H=2, D=1, A=0)
- **Evaluation**: Accuracy, Brier Score

### 2. Scoreline Prediction (Regression)
- **Model**: `XGBoostRegressor`
- **Targets**: `HomeGoals`, `AwayGoals`
- **Evaluation**: RMSE, strict scoreline match %

---

## 🔁 Rolling Backtesting Strategy

To simulate real-world forecasting:

- **Rolling window training**: Train on seasons 1...N-1, test on season N.
- Models are trained separately for each season in the backtest.
- Avoids look-ahead bias and allows tracking of accuracy over time.

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | For match outcome prediction |
| **RMSE** | For continuous goal forecasts |
| **Brier Score** | Measures the accuracy of probabilistic forecasts for classification tasks |


