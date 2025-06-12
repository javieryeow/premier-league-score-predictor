import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("../data/final_premier_league.csv")
df = df.sort_values(by=["Season", "Date"]).reset_index(drop=True)

# classifier target variables
outcome_mapping = {"H": 2, "D": 1, "A": 0}
df["MatchOutcome"] = df["FullTimeResult"].map(outcome_mapping)

# regression target variables
df["HomeGoals"] = df["FullTimeHomeTeamGoals"]
df["AwayGoals"] = df["FullTimeAwayTeamGoals"]
df["HomePoints"] = df["MatchOutcome"].map({2: 3, 1: 1, 0: 0})
df["AwayPoints"] = df["MatchOutcome"].map({0: 3, 1: 1, 2: 0})

# target encoding for matchup
matchup_target_encoding = (
    df.groupby("Matchup")["HomePoints"].agg(["mean", "count"])
).reset_index()
matchup_target_encoding.rename(columns={"mean": "MatchupAvgPoints", "count": "MatchupGamesPlayed"}, inplace=True)
df = df.merge(matchup_target_encoding, on="Matchup", how="left")
league_avg_points = df["HomePoints"].mean()
df["MatchupAvgPoints"].fillna(league_avg_points, inplace=True)

# target encoding for team-level point averages per game
# home_team_encoding = (
#     df.groupby("HomeTeam")["HomePoints"].mean().reset_index()
# )
# home_team_encoding.rename(columns={"HomePoints": "HomeTeamAvgPoints"}, inplace=True)
# df = df.merge(home_team_encoding, on="HomeTeam", how="left")

# away_team_encoding = (
#     df.groupby("AwayTeam")["AwayPoints"].mean().reset_index()
# )
# away_team_encoding.rename(columns={"AwayPoints": "AwayTeamAvgPoints"}, inplace=True)
# df = df.merge(away_team_encoding, on="AwayTeam", how="left")

# league_avg_points = df["HomePoints"].mean()
# df["HomeTeamAvgPoints"].fillna(league_avg_points, inplace=True)
# df["AwayTeamAvgPoints"].fillna(league_avg_points, inplace=True)


feature_cols = [
    "HomeGoalScoringForm",
    "AwayGoalScoringForm",
    "HomeGoalsConceded",
    "AwayGoalsConceded",
    "HomeWinningForm",
    "AwayWinningForm",
    "HomeRecentGoalScoringForm",
    "HomeRecentGoalsConceded",
    "HomeRecentWinningForm",
    "AwayRecentGoalScoringForm",
    "AwayRecentGoalsConceded",
    "AwayRecentWinningForm",
    "HomeDaysSinceLastMatch",
    "AwayDaysSinceLastMatch",
    "HomeEloBefore",
    "HomeEloAfter",
    "AwayEloBefore",
    "HomeTiltBefore",
    "AwayTiltBefore",
    "H2H_TotalMatches",
    "H2H_HomeGoalsScored",
    "H2H_AwayGoalsScored",
    "H2H_HomeWinRate",
    "HomeGoalDifferenceForm",
    "AwayGoalDifferenceForm",
    "HomeSeasonToDateGoalsScored",
    "HomeSeasonToDateGoalsConceded",
    "HomeSeasonToDateGoalDifference",
    "HomeSeasonToDatePoints",
    "AwaySeasonToDateGoalsScored",
    "AwaySeasonToDateGoalsConceded",
    "AwaySeasonToDateGoalDifference",
    "AwaySeasonToDatePoints",
    "HomeAvgShotsTaken",
    "HomeAvgShotsOnTarget",
    "HomeAvgCornersTaken",
    "AwayAvgShotsTaken",
    "AwayAvgShotsOnTarget",
    "AwayAvgCornersTaken",
    "MatchupAvgPoints",
    "B365HomeProbNorm",
    "B365DrawProbNorm",
    "B365AwayProbNorm",
    # "HomeTeamAvgPoints",
    # "AwayTeamAvgPoints"
]

# rolling backtest loop
seasons = sorted(df["Season"].unique())
all_results = []

for i in range(3, len(seasons)):  # Start after first few seasons to give training data
    
    train_seasons = seasons[:i]
    test_season = seasons[i]
    
    train_df = df[df["Season"].isin(train_seasons)]
    test_df = df[df["Season"] == test_season]
    
    # Prepare train and test sets
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Targets
    y_class_train = train_df["MatchOutcome"]
    y_class_test = test_df["MatchOutcome"]
    
    y_home_train = train_df["HomeGoals"]
    y_home_test = test_df["HomeGoals"]
    
    y_away_train = train_df["AwayGoals"]
    y_away_test = test_df["AwayGoals"]
    
    # # Outcome Model
    # outcome_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    # outcome_model.fit(X_train, y_class_train)
    # y_class_pred = outcome_model.predict(X_test)
    # class_acc = np.mean(y_class_pred == y_class_test)
    
    # # Home Goals Model
    # home_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    # home_model.fit(X_train, y_home_train)
    # y_home_pred = home_model.predict(X_test)
    # home_rmse = root_mean_squared_error(y_home_test, y_home_pred)

    # # Away Goals Model
    # away_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    # away_model.fit(X_train, y_away_train)
    # y_away_pred = away_model.predict(X_test)
    # away_rmse = root_mean_squared_error(y_away_test, y_away_pred)
    
    # --- Outcome model (XGBoost Classification) ---
    outcome_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.02,
        random_state=42,
        colsample_bytree=0.5,
        gamma=2,
        reg_alpha=1,
        reg_lambda=3,
        subsample=0.9,
    )
    outcome_model.fit(X_train, y_class_train)
    y_class_pred = outcome_model.predict(X_test)
    class_acc = np.mean(y_class_pred == y_class_test)
    
    # --- Home Goals model (XGBoost Regression) ---
    home_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.02,
        random_state=42,
        colsample_bytree=0.5,
        gamma=2,
        reg_alpha=1,
        reg_lambda=3,
        subsample=0.9,
    )
    home_model.fit(X_train, y_home_train)
    y_home_pred = home_model.predict(X_test)
    home_rmse = root_mean_squared_error(y_home_test, y_home_pred)
    
    # --- Away Goals model (XGBoost Regression) ---
    away_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.02,
        random_state=42,
        gamma=2,
        reg_alpha=1,
        reg_lambda=3,
        subsample=0.9,
    )
    away_model.fit(X_train, y_away_train)
    y_away_pred = away_model.predict(X_test)
    away_rmse = root_mean_squared_error(y_away_test, y_away_pred)
    
    # Store results
    all_results.append({
        "TrainSeasons": train_seasons,
        "TestSeason": test_season,
        "OutcomeAccuracy": class_acc,
        "HomeGoalsRMSE": home_rmse,
        "AwayGoalsRMSE": away_rmse
    })
    
    print(f"Test season {test_season}: Acc={class_acc:.3f}, HomeRMSE={home_rmse:.3f}, AwayRMSE={away_rmse:.3f}")
    
result_df = pd.DataFrame(all_results)
print(result_df)
print("\nAverage metrics across all folds:")
print(result_df.mean(numeric_only=True))



