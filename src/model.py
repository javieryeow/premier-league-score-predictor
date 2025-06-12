import pandas as pd
import numpy as np 

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import root_mean_squared_error


df = pd.read_csv("../data/final_premier_league.csv")
df = df.sort_values(by=["Season", "Date"]).reset_index(drop=True)

# classifier target variables
outcome_mapping = {"H": 2, "D": 1, "A": 0}
df["MatchOutcome"] = df["FullTimeResult"].map(outcome_mapping)

# regression target variables
df["HomeGoals"] = df["FullTimeHomeTeamGoals"]
df["AwayGoals"] = df["FullTimeAwayTeamGoals"]

# target encoding for matchup
df["HomePoints"] = df["MatchOutcome"].map({2: 3, 1: 1, 0: 0})
matchup_target_encoding = (
    df.groupby("Matchup")["HomePoints"].agg(["mean", "count"])
).reset_index()
matchup_target_encoding.rename(columns={"mean": "MatchupAvgPoints", "count": "MatchupGamesPlayed"}, inplace=True)
df = df.merge(matchup_target_encoding, on="Matchup", how="left")
league_avg_points = df["HomePoints"].mean()
df["MatchupAvgPoints"].fillna(league_avg_points, inplace=True)


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
    y_class_proba = outcome_model.predict_proba(X_test)
    class_acc = np.mean(y_class_pred == y_class_test)
    
    brier_total = 0

    for i in range(len(y_class_test)):
        true_outcome = y_class_test.iloc[i]  # 0=A, 1=D, 2=H
        proba_vector = y_class_proba[i]      # predicted [A, D, H] probabilities

        # Build true outcome one-hot encoding
        true_vector = np.zeros(3)
        true_vector[true_outcome] = 1

        # Brier score for this match
        brier = np.sum((proba_vector - true_vector) ** 2)
        brier_total += brier

    # Average Brier score for the test season
    brier_score = brier_total / len(y_class_test)
    
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
    y_home_pred_discrete = np.round(y_home_pred).astype(int)
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
    y_away_pred_discrete = np.round(y_away_pred).astype(int)
    away_rmse = root_mean_squared_error(y_away_test, y_away_pred)
    
    # evaluate strictly correct prediction of scoreline
    scoreline_accuracy = np.mean((y_home_pred_discrete == y_home_test) & (y_away_pred_discrete == y_away_test))
    print(f"Scoreline prediction accuracy: {scoreline_accuracy:.4f}")
    
    # Store results
    all_results.append({
        "TrainSeasons": train_seasons,
        "TestSeason": test_season,
        "OutcomeAccuracy": class_acc,
        "HomeGoalsRMSE": home_rmse,
        "AwayGoalsRMSE": away_rmse,
        "BrierScore": brier_score,
    })
    
    print(f"Test season {test_season}: Acc={class_acc:.3f}, HomeRMSE={home_rmse:.3f}, AwayRMSE={away_rmse:.3f}, Brier={brier_score:.3f},")
    
result_df = pd.DataFrame(all_results)
print(result_df)
print("\nAverage metrics across all folds:")
print(result_df.mean(numeric_only=True))



