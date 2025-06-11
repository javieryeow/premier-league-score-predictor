import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("../data/final_premier_league.csv")
df = df.sort_values(by=["Season", "Date"]).reset_index(drop=True)

# need to encode Matchup column since model can only take in numerical data
le = LabelEncoder()
df["MatchupEncoded"] = le.fit_transform(df["Matchup"])

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
    "MatchupEncoded"
]

outcome_mapping = {"H": 2, "D": 1, "A": 0}
df["MatchOutcome"] = df["FullTimeResult"].map(outcome_mapping)

df["HomeGoals"] = df["FullTimeHomeTeamGoals"]
df["AwayGoals"] = df["FullTimeAwayTeamGoals"]

# rolling backtest loop
seasons = sorted(df["Season"].unique())
all_results = []

for i in range(3, len(seasons)-1):  # Start after first few seasons to give training data
    
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
    
    # Outcome Model
    outcome_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    outcome_model.fit(X_train, y_class_train)
    y_class_pred = outcome_model.predict(X_test)
    class_acc = np.mean(y_class_pred == y_class_test)
    
    # Home Goals Model
    home_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    home_model.fit(X_train, y_home_train)
    y_home_pred = home_model.predict(X_test)
    home_rmse = root_mean_squared_error(y_home_test, y_home_pred)

    # Away Goals Model
    away_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
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
print(result_df.mean())



