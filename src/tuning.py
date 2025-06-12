import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("../data/final_premier_league.csv")
df = df.sort_values(by=["Season", "Date"]).reset_index(drop=True)
outcome_mapping = {"H": 2, "D": 1, "A": 0}
df["MatchOutcome"] = df["FullTimeResult"].map(outcome_mapping)
df["HomePoints"] = df["MatchOutcome"].map({2: 3, 1: 1, 0: 0})

# target encoding for matchup
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

# use dataset before 2021-2022 for training, and 2021-2022 onwards for testing
train_df = df[df["Season"] < "2021-2022"]
test_df = df[df["Season"] >= "2021-2022"]

X_train = train_df[feature_cols]
y_train = train_df["MatchOutcome"]

X_test = test_df[feature_cols]
y_test = test_df["MatchOutcome"]

# parameters to select from
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.02, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'gamma': [0, 2],
    'reg_alpha': [0, 1],
    'reg_lambda': [1, 3]
}

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3, 
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# best hyperparameters found
print("Best parameters:", grid_search.best_params_)

# using best estimator to predict on test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy on unseen data: {test_accuracy:.4f}")
