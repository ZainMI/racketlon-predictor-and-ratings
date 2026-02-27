import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv("data/features_engineered.csv")

# Keep chronological order
df = df.sort_values("datetime").reset_index(drop=True)

# --------------------------------------------------
# Select features
# --------------------------------------------------

# Drop non-numeric + identifiers + targets
drop_cols = [
    "match_index",
    "tournament_id",
    "datetime",
    "match_date",
    "match_time",
    "draw",
    "draw_id",
    "round",
    "location",
    "team1_players",
    "team2_players",
    "team1_player_ids",
    "team2_player_ids",
    "team1_nationalities",
    "team2_nationalities",
    "team1_club_ids",
    "team2_club_ids",
    "winner_side",
    "status_message",
    "mode",
    # targets
    "y_total_p1",
    "y_total_p2",
    "y_winner_p1",
    "TT_y_diff",
    "TT_y_total",
    "TT_y_win_p1",
    "BD_y_diff",
    "BD_y_total",
    "BD_y_win_p1",
    "SQ_y_diff",
    "SQ_y_total",
    "SQ_y_win_p1",
    "TN_y_diff",
    "TN_y_total",
    "TN_y_win_p1",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["y_winner_p1"]

# Remove rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]

# Fill NaNs in numeric features
X = X.fillna(0)

# --------------------------------------------------
# Chronological split
# --------------------------------------------------

n = len(X)
split = int(n * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# --------------------------------------------------
# Model
# --------------------------------------------------

model = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))
print("Brier Score:", brier_score_loss(y_test, probs))
