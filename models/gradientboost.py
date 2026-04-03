import pandas as pd
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42

# -------------------------------------------------
# Toggle
# -------------------------------------------------
PREDICT_ONE_MATCH = True
PLAYER1 = "zain magdon-ismail"
PLAYER2 = "anant gupta"

DATA_PATH = "data/matches_cleaned.csv"
TRAIN_RATIO = 0.8
SHRINK = 10

PREDICT_TENNIS_INDEPENDENTLY = True

# Gradient boosting params
GB_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)


# -------------------------------------------------
# Load
# -------------------------------------------------
def read_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y%m%d")

    if "match_time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["match_date"].astype(str) + " " + df["match_time"].astype(str),
            errors="coerce",
        )
        df = df.sort_values("datetime")
    else:
        df = df.sort_values("match_date")

    return df.reset_index(drop=True)


# -------------------------------------------------
# Running Stats
# -------------------------------------------------
@dataclass
class RunningStats:
    diff_sum: Dict[str, Dict[str, float]]
    total_sum: Dict[str, Dict[str, float]]
    count: Dict[str, Dict[str, int]]

    def __init__(self):
        self.diff_sum = defaultdict(lambda: defaultdict(float))
        self.total_sum = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(lambda: defaultdict(int))

    def get_avgs(self, player: str, sport: str, shrink: int = 10):
        c = self.count[player][sport]
        d = self.diff_sum[player][sport]
        t = self.total_sum[player][sport]

        return (
            d / (c + shrink),
            t / (c + shrink),
        )

    def update(self, p1: str, p2: str, sport: str, s1: float, s2: float):
        diff = s1 - s2
        total = s1 + s2

        self.diff_sum[p1][sport] += diff
        self.total_sum[p1][sport] += total
        self.count[p1][sport] += 1

        self.diff_sum[p2][sport] -= diff
        self.total_sum[p2][sport] += total
        self.count[p2][sport] += 1


# -------------------------------------------------
# Feature Builder
# -------------------------------------------------
def build_dataset(df: pd.DataFrame, shrink: int = 10):
    stats = RunningStats()
    X_rows = []
    targets = {s: {"diff": [], "total": [], "row": []} for s in SPORTS}

    for i, row in df.iterrows():
        p1 = str(row["team1_players"]).strip().lower()
        p2 = str(row["team2_players"]).strip().lower()

        feat = {}

        # same per-sport features as your ridge baseline
        for s in SPORTS:
            p1_d, p1_t = stats.get_avgs(p1, s, shrink)
            p2_d, p2_t = stats.get_avgs(p2, s, shrink)

            feat[f"{s}_delta_diff"] = p1_d - p2_d
            feat[f"{s}_delta_total"] = p1_t - p2_t
            feat[f"{s}_sum_total"] = p1_t + p2_t

        X_rows.append(feat)

        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")

            if pd.isna(s1) or pd.isna(s2):
                continue

            s1 = float(s1)
            s2 = float(s2)

            targets[s]["diff"].append(s1 - s2)
            targets[s]["total"].append(s1 + s2)
            targets[s]["row"].append(i)

            stats.update(p1, p2, s, s1, s2)

    return pd.DataFrame(X_rows), targets


def build_running_stats(df: pd.DataFrame, end_idx: int):
    stats = RunningStats()

    for i in range(end_idx):
        row = df.iloc[i]
        p1 = str(row["team1_players"]).strip().lower()
        p2 = str(row["team2_players"]).strip().lower()

        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue
            stats.update(p1, p2, s, float(s1), float(s2))

    return stats


# -------------------------------------------------
# Reconstruction
# -------------------------------------------------
def reconstruct_scores(pred_diff, pred_total):
    pred_diff = np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    pred_total = np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT)

    s1 = 0.5 * (pred_total + pred_diff)
    s2 = 0.5 * (pred_total - pred_diff)

    s1 = np.clip(s1, 0, MAX_POINTS_PER_SPORT)
    s2 = np.clip(s2, 0, MAX_POINTS_PER_SPORT)

    return s1, s2


def round_and_clip_score(x: float, upper: int = 21) -> int:
    return max(0, min(upper, int(round(float(x)))))


def decode_full_game_score(
    pred_diff: float, pred_total: float
) -> tuple[int, int]:
    pred_diff = float(
        np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    )
    pred_total = float(np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT))

    raw_s1, raw_s2 = reconstruct_scores(pred_diff, pred_total)
    raw_s1 = float(raw_s1)
    raw_s2 = float(raw_s2)

    margin_loser = 21 - abs(pred_diff)
    margin_loser = max(0.0, min(20.0, margin_loser))

    if pred_diff >= 0:
        loser_from_raw = min(raw_s2, 20.0)
        loser_score = 0.5 * loser_from_raw + 0.5 * margin_loser
        s1 = 21
        s2 = round_and_clip_score(loser_score, 20)
    else:
        loser_from_raw = min(raw_s1, 20.0)
        loser_score = 0.5 * loser_from_raw + 0.5 * margin_loser
        s1 = round_and_clip_score(loser_score, 20)
        s2 = 21

    return s1, s2


def decode_tennis_score(
    pred_diff: float, pred_total: float, running_diff_before_tn: int
) -> tuple[int, int]:
    pred_diff = float(
        np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    )
    pred_total = float(np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT))

    raw_s1, raw_s2 = reconstruct_scores(pred_diff, pred_total)
    raw_s1 = round_and_clip_score(raw_s1, 21)
    raw_s2 = round_and_clip_score(raw_s2, 21)

    if running_diff_before_tn == 0:
        if pred_diff >= 0:
            s1 = max(1, raw_s1)
            s2 = min(raw_s2, max(0, s1 - 1))
        else:
            s2 = max(1, raw_s2)
            s1 = min(raw_s1, max(0, s2 - 1))
        return s1, s2

    if running_diff_before_tn > 0:
        p2_needed = running_diff_before_tn + 1

        if pred_diff >= 0:
            return 1, 0

        s2 = max(p2_needed, raw_s2)
        s2 = min(21, s2)

        s1 = int(round(s2 + pred_diff))
        s1 = max(0, min(21, s1))

        if s1 >= s2:
            s1 = max(0, s2 - 1)

        if (s2 - s1) >= p2_needed:
            return s1, s2

        return max(0, 21 - p2_needed), 21

    p1_needed = -running_diff_before_tn + 1

    if pred_diff < 0:
        return 0, 1

    s1 = max(p1_needed, raw_s1)
    s1 = min(21, s1)

    s2 = int(round(s1 - pred_diff))
    s2 = max(0, min(21, s2))

    if s2 >= s1:
        s2 = max(0, s1 - 1)

    if (s1 - s2) >= p1_needed:
        return s1, s2

    return 21, max(0, 21 - p1_needed)


# -------------------------------------------------
# Models
# -------------------------------------------------
def make_regressor():
    return HistGradientBoostingRegressor(**GB_PARAMS)


def fit_models(df: pd.DataFrame, X: pd.DataFrame, targets, split: int):
    sport_models = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))

    for s in SPORTS:
        rows = np.array(targets[s]["row"])
        if len(rows) < 20:
            continue

        feat_cols = [f"{s}_delta_diff", f"{s}_delta_total", f"{s}_sum_total"]

        X_s = X.loc[rows, feat_cols].values
        y_diff = np.array(targets[s]["diff"])
        y_total = np.array(targets[s]["total"])

        train_mask = rows < split
        test_mask = rows >= split

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, X_test = X_s[train_mask], X_s[test_mask]
        y_diff_train, y_diff_test = y_diff[train_mask], y_diff[test_mask]
        y_total_train, y_total_test = y_total[train_mask], y_total[test_mask]

        model_diff = make_regressor()
        model_total = make_regressor()

        model_diff.fit(X_train, y_diff_train)
        model_total.fit(X_train, y_total_train)

        pred_diff = np.clip(model_diff.predict(X_test), -21, 21)
        pred_total = np.clip(model_total.predict(X_test), 0, 42)

        for idx, row_idx in enumerate(rows[test_mask]):
            if s in ["TT", "BD", "SQ"]:
                s1_hat, s2_hat = decode_full_game_score(
                    pred_diff[idx], pred_total[idx]
                )
            else:
                if PREDICT_TENNIS_INDEPENDENTLY:
                    s1_hat, s2_hat = decode_full_game_score(
                        pred_diff[idx], pred_total[idx]
                    )
                else:
                    running_diff_before_tn = int(
                        round(total_pred_p1[row_idx] - total_pred_p2[row_idx])
                    )
                    s1_hat, s2_hat = decode_tennis_score(
                        pred_diff[idx],
                        pred_total[idx],
                        running_diff_before_tn,
                    )

            total_pred_p1[row_idx] += s1_hat
            total_pred_p2[row_idx] += s2_hat

        print(f"\n{s} TEST RESULTS")
        print("Diff MAE:", mean_absolute_error(y_diff_test, pred_diff))
        print("Total MAE:", mean_absolute_error(y_total_test, pred_total))

        sport_models[s] = {
            "model_diff": model_diff,
            "model_total": model_total,
            "feat_cols": feat_cols,
        }

    return sport_models, total_pred_p1, total_pred_p2


# -------------------------------------------------
# One match
# -------------------------------------------------
def predict_match(
    player1: str, player2: str, stats: RunningStats, sport_models
):
    p1_key = player1.strip().lower()
    p2_key = player2.strip().lower()

    total_p1 = 0
    total_p2 = 0

    print(f"{player1} {player2}")

    for s in SPORTS:
        if s not in sport_models:
            continue

        p1_d, p1_t = stats.get_avgs(p1_key, s, SHRINK)
        p2_d, p2_t = stats.get_avgs(p2_key, s, SHRINK)

        feat = np.array([[p1_d - p2_d, p1_t - p2_t, p1_t + p2_t]])

        pred_diff = float(sport_models[s]["model_diff"].predict(feat)[0])
        pred_total = float(sport_models[s]["model_total"].predict(feat)[0])

        if s in ["TT", "BD", "SQ"]:
            s1, s2 = decode_full_game_score(pred_diff, pred_total)
        else:
            if PREDICT_TENNIS_INDEPENDENTLY:
                s1, s2 = decode_full_game_score(pred_diff, pred_total)
            else:
                running_diff_before_tn = total_p1 - total_p2
                s1, s2 = decode_tennis_score(
                    pred_diff, pred_total, running_diff_before_tn
                )

        total_p1 += s1
        total_p2 += s2

        print(f"{s}: {s1}-{s2}")

    diff = total_p1 - total_p2

    print()
    print(f"Total diff {diff:+d}")
    print()

    if diff > 0:
        print(f"{player1} Wins")
    elif diff < 0:
        print(f"{player2} Wins")
    else:
        print("Draw")


# -------------------------------------------------
# Main
# -------------------------------------------------
def train_model(path: str):
    df = read_matches(path)

    df["team1_players"] = (
        df["team1_players"].astype(str).str.strip().str.lower()
    )
    df["team2_players"] = (
        df["team2_players"].astype(str).str.strip().str.lower()
    )

    X, targets = build_dataset(df, shrink=SHRINK)
    split = int(len(df) * TRAIN_RATIO)

    sport_models, total_pred_p1, total_pred_p2 = fit_models(
        df, X, targets, split
    )

    true_p1 = []
    true_p2 = []
    pred_p1 = []
    pred_p2 = []

    for i in range(split, len(df)):
        t1 = 0
        t2 = 0
        for s in SPORTS:
            s1 = df.loc[i, f"{s}_p1"]
            s2 = df.loc[i, f"{s}_p2"]
            if not pd.isna(s1) and not pd.isna(s2):
                t1 += s1
                t2 += s2

        true_p1.append(t1)
        true_p2.append(t2)
        pred_p1.append(total_pred_p1[i])
        pred_p2.append(total_pred_p2[i])

    true_p1 = np.array(true_p1)
    true_p2 = np.array(true_p2)
    pred_p1 = np.array(pred_p1)
    pred_p2 = np.array(pred_p2)

    print("\n=== MATCH LEVEL ===")
    print("P1 MAE:", mean_absolute_error(true_p1, pred_p1))
    print("P2 MAE:", mean_absolute_error(true_p2, pred_p2))
    print(
        "Winner Accuracy:", ((pred_p1 > pred_p2) == (true_p1 > true_p2)).mean()
    )

    if PREDICT_ONE_MATCH:
        print()
        stats = build_running_stats(df, split)
        predict_match(PLAYER1, PLAYER2, stats, sport_models)


if __name__ == "__main__":
    train_model(DATA_PATH)
