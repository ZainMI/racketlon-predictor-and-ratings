import pandas as pd
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 84  # 21 + 21


# -------------------------------------------------
# Load + Sort Chronologically
# -------------------------------------------------
def read_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Convert YYYYMMDD → datetime
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y%m%d")

    # Combine date + time if available
    if "match_time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["match_date"].astype(str) + " " + df["match_time"],
            errors="coerce",
        )
        df = df.sort_values("datetime")
    else:
        df = df.sort_values("match_date")

    df = df.reset_index(drop=True)
    return df


# -------------------------------------------------
# Running Pre-Match Averages (NO LEAKAGE)
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

    def get_avgs(self, player, sport, shrink=10):
        c = self.count[player][sport]
        d = self.diff_sum[player][sport]
        t = self.total_sum[player][sport]

        return (
            d / (c + shrink),
            t / (c + shrink),
        )

    def update(self, p1, p2, sport, s1, s2):
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
def build_dataset(df, shrink=10):

    stats = RunningStats()
    X_rows = []
    targets = {s: {"diff": [], "total": [], "row": []} for s in SPORTS}

    for i, row in df.iterrows():
        p1 = row["team1_players"]
        p2 = row["team2_players"]

        feat = {}

        # --- pre-match features ---
        for s in SPORTS:
            p1_d, p1_t = stats.get_avgs(p1, s, shrink)
            p2_d, p2_t = stats.get_avgs(p2, s, shrink)

            feat[f"{s}_delta_diff"] = p1_d - p2_d
            feat[f"{s}_delta_total"] = p1_t - p2_t
            feat[f"{s}_sum_total"] = p1_t + p2_t

        X_rows.append(feat)

        # --- store targets + update ---
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

    X = pd.DataFrame(X_rows)
    return X, targets


# -------------------------------------------------
# Score Reconstruction
# -------------------------------------------------
def reconstruct_scores(pred_diff, pred_total):

    pred_diff = np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    pred_total = np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT)

    s1 = 0.5 * (pred_total + pred_diff)
    s2 = 0.5 * (pred_total - pred_diff)

    s1 = np.clip(s1, 0, MAX_POINTS_PER_SPORT)
    s2 = np.clip(s2, 0, MAX_POINTS_PER_SPORT)

    return s1, s2


# -------------------------------------------------
# Train + Evaluate
# -------------------------------------------------
def train_model(path):

    df = read_matches(path)
    X, targets = build_dataset(df)

    n = len(df)
    split = int(n * 0.8)

    total_pred_p1 = np.zeros(n)
    total_pred_p2 = np.zeros(n)

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

        X_train, X_test = X_s[train_mask], X_s[test_mask]
        y_diff_train, y_diff_test = y_diff[train_mask], y_diff[test_mask]
        y_total_train, y_total_test = y_total[train_mask], y_total[test_mask]

        model_diff = Ridge(alpha=1.0)
        model_total = Ridge(alpha=1.0)

        model_diff.fit(X_train, y_diff_train)
        model_total.fit(X_train, y_total_train)

        pred_diff = model_diff.predict(X_test)
        pred_total = model_total.predict(X_test)

        s1_hat, s2_hat = reconstruct_scores(pred_diff, pred_total)

        total_pred_p1[rows[test_mask]] += s1_hat
        total_pred_p2[rows[test_mask]] += s2_hat

        print(f"\n{s} TEST RESULTS")
        print(
            "Diff MAE:",
            mean_absolute_error(y_diff_test, np.clip(pred_diff, -21, 21)),
        )
        print(
            "Total MAE:",
            mean_absolute_error(y_total_test, np.clip(pred_total, 0, 42)),
        )

    # ----- Match-level evaluation -----
    true_p1 = []
    true_p2 = []
    pred_p1 = []
    pred_p2 = []

    for i in range(split, n):
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

    acc = ((pred_p1 > pred_p2) == (true_p1 > true_p2)).mean()
    print("Winner Accuracy:", acc)


if __name__ == "__main__":
    train_model("data/matches_cleaned.csv")
