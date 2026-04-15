import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42

DATA_PATH = "data/matches_cleaned.csv"
OUTPUT_DIR = "finished_models/simple_benchmark/artifacts/predictor_package"
TRAIN_RATIO = 0.8
SHRINK = 5.0
PREDICT_TENNIS_INDEPENDENTLY = True


# -------------------------------------------------
# IO
# -------------------------------------------------
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        d = pd.to_datetime(df["match_date"], format="%Y%m%d", errors="coerce")
        if "match_time" in df.columns:
            dt = pd.to_datetime(
                d.astype(str) + " " + df["match_time"].astype(str),
                errors="coerce",
            )
            df["datetime"] = dt.fillna(d)
        else:
            df["datetime"] = d

    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["team1_players", "team2_players"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


# -------------------------------------------------
# Pickle-safe running stats
# -------------------------------------------------
def float_dd():
    return defaultdict(float)


def int_dd():
    return defaultdict(int)


@dataclass
class RunningPlayerStats:
    diff_sum: dict
    total_sum: dict
    count: dict

    def __init__(self):
        self.diff_sum = defaultdict(float_dd)
        self.total_sum = defaultdict(float_dd)
        self.count = defaultdict(int_dd)

    def get_avg_diff(
        self, player: str, sport: str, shrink: float = 5.0
    ) -> float:
        return self.diff_sum[player][sport] / (
            self.count[player][sport] + shrink
        )

    def get_avg_total(
        self, player: str, sport: str, shrink: float = 5.0
    ) -> float:
        return self.total_sum[player][sport] / (
            self.count[player][sport] + shrink
        )

    def update(self, p1: str, p2: str, sport: str, s1: float, s2: float):
        diff = float(s1) - float(s2)
        total = float(s1) + float(s2)

        self.diff_sum[p1][sport] += diff
        self.total_sum[p1][sport] += total
        self.count[p1][sport] += 1

        self.diff_sum[p2][sport] -= diff
        self.total_sum[p2][sport] += total
        self.count[p2][sport] += 1


def stats_to_plain_dict(stats: RunningPlayerStats) -> dict:
    return {
        "diff_sum": {
            player: dict(sport_map)
            for player, sport_map in stats.diff_sum.items()
        },
        "total_sum": {
            player: dict(sport_map)
            for player, sport_map in stats.total_sum.items()
        },
        "count": {
            player: dict(sport_map) for player, sport_map in stats.count.items()
        },
    }


def stats_from_plain_dict(obj: dict) -> RunningPlayerStats:
    stats = RunningPlayerStats()

    for player, sport_map in obj.get("diff_sum", {}).items():
        for sport, val in sport_map.items():
            stats.diff_sum[player][sport] = float(val)

    for player, sport_map in obj.get("total_sum", {}).items():
        for sport, val in sport_map.items():
            stats.total_sum[player][sport] = float(val)

    for player, sport_map in obj.get("count", {}).items():
        for sport, val in sport_map.items():
            stats.count[player][sport] = int(val)

    return stats


# -------------------------------------------------
# Score decoding
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
        return 21, round_and_clip_score(loser_score, 20)

    loser_from_raw = min(raw_s1, 20.0)
    loser_score = 0.5 * loser_from_raw + 0.5 * margin_loser
    return round_and_clip_score(loser_score, 20), 21


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
# Predictor package
# -------------------------------------------------
@dataclass
class PredictorPackage:
    player_stats: RunningPlayerStats
    shrink: float

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "predictor.pkl", "rb") as f:
            obj = pickle.load(f)

        stats = stats_from_plain_dict(obj["player_stats"])
        return cls(
            player_stats=stats,
            shrink=float(obj["shrink"]),
        )

    def predict_pair(self, player1: str, player2: str):
        p1 = player1.strip().lower()
        p2 = player2.strip().lower()

        total_p1 = 0
        total_p2 = 0
        sports_out = {}

        for sport in SPORTS:
            p1_avg_diff = self.player_stats.get_avg_diff(p1, sport, self.shrink)
            p2_avg_diff = self.player_stats.get_avg_diff(p2, sport, self.shrink)

            p1_avg_total = self.player_stats.get_avg_total(
                p1, sport, self.shrink
            )
            p2_avg_total = self.player_stats.get_avg_total(
                p2, sport, self.shrink
            )

            pred_diff = float(np.clip(p1_avg_diff - p2_avg_diff, -21, 21))
            pred_total = float(
                np.clip(0.5 * (p1_avg_total + p2_avg_total), 18, 42)
            )

            if sport in ["TT", "BD", "SQ"]:
                s1, s2 = decode_full_game_score(pred_diff, pred_total)
            else:
                if PREDICT_TENNIS_INDEPENDENTLY:
                    s1, s2 = decode_full_game_score(pred_diff, pred_total)
                else:
                    s1, s2 = decode_tennis_score(
                        pred_diff, pred_total, total_p1 - total_p2
                    )

            total_p1 += s1
            total_p2 += s2

            sports_out[sport] = {
                "score_p1": int(s1),
                "score_p2": int(s2),
                "pred_diff": float(pred_diff),
                "pred_total": float(pred_total),
                "p1_avg_diff": float(p1_avg_diff),
                "p2_avg_diff": float(p2_avg_diff),
            }

        total_diff = total_p1 - total_p2
        winner = (
            player1 if total_diff > 0 else player2 if total_diff < 0 else "Draw"
        )

        return {
            "player1": player1,
            "player2": player2,
            "sports": sports_out,
            "total_p1": int(total_p1),
            "total_p2": int(total_p2),
            "total_diff": int(total_diff),
            "winner": winner,
        }


# -------------------------------------------------
# Train / evaluate
# -------------------------------------------------
def train_and_package():
    outdir = ensure_dir(OUTPUT_DIR)
    df = read_matches(DATA_PATH)

    split = int(len(df) * TRAIN_RATIO)

    stats = RunningPlayerStats()

    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))

    diff_targets = {s: [] for s in SPORTS}
    diff_preds = {s: [] for s in SPORTS}
    total_targets = {s: [] for s in SPORTS}
    total_preds = {s: [] for s in SPORTS}

    for i, row in df.iterrows():
        p1 = row["team1_players"]
        p2 = row["team2_players"]

        pred_by_sport = {}

        for sport in SPORTS:
            p1_avg_diff = stats.get_avg_diff(p1, sport, SHRINK)
            p2_avg_diff = stats.get_avg_diff(p2, sport, SHRINK)

            p1_avg_total = stats.get_avg_total(p1, sport, SHRINK)
            p2_avg_total = stats.get_avg_total(p2, sport, SHRINK)

            pred_diff = float(np.clip(p1_avg_diff - p2_avg_diff, -21, 21))
            pred_total = float(
                np.clip(0.5 * (p1_avg_total + p2_avg_total), 18, 42)
            )
            pred_by_sport[sport] = (pred_diff, pred_total)

        if i >= split:
            running_p1 = 0
            running_p2 = 0

            for sport in SPORTS:
                s1 = row.get(f"{sport}_p1")
                s2 = row.get(f"{sport}_p2")
                if pd.isna(s1) or pd.isna(s2):
                    continue

                actual_diff = float(s1) - float(s2)
                actual_total = float(s1) + float(s2)

                pred_diff, pred_total = pred_by_sport[sport]

                diff_targets[sport].append(actual_diff)
                diff_preds[sport].append(pred_diff)
                total_targets[sport].append(actual_total)
                total_preds[sport].append(pred_total)

                if sport in ["TT", "BD", "SQ"]:
                    s1_hat, s2_hat = decode_full_game_score(
                        pred_diff, pred_total
                    )
                else:
                    if PREDICT_TENNIS_INDEPENDENTLY:
                        s1_hat, s2_hat = decode_full_game_score(
                            pred_diff, pred_total
                        )
                    else:
                        s1_hat, s2_hat = decode_tennis_score(
                            pred_diff, pred_total, running_p1 - running_p2
                        )

                running_p1 += s1_hat
                running_p2 += s2_hat
                total_pred_p1[i] += s1_hat
                total_pred_p2[i] += s2_hat

        for sport in SPORTS:
            s1 = row.get(f"{sport}_p1")
            s2 = row.get(f"{sport}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue
            stats.update(p1, p2, sport, float(s1), float(s2))

    metrics = {}
    for sport in SPORTS:
        if len(diff_targets[sport]) == 0:
            continue

        diff_mae = float(
            mean_absolute_error(diff_targets[sport], diff_preds[sport])
        )
        total_mae = float(
            mean_absolute_error(total_targets[sport], total_preds[sport])
        )

        print(f"\n{sport} TEST RESULTS")
        print("Diff MAE:", diff_mae)
        print("Total MAE:", total_mae)

        metrics[sport] = {
            "diff_mae": diff_mae,
            "total_mae": total_mae,
            "test_n": len(diff_targets[sport]),
        }

    true_diff = []
    pred_diff = []

    for i in range(split, len(df)):
        row = df.iloc[i]
        match_total_diff = 0.0
        has_any = False

        for sport in SPORTS:
            s1 = row.get(f"{sport}_p1")
            s2 = row.get(f"{sport}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue
            match_total_diff += float(s1) - float(s2)
            has_any = True

        if not has_any:
            continue

        true_diff.append(match_total_diff)
        pred_diff.append(float(total_pred_p1[i] - total_pred_p2[i]))

    true_diff = np.array(true_diff)
    pred_diff = np.array(pred_diff)

    match_mae = float(mean_absolute_error(true_diff, pred_diff))
    winner_acc = float(((pred_diff > 0) == (true_diff > 0)).mean())

    print("\n=== MATCH LEVEL ===")
    print("Total Diff MAE:", match_mae)
    print("Winner Accuracy:", winner_acc)

    metadata = {
        "data_path": DATA_PATH,
        "train_ratio": TRAIN_RATIO,
        "shrink": SHRINK,
        "n_rows": int(len(df)),
        "sport_metrics": metrics,
        "match_metrics": {
            "total_diff_mae": match_mae,
            "winner_accuracy": winner_acc,
        },
    }

    with open(outdir / "predictor.pkl", "wb") as f:
        pickle.dump(
            {
                "player_stats": stats_to_plain_dict(stats),
                "shrink": SHRINK,
            },
            f,
        )

    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved simple benchmark package to: {outdir}")


if __name__ == "__main__":
    train_and_package()
