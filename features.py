# features.py
#
# Build modeling features with NO leakage:
# - Margin-Elo (per sport): rating_p1, rating_p2, rating_diff, pred_diff, games_p1, games_p2
# - Head-to-head up to that match (per sport + overall): h2h_games, h2h_winrate_p1, h2h_avg_diff_p1, h2h_days_since_last, etc.
#
# Output: data/features.csv

import math
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

SPORTS = ["TT", "BD", "SQ", "TN"]

# -----------------------
# Identity config
# -----------------------
USE_IDS = True  # strongly recommended

# -----------------------
# Margin-Elo config
# -----------------------
BASE = 0.0
MAX_DIFF = 21.0

# Link function: pred_diff = 21 * tanh(alpha * (r1-r2) / 2)
ALPHAS = {"TT": 0.010, "BD": 0.010, "SQ": 0.010, "TN": 0.010}

# Learning rate schedule (dynamic, like K)
ETA_MIN = 0.02
ETA_MAX = 0.20
ETA_TAU = 15.0

# Small regularization to prevent drift
L2 = 1e-5


# -----------------------
# Utilities
# -----------------------
def eta_eff(games_played: int) -> float:
    gp = max(0, int(games_played))
    return ETA_MIN + (ETA_MAX - ETA_MIN) * math.exp(-gp / ETA_TAU)


def clip_diff(d: float) -> float:
    return float(np.clip(d, -MAX_DIFF, MAX_DIFF))


def rating_to_score_diff(x: float, alpha: float) -> float:
    # stable saturating mapping
    return MAX_DIFF * math.tanh((alpha * x) / 2.0)


def d_pred_dx(x: float, alpha: float) -> float:
    # derivative of 21*tanh(alpha*x/2)
    t = math.tanh((alpha * x) / 2.0)
    return MAX_DIFF * (alpha / 2.0) * (1.0 - t * t)


def safe_datetime(df: pd.DataFrame) -> pd.Series:
    # If datetime exists and parses, use it
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.notna().any():
            return dt

    # Else build from match_date + match_time
    d = pd.to_datetime(df["match_date"], format="%Y%m%d", errors="coerce")
    if "match_time" in df.columns:
        dt = pd.to_datetime(
            d.astype(str) + " " + df["match_time"].astype(str), errors="coerce"
        )
        return dt.fillna(d)
    return d


def player_key(row: pd.Series, side: int) -> str:
    """
    side=1 -> team1, side=2 -> team2
    """
    if USE_IDS:
        col = f"team{side}_player_ids"
        if col in row and not pd.isna(row[col]):
            return str(row[col])
    # fallback to name
    return str(row.get(f"team{side}_players", "")).strip().lower()


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def get_sport_diff(row: pd.Series, sport: str) -> Optional[float]:
    a = row.get(f"{sport}_p1")
    b = row.get(f"{sport}_p2")
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


# -----------------------
# H2H state
# -----------------------
@dataclass
class H2HStats:
    games: int = 0
    wins_a: int = 0
    sum_diff_a: float = 0.0
    last_dt: Optional[pd.Timestamp] = None


def h2h_update(
    st: H2HStats, a_is_p1: bool, diff_p1: float, dt: pd.Timestamp
) -> None:
    st.games += 1
    diff_a = diff_p1 if a_is_p1 else -diff_p1
    st.sum_diff_a += diff_a
    if diff_a > 0:
        st.wins_a += 1
    st.last_dt = dt


def h2h_features(
    st: H2HStats, a_is_p1: bool, dt: pd.Timestamp, prefix: str
) -> Dict[str, float]:
    out = {}
    out[f"{prefix}h2h_games"] = st.games

    if st.games > 0:
        winrate_a = st.wins_a / st.games
        avgdiff_a = st.sum_diff_a / st.games
    else:
        winrate_a = 0.5
        avgdiff_a = 0.0

    # convert from "a perspective" to p1 perspective
    out[f"{prefix}h2h_winrate_p1"] = winrate_a if a_is_p1 else (1.0 - winrate_a)
    out[f"{prefix}h2h_avg_diff_p1"] = avgdiff_a if a_is_p1 else (-avgdiff_a)

    if st.last_dt is None or pd.isna(dt):
        out[f"{prefix}h2h_days_since_last"] = np.nan
    else:
        out[f"{prefix}h2h_days_since_last"] = float((dt - st.last_dt).days)

    return out


# -----------------------
# Margin-Elo state
# -----------------------
class MarginElo:
    def __init__(self):
        self.R = defaultdict(lambda: {s: BASE for s in SPORTS})
        self.games = defaultdict(lambda: {s: 0 for s in SPORTS})

    def predict(self, p1: str, p2: str) -> Dict[str, float]:
        pred = {}
        for s in SPORTS:
            x = self.R[p1][s] - self.R[p2][s]
            pred[s] = rating_to_score_diff(x, ALPHAS[s])
        return pred

    def update(
        self, p1: str, p2: str, diffs: Dict[str, Optional[float]]
    ) -> None:
        for s in SPORTS:
            d = diffs.get(s, None)
            if d is None:
                continue

            d = clip_diff(d)
            r1 = self.R[p1][s]
            r2 = self.R[p2][s]
            x = r1 - r2

            pred = rating_to_score_diff(x, ALPHAS[s])
            err = d - pred

            g1 = self.games[p1][s]
            g2 = self.games[p2][s]
            eta = 0.5 * (eta_eff(g1) + eta_eff(g2))

            grad = d_pred_dx(x, ALPHAS[s])

            # step is proportional to error and derivative of predictor
            step = eta * err * grad

            # small L2 on the diff (keeps things bounded)
            step -= eta * L2 * x

            self.R[p1][s] = r1 + step
            self.R[p2][s] = r2 - step

            self.games[p1][s] += 1
            self.games[p2][s] += 1


# -----------------------
# Main feature builder
# -----------------------
def build_features(
    in_csv: str = "data/matches_cleaned.csv",
    out_csv: str = "data/features.csv",
) -> None:
    df = pd.read_csv(in_csv, low_memory=False)

    df["datetime"] = safe_datetime(df)
    df = df.sort_values("datetime").reset_index(drop=True)

    elo = MarginElo()

    # H2H tables: overall and per sport
    h2h_overall: Dict[Tuple[str, str], H2HStats] = defaultdict(H2HStats)
    h2h_sport: Dict[str, Dict[Tuple[str, str], H2HStats]] = {
        s: defaultdict(H2HStats) for s in SPORTS
    }

    out_rows: List[dict] = []

    for i, row in df.iterrows():
        dt = row["datetime"]
        p1 = player_key(row, 1)
        p2 = player_key(row, 2)

        # Pre-match snapshot features
        out = {
            "match_index": i,
            "datetime": dt,
            "p1": row.get("team1_players", p1),
            "p2": row.get("team2_players", p2),
            "p1_key": p1,
            "p2_key": p2,
        }

        # ----- Elo features (pre-match) -----
        pred = elo.predict(p1, p2)
        for s in SPORTS:
            out[f"{s}_rating_p1"] = elo.R[p1][s]
            out[f"{s}_rating_p2"] = elo.R[p2][s]
            out[f"{s}_rating_diff"] = elo.R[p1][s] - elo.R[p2][s]
            out[f"{s}_pred_diff"] = pred[s]
            out[f"{s}_games_p1"] = elo.games[p1][s]
            out[f"{s}_games_p2"] = elo.games[p2][s]
            out[f"{s}_games_diff"] = elo.games[p1][s] - elo.games[p2][s]

        # ----- H2H features (pre-match) -----
        pk = pair_key(p1, p2)
        a = pk[0]
        a_is_p1 = a == p1

        # overall H2H
        out.update(h2h_features(h2h_overall[pk], a_is_p1, dt, prefix=""))

        # per sport H2H
        for s in SPORTS:
            out.update(
                h2h_features(h2h_sport[s][pk], a_is_p1, dt, prefix=f"{s}_")
            )

        out_rows.append(out)

        # ----- Update states AFTER computing features -----
        diffs = {s: get_sport_diff(row, s) for s in SPORTS}

        # update Elo using available sport diffs
        elo.update(p1, p2, diffs)

        # update H2H per sport and overall
        total_diff = 0.0
        any_sport = False
        for s in SPORTS:
            d = diffs[s]
            if d is None:
                continue
            any_sport = True
            total_diff += d
            h2h_update(h2h_sport[s][pk], a_is_p1, d, dt)

        if any_sport:
            h2h_update(h2h_overall[pk], a_is_p1, total_diff, dt)

    feats = pd.DataFrame(out_rows)

    # If you want, you can merge back some original columns here.
    # For now, output only engineered features + player/date keys.
    feats.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with shape {feats.shape} (USE_IDS={USE_IDS})")


if __name__ == "__main__":
    build_features()
