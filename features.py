# features.py
#
# Builds a SINGLE training-ready dataset with:
# - leakage-safe pre-match features (margin-Elo + H2H up to that point)
# - targets (per-sport diffs + total diff + winner)
#
# Output: data/data.csv

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

SPORTS = ["TT", "BD", "SQ", "TN"]

# -----------------------
# Identity config
# -----------------------
USE_IDS = True  # recommended; uses team*_player_ids as identity key if present

# -----------------------
# Margin-Elo config
# -----------------------
BASE = 0.0
MAX_DIFF = 21.0

# pred_diff = 21 * tanh(alpha*(r1-r2)/2)
ALPHAS = {"TT": 0.010, "BD": 0.010, "SQ": 0.010, "TN": 0.010}

# Learning-rate schedule (dynamic)
ETA_MIN = 0.02
ETA_MAX = 0.20
ETA_TAU = 15.0

# small regularization on rating differences
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
    return MAX_DIFF * math.tanh((alpha * x) / 2.0)


def d_pred_dx(x: float, alpha: float) -> float:
    t = math.tanh((alpha * x) / 2.0)
    return MAX_DIFF * (alpha / 2.0) * (1.0 - t * t)


def safe_datetime(df: pd.DataFrame) -> pd.Series:
    # Use existing datetime if present
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.notna().any():
            return dt

    # Otherwise build from match_date + match_time
    d = pd.to_datetime(df["match_date"], format="%Y%m%d", errors="coerce")
    if "match_time" in df.columns:
        dt = pd.to_datetime(
            d.astype(str) + " " + df["match_time"].astype(str), errors="coerce"
        )
        return dt.fillna(d)
    return d


def player_key(row: pd.Series, side: int) -> str:
    if USE_IDS:
        col = f"team{side}_player_ids"
        if col in row and not pd.isna(row[col]):
            return str(row[col])
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
            d = diffs.get(s)
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
            step = eta * err * grad
            step -= eta * L2 * x

            self.R[p1][s] = r1 + step
            self.R[p2][s] = r2 - step

            self.games[p1][s] += 1
            self.games[p2][s] += 1


# -----------------------
# Main
# -----------------------
def build_training_data(
    in_csv: str = "data/matches_cleaned.csv",
    out_csv: str = "data/data.csv",
) -> None:
    df = pd.read_csv(in_csv, low_memory=False)
    df["datetime"] = safe_datetime(df)
    df = df.sort_values("datetime").reset_index(drop=True)

    elo = MarginElo()

    h2h_overall: Dict[Tuple[str, str], H2HStats] = defaultdict(H2HStats)
    h2h_sport: Dict[str, Dict[Tuple[str, str], H2HStats]] = {
        s: defaultdict(H2HStats) for s in SPORTS
    }

    out_rows: List[dict] = []

    for i, row in df.iterrows():
        dt = row["datetime"]
        p1_key = player_key(row, 1)
        p2_key = player_key(row, 2)

        p1_name = str(row.get("team1_players", "")).strip().lower()
        p2_name = str(row.get("team2_players", "")).strip().lower()

        # -------- Pre-match features (NO leakage) --------
        out = {
            "match_index": i,
            "datetime": dt,
            "p1_key": p1_key,
            "p2_key": p2_key,
            "p1_name": p1_name,
            "p2_name": p2_name,
        }

        pred = elo.predict(p1_key, p2_key)
        for s in SPORTS:
            out[f"{s}_rating_diff"] = elo.R[p1_key][s] - elo.R[p2_key][s]
            out[f"{s}_pred_diff"] = pred[s]
            out[f"{s}_games_p1"] = elo.games[p1_key][s]
            out[f"{s}_games_p2"] = elo.games[p2_key][s]
            out[f"{s}_games_diff"] = elo.games[p1_key][s] - elo.games[p2_key][s]

        pk = pair_key(p1_key, p2_key)
        a = pk[0]
        a_is_p1 = a == p1_key

        # overall h2h
        out.update(h2h_features(h2h_overall[pk], a_is_p1, dt, prefix=""))
        # per sport h2h
        for s in SPORTS:
            out.update(
                h2h_features(h2h_sport[s][pk], a_is_p1, dt, prefix=f"{s}_")
            )

        # -------- Targets (OK to include in the same row) --------
        diffs = {s: get_sport_diff(row, s) for s in SPORTS}
        total = 0.0
        any_sport = False
        for s in SPORTS:
            d = diffs[s]
            has = 1 if d is not None else 0
            out[f"has_{s}"] = has
            if d is not None:
                d = clip_diff(d)
                out[f"{s}_y_diff"] = d
                total += d
                any_sport = True
            else:
                out[f"{s}_y_diff"] = np.nan

        out["y_total_diff"] = total if any_sport else np.nan
        out["y_winner_p1"] = (1 if total > 0 else 0) if any_sport else np.nan

        out_rows.append(out)

        # -------- Update states AFTER features (no leakage) --------
        elo.update(p1_key, p2_key, diffs)

        # update h2h per sport + overall
        total_diff_for_h2h = 0.0
        any_for_h2h = False
        for s in SPORTS:
            d = diffs[s]
            if d is None:
                continue
            any_for_h2h = True
            total_diff_for_h2h += d
            h2h_update(h2h_sport[s][pk], a_is_p1, d, dt)
        if any_for_h2h:
            h2h_update(h2h_overall[pk], a_is_p1, total_diff_for_h2h, dt)

    out_df = pd.DataFrame(out_rows)

    # Keep only "necessary" columns:
    # (Everything in out_df is already necessary by design, but we’ll enforce order.)
    id_cols = [
        "match_index",
        "datetime",
        "p1_key",
        "p2_key",
        "p1_name",
        "p2_name",
    ]
    feature_cols = [
        c
        for c in out_df.columns
        if c not in id_cols
        and not c.endswith("_y_diff")
        and not c.startswith("y_")
    ]
    target_cols = [
        c
        for c in out_df.columns
        if c.endswith("_y_diff") or c in ("y_total_diff", "y_winner_p1")
    ]

    out_df = out_df[id_cols + sorted(feature_cols) + sorted(target_cols)]
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with shape {out_df.shape} (USE_IDS={USE_IDS})")


if __name__ == "__main__":
    build_training_data()
