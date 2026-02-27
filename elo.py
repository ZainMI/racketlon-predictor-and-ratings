import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List

SPORTS = ["TT", "BD", "SQ", "TN"]

# Elo hyperparameters
BASE_RATING = 1500.0
K = 24.0  # try 16, 24, 32
ELO_SCALE = 400.0  # standard chess scale
MAX_DIFF = 21.0  # your stated cap for margin scaling


def read_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y%m%d")
    if "match_time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["match_date"].astype(str) + " " + df["match_time"],
            errors="coerce",
        )
        df = df.sort_values("datetime")
    else:
        df = df.sort_values("match_date")
    return df.reset_index(drop=True)


def expected_score(r_a: float, r_b: float, scale: float = ELO_SCALE) -> float:
    # Expected probability A beats B
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))


def mov_multiplier(diff: float) -> float:
    """
    Margin-of-victory multiplier using bounded diff.
    - diff is (s1 - s2)
    - we cap |diff| at MAX_DIFF to respect your rule
    - multiplier is in [1, 2]-ish by default
    """
    d = min(abs(diff), MAX_DIFF)
    # Smooth, bounded: 1 + d/MAX_DIFF in [1,2]
    return 1.0 + (d / MAX_DIFF)


@dataclass
class EloState:
    # ratings[player][sport] -> float
    ratings: Dict[str, Dict[str, float]]

    def __init__(self):
        self.ratings = defaultdict(lambda: defaultdict(lambda: BASE_RATING))

    def get(self, player: str, sport: str) -> float:
        return float(self.ratings[player][sport])

    def set(self, player: str, sport: str, value: float) -> None:
        self.ratings[player][sport] = float(value)


def compute_elo_features_and_update(
    df: pd.DataFrame, use_margin: bool = True, k: float = K
) -> Tuple[pd.DataFrame, EloState]:
    """
    Returns:
      X_elo: per-match features (pre-match rating diffs + win probs)
      state: final Elo state after processing all matches
    """
    state = EloState()
    rows: List[dict] = []

    for i, row in df.iterrows():
        p1 = row["team1_players"]
        p2 = row["team2_players"]

        feat = {"match_index": i, "p1": p1, "p2": p2}

        # --- Pre-match features per sport ---
        for s in SPORTS:
            r1 = state.get(p1, s)
            r2 = state.get(p2, s)
            feat[f"{s}_elo_p1"] = r1
            feat[f"{s}_elo_p2"] = r2
            feat[f"{s}_elo_diff"] = r1 - r2
            feat[f"{s}_elo_pwin_p1"] = expected_score(r1, r2)

        rows.append(feat)

        # --- Update ratings after match results (no leakage) ---
        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue

            s1 = float(s1)
            s2 = float(s2)
            diff = s1 - s2

            # outcome from p1 perspective
            if diff > 0:
                a = 1.0
            elif diff < 0:
                a = 0.0
            else:
                a = 0.5  # ties unlikely, but safe

            r1 = state.get(p1, s)
            r2 = state.get(p2, s)
            e1 = expected_score(r1, r2)

            mult = mov_multiplier(diff) if use_margin else 1.0
            delta = k * mult * (a - e1)

            state.set(p1, s, r1 + delta)
            state.set(p2, s, r2 - delta)

    X_elo = pd.DataFrame(rows)
    return X_elo, state


# --- Example usage: make Elo features and train your regressors on them ---
if __name__ == "__main__":
    df = read_matches("data/matches_cleaned.csv")
    X_elo, state = compute_elo_features_and_update(df, use_margin=True, k=24.0)

    # Example: show current Elo for a player
    player = "zain magdon-ismail"
    print({s: state.get(player, s) for s in SPORTS})

    # Save features for modeling
    X_elo.to_csv("elo_features.csv", index=False)
    print("Saved elo_features.csv")
