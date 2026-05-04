# features.py
#
# Builds a SINGLE training-ready dataset with:
# - leakage-safe pre-match features
# - ONE recency-sensitive rating per player per sport
# - no snapshot ratings
# - H2H features
# - recent-form / dominance / volatility / residual features
# - long-term shrunk performance features
# - targets
# - FINAL POST-HISTORY INFERENCE STATE for synthetic future matchups
#
# Outputs:
#   data/data.csv
#   data/inference_state.pkl

import math
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

SPORTS = ["TT", "BD", "SQ", "TN"]

# =================================================
# Identity config
# =================================================
USE_IDS = True

# =================================================
# One-rating-per-sport config
# =================================================
BASE = 0.0
MAX_DIFF = 21.0

ALPHAS = {
    "TT": 0.060,
    "BD": 0.080,
    "SQ": 0.060,
    "TN": 0.040,
}

ETA_MIN = 0.04
ETA_MAX = 0.30
ETA_TAU = 20.0
L2 = 1e-5

MARGIN_MULT_BY_SPORT = {
    "TT": 0.6,
    "BD": 0.8,
    "SQ": 0.6,
    "TN": 0.4,
}

# Longer inactivity => next match updates rating more.
# Rating itself does NOT shrink.
TIME_C = {
    "TT": 0.6,
    "BD": 0.7,
    "SQ": 0.6,
    "TN": 0.5,
}

TIME_TAU_DAYS = {
    "TT": 90.0,
    "BD": 120.0,
    "SQ": 90.0,
    "TN": 120.0,
}

# =================================================
# Feature config
# =================================================
BLOWOUT_WIN_THRESHOLD = 10.0
BLOWOUT_LOSS_THRESHOLD = -10.0
CLOSE_MATCH_ABS_THRESHOLD = 3.0
FAVORITE_EXPECTED_DIFF_THRESHOLD = 5.0
EWM_ALPHA = 0.25
LONG_TERM_SHRINK = 10.0


# =================================================
# Shared utilities
# =================================================
def safe_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.notna().any():
            return dt

    d = pd.to_datetime(df["match_date"], format="%Y%m%d", errors="coerce")
    if "match_time" in df.columns:
        dt = pd.to_datetime(
            d.astype(str) + " " + df["match_time"].astype(str),
            errors="coerce",
        )
        return dt.fillna(d)
    return d


def normalize_player_name(x) -> str:
    return str(x).strip().lower()


def player_key(row: pd.Series, side: int) -> str:
    """
    Generates a unique key using:
    name + country

    Format:
    name__country
    """
    name = normalize_player_name(row.get(f"team{side}_players", "unknown"))
    country = str(row.get(f"team{side}_nationalities", "")).strip().lower()

    # Fallback if country missing
    if not country or country == "nan":
        return f"name::{name}"

    return f"{name}__{country}"


def player_name(row: pd.Series, side: int) -> str:
    return normalize_player_name(row.get(f"team{side}_players", ""))


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def get_sport_diff(row: pd.Series, sport: str) -> Optional[float]:
    a = row.get(f"{sport}_p1")
    b = row.get(f"{sport}_p2")
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def get_sport_total(row: pd.Series, sport: str) -> Optional[float]:
    a = row.get(f"{sport}_p1")
    b = row.get(f"{sport}_p2")
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) + float(b)


def get_month_key(dt: pd.Timestamp) -> Optional[str]:
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def clip_diff(d: float) -> float:
    return float(np.clip(d, -MAX_DIFF, MAX_DIFF))


def rating_to_score_diff(x: float, alpha: float) -> float:
    return MAX_DIFF * math.tanh((alpha * x) / 2.0)


def d_pred_dx(x: float, alpha: float) -> float:
    t = math.tanh((alpha * x) / 2.0)
    return MAX_DIFF * (alpha / 2.0) * (1.0 - t * t)


def eta_eff(games_played: int) -> float:
    gp = max(0, int(games_played))
    return ETA_MIN + (ETA_MAX - ETA_MIN) * math.exp(-gp / ETA_TAU)


def days_since(last_dt: Optional[pd.Timestamp], dt: pd.Timestamp) -> float:
    if last_dt is None or pd.isna(last_dt) or pd.isna(dt):
        return 0.0
    return float(max(0, (dt - last_dt).days))


def time_multiplier(sport: str, gap_days: float) -> float:
    return 1.0 + TIME_C[sport] * (
        1.0 - math.exp(-gap_days / TIME_TAU_DAYS[sport])
    )


def add_prefixed(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def add_pairwise_deltas(
    out: Dict[str, float],
    sport: str,
    p1_feats: Dict[str, float],
    p2_feats: Dict[str, float],
) -> None:
    for k in p1_feats.keys():
        out[f"{sport}_{k}_diff_p1_p2"] = float(p1_feats[k]) - float(p2_feats[k])


# =================================================
# Rolling recent-form state
# =================================================
@dataclass
class RollingWindow:
    maxlen: int
    values: deque = field(default_factory=deque)
    sum_: float = 0.0
    sumsq_: float = 0.0

    def push(self, x: float) -> None:
        x = float(x)
        if len(self.values) == self.maxlen:
            old = self.values.popleft()
            self.sum_ -= old
            self.sumsq_ -= old * old
        self.values.append(x)
        self.sum_ += x
        self.sumsq_ += x * x

    def count(self) -> int:
        return len(self.values)

    def mean(self, default: float = 0.0) -> float:
        n = len(self.values)
        return float(self.sum_ / n) if n > 0 else float(default)

    def std(self, default: float = 0.0) -> float:
        n = len(self.values)
        if n <= 1:
            return float(default)
        mean = self.sum_ / n
        var = max(0.0, self.sumsq_ / n - mean * mean)
        return float(math.sqrt(var))


@dataclass
class PlayerSportRecentState:
    n_matches: int = 0
    last_dt: Optional[pd.Timestamp] = None

    diff5: RollingWindow = field(default_factory=lambda: RollingWindow(5))
    diff10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    diff20: RollingWindow = field(default_factory=lambda: RollingWindow(20))

    total5: RollingWindow = field(default_factory=lambda: RollingWindow(5))
    total10: RollingWindow = field(default_factory=lambda: RollingWindow(10))

    win5: RollingWindow = field(default_factory=lambda: RollingWindow(5))
    win10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    win20: RollingWindow = field(default_factory=lambda: RollingWindow(20))

    blowout_win10: RollingWindow = field(
        default_factory=lambda: RollingWindow(10)
    )
    blowout_loss10: RollingWindow = field(
        default_factory=lambda: RollingWindow(10)
    )
    close10: RollingWindow = field(default_factory=lambda: RollingWindow(10))

    resid10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    resid20: RollingWindow = field(default_factory=lambda: RollingWindow(20))
    resid_pos10: RollingWindow = field(
        default_factory=lambda: RollingWindow(10)
    )

    fav_win10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    fav_diff10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    dog_win10: RollingWindow = field(default_factory=lambda: RollingWindow(10))
    dog_diff10: RollingWindow = field(default_factory=lambda: RollingWindow(10))

    ewm_diff: float = 0.0
    ewm_initialized: bool = False


def recent_state_features(
    st: PlayerSportRecentState,
    dt: Optional[pd.Timestamp],
) -> Dict[str, float]:
    if st.last_dt is None or dt is None or pd.isna(dt):
        days_since_last_match = -1.0
    else:
        days_since_last_match = float((dt - st.last_dt).days)

    return {
        "matches_played": float(st.n_matches),
        "days_since_last_match": days_since_last_match,
        "recent_n_5": float(st.diff5.count()),
        "recent_n_10": float(st.diff10.count()),
        "recent_n_20": float(st.diff20.count()),
        "diff_mean_5": st.diff5.mean(),
        "diff_mean_10": st.diff10.mean(),
        "diff_mean_20": st.diff20.mean(),
        "diff_std_10": st.diff10.std(),
        "diff_std_20": st.diff20.std(),
        "total_mean_5": st.total5.mean(),
        "total_mean_10": st.total10.mean(),
        "total_std_10": st.total10.std(),
        "winrate_5": st.win5.mean(),
        "winrate_10": st.win10.mean(),
        "winrate_20": st.win20.mean(),
        "blowout_win_rate_10": st.blowout_win10.mean(),
        "blowout_loss_rate_10": st.blowout_loss10.mean(),
        "close_match_rate_10": st.close10.mean(),
        "resid_mean_10": st.resid10.mean(),
        "resid_mean_20": st.resid20.mean(),
        "resid_std_10": st.resid10.std(),
        "resid_pos_rate_10": st.resid_pos10.mean(),
        "ewm_diff": float(st.ewm_diff if st.ewm_initialized else 0.0),
        "momentum_diff_5_20": st.diff5.mean() - st.diff20.mean(),
        "momentum_win_5_20": st.win5.mean() - st.win20.mean(),
        "fav_count_10": float(st.fav_diff10.count()),
        "fav_winrate_10": st.fav_win10.mean(),
        "fav_diff_mean_10": st.fav_diff10.mean(),
        "dog_count_10": float(st.dog_diff10.count()),
        "dog_winrate_10": st.dog_win10.mean(),
        "dog_diff_mean_10": st.dog_diff10.mean(),
    }


def update_recent_state(
    st: PlayerSportRecentState,
    actual_diff_player_pov: float,
    actual_total: float,
    expected_diff_player_pov: float,
    dt: pd.Timestamp,
    ewm_alpha: float = EWM_ALPHA,
) -> None:
    actual_diff_player_pov = float(actual_diff_player_pov)
    actual_total = float(actual_total)
    expected_diff_player_pov = float(expected_diff_player_pov)

    residual = actual_diff_player_pov - expected_diff_player_pov
    win = 1.0 if actual_diff_player_pov > 0 else 0.0
    blowout_win = (
        1.0 if actual_diff_player_pov >= BLOWOUT_WIN_THRESHOLD else 0.0
    )
    blowout_loss = (
        1.0 if actual_diff_player_pov <= BLOWOUT_LOSS_THRESHOLD else 0.0
    )
    close_match = (
        1.0 if abs(actual_diff_player_pov) <= CLOSE_MATCH_ABS_THRESHOLD else 0.0
    )
    resid_pos = 1.0 if residual > 0 else 0.0

    st.diff5.push(actual_diff_player_pov)
    st.diff10.push(actual_diff_player_pov)
    st.diff20.push(actual_diff_player_pov)

    st.total5.push(actual_total)
    st.total10.push(actual_total)

    st.win5.push(win)
    st.win10.push(win)
    st.win20.push(win)

    st.blowout_win10.push(blowout_win)
    st.blowout_loss10.push(blowout_loss)
    st.close10.push(close_match)

    st.resid10.push(residual)
    st.resid20.push(residual)
    st.resid_pos10.push(resid_pos)

    if expected_diff_player_pov >= FAVORITE_EXPECTED_DIFF_THRESHOLD:
        st.fav_win10.push(win)
        st.fav_diff10.push(actual_diff_player_pov)
    elif expected_diff_player_pov <= -FAVORITE_EXPECTED_DIFF_THRESHOLD:
        st.dog_win10.push(win)
        st.dog_diff10.push(actual_diff_player_pov)

    if not st.ewm_initialized:
        st.ewm_diff = actual_diff_player_pov
        st.ewm_initialized = True
    else:
        st.ewm_diff = (
            ewm_alpha * actual_diff_player_pov + (1.0 - ewm_alpha) * st.ewm_diff
        )

    st.n_matches += 1
    st.last_dt = dt


# =================================================
# Long-term shrunk state
# =================================================
@dataclass
class PlayerSportLongTermState:
    n_matches: int = 0
    diff_sum: float = 0.0
    total_sum: float = 0.0
    wins_sum: float = 0.0

    def features(self, shrink: float = LONG_TERM_SHRINK) -> Dict[str, float]:
        denom = self.n_matches + shrink
        return {
            "long_n": float(self.n_matches),
            "long_diff_mean": float(self.diff_sum / denom),
            "long_total_mean": float(self.total_sum / denom),
            "long_winrate": float(self.wins_sum / denom),
        }

    def update(
        self, actual_diff_player_pov: float, actual_total: float
    ) -> None:
        self.n_matches += 1
        self.diff_sum += float(actual_diff_player_pov)
        self.total_sum += float(actual_total)
        self.wins_sum += 1.0 if actual_diff_player_pov > 0 else 0.0


# =================================================
# H2H state
# =================================================
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
    st: H2HStats,
    a_is_p1: bool,
    dt: Optional[pd.Timestamp],
    prefix: str,
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

    if st.last_dt is None or dt is None or pd.isna(dt):
        out[f"{prefix}h2h_days_since_last"] = np.nan
    else:
        out[f"{prefix}h2h_days_since_last"] = float((dt - st.last_dt).days)

    return out


# =================================================
# One-rating-per-sport state
# =================================================
class DecayedUpdateMarginElo:
    def __init__(self):
        self.R = defaultdict(lambda: {s: BASE for s in SPORTS})
        self.games = defaultdict(lambda: {s: 0 for s in SPORTS})
        self.last_dt = defaultdict(lambda: {s: None for s in SPORTS})

    def predict(self, p1: str, p2: str) -> Dict[str, float]:
        pred = {}
        for s in SPORTS:
            x = self.R[p1][s] - self.R[p2][s]
            pred[s] = rating_to_score_diff(x, ALPHAS[s])
        return pred

    def time_mult(self, player: str, sport: str, dt: pd.Timestamp) -> float:
        gap = days_since(self.last_dt[player][sport], dt)
        return time_multiplier(sport, gap)

    def update(
        self,
        p1: str,
        p2: str,
        diffs: Dict[str, Optional[float]],
        dt: pd.Timestamp,
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

            tm1 = self.time_mult(p1, s, dt)
            tm2 = self.time_mult(p2, s, dt)
            time_mult_avg = 0.5 * (tm1 + tm2)

            grad = d_pred_dx(x, ALPHAS[s])
            margin_mult = 1.0 + MARGIN_MULT_BY_SPORT[s] * (abs(d) / MAX_DIFF)

            step = eta * time_mult_avg * margin_mult * err * grad
            step -= eta * L2 * x

            self.R[p1][s] = r1 + step
            self.R[p2][s] = r2 - step

            self.games[p1][s] += 1
            self.games[p2][s] += 1
            self.last_dt[p1][s] = dt
            self.last_dt[p2][s] = dt


# =================================================
# Final inference-state export
# =================================================
def build_final_player_state_record(
    player_name_norm: str,
    last_dt: Optional[pd.Timestamp],
    elo: DecayedUpdateMarginElo,
    recent_state: Dict[str, Dict[str, PlayerSportRecentState]],
    long_term_state: Dict[str, Dict[str, PlayerSportLongTermState]],
) -> dict:
    rec = {
        "player_name": player_name_norm,
        "player_key": player_name_norm,
        "last_datetime": last_dt,
    }

    total_pred = 0.0

    for s in SPORTS:
        rating = float(elo.R[player_name_norm][s])
        games = float(elo.games[player_name_norm][s])

        rec[f"{s}_rating_p1"] = rating
        rec[f"{s}_rating_p2"] = 0.0
        rec[f"{s}_rating_diff"] = rating
        rec[f"{s}_pred_diff"] = rating_to_score_diff(rating, ALPHAS[s])
        rec[f"{s}_games_p1"] = games
        rec[f"{s}_games_p2"] = 0.0
        rec[f"{s}_games_diff"] = games
        total_pred += rec[f"{s}_pred_diff"]

        rec[f"{s}_days_since_last_p1"] = 0.0
        rec[f"{s}_time_mult_p1"] = 1.0

        recent_feats = recent_state_features(
            recent_state[player_name_norm][s], last_dt
        )
        for k, v in recent_feats.items():
            rec[f"{s}_p1_recent_{k}"] = v

        long_feats = long_term_state[player_name_norm][s].features(
            shrink=LONG_TERM_SHRINK
        )
        for k, v in long_feats.items():
            rec[f"{s}_p1_{k}"] = v

    rec["rating_total_pred_diff"] = total_pred
    return rec


def build_inference_state(
    latest_player_dt_by_name: Dict[str, pd.Timestamp],
    elo: DecayedUpdateMarginElo,
    recent_state: Dict[str, Dict[str, PlayerSportRecentState]],
    long_term_state: Dict[str, Dict[str, PlayerSportLongTermState]],
    h2h_overall: Dict[Tuple[str, str], H2HStats],
    h2h_sport: Dict[str, Dict[Tuple[str, str], H2HStats]],
) -> dict:
    player_states_by_name = {}

    for player_name_norm in latest_player_dt_by_name.keys():
        last_dt = latest_player_dt_by_name.get(player_name_norm)
        player_states_by_name[player_name_norm] = (
            build_final_player_state_record(
                player_name_norm=player_name_norm,
                last_dt=last_dt,
                elo=elo,
                recent_state=recent_state,
                long_term_state=long_term_state,
            )
        )

    pair_h2h = {}
    for pk, st in h2h_overall.items():
        a_name = pk[0]
        b_name = pk[1]

        pair_h2h[pk] = {
            "overall": h2h_features(st, a_is_p1=True, dt=None, prefix=""),
            "sports": {
                s: h2h_features(
                    h2h_sport[s][pk],
                    a_is_p1=True,
                    dt=None,
                    prefix=f"{s}_",
                )
                for s in SPORTS
            },
            "a_name": a_name,
            "b_name": b_name,
            "last_dt": st.last_dt,
        }

    return {
        "player_states_by_name": player_states_by_name,
        "pair_h2h": pair_h2h,
    }


# =================================================
# Main builder
# =================================================
def build_training_data(
    in_csv: str = "data/matches_cleaned.csv",
    out_csv: str = "data/data.csv",
    out_inference_state: str = "data/inference_state.pkl",
) -> None:
    df = pd.read_csv(in_csv, low_memory=False)
    df["datetime"] = safe_datetime(df)
    df = df.sort_values("datetime").reset_index(drop=True)

    elo = DecayedUpdateMarginElo()

    h2h_overall: Dict[Tuple[str, str], H2HStats] = defaultdict(H2HStats)
    h2h_sport: Dict[str, Dict[Tuple[str, str], H2HStats]] = {
        s: defaultdict(H2HStats) for s in SPORTS
    }

    recent_state: Dict[str, Dict[str, PlayerSportRecentState]] = defaultdict(
        lambda: {s: PlayerSportRecentState() for s in SPORTS}
    )

    long_term_state: Dict[str, Dict[str, PlayerSportLongTermState]] = (
        defaultdict(lambda: {s: PlayerSportLongTermState() for s in SPORTS})
    )

    latest_player_dt_by_name: Dict[str, pd.Timestamp] = {}
    out_rows: List[dict] = []

    for i, row in df.iterrows():
        dt = row["datetime"]
        p1_key = player_key(row, 1)
        p2_key = player_key(row, 2)

        p1_name = player_name(row, 1)
        p2_name = player_name(row, 2)

        latest_player_dt_by_name[p1_name] = dt
        latest_player_dt_by_name[p2_name] = dt

        out = {
            "match_index": i,
            "datetime": dt,
            "month_key": get_month_key(dt),
            "p1_key": p1_key,
            "p2_key": p2_key,
            "p1_name": p1_name,
            "p2_name": p2_name,
        }

        pred = elo.predict(p1_key, p2_key)
        for s in SPORTS:
            out[f"{s}_rating_p1"] = elo.R[p1_key][s]
            out[f"{s}_rating_p2"] = elo.R[p2_key][s]
            out[f"{s}_rating_diff"] = elo.R[p1_key][s] - elo.R[p2_key][s]
            out[f"{s}_pred_diff"] = pred[s]
            out[f"{s}_games_p1"] = elo.games[p1_key][s]
            out[f"{s}_games_p2"] = elo.games[p2_key][s]
            out[f"{s}_games_diff"] = elo.games[p1_key][s] - elo.games[p2_key][s]

            out[f"{s}_days_since_last_p1"] = days_since(
                elo.last_dt[p1_key][s], dt
            )
            out[f"{s}_days_since_last_p2"] = days_since(
                elo.last_dt[p2_key][s], dt
            )
            out[f"{s}_time_mult_p1"] = elo.time_mult(p1_key, s, dt)
            out[f"{s}_time_mult_p2"] = elo.time_mult(p2_key, s, dt)
            out[f"{s}_time_mult_diff"] = (
                out[f"{s}_time_mult_p1"] - out[f"{s}_time_mult_p2"]
            )

        out["rating_total_pred_diff"] = sum(pred[s] for s in SPORTS)
        out["rating_winner_p1"] = 1 if out["rating_total_pred_diff"] > 0 else 0

        pk = pair_key(p1_key, p2_key)
        a = pk[0]
        a_is_p1 = a == p1_key

        out.update(h2h_features(h2h_overall[pk], a_is_p1, dt, prefix=""))
        for s in SPORTS:
            out.update(
                h2h_features(h2h_sport[s][pk], a_is_p1, dt, prefix=f"{s}_")
            )

        for s in SPORTS:
            p1_recent = recent_state[p1_key][s]
            p2_recent = recent_state[p2_key][s]

            p1_feats = recent_state_features(p1_recent, dt)
            p2_feats = recent_state_features(p2_recent, dt)

            out.update(add_prefixed(p1_feats, f"{s}_p1_recent_"))
            out.update(add_prefixed(p2_feats, f"{s}_p2_recent_"))
            add_pairwise_deltas(out, s, p1_feats, p2_feats)

        for s in SPORTS:
            p1_long = long_term_state[p1_key][s]
            p2_long = long_term_state[p2_key][s]

            p1_long_feats = p1_long.features(shrink=LONG_TERM_SHRINK)
            p2_long_feats = p2_long.features(shrink=LONG_TERM_SHRINK)

            out.update(add_prefixed(p1_long_feats, f"{s}_p1_"))
            out.update(add_prefixed(p2_long_feats, f"{s}_p2_"))
            add_pairwise_deltas(out, s, p1_long_feats, p2_long_feats)

        diffs = {s: get_sport_diff(row, s) for s in SPORTS}
        totals = {s: get_sport_total(row, s) for s in SPORTS}

        total_diff = 0.0
        any_sport = False

        for s in SPORTS:
            d = diffs[s]
            t = totals[s]

            out[f"has_{s}"] = 1 if d is not None else 0

            if d is not None and t is not None:
                d = clip_diff(d)
                out[f"{s}_y_diff"] = d
                out[f"{s}_y_total"] = float(t)
                total_diff += d
                any_sport = True
            else:
                out[f"{s}_y_diff"] = np.nan
                out[f"{s}_y_total"] = np.nan

        out["y_total_diff"] = total_diff if any_sport else np.nan
        out["y_winner_p1"] = (
            (1 if total_diff > 0 else 0) if any_sport else np.nan
        )

        out_rows.append(out)

        # Update state AFTER row
        elo.update(p1_key, p2_key, diffs, dt)

        total_diff_for_h2h = 0.0
        any_for_h2h = False
        for s in SPORTS:
            d = diffs[s]
            t = totals[s]
            if d is None or t is None:
                continue

            d = clip_diff(d)
            any_for_h2h = True
            total_diff_for_h2h += d

            h2h_update(h2h_sport[s][pk], a_is_p1, d, dt)

            update_recent_state(
                recent_state[p1_key][s],
                actual_diff_player_pov=d,
                actual_total=t,
                expected_diff_player_pov=pred[s],
                dt=dt,
            )
            update_recent_state(
                recent_state[p2_key][s],
                actual_diff_player_pov=-d,
                actual_total=t,
                expected_diff_player_pov=-pred[s],
                dt=dt,
            )

            long_term_state[p1_key][s].update(
                actual_diff_player_pov=d,
                actual_total=t,
            )
            long_term_state[p2_key][s].update(
                actual_diff_player_pov=-d,
                actual_total=t,
            )

        if any_for_h2h:
            h2h_update(h2h_overall[pk], a_is_p1, total_diff_for_h2h, dt)

    out_df = pd.DataFrame(out_rows)

    id_cols = [
        "match_index",
        "datetime",
        "month_key",
        "p1_key",
        "p2_key",
        "p1_name",
        "p2_name",
    ]

    keep_feature_cols = []

    for s in SPORTS:
        keep_feature_cols += [
            # core rating features
            f"{s}_rating_p1",
            f"{s}_rating_p2",
            f"{s}_rating_diff",
            f"{s}_games_p1",
            f"{s}_games_p2",
            f"{s}_games_diff",
            f"{s}_days_since_last_p1",
            f"{s}_days_since_last_p2",
            f"{s}_time_mult_p1",
            f"{s}_time_mult_p2",
            f"{s}_time_mult_diff",
            # overall H2H in this sport
            f"{s}_h2h_games",
            f"{s}_h2h_avg_diff_p1",
            f"{s}_h2h_winrate_p1",
            f"{s}_h2h_days_since_last",
            # recent-form features actually used
            f"{s}_p1_recent_diff_mean_10",
            f"{s}_p2_recent_diff_mean_10",
            f"{s}_diff_mean_10_diff_p1_p2",
            f"{s}_p1_recent_resid_mean_10",
            f"{s}_p2_recent_resid_mean_10",
            f"{s}_resid_mean_10_diff_p1_p2",
            f"{s}_p1_recent_diff_std_10",
            f"{s}_p2_recent_diff_std_10",
            f"{s}_diff_std_10_diff_p1_p2",
            f"{s}_p1_recent_momentum_diff_5_20",
            f"{s}_p2_recent_momentum_diff_5_20",
            f"{s}_momentum_diff_5_20_diff_p1_p2",
            # long-term shrunk features
            f"{s}_p1_long_n",
            f"{s}_p2_long_n",
            f"{s}_long_n_diff_p1_p2",
            f"{s}_p1_long_diff_mean",
            f"{s}_p2_long_diff_mean",
            f"{s}_long_diff_mean_diff_p1_p2",
            f"{s}_p1_long_total_mean",
            f"{s}_p2_long_total_mean",
            f"{s}_long_total_mean_diff_p1_p2",
            f"{s}_p1_long_winrate",
            f"{s}_p2_long_winrate",
            f"{s}_long_winrate_diff_p1_p2",
        ]

    keep_feature_cols += [
        # overall match H2H
        "h2h_games",
        "h2h_avg_diff_p1",
        "h2h_winrate_p1",
        "h2h_days_since_last",
    ]

    target_cols = [
        c
        for c in out_df.columns
        if c.endswith("_y_diff")
        or c.endswith("_y_total")
        or c in ("y_total_diff", "y_winner_p1")
    ]

    ordered_cols = []
    for c in id_cols + keep_feature_cols + sorted(target_cols):
        if c in out_df.columns and c not in ordered_cols:
            ordered_cols.append(c)

    out_df = out_df[ordered_cols]
    out_df.to_csv(out_csv, index=False)

    inference_state = build_inference_state(
        latest_player_dt_by_name=latest_player_dt_by_name,
        elo=elo,
        recent_state=recent_state,
        long_term_state=long_term_state,
        h2h_overall=h2h_overall,
        h2h_sport=h2h_sport,
    )

    with open(out_inference_state, "wb") as f:
        pickle.dump(inference_state, f)

    print(
        f"Saved {out_csv} with shape {out_df.shape} | "
        f"Saved inference state to {out_inference_state} | "
        f"USE_IDS={USE_IDS}"
    )


if __name__ == "__main__":
    build_training_data()
