# features_v4.py
#
# Builds a SINGLE training-ready dataset with:
# - leakage-safe pre-match features (margin-Elo + H2H up to that point)
# - monthly snapshot rating features computed directly from raw matches
# - explicit per-player current ratings for synthetic future matchup construction
# - targets:
#     * per-sport diffs      -> {SPORT}_y_diff
#     * per-sport totals     -> {SPORT}_y_total
#     * match total diff     -> y_total_diff
#     * match winner         -> y_winner_p1
#
# Output: data/data.csv

import math
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SPORTS = ["TT", "BD", "SQ", "TN"]
SPORTS_LOWER = ["tt", "bd", "sq", "tn"]
SPORT_UPPER_TO_LOWER = {"TT": "tt", "BD": "bd", "SQ": "sq", "TN": "tn"}

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


def player_key(row: pd.Series, side: int) -> str:
    if USE_IDS:
        col = f"team{side}_player_ids"
        if col in row and not pd.isna(row[col]):
            return str(row[col])
    return str(row.get(f"team{side}_players", "")).strip().lower()


def player_name(row: pd.Series, side: int) -> str:
    return str(row.get(f"team{side}_players", "")).strip().lower()


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


def previous_month(year: int, month: int) -> Tuple[int, int]:
    month -= 1
    if month == 0:
        return year - 1, 12
    return year, month


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


# =================================================
# Old monthly-ratings logic, adapted to raw CSV
# =================================================
def row_to_old_match_format(row: pd.Series) -> dict:
    return {
        "p1": player_name(row, 1),
        "p2": player_name(row, 2),
        "tt_p1": row.get("TT_p1"),
        "tt_p2": row.get("TT_p2"),
        "bd_p1": row.get("BD_p1"),
        "bd_p2": row.get("BD_p2"),
        "sq_p1": row.get("SQ_p1"),
        "sq_p2": row.get("SQ_p2"),
        "tn_p1": row.get("TN_p1"),
        "tn_p2": row.get("TN_p2"),
        "date": (
            row["datetime"].to_pydatetime()
            if pd.notna(row["datetime"])
            else None
        ),
    }


def compute_history(player, date, history, within):
    matches = history.get(player, [])

    tt = bd = sq = tn = 0.0
    tennis_null = 0
    used = 0

    for row in matches:
        if row["date"] is None or date is None:
            continue
        if row["date"] > date:
            continue

        max_history = pd.Timedelta(days=within * 365)
        if pd.Timestamp(date) - pd.Timestamp(row["date"]) > max_history:
            continue

        slot = "p1" if row["p1"] == player else "p2"
        op_slot = "p2" if slot == "p1" else "p1"

        tt += float(row[f"tt_{slot}"] or 0) - float(row[f"tt_{op_slot}"] or 0)
        bd += float(row[f"bd_{slot}"] or 0) - float(row[f"bd_{op_slot}"] or 0)
        sq += float(row[f"sq_{slot}"] or 0) - float(row[f"sq_{op_slot}"] or 0)

        if (
            row[f"tn_{slot}"] is None
            or row[f"tn_{op_slot}"] is None
            or pd.isna(row[f"tn_{slot}"])
            or pd.isna(row[f"tn_{op_slot}"])
        ):
            tennis_null += 1
        else:
            tn += float(row[f"tn_{slot}"]) - float(row[f"tn_{op_slot}"])

        used += 1

    tn_count = used - tennis_null
    if used == 0:
        return {"tt": 0.0, "bd": 0.0, "sq": 0.0, "tn": 0.0}

    return {
        "tt": tt / used,
        "bd": bd / used,
        "sq": sq / used,
        "tn": tn / (tn_count + 1e-13),
    }


def calculate_weight(match_date, ref_date, half_life=365):
    if match_date is None or ref_date is None:
        return 0.0
    delta_days = (ref_date - match_date).days
    if delta_days < 0:
        return 0.0
    decay_rate = np.log(2) / half_life
    return np.exp(-decay_rate * delta_days)


def old_get_diff_by_sport(sport, row):
    p1_score = row.get(f"{sport}_p1")
    p2_score = row.get(f"{sport}_p2")
    if (
        p1_score is None
        or p2_score is None
        or pd.isna(p1_score)
        or pd.isna(p2_score)
    ):
        return 0.0
    return float(p1_score) - float(p2_score)


def get_rating_diff(r1, r2):
    return {
        sport: float(r1[sport]) - float(r2[sport]) for sport in SPORTS_LOWER
    }


def get_actual_diff_old(row):
    return {
        "tt": old_get_diff_by_sport("tt", row),
        "bd": old_get_diff_by_sport("bd", row),
        "sq": old_get_diff_by_sport("sq", row),
        "tn": old_get_diff_by_sport("tn", row),
    }


def compute_error(actual_diff, expected_diff):
    return {
        sport: actual_diff[sport] - expected_diff[sport]
        for sport in actual_diff
    }


def get_eta(match_count, base_eta=1.0):
    scale = base_eta / np.sqrt(1 + match_count)
    return {"tt": scale, "bd": scale, "sq": scale, "tn": scale}


def update_all_ratings(train_data, ratings, history, ref_date, base_eta=1.0):
    match_counts = defaultdict(int)

    for row in train_data:
        p1 = row["p1"]
        p2 = row["p2"]

        for player in [p1, p2]:
            if player not in ratings:
                his = compute_history(player, ref_date, history, within=100)
                ratings[player] = {
                    "tt": his["tt"],
                    "bd": his["bd"],
                    "sq": his["sq"],
                    "tn": his["tn"],
                }

        r1 = ratings[p1]
        r2 = ratings[p2]

        actual_diff = get_actual_diff_old(row)
        expected_diff = get_rating_diff(r1, r2)
        error = compute_error(actual_diff, expected_diff)

        weight = calculate_weight(row["date"], ref_date)

        eta_p1 = get_eta(match_counts[p1], base_eta)
        eta_p2 = get_eta(match_counts[p2], base_eta)

        movement_p1 = {
            sport: weight * eta_p1[sport] * error[sport] for sport in error
        }
        movement_p2 = {
            sport: -weight * eta_p2[sport] * error[sport] for sport in error
        }

        ratings[p1] = {sport: r1[sport] + movement_p1[sport] for sport in r1}
        ratings[p2] = {sport: r2[sport] + movement_p2[sport] for sport in r2}

        match_counts[p1] += 1
        match_counts[p2] += 1


def loss_fn(alpha, x, y):
    pred = rating_to_score_diff_vec(x, alpha)
    return np.mean((pred - y) ** 2)


def rating_to_score_diff_vec(x, alpha):
    return 21 * (1 - np.exp(-alpha * x)) / (1 + np.exp(-alpha * x))


def best_alpha(train_data, ratings):
    x = {"tt": [], "bd": [], "sq": [], "tn": []}
    y = {"tt": [], "bd": [], "sq": [], "tn": []}

    for row in train_data:
        p1 = row["p1"]
        p2 = row["p2"]
        if p1 not in ratings or p2 not in ratings:
            continue

        ex = get_rating_diff(ratings[p1], ratings[p2])
        ac = get_actual_diff_old(row)

        for sport in x:
            x[sport].append(ex[sport])
            y[sport].append(ac[sport])

    ret = {}
    for sport in x:
        if len(x[sport]) == 0:
            ret[sport] = 0.01
            continue

        res = minimize_scalar(
            loss_fn,
            bounds=(0.001, 2),
            args=(np.array(x[sport]), np.array(y[sport])),
            method="bounded",
        )
        ret[sport] = float(res.x)

    return ret


def build_ratings_by_month_from_matches(df: pd.DataFrame):
    all_matches = [row_to_old_match_format(r) for _, r in df.iterrows()]
    all_matches = sorted(all_matches, key=lambda x: x["date"])

    history = defaultdict(list)
    for row in all_matches:
        history[row["p1"]].append(row)
        history[row["p2"]].append(row)

    ratings_by_month = defaultdict(
        lambda: {"tt": 0.0, "bd": 0.0, "sq": 0.0, "tn": 0.0}
    )
    monthly_matches = defaultdict(list)

    for row in all_matches:
        year_month = (row["date"].year, row["date"].month)
        monthly_matches[year_month].append(row)

    sorted_months = sorted(monthly_matches.keys())
    grouped = [(ym, monthly_matches[ym]) for ym in sorted_months]

    results = {}

    for i, (ym, rows) in enumerate(grouped):
        year, month = ym
        key = f"{year:04d}-{month:02d}"

        if month == 12:
            ref_date = pd.Timestamp(year + 1, 1, 1).to_pydatetime()
        else:
            ref_date = pd.Timestamp(year, month + 1, 1).to_pydatetime()

        for j in range(1, 10):
            update_all_ratings(
                rows,
                ratings_by_month,
                history,
                ref_date=ref_date,
                base_eta=float(1 / j),
            )

        best_alphas = best_alpha(rows, ratings_by_month)

        results[key] = {
            "ratings": copy.deepcopy(dict(ratings_by_month)),
            "alphas": best_alphas,
        }

    return results


def get_last_month_snapshot(
    ratings_by_month: Dict[str, dict],
    dt: pd.Timestamp,
    include_current_month: bool = True,
    lower_year_bound: int = 2010,
) -> Tuple[Optional[str], Optional[dict]]:
    if pd.isna(dt):
        return None, None

    year = int(dt.year)
    month = int(dt.month)

    if not include_current_month:
        year, month = previous_month(year, month)

    while year >= lower_year_bound:
        key = f"{year:04d}-{month:02d}"
        if key in ratings_by_month:
            snap = ratings_by_month[key]
            if isinstance(snap, dict):
                return key, snap
        year, month = previous_month(year, month)

    return None, None


def get_snapshot_player_ratings(
    snapshot: Optional[dict], player_name_norm: str
) -> dict:
    if snapshot is None:
        return {}
    ratings_block = snapshot.get("ratings", {})
    rec = ratings_block.get(player_name_norm)
    if not isinstance(rec, dict):
        return {}
    return rec


def get_snapshot_rating_diff_fields(
    p1_name_norm: str,
    p2_name_norm: str,
    snapshot: Optional[dict],
) -> Dict[str, float]:
    p1 = get_snapshot_player_ratings(snapshot, p1_name_norm)
    p2 = get_snapshot_player_ratings(snapshot, p2_name_norm)

    alphas = snapshot.get("alphas", {}) if isinstance(snapshot, dict) else {}

    out: Dict[str, float] = {}
    p1_found = 1 if isinstance(p1, dict) and len(p1) > 0 else 0
    p2_found = 1 if isinstance(p2, dict) and len(p2) > 0 else 0

    for sport_upper in SPORTS:
        sport_lower = SPORT_UPPER_TO_LOWER[sport_upper]
        alpha = float(alphas.get(sport_lower, ALPHAS[sport_upper]))

        r1 = float(p1.get(sport_lower, 0.0))
        r2 = float(p2.get(sport_lower, 0.0))
        diff = r1 - r2

        out[f"{sport_upper}_snapshot_rating_p1"] = r1
        out[f"{sport_upper}_snapshot_rating_p2"] = r2
        out[f"{sport_upper}_snapshot_rating_diff"] = diff
        out[f"{sport_upper}_snapshot_pred_diff"] = rating_to_score_diff(
            diff, alpha
        )
        out[f"{sport_upper}_snapshot_p1_found"] = p1_found
        out[f"{sport_upper}_snapshot_p2_found"] = p2_found

    total_pred = sum(out[f"{s}_snapshot_pred_diff"] for s in SPORTS)
    out["snapshot_total_pred_diff"] = total_pred
    out["snapshot_winner_p1"] = 1 if total_pred > 0 else 0
    return out


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


# =================================================
# Margin-Elo state
# =================================================
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


# =================================================
# Main builder
# =================================================
def build_training_data(
    in_csv: str = "data/matches_cleaned.csv",
    out_csv: str = "data/data.csv",
    include_current_month_snapshot: bool = True,
) -> None:
    df = pd.read_csv(in_csv, low_memory=False)
    df["datetime"] = safe_datetime(df)
    df = df.sort_values("datetime").reset_index(drop=True)

    print("Building monthly snapshot ratings in memory...")
    ratings_by_month = build_ratings_by_month_from_matches(df)

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

        p1_name = player_name(row, 1)
        p2_name = player_name(row, 2)

        out = {
            "match_index": i,
            "datetime": dt,
            "month_key": get_month_key(dt),
            "p1_key": p1_key,
            "p2_key": p2_key,
            "p1_name": p1_name,
            "p2_name": p2_name,
        }

        # Current running margin-Elo features
        pred = elo.predict(p1_key, p2_key)
        for s in SPORTS:
            # NEW: explicit individual current ratings
            out[f"{s}_rating_p1"] = elo.R[p1_key][s]
            out[f"{s}_rating_p2"] = elo.R[p2_key][s]

            out[f"{s}_rating_diff"] = elo.R[p1_key][s] - elo.R[p2_key][s]
            out[f"{s}_pred_diff"] = pred[s]
            out[f"{s}_games_p1"] = elo.games[p1_key][s]
            out[f"{s}_games_p2"] = elo.games[p2_key][s]
            out[f"{s}_games_diff"] = elo.games[p1_key][s] - elo.games[p2_key][s]

        # Monthly snapshot features
        snapshot_key, snapshot = get_last_month_snapshot(
            ratings_by_month,
            dt,
            include_current_month=include_current_month_snapshot,
        )
        out["snapshot_key"] = snapshot_key
        out["snapshot_found"] = 1 if snapshot is not None else 0
        out.update(
            get_snapshot_rating_diff_fields(p1_name, p2_name, snapshot or {})
        )

        # H2H features
        pk = pair_key(p1_key, p2_key)
        a = pk[0]
        a_is_p1 = a == p1_key

        out.update(h2h_features(h2h_overall[pk], a_is_p1, dt, prefix=""))
        for s in SPORTS:
            out.update(
                h2h_features(h2h_sport[s][pk], a_is_p1, dt, prefix=f"{s}_")
            )

        # Targets
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

        # Update state after features
        elo.update(p1_key, p2_key, diffs)

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

    id_cols = [
        "match_index",
        "datetime",
        "month_key",
        "snapshot_key",
        "snapshot_found",
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
        and not c.endswith("_y_total")
        and not c.startswith("y_")
    ]

    target_cols = [
        c
        for c in out_df.columns
        if c.endswith("_y_diff")
        or c.endswith("_y_total")
        or c in ("y_total_diff", "y_winner_p1")
    ]

    out_df = out_df[id_cols + sorted(feature_cols) + sorted(target_cols)]
    out_df.to_csv(out_csv, index=False)

    print(
        f"Saved {out_csv} with shape {out_df.shape} "
        f"(USE_IDS={USE_IDS}, include_current_month_snapshot={include_current_month_snapshot})"
    )


if __name__ == "__main__":
    build_training_data()
