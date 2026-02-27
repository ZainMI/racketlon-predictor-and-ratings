# Leakage-free feature engineering for Racketlon:
# - Per-sport Elo (with optional margin scaling)
# - Per-sport experience + confidence
# - Per-sport rolling recent margin/total
# - Per-sport H2H (head-to-head) features with shrinkage
# - Rest features (days since last match overall + per sport)
# Saves: features_engineered.csv

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional, Any, List

SPORTS = ["TT", "BD", "SQ", "TN"]

# -----------------------------
# Elo hyperparameters
# -----------------------------
BASE_RATING = 1500.0
ELO_SCALE = 400.0
K = 24.0

# Margin scaling (optional)
USE_MARGIN = True
MAX_DIFF_FOR_MOV = 21.0  # cap the effect (safe even if deuce exists)

# Recency window
RECENT_N = 5

# Confidence from games played: conf = 1 - exp(-games/tau)
CONF_TAU = 15.0

# H2H shrinkage priors
H2H_WIN_PRIOR_GAMES = 6.0  # larger => shrink winrate toward 0.5 more
H2H_DIFF_PRIOR_GAMES = 6.0  # larger => shrink avg diff toward 0 more


def ordered_pair(a: str, b: str) -> Tuple[str, str]:
    """Canonicalize a pair key so (A,B)==(B,A) with deterministic ordering."""
    return (a, b) if a <= b else (b, a)


def expected_score(r_a: float, r_b: float, scale: float = ELO_SCALE) -> float:
    """Expected probability A beats B."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))


def mov_multiplier(diff: float) -> float:
    """
    Margin-of-victory multiplier using bounded diff.
    Produces a smooth multiplier in [1,2] based on |diff| capped at MAX_DIFF_FOR_MOV.
    """
    d = min(abs(float(diff)), MAX_DIFF_FOR_MOV)
    return 1.0 + (d / MAX_DIFF_FOR_MOV)


def safe_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses match_date (YYYYMMDD) and combines with match_time when possible.
    Sorts chronologically and returns a new DataFrame.
    """
    df = df.copy()

    if "match_date" not in df.columns:
        raise ValueError("Expected a 'match_date' column (YYYYMMDD).")

    df["match_date"] = pd.to_datetime(
        df["match_date"], format="%Y%m%d", errors="coerce"
    )

    if "match_time" in df.columns:
        dt = pd.to_datetime(
            df["match_date"].astype(str) + " " + df["match_time"].astype(str),
            errors="coerce",
        )
        df["datetime"] = dt
        df["datetime"] = df["datetime"].fillna(df["match_date"])
    else:
        df["datetime"] = df["match_date"]

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def main(
    input_csv: str = "data/matches_cleaned.csv",
    output_csv: str = "features_engineered.csv",
) -> None:
    df = pd.read_csv(input_csv)
    df = safe_datetime(df)

    # -----------------------------
    # State
    # -----------------------------

    # Elo ratings and games played per (player, sport)
    elo: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: BASE_RATING)
    )
    games: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Recent per (player, sport)
    recent_diffs: Dict[str, Dict[str, Deque[float]]] = defaultdict(
        lambda: defaultdict(lambda: deque(maxlen=RECENT_N))
    )
    recent_totals: Dict[str, Dict[str, Deque[float]]] = defaultdict(
        lambda: defaultdict(lambda: deque(maxlen=RECENT_N))
    )

    # Last played times (overall and per sport)
    last_played_any: Dict[str, pd.Timestamp] = {}
    last_played_sport: Dict[str, Dict[str, pd.Timestamp]] = defaultdict(dict)

    # --- H2H state: tracked per unordered pair, per sport ---
    # Stored from the "pair[0]" player's perspective for consistency.
    h2h_games = defaultdict(
        lambda: defaultdict(int)
    )  # count of prior meetings in that sport
    h2h_wins_first = defaultdict(lambda: defaultdict(float))  # wins for pair[0]
    h2h_diff_sum_first = defaultdict(
        lambda: defaultdict(float)
    )  # sum of (pair[0] score - pair[1] score)
    h2h_last_dt = defaultdict(
        lambda: defaultdict(lambda: None)
    )  # last meeting datetime per sport

    out_rows: List[dict] = []

    # -----------------------------
    # Chronological processing loop
    # -----------------------------
    for i, row in df.iterrows():
        p1 = str(row["team1_players"])
        p2 = str(row["team2_players"])
        dt: pd.Timestamp = row["datetime"]

        pair = ordered_pair(p1, p2)
        p1_is_first = pair[0] == p1

        # ---- Base metadata passthrough (helpful for later joins/debug) ----
        out: Dict[str, Any] = {
            "match_index": i,
            "tournament_id": row.get("tournament_id"),
            "datetime": dt,
            "match_date": row.get("match_date"),
            "match_time": row.get("match_time"),
            "draw": row.get("draw"),
            "draw_id": row.get("draw_id"),
            "round": row.get("round"),
            "location": row.get("location"),
            "team1_players": p1,
            "team2_players": p2,
            "team1_player_ids": row.get("team1_player_ids"),
            "team2_player_ids": row.get("team2_player_ids"),
            "team1_nationalities": row.get("team1_nationalities"),
            "team2_nationalities": row.get("team2_nationalities"),
            "team1_club_ids": row.get("team1_club_ids"),
            "team2_club_ids": row.get("team2_club_ids"),
            "winner_side": row.get("winner_side"),
            "status_message": row.get("status_message"),
            "mode": row.get("mode"),
        }

        # ---- Global rest features (overall) ----
        def days_since(last_map: Dict[str, pd.Timestamp], player: str) -> float:
            last = last_map.get(player)
            if last is None or pd.isna(dt) or pd.isna(last):
                return np.nan
            return float((dt - last).days)

        out["days_since_last_match_p1"] = days_since(last_played_any, p1)
        out["days_since_last_match_p2"] = days_since(last_played_any, p2)

        # Flags
        out["same_nationality"] = float(
            row.get("team1_nationalities") == row.get("team2_nationalities")
        )
        out["same_club"] = float(
            row.get("team1_club_ids") == row.get("team2_club_ids")
        )

        # ---- Per-sport pre-match features (Elo, experience, recency, H2H) ----
        for s in SPORTS:
            # Elo + win prob
            r1 = float(elo[p1][s])
            r2 = float(elo[p2][s])
            pwin = expected_score(r1, r2)

            out[f"{s}_elo_p1"] = r1
            out[f"{s}_elo_p2"] = r2
            out[f"{s}_elo_diff"] = r1 - r2
            out[f"{s}_pwin_p1"] = float(pwin)

            # Experience + confidence
            g1 = int(games[p1][s])
            g2 = int(games[p2][s])
            out[f"{s}_games_p1"] = g1
            out[f"{s}_games_p2"] = g2
            out[f"{s}_games_diff"] = g1 - g2
            out[f"{s}_conf_p1"] = float(1.0 - np.exp(-g1 / CONF_TAU))
            out[f"{s}_conf_p2"] = float(1.0 - np.exp(-g2 / CONF_TAU))

            # Recency stats (last N)
            rd1 = recent_diffs[p1][s]
            rd2 = recent_diffs[p2][s]
            rt1 = recent_totals[p1][s]
            rt2 = recent_totals[p2][s]

            out[f"{s}_recent_margin_p1"] = (
                float(np.mean(rd1)) if len(rd1) else 0.0
            )
            out[f"{s}_recent_margin_p2"] = (
                float(np.mean(rd2)) if len(rd2) else 0.0
            )
            out[f"{s}_recent_total_p1"] = (
                float(np.mean(rt1)) if len(rt1) else 0.0
            )
            out[f"{s}_recent_total_p2"] = (
                float(np.mean(rt2)) if len(rt2) else 0.0
            )

            # Sport-specific rest (days since last played that sport)
            last_s1 = last_played_sport[p1].get(s)
            last_s2 = last_played_sport[p2].get(s)
            out[f"{s}_days_since_last_p1"] = (
                float((dt - last_s1).days)
                if last_s1 is not None and not pd.isna(dt)
                else np.nan
            )
            out[f"{s}_days_since_last_p2"] = (
                float((dt - last_s2).days)
                if last_s2 is not None and not pd.isna(dt)
                else np.nan
            )

            # Data availability flag for this row (label availability, not truly "pre-match")
            s1_val = row.get(f"{s}_p1")
            s2_val = row.get(f"{s}_p2")
            out[f"has_{s}"] = float(not (pd.isna(s1_val) or pd.isna(s2_val)))

            # ---- H2H features (pre-match only) ----
            g = int(h2h_games[pair][s])
            wins_first = float(h2h_wins_first[pair][s])
            diff_sum_first = float(h2h_diff_sum_first[pair][s])

            # Convert stored first-player perspective -> p1 perspective
            if p1_is_first:
                wins_p1 = wins_first
                diff_sum_p1 = diff_sum_first
            else:
                wins_p1 = g - wins_first
                diff_sum_p1 = -diff_sum_first

            # Shrunk winrate toward 0.5
            denom_win = g + H2H_WIN_PRIOR_GAMES
            winrate_p1 = (
                (wins_p1 + 0.5 * H2H_WIN_PRIOR_GAMES) / denom_win
                if denom_win > 0
                else 0.5
            )

            # Shrunk avg diff toward 0
            denom_diff = g + H2H_DIFF_PRIOR_GAMES
            avg_diff_p1 = diff_sum_p1 / denom_diff if denom_diff > 0 else 0.0

            out[f"{s}_h2h_games"] = g
            out[f"{s}_h2h_winrate_p1"] = float(winrate_p1)
            out[f"{s}_h2h_avg_diff_p1"] = float(avg_diff_p1)

            last_h2h = h2h_last_dt[pair][s]
            out[f"{s}_h2h_days_since_last"] = (
                float((dt - last_h2h).days)
                if last_h2h is not None and not pd.isna(dt)
                else np.nan
            )

        # -----------------------------
        # Labels/targets (optional but useful)
        # -----------------------------
        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")
            if pd.isna(s1) or pd.isna(s2):
                out[f"{s}_y_diff"] = np.nan
                out[f"{s}_y_total"] = np.nan
                out[f"{s}_y_win_p1"] = np.nan
            else:
                s1f = float(s1)
                s2f = float(s2)
                out[f"{s}_y_diff"] = s1f - s2f
                out[f"{s}_y_total"] = s1f + s2f
                out[f"{s}_y_win_p1"] = float(s1f > s2f)

        # Match-level totals across recorded sports
        true_total_p1 = 0.0
        true_total_p2 = 0.0
        any_sport = False
        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue
            any_sport = True
            true_total_p1 += float(s1)
            true_total_p2 += float(s2)
        out["y_total_p1"] = true_total_p1 if any_sport else np.nan
        out["y_total_p2"] = true_total_p2 if any_sport else np.nan
        out["y_winner_p1"] = (
            float(true_total_p1 > true_total_p2) if any_sport else np.nan
        )

        out_rows.append(out)

        # -----------------------------
        # Post-match updates (Elo, recency, rest, H2H)
        # -----------------------------
        if not pd.isna(dt):
            last_played_any[p1] = dt
            last_played_any[p2] = dt

        for s in SPORTS:
            s1 = row.get(f"{s}_p1")
            s2 = row.get(f"{s}_p2")
            if pd.isna(s1) or pd.isna(s2):
                continue

            s1f = float(s1)
            s2f = float(s2)
            diff = s1f - s2f
            total = s1f + s2f

            # Outcome from p1 perspective
            if diff > 0:
                a = 1.0
            elif diff < 0:
                a = 0.0
            else:
                a = 0.5

            # Elo update
            r1 = float(elo[p1][s])
            r2 = float(elo[p2][s])
            e1 = expected_score(r1, r2)
            mult = mov_multiplier(diff) if USE_MARGIN else 1.0
            delta = K * mult * (a - e1)

            elo[p1][s] = r1 + delta
            elo[p2][s] = r2 - delta

            # Games + recency
            games[p1][s] += 1
            games[p2][s] += 1

            recent_diffs[p1][s].append(diff)
            recent_diffs[p2][s].append(-diff)
            recent_totals[p1][s].append(total)
            recent_totals[p2][s].append(total)

            if not pd.isna(dt):
                last_played_sport[p1][s] = dt
                last_played_sport[p2][s] = dt

            # H2H update (store from pair[0] perspective)
            pair = ordered_pair(p1, p2)
            p1_is_first = pair[0] == p1

            h2h_games[pair][s] += 1
            if p1_is_first:
                # pair[0] == p1
                h2h_wins_first[pair][s] += a
                h2h_diff_sum_first[pair][s] += diff
            else:
                # pair[0] == p2, from pair[0] perspective:
                h2h_wins_first[pair][s] += 1.0 - a
                h2h_diff_sum_first[pair][s] += -diff

            if not pd.isna(dt):
                h2h_last_dt[pair][s] = dt

    feats = pd.DataFrame(out_rows)

    # Make timestamps CSV-friendly
    feats["datetime"] = pd.to_datetime(feats["datetime"], errors="coerce")
    feats["match_date"] = pd.to_datetime(feats["match_date"], errors="coerce")

    feats.to_csv(output_csv, index=False)
    print(f"Saved {len(feats)} rows to {output_csv}")


if __name__ == "__main__":
    main()
