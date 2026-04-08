import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42
BASE = 0.0

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_PATH = "data/data.csv"
TRAIN_RATIO = 0.8

PREDICT_ONE_MATCH = True
PLAYER1 = "zain magdon-ismail"
PLAYER2 = "luke griffiths"

# If True, TN is decoded as a normal full-score sport.
# If False, TN is truncated by racketlon stop rules.
PREDICT_TENNIS_INDEPENDENTLY = True

CATBOOST_PARAMS = dict(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=False,
)


# -------------------------------------------------
# Load
# -------------------------------------------------
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["p1_name", "p2_name", "p1_key", "p2_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


# -------------------------------------------------
# Feature selection
# -------------------------------------------------
def get_feature_columns(df: pd.DataFrame):
    drop_exact = {
        "match_index",
        "datetime",
        "month_key",
        "snapshot_key",
        "p1_name",
        "p2_name",
        "p1_key",
        "p2_key",
        "y_total_diff",
        "y_winner_p1",
        "snapshot_winner_p1",
        "snapshot_total_pred_diff",
    }

    feature_cols = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if c.endswith("_y_diff"):
            continue
        if c.endswith("_y_total"):
            continue
        if c.startswith("has_"):
            continue
        if "_pred_diff" in c:
            continue
        feature_cols.append(c)

    return sorted(feature_cols)


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
# Training
# -------------------------------------------------
def make_model():
    return CatBoostRegressor(**CATBOOST_PARAMS)


def train_models(df: pd.DataFrame):
    split = int(len(df) * TRAIN_RATIO)
    feature_cols = get_feature_columns(df)

    print(f"Using {len(feature_cols)} features")
    print(
        "Dropped all *_pred_diff, snapshot_total_pred_diff, has_*, and target columns."
    )

    sport_models = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))

    for sport in SPORTS:
        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"

        if diff_col not in df.columns or total_col not in df.columns:
            print(f"Skipping {sport}: missing {diff_col} or {total_col}")
            continue

        mask = df[diff_col].notna() & df[total_col].notna()
        rows = np.where(mask.values)[0]
        if len(rows) < 20:
            print(f"Skipping {sport}: only {len(rows)} usable rows")
            continue

        X = df.loc[rows, feature_cols].copy()
        y_diff = df.loc[rows, diff_col].astype(float).values
        y_total = df.loc[rows, total_col].astype(float).values

        train_mask = rows < split
        test_mask = rows >= split

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"Skipping {sport}: empty train or test split")
            continue

        X_train = X.iloc[train_mask]
        X_test = X.iloc[test_mask]

        y_diff_train = y_diff[train_mask]
        y_diff_test = y_diff[test_mask]
        y_total_train = y_total[train_mask]
        y_total_test = y_total[test_mask]

        model_diff = make_model()
        model_total = make_model()

        model_diff.fit(X_train, y_diff_train)
        model_total.fit(X_train, y_total_train)

        pred_diff = np.clip(model_diff.predict(X_test), -21, 21)
        pred_total = np.clip(model_total.predict(X_test), 0, 42)

        test_rows = rows[test_mask]
        for i, row_idx in enumerate(test_rows):
            if sport in ["TT", "BD", "SQ"]:
                s1_hat, s2_hat = decode_full_game_score(
                    pred_diff[i], pred_total[i]
                )
            else:
                if PREDICT_TENNIS_INDEPENDENTLY:
                    s1_hat, s2_hat = decode_full_game_score(
                        pred_diff[i], pred_total[i]
                    )
                else:
                    running_diff_before_tn = int(
                        round(total_pred_p1[row_idx] - total_pred_p2[row_idx])
                    )
                    s1_hat, s2_hat = decode_tennis_score(
                        pred_diff[i],
                        pred_total[i],
                        running_diff_before_tn,
                    )

            total_pred_p1[row_idx] += s1_hat
            total_pred_p2[row_idx] += s2_hat

        print(f"\n{sport} TEST RESULTS")
        print("Diff MAE:", mean_absolute_error(y_diff_test, pred_diff))
        print("Total MAE:", mean_absolute_error(y_total_test, pred_total))

        sport_models[sport] = {
            "model_diff": model_diff,
            "model_total": model_total,
            "feature_cols": feature_cols,
        }

    return sport_models, total_pred_p1, total_pred_p2, split


# -------------------------------------------------
# Match-level evaluation
# -------------------------------------------------
def evaluate_match_level(
    df: pd.DataFrame, total_pred_p1, total_pred_p2, split: int
):
    true_diff = []
    pred_diff = []

    for i in range(split, len(df)):
        if "y_total_diff" not in df.columns or pd.isna(
            df.loc[i, "y_total_diff"]
        ):
            continue

        true_diff.append(float(df.loc[i, "y_total_diff"]))
        pred_diff.append(float(total_pred_p1[i] - total_pred_p2[i]))

    true_diff = np.array(true_diff)
    pred_diff = np.array(pred_diff)

    print("\n=== MATCH LEVEL ===")
    print("Total Diff MAE:", mean_absolute_error(true_diff, pred_diff))
    print("Winner Accuracy:", ((pred_diff > 0) == (true_diff > 0)).mean())


# -------------------------------------------------
# Synthetic inference from latest player states
# -------------------------------------------------
def orient_row_to_player_as_p1(row: pd.Series, player_name: str) -> pd.Series:
    player_name = player_name.strip().lower()
    out = row.copy()

    if out["p1_name"] == player_name:
        return out

    if out["p2_name"] != player_name:
        raise ValueError(f"Player '{player_name}' not found in supplied row.")

    for a, b in [("p1_name", "p2_name"), ("p1_key", "p2_key")]:
        if a in out.index and b in out.index:
            out[a], out[b] = out[b], out[a]

    if "h2h_avg_diff_p1" in out.index and pd.notna(out["h2h_avg_diff_p1"]):
        out["h2h_avg_diff_p1"] = -out["h2h_avg_diff_p1"]
    if "h2h_winrate_p1" in out.index and pd.notna(out["h2h_winrate_p1"]):
        out["h2h_winrate_p1"] = 1.0 - out["h2h_winrate_p1"]

    for sport in SPORTS:
        swap_pairs = [
            (f"{sport}_rating_p1", f"{sport}_rating_p2"),
            (f"{sport}_games_p1", f"{sport}_games_p2"),
            (f"{sport}_snapshot_rating_p1", f"{sport}_snapshot_rating_p2"),
            (f"{sport}_snapshot_p1_found", f"{sport}_snapshot_p2_found"),
        ]
        for a, b in swap_pairs:
            if a in out.index and b in out.index:
                out[a], out[b] = out[b], out[a]

        directional = [
            f"{sport}_rating_diff",
            f"{sport}_games_diff",
            f"{sport}_h2h_avg_diff_p1",
            f"{sport}_snapshot_rating_diff",
        ]
        for col in directional:
            if col in out.index and pd.notna(out[col]):
                out[col] = -out[col]

        if f"{sport}_h2h_winrate_p1" in out.index and pd.notna(
            out[f"{sport}_h2h_winrate_p1"]
        ):
            out[f"{sport}_h2h_winrate_p1"] = (
                1.0 - out[f"{sport}_h2h_winrate_p1"]
            )

    return out


def get_latest_player_state(df: pd.DataFrame, player_name: str) -> pd.Series:
    player_name = player_name.strip().lower()

    mask = (df["p1_name"] == player_name) | (df["p2_name"] == player_name)
    candidates = df[mask]
    if candidates.empty:
        raise ValueError(f"No history found for player '{player_name}'")

    row = candidates.iloc[-1].copy()
    return orient_row_to_player_as_p1(row, player_name)


def get_latest_pair_h2h_row(df: pd.DataFrame, player1: str, player2: str):
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()

    mask_direct = (df["p1_name"] == p1) & (df["p2_name"] == p2)
    mask_reverse = (df["p1_name"] == p2) & (df["p2_name"] == p1)

    candidates = df[mask_direct | mask_reverse]
    if candidates.empty:
        return None

    row = candidates.iloc[-1].copy()
    return orient_row_to_player_as_p1(row, p1)


def build_synthetic_match_row(
    df: pd.DataFrame, player1: str, player2: str, feature_cols
) -> pd.Series:
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()

    if p1 == p2:
        raise ValueError("PLAYER1 and PLAYER2 must be different.")

    s1 = get_latest_player_state(df, p1)
    s2 = get_latest_player_state(df, p2)
    pair_row = get_latest_pair_h2h_row(df, p1, p2)

    row = {}

    row["snapshot_found"] = float(
        max(
            int(
                s1.get("snapshot_found", 0)
                if pd.notna(s1.get("snapshot_found", 0))
                else 0
            ),
            int(
                s2.get("snapshot_found", 0)
                if pd.notna(s2.get("snapshot_found", 0))
                else 0
            ),
        )
    )

    row["h2h_games"] = 0.0
    row["h2h_avg_diff_p1"] = 0.0
    row["h2h_winrate_p1"] = 0.5
    row["h2h_days_since_last"] = np.nan

    if pair_row is not None:
        for col in [
            "h2h_games",
            "h2h_avg_diff_p1",
            "h2h_winrate_p1",
            "h2h_days_since_last",
        ]:
            if col in pair_row.index:
                row[col] = pair_row[col]

    for sport in SPORTS:
        r1 = float(s1.get(f"{sport}_rating_p1", BASE))
        r2 = float(s2.get(f"{sport}_rating_p1", BASE))
        g1 = float(s1.get(f"{sport}_games_p1", 0.0))
        g2 = float(s2.get(f"{sport}_games_p1", 0.0))

        row[f"{sport}_rating_p1"] = r1
        row[f"{sport}_rating_p2"] = r2
        row[f"{sport}_rating_diff"] = r1 - r2
        row[f"{sport}_games_p1"] = g1
        row[f"{sport}_games_p2"] = g2
        row[f"{sport}_games_diff"] = g1 - g2

        sr1 = float(s1.get(f"{sport}_snapshot_rating_p1", 0.0))
        sr2 = float(s2.get(f"{sport}_snapshot_rating_p1", 0.0))
        sf1 = int(
            s1.get(f"{sport}_snapshot_p1_found", 0)
            if pd.notna(s1.get(f"{sport}_snapshot_p1_found", 0))
            else 0
        )
        sf2 = int(
            s2.get(f"{sport}_snapshot_p1_found", 0)
            if pd.notna(s2.get(f"{sport}_snapshot_p1_found", 0))
            else 0
        )

        row[f"{sport}_snapshot_rating_p1"] = sr1
        row[f"{sport}_snapshot_rating_p2"] = sr2
        row[f"{sport}_snapshot_rating_diff"] = sr1 - sr2
        row[f"{sport}_snapshot_p1_found"] = sf1
        row[f"{sport}_snapshot_p2_found"] = sf2

        row[f"{sport}_h2h_games"] = 0.0
        row[f"{sport}_h2h_avg_diff_p1"] = 0.0
        row[f"{sport}_h2h_winrate_p1"] = 0.5
        row[f"{sport}_h2h_days_since_last"] = np.nan

        if pair_row is not None:
            for col in [
                f"{sport}_h2h_games",
                f"{sport}_h2h_avg_diff_p1",
                f"{sport}_h2h_winrate_p1",
                f"{sport}_h2h_days_since_last",
            ]:
                if col in pair_row.index:
                    row[col] = pair_row[col]

    final = {}
    for col in feature_cols:
        final[col] = row.get(col, np.nan)

    return pd.Series(final)


# -------------------------------------------------
# One-off prediction
# -------------------------------------------------
def predict_match_from_synthetic_row(
    player1: str, player2: str, row: pd.Series, sport_models
):
    total_p1 = 0
    total_p2 = 0

    print(f"{player1} {player2}")

    for sport in SPORTS:
        if sport not in sport_models:
            continue

        feature_cols = sport_models[sport]["feature_cols"]
        X_one = pd.DataFrame([row[feature_cols].to_dict()])

        pred_diff = float(sport_models[sport]["model_diff"].predict(X_one)[0])
        pred_total = float(sport_models[sport]["model_total"].predict(X_one)[0])

        pred_diff = float(np.clip(pred_diff, -21, 21))
        pred_total = float(np.clip(pred_total, 0, 42))

        if sport in ["TT", "BD", "SQ"]:
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

        print(f"{sport}: {s1}-{s2}")

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
def main():
    df = read_data(DATA_PATH)

    sport_models, total_pred_p1, total_pred_p2, split = train_models(df)
    evaluate_match_level(df, total_pred_p1, total_pred_p2, split)

    if PREDICT_ONE_MATCH:
        print()
        feature_cols = next(iter(sport_models.values()))["feature_cols"]
        row = build_synthetic_match_row(df, PLAYER1, PLAYER2, feature_cols)
        predict_match_from_synthetic_row(PLAYER1, PLAYER2, row, sport_models)


if __name__ == "__main__":
    main()
