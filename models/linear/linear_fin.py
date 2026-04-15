import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, confusion_matrix

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42
BASE = 0.0

DATA_PATH = "data/data.csv"
INFERENCE_STATE_PATH = "data/inference_state.pkl"
OUTPUT_DIR = "finished_models/linear/artifacts/predictor_package"
TRAIN_RATIO = 0.8
RIDGE_ALPHA = 3.0
PREDICT_TENNIS_INDEPENDENTLY = True


# -------------------------------------------------
# IO
# -------------------------------------------------
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_data(path):
    df = pd.read_csv(path, low_memory=False)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["p1_name", "p2_name", "p1_key", "p2_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def load_inference_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
# Feature selection
# -------------------------------------------------
def get_compact_feature_cols(df, sport):
    wanted = [
        f"{sport}_rating_p1",
        f"{sport}_rating_p2",
        f"{sport}_rating_diff",
        f"{sport}_pred_diff",
        f"{sport}_games_p1",
        f"{sport}_games_p2",
        f"{sport}_games_diff",
        f"{sport}_days_since_last_p1",
        f"{sport}_days_since_last_p2",
        f"{sport}_time_mult_p1",
        f"{sport}_time_mult_p2",
        f"{sport}_time_mult_diff",
        "h2h_games",
        "h2h_avg_diff_p1",
        "h2h_winrate_p1",
        "h2h_days_since_last",
        f"{sport}_h2h_games",
        f"{sport}_h2h_avg_diff_p1",
        f"{sport}_h2h_winrate_p1",
        f"{sport}_h2h_days_since_last",
        f"{sport}_p1_recent_diff_mean_10",
        f"{sport}_p2_recent_diff_mean_10",
        f"{sport}_diff_mean_10_diff_p1_p2",
        f"{sport}_p1_recent_resid_mean_10",
        f"{sport}_p2_recent_resid_mean_10",
        f"{sport}_resid_mean_10_diff_p1_p2",
        f"{sport}_p1_recent_blowout_win_rate_10",
        f"{sport}_p2_recent_blowout_win_rate_10",
        f"{sport}_blowout_win_rate_10_diff_p1_p2",
        f"{sport}_p1_recent_diff_std_10",
        f"{sport}_p2_recent_diff_std_10",
        f"{sport}_diff_std_10_diff_p1_p2",
        f"{sport}_p1_recent_momentum_diff_5_20",
        f"{sport}_p2_recent_momentum_diff_5_20",
        f"{sport}_momentum_diff_5_20_diff_p1_p2",
    ]
    return [c for c in wanted if c in df.columns]


# -------------------------------------------------
# Calibration
# -------------------------------------------------
def fit_linear_calibrator(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_pred) < 1e-9:
        return {"a": 0.0, "b": 1.0}
    b, a = np.polyfit(y_pred, y_true, 1)
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 1.0
    b = float(np.clip(b, 0.7, 1.6))
    a = float(np.clip(a, -5.0, 5.0))
    return {"a": a, "b": b}


def apply_linear_calibrator(y_pred, cal):
    return cal["a"] + cal["b"] * np.asarray(y_pred, dtype=float)


# -------------------------------------------------
# Synthetic inference row
# -------------------------------------------------
def build_synthetic_match_row(inference_state, player1, player2, feature_cols):
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()
    if p1 == p2:
        raise ValueError("PLAYER1 and PLAYER2 must be different.")

    player_states = inference_state["player_states_by_name"]
    if p1 not in player_states:
        raise ValueError(f"No history found for player '{player1}'")
    if p2 not in player_states:
        raise ValueError(f"No history found for player '{player2}'")

    s1 = player_states[p1]
    s2 = player_states[p2]

    row = {}

    row["h2h_games"] = 0.0
    row["h2h_avg_diff_p1"] = 0.0
    row["h2h_winrate_p1"] = 0.5
    row["h2h_days_since_last"] = 9999.0

    pk = (p1, p2) if p1 <= p2 else (p2, p1)
    pair_h2h = inference_state["pair_h2h"].get(pk)

    if pair_h2h is not None:
        overall = pair_h2h["overall"]
        a_name = pair_h2h["a_name"]
        b_name = pair_h2h["b_name"]

        row["h2h_games"] = overall["h2h_games"]
        if p1 == a_name and p2 == b_name:
            row["h2h_avg_diff_p1"] = overall["h2h_avg_diff_p1"]
            row["h2h_winrate_p1"] = overall["h2h_winrate_p1"]
        else:
            row["h2h_avg_diff_p1"] = -overall["h2h_avg_diff_p1"]
            row["h2h_winrate_p1"] = 1.0 - overall["h2h_winrate_p1"]

    for sport in SPORTS:
        r1 = float(s1.get(f"{sport}_rating_p1", BASE))
        r2 = float(s2.get(f"{sport}_rating_p1", BASE))
        g1 = float(s1.get(f"{sport}_games_p1", 0.0))
        g2 = float(s2.get(f"{sport}_games_p1", 0.0))

        row[f"{sport}_rating_p1"] = r1
        row[f"{sport}_rating_p2"] = r2
        row[f"{sport}_rating_diff"] = r1 - r2
        row[f"{sport}_pred_diff"] = float(
            s1.get(f"{sport}_pred_diff", 0.0)
            - s2.get(f"{sport}_pred_diff", 0.0)
        )
        row[f"{sport}_games_p1"] = g1
        row[f"{sport}_games_p2"] = g2
        row[f"{sport}_games_diff"] = g1 - g2

        row[f"{sport}_days_since_last_p1"] = float(
            s1.get(f"{sport}_days_since_last_p1", 0.0)
        )
        row[f"{sport}_days_since_last_p2"] = float(
            s2.get(f"{sport}_days_since_last_p1", 0.0)
        )
        row[f"{sport}_time_mult_p1"] = float(
            s1.get(f"{sport}_time_mult_p1", 1.0)
        )
        row[f"{sport}_time_mult_p2"] = float(
            s2.get(f"{sport}_time_mult_p1", 1.0)
        )
        row[f"{sport}_time_mult_diff"] = (
            row[f"{sport}_time_mult_p1"] - row[f"{sport}_time_mult_p2"]
        )

        row[f"{sport}_h2h_games"] = 0.0
        row[f"{sport}_h2h_avg_diff_p1"] = 0.0
        row[f"{sport}_h2h_winrate_p1"] = 0.5
        row[f"{sport}_h2h_days_since_last"] = 9999.0

        if pair_h2h is not None:
            sport_h = pair_h2h["sports"][sport]
            a_name = pair_h2h["a_name"]
            b_name = pair_h2h["b_name"]
            row[f"{sport}_h2h_games"] = sport_h[f"{sport}_h2h_games"]
            if p1 == a_name and p2 == b_name:
                row[f"{sport}_h2h_avg_diff_p1"] = sport_h[
                    f"{sport}_h2h_avg_diff_p1"
                ]
                row[f"{sport}_h2h_winrate_p1"] = sport_h[
                    f"{sport}_h2h_winrate_p1"
                ]
            else:
                row[f"{sport}_h2h_avg_diff_p1"] = -sport_h[
                    f"{sport}_h2h_avg_diff_p1"
                ]
                row[f"{sport}_h2h_winrate_p1"] = (
                    1.0 - sport_h[f"{sport}_h2h_winrate_p1"]
                )

        recent_suffixes = [
            "diff_mean_10",
            "resid_mean_10",
            "blowout_win_rate_10",
            "diff_std_10",
            "momentum_diff_5_20",
        ]
        for suffix in recent_suffixes:
            p1_col = f"{sport}_p1_recent_{suffix}"
            p2_col = f"{sport}_p2_recent_{suffix}"
            row[p1_col] = float(s1.get(p1_col, 0.0))
            row[p2_col] = float(s2.get(p2_col, 0.0))
            row[f"{sport}_{suffix}_diff_p1_p2"] = row[p1_col] - row[p2_col]

    return pd.Series({col: row.get(col, np.nan) for col in feature_cols})


# -------------------------------------------------
# Package
# -------------------------------------------------
@dataclass
class PredictorPackage:
    models: dict
    feature_cols: list
    inference_state: dict

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "models.pkl", "rb") as f:
            models = pickle.load(f)
        with open(directory / "feature_cols.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        with open(directory / "inference_state.pkl", "rb") as f:
            inference_state = pickle.load(f)
        return cls(
            models=models,
            feature_cols=feature_cols,
            inference_state=inference_state,
        )

    def predict_pair(self, player1, player2):
        total_p1 = 0
        total_p2 = 0
        sports_out = {}

        for sport in SPORTS:
            pack = self.models[sport]
            row = build_synthetic_match_row(
                self.inference_state, player1, player2, pack["feat_cols"]
            )
            X_one = pd.DataFrame([row[pack["feat_cols"]].to_dict()]).fillna(0.0)

            pred_diff_raw = float(pack["model_diff"].predict(X_one)[0])
            pred_diff = float(
                np.clip(
                    apply_linear_calibrator(
                        np.array([pred_diff_raw]), pack["diff_calibrator"]
                    )[0],
                    -21,
                    21,
                )
            )
            pred_total = float(
                np.clip(pack["model_total"].predict(X_one)[0], 0, 42)
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
# Train
# -------------------------------------------------
def train_and_package():
    outdir = ensure_dir(OUTPUT_DIR)

    df = read_data(DATA_PATH)
    inference_state = load_inference_state(INFERENCE_STATE_PATH)
    split = int(len(df) * TRAIN_RATIO)

    sport_models = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))
    metrics = {}

    for sport in SPORTS:
        feat_cols = get_compact_feature_cols(df, sport)
        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"

        mask = df[diff_col].notna() & df[total_col].notna()
        rows = np.where(mask.values)[0]

        X = df.loc[rows, feat_cols].copy().fillna(0.0)
        y_diff = df.loc[rows, diff_col].astype(float).values
        y_total = df.loc[rows, total_col].astype(float).values

        train_mask = rows < split
        test_mask = rows >= split

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = X.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_diff_train = y_diff[train_mask]
        y_diff_test = y_diff[test_mask]
        y_total_train = y_total[train_mask]
        y_total_test = y_total[test_mask]

        model_diff = Ridge(alpha=RIDGE_ALPHA)
        model_total = Ridge(alpha=RIDGE_ALPHA)

        model_diff.fit(X_train, y_diff_train)
        model_total.fit(X_train, y_total_train)

        pred_diff_train = model_diff.predict(X_train)
        diff_calibrator = fit_linear_calibrator(y_diff_train, pred_diff_train)

        pred_diff = np.clip(
            apply_linear_calibrator(
                model_diff.predict(X_test), diff_calibrator
            ),
            -21,
            21,
        )
        pred_total = np.clip(model_total.predict(X_test), 0, 42)

        for idx, row_idx in enumerate(rows[test_mask]):
            if sport in ["TT", "BD", "SQ"]:
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
                        pred_diff[idx], pred_total[idx], running_diff_before_tn
                    )

            total_pred_p1[row_idx] += s1_hat
            total_pred_p2[row_idx] += s2_hat

        diff_mae = float(mean_absolute_error(y_diff_test, pred_diff))
        total_mae = float(mean_absolute_error(y_total_test, pred_total))

        print(f"\n{sport} TEST RESULTS")
        print("Diff MAE:", diff_mae)
        print("Total MAE:", total_mae)

        metrics[sport] = {
            "diff_mae": diff_mae,
            "total_mae": total_mae,
            "n_features": len(feat_cols),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
        }

        sport_models[sport] = {
            "model_diff": model_diff,
            "model_total": model_total,
            "feat_cols": feat_cols,
            "diff_calibrator": diff_calibrator,
        }

    true_diff = []
    pred_diff = []

    for i in range(split, len(df)):
        if pd.isna(df.loc[i, "y_total_diff"]):
            continue
        true_diff.append(float(df.loc[i, "y_total_diff"]))
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
        "inference_state_path": INFERENCE_STATE_PATH,
        "train_ratio": TRAIN_RATIO,
        "ridge_alpha": RIDGE_ALPHA,
        "n_rows": int(len(df)),
        "sport_metrics": metrics,
        "match_metrics": {
            "total_diff_mae": match_mae,
            "winner_accuracy": winner_acc,
        },
    }

    with open(outdir / "models.pkl", "wb") as f:
        pickle.dump(sport_models, f)
    with open(outdir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(
            {sport: sport_models[sport]["feat_cols"] for sport in sport_models},
            f,
            indent=2,
        )
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(outdir / "inference_state.pkl", "wb") as f:
        pickle.dump(inference_state, f)

    print(f"\nSaved baseline package to: {outdir}")


if __name__ == "__main__":
    train_and_package()
