import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42
BASE = 0.0

DATA_PATH = "data/data.csv"
INFERENCE_STATE_PATH = "data/inference_state.pkl"
OUTPUT_DIR = "full/catboost/artifacts/predictor_package"
TRAIN_RATIO = 0.8
PREDICT_TENNIS_INDEPENDENTLY = True

ALPHAS = {
    "TT": 0.060,
    "BD": 0.080,
    "SQ": 0.060,
    "TN": 0.040,
}

BASE_DIFF_MODEL_PARAMS = dict(
    iterations=900,
    depth=5,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=12,
    verbose=False,
    l2_leaf_reg=6,
)

RESID_MODEL_PARAMS = dict(
    iterations=700,
    depth=5,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=12,
    verbose=False,
    l2_leaf_reg=6,
)

TOTAL_MODEL_PARAMS = dict(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=12,
    verbose=False,
    l2_leaf_reg=6,
)

RESIDUAL_SCALE = 0.70


# -------------------------------------------------
# IO
# -------------------------------------------------
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def augment_with_mirrored_rows(df: pd.DataFrame) -> pd.DataFrame:
    mirrored = df.copy()

    for a, b in [("p1_name", "p2_name"), ("p1_key", "p2_key")]:
        if a in mirrored.columns and b in mirrored.columns:
            mirrored[a], mirrored[b] = mirrored[b].copy(), mirrored[a].copy()

    cols = list(mirrored.columns)

    for col in cols:
        if "_p1" in col:
            other = col.replace("_p1", "_p2")
            if other in mirrored.columns:
                mirrored[col], mirrored[other] = (
                    mirrored[other].copy(),
                    mirrored[col].copy(),
                )

    target_diff_cols = {f"{sport}_y_diff" for sport in SPORTS}
    target_diff_cols.add("y_total_diff")

    for col in target_diff_cols:
        if col in mirrored.columns:
            mirrored[col] = -mirrored[col]

    if "y_winner_p1" in mirrored.columns:
        mirrored["y_winner_p1"] = 1 - mirrored["y_winner_p1"]

    if "rating_winner_p1" in mirrored.columns:
        mirrored["rating_winner_p1"] = 1 - mirrored["rating_winner_p1"]

    for col in mirrored.columns:
        if col in target_diff_cols:
            continue

        should_negate = (
            col.endswith("_diff")
            or col.endswith("_diff_p1_p2")
            or col.endswith("_avg_diff_p1")
        )

        if should_negate and pd.api.types.is_numeric_dtype(mirrored[col]):
            mirrored[col] = -mirrored[col]

    for col in mirrored.columns:
        if col.endswith("_winrate_p1") and pd.api.types.is_numeric_dtype(
            mirrored[col]
        ):
            mirrored[col] = 1.0 - mirrored[col]

    df = df.copy()
    df["is_mirrored_row"] = 0
    mirrored["is_mirrored_row"] = 1

    out = pd.concat([df, mirrored], ignore_index=True)

    if "datetime" in out.columns:
        out = out.sort_values(["datetime", "is_mirrored_row"]).reset_index(
            drop=True
        )

    return out


def read_data(path):
    df = pd.read_csv(path, low_memory=False)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["p1_name", "p2_name", "p1_key", "p2_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    df = augment_with_mirrored_rows(df)
    return df


def load_inference_state(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Score utilities
# -------------------------------------------------
def rating_to_score_diff(x: float, alpha: float) -> float:
    return MAX_DIFF_PER_SPORT * np.tanh((alpha * x) / 2.0)


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
# Calibration / weights
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

    b = float(np.clip(b, 0.7, 1.8))
    a = float(np.clip(a, -5.0, 5.0))
    return {"a": a, "b": b}


def apply_linear_calibrator(y_pred, cal):
    return cal["a"] + cal["b"] * np.asarray(y_pred, dtype=float)


def diff_weights(y):
    a = np.abs(np.asarray(y, dtype=float))
    return (
        1.0
        + 0.15 * (a >= 5).astype(float)
        + 0.30 * (a >= 8).astype(float)
        + 0.45 * (a >= 12).astype(float)
    )


def residual_weights(y_true, base_pred):
    a = np.abs(np.asarray(y_true, dtype=float))
    return (
        1.0
        + 0.15 * (a >= 5).astype(float)
        + 0.30 * (a >= 8).astype(float)
        + 0.45 * (a >= 12).astype(float)
    )


def total_weights(y_total):
    y_total = np.asarray(y_total, dtype=float)
    return 1.0 + 0.02 * np.abs(y_total - 21.0)


# -------------------------------------------------
# Optional plots
# -------------------------------------------------
def save_scatter_plot(x, y, xlabel, ylabel, title, outpath):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.35)
    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_confusion_plot(y_true, y_pred, outpath):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["P2 win", "P1 win"])
    plt.yticks([0, 1], ["P2 win", "P1 win"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Winner confusion matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -------------------------------------------------
# Feature selection
# -------------------------------------------------
def get_feature_columns(df):
    drop_exact = {
        "match_index",
        "datetime",
        "month_key",
        "p1_name",
        "p2_name",
        "p1_key",
        "p2_key",
        "y_total_diff",
        "y_winner_p1",
        "rating_winner_p1",
        "is_mirrored_row",
    }

    feature_cols = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if c.endswith("_y_diff") or c.endswith("_y_total"):
            continue
        if c.startswith("has_"):
            continue
        feature_cols.append(c)
    return sorted(feature_cols)


def get_base_feature_cols(df, sport):
    wanted = [
        f"{sport}_rating_diff",
        f"{sport}_games_diff",
        f"{sport}_long_diff_mean_diff_p1_p2",
        f"{sport}_long_winrate_diff_p1_p2",
        "h2h_games",
        "h2h_avg_diff_p1",
        "h2h_winrate_p1",
        f"{sport}_h2h_games",
        f"{sport}_h2h_avg_diff_p1",
        f"{sport}_h2h_winrate_p1",
    ]
    return [c for c in wanted if c in df.columns]


def get_residual_feature_cols(df, sport):
    wanted = [
        f"{sport}_diff_mean_10_diff_p1_p2",
        f"{sport}_resid_mean_10_diff_p1_p2",
        f"{sport}_diff_std_10_diff_p1_p2",
        f"{sport}_momentum_diff_5_20_diff_p1_p2",
        f"{sport}_p1_recent_diff_std_10",
        f"{sport}_p2_recent_diff_std_10",
    ]
    return [c for c in wanted if c in df.columns]


def get_total_feature_cols(df, sport):
    wanted = [
        f"{sport}_rating_diff",
        f"{sport}_games_diff",
        f"{sport}_time_mult_diff",
        "h2h_games",
        "h2h_avg_diff_p1",
        "h2h_winrate_p1",
        f"{sport}_h2h_games",
        f"{sport}_h2h_avg_diff_p1",
        f"{sport}_h2h_winrate_p1",
        f"{sport}_diff_mean_10_diff_p1_p2",
        f"{sport}_resid_mean_10_diff_p1_p2",
        f"{sport}_diff_std_10_diff_p1_p2",
        f"{sport}_momentum_diff_5_20_diff_p1_p2",
        f"{sport}_long_diff_mean_diff_p1_p2",
        f"{sport}_long_total_mean_diff_p1_p2",
        f"{sport}_long_winrate_diff_p1_p2",
        f"{sport}_p1_long_n",
        f"{sport}_p2_long_n",
    ]
    return [c for c in wanted if c in df.columns]


# -------------------------------------------------
# Synthetic inference row
# -------------------------------------------------
def build_synthetic_match_row(
    inference_state, player1_key, player2_key, feature_cols
):
    if player1_key == player2_key:
        raise ValueError("PLAYER1 and PLAYER2 must be different.")

    player_states = inference_state["player_states_by_key"]
    if player1_key not in player_states:
        raise ValueError(f"No history found for player '{player1_key}'")
    if player2_key not in player_states:
        raise ValueError(f"No history found for player '{player2_key}'")

    s1 = player_states[player1_key]
    s2 = player_states[player2_key]

    row = {}

    row["h2h_games"] = 0.0
    row["h2h_avg_diff_p1"] = 0.0
    row["h2h_winrate_p1"] = 0.5
    row["h2h_days_since_last"] = 9999.0

    pk = (
        (player1_key, player2_key)
        if player1_key <= player2_key
        else (player2_key, player1_key)
    )
    pair_h2h = inference_state["pair_h2h"].get(pk)

    if pair_h2h is not None:
        overall = pair_h2h["overall"]
        a_key = pair_h2h["a_name"]
        b_key = pair_h2h["b_name"]

        row["h2h_games"] = overall["h2h_games"]
        row["h2h_days_since_last"] = float(
            overall.get("h2h_days_since_last", 9999.0)
        )

        if player1_key == a_key and player2_key == b_key:
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
            rating_to_score_diff(row[f"{sport}_rating_diff"], ALPHAS[sport])
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
            a_key = pair_h2h["a_name"]
            b_key = pair_h2h["b_name"]

            row[f"{sport}_h2h_games"] = sport_h[f"{sport}_h2h_games"]
            row[f"{sport}_h2h_days_since_last"] = float(
                sport_h.get(f"{sport}_h2h_days_since_last", 9999.0)
            )

            if player1_key == a_key and player2_key == b_key:
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

        long_suffixes = [
            "long_n",
            "long_diff_mean",
            "long_total_mean",
            "long_winrate",
        ]
        for suffix in long_suffixes:
            p1_col = f"{sport}_p1_{suffix}"
            p2_col = f"{sport}_p2_{suffix}"
            row[p1_col] = float(s1.get(p1_col, 0.0))
            row[p2_col] = float(s2.get(p2_col, 0.0))
            row[f"{sport}_{suffix}_diff_p1_p2"] = row[p1_col] - row[p2_col]

    return pd.Series({col: row.get(col, 0.0) for col in feature_cols})


# -------------------------------------------------
# Package
# -------------------------------------------------
@dataclass
class PredictorPackage:
    models: dict
    feature_cols: dict
    inference_state: dict
    metadata: Optional[dict] = None

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "models.pkl", "rb") as f:
            models = pickle.load(f)
        with open(directory / "feature_cols.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        with open(directory / "inference_state.pkl", "rb") as f:
            inference_state = pickle.load(f)

        metadata = None
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            models=models,
            feature_cols=feature_cols,
            inference_state=inference_state,
            metadata=metadata,
        )

    def predict_pair(self, player1_key, player2_key):
        total_p1 = 0
        total_p2 = 0
        sports_out = {}

        p1_state = self.get_player_state(player1_key)
        p2_state = self.get_player_state(player2_key)

        p1_name = p1_state.get("player_name", player1_key)
        p2_name = p2_state.get("player_name", player2_key)

        for sport in SPORTS:
            pack = self.models[sport]

            row = build_synthetic_match_row(
                self.inference_state,
                player1_key,
                player2_key,
                pack["all_feat_cols"],
            )

            X_base = pd.DataFrame(
                [row[pack["base_feat_cols"]].to_dict()]
            ).fillna(0.0)
            X_resid = pd.DataFrame(
                [row[pack["resid_feat_cols"]].to_dict()]
            ).fillna(0.0)
            X_total = pd.DataFrame(
                [row[pack["total_feat_cols"]].to_dict()]
            ).fillna(0.0)

            base_pred = float(pack["model_diff_base"].predict(X_base)[0])

            X_resid = X_resid.copy()
            X_resid["base_pred_diff"] = base_pred
            X_resid["abs_base_pred_diff"] = abs(base_pred)

            resid_pred = float(pack["model_diff_resid"].predict(X_resid)[0])

            pred_diff_raw = base_pred + RESIDUAL_SCALE * resid_pred

            pred_diff = float(
                np.clip(
                    apply_linear_calibrator(
                        np.array([pred_diff_raw]),
                        pack["diff_calibrator"],
                    )[0],
                    -21,
                    21,
                )
            )

            pred_total = float(
                np.clip(pack["model_total"].predict(X_total)[0], 0, 42)
            )

            if sport in ["TT", "BD", "SQ"]:
                s1, s2 = decode_full_game_score(pred_diff, pred_total)
            else:
                if PREDICT_TENNIS_INDEPENDENTLY:
                    s1, s2 = decode_full_game_score(pred_diff, pred_total)
                else:
                    s1, s2 = decode_tennis_score(
                        pred_diff,
                        pred_total,
                        total_p1 - total_p2,
                    )

            s1 = int(s1)
            s2 = int(s2)

            total_p1 += s1
            total_p2 += s2

            sports_out[sport] = {
                "score_p1": s1,
                "score_p2": s2,
                "base_pred_diff": float(base_pred),
                "resid_pred_diff": float(resid_pred),
                "pred_diff": float(pred_diff),
                "pred_total": float(pred_total),
            }

        total_p1 = int(total_p1)
        total_p2 = int(total_p2)
        total_diff = int(total_p1 - total_p2)

        winner = (
            p1_name if total_diff > 0 else p2_name if total_diff < 0 else "Draw"
        )

        return {
            "player1": player1_key,
            "player2": player2_key,
            "player1_key": player1_key,
            "player2_key": player2_key,
            "player1_name": p1_name,
            "player2_name": p2_name,
            "sports": sports_out,
            # official displayed prediction totals
            "total_p1": total_p1,
            "total_p2": total_p2,
            "total_diff": total_diff,
            "winner": winner,
        }

    def get_player_state(self, player_key):
        state = self.inference_state["player_states_by_key"].get(player_key)
        if state is None:
            raise ValueError(f"No history found for player '{player_key}'")
        return state


# -------------------------------------------------
# Public convenience helpers
# -------------------------------------------------
def load_predictor(directory: str = OUTPUT_DIR) -> PredictorPackage:
    return PredictorPackage.load(directory)


def predict_match(
    predictor: PredictorPackage,
    player1_key: str,
    player2_key: str,
) -> dict:
    return predictor.predict_pair(player1_key, player2_key)


def get_player_ratings(
    predictor: PredictorPackage,
    player_key: str,
) -> dict:
    state = predictor.get_player_state(player_key)
    out = {
        "player_key": player_key,
        "player_name": state.get("player_name", player_key),
        "country": state.get("player_country", ""),
        "sports": {},
    }

    for sport in SPORTS:
        out["sports"][sport] = {
            "rating": float(state.get(f"{sport}_rating_p1", 0.0)),
            "pred_diff_vs_avg": float(state.get(f"{sport}_pred_diff", 0.0)),
            "games_played": int(state.get(f"{sport}_games_p1", 0.0)),
            "days_since_last": float(
                state.get(f"{sport}_days_since_last_p1", 0.0)
            ),
            "time_multiplier": float(state.get(f"{sport}_time_mult_p1", 1.0)),
            "diff_mean_10": float(
                state.get(f"{sport}_p1_recent_diff_mean_10", 0.0)
            ),
            "resid_mean_10": float(
                state.get(f"{sport}_p1_recent_resid_mean_10", 0.0)
            ),
            "diff_std_10": float(
                state.get(f"{sport}_p1_recent_diff_std_10", 0.0)
            ),
            "momentum_diff_5_20": float(
                state.get(f"{sport}_p1_recent_momentum_diff_5_20", 0.0)
            ),
            "long_n": float(state.get(f"{sport}_p1_long_n", 0.0)),
            "long_diff_mean": float(
                state.get(f"{sport}_p1_long_diff_mean", 0.0)
            ),
            "long_total_mean": float(
                state.get(f"{sport}_p1_long_total_mean", 0.0)
            ),
            "long_winrate": float(state.get(f"{sport}_p1_long_winrate", 0.0)),
        }

    return out


# -------------------------------------------------
# Internal train helpers
# -------------------------------------------------
def _train_single_sport_models(
    df: pd.DataFrame,
    sport: str,
    fit_rows_mask: np.ndarray,
):
    base_feat_cols = get_base_feature_cols(df, sport)
    resid_feat_cols = get_residual_feature_cols(df, sport)
    total_feat_cols = get_total_feature_cols(df, sport)
    all_feat_cols = sorted(
        set(base_feat_cols + resid_feat_cols + total_feat_cols)
    )

    diff_col = f"{sport}_y_diff"
    total_col = f"{sport}_y_total"

    mask = df[diff_col].notna() & df[total_col].notna() & fit_rows_mask
    rows = np.where(mask.values)[0]

    if len(rows) == 0:
        return None

    X_base = df.loc[rows, base_feat_cols].copy().fillna(0.0)
    X_resid = df.loc[rows, resid_feat_cols].copy().fillna(0.0)
    X_total = df.loc[rows, total_feat_cols].copy().fillna(0.0)

    y_diff = df.loc[rows, diff_col].astype(float).values
    y_total = df.loc[rows, total_col].astype(float).values

    model_diff_base = CatBoostRegressor(**BASE_DIFF_MODEL_PARAMS)
    model_diff_base.fit(
        X_base,
        y_diff,
        sample_weight=diff_weights(y_diff),
    )

    base_pred = model_diff_base.predict(X_base)
    resid_train_target = y_diff - base_pred

    X_resid_aug = X_resid.copy()
    X_resid_aug["base_pred_diff"] = base_pred
    X_resid_aug["abs_base_pred_diff"] = np.abs(base_pred)

    model_diff_resid = CatBoostRegressor(**RESID_MODEL_PARAMS)
    model_diff_resid.fit(
        X_resid_aug,
        resid_train_target,
        sample_weight=residual_weights(y_diff, base_pred),
    )

    resid_pred = model_diff_resid.predict(X_resid_aug)
    pred_diff_raw = base_pred + RESIDUAL_SCALE * resid_pred
    diff_calibrator = fit_linear_calibrator(y_diff, pred_diff_raw)

    model_total = CatBoostRegressor(**TOTAL_MODEL_PARAMS)
    model_total.fit(
        X_total,
        y_total,
        sample_weight=total_weights(y_total),
    )

    return {
        "model_diff_base": model_diff_base,
        "model_diff_resid": model_diff_resid,
        "model_total": model_total,
        "base_feat_cols": base_feat_cols,
        "resid_feat_cols": resid_feat_cols,
        "total_feat_cols": total_feat_cols,
        "all_feat_cols": all_feat_cols,
        "diff_calibrator": diff_calibrator,
        "n_rows_fit": int(len(rows)),
    }


def _evaluate_models(
    df: pd.DataFrame,
    sport_models: dict,
    eval_rows_mask: np.ndarray,
):
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))
    metrics = {}

    for sport in SPORTS:
        pack = sport_models[sport]

        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"

        mask = df[diff_col].notna() & df[total_col].notna() & eval_rows_mask
        rows = np.where(mask.values)[0]

        if len(rows) == 0:
            continue

        X_base = df.loc[rows, pack["base_feat_cols"]].copy().fillna(0.0)
        X_resid = df.loc[rows, pack["resid_feat_cols"]].copy().fillna(0.0)
        X_total = df.loc[rows, pack["total_feat_cols"]].copy().fillna(0.0)

        y_diff = df.loc[rows, diff_col].astype(float).values
        y_total = df.loc[rows, total_col].astype(float).values

        base_pred = pack["model_diff_base"].predict(X_base)

        X_resid_aug = X_resid.copy()
        X_resid_aug["base_pred_diff"] = base_pred
        X_resid_aug["abs_base_pred_diff"] = np.abs(base_pred)

        resid_pred = pack["model_diff_resid"].predict(X_resid_aug)
        pred_diff_raw = base_pred + RESIDUAL_SCALE * resid_pred

        pred_diff = np.clip(
            apply_linear_calibrator(pred_diff_raw, pack["diff_calibrator"]),
            -21,
            21,
        )
        pred_total = np.clip(pack["model_total"].predict(X_total), 0, 42)

        for idx, row_idx in enumerate(rows):
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

        diff_mae = float(mean_absolute_error(y_diff, pred_diff))
        total_mae = float(mean_absolute_error(y_total, pred_total))
        diff_std_ratio = float(np.std(pred_diff) / max(np.std(y_diff), 1e-9))

        metrics[sport] = {
            "diff_mae": diff_mae,
            "total_mae": total_mae,
            "diff_std_ratio_pred_over_actual": diff_std_ratio,
            "n_eval_rows": int(len(rows)),
        }

    true_diff = []
    pred_diff_match = []

    for i in np.where(eval_rows_mask)[0]:
        if pd.isna(df.loc[i, "y_total_diff"]):
            continue
        true_diff.append(float(df.loc[i, "y_total_diff"]))
        pred_diff_match.append(float(total_pred_p1[i] - total_pred_p2[i]))

    true_diff = np.array(true_diff)
    pred_diff_match = np.array(pred_diff_match)

    match_metrics = None
    if len(true_diff) > 0:
        match_metrics = {
            "total_diff_mae": float(
                mean_absolute_error(true_diff, pred_diff_match)
            ),
            "winner_accuracy": float(
                ((pred_diff_match > 0) == (true_diff > 0)).mean()
            ),
        }

    return {
        "sport_metrics": metrics,
        "match_metrics": match_metrics,
        "total_pred_p1": total_pred_p1,
        "total_pred_p2": total_pred_p2,
    }


def _save_package(
    sport_models: dict,
    inference_state: dict,
    outdir: str,
    metadata: dict,
):
    outdir = ensure_dir(outdir)

    with open(Path(outdir) / "models.pkl", "wb") as f:
        pickle.dump(sport_models, f)

    with open(Path(outdir) / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                sport: {
                    "base_feat_cols": sport_models[sport]["base_feat_cols"],
                    "resid_feat_cols": sport_models[sport]["resid_feat_cols"],
                    "total_feat_cols": sport_models[sport]["total_feat_cols"],
                    "all_feat_cols": sport_models[sport]["all_feat_cols"],
                }
                for sport in sport_models
            },
            f,
            indent=2,
        )

    with open(Path(outdir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(Path(outdir) / "inference_state.pkl", "wb") as f:
        pickle.dump(inference_state, f)


# -------------------------------------------------
# Public train functions
# -------------------------------------------------
def train_eval_and_package(
    data_path: str = DATA_PATH,
    inference_state_path: str = INFERENCE_STATE_PATH,
    output_dir: str = OUTPUT_DIR,
    train_ratio: float = TRAIN_RATIO,
) -> PredictorPackage:
    df = read_data(data_path)
    inference_state = load_inference_state(inference_state_path)

    split = int(len(df) * train_ratio)
    train_rows_mask = np.arange(len(df)) < split
    test_rows_mask = np.arange(len(df)) >= split

    sport_models = {}
    for sport in SPORTS:
        pack = _train_single_sport_models(df, sport, train_rows_mask)
        if pack is not None:
            sport_models[sport] = pack

    eval_out = _evaluate_models(df, sport_models, test_rows_mask)

    metadata = {
        "mode": "train_eval",
        "data_path": data_path,
        "inference_state_path": inference_state_path,
        "train_ratio": train_ratio,
        "n_rows": int(len(df)),
        "sport_metrics": eval_out["sport_metrics"],
        "match_metrics": eval_out["match_metrics"],
    }

    _save_package(
        sport_models=sport_models,
        inference_state=inference_state,
        outdir=output_dir,
        metadata=metadata,
    )

    return PredictorPackage.load(output_dir)


def train_full_and_package(
    data_path: str = DATA_PATH,
    inference_state_path: str = INFERENCE_STATE_PATH,
    output_dir: str = OUTPUT_DIR,
) -> PredictorPackage:
    df = read_data(data_path)
    inference_state = load_inference_state(inference_state_path)

    full_rows_mask = np.ones(len(df), dtype=bool)

    sport_models = {}
    fit_metrics = {}
    for sport in SPORTS:
        pack = _train_single_sport_models(df, sport, full_rows_mask)
        if pack is not None:
            sport_models[sport] = pack
            fit_metrics[sport] = {"n_fit_rows": pack["n_rows_fit"]}

    metadata = {
        "mode": "train_full",
        "data_path": data_path,
        "inference_state_path": inference_state_path,
        "n_rows": int(len(df)),
        "sport_metrics": fit_metrics,
        "match_metrics": None,
    }

    _save_package(
        sport_models=sport_models,
        inference_state=inference_state,
        outdir=output_dir,
        metadata=metadata,
    )

    return PredictorPackage.load(output_dir)


# -------------------------------------------------
# Example CLI usage
# -------------------------------------------------
if __name__ == "__main__":
    predictor = train_eval_and_package()
    print("Saved evaluated package.")
