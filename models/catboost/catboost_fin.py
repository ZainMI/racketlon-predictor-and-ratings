import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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
OUTPUT_DIR = "finished_models/catboost/artifacts/predictor_package"
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
# Plots
# -------------------------------------------------
def save_scatter_plot(x, y, xlabel, ylabel, title, outpath):
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


def save_feature_importance_plot(model, feature_cols, title, outpath, top_n=20):
    importances = model.get_feature_importance()
    order = np.argsort(importances)[::-1][:top_n]
    labels = [feature_cols[i] for i in order][::-1]
    vals = importances[order][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
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
        row["h2h_days_since_last"] = float(
            overall.get("h2h_days_since_last", 9999.0)
        )

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
            a_name = pair_h2h["a_name"]
            b_name = pair_h2h["b_name"]

            row[f"{sport}_h2h_games"] = sport_h[f"{sport}_h2h_games"]
            row[f"{sport}_h2h_days_since_last"] = float(
                sport_h.get(f"{sport}_h2h_days_since_last", 9999.0)
            )

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
                self.inference_state, player1, player2, pack["all_feat_cols"]
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
                        np.array([pred_diff_raw]), pack["diff_calibrator"]
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
                        pred_diff, pred_total, total_p1 - total_p2
                    )

            total_p1 += s1
            total_p2 += s2

            sports_out[sport] = {
                "score_p1": int(s1),
                "score_p2": int(s2),
                "base_pred_diff": float(base_pred),
                "resid_pred_diff": float(resid_pred),
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
    plots_dir = ensure_dir(outdir / "plots")

    df = read_data(DATA_PATH)
    inference_state = load_inference_state(INFERENCE_STATE_PATH)
    split = int(len(df) * TRAIN_RATIO)
    all_feature_cols = get_feature_columns(df)

    sport_models = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))
    metrics = {}

    print(f"Using {len(all_feature_cols)} total columns")
    print(
        "Training compact CatBoost with base + residual model, no blowout features."
    )

    for sport in SPORTS:
        base_feat_cols = get_base_feature_cols(df, sport)
        resid_feat_cols = get_residual_feature_cols(df, sport)
        total_feat_cols = get_total_feature_cols(df, sport)
        all_feat_cols = sorted(
            set(base_feat_cols + resid_feat_cols + total_feat_cols)
        )

        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"

        mask = df[diff_col].notna() & df[total_col].notna()
        rows = np.where(mask.values)[0]

        X_base = df.loc[rows, base_feat_cols].copy().fillna(0.0)
        X_resid = df.loc[rows, resid_feat_cols].copy().fillna(0.0)
        X_total = df.loc[rows, total_feat_cols].copy().fillna(0.0)

        y_diff = df.loc[rows, diff_col].astype(float).values
        y_total = df.loc[rows, total_col].astype(float).values

        train_mask = rows < split
        test_mask = rows >= split

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_base_train = X_base.iloc[train_mask]
        X_base_test = X_base.iloc[test_mask]

        X_resid_train = X_resid.iloc[train_mask]
        X_resid_test = X_resid.iloc[test_mask]

        X_total_train = X_total.iloc[train_mask]
        X_total_test = X_total.iloc[test_mask]

        y_diff_train = y_diff[train_mask]
        y_diff_test = y_diff[test_mask]
        y_total_train = y_total[train_mask]
        y_total_test = y_total[test_mask]

        model_diff_base = CatBoostRegressor(**BASE_DIFF_MODEL_PARAMS)
        model_diff_base.fit(
            X_base_train,
            y_diff_train,
            sample_weight=diff_weights(y_diff_train),
        )

        base_pred_train = model_diff_base.predict(X_base_train)
        base_pred_test = model_diff_base.predict(X_base_test)

        resid_train_target = y_diff_train - base_pred_train

        X_resid_train_aug = X_resid_train.copy()
        X_resid_test_aug = X_resid_test.copy()
        X_resid_train_aug["base_pred_diff"] = base_pred_train
        X_resid_train_aug["abs_base_pred_diff"] = np.abs(base_pred_train)
        X_resid_test_aug["base_pred_diff"] = base_pred_test
        X_resid_test_aug["abs_base_pred_diff"] = np.abs(base_pred_test)

        model_diff_resid = CatBoostRegressor(**RESID_MODEL_PARAMS)
        model_diff_resid.fit(
            X_resid_train_aug,
            resid_train_target,
            sample_weight=residual_weights(y_diff_train, base_pred_train),
        )

        resid_pred_train = model_diff_resid.predict(X_resid_train_aug)
        resid_pred_test = model_diff_resid.predict(X_resid_test_aug)

        pred_diff_train_raw = (
            base_pred_train + RESIDUAL_SCALE * resid_pred_train
        )
        pred_diff_test_raw = base_pred_test + RESIDUAL_SCALE * resid_pred_test

        diff_calibrator = fit_linear_calibrator(
            y_diff_train, pred_diff_train_raw
        )

        pred_diff = np.clip(
            apply_linear_calibrator(pred_diff_test_raw, diff_calibrator),
            -21,
            21,
        )

        model_total = CatBoostRegressor(**TOTAL_MODEL_PARAMS)
        model_total.fit(
            X_total_train,
            y_total_train,
            sample_weight=total_weights(y_total_train),
        )
        pred_total = np.clip(model_total.predict(X_total_test), 0, 42)

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
        diff_std_ratio = float(
            np.std(pred_diff) / max(np.std(y_diff_test), 1e-9)
        )

        print(f"\n{sport} TEST RESULTS")
        print("Diff MAE:", diff_mae)
        print("Total MAE:", total_mae)
        print("Diff std ratio pred/actual:", diff_std_ratio)

        metrics[sport] = {
            "diff_mae": diff_mae,
            "total_mae": total_mae,
            "diff_std_ratio_pred_over_actual": diff_std_ratio,
            "n_base_features": len(base_feat_cols),
            "n_resid_features": len(resid_feat_cols) + 2,
            "n_total_features": len(total_feat_cols),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
        }

        save_scatter_plot(
            y_diff_test,
            pred_diff,
            "Actual diff",
            "Predicted diff",
            f"{sport} diff: actual vs predicted",
            plots_dir / f"{sport.lower()}_diff_scatter.png",
        )
        save_scatter_plot(
            y_total_test,
            pred_total,
            "Actual total",
            "Predicted total",
            f"{sport} total: actual vs predicted",
            plots_dir / f"{sport.lower()}_total_scatter.png",
        )
        save_feature_importance_plot(
            model_diff_base,
            base_feat_cols,
            f"{sport} base diff feature importance",
            plots_dir / f"{sport.lower()}_base_diff_feature_importance.png",
        )
        save_feature_importance_plot(
            model_diff_resid,
            resid_feat_cols + ["base_pred_diff", "abs_base_pred_diff"],
            f"{sport} residual diff feature importance",
            plots_dir / f"{sport.lower()}_resid_diff_feature_importance.png",
        )
        save_feature_importance_plot(
            model_total,
            total_feat_cols,
            f"{sport} total feature importance",
            plots_dir / f"{sport.lower()}_total_feature_importance.png",
        )

        sport_models[sport] = {
            "model_diff_base": model_diff_base,
            "model_diff_resid": model_diff_resid,
            "model_total": model_total,
            "base_feat_cols": base_feat_cols,
            "resid_feat_cols": resid_feat_cols,
            "total_feat_cols": total_feat_cols,
            "all_feat_cols": all_feat_cols,
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

    save_scatter_plot(
        true_diff,
        pred_diff,
        "Actual total diff",
        "Predicted total diff",
        "Match total diff: actual vs predicted",
        plots_dir / "match_total_diff_scatter.png",
    )
    save_confusion_plot(
        (true_diff > 0).astype(int),
        (pred_diff > 0).astype(int),
        plots_dir / "winner_confusion_matrix.png",
    )

    metadata = {
        "data_path": DATA_PATH,
        "inference_state_path": INFERENCE_STATE_PATH,
        "train_ratio": TRAIN_RATIO,
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
            {
                sport: {
                    "base_feat_cols": sport_models[sport]["base_feat_cols"],
                    "resid_feat_cols": sport_models[sport]["resid_feat_cols"],
                    "total_feat_cols": sport_models[sport]["total_feat_cols"],
                }
                for sport in sport_models
            },
            f,
            indent=2,
        )
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(outdir / "inference_state.pkl", "wb") as f:
        pickle.dump(inference_state, f)

    print(f"\nSaved CatBoost package to: {outdir}")


if __name__ == "__main__":
    train_and_package()
