import json
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mean_absolute_error

SPORTS = ["TT", "BD", "SQ", "TN"]

MAX_POINTS_PER_SPORT = 21
MAX_DIFF_PER_SPORT = 21
MAX_TOTAL_PER_SPORT = 42
BASE = 0.0

# -------------------- Config --------------------
DATA_PATH = "data/data.csv"
INFERENCE_STATE_PATH = "data/inference_state.pkl"
OUTPUT_DIR = "finished_models/player_embedding/artifacts/predictor_package"

TRAIN_RATIO = 0.8
VAL_RATIO_WITHIN_TRAIN = 0.125  # 70/10/20 overall when TRAIN_RATIO=0.8

PREDICT_TENNIS_INDEPENDENTLY = True
SEED = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 16
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.15

EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

DEMO_PLAYER1 = "zain magdon-ismail"
DEMO_PLAYER2 = "patrick moran"

ALPHAS = {
    "TT": 0.060,
    "BD": 0.080,
    "SQ": 0.060,
    "TN": 0.040,
}


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["p1_name", "p2_name", "p1_key", "p2_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def load_inference_state(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
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


def get_compact_feature_cols(df: pd.DataFrame, sport: str) -> List[str]:
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
        f"{sport}_p1_long_n",
        f"{sport}_p2_long_n",
        f"{sport}_long_n_diff_p1_p2",
        f"{sport}_p1_long_diff_mean",
        f"{sport}_p2_long_diff_mean",
        f"{sport}_long_diff_mean_diff_p1_p2",
        f"{sport}_p1_long_total_mean",
        f"{sport}_p2_long_total_mean",
        f"{sport}_long_total_mean_diff_p1_p2",
        f"{sport}_p1_long_winrate",
        f"{sport}_p2_long_winrate",
        f"{sport}_long_winrate_diff_p1_p2",
    ]
    return [c for c in wanted if c in df.columns]


def build_player_index(df: pd.DataFrame) -> Dict[str, int]:
    players = sorted(
        set(df["p1_key"].dropna().tolist())
        | set(df["p2_key"].dropna().tolist())
    )
    return {p: i + 1 for i, p in enumerate(players)}


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
) -> Tuple[int, int]:
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
        return 21, round_and_clip_score(loser_score, 20)
    loser_from_raw = min(raw_s1, 20.0)
    loser_score = 0.5 * loser_from_raw + 0.5 * margin_loser
    return round_and_clip_score(loser_score, 20), 21


def decode_tennis_score(
    pred_diff: float, pred_total: float, running_diff_before_tn: int
) -> Tuple[int, int]:
    pred_diff = float(
        np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    )
    pred_total = float(np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT))
    raw_s1, raw_s2 = reconstruct_scores(pred_diff, pred_total)
    raw_s1 = round_and_clip_score(raw_s1, 21)
    raw_s2 = round_and_clip_score(raw_s2, 21)

    if running_diff_before_tn == 0:
        if pred_diff >= 0:
            return max(1, raw_s1), min(raw_s2, max(0, max(1, raw_s1) - 1))
        return min(raw_s1, max(0, max(1, raw_s2) - 1)), max(1, raw_s2)

    if running_diff_before_tn > 0:
        p2_needed = running_diff_before_tn + 1
        if pred_diff >= 0:
            return 1, 0
        s2 = min(21, max(p2_needed, raw_s2))
        s1 = max(0, min(21, int(round(s2 + pred_diff))))
        if s1 >= s2:
            s1 = max(0, s2 - 1)
        if (s2 - s1) >= p2_needed:
            return s1, s2
        return max(0, 21 - p2_needed), 21

    p1_needed = -running_diff_before_tn + 1
    if pred_diff < 0:
        return 0, 1
    s1 = min(21, max(p1_needed, raw_s1))
    s2 = max(0, min(21, int(round(s1 - pred_diff))))
    if s2 >= s1:
        s2 = max(0, s1 - 1)
    if (s1 - s2) >= p1_needed:
        return s1, s2
    return 21, max(0, 21 - p1_needed)


def build_synthetic_match_row(
    inference_state: dict,
    player1: str,
    player2: str,
    feature_cols: List[str],
) -> pd.Series:
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

    return pd.Series({c: row.get(c, 0.0) for c in feature_cols})


class SportEmbeddingNet(nn.Module):
    def __init__(self, n_players: int, n_features: int, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(n_players + 1, embed_dim)
        input_dim = n_features + embed_dim * 4

        layers = []
        prev = input_dim
        for h in HIDDEN_DIMS:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)

    def forward(self, p1_idx, p2_idx, x_num):
        e1 = self.embed(p1_idx)
        e2 = self.embed(p2_idx)
        x = torch.cat([e1, e2, e1 - e2, e1 * e2, x_num], dim=1)
        z = self.backbone(x)
        return self.head(z)


def to_tensor_batch(
    df_slice: pd.DataFrame,
    feature_cols: List[str],
    player_to_idx: Dict[str, int],
    feat_mean: pd.Series,
    feat_std: pd.Series,
):
    p1_idx = torch.tensor(
        [player_to_idx.get(x, 0) for x in df_slice["p1_key"].tolist()],
        dtype=torch.long,
        device=DEVICE,
    )
    p2_idx = torch.tensor(
        [player_to_idx.get(x, 0) for x in df_slice["p2_key"].tolist()],
        dtype=torch.long,
        device=DEVICE,
    )

    x_num = df_slice[feature_cols].copy()
    x_num = x_num.fillna(feat_mean)
    x_num = (x_num - feat_mean) / feat_std
    x_num = x_num.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x_num_t = torch.tensor(x_num.values, dtype=torch.float32, device=DEVICE)
    return p1_idx, p2_idx, x_num_t


def iterate_minibatches(n, batch_size):
    idx = np.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


@dataclass
class PredictorPackage:
    models: Dict[str, nn.Module]
    model_meta: Dict[str, dict]
    feature_cols_by_sport: Dict[str, List[str]]
    player_to_idx: Dict[str, int]
    inference_state: dict

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        bundle = torch.load(
            directory / "player_embedding_package.pt",
            map_location=DEVICE,
            weights_only=False,
        )

        with open(directory / "inference_state.pkl", "rb") as f:
            inference_state = pickle.load(f)

        models = {}
        for sport in bundle["sports"]:
            cfg = bundle["config"]
            feat_cols = bundle["feature_cols_by_sport"][sport]
            model = SportEmbeddingNet(
                n_players=bundle["n_players"],
                n_features=len(feat_cols),
                embed_dim=cfg["embed_dim"],
            ).to(DEVICE)
            model.load_state_dict(bundle["state_dicts"][sport])
            model.eval()
            models[sport] = model

        return cls(
            models=models,
            model_meta=bundle["model_meta"],
            feature_cols_by_sport=bundle["feature_cols_by_sport"],
            player_to_idx=bundle["player_to_idx"],
            inference_state=inference_state,
        )

    def get_player_key(self, player_name: str) -> str:
        player_name = player_name.strip().lower()
        if player_name not in self.inference_state["player_states_by_name"]:
            raise ValueError(f"No history found for player '{player_name}'")
        return player_name

    def predict_pair(self, player1: str, player2: str) -> dict:
        total_p1 = 0
        total_p2 = 0
        sports_out = {}

        for sport in SPORTS:
            if sport not in self.models:
                continue

            feat_cols = self.feature_cols_by_sport[sport]
            row = build_synthetic_match_row(
                self.inference_state, player1, player2, feat_cols
            )
            one_row = pd.DataFrame([row.to_dict()]).fillna(0.0)
            one_row["p1_key"] = self.get_player_key(player1)
            one_row["p2_key"] = self.get_player_key(player2)

            meta = self.model_meta[sport]
            feat_mean = pd.Series(meta["feat_mean"])
            feat_std = pd.Series(meta["feat_std"])

            p1_idx, p2_idx, x_num = to_tensor_batch(
                one_row,
                feat_cols,
                self.player_to_idx,
                feat_mean,
                feat_std,
            )

            with torch.no_grad():
                out = self.models[sport](p1_idx, p2_idx, x_num).cpu().numpy()[0]

            pred_diff = float(np.clip(out[0], -21, 21))
            pred_total = float(np.clip(out[1], 0, 42))

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
                "pred_diff": pred_diff,
                "pred_total": pred_total,
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


def save_loss_plot(history_train, history_val, title, outpath):
    plt.figure(figsize=(6, 4))
    plt.plot(history_train, label="train")
    plt.plot(history_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def train_one_sport_model(
    sport: str,
    df: pd.DataFrame,
    rows: np.ndarray,
    feature_cols: List[str],
    player_to_idx: Dict[str, int],
    split_train: int,
    split_val: int,
):
    diff_col = f"{sport}_y_diff"
    total_col = f"{sport}_y_total"

    train_mask = rows < split_train
    val_mask = (rows >= split_train) & (rows < split_val)
    test_mask = rows >= split_val

    df_train = df.loc[rows[train_mask]].copy()
    df_val = df.loc[rows[val_mask]].copy()
    df_test = df.loc[rows[test_mask]].copy()

    feat_mean = df_train[feature_cols].mean(numeric_only=True)
    feat_std = df_train[feature_cols].std(numeric_only=True).replace(0, 1.0)

    model = SportEmbeddingNet(
        max(player_to_idx.values()) if player_to_idx else 0,
        len(feature_cols),
        EMBED_DIM,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.SmoothL1Loss()

    y_train = torch.tensor(
        df_train[[diff_col, total_col]].values,
        dtype=torch.float32,
        device=DEVICE,
    )
    y_val = torch.tensor(
        df_val[[diff_col, total_col]].values,
        dtype=torch.float32,
        device=DEVICE,
    )

    p1_train, p2_train, x_train = to_tensor_batch(
        df_train, feature_cols, player_to_idx, feat_mean, feat_std
    )
    p1_val, p2_val, x_val = to_tensor_batch(
        df_val, feature_cols, player_to_idx, feat_mean, feat_std
    )

    best_state = None
    best_val = float("inf")
    patience = 0
    train_history = []
    val_history = []

    for _ in range(EPOCHS):
        model.train()
        running = 0.0
        for batch_idx in iterate_minibatches(len(df_train), BATCH_SIZE):
            optimizer.zero_grad()
            out = model(
                p1_train[batch_idx], p2_train[batch_idx], x_train[batch_idx]
            )
            loss = loss_fn(out[:, 0], y_train[batch_idx, 0]) + loss_fn(
                out[:, 1], y_train[batch_idx, 1]
            )
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(batch_idx)

        train_epoch_loss = running / max(1, len(df_train))
        train_history.append(train_epoch_loss)

        model.eval()
        with torch.no_grad():
            val_out = model(p1_val, p2_val, x_val)
            val_loss = float(
                loss_fn(val_out[:, 0], y_val[:, 0]).item()
                + loss_fn(val_out[:, 1], y_val[:, 1]).item()
            )
        val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    p1_test, p2_test, x_test = to_tensor_batch(
        df_test, feature_cols, player_to_idx, feat_mean, feat_std
    )
    with torch.no_grad():
        out = model(p1_test, p2_test, x_test).cpu().numpy()

    pred_diff = np.clip(out[:, 0], -21, 21)
    pred_total = np.clip(out[:, 1], 0, 42)

    y_diff_test = df_test[diff_col].values.astype(float)
    y_total_test = df_test[total_col].values.astype(float)

    return model, {
        "feat_mean": feat_mean.to_dict(),
        "feat_std": feat_std.to_dict(),
        "rows_test": df_test.index.values.tolist(),
        "pred_diff": pred_diff.tolist(),
        "pred_total": pred_total.tolist(),
        "diff_mae": float(mean_absolute_error(y_diff_test, pred_diff)),
        "total_mae": float(mean_absolute_error(y_total_test, pred_total)),
        "train_n": int(len(df_train)),
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "train_history": train_history,
        "val_history": val_history,
    }


def main():
    set_seed(SEED)
    outdir = ensure_dir(OUTPUT_DIR)
    plots_dir = ensure_dir(outdir / "plots")

    df = read_data(DATA_PATH)
    inference_state = load_inference_state(INFERENCE_STATE_PATH)
    player_to_idx = build_player_index(df)

    n = len(df)
    split_train = int(n * (TRAIN_RATIO - TRAIN_RATIO * VAL_RATIO_WITHIN_TRAIN))
    split_val = int(n * TRAIN_RATIO)

    print(f"Device: {DEVICE}")
    print(f"Rows: {n}")
    print(f"Train end: {split_train}")
    print(f"Val end:   {split_val}")

    models = {}
    model_meta = {}
    feature_cols_by_sport = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))

    for sport in SPORTS:
        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"
        feat_cols = get_compact_feature_cols(df, sport)

        mask = df[diff_col].notna() & df[total_col].notna()
        rows = np.where(mask.values)[0]
        if len(rows) < 50:
            print(f"Skipping {sport}: not enough usable rows")
            continue

        model, meta = train_one_sport_model(
            sport,
            df,
            rows,
            feat_cols,
            player_to_idx,
            split_train,
            split_val,
        )

        print(f"\n{sport} TEST RESULTS")
        print("Diff MAE:", meta["diff_mae"])
        print("Total MAE:", meta["total_mae"])

        save_scatter_plot(
            df.loc[meta["rows_test"], diff_col].values.astype(float),
            np.array(meta["pred_diff"]),
            "Actual diff",
            "Predicted diff",
            f"{sport} diff: actual vs predicted",
            plots_dir / f"{sport.lower()}_diff_scatter.png",
        )
        save_scatter_plot(
            df.loc[meta["rows_test"], total_col].values.astype(float),
            np.array(meta["pred_total"]),
            "Actual total",
            "Predicted total",
            f"{sport} total: actual vs predicted",
            plots_dir / f"{sport.lower()}_total_scatter.png",
        )
        save_loss_plot(
            meta["train_history"],
            meta["val_history"],
            f"{sport} train/val loss",
            plots_dir / f"{sport.lower()}_loss.png",
        )

        models[sport] = model
        model_meta[sport] = meta
        feature_cols_by_sport[sport] = feat_cols

        for i, row_idx in enumerate(meta["rows_test"]):
            pdiff = meta["pred_diff"][i]
            ptotal = meta["pred_total"][i]
            if sport in ["TT", "BD", "SQ"]:
                s1_hat, s2_hat = decode_full_game_score(pdiff, ptotal)
            else:
                if PREDICT_TENNIS_INDEPENDENTLY:
                    s1_hat, s2_hat = decode_full_game_score(pdiff, ptotal)
                else:
                    s1_hat, s2_hat = decode_tennis_score(
                        pdiff,
                        ptotal,
                        int(
                            round(
                                total_pred_p1[row_idx] - total_pred_p2[row_idx]
                            )
                        ),
                    )
            total_pred_p1[row_idx] += s1_hat
            total_pred_p2[row_idx] += s2_hat

    true_diff = []
    pred_diff = []
    for i in range(split_val, len(df)):
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

    bundle = {
        "sports": list(models.keys()),
        "feature_cols_by_sport": feature_cols_by_sport,
        "player_to_idx": player_to_idx,
        "n_players": max(player_to_idx.values()) if player_to_idx else 0,
        "config": {
            "embed_dim": EMBED_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
        },
        "model_meta": model_meta,
        "state_dicts": {s: models[s].state_dict() for s in models},
        "metrics": {
            "sport_metrics": {
                s: {
                    "diff_mae": model_meta[s]["diff_mae"],
                    "total_mae": model_meta[s]["total_mae"],
                    "train_n": model_meta[s]["train_n"],
                    "val_n": model_meta[s]["val_n"],
                    "test_n": model_meta[s]["test_n"],
                    "n_features": len(feature_cols_by_sport[s]),
                }
                for s in model_meta
            },
            "match_metrics": {
                "total_diff_mae": match_mae,
                "winner_accuracy": winner_acc,
            },
        },
    }

    torch.save(bundle, outdir / "player_embedding_package.pt")

    with open(outdir / "inference_state.pkl", "wb") as f:
        pickle.dump(inference_state, f)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    predictor = PredictorPackage.load(outdir)
    demo = predictor.predict_pair(DEMO_PLAYER1, DEMO_PLAYER2)
    with open(outdir / "demo_prediction.json", "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2)

    print(f"\nSaved package to: {outdir}")


if __name__ == "__main__":
    main()
