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
OUTPUT_DIR = "finished_models/catboost/artifacts/predictor_package"
TRAIN_RATIO = 0.8
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


def get_feature_columns(df):
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
        if c.endswith("_y_diff") or c.endswith("_y_total"):
            continue
        if c.startswith("has_"):
            continue
        if "_pred_diff" in c:
            continue
        feature_cols.append(c)
    return sorted(feature_cols)


def reconstruct_scores(pred_diff, pred_total):
    pred_diff = np.clip(pred_diff, -MAX_DIFF_PER_SPORT, MAX_DIFF_PER_SPORT)
    pred_total = np.clip(pred_total, 0, MAX_TOTAL_PER_SPORT)
    s1 = 0.5 * (pred_total + pred_diff)
    s2 = 0.5 * (pred_total - pred_diff)
    s1 = np.clip(s1, 0, MAX_POINTS_PER_SPORT)
    s2 = np.clip(s2, 0, MAX_POINTS_PER_SPORT)
    return s1, s2


def round_and_clip_score(x, upper=21):
    return max(0, min(upper, int(round(float(x)))))


def decode_full_game_score(pred_diff, pred_total):
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


def decode_tennis_score(pred_diff, pred_total, running_diff_before_tn):
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


def orient_row_to_player_as_p1(row, player_name):
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
        for a, b in [
            (f"{sport}_rating_p1", f"{sport}_rating_p2"),
            (f"{sport}_games_p1", f"{sport}_games_p2"),
            (f"{sport}_snapshot_rating_p1", f"{sport}_snapshot_rating_p2"),
            (f"{sport}_snapshot_p1_found", f"{sport}_snapshot_p2_found"),
        ]:
            if a in out.index and b in out.index:
                out[a], out[b] = out[b], out[a]

        for col in [
            f"{sport}_rating_diff",
            f"{sport}_games_diff",
            f"{sport}_h2h_avg_diff_p1",
            f"{sport}_snapshot_rating_diff",
        ]:
            if col in out.index and pd.notna(out[col]):
                out[col] = -out[col]

        wr = f"{sport}_h2h_winrate_p1"
        if wr in out.index and pd.notna(out[wr]):
            out[wr] = 1.0 - out[wr]

    return out


def get_latest_player_state(df, player_name):
    player_name = player_name.strip().lower()
    mask = (df["p1_name"] == player_name) | (df["p2_name"] == player_name)
    candidates = df[mask]
    if candidates.empty:
        raise ValueError(f"No history found for player '{player_name}'")
    row = candidates.iloc[-1].copy()
    return orient_row_to_player_as_p1(row, player_name)


def get_latest_pair_h2h_row(df, player1, player2):
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()
    mask_direct = (df["p1_name"] == p1) & (df["p2_name"] == p2)
    mask_reverse = (df["p1_name"] == p2) & (df["p2_name"] == p1)
    candidates = df[mask_direct | mask_reverse]
    if candidates.empty:
        return None
    row = candidates.iloc[-1].copy()
    return orient_row_to_player_as_p1(row, p1)


def build_synthetic_match_row(df, player1, player2, feature_cols):
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

    return pd.Series({col: row.get(col, np.nan) for col in feature_cols})


@dataclass
class PredictorPackage:
    models: dict
    feature_cols: list
    df: pd.DataFrame

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with open(directory / "models.pkl", "rb") as f:
            models = pickle.load(f)
        with open(directory / "feature_cols.json", "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        df = pd.read_pickle(directory / "data_snapshot.pkl")
        return cls(models=models, feature_cols=feature_cols, df=df)

    def predict_pair(self, player1, player2):
        row = build_synthetic_match_row(
            self.df, player1, player2, self.feature_cols
        )

        total_p1 = 0
        total_p2 = 0
        sports_out = {}

        for sport in SPORTS:
            model_diff = self.models[sport]["model_diff"]
            model_total = self.models[sport]["model_total"]
            X_one = pd.DataFrame([row[self.feature_cols].to_dict()])

            pred_diff = float(np.clip(model_diff.predict(X_one)[0], -21, 21))
            pred_total = float(np.clip(model_total.predict(X_one)[0], 0, 42))

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


def train_and_package():
    outdir = ensure_dir(OUTPUT_DIR)
    plots_dir = ensure_dir(outdir / "plots")

    df = read_data(DATA_PATH)
    split = int(len(df) * TRAIN_RATIO)
    feature_cols = get_feature_columns(df)

    print(f"Using {len(feature_cols)} features")
    print(
        "Dropped all *_pred_diff, snapshot_total_pred_diff, has_*, and target columns."
    )

    sport_models = {}
    total_pred_p1 = np.zeros(len(df))
    total_pred_p2 = np.zeros(len(df))
    metrics = {}

    for sport in SPORTS:
        diff_col = f"{sport}_y_diff"
        total_col = f"{sport}_y_total"
        mask = df[diff_col].notna() & df[total_col].notna()
        rows = np.where(mask.values)[0]

        X = df.loc[rows, feature_cols].copy()
        y_diff = df.loc[rows, diff_col].astype(float).values
        y_total = df.loc[rows, total_col].astype(float).values

        train_mask = rows < split
        test_mask = rows >= split

        X_train = X.iloc[train_mask]
        X_test = X.iloc[test_mask]
        y_diff_train = y_diff[train_mask]
        y_diff_test = y_diff[test_mask]
        y_total_train = y_total[train_mask]
        y_total_test = y_total[test_mask]

        model_diff = CatBoostRegressor(**CATBOOST_PARAMS)
        model_total = CatBoostRegressor(**CATBOOST_PARAMS)

        model_diff.fit(X_train, y_diff_train)
        model_total.fit(X_train, y_total_train)

        pred_diff = np.clip(model_diff.predict(X_test), -21, 21)
        pred_total = np.clip(model_total.predict(X_test), 0, 42)

        for i, row_idx in enumerate(rows[test_mask]):
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
                    s1_hat, s2_hat = decode_tennis_score(
                        pred_diff[i],
                        pred_total[i],
                        int(
                            round(
                                total_pred_p1[row_idx] - total_pred_p2[row_idx]
                            )
                        ),
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
            model_diff,
            feature_cols,
            f"{sport} diff feature importance",
            plots_dir / f"{sport.lower()}_diff_feature_importance.png",
        )
        save_feature_importance_plot(
            model_total,
            feature_cols,
            f"{sport} total feature importance",
            plots_dir / f"{sport.lower()}_total_feature_importance.png",
        )

        sport_models[sport] = {
            "model_diff": model_diff,
            "model_total": model_total,
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
        "train_ratio": TRAIN_RATIO,
        "predict_tennis_independently": PREDICT_TENNIS_INDEPENDENTLY,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "sport_metrics": metrics,
        "match_metrics": {
            "total_diff_mae": match_mae,
            "winner_accuracy": winner_acc,
        },
    }

    with open(outdir / "models.pkl", "wb") as f:
        pickle.dump(sport_models, f)
    with open(outdir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    df.to_pickle(outdir / "data_snapshot.pkl")

    predictor = PredictorPackage(
        models=sport_models, feature_cols=feature_cols, df=df
    )
    demo = predictor.predict_pair("zain magdon-ismail", "patrick moran")
    with open(outdir / "demo_prediction.json", "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2)

    print(f"\nSaved package to: {outdir}")


if __name__ == "__main__":
    train_and_package()
