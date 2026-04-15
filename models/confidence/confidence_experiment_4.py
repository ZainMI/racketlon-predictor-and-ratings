import sys
from pathlib import Path

# add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KDTree

from models.catboost.catboost_fin import (
    PredictorPackage as CatBoostPredictorPackage,
)


SPORTS = ["TT", "BD", "SQ", "TN"]

DATA_PATH = "data/data.csv"
INFERENCE_STATE_PATH = "data/inference_state.pkl"
CATBOOST_PACKAGE_DIR = "models/catboost/artifacts/predictor_package"
OUTPUT_DIR = "models/confidence/artifacts/experiment_4"

TRAIN_RATIO = 0.8
K = 9

FEATURE_COLS = [
    "TT_rating_diff",
    "BD_rating_diff",
    "SQ_rating_diff",
    "TN_rating_diff",
    "TT_diff_mean_10_diff_p1_p2",
    "BD_diff_mean_10_diff_p1_p2",
    "SQ_diff_mean_10_diff_p1_p2",
    "TN_diff_mean_10_diff_p1_p2",
    "TT_resid_mean_10_diff_p1_p2",
    "BD_resid_mean_10_diff_p1_p2",
    "SQ_resid_mean_10_diff_p1_p2",
    "TN_resid_mean_10_diff_p1_p2",
]
TARGET_COL = "y_total_diff"


# -------------------------------------------------
# IO
# -------------------------------------------------
def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


# -------------------------------------------------
# Confidence helpers
# -------------------------------------------------
def consistency_confidence(targets: np.ndarray) -> float:
    if len(targets) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.std(targets)))


def density_confidence(distances: np.ndarray) -> float:
    if len(distances) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(distances)))


def combined_confidence(
    consistency_conf: float,
    density_conf: float,
    consistency_weight: float = 0.75,
) -> float:
    consistency_weight = float(np.clip(consistency_weight, 0.0, 1.0))
    return (
        consistency_weight * consistency_conf
        + (1.0 - consistency_weight) * density_conf
    )


def spearman_corr(x, y) -> float:
    s = pd.Series(x).corr(pd.Series(y), method="spearman")
    return float(s) if pd.notna(s) else np.nan


def quartile_error_summary(confidence: np.ndarray, abs_err: np.ndarray) -> dict:
    order = np.argsort(confidence)
    q = len(order) // 4

    if q == 0:
        return {
            "low_q_mae": np.nan,
            "high_q_mae": np.nan,
            "high_minus_low": np.nan,
        }

    low_idx = order[:q]
    high_idx = order[-q:]

    low_mae = float(np.mean(abs_err[low_idx]))
    high_mae = float(np.mean(abs_err[high_idx]))

    return {
        "low_q_mae": low_mae,
        "high_q_mae": high_mae,
        "high_minus_low": high_mae - low_mae,
    }


# -------------------------------------------------
# Query-building
# -------------------------------------------------
def build_query_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray([float(row[c]) for c in FEATURE_COLS], dtype=float)


def build_query_from_players(
    inference_state: dict, player1: str, player2: str
) -> np.ndarray:
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()

    player_states = inference_state["player_states_by_name"]
    if p1 not in player_states:
        raise ValueError(f"No history found for player '{player1}'")
    if p2 not in player_states:
        raise ValueError(f"No history found for player '{player2}'")

    s1 = player_states[p1]
    s2 = player_states[p2]

    vals = []

    for sport in SPORTS:
        r1 = float(s1.get(f"{sport}_rating_p1", 0.0))
        r2 = float(s2.get(f"{sport}_rating_p1", 0.0))
        vals.append(r1 - r2)

    for sport in SPORTS:
        vals.append(
            float(s1.get(f"{sport}_diff_mean_10_diff_p1_p2", 0.0))
            if f"{sport}_diff_mean_10_diff_p1_p2" in s1
            else float(s1.get(f"{sport}_p1_recent_diff_mean_10", 0.0))
            - float(s2.get(f"{sport}_p1_recent_diff_mean_10", 0.0))
        )

    for sport in SPORTS:
        vals.append(
            float(s1.get(f"{sport}_resid_mean_10_diff_p1_p2", 0.0))
            if f"{sport}_resid_mean_10_diff_p1_p2" in s1
            else float(s1.get(f"{sport}_p1_recent_resid_mean_10", 0.0))
            - float(s2.get(f"{sport}_p1_recent_resid_mean_10", 0.0))
        )

    return np.asarray(vals, dtype=float)


def knn_confidence_query(
    tree: KDTree, train_y: np.ndarray, q: np.ndarray, k: int
) -> dict:
    dists, idx = tree.query(q.reshape(1, -1), k=k)
    dists = dists[0]
    idx = idx[0]

    nn_targets = train_y[idx]

    consistency_conf = consistency_confidence(nn_targets)
    density_conf = density_confidence(dists)
    combo_conf = combined_confidence(
        consistency_conf=consistency_conf,
        density_conf=density_conf,
        consistency_weight=0.75,
    )

    return {
        "consistency_confidence": float(consistency_conf),
        "density_confidence": float(density_conf),
        "combined_confidence": float(combo_conf),
        "neighbor_target_std": float(np.std(nn_targets)),
        "neighbor_mean_distance": float(np.mean(dists)),
        "neighbor_targets_mean": float(np.mean(nn_targets)),
        "neighbor_targets": nn_targets,
        "neighbor_distances": dists,
    }


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def run_experiment():
    outdir = ensure_dir(OUTPUT_DIR)

    df = read_data(DATA_PATH)
    inference_state = load_inference_state(INFERENCE_STATE_PATH)
    catboost_predictor = CatBoostPredictorPackage.load(CATBOOST_PACKAGE_DIR)

    needed_cols = FEATURE_COLS + [TARGET_COL, "p1_name", "p2_name"]
    work = df[needed_cols].copy().dropna().reset_index(drop=True)

    split = int(len(work) * TRAIN_RATIO)
    train_df = work.iloc[:split].copy().reset_index(drop=True)
    test_df = work.iloc[split:].copy().reset_index(drop=True)

    X_train = train_df[FEATURE_COLS].astype(float).values
    y_train = train_df[TARGET_COL].astype(float).values

    y_test = test_df[TARGET_COL].astype(float).values

    print(f"Rows used: {len(work)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Features:   {FEATURE_COLS}")
    print(f"k:          {K}")

    t0 = time.perf_counter()
    tree = KDTree(X_train)
    build_ms = 1000.0 * (time.perf_counter() - t0)

    preds = []
    abs_err = []

    consistency_conf_list = []
    density_conf_list = []
    combined_conf_list = []

    target_std_list = []
    mean_dist_list = []
    local_mean_target_list = []

    t0 = time.perf_counter()
    for _, row in test_df.iterrows():
        p1 = row["p1_name"]
        p2 = row["p2_name"]
        y_true = float(row[TARGET_COL])

        pred_pack = catboost_predictor.predict_pair(p1, p2)
        pred_total_diff = float(pred_pack["total_diff"])

        q = build_query_from_row(row)
        conf_pack = knn_confidence_query(tree, y_train, q, K)

        err = abs(pred_total_diff - y_true)

        preds.append(pred_total_diff)
        abs_err.append(err)

        consistency_conf_list.append(conf_pack["consistency_confidence"])
        density_conf_list.append(conf_pack["density_confidence"])
        combined_conf_list.append(conf_pack["combined_confidence"])

        target_std_list.append(conf_pack["neighbor_target_std"])
        mean_dist_list.append(conf_pack["neighbor_mean_distance"])
        local_mean_target_list.append(conf_pack["neighbor_targets_mean"])

    query_ms = 1000.0 * (time.perf_counter() - t0)

    preds = np.asarray(preds, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)

    consistency_conf_list = np.asarray(consistency_conf_list, dtype=float)
    density_conf_list = np.asarray(density_conf_list, dtype=float)
    combined_conf_list = np.asarray(combined_conf_list, dtype=float)

    target_std_list = np.asarray(target_std_list, dtype=float)
    mean_dist_list = np.asarray(mean_dist_list, dtype=float)
    local_mean_target_list = np.asarray(local_mean_target_list, dtype=float)

    overall_mae = float(mean_absolute_error(y_test, preds))
    winner_acc = float(((preds > 0) == (y_test > 0)).mean())

    consistency_s = spearman_corr(consistency_conf_list, abs_err)
    density_s = spearman_corr(density_conf_list, abs_err)
    combined_s = spearman_corr(combined_conf_list, abs_err)

    consistency_q = quartile_error_summary(consistency_conf_list, abs_err)
    density_q = quartile_error_summary(density_conf_list, abs_err)
    combined_q = quartile_error_summary(combined_conf_list, abs_err)

    target_std_s = spearman_corr(target_std_list, abs_err)
    mean_dist_s = spearman_corr(mean_dist_list, abs_err)

    print("\n=== CATBOOST PREDICTION ===")
    print("Total Diff MAE:", overall_mae)
    print("Winner Accuracy:", winner_acc)

    print("\n=== RUNTIME ===")
    print("KD-tree build time (ms):", build_ms)
    print("Total query time (ms):", query_ms)
    print("Avg query time (ms):", query_ms / max(len(test_df), 1))

    print("\n=== CONFIDENCE: CONSISTENCY ONLY ===")
    print("Spearman(confidence, abs_error):", consistency_s)
    print("Low-confidence quartile MAE:", consistency_q["low_q_mae"])
    print("High-confidence quartile MAE:", consistency_q["high_q_mae"])

    print("\n=== CONFIDENCE: DENSITY ONLY ===")
    print("Spearman(confidence, abs_error):", density_s)
    print("Low-confidence quartile MAE:", density_q["low_q_mae"])
    print("High-confidence quartile MAE:", density_q["high_q_mae"])

    print("\n=== CONFIDENCE: COMBINED ===")
    print("Spearman(confidence, abs_error):", combined_s)
    print("Low-confidence quartile MAE:", combined_q["low_q_mae"])
    print("High-confidence quartile MAE:", combined_q["high_q_mae"])

    print("\n=== RAW LOCAL SIGNALS ===")
    print("Spearman(neighbor_target_std, abs_error):", target_std_s)
    print("Spearman(mean_neighbor_distance, abs_error):", mean_dist_s)

    results_df = pd.DataFrame(
        {
            "p1_name": test_df["p1_name"].values,
            "p2_name": test_df["p2_name"].values,
            "y_true": y_test,
            "y_pred_catboost": preds,
            "abs_error": abs_err,
            "consistency_confidence": consistency_conf_list,
            "density_confidence": density_conf_list,
            "combined_confidence": combined_conf_list,
            "neighbor_target_std": target_std_list,
            "mean_neighbor_distance": mean_dist_list,
            "local_mean_target": local_mean_target_list,
        }
    )
    results_df.to_csv(outdir / "experiment_4_results.csv", index=False)

    summary = {
        "data_path": DATA_PATH,
        "inference_state_path": INFERENCE_STATE_PATH,
        "catboost_package_dir": CATBOOST_PACKAGE_DIR,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_ratio": TRAIN_RATIO,
        "k": K,
        "n_rows_used": int(len(work)),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "prediction_metrics": {
            "catboost_total_diff_mae": overall_mae,
            "catboost_winner_accuracy": winner_acc,
        },
        "runtime_ms": {
            "tree_build_ms": build_ms,
            "total_query_ms": query_ms,
            "avg_query_ms": query_ms / max(len(test_df), 1),
        },
        "confidence_metrics": {
            "consistency_only": {
                "spearman_conf_vs_abs_error": consistency_s,
                **consistency_q,
            },
            "density_only": {
                "spearman_conf_vs_abs_error": density_s,
                **density_q,
            },
            "combined": {
                "spearman_conf_vs_abs_error": combined_s,
                **combined_q,
            },
            "raw_local_signals": {
                "spearman_neighbor_target_std_vs_abs_error": target_std_s,
                "spearman_mean_neighbor_distance_vs_abs_error": mean_dist_s,
            },
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {outdir}")


if __name__ == "__main__":
    run_experiment()
