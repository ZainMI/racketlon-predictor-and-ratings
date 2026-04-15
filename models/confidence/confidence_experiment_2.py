import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KDTree


SPORTS = ["TT", "BD", "SQ", "TN"]

DATA_PATH = "data/data.csv"
OUTPUT_DIR = "models/confidence/artifacts/experiment_2"

FEATURE_COLS = [
    "TT_rating_diff",
    "BD_rating_diff",
    "SQ_rating_diff",
    "TN_rating_diff",
]
TARGET_COL = "y_total_diff"

TRAIN_RATIO = 0.8
K = 9


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


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def weighted_mean(values: np.ndarray, distances: np.ndarray) -> float:
    w = 1.0 / (distances + 1e-9)
    return float(np.sum(w * values) / np.sum(w))


def density_confidence(distances: np.ndarray) -> float:
    if len(distances) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(distances)))


def consistency_confidence(targets: np.ndarray) -> float:
    if len(targets) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.std(targets)))


def combined_confidence(
    density_conf: float,
    consistency_conf: float,
    density_weight: float = 0.5,
) -> float:
    density_weight = float(np.clip(density_weight, 0.0, 1.0))
    return (
        density_weight * density_conf
        + (1.0 - density_weight) * consistency_conf
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


def run_knn_query(
    tree: KDTree, train_y: np.ndarray, q: np.ndarray, k: int
) -> dict:
    dists, idx = tree.query(q.reshape(1, -1), k=k)
    dists = dists[0]
    idx = idx[0]

    nn_targets = train_y[idx]
    pred = weighted_mean(nn_targets, dists)

    d_conf = density_confidence(dists)
    c_conf = consistency_confidence(nn_targets)
    combo_conf = combined_confidence(d_conf, c_conf, density_weight=0.5)

    return {
        "pred": float(pred),
        "density_conf": float(d_conf),
        "consistency_conf": float(c_conf),
        "combined_conf": float(combo_conf),
        "mean_neighbor_distance": float(np.mean(dists)),
        "neighbor_target_std": float(np.std(nn_targets)),
        "neighbor_targets_mean": float(np.mean(nn_targets)),
        "neighbor_count": int(len(nn_targets)),
    }


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def run_experiment():
    outdir = ensure_dir(OUTPUT_DIR)

    df = read_data(DATA_PATH)
    work = (
        df[FEATURE_COLS + [TARGET_COL]].copy().dropna().reset_index(drop=True)
    )

    X = work[FEATURE_COLS].astype(float).values
    y = work[TARGET_COL].astype(float).values

    split = int(len(work) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Rows used: {len(work)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows:  {len(X_test)}")
    print(f"Features:   {FEATURE_COLS}")
    print(f"k:          {K}")

    t0 = time.perf_counter()
    tree = KDTree(X_train)
    build_ms = 1000.0 * (time.perf_counter() - t0)

    preds = []
    abs_err = []

    density_conf_list = []
    consistency_conf_list = []
    combined_conf_list = []

    mean_dist_list = []
    target_std_list = []
    local_mean_target_list = []

    t0 = time.perf_counter()
    for q, y_true in zip(X_test, y_test):
        out = run_knn_query(tree, y_train, q, K)

        pred = out["pred"]
        err = abs(pred - y_true)

        preds.append(pred)
        abs_err.append(err)

        density_conf_list.append(out["density_conf"])
        consistency_conf_list.append(out["consistency_conf"])
        combined_conf_list.append(out["combined_conf"])

        mean_dist_list.append(out["mean_neighbor_distance"])
        target_std_list.append(out["neighbor_target_std"])
        local_mean_target_list.append(out["neighbor_targets_mean"])

    query_ms = 1000.0 * (time.perf_counter() - t0)

    preds = np.asarray(preds, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)

    density_conf_list = np.asarray(density_conf_list, dtype=float)
    consistency_conf_list = np.asarray(consistency_conf_list, dtype=float)
    combined_conf_list = np.asarray(combined_conf_list, dtype=float)

    mean_dist_list = np.asarray(mean_dist_list, dtype=float)
    target_std_list = np.asarray(target_std_list, dtype=float)
    local_mean_target_list = np.asarray(local_mean_target_list, dtype=float)

    overall_mae = float(mean_absolute_error(y_test, preds))
    winner_acc = float(((preds > 0) == (y_test > 0)).mean())

    density_q = quartile_error_summary(density_conf_list, abs_err)
    consistency_q = quartile_error_summary(consistency_conf_list, abs_err)
    combined_q = quartile_error_summary(combined_conf_list, abs_err)

    density_s = spearman_corr(density_conf_list, abs_err)
    consistency_s = spearman_corr(consistency_conf_list, abs_err)
    combined_s = spearman_corr(combined_conf_list, abs_err)

    mean_dist_s = spearman_corr(mean_dist_list, abs_err)
    target_std_s = spearman_corr(target_std_list, abs_err)

    print("\n=== PREDICTION ===")
    print("Weighted k-NN Total Diff MAE:", overall_mae)
    print("Weighted k-NN Winner Accuracy:", winner_acc)

    print("\n=== RUNTIME ===")
    print("Tree build time (ms):", build_ms)
    print("Total query time (ms):", query_ms)
    print("Avg query time (ms):", query_ms / max(len(X_test), 1))

    print("\n=== CONFIDENCE: DENSITY ONLY ===")
    print("Spearman(confidence, abs_error):", density_s)
    print("Low-confidence quartile MAE:", density_q["low_q_mae"])
    print("High-confidence quartile MAE:", density_q["high_q_mae"])

    print("\n=== CONFIDENCE: CONSISTENCY ONLY ===")
    print("Spearman(confidence, abs_error):", consistency_s)
    print("Low-confidence quartile MAE:", consistency_q["low_q_mae"])
    print("High-confidence quartile MAE:", consistency_q["high_q_mae"])

    print("\n=== CONFIDENCE: COMBINED ===")
    print("Spearman(confidence, abs_error):", combined_s)
    print("Low-confidence quartile MAE:", combined_q["low_q_mae"])
    print("High-confidence quartile MAE:", combined_q["high_q_mae"])

    print("\n=== RAW LOCAL SIGNALS ===")
    print("Spearman(mean_neighbor_distance, abs_error):", mean_dist_s)
    print("Spearman(neighbor_target_std, abs_error):", target_std_s)

    results_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": preds,
            "abs_error": abs_err,
            "density_confidence": density_conf_list,
            "consistency_confidence": consistency_conf_list,
            "combined_confidence": combined_conf_list,
            "mean_neighbor_distance": mean_dist_list,
            "neighbor_target_std": target_std_list,
            "local_mean_target": local_mean_target_list,
        }
    )
    results_df.to_csv(outdir / "experiment_2_results.csv", index=False)

    summary = {
        "data_path": DATA_PATH,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_ratio": TRAIN_RATIO,
        "k": K,
        "n_rows_used": int(len(work)),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "prediction_metrics": {
            "total_diff_mae": overall_mae,
            "winner_accuracy": winner_acc,
        },
        "runtime_ms": {
            "tree_build_ms": build_ms,
            "total_query_ms": query_ms,
            "avg_query_ms": query_ms / max(len(X_test), 1),
        },
        "confidence_metrics": {
            "density_only": {
                "spearman_conf_vs_abs_error": density_s,
                **density_q,
            },
            "consistency_only": {
                "spearman_conf_vs_abs_error": consistency_s,
                **consistency_q,
            },
            "combined": {
                "spearman_conf_vs_abs_error": combined_s,
                **combined_q,
            },
            "raw_local_signals": {
                "spearman_mean_neighbor_distance_vs_abs_error": mean_dist_s,
                "spearman_neighbor_target_std_vs_abs_error": target_std_s,
            },
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {outdir}")


if __name__ == "__main__":
    run_experiment()
