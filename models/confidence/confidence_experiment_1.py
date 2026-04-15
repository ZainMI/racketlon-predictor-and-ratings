import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KDTree


SPORTS = ["TT", "BD", "SQ", "TN"]

DATA_PATH = "data/data.csv"
OUTPUT_DIR = "models/confidence/artifacts/experiment_1"

FEATURE_COLS = [
    "TT_rating_diff",
    "BD_rating_diff",
    "SQ_rating_diff",
    "TN_rating_diff",
]
TARGET_COL = "y_total_diff"

TRAIN_RATIO = 0.8
K = 9

# If None, radius is chosen automatically from train-set neighbor distances.
RADIUS = None


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
# Confidence + prediction helpers
# -------------------------------------------------
def weighted_mean(values: np.ndarray, distances: np.ndarray) -> float:
    w = 1.0 / (distances + 1e-9)
    return float(np.sum(w * values) / np.sum(w))


def knn_confidence_from_distances(distances: np.ndarray) -> float:
    if len(distances) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(distances)))


def radius_confidence_from_count(count: int, scale: float = 10.0) -> float:
    # smooth density-style confidence in [0,1)
    return float(count / (count + scale))


def spearman_corr(x, y) -> float:
    s = pd.Series(x).corr(pd.Series(y), method="spearman")
    return float(s) if pd.notna(s) else np.nan


def quartile_error_summary(confidence: np.ndarray, abs_err: np.ndarray):
    order = np.argsort(confidence)
    q = len(order) // 4

    if q == 0:
        return {
            "low_q_mae": np.nan,
            "high_q_mae": np.nan,
            "gap_high_minus_low": np.nan,
        }

    low_idx = order[:q]
    high_idx = order[-q:]

    low_mae = float(np.mean(abs_err[low_idx]))
    high_mae = float(np.mean(abs_err[high_idx]))

    return {
        "low_q_mae": low_mae,
        "high_q_mae": high_mae,
        "gap_high_minus_low": high_mae - low_mae,
    }


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def run_experiment():
    outdir = ensure_dir(OUTPUT_DIR)

    df = read_data(DATA_PATH)

    keep = FEATURE_COLS + [TARGET_COL]
    work = df[keep].copy().dropna().reset_index(drop=True)

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

    # Build tree
    t0 = time.perf_counter()
    tree = KDTree(X_train)
    build_ms = 1000.0 * (time.perf_counter() - t0)

    # Pick a radius automatically from the train set if not provided.
    # We use the median distance to the k-th nearest neighbor among train points.
    if RADIUS is None:
        # query train against itself, need k+1 because first neighbor is itself
        d_train, _ = tree.query(X_train, k=min(K + 1, len(X_train)))
        kth = d_train[:, -1]
        radius = float(np.median(kth))
    else:
        radius = float(RADIUS)

    print(f"Tree build time (ms): {build_ms:.3f}")
    print(f"Radius used:          {radius:.6f}")

    preds = []
    abs_err = []

    knn_conf = []
    radius_conf = []

    knn_neighbor_mean_dist = []
    radius_neighbor_count = []
    radius_neighbor_target_std = []

    t0 = time.perf_counter()
    for q, y_true in zip(X_test, y_test):
        # -----------------------------
        # k-NN query and weighted prediction
        # -----------------------------
        dists, idx = tree.query(q.reshape(1, -1), k=min(K, len(X_train)))
        dists = dists[0]
        idx = idx[0]

        nn_targets = y_train[idx]
        pred = weighted_mean(nn_targets, dists)

        preds.append(pred)
        err = abs(pred - y_true)
        abs_err.append(err)

        knn_conf.append(knn_confidence_from_distances(dists))
        knn_neighbor_mean_dist.append(float(np.mean(dists)))

        # -----------------------------
        # radius query for density confidence
        # -----------------------------
        ind, dist = tree.query_radius(
            q.reshape(1, -1), r=radius, return_distance=True, sort_results=True
        )
        ridx = ind[0]
        rdists = dist[0]

        count = int(len(ridx))
        radius_neighbor_count.append(count)

        if count > 0:
            local_targets = y_train[ridx]
            local_std = float(np.std(local_targets))
        else:
            local_std = np.nan

        radius_neighbor_target_std.append(local_std)
        radius_conf.append(radius_confidence_from_count(count, scale=float(K)))

    query_ms = 1000.0 * (time.perf_counter() - t0)

    preds = np.asarray(preds, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)
    knn_conf = np.asarray(knn_conf, dtype=float)
    radius_conf = np.asarray(radius_conf, dtype=float)
    knn_neighbor_mean_dist = np.asarray(knn_neighbor_mean_dist, dtype=float)
    radius_neighbor_count = np.asarray(radius_neighbor_count, dtype=float)
    radius_neighbor_target_std = np.asarray(
        radius_neighbor_target_std, dtype=float
    )

    overall_mae = float(mean_absolute_error(y_test, preds))
    winner_acc = float(((preds > 0) == (y_test > 0)).mean())

    # Confidence-quality comparisons
    knn_q = quartile_error_summary(knn_conf, abs_err)
    radius_q = quartile_error_summary(radius_conf, abs_err)

    # Since larger confidence should mean smaller error,
    # a good sign is a NEGATIVE correlation between confidence and abs error.
    knn_spearman = spearman_corr(knn_conf, abs_err)
    radius_spearman = spearman_corr(radius_conf, abs_err)

    # Also report direct geometric signals
    mean_dist_spearman = spearman_corr(knn_neighbor_mean_dist, abs_err)
    count_spearman = spearman_corr(radius_neighbor_count, abs_err)

    print("\n=== PREDICTION ===")
    print("Weighted k-NN Total Diff MAE:", overall_mae)
    print("Weighted k-NN Winner Accuracy:", winner_acc)

    print("\n=== RUNTIME ===")
    print("Tree build time (ms):", build_ms)
    print("Total query time (ms):", query_ms)
    print("Avg query time (ms):", query_ms / max(len(X_test), 1))

    print("\n=== CONFIDENCE: K-NN DISTANCE ===")
    print("Spearman(confidence, abs_error):", knn_spearman)
    print("Low-confidence quartile MAE:", knn_q["low_q_mae"])
    print("High-confidence quartile MAE:", knn_q["high_q_mae"])

    print("\n=== CONFIDENCE: RADIUS DENSITY ===")
    print("Spearman(confidence, abs_error):", radius_spearman)
    print("Low-confidence quartile MAE:", radius_q["low_q_mae"])
    print("High-confidence quartile MAE:", radius_q["high_q_mae"])

    print("\n=== RAW GEOMETRIC SIGNALS ===")
    print("Spearman(mean_neighbor_distance, abs_error):", mean_dist_spearman)
    print("Spearman(radius_neighbor_count, abs_error):", count_spearman)

    results_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": preds,
            "abs_error": abs_err,
            "knn_confidence": knn_conf,
            "radius_confidence": radius_conf,
            "knn_mean_neighbor_distance": knn_neighbor_mean_dist,
            "radius_neighbor_count": radius_neighbor_count,
            "radius_neighbor_target_std": radius_neighbor_target_std,
        }
    )
    results_df.to_csv(outdir / "experiment_1_results.csv", index=False)

    summary = {
        "data_path": DATA_PATH,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_ratio": TRAIN_RATIO,
        "k": K,
        "radius": radius,
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
            "knn_distance_confidence": {
                "spearman_conf_vs_abs_error": knn_spearman,
                **knn_q,
            },
            "radius_density_confidence": {
                "spearman_conf_vs_abs_error": radius_spearman,
                **radius_q,
            },
            "raw_geometry": {
                "spearman_mean_neighbor_distance_vs_abs_error": mean_dist_spearman,
                "spearman_radius_neighbor_count_vs_abs_error": count_spearman,
            },
        },
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {outdir}")


if __name__ == "__main__":
    run_experiment()
