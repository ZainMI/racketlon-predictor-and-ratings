# methods/method_gbdt.py
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

SPORTS = ["TT", "BD", "SQ", "TN"]
MAX_DIFF = 21.0

# -----------------------
# Feature selection
# -----------------------
DROP_ALWAYS = {
    "match_index",
    "datetime",
    "p1",
    "p2",
    "p1_key",
    "p2_key",
}


def is_target_col(c: str) -> bool:
    # adapt if your target names differ
    return c.endswith("_y_diff") or c in (
        "y_winner_p1",
        "y_total_p1",
        "y_total_p2",
    )


def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=["number"]).copy()
    # Drop targets if present
    X = X.drop(
        columns=[c for c in X.columns if is_target_col(c)], errors="ignore"
    )
    # Drop always
    X = X.drop(
        columns=[c for c in DROP_ALWAYS if c in X.columns], errors="ignore"
    )
    return X


def clip_diff(y: np.ndarray) -> np.ndarray:
    return np.clip(y, -MAX_DIFF, MAX_DIFF)


# -----------------------
# Training
# -----------------------
def train(df: pd.DataFrame):
    """
    Expects df to include per-sport targets like TT_y_diff, BD_y_diff, ...
    and a column y_winner_p1 (0/1) OR can be derived later.
    """
    # time-safe split (simple): last 20% by datetime
    df = df.sort_values("datetime").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    X_train = make_feature_matrix(train_df)
    X_test = make_feature_matrix(test_df)

    sport_models = {}
    for s in SPORTS:
        y_col = f"{s}_y_diff"
        if y_col not in df.columns:
            raise ValueError(f"Missing target column {y_col}")

        y_train = clip_diff(train_df[y_col].to_numpy(dtype=float))
        y_test = clip_diff(test_df[y_col].to_numpy(dtype=float))

        # Good default regressor
        reg = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.06,
            max_iter=500,
            min_samples_leaf=30,
            l2_regularization=0.0,
            random_state=42,
        )
        reg.fit(X_train, y_train)

        sport_models[s] = reg

    # ---- Calibrator: total_diff_hat -> P(p1 wins) ----
    # Derive winner label if not present:
    if "y_winner_p1" in df.columns:
        ywin_train = train_df["y_winner_p1"].astype(int).to_numpy()
        ywin_test = test_df["y_winner_p1"].astype(int).to_numpy()
    else:
        # if you have per-player totals in your dataset, use them; else use sum of diffs
        # Here we use sum of true per-sport diffs: winner = total_diff > 0
        true_total_train = sum(train_df[f"{s}_y_diff"] for s in SPORTS)
        true_total_test = sum(test_df[f"{s}_y_diff"] for s in SPORTS)
        ywin_train = (true_total_train > 0).astype(int).to_numpy()
        ywin_test = (true_total_test > 0).astype(int).to_numpy()

    # Predicted totals
    pred_total_train = np.zeros(len(train_df), dtype=float)
    pred_total_test = np.zeros(len(test_df), dtype=float)
    for s in SPORTS:
        pred_total_train += sport_models[s].predict(X_train)
        pred_total_test += sport_models[s].predict(X_test)

    # Logistic calibration (simple & usually strong)
    calib = LogisticRegression(solver="lbfgs")
    calib.fit(pred_total_train.reshape(-1, 1), ywin_train)

    # Quick eval prints
    p_test = calib.predict_proba(pred_total_test.reshape(-1, 1))[:, 1]
    auc = (
        roc_auc_score(ywin_test, p_test)
        if len(np.unique(ywin_test)) > 1
        else float("nan")
    )
    brier = brier_score_loss(ywin_test, p_test)
    acc = ((p_test >= 0.5).astype(int) == ywin_test).mean()

    print(
        f"[GBDT] Winner Accuracy: {acc:.3f}  AUC: {auc:.3f}  Brier: {brier:.3f}"
    )

    bundle = {
        "sport_models": sport_models,
        "calibrator": calib,
        "feature_columns": list(X_train.columns),
    }
    return bundle


# -----------------------
# Prediction
# -----------------------
def predict_row(bundle, row: pd.Series) -> dict:
    """
    row should contain the same feature columns as training (numeric).
    """
    cols = bundle["feature_columns"]
    X = row.reindex(cols).to_frame().T
    # fill missing numeric features
    X = X.fillna(0.0)

    per_sport = {}
    total = 0.0

    for s in SPORTS:
        diff = float(bundle["sport_models"][s].predict(X)[0])
        diff = float(np.clip(diff, -MAX_DIFF, MAX_DIFF))
        total += diff

        # Convert sport diff to a sport-win probability (optional heuristic)
        # This is NOT calibrated; it’s just a monotonic mapping.
        p_s = 1.0 / (1.0 + np.exp(-0.25 * diff))
        per_sport[s] = {"diff": diff, "p1_win_prob": float(p_s)}

    p_win = float(
        bundle["calibrator"].predict_proba(np.array([[total]]))[:, 1][0]
    )
    winner = "p1" if p_win >= 0.5 else "p2"

    return {
        "per_sport": per_sport,
        "total_diff": float(total),
        "p1_win_prob": p_win,
        "winner": winner,
        "confidence": p_win if winner == "p1" else (1.0 - p_win),
    }


def save(bundle, path: str):
    joblib.dump(bundle, path)


def load(path: str):
    return joblib.load(path)
