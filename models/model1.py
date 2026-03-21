import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SPORTS = ["TT", "BD", "SQ", "TN"]
ALPHAS = {"TT": 0.010, "BD": 0.010, "SQ": 0.010, "TN": 0.010}
TEST_RATIO = 0.2
MAX_DIFF = 21.0


def get_diff_by_sport(sport, row):
    value = row.get(f"{sport}_y_diff")
    if pd.isna(value):
        return np.nan
    return float(value)


def get_rating_diff(row):
    return {
        sport: float(row.get(f"{sport}_rating_diff", 0.0)) for sport in SPORTS
    }


def get_actual_diff(row):
    return {sport: get_diff_by_sport(sport, row) for sport in SPORTS}


def compute_error(actual_diff, expected_diff):
    ret = {}
    for sport in actual_diff:
        a = actual_diff[sport]
        e = expected_diff[sport]
        if pd.isna(a):
            ret[sport] = np.nan
        else:
            ret[sport] = a - e
    return ret


def rating_to_score_diff(x, alpha):
    return 21 * (1 - np.exp(-alpha * x)) / (1 + np.exp(-alpha * x))


def get_score_diffs(rating_diff, alphas):
    ret = {}
    for sport in rating_diff:
        ret[sport] = rating_to_score_diff(rating_diff[sport], alphas[sport])
    return ret


def expected_match_results(row, alphas):
    rating_diff = get_rating_diff(row)
    expected_diff = get_score_diffs(rating_diff, alphas)

    total_diff = 0.0
    for sport in SPORTS:
        has_col = f"has_{sport}"
        if has_col in row and int(row[has_col]) == 1:
            total_diff += expected_diff[sport]

    if total_diff > 0:
        winner = 1
    elif total_diff < 0:
        winner = 0
    else:
        winner = -1

    return expected_diff, total_diff, winner


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(df, alphas):
    split = int(len(df) * (1 - TEST_RATIO))
    test_df = df.iloc[split:].copy()

    sport_true = {sport: [] for sport in SPORTS}
    sport_pred = {sport: [] for sport in SPORTS}

    total_true = []
    total_pred = []

    winner_true = []
    winner_pred = []

    for _, row in test_df.iterrows():
        expected_diff, pred_total_diff, _ = expected_match_results(row, alphas)
        actual_diff = get_actual_diff(row)

        true_total_diff = 0.0
        any_sport = False

        for sport in SPORTS:
            if pd.isna(actual_diff[sport]):
                continue

            any_sport = True
            sport_true[sport].append(actual_diff[sport])
            sport_pred[sport].append(
                np.clip(expected_diff[sport], -MAX_DIFF, MAX_DIFF)
            )
            true_total_diff += actual_diff[sport]

        if not any_sport:
            continue

        total_true.append(true_total_diff)
        total_pred.append(pred_total_diff)

        # Only evaluate winner accuracy on rows with a non-draw true result
        if true_total_diff != 0:
            winner_true.append(1 if true_total_diff > 0 else 0)

            if pred_total_diff > 0:
                winner_pred.append(1)
            elif pred_total_diff < 0:
                winner_pred.append(0)
            else:
                winner_pred.append(0)

    for sport in SPORTS:
        if len(sport_true[sport]) == 0:
            print(f"\n{sport} TEST RESULTS")
            print("No usable rows")
            continue

        y_true = np.array(sport_true[sport], dtype=float)
        y_pred = np.array(sport_pred[sport], dtype=float)

        print(f"\n{sport} TEST RESULTS")
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("RMSE:", rmse(y_true, y_pred))
        print("R^2:", r2_score(y_true, y_pred))

    total_true = np.array(total_true, dtype=float)
    total_pred = np.array(total_pred, dtype=float)

    print("\n=== MATCH LEVEL ===")
    print("Total Diff MAE:", mean_absolute_error(total_true, total_pred))
    print("Total Diff RMSE:", rmse(total_true, total_pred))
    print("Total Diff R^2:", r2_score(total_true, total_pred))

    if len(winner_true) == 0:
        print("Winner Accuracy: no non-draw matches available")
    else:
        winner_true = np.array(winner_true, dtype=int)
        winner_pred = np.array(winner_pred, dtype=int)
        print("Winner Accuracy:", (winner_true == winner_pred).mean())


def inspect_example_matches(df, alphas, n=5):
    split = int(len(df) * (1 - TEST_RATIO))
    test_df = df.iloc[split:].copy()

    print("\n=== SAMPLE PREDICTIONS ===")
    shown = 0

    for _, row in test_df.iterrows():
        expected_diff, pred_total_diff, pred_winner = expected_match_results(
            row, alphas
        )
        actual_diff = get_actual_diff(row)

        print(f"\nMatch: {row['p1_name']} vs {row['p2_name']}")
        print("Expected score differential:")
        for sport in SPORTS:
            if int(row.get(f"has_{sport}", 0)) == 1:
                print(f"  {sport}: {expected_diff[sport]:+.2f}")
            else:
                print(f"  {sport}: not played")

        print("Actual score differential:")
        for sport in SPORTS:
            if pd.isna(actual_diff[sport]):
                print(f"  {sport}: not played")
            else:
                print(f"  {sport}: {actual_diff[sport]:+.2f}")

        print(f"Predicted total diff: {pred_total_diff:+.2f}")

        if pred_winner == 1:
            print(f"Predicted winner: {row['p1_name']}")
        elif pred_winner == 0:
            print(f"Predicted winner: {row['p2_name']}")
        else:
            print("Predicted winner: Draw")

        shown += 1
        if shown >= n:
            break


def main(path="data/data.csv"):
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Loaded: {path}")
    print(f"Shape: {df.shape}")

    evaluate_model(df, ALPHAS)
    inspect_example_matches(df, ALPHAS, n=5)


if __name__ == "__main__":
    main()
