from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from funcs import load_predictor, predict_match, get_player_ratings

# Change this import path:
# from your_module_name import ...
# to wherever your functions live, for example:
# from funcs import load_predictor, predict_match, get_player_ratings

MODEL_DIR = "full/catboost/artifacts/predictor_package"

PLAYER1 = "zain magdon-ismail"
PLAYER2 = "patrick moran"


def pretty_print_prediction(result: dict) -> None:
    p1 = result["player1"]
    p2 = result["player2"]

    print("\n🏓 Match Prediction\n")
    print(f"{p1.title()} vs {p2.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        if sport not in result["sports"]:
            continue
        s = result["sports"][sport]
        print(
            f"{sport}: {s['score_p1']:>2} - {s['score_p2']:<2}   "
            f"(diff {s['pred_diff']:+.1f})"
        )

    print("\n---\n")
    print("📊 Total\n")
    print(f"{p1.title():<22}: {result['total_p1']}")
    print(f"{p2.title():<22}: {result['total_p2']}")
    print("-" * 32)
    print(f"Total Diff: {result['total_diff']:+d}\n")
    print(f"🏆 Predicted Winner: {result['winner'].title()}\n")


def pretty_print_player_ratings(ratings: dict) -> None:
    player = ratings["player"]
    print(f"\n📈 Player Ratings: {player.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        s = ratings["sports"][sport]
        print(f"=== {sport} ===")
        print(f"Rating:              {s['rating']:.2f}")
        print(f"Pred diff (vs avg):  {s['pred_diff_vs_avg']:.2f}")
        print(f"Games played:        {s['games_played']}")
        print(f"Days since last:     {s['days_since_last']:.1f}")
        print(f"Time multiplier:     {s['time_multiplier']:.2f}")
        print(f"diff_mean_10:        {s['diff_mean_10']:.2f}")
        print(f"resid_mean_10:       {s['resid_mean_10']:.2f}")
        print(f"diff_std_10:         {s['diff_std_10']:.2f}")
        print(f"momentum_diff_5_20:  {s['momentum_diff_5_20']:.2f}")
        print(f"long_n:              {s['long_n']:.0f}")
        print(f"long_diff_mean:      {s['long_diff_mean']:.2f}")
        print(f"long_total_mean:     {s['long_total_mean']:.2f}")
        print(f"long_winrate:        {s['long_winrate']:.2f}")
        print()


def main() -> None:
    predictor = load_predictor(MODEL_DIR)

    result = predict_match(predictor, PLAYER1, PLAYER2)
    pretty_print_prediction(result)

    p1_ratings = get_player_ratings(predictor, PLAYER1)
    p2_ratings = get_player_ratings(predictor, PLAYER2)

    pretty_print_player_ratings(p1_ratings)
    pretty_print_player_ratings(p2_ratings)


if __name__ == "__main__":
    main()
