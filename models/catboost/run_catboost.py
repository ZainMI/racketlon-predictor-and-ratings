from catboost_fin import PredictorPackage
import numpy as np

predictor = PredictorPackage.load(
    "finished_models/catboost/artifacts/predictor_package"
)

# player1 = "bastian böhm"
player2 = "luke griffiths"
# player2 = "patrick moran"
player1 = "zain magdon-ismail"


def pretty_print_prediction(result):
    p1 = result["player1"]
    p2 = result["player2"]

    print(f"\n🏓 Match Prediction\n")
    print(f"{p1.title()} vs {p2.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        s = result["sports"][sport]
        diff = s["pred_diff"]
        print(
            f"{sport}: {s['score_p1']:>2} - {s['score_p2']:<2}   (diff {diff:+.1f})"
        )

    print("\n---\n")

    print("📊 Total\n")
    print(f"{p1.title():<22}: {result['total_p1']}")
    print(f"{p2.title():<22}: {result['total_p2']}")
    print("-" * 32)
    print(f"Total Diff: {result['total_diff']:+d}\n")

    print(f"🏆 Predicted Winner: {result['winner'].title()}\n")


result = predictor.predict_pair(player1, player2)
pretty_print_prediction(result)


def print_player_ratings(predictor, player):
    p = player.strip().lower()
    state = predictor.inference_state["player_states_by_name"].get(p)

    if state is None:
        print(f"No state found for {player}")
        return

    print(f"\n📈 Player State: {player.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        print(f"=== {sport} ===")
        print(f"Rating:              {state.get(f'{sport}_rating_p1', 0):.2f}")
        print(f"Pred diff (vs avg):  {state.get(f'{sport}_pred_diff', 0):.2f}")
        print(f"Games played:        {state.get(f'{sport}_games_p1', 0):.0f}")

        if f"{sport}_days_since_last_p1" in state:
            print(
                f"Days since last:     {state.get(f'{sport}_days_since_last_p1', 0):.1f}"
            )
        if f"{sport}_time_mult_p1" in state:
            print(
                f"Time multiplier:     {state.get(f'{sport}_time_mult_p1', 1):.2f}"
            )

        recent_keys = [
            f"{sport}_p1_recent_diff_mean_10",
            f"{sport}_p1_recent_resid_mean_10",
            f"{sport}_p1_recent_blowout_win_rate_10",
            f"{sport}_p1_recent_diff_std_10",
            f"{sport}_p1_recent_momentum_diff_5_20",
        ]
        for k in recent_keys:
            if k in state:
                label = k.replace(f"{sport}_p1_recent_", "")
                print(f"{label:20s} {state[k]:.2f}")
        print()


print_player_ratings(predictor, player1)
print_player_ratings(predictor, player2)
