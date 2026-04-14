from benchmark import PredictorPackage


PACKAGE_DIR = "finished_models/benchmark/artifacts/predictor_package"

PLAYER1 = "zain magdon-ismail"
PLAYER2 = "luke griffiths"


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


def print_matchup_core(predictor, player1, player2):
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()

    s1 = predictor.inference_state["player_states_by_name"].get(p1)
    s2 = predictor.inference_state["player_states_by_name"].get(p2)

    if s1 is None or s2 is None:
        print("\nNo matchup debug available.\n")
        return

    print(f"\n🔎 Matchup Core: {player1.title()} vs {player2.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        r1 = float(s1.get(f"{sport}_rating_p1", 0.0))
        r2 = float(s2.get(f"{sport}_rating_p1", 0.0))
        pd1 = float(s1.get(f"{sport}_pred_diff", 0.0))
        pd2 = float(s2.get(f"{sport}_pred_diff", 0.0))

        d1 = float(s1.get(f"{sport}_p1_recent_diff_mean_10", 0.0))
        d2 = float(s2.get(f"{sport}_p1_recent_diff_mean_10", 0.0))

        rres1 = float(s1.get(f"{sport}_p1_recent_resid_mean_10", 0.0))
        rres2 = float(s2.get(f"{sport}_p1_recent_resid_mean_10", 0.0))

        print(f"=== {sport} ===")
        print(f"rating_diff:         {r1 - r2:+.2f}")
        print(f"pred_diff_vs_avg:    {pd1 - pd2:+.2f}")
        print(f"recent_diff_mean10:  {d1 - d2:+.2f}")
        print(f"recent_resid_mean10: {rres1 - rres2:+.2f}")
        print()


if __name__ == "__main__":
    predictor = PredictorPackage.load(PACKAGE_DIR)

    result = predictor.predict_pair(PLAYER1, PLAYER2)
    pretty_print_prediction(result)

    print_player_ratings(predictor, PLAYER1)
    print_player_ratings(predictor, PLAYER2)

    print_matchup_core(predictor, PLAYER1, PLAYER2)
