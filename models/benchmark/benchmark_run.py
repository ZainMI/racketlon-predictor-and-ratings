from benchmark_fin import PredictorPackage

predictor = PredictorPackage.load(
    "finished_models/simple_benchmark/artifacts/predictor_package"
)

# player1 = "bastian böhm"
# player2 = "luke griffiths"
player2 = "patrick moran"
player1 = "zain magdon-ismail"


def pretty_print_prediction(result):
    p1 = result["player1"]
    p2 = result["player2"]

    print(f"\n🏓 Match Prediction\n")
    print(f"{p1.title()} vs {p2.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        if sport not in result["sports"]:
            continue
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


def print_player_state(predictor, player):
    p = player.strip().lower()
    stats = predictor.player_stats

    print(f"\n📈 Player State: {player.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        avg_diff = stats.get_avg_diff(p, sport, predictor.shrink)
        avg_total = stats.get_avg_total(p, sport, predictor.shrink)
        games = stats.count[p][sport]

        print(f"=== {sport} ===")
        print(f"Avg diff:      {avg_diff:>6.2f}")
        print(f"Avg total:     {avg_total:>6.2f}")
        print(f"Games played:  {games:>6d}")
        print()


def print_matchup_core(predictor, player1, player2):
    p1 = player1.strip().lower()
    p2 = player2.strip().lower()
    stats = predictor.player_stats

    print(f"\n🔎 Matchup Core: {player1.title()} vs {player2.title()}\n")

    for sport in ["TT", "BD", "SQ", "TN"]:
        p1_avg_diff = stats.get_avg_diff(p1, sport, predictor.shrink)
        p2_avg_diff = stats.get_avg_diff(p2, sport, predictor.shrink)

        p1_avg_total = stats.get_avg_total(p1, sport, predictor.shrink)
        p2_avg_total = stats.get_avg_total(p2, sport, predictor.shrink)

        pred_diff = p1_avg_diff - p2_avg_diff
        pred_total = 0.5 * (p1_avg_total + p2_avg_total)

        print(f"=== {sport} ===")
        print(f"p1_avg_diff:   {p1_avg_diff:+8.2f}")
        print(f"p2_avg_diff:   {p2_avg_diff:+8.2f}")
        print(f"pred_diff:     {pred_diff:+8.2f}")
        print(f"pred_total:    {pred_total:>8.2f}")
        print()


result = predictor.predict_pair(player1, player2)
pretty_print_prediction(result)
print_player_state(predictor, player1)
print_player_state(predictor, player2)
print_matchup_core(predictor, player1, player2)
