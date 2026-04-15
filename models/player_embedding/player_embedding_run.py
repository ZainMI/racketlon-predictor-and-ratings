from player_embedding_fin import PredictorPackage

predictor = PredictorPackage.load(
    "finished_models/player_embedding/artifacts/predictor_package"
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


result = predictor.predict_pair(player1, player2)
pretty_print_prediction(result)
