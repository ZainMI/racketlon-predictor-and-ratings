import pandas as pd
from collections import defaultdict
from typing import Dict, Any

SPORTS = ["TT", "BD", "SQ", "TN"]


def read_matches(file_path):
    return pd.read_csv(file_path)


def compute_player_sport_averages(matches):
    """
    Returns dict: { player_name: { sport: avg_diff, ... }, ... }
    avg_diff is computed from the player's perspective:
      avg_diff = mean( p_score - opponent_score )
    Positive => player generally outscores opponent in that sport.
    """
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    # iterate rows once
    for row in matches.itertuples(index=False):
        # Use getattr so namedtuple access works
        p1 = getattr(row, "team1_players")
        p2 = getattr(row, "team2_players")

        for sport in SPORTS:
            attr_p1 = f"{sport}_p1"
            attr_p2 = f"{sport}_p2"

            s1_raw = getattr(row, attr_p1, None)
            s2_raw = getattr(row, attr_p2, None)

            if s1_raw is None or s2_raw is None:
                continue
            if pd.isna(s1_raw) or pd.isna(s2_raw):
                continue

            s1 = float(s1_raw)
            s2 = float(s2_raw)

            diff = s1 - s2  # p1 perspective

            # update totals and counts only when valid
            totals[p1][sport] += diff
            counts[p1][sport] += 1

            totals[p2][sport] += -diff
            counts[p2][sport] += 1

    averages = {}
    for player, sport_totals in totals.items():
        averages[player] = {}
        for sport in SPORTS:
            c = counts[player].get(sport, 0)
            if c > 0:
                averages[player][sport] = sport_totals.get(sport, 0.0) / c
            else:
                averages[player][sport] = 0.0
    return averages


def initialize_ratings(matches):
    """
    Wrapper that computes and returns per-player sport averages (initial ratings).
    """
    return compute_player_sport_averages(matches)


def main():
    matches = read_matches("data/matches_cleaned.csv")
    ratings = initialize_ratings(matches)
    print(ratings.get("zain magdon-ismail"))


if __name__ == "__main__":
    main()
