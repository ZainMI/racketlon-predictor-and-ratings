import csv
import re


def draw_names(input_csv, output_file):
    draws = set()

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            draw = row.get("draw")
            if draw:
                draws.add(draw.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        for draw in sorted(draws):
            f.write(draw + "\n")

    print(f"Saved {len(draws)} draw names to {output_file}")


INVALID_DRAWS = ["double", "doubles", "mixed", "league", "team"]


def clean_by_draw_name(
    input_csv="matches.csv", output_csv="matches_name_cleaned.csv"
):
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            kept = 0
            removed = 0

            for row in reader:
                draw = (row.get("draw") or "").lower()
                if any(bad in draw for bad in INVALID_DRAWS):
                    removed += 1
                    continue

                writer.writerow(row)
                kept += 1

    print(f"Saved cleaned file to {output_csv}")
    print(f"Kept: {kept} rows | Removed: {removed} rows")
    return output_csv


def filter_by_completed(input_csv, output_csv="matches_completed.csv"):
    """
    Keeps only matches with a squash score.
    Assumes 'raw_points' contains the score like: 21-10|7-21|10-21
    If raw_points is empty/missing => not completed => removed.
    """
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            kept = 0
            removed = 0

            for row in reader:
                raw_points = (row.get("raw_points") or "").strip()
                if not raw_points:
                    removed += 1
                    continue

                writer.writerow(row)
                kept += 1

    print(f"Saved completed-only file to {output_csv}")
    print(f"Kept: {kept} rows | Removed (no score): {removed} rows")
    return output_csv


def normalize_player_names(
    input_csv, output_csv="matches_names_normalized.csv"
):
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                for field in ("team1_players", "team2_players"):
                    name = row.get(field) or ""

                    # remove seeding like [3/4], [2], etc
                    name = re.sub(r"\s*\[.*?\]\s*", "", name)

                    # normalize
                    row[field] = name.strip().lower()

                writer.writerow(row)

    print(f"Saved name-normalized file to {output_csv}")
    return output_csv


def filter_by_date_present(input_csv, output_csv="matches_with_date.csv"):
    """
    Removes rows with no match_date.
    """
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            kept = 0
            removed = 0

            for row in reader:
                match_date = (row.get("match_date") or "").strip()

                if not match_date:
                    removed += 1
                    continue

                writer.writerow(row)
                kept += 1

    print(f"Saved date-filtered file to {output_csv}")
    print(f"Kept: {kept} | Removed (no date): {removed}")
    return output_csv


def main():
    input_csv = "matches.csv"

    name_cleaned = clean_by_draw_name(input_csv, "matches_name_cleaned.csv")

    completed_cleaned = filter_by_completed(
        name_cleaned, "matches_name_and_completed_cleaned.csv"
    )

    normalized = normalize_player_names(completed_cleaned)

    date_filtered = filter_by_date_present(normalized)


if __name__ == "__main__":
    main()
