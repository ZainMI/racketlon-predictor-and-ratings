import csv
import re

INVALID_DRAWS = ["double", "doubles", "mixed", "league", "team"]


def clean_matches(
    input_csv="data/matches.csv",
    output_csv="data/matches_cleaned.csv",
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
                # 1️⃣ Filter invalid draw names
                draw = (row.get("draw") or "").lower()
                if any(bad in draw for bad in INVALID_DRAWS):
                    removed += 1
                    continue

                # 2️⃣ Filter incomplete matches (no score)
                raw_points = (row.get("raw_points") or "").strip()
                if not raw_points:
                    removed += 1
                    continue

                # 3️⃣ Remove rows without match_date
                match_date = (row.get("match_date") or "").strip()
                if not match_date:
                    removed += 1
                    continue

                # 4️⃣ Normalize player names
                for field in ("team1_players", "team2_players"):
                    name = row.get(field) or ""

                    # Remove seeding like [3/4], [2], etc.
                    name = re.sub(r"\s*\[.*?\]\s*", "", name)

                    # Normalize
                    row[field] = name.strip().lower()

                writer.writerow(row)
                kept += 1

    print(f"Saved cleaned file to {output_csv}")
    print(f"Kept: {kept} rows | Removed: {removed} rows")


if __name__ == "__main__":
    clean_matches()
