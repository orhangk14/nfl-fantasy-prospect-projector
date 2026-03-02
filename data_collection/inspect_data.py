# data_collection/inspect_data.py
"""
Quick inspection of all our data files.
Shows row counts, columns, sample data, missing values.

Run: python -m data_collection.inspect_data
"""

import csv
import os
from collections import Counter

DATA_DIR = "data/raw"

FILES = [
    "nfl_roster_players.csv",
    "player_details.csv",
    "college/college_stats.csv",
    "college/college_games_played.csv",
    "nfl/nfl_stats.csv",
    "college/prospects_2026_college_stats.csv",
    "college/prospects_2026_games_played.csv",
    "combine_2026_fantasy.csv",
    "college/historiccombinedata.csv",
    "prospects_2026_ids.csv",
]


def inspect_file(filepath):
    full = os.path.join(DATA_DIR, filepath)
    if not os.path.exists(full):
        print(f"  ❌ NOT FOUND: {full}")
        return

    rows = []
    with open(full, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"  Rows: {len(rows)}")
    print(f"  Columns ({len(headers)}): {headers}")

    # Missing value counts
    missing = {}
    for col in headers:
        empty = sum(1 for r in rows if not r.get(col, '').strip())
        if empty > 0:
            missing[col] = f"{empty}/{len(rows)} ({100*empty/len(rows):.0f}%)"
    if missing:
        print(f"  Missing values:")
        for col, val in list(missing.items())[:10]:
            print(f"    {col}: {val}")

    # Position breakdown if applicable
    for pos_col in ['position', 'POS']:
        if pos_col in headers:
            positions = Counter(r[pos_col] for r in rows)
            print(f"  Positions: {dict(positions)}")

    # Stat category breakdown
    if 'stat_category' in headers:
        cats = Counter(r['stat_category'] for r in rows)
        print(f"  Stat categories: {dict(cats)}")

    # Season range
    if 'season' in headers:
        seasons = [r['season'] for r in rows if r['season'].strip()]
        if seasons:
            print(f"  Season range: {min(seasons)} - {max(seasons)}")

    # Sample rows
    print(f"  Sample (first 3):")
    for row in rows[:3]:
        compact = {k: v for k, v in row.items() if v and v.strip()}
        print(f"    {compact}")


def main():
    print("=" * 70)
    print("DATA INSPECTION")
    print("=" * 70)

    for filepath in FILES:
        print(f"\n{'─' * 70}")
        print(f"📄 {filepath}")
        print(f"{'─' * 70}")
        inspect_file(filepath)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()