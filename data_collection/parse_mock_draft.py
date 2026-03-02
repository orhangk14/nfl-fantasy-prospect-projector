# data_collection/parse_mock_draft.py
"""
Parses mock draft data and creates a draft capital mapping for 2026 prospects.
Maps mock draft picks to our prospect list for draft capital features.

Run: python -m data_collection.parse_mock_draft
"""

import csv
import os
import re
from difflib import get_close_matches

RAW_DIR = "data/raw"

# Mock draft data parsed from Tankathon
# Format: (pick, name, position, school)
MOCK_DRAFT_2026 = [
    (1, "Fernando Mendoza", "QB", "Indiana"),
    (9, "Jeremiyah Love", "RB", "Notre Dame"),
    (11, "Carnell Tate", "WR", "Ohio State"),
    (13, "Makai Lemon", "WR", "USC"),
    (15, "Kenyon Sadiq", "TE", "Oregon"),
    (16, "Jordyn Tyson", "WR", "Arizona State"),
    (21, "Denzel Boston", "WR", "Washington"),
    (24, "KC Concepcion", "WR", "Texas A&M"),
    (34, "Ty Simpson", "QB", "Alabama"),
    (35, "Omar Cooper Jr.", "WR", "Indiana"),
    (48, "Malachi Fields", "WR", "Notre Dame"),
    (49, "Jadarian Price", "RB", "Notre Dame"),
    (58, "Chris Bell", "WR", "Louisville"),
    (60, "Chris Brazzell II", "WR", "Tennessee"),
    (62, "Eli Stowers", "TE", "Vanderbilt"),
    (63, "Zachariah Branch", "WR", "Georgia"),
    (64, "Germie Bernard", "WR", "Alabama"),
    (65, "Emmett Johnson", "RB", "Nebraska"),
    (67, "Elijah Sarratt", "WR", "Indiana"),
    (68, "Max Klare", "TE", "Ohio State"),
    (71, "Antonio Williams", "WR", "Clemson"),
    (74, "Michael Trigg", "TE", "Baylor"),
    (76, "Garrett Nussmeier", "QB", "LSU"),
    (80, "Ja'Kobi Lane", "WR", "USC"),
    (83, "Ted Hurst", "WR", "Georgia State"),
    (84, "Jonah Coleman", "RB", "Washington"),
    (105, "Deion Burks", "WR", "Oklahoma"),
    (106, "Mike Washington Jr.", "RB", "Arkansas"),
    (107, "Skyler Bell", "WR", "UConn"),
    (108, "Nicholas Singleton", "RB", "Penn State"),
    (109, "Brenen Thompson", "WR", "Mississippi State"),
    (111, "Carson Beck", "QB", "Miami"),
    (119, "Jack Endries", "TE", "Texas"),
    (122, "CJ Daniels", "WR", "Miami"),
    (124, "Justin Joly", "TE", "NC State"),
    (126, "Kaytron Allen", "RB", "Penn State"),
    (132, "De'Zhaun Stribling", "WR", "Ole Miss"),
    (135, "Josh Cameron", "WR", "Baylor"),
    (136, "Demond Claiborne", "RB", "Wake Forest"),
]

# Fantasy-relevant positions only
FANTASY_POS = {'QB', 'RB', 'WR', 'TE'}


def pick_to_round(pick):
    """Convert overall pick number to round."""
    if pick <= 32:
        return 1
    elif pick <= 64:
        return 2
    elif pick <= 100:
        return 3
    elif pick <= 138:
        return 4
    elif pick <= 176:
        return 5
    elif pick <= 220:
        return 6
    else:
        return 7


def load_prospect_ids():
    """Load our prospect ID mapping."""
    path = os.path.join(RAW_DIR, "prospects_2026_ids.csv")
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_name(name):
    """Normalize a name for fuzzy matching."""
    name = name.lower().strip()
    # Remove suffixes
    for suffix in [' jr.', ' jr', ' ii', ' iii', ' iv', ' sr.', ' sr']:
        name = name.replace(suffix, '')
    return name.strip()


def match_prospects():
    """Match mock draft picks to our prospect list."""
    prospects = load_prospect_ids()

    # Build name lookup
    prospect_names = {normalize_name(p['name']): p for p in prospects}
    prospect_name_list = list(prospect_names.keys())

    matches = []
    unmatched_mock = []
    unmatched_prospects = set(p['name'] for p in prospects)

    for pick, name, pos, school in MOCK_DRAFT_2026:
        if pos not in FANTASY_POS:
            continue

        norm = normalize_name(name)
        rnd = pick_to_round(pick)

        # Direct match
        if norm in prospect_names:
            p = prospect_names[norm]
            matches.append({
                'espn_id': p['espn_id'],
                'name': p['name'],
                'position': pos,
                'school': school,
                'mock_pick': pick,
                'mock_round': rnd,
            })
            unmatched_prospects.discard(p['name'])
            continue

        # Fuzzy match
        close = get_close_matches(norm, prospect_name_list, n=1, cutoff=0.75)
        if close:
            p = prospect_names[close[0]]
            # Verify position matches
            if p['position'] == pos:
                matches.append({
                    'espn_id': p['espn_id'],
                    'name': p['name'],
                    'position': pos,
                    'school': school,
                    'mock_pick': pick,
                    'mock_round': rnd,
                    'matched_from': name,
                })
                unmatched_prospects.discard(p['name'])
                continue

        unmatched_mock.append((pick, name, pos, school))

    return matches, unmatched_mock, unmatched_prospects


def assign_undrafted_estimates(unmatched_prospects, prospects):
    """
    For prospects not in the mock draft, assign estimated draft capital.
    Use their big board rank from combine data as a proxy, or assign
    late-round / UDFA estimates.
    """
    estimates = []

    for p_name in unmatched_prospects:
        prospect = next((p for p in prospects if p['name'] == p_name), None)
        if not prospect:
            continue

        # Default: round 5-7 / UDFA range
        estimates.append({
            'espn_id': prospect['espn_id'],
            'name': prospect['name'],
            'position': prospect['position'],
            'mock_pick': 180,  # late round estimate
            'mock_round': 6,
            'is_estimate': True,
        })

    return estimates


def main():
    print("=" * 70)
    print("PARSING 2026 MOCK DRAFT")
    print("=" * 70)

    prospects = load_prospect_ids()
    matches, unmatched_mock, unmatched_prospects = match_prospects()
    estimates = assign_undrafted_estimates(unmatched_prospects, prospects)

    all_draft_capital = matches + estimates

    print(f"\n  Matched to mock draft: {len(matches)}")
    print(f"  Unmatched mock picks (non-fantasy or missing): {len(unmatched_mock)}")
    print(f"  Prospects not in mock (estimated): {len(estimates)}")
    print(f"  Total with draft capital: {len(all_draft_capital)}")

    # Print matches
    print(f"\n{'─' * 70}")
    print(f"{'Pick':<6} {'Rd':<4} {'Name':<30} {'Pos':<4} {'School':<20} {'Note'}")
    print(f"{'─' * 70}")

    for m in sorted(all_draft_capital, key=lambda x: x['mock_pick']):
        note = '(estimated)' if m.get('is_estimate') else ''
        if m.get('matched_from'):
            note = f'(matched from: {m["matched_from"]})'
        print(f"{m['mock_pick']:<6} {m['mock_round']:<4} {m['name']:<30} "
              f"{m['position']:<4} {m.get('school', ''):<20} {note}")

    if unmatched_mock:
        print(f"\n⚠️  Mock picks not matched:")
        for pick, name, pos, school in unmatched_mock:
            print(f"  Pick {pick}: {name} ({pos}, {school})")

    if unmatched_prospects:
        print(f"\n⚠️  Prospects without mock draft pick:")
        for name in sorted(unmatched_prospects):
            print(f"  {name}")

    # Save
    output_path = os.path.join(RAW_DIR, "mock_draft_2026.csv")
    if all_draft_capital:
        keys = ['espn_id', 'name', 'position', 'school', 'mock_pick',
                'mock_round', 'is_estimate', 'matched_from']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_draft_capital)

    print(f"\n✅ Saved: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()