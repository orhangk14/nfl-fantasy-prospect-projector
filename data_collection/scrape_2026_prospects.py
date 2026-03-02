# data_collection/scrape_2026_prospects.py
"""
Scrape college stats for 2026 NFL Draft prospects.
Fantasy positions only: QB, RB, WR, TE.

Run: python -m data_collection.scrape_2026_prospects
"""

import requests
import time
import json
import csv
import os

DELAY = 3.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_DIR = "data/raw"

# 2026 prospects - fantasy positions only
# From Tankathon big board
PROSPECTS_2026 = [
    # QBs
    {"name": "Fernando Mendoza", "position": "QB", "school": "Indiana", "height": '6\'5"', "weight": "236 lbs"},
    {"name": "Ty Simpson", "position": "QB", "school": "Alabama", "height": '6\'1"', "weight": "211 lbs"},
    {"name": "Garrett Nussmeier", "position": "QB", "school": "LSU", "height": '6\'1.5"', "weight": "203 lbs"},
    {"name": "Carson Beck", "position": "QB", "school": "Miami", "height": '6\'5"', "weight": "233 lbs"},
    {"name": "Drew Allar", "position": "QB", "school": "Penn State", "height": '6\'5.5"', "weight": "228 lbs"},
    {"name": "Cade Klubnik", "position": "QB", "school": "Clemson", "height": '6\'2.5"', "weight": "207 lbs"},
    {"name": "Sawyer Robertson", "position": "QB", "school": "Baylor", "height": '6\'4"', "weight": "216 lbs"},
    
    # RBs
    {"name": "Jeremiyah Love", "position": "RB", "school": "Notre Dame", "height": '6\'0"', "weight": "212 lbs"},
    {"name": "Jadarian Price", "position": "RB", "school": "Notre Dame", "height": '5\'10.5"', "weight": "203 lbs"},
    {"name": "Emmett Johnson", "position": "RB", "school": "Nebraska", "height": '5\'10.5"', "weight": "202 lbs"},
    {"name": "Jonah Coleman", "position": "RB", "school": "Washington", "height": '5\'8"', "weight": "220 lbs"},
    {"name": "Mike Washington Jr.", "position": "RB", "school": "Arkansas", "height": '6\'1"', "weight": "223 lbs"},
    {"name": "Demond Claiborne", "position": "RB", "school": "Wake Forest", "height": '5\'10"', "weight": "188 lbs"},
    {"name": "Kaytron Allen", "position": "RB", "school": "Penn State", "height": '5\'11.5"', "weight": "216 lbs"},
    {"name": "Nicholas Singleton", "position": "RB", "school": "Penn State", "height": '6\'0.5"', "weight": "219 lbs"},
    {"name": "J'Mari Taylor", "position": "RB", "school": "Virginia", "height": '5\'10"', "weight": "199 lbs"},
    
    # WRs
    {"name": "Carnell Tate", "position": "WR", "school": "Ohio State", "height": '6\'2.5"', "weight": "192 lbs"},
    {"name": "Makai Lemon", "position": "WR", "school": "USC", "height": '5\'11"', "weight": "192 lbs"},
    {"name": "Jordyn Tyson", "position": "WR", "school": "Arizona State", "height": '6\'2"', "weight": "203 lbs"},
    {"name": "Denzel Boston", "position": "WR", "school": "Washington", "height": '6\'3.5"', "weight": "212 lbs"},
    {"name": "KC Concepcion", "position": "WR", "school": "Texas A&M", "height": '5\'11.5"', "weight": "196 lbs"},
    {"name": "Omar Cooper Jr.", "position": "WR", "school": "Indiana", "height": '6\'0"', "weight": "199 lbs"},
    {"name": "Malachi Fields", "position": "WR", "school": "Notre Dame", "height": '6\'4.5"', "weight": "218 lbs"},
    {"name": "Chris Bell", "position": "WR", "school": "Louisville", "height": '6\'2"', "weight": "222 lbs"},
    {"name": "Chris Brazzell II", "position": "WR", "school": "Tennessee", "height": '6\'4"', "weight": "198 lbs"},
    {"name": "Zachariah Branch", "position": "WR", "school": "Georgia", "height": '5\'8.5"', "weight": "177 lbs"},
    {"name": "Ted Hurst", "position": "WR", "school": "Georgia State", "height": '6\'4"', "weight": "206 lbs"},
    {"name": "Deion Burks", "position": "WR", "school": "Oklahoma", "height": '5\'10"', "weight": "180 lbs"},
    {"name": "Brenen Thompson", "position": "WR", "school": "Mississippi State", "height": '5\'9.5"', "weight": "164 lbs"},
    {"name": "Germie Bernard", "position": "WR", "school": "Alabama", "height": '6\'1.5"', "weight": "206 lbs"},
    {"name": "Skyler Bell", "position": "WR", "school": "UConn", "height": '5\'11.5"', "weight": "192 lbs"},
    {"name": "Elijah Sarratt", "position": "WR", "school": "Indiana", "height": '6\'2.5"', "weight": "210 lbs"},
    {"name": "Ja'Kobi Lane", "position": "WR", "school": "USC", "height": '6\'4.5"', "weight": "200 lbs"},
    {"name": "Josh Cameron", "position": "WR", "school": "Baylor", "height": '6\'1.5"', "weight": "220 lbs"},
    {"name": "Reggie Virgil", "position": "WR", "school": "Texas Tech", "height": '6\'2.5"', "weight": "187 lbs"},
    {"name": "Kevin Coleman Jr.", "position": "WR", "school": "Missouri", "height": '5\'10.5"', "weight": "179 lbs"},
    {"name": "De'Zhaun Stribling", "position": "WR", "school": "Ole Miss", "height": '6\'2"', "weight": "207 lbs"},
    {"name": "Bryce Lance", "position": "WR", "school": "North Dakota State", "height": '6\'3.5"', "weight": "204 lbs"},
    {"name": "Aaron Anderson", "position": "WR", "school": "LSU", "height": '5\'8"', "weight": "191 lbs"},
    {"name": "Eric McAlister", "position": "WR", "school": "TCU", "height": '6\'3.5"', "weight": "194 lbs"},
    {"name": "Eric Rivers", "position": "WR", "school": "Georgia Tech", "height": '5\'10"', "weight": "176 lbs"},
    {"name": "CJ Daniels", "position": "WR", "school": "Miami", "height": '6\'2.5"', "weight": "202 lbs"},
    {"name": "Antonio Williams", "position": "WR", "school": "Clemson", "height": '5\'11.5"', "weight": "187 lbs"},
    
    # TEs
    {"name": "Kenyon Sadiq", "position": "TE", "school": "Oregon", "height": '6\'3"', "weight": "241 lbs"},
    {"name": "Eli Stowers", "position": "TE", "school": "Vanderbilt", "height": '6\'4"', "weight": "239 lbs"},
    {"name": "Max Klare", "position": "TE", "school": "Ohio State", "height": '6\'4.5"', "weight": "246 lbs"},
    {"name": "Michael Trigg", "position": "TE", "school": "Baylor", "height": '6\'4"', "weight": "240 lbs"},
    {"name": "Jack Endries", "position": "TE", "school": "Texas", "height": '6\'4.5"', "weight": "245 lbs"},
    {"name": "Justin Joly", "position": "TE", "school": "NC State", "height": '6\'3.5"', "weight": "241 lbs"},
    {"name": "Sam Roush", "position": "TE", "school": "Stanford", "height": '6\'6"', "weight": "267 lbs"},
    {"name": "Eli Raridon", "position": "TE", "school": "Notre Dame", "height": '6\'6"', "weight": "245 lbs"},
    {"name": "Joe Royer", "position": "TE", "school": "Cincinnati", "height": '6\'5"', "weight": "247 lbs"},
    {"name": "Oscar Delp", "position": "TE", "school": "Georgia", "height": '6\'5"', "weight": "245 lbs"},
    {"name": "Dallen Bentley", "position": "TE", "school": "Utah", "height": '6\'4"', "weight": "253 lbs"},
    {"name": "Nate Boerkircher", "position": "TE", "school": "Texas A&M", "height": '6\'5.5"', "weight": "245 lbs"},
    {"name": "Marlin Klein", "position": "TE", "school": "Michigan", "height": '6\'6"', "weight": "248 lbs"},
]


class Scraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.request_count = 0
        self.last_request = 0

    def get(self, url, retries=3):
        for attempt in range(retries):
            elapsed = time.time() - self.last_request
            if elapsed < DELAY:
                time.sleep(DELAY - elapsed)
            try:
                resp = self.session.get(url, timeout=30)
                self.last_request = time.time()
                self.request_count += 1
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    time.sleep(DELAY * (2 ** attempt) + 5)
                elif resp.status_code == 404:
                    return None
            except Exception as e:
                print(f"    ❌ {e}")
                time.sleep(DELAY * 2)
        return None


def _save_csv(filepath, rows):
    if not rows:
        return
    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 70)
    print("2026 NFL DRAFT PROSPECTS - COLLEGE STATS SCRAPER")
    print(f"Prospects to scrape: {len(PROSPECTS_2026)}")
    print("=" * 70)

    scraper = Scraper()

    # =========================================================
    # STEP 1: Find ESPN IDs via search
    # =========================================================
    print("\n📋 STEP 1: Finding ESPN IDs...")

    prospect_ids = []
    not_found = []

    for i, prospect in enumerate(PROSPECTS_2026):
        name = prospect['name']
        school = prospect['school']
        print(f"  [{i+1}/{len(PROSPECTS_2026)}] Searching: {name} ({school})...", end=" ")

        data = scraper.get(
            f"https://site.web.api.espn.com/apis/common/v3/search?query={name.replace(' ', '+')}&limit=10&type=player"
        )

        found = False
        if data and data.get('items'):
            for item in data['items']:
                item_name = item.get('displayName', '').lower()
                search_name = name.lower()
                
                # Match by name (fuzzy - at least last name must match)
                last_name = search_name.split()[-1]
                if last_name in item_name.lower():
                    espn_id = item.get('id')
                    league = item.get('league', '')
                    print(f"✅ ID: {espn_id} ({league})")
                    
                    prospect_entry = {**prospect, 'espn_id': espn_id, 'espn_league': league}
                    prospect_ids.append(prospect_entry)
                    found = True
                    break

        if not found:
            print(f"❌ NOT FOUND")
            not_found.append(prospect)

    print(f"\n  Found: {len(prospect_ids)}/{len(PROSPECTS_2026)}")
    if not_found:
        print(f"  Missing: {[p['name'] for p in not_found]}")

    # Save prospect list with IDs
    _save_csv(os.path.join(OUTPUT_DIR, "prospects_2026_ids.csv"), prospect_ids)

    # =========================================================
    # STEP 2: Get college stats for each prospect
    # =========================================================
    print(f"\n📋 STEP 2: Scraping college stats...")
    est_min = len(prospect_ids) * DELAY / 60
    print(f"  ~{est_min:.0f} minutes")

    all_college_rows = []

    for i, prospect in enumerate(prospect_ids):
        pid = prospect['espn_id']
        name = prospect['name']

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(prospect_ids)}] {name}...")

        data = scraper.get(
            f"https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{pid}/stats"
        )

        if not data or 'categories' not in data:
            continue

        for cat in data['categories']:
            cat_name = cat.get('name', '')
            names = cat.get('names', [])
            if cat_name not in ('passing', 'rushing', 'receiving'):
                continue
            for season_row in cat.get('statistics', []):
                year = season_row.get('season', {}).get('year', '')
                team_slug = season_row.get('teamSlug', '')
                stats = season_row.get('stats', [])
                row = {
                    'espn_id': pid,
                    'name': name,
                    'position': prospect['position'],
                    'height': prospect['height'],
                    'weight': prospect['weight'],
                    'school': prospect['school'],
                    'season': year,
                    'college_team': team_slug,
                    'stat_category': cat_name,
                    'is_2026_prospect': True,
                }
                for j, stat_name in enumerate(names):
                    if j < len(stats):
                        row[stat_name] = stats[j]
                all_college_rows.append(row)

    _save_csv(os.path.join(OUTPUT_DIR, "college", "prospects_2026_college_stats.csv"), all_college_rows)

    # =========================================================
    # STEP 3: Get college gamelogs for GP
    # =========================================================
    print(f"\n📋 STEP 3: Scraping college gamelogs for GP...")

    # Build season list per prospect
    prospect_seasons = {}
    for row in all_college_rows:
        pid = row['espn_id']
        if pid not in prospect_seasons:
            prospect_seasons[pid] = {'name': row['name'], 'seasons': set()}
        prospect_seasons[pid]['seasons'].add(str(row['season']))

    all_gp_rows = []

    for idx, (pid, info) in enumerate(prospect_seasons.items()):
        for season in sorted(info['seasons']):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(prospect_seasons)}] {info['name']} {season}")

            data = scraper.get(
                f"https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{pid}/gamelog?season={season}"
            )

            if not data:
                continue

            reg_games = 0
            post_games = 0

            for st in data.get('seasonTypes', []):
                st_name = st.get('displayName', '').lower()
                for cat in st.get('categories', []):
                    events = cat.get('events', [])
                    if 'regular' in st_name:
                        reg_games = max(reg_games, len(events))
                    elif 'post' in st_name:
                        post_games = max(post_games, len(events))

            if reg_games == 0:
                reg_games = len(data.get('events', {}))

            all_gp_rows.append({
                'espn_id': pid,
                'name': info['name'],
                'season': season,
                'regular_season_games': reg_games,
                'postseason_games': post_games,
            })

    _save_csv(os.path.join(OUTPUT_DIR, "college", "prospects_2026_games_played.csv"), all_gp_rows)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"✅ COMPLETE")
    print(f"  Requests: {scraper.request_count}")
    print(f"  Prospects found: {len(prospect_ids)}")
    print(f"  College stat rows: {len(all_college_rows)}")
    print(f"  GP rows: {len(all_gp_rows)}")
    print(f"\n  Files:")
    print(f"    {OUTPUT_DIR}/prospects_2026_ids.csv")
    print(f"    {OUTPUT_DIR}/college/prospects_2026_college_stats.csv")
    print(f"    {OUTPUT_DIR}/college/prospects_2026_games_played.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()