# data_collection/scrape_all.py
"""
Master scraper for NFL Fantasy Prospect Projector.

Pipeline:
1. Scrape all 32 NFL rosters → fantasy-relevant players (QB/RB/WR/TE)
2. For each player: athlete detail (draft, measurables)
3. For each player: college career stats
4. For each player: college gamelogs (for GP count)
5. For each player: NFL career stats
6. Save to CSVs

Run from project root:
    python -m data_collection.scrape_all

Expected runtime: ~4-6 hours (rate limited at 3s/request)
Can be interrupted and resumed (saves progress).
"""

import requests
import time
import json
import csv
import os
import re
import sys
from datetime import datetime

DELAY = 3.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

NFL_TEAM_IDS = [
    22, 1, 33, 2, 29, 3, 4, 5, 6, 7, 8, 9, 34, 11, 30, 12,
    13, 24, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 10, 28
]

OUTPUT_DIR = "data/raw"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "scrape_progress.json")


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
                    wait = DELAY * (2 ** attempt) + 5
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code == 404:
                    return None
                else:
                    print(f"    HTTP {resp.status_code}: {url}")
                    time.sleep(DELAY * 2)
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(DELAY * 2)
        
        return None


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_players": [], "step": "rosters"}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def parse_draft_string(draft_str):
    """Parse '2021: Rd 1, Pk 5 (CIN)' into components."""
    if not draft_str:
        return None, None, None, None
    
    match = re.match(r'(\d{4}):\s*Rd\s*(\d+),\s*Pk\s*(\d+)\s*\((\w+)\)', draft_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    return None, None, None, None


def parse_stat_value(val):
    """Convert stat string like '1,455' to float."""
    if val is None or val == '' or val == '--':
        return 0.0
    try:
        return float(val.replace(',', ''))
    except (ValueError, AttributeError):
        return 0.0


# =========================================================
# STEP 1: Get all fantasy-relevant players from NFL rosters
# =========================================================
def scrape_rosters(scraper):
    """Get all QB/RB/WR/TE from all 32 NFL rosters."""
    
    output_file = os.path.join(OUTPUT_DIR, "nfl_roster_players.csv")
    
    # Check if already done
    if os.path.exists(output_file):
        print(f"  Roster file exists, loading...")
        players = []
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                players.append(row)
        print(f"  Loaded {len(players)} players from cache")
        return players
    
    players = []
    seen_ids = set()
    
    for team_id in NFL_TEAM_IDS:
        print(f"  Scraping team {team_id}...")
        data = scraper.get(
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
        )
        
        if not data or 'athletes' not in data:
            print(f"    No roster data for team {team_id}")
            continue
        
        team_name = "unknown"
        
        for group in data['athletes']:
            for player in group.get('items', []):
                pos = player.get('position', {}).get('abbreviation', '')
                
                if pos not in FANTASY_POSITIONS:
                    continue
                
                pid = player.get('id')
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                
                college = player.get('college', {})
                
                players.append({
                    'espn_id': pid,
                    'name': player.get('displayName', ''),
                    'position': pos,
                    'height': player.get('displayHeight', ''),
                    'weight': player.get('displayWeight', ''),
                    'age': player.get('age', ''),
                    'dob': player.get('dateOfBirth', ''),
                    'college_name': college.get('name', ''),
                    'college_id': college.get('id', ''),
                    'experience_years': player.get('experience', {}).get('years', ''),
                    'team_id': team_id,
                })
    
    # Save
    if players:
        fieldnames = list(players[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(players)
    
    print(f"  Found {len(players)} fantasy-relevant players across 32 rosters")
    return players


# =========================================================
# STEP 2: Get athlete detail (draft info, full measurables)
# =========================================================
def scrape_athlete_details(scraper, players):
    """Get draft info and full bio for each player."""
    
    output_file = os.path.join(OUTPUT_DIR, "player_details.csv")
    
    if os.path.exists(output_file):
        print(f"  Details file exists, loading...")
        details = []
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                details.append(row)
        print(f"  Loaded {len(details)} player details from cache")
        return details
    
    details = []
    total = len(players)
    
    for i, player in enumerate(players):
        pid = player['espn_id']
        print(f"  [{i+1}/{total}] {player['name']} ({pid})...")
        
        data = scraper.get(
            f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{pid}"
        )
        
        if not data or 'athlete' not in data:
            print(f"    No detail data")
            continue
        
        ath = data['athlete']
        
        draft_year, draft_round, draft_pick, draft_team = parse_draft_string(
            ath.get('displayDraft', '')
        )
        
        details.append({
            'espn_id': pid,
            'name': ath.get('displayName', ''),
            'position': ath.get('position', {}).get('abbreviation', ''),
            'height': ath.get('displayHeight', ''),
            'weight': ath.get('displayWeight', ''),
            'dob': ath.get('displayDOB', ''),
            'age': ath.get('age', ''),
            'birth_place': ath.get('displayBirthPlace', ''),
            'college': ath.get('college', {}).get('name', ''),
            'college_id': ath.get('college', {}).get('id', ''),
            'draft_year': draft_year,
            'draft_round': draft_round,
            'draft_pick': draft_pick,
            'draft_team': draft_team,
            'experience': ath.get('displayExperience', ''),
            'hand': ath.get('hand', {}).get('abbreviation', '') if isinstance(ath.get('hand'), dict) else '',
            'college_athlete_id': ath.get('collegeAthlete', {}).get('id', ''),
        })
        
        # Save every 50 players (in case of crash)
        if (i + 1) % 50 == 0:
            _save_csv(output_file, details)
            print(f"    Checkpoint saved ({len(details)} players)")
    
    _save_csv(output_file, details)
    print(f"  Got details for {len(details)} players")
    return details


# =========================================================
# STEP 3: Get college career stats
# =========================================================
def scrape_college_stats(scraper, players):
    """Get college season-by-season stats for each player."""
    
    output_file = os.path.join(OUTPUT_DIR, "college", "college_stats.csv")
    
    if os.path.exists(output_file):
        print(f"  College stats file exists, loading...")
        rows = []
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"  Loaded {len(rows)} college stat rows from cache")
        return rows
    
    all_rows = []
    total = len(players)
    
    for i, player in enumerate(players):
        pid = player['espn_id']
        pos = player['position']
        print(f"  [{i+1}/{total}] College stats: {player['name']}...")
        
        data = scraper.get(
            f"https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{pid}/stats"
        )
        
        if not data or 'categories' not in data:
            continue
        
        # Parse each category
        for cat in data['categories']:
            cat_name = cat.get('name', '')
            names = cat.get('names', [])
            
            # Only care about offensive stats
            if cat_name not in ('passing', 'rushing', 'receiving'):
                continue
            
            for season_row in cat.get('statistics', []):
                year = season_row.get('season', {}).get('year', '')
                team_slug = season_row.get('teamSlug', '')
                stats = season_row.get('stats', [])
                
                row = {
                    'espn_id': pid,
                    'name': player['name'],
                    'position': pos,
                    'season': year,
                    'college_team': team_slug,
                    'stat_category': cat_name,
                }
                
                # Map stat names to values
                for j, stat_name in enumerate(names):
                    if j < len(stats):
                        row[stat_name] = stats[j]
                
                all_rows.append(row)
        
        if (i + 1) % 50 == 0:
            _save_csv(output_file, all_rows)
            print(f"    Checkpoint saved ({len(all_rows)} rows)")
    
    _save_csv(output_file, all_rows)
    print(f"  Got {len(all_rows)} college stat rows")
    return all_rows


# =========================================================
# STEP 4: Get college gamelogs (for GP count per season)
# =========================================================
def scrape_college_gamelogs(scraper, players, college_stats):
    """Get game count per season from college gamelogs."""
    
    output_file = os.path.join(OUTPUT_DIR, "college", "college_games_played.csv")
    
    if os.path.exists(output_file):
        print(f"  College GP file exists, loading...")
        rows = []
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"  Loaded {len(rows)} college GP rows from cache")
        return rows
    
    # Figure out which seasons each player played
    player_seasons = {}
    for row in college_stats:
        pid = row['espn_id']
        season = row['season']
        if pid not in player_seasons:
            player_seasons[pid] = set()
        player_seasons[pid].add(str(season))
    
    all_rows = []
    total = len(player_seasons)
    
    for idx, (pid, seasons) in enumerate(player_seasons.items()):
        player_name = next(
            (p['name'] for p in players if p['espn_id'] == pid), pid
        )
        
        for season in sorted(seasons):
            print(f"  [{idx+1}/{total}] Gamelog: {player_name} {season}...")
            
            data = scraper.get(
                f"https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{pid}/gamelog?season={season}"
            )
            
            if not data:
                continue
            
            # Count regular season games only
            reg_season_games = 0
            post_season_games = 0
            
            for st in data.get('seasonTypes', []):
                st_name = st.get('displayName', '').lower()
                for cat in st.get('categories', []):
                    events = cat.get('events', [])
                    if 'regular' in st_name:
                        reg_season_games = max(reg_season_games, len(events))
                    elif 'post' in st_name:
                        post_season_games = max(post_season_games, len(events))
            
            # Fallback: count total events if seasonTypes not clear
            if reg_season_games == 0:
                total_events = len(data.get('events', {}))
                reg_season_games = total_events  # best guess
            
            all_rows.append({
                'espn_id': pid,
                'name': player_name,
                'season': season,
                'regular_season_games': reg_season_games,
                'postseason_games': post_season_games,
            })
        
        if (idx + 1) % 30 == 0:
            _save_csv(output_file, all_rows)
            print(f"    Checkpoint saved ({len(all_rows)} rows)")
    
    _save_csv(output_file, all_rows)
    print(f"  Got {len(all_rows)} college GP rows")
    return all_rows


# =========================================================
# STEP 5: Get NFL career stats
# =========================================================
def scrape_nfl_stats(scraper, players):
    """Get NFL season-by-season stats for each player."""
    
    output_file = os.path.join(OUTPUT_DIR, "nfl", "nfl_stats.csv")
    
    if os.path.exists(output_file):
        print(f"  NFL stats file exists, loading...")
        rows = []
        with open(output_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"  Loaded {len(rows)} NFL stat rows from cache")
        return rows
    
    all_rows = []
    total = len(players)
    
    for i, player in enumerate(players):
        pid = player['espn_id']
        pos = player['position']
        print(f"  [{i+1}/{total}] NFL stats: {player['name']}...")
        
        data = scraper.get(
            f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{pid}/stats"
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
                    'name': player['name'],
                    'position': pos,
                    'season': year,
                    'nfl_team': team_slug,
                    'stat_category': cat_name,
                }
                
                for j, stat_name in enumerate(names):
                    if j < len(stats):
                        row[stat_name] = stats[j]
                
                all_rows.append(row)
        
        if (i + 1) % 50 == 0:
            _save_csv(output_file, all_rows)
            print(f"    Checkpoint saved ({len(all_rows)} rows)")
    
    _save_csv(output_file, all_rows)
    print(f"  Got {len(all_rows)} NFL stat rows")
    return all_rows


# =========================================================
# Utility
# =========================================================
def _save_csv(filepath, rows):
    """Save list of dicts to CSV."""
    if not rows:
        return
    
    # Get all unique keys across all rows
    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Main
# =========================================================
def main():
    start_time = datetime.now()
    print("=" * 70)
    print("NFL FANTASY PROSPECT PROJECTOR - DATA SCRAPER")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "college"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "nfl"), exist_ok=True)
    
    scraper = Scraper()
    
    # STEP 1: Rosters
    print("\n📋 STEP 1: Scraping NFL rosters...")
    players = scrape_rosters(scraper)
    print(f"  Total requests so far: {scraper.request_count}")
    
    # STEP 2: Athlete details (draft info)
    print("\n📋 STEP 2: Scraping athlete details...")
    details = scrape_athlete_details(scraper, players)
    print(f"  Total requests so far: {scraper.request_count}")
    
    # STEP 3: College stats
    print("\n📋 STEP 3: Scraping college stats...")
    college_stats = scrape_college_stats(scraper, players)
    print(f"  Total requests so far: {scraper.request_count}")
    
    # STEP 4: College gamelogs (GP)
    print("\n📋 STEP 4: Scraping college gamelogs for GP...")
    college_gp = scrape_college_gamelogs(scraper, players, college_stats)
    print(f"  Total requests so far: {scraper.request_count}")
    
    # STEP 5: NFL stats
    print("\n📋 STEP 5: Scraping NFL stats...")
    nfl_stats = scrape_nfl_stats(scraper, players)
    print(f"  Total requests so far: {scraper.request_count}")
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"  Time elapsed: {elapsed}")
    print(f"  Total API requests: {scraper.request_count}")
    print(f"  Players found: {len(players)}")
    print(f"  Player details: {len(details)}")
    print(f"  College stat rows: {len(college_stats)}")
    print(f"  College GP rows: {len(college_gp)}")
    print(f"  NFL stat rows: {len(nfl_stats)}")
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  nfl_roster_players.csv")
    print(f"  player_details.csv")
    print(f"  college/college_stats.csv")
    print(f"  college/college_games_played.csv")
    print(f"  nfl/nfl_stats.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()