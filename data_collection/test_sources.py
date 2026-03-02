# data_collection/test_sources.py
"""
Round 5: Get the missing pieces:
1. displayDraft value from athlete detail
2. College gamelog game count (to derive GP)
3. NFL stats for Chase - need to see all seasons including 2024/2025
4. Test if collegeAthlete ID links work for college→NFL mapping
"""

import requests
import time
import json

DELAY = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    print(f"  ❌ {resp.status_code}: {url}")
    return None


def main():
    print("ROUND 5: MISSING PIECES")
    print("=" * 70)
    
    # =========================================================
    # 1. Get displayDraft + other display fields from Chase
    # =========================================================
    print("\n1. CHASE - DISPLAY FIELDS (draft, experience, etc)")
    data = fetch(
        "https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/4362628"
    )
    if data and 'athlete' in data:
        ath = data['athlete']
        display_fields = [
            'displayDraft', 'displayExperience', 'displayBirthPlace',
            'displayHeight', 'displayWeight', 'displayDOB', 'age',
            'hand', 'collegeAthlete'
        ]
        for field in display_fields:
            if field in ath:
                val = ath[field]
                if isinstance(val, dict):
                    print(f"  {field}: {json.dumps(val)}")
                else:
                    print(f"  {field}: {val}")
    
    time.sleep(DELAY)
    
    # =========================================================
    # 2. Get the same for Travis Hunter (2025 draft class)
    # =========================================================
    print("\n\n2. TRAVIS HUNTER - DISPLAY FIELDS")
    data = fetch(
        "https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/4685415"
    )
    if data and 'athlete' in data:
        ath = data['athlete']
        display_fields = [
            'displayDraft', 'displayExperience', 'displayBirthPlace',
            'displayHeight', 'displayWeight', 'displayDOB', 'age',
            'hand', 'collegeAthlete', 'debutYear'
        ]
        for field in display_fields:
            if field in ath:
                val = ath[field]
                if isinstance(val, dict):
                    print(f"  {field}: {json.dumps(val)}")
                else:
                    print(f"  {field}: {val}")
    
    time.sleep(DELAY)
    
    # =========================================================
    # 3. Get a few more players' draft info to confirm format
    #    Burrow, Barkley, CeeDee Lamb
    # =========================================================
    print("\n\n3. MULTIPLE PLAYERS - DRAFT INFO")
    
    players = {
        "Joe Burrow": "3915511",
        "Saquon Barkley": "3929630",
    }
    
    # Search for CeeDee Lamb
    data = fetch(
        "https://site.web.api.espn.com/apis/common/v3/search?query=ceedee+lamb&limit=1&type=player"
    )
    if data and data.get('items'):
        lamb_id = data['items'][0]['id']
        players["CeeDee Lamb"] = lamb_id
        print(f"  CeeDee Lamb ID: {lamb_id}")
    
    time.sleep(DELAY)
    
    for name, pid in players.items():
        print(f"\n  --- {name} (ID: {pid}) ---")
        data = fetch(
            f"https://site.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{pid}"
        )
        if data and 'athlete' in data:
            ath = data['athlete']
            print(f"    displayDraft: {ath.get('displayDraft', 'NOT FOUND')}")
            print(f"    position: {ath.get('position', {}).get('abbreviation', 'N/A')}")
            print(f"    college: {ath.get('college', {}).get('name', 'N/A')}")
            print(f"    displayHeight: {ath.get('displayHeight', 'N/A')}")
            print(f"    displayWeight: {ath.get('displayWeight', 'N/A')}")
            if 'collegeAthlete' in ath:
                print(f"    collegeAthlete: {json.dumps(ath['collegeAthlete'])}")
        time.sleep(DELAY)
    
    # =========================================================
    # 4. NFL roster scrape - check how many players + fields
    #    Use one team, look for draft info on roster entries
    # =========================================================
    print("\n\n4. BENGALS ROSTER - CHECK FOR DRAFT INFO IN ROSTER ENTRIES")
    data = fetch(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/4/roster"
    )
    if data and 'athletes' in data:
        for group in data['athletes']:
            pos_group = group.get('position', 'unknown')
            items = group.get('items', [])
            print(f"\n  Position group: {pos_group} ({len(items)} players)")
            
            # Print first 2 players with ALL their fields
            for player in items[:2]:
                print(f"\n    Player: {player.get('displayName')}")
                print(f"    All keys: {list(player.keys())}")
                # Print specific fields
                for key in ['position', 'experience', 'draft', 'debutYear',
                           'displayHeight', 'displayWeight', 'age',
                           'college', 'birthPlace']:
                    if key in player:
                        val = player[key]
                        if isinstance(val, dict):
                            print(f"    {key}: {json.dumps(val)[:300]}")
                        else:
                            print(f"    {key}: {val}")
            
            # Only need to see offense
            if pos_group == 'offense':
                break
    
    time.sleep(DELAY)
    
    # =========================================================
    # 5. Chase NFL career stats - print ALL seasons clearly
    # =========================================================
    print("\n\n5. CHASE NFL STATS - ALL SEASONS")
    data = fetch(
        "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/4362628/stats"
    )
    if data and 'categories' in data:
        for cat in data['categories']:
            if cat['name'] == 'receiving':
                print(f"  Receiving stats:")
                print(f"  Labels: {cat['labels']}")
                print(f"  Names: {cat['names']}")
                for row in cat.get('statistics', []):
                    season = row['season']['displayName']
                    team = row['teamSlug']
                    stats = row['stats']
                    print(f"    {season} ({team}): {stats}")
                print(f"  Career totals: {cat.get('totals')}")
    
    time.sleep(DELAY)
    
    # =========================================================
    # 6. How many games in college - gamelog seasons available
    #    Chase played 2018 and 2019, opted out 2020
    # =========================================================
    print("\n\n6. CHASE COLLEGE GAMELOGS - COUNT GAMES PER SEASON")
    for year in [2018, 2019, 2020]:
        print(f"\n  Season {year}:")
        data = fetch(
            f"https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/4362628/gamelog?season={year}"
        )
        if data:
            events = data.get('events', {})
            print(f"    Games found: {len(events)}")
            
            # Get totals from seasonTypes
            for st in data.get('seasonTypes', []):
                for cat in st.get('categories', []):
                    if 'totals' in cat:
                        print(f"    {st.get('displayName', '')} totals: {cat['totals'][:5]}")
        time.sleep(DELAY)
    
    # =========================================================
    # 7. ALL 32 NFL TEAMS - just IDs and names
    # =========================================================
    print("\n\n7. ALL NFL TEAM IDS")
    data = fetch(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    )
    if data:
        for sport in data.get('sports', []):
            for league in sport.get('leagues', []):
                for t in league.get('teams', []):
                    team = t.get('team', t)
                    print(f"  {team.get('id'):>3}: {team.get('abbreviation'):>4} - {team.get('displayName')}")
    
    print(f"\n\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()