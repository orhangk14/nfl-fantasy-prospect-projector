# data_collection/test_combine_sources.py
"""
Test sources for NFL Combine athletic testing data.
Run: python -m data_collection.test_combine_sources
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


def test(url, label):
    print(f"\n{'=' * 70}")
    print(f"📥 {label}")
    print(f"   {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        print(f"   Status: {resp.status_code} | Size: {len(resp.text)} chars")
        text = resp.text
        if "just a moment" in text.lower() or "cloudflare" in text.lower():
            print("   ⛔ CLOUDFLARE")
            return
        if resp.status_code != 200:
            print(f"   {text[:500]}")
            return
        if 'json' in resp.headers.get('content-type', ''):
            data = resp.json()
            print(f"   Type: {type(data)}")
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"   Length: {len(data)}")
            print(json.dumps(data, indent=2)[:3000])
        else:
            print(text[:2000])
    except Exception as e:
        print(f"   ❌ {e}")


def main():
    print("COMBINE DATA SOURCE TEST")
    print("=" * 70)

    tests = [
        # NFL.com combine
        (
            "https://nfl-api.com/api/draft/combine?year=2024",
            "NFL API: 2024 Combine"
        ),
        (
            "https://api.nfl.com/v3/shield/?query=%7B%20combine(year%3A2024)%20%7B%20players%20%7B%20firstName%20lastName%20position%20fortyYardDash%20%7D%20%7D%20%7D",
            "NFL Shield API: 2024 Combine GraphQL"
        ),
        
        # ESPN core API combine attempts
        (
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/draft/rounds/1/picks?limit=32",
            "ESPN Core: 2024 Draft Round 1 Picks"
        ),
        (
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2025/draft/rounds/1/picks?limit=32",
            "ESPN Core: 2025 Draft Round 1 Picks"
        ),
        
        # Try undocumented ESPN combine
        (
            "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/4362628/bio",
            "ESPN: Chase Bio (might have combine)"
        ),
        
        # The athlete detail had displayHeight/Weight
        # but let's check if the core API has more measurables
        (
            "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/athletes/4362628",
            "ESPN Core: Chase athlete (check for combine fields)"
        ),
        
        # Try a public combine dataset API
        (
            "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p=Jamarr%20Chase",
            "TheSportsDB: Search Chase"
        ),
    ]

    for i, (url, label) in enumerate(tests):
        test(url, label)
        if i < len(tests) - 1:
            time.sleep(DELAY)

    print(f"\n\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()