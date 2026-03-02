# feature_engineering/build_features.py
"""
Processes all raw CSVs into clean player profiles.
Outputs: data/processed/player_profiles.csv (one row per player)

Run: python -m feature_engineering.build_features
"""

import csv
import os
import re
import json
from collections import defaultdict

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

FANTASY_POS = {'QB', 'RB', 'WR', 'TE'}

# Map historic combine positions to our standard
POS_MAP = {
    'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'FB': 'RB',
}


def load_csv(path):
    full = os.path.join(RAW_DIR, path) if not path.startswith('data') else path
    rows = []
    with open(full, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val):
    if not val or val.strip() in ('', '--', 'N/A'):
        return None
    try:
        return float(val.replace(',', ''))
    except (ValueError, TypeError):
        return None


def parse_height_inches(height_str):
    """Convert '6\' 2"' or '6\'2.5"' to inches."""
    if not height_str:
        return None
    clean = height_str.replace('\\', '').replace('"', '').strip()
    match = re.match(r"(\d+)['\s]+(\d+\.?\d*)", clean)
    if match:
        return int(match.group(1)) * 12 + float(match.group(2))
    return safe_float(height_str)


def parse_weight_lbs(weight_str):
    if not weight_str:
        return None
    match = re.match(r"(\d+)", weight_str)
    return int(match.group(1)) if match else None


def build_college_profiles(college_stats, college_gp, is_prospect=False):
    """
    Build per-player college profiles from stat rows and GP data.
    Returns dict: espn_id -> profile dict
    """
    # Index GP by (espn_id, season)
    gp_lookup = {}
    for row in college_gp:
        key = (row['espn_id'], str(row['season']))
        gp_lookup[key] = safe_float(row['regular_season_games']) or 0

    # Group stats by player
    player_stats = defaultdict(lambda: defaultdict(dict))
    player_meta = {}

    for row in college_stats:
        pid = row['espn_id']
        season = str(row['season'])
        cat = row['stat_category']

        if pid not in player_meta:
            player_meta[pid] = {
                'name': row['name'],
                'position': row['position'],
            }
            if is_prospect:
                player_meta[pid]['school'] = row.get('school', '')
                player_meta[pid]['height'] = row.get('height', '')
                player_meta[pid]['weight'] = row.get('weight', '')

        if cat == 'receiving':
            player_stats[pid][season]['rec'] = safe_float(row.get('receptions'))
            player_stats[pid][season]['rec_yds'] = safe_float(row.get('receivingYards'))
            player_stats[pid][season]['rec_td'] = safe_float(row.get('receivingTouchdowns'))
            player_stats[pid][season]['ypr'] = safe_float(row.get('yardsPerReception'))
            player_stats[pid][season]['long_rec'] = safe_float(row.get('longReception'))
        elif cat == 'rushing':
            player_stats[pid][season]['rush_att'] = safe_float(row.get('rushingAttempts'))
            player_stats[pid][season]['rush_yds'] = safe_float(row.get('rushingYards'))
            player_stats[pid][season]['rush_td'] = safe_float(row.get('rushingTouchdowns'))
            player_stats[pid][season]['ypc'] = safe_float(row.get('yardsPerRushAttempt'))
            player_stats[pid][season]['long_rush'] = safe_float(row.get('longRushing'))
        elif cat == 'passing':
            player_stats[pid][season]['cmp'] = safe_float(row.get('completions'))
            player_stats[pid][season]['pass_att'] = safe_float(row.get('passingAttempts'))
            player_stats[pid][season]['cmp_pct'] = safe_float(row.get('completionPct'))
            player_stats[pid][season]['pass_yds'] = safe_float(row.get('passingYards'))
            player_stats[pid][season]['pass_td'] = safe_float(row.get('passingTouchdowns'))
            player_stats[pid][season]['int'] = safe_float(row.get('interceptions'))
            player_stats[pid][season]['ypa'] = safe_float(row.get('yardsPerPassAttempt'))
            player_stats[pid][season]['qb_rating'] = safe_float(row.get('QBRating'))
            player_stats[pid][season]['sacks'] = safe_float(row.get('sacks'))

    # Build profiles
    profiles = {}

    for pid, seasons_data in player_stats.items():
        meta = player_meta.get(pid, {})
        pos = meta.get('position', '')
        if pos not in FANTASY_POS:
            continue

        season_list = []
        for season, stats in sorted(seasons_data.items()):
            gp = gp_lookup.get((pid, season), None)
            stats['season'] = int(season)
            stats['gp'] = gp
            season_list.append(stats)

        if not season_list:
            continue

        profile = _compute_profile(season_list, pos)
        profile['espn_id'] = pid
        profile['name'] = meta.get('name', '')
        profile['position'] = pos
        profile['is_prospect'] = is_prospect

        if is_prospect:
            profile['school'] = meta.get('school', '')

        profiles[pid] = profile

    return profiles


def _compute_profile(seasons, pos):
    """
    Compute career profile from season list.
    
    v2: Adds peak season per-game stats as separate features.
    Best season gets its own feature set for similarity matching,
    capturing guys like Ja'Marr Chase / Puka Nacua who had one
    dominant season surrounded by quiet ones.
    
    Weights for career averages: best 50%, 2nd 30%, rest 20%.
    """
    profile = {}

    # Compute per-season fantasy PPG (college PPR proxy)
    for s in seasons:
        gp = s.get('gp') or 1
        if gp == 0:
            gp = 1

        rec = s.get('rec', 0) or 0
        rec_yds = s.get('rec_yds', 0) or 0
        rec_td = s.get('rec_td', 0) or 0
        rush_att = s.get('rush_att', 0) or 0
        rush_yds = s.get('rush_yds', 0) or 0
        rush_td = s.get('rush_td', 0) or 0
        pass_yds = s.get('pass_yds', 0) or 0
        pass_td = s.get('pass_td', 0) or 0
        ints = s.get('int', 0) or 0

        if pos == 'QB':
            ppr = (pass_yds * 0.04 + pass_td * 4 - ints * 2 +
                   rush_yds * 0.1 + rush_td * 6)
        else:
            ppr = (rec * 1.0 + rec_yds * 0.1 + rec_td * 6 +
                   rush_yds * 0.1 + rush_td * 6)

        s['ppr_total'] = ppr
        s['ppr_ppg'] = ppr / gp
        s['gp_val'] = gp

        # Per-game rates
        s['rec_pg'] = rec / gp
        s['rec_yds_pg'] = rec_yds / gp
        s['rec_td_pg'] = rec_td / gp
        s['rush_att_pg'] = rush_att / gp
        s['rush_yds_pg'] = rush_yds / gp
        s['rush_td_pg'] = rush_td / gp

        if pos == 'QB':
            s['pass_att_pg'] = (s.get('pass_att', 0) or 0) / gp
            s['pass_yds_pg'] = pass_yds / gp
            s['pass_td_pg'] = pass_td / gp
            s['rush_yds_pg_qb'] = rush_yds / gp

    # Sort by PPR PPG descending (best season first)
    sorted_seasons = sorted(seasons, key=lambda x: x.get('ppr_ppg', 0), reverse=True)

    # ─── Weighted career average ────────────────────────────────────
    n = len(sorted_seasons)
    if n == 1:
        weights = [1.0]
    elif n == 2:
        weights = [0.60, 0.40]
    elif n == 3:
        weights = [0.50, 0.30, 0.20]
    else:
        weights = [0.50, 0.30]
        remaining = 0.20 / (n - 2)
        weights.extend([remaining] * (n - 2))

    numeric_keys = [
        'ppr_ppg', 'rec_pg', 'rec_yds_pg', 'rec_td_pg',
        'rush_att_pg', 'rush_yds_pg', 'rush_td_pg',
        'ypr', 'ypc',
    ]
    if pos == 'QB':
        numeric_keys += [
            'pass_att_pg', 'pass_yds_pg', 'pass_td_pg',
            'cmp_pct', 'ypa', 'qb_rating', 'rush_yds_pg_qb',
        ]

    for key in numeric_keys:
        values = [(s.get(key), w) for s, w in zip(sorted_seasons, weights)
                  if s.get(key) is not None]
        if values:
            total_w = sum(w for _, w in values)
            profile[f'college_{key}'] = sum(v * w for v, w in values) / total_w

    # ─── Peak season stats (BEST single season per-game) ───────────
    best = sorted_seasons[0]
    best_gp = best.get('gp_val', 1) or 1

    profile['peak_ppr_ppg'] = best.get('ppr_ppg', 0)
    profile['peak_season_year'] = best.get('season', 0)
    profile['peak_gp'] = best_gp

    # Peak totals
    profile['peak_rec'] = best.get('rec', 0) or 0
    profile['peak_rec_yds'] = best.get('rec_yds', 0) or 0
    profile['peak_rec_td'] = best.get('rec_td', 0) or 0
    profile['peak_rush_yds'] = best.get('rush_yds', 0) or 0
    profile['peak_rush_td'] = best.get('rush_td', 0) or 0

    # Peak per-game rates
    profile['peak_rec_pg'] = best.get('rec_pg', 0)
    profile['peak_rec_yds_pg'] = best.get('rec_yds_pg', 0)
    profile['peak_rec_td_pg'] = best.get('rec_td_pg', 0)
    profile['peak_rush_att_pg'] = best.get('rush_att_pg', 0)
    profile['peak_rush_yds_pg'] = best.get('rush_yds_pg', 0)
    profile['peak_rush_td_pg'] = best.get('rush_td_pg', 0)
    profile['peak_ypr'] = best.get('ypr')
    profile['peak_ypc'] = best.get('ypc')

    if pos == 'QB':
        profile['peak_pass_yds'] = best.get('pass_yds', 0) or 0
        profile['peak_pass_td'] = best.get('pass_td', 0) or 0
        profile['peak_cmp_pct'] = best.get('cmp_pct', 0) or 0
        profile['peak_ypa'] = best.get('ypa', 0) or 0
        profile['peak_qb_rating'] = best.get('qb_rating', 0) or 0
        profile['peak_pass_yds_pg'] = best.get('pass_yds_pg', 0)
        profile['peak_pass_td_pg'] = best.get('pass_td_pg', 0)
        profile['peak_rush_yds_pg_qb'] = best.get('rush_yds_pg_qb', 0)

    # ─── Peak season TD rate ───────────────────────────────────────
    peak_touches_pg = (best.get('rec_pg', 0) or 0) + (best.get('rush_att_pg', 0) or 0)
    peak_td_pg = (best.get('rec_td_pg', 0) or 0) + (best.get('rush_td_pg', 0) or 0)
    if peak_touches_pg > 0:
        profile['peak_td_rate'] = peak_td_pg / peak_touches_pg
    else:
        profile['peak_td_rate'] = None

    # ─── Breakout ratio: peak PPG vs career avg PPG ────────────────
    career_ppg = profile.get('college_ppr_ppg', 0) or 0
    peak_ppg = profile.get('peak_ppr_ppg', 0) or 0
    if career_ppg > 0:
        profile['breakout_ratio'] = peak_ppg / career_ppg
    else:
        profile['breakout_ratio'] = None

    # ─── Last season stats (most recent = most relevant) ───────────
    by_year = sorted(seasons, key=lambda x: x.get('season', 0), reverse=True)
    last = by_year[0]
    profile['last_season_ppr_ppg'] = last.get('ppr_ppg', 0)
    profile['last_season_year'] = last.get('season', 0)
    profile['last_season_rec_pg'] = last.get('rec_pg', 0)
    profile['last_season_rec_yds_pg'] = last.get('rec_yds_pg', 0)

    if pos == 'QB':
        profile['last_season_pass_yds_pg'] = last.get('pass_yds_pg', 0)
        profile['last_season_pass_td_pg'] = last.get('pass_td_pg', 0)

    # ─── Career totals ─────────────────────────────────────────────
    profile['total_college_seasons'] = n
    profile['total_college_games'] = sum(s.get('gp_val', 0) for s in seasons)

    threshold = 10 if pos == 'QB' else 5
    profile['productive_seasons'] = sum(
        1 for s in seasons if s.get('ppr_ppg', 0) > threshold
    )

    # ─── Season-over-season improvement ────────────────────────────
    by_year_ppg = [(s.get('season', 0), s.get('ppr_ppg', 0)) for s in seasons
                   if s.get('season') and s.get('ppr_ppg')]
    by_year_ppg.sort()
    if len(by_year_ppg) >= 2:
        first_ppg = by_year_ppg[0][1]
        last_ppg = by_year_ppg[-1][1]
        if first_ppg > 0:
            profile['improvement_ratio'] = last_ppg / first_ppg
        else:
            profile['improvement_ratio'] = None
    else:
        profile['improvement_ratio'] = None

    return profile


def build_nfl_outcomes(nfl_stats):
    """Build NFL fantasy outcomes per player."""
    player_seasons = defaultdict(lambda: defaultdict(dict))
    player_meta = {}

    for row in nfl_stats:
        pid = row['espn_id']
        season = str(row['season'])
        cat = row['stat_category']

        if pid not in player_meta:
            player_meta[pid] = {
                'name': row['name'],
                'position': row['position'],
            }

        gp = safe_float(row.get('gamesPlayed'))
        if gp:
            player_seasons[pid][season]['gp'] = gp

        if cat == 'receiving':
            player_seasons[pid][season]['rec'] = safe_float(row.get('receptions'))
            player_seasons[pid][season]['targets'] = safe_float(row.get('receivingTargets'))
            player_seasons[pid][season]['rec_yds'] = safe_float(row.get('receivingYards'))
            player_seasons[pid][season]['rec_td'] = safe_float(row.get('receivingTouchdowns'))
        elif cat == 'rushing':
            player_seasons[pid][season]['rush_att'] = safe_float(row.get('rushingAttempts'))
            player_seasons[pid][season]['rush_yds'] = safe_float(row.get('rushingYards'))
            player_seasons[pid][season]['rush_td'] = safe_float(row.get('rushingTouchdowns'))
        elif cat == 'passing':
            player_seasons[pid][season]['pass_yds'] = safe_float(row.get('passingYards'))
            player_seasons[pid][season]['pass_td'] = safe_float(row.get('passingTouchdowns'))
            player_seasons[pid][season]['ints'] = safe_float(row.get('interceptions'))
            player_seasons[pid][season]['rush_yds_qb'] = safe_float(row.get('rushingYards'))
            player_seasons[pid][season]['rush_td_qb'] = safe_float(row.get('rushingTouchdowns'))

    outcomes = {}

    for pid, seasons_data in player_seasons.items():
        meta = player_meta.get(pid, {})
        pos = meta.get('position', '')

        season_pprs = []
        for season, stats in sorted(seasons_data.items()):
            gp = stats.get('gp', 0) or 0
            if gp == 0:
                continue

            rec = stats.get('rec', 0) or 0
            rec_yds = stats.get('rec_yds', 0) or 0
            rec_td = stats.get('rec_td', 0) or 0
            rush_yds = stats.get('rush_yds', 0) or 0
            rush_td = stats.get('rush_td', 0) or 0

            if pos == 'QB':
                pass_yds = stats.get('pass_yds', 0) or 0
                pass_td = stats.get('pass_td', 0) or 0
                ints = stats.get('ints', 0) or 0
                qb_rush_yds = stats.get('rush_yds_qb', 0) or rush_yds
                qb_rush_td = stats.get('rush_td_qb', 0) or rush_td

                ppr_total = (pass_yds * 0.04 + pass_td * 4 - ints * 2 +
                             qb_rush_yds * 0.1 + qb_rush_td * 6 +
                             rec * 1.0 + rec_yds * 0.1 + rec_td * 6)
            else:
                ppr_total = (rec * 1.0 + rec_yds * 0.1 + rec_td * 6 +
                             rush_yds * 0.1 + rush_td * 6)

            ppg = ppr_total / gp
            season_pprs.append({
                'season': int(season),
                'gp': gp,
                'ppr_total': ppr_total,
                'ppr_ppg': ppg,
            })

        if not season_pprs:
            continue

        # Sort by season year
        season_pprs.sort(key=lambda x: x['season'])

        # Rookie year = first season
        rookie = season_pprs[0]

        # Best season
        best = max(season_pprs, key=lambda x: x['ppr_ppg'])

        # Peak 3-year window
        ppg_values = [s['ppr_ppg'] for s in season_pprs]
        if len(ppg_values) >= 3:
            peak_3yr = max(
                sum(ppg_values[i:i+3]) / 3
                for i in range(len(ppg_values) - 2)
            )
        elif len(ppg_values) >= 1:
            peak_3yr = sum(ppg_values) / len(ppg_values)
        else:
            peak_3yr = 0

        # Dynasty value = average PPG over first 5 seasons (or career if shorter)
        first_5 = season_pprs[:5]
        dynasty_ppg = sum(s['ppr_ppg'] for s in first_5) / len(first_5) if first_5 else 0

        outcomes[pid] = {
            'nfl_seasons_played': len(season_pprs),
            'rookie_ppr_ppg': rookie['ppr_ppg'],
            'rookie_ppr_total': rookie['ppr_total'],
            'rookie_gp': rookie['gp'],
            'best_season_ppr_ppg': best['ppr_ppg'],
            'best_season_ppr_total': best['ppr_total'],
            'peak_3yr_ppr_ppg': peak_3yr,
            'dynasty_ppg': dynasty_ppg,
            'career_ppr_total': sum(s['ppr_total'] for s in season_pprs),
        }

    return outcomes


def build_combine_lookup(historic_combine):
    """Build lookup: (name_lower, college_lower) -> combine data."""
    lookup = {}
    for row in historic_combine:
        pos = POS_MAP.get(row.get('POS', ''))
        if not pos:
            continue
        name = row.get('Name', '').strip().lower()
        college = row.get('College', '').strip().lower()
        key = (name, college)
        lookup[key] = {
            'combine_height': safe_float(row.get('Height (in)')),
            'combine_weight': safe_float(row.get('Weight (lbs)')),
            'combine_40': safe_float(row.get('40 Yard')),
            'combine_bench': safe_float(row.get('Bench Press')),
            'combine_vert': safe_float(row.get('Vert Leap (in)')),
            'combine_broad': safe_float(row.get('Broad Jump (in)')),
            'combine_shuttle': safe_float(row.get('Shuttle')),
            'combine_3cone': safe_float(row.get('3Cone')),
            'combine_hand': safe_float(row.get('Hand Size (in)')),
            'combine_arm': safe_float(row.get('Arm Length (in)')),
        }
    return lookup


def build_2026_combine_lookup(combine_2026):
    """Build lookup for 2026 prospects: name_lower -> combine data."""
    lookup = {}
    for row in combine_2026:
        name = row.get('Name', '').strip().lower()
        lookup[name] = {
            'combine_height': safe_float(row.get('Height_in')),
            'combine_weight': safe_float(row.get('Weight_lbs')),
            'combine_40': safe_float(row.get('40_Yard')),
            'combine_bench': safe_float(row.get('Bench_Press')),
            'combine_vert': safe_float(row.get('Vert_Leap_in')),
            'combine_broad': safe_float(row.get('Broad_Jump_in')),
            'combine_shuttle': safe_float(row.get('Shuttle')),
            'combine_3cone': safe_float(row.get('3Cone')),
            'combine_hand': safe_float(row.get('Hand_Size_in')),
            'combine_arm': safe_float(row.get('Arm_Length_in')),
            'big_board_rank': safe_float(row.get('Big_Board_Rank')),
        }
    return lookup


def classify_archetype(profile):
    """Classify player into archetype based on stats + measurables."""
    pos = profile.get('position', '')

    if pos == 'WR':
        height = profile.get('combine_height') or profile.get('height_inches')
        weight = profile.get('combine_weight') or profile.get('weight_lbs')
        forty = profile.get('combine_40')
        ypr = profile.get('college_ypr', 0) or 0
        rec_pg = profile.get('college_rec_pg', 0) or 0
        rush_yds_pg = profile.get('college_rush_yds_pg', 0) or 0

        scores = {'X_OUTSIDE': 0, 'SLOT': 0, 'DEEP_THREAT': 0, 'YAC_GADGET': 0}

        if height and height >= 73:
            scores['X_OUTSIDE'] += 2
        if height and height <= 71:
            scores['SLOT'] += 2

        if weight and weight >= 210:
            scores['X_OUTSIDE'] += 1
        if weight and weight <= 190:
            scores['SLOT'] += 1
            scores['YAC_GADGET'] += 1

        if forty and forty <= 4.38:
            scores['DEEP_THREAT'] += 3
        elif forty and forty <= 4.45:
            scores['DEEP_THREAT'] += 1

        if ypr >= 17:
            scores['DEEP_THREAT'] += 2
        elif ypr >= 14:
            scores['X_OUTSIDE'] += 1
        elif ypr <= 11:
            scores['SLOT'] += 2

        if rec_pg >= 5.5:
            scores['SLOT'] += 2
        elif rec_pg >= 4:
            scores['SLOT'] += 1

        if rush_yds_pg >= 5:
            scores['YAC_GADGET'] += 3
        elif rush_yds_pg >= 2:
            scores['YAC_GADGET'] += 1

        sorted_arch = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_arch[0][0], scores

    elif pos == 'RB':
        ypc = profile.get('college_ypc', 0) or 0
        rec_pg = profile.get('college_rec_pg', 0) or 0
        rush_att_pg = profile.get('college_rush_att_pg', 0) or 0
        forty = profile.get('combine_40')
        weight = profile.get('combine_weight') or profile.get('weight_lbs')

        scores = {'WORKHORSE': 0, 'SPEED_BACK': 0, 'RECEIVING_BACK': 0, 'POWER_BACK': 0}

        if rush_att_pg >= 16:
            scores['WORKHORSE'] += 2
        if rush_att_pg >= 12 and rec_pg >= 2:
            scores['WORKHORSE'] += 2

        if rec_pg >= 3:
            scores['RECEIVING_BACK'] += 3
        elif rec_pg >= 2:
            scores['RECEIVING_BACK'] += 1

        if ypc >= 6:
            scores['SPEED_BACK'] += 2
        if forty and forty <= 4.42:
            scores['SPEED_BACK'] += 2

        if weight and weight >= 220:
            scores['POWER_BACK'] += 2
        if ypc and ypc <= 4.5:
            scores['POWER_BACK'] += 1

        sorted_arch = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_arch[0][0], scores

    elif pos == 'QB':
        rush_yds_pg = profile.get('college_rush_yds_pg', 0) or profile.get('college_rush_yds_pg_qb', 0) or 0
        cmp_pct = profile.get('college_cmp_pct', 0) or 0
        ypa = profile.get('college_ypa', 0) or 0
        pass_att_pg = profile.get('college_pass_att_pg', 0) or 0

        scores = {'POCKET_PASSER': 0, 'DUAL_THREAT': 0, 'SCRAMBLER': 0}

        if rush_yds_pg >= 40:
            scores['DUAL_THREAT'] += 3
        elif rush_yds_pg >= 20:
            scores['SCRAMBLER'] += 2

        if rush_yds_pg < 15:
            scores['POCKET_PASSER'] += 2

        if cmp_pct >= 65:
            scores['POCKET_PASSER'] += 2
        if ypa >= 8.5:
            scores['POCKET_PASSER'] += 1

        if pass_att_pg >= 30:
            scores['POCKET_PASSER'] += 1

        sorted_arch = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_arch[0][0], scores

    elif pos == 'TE':
        rec_pg = profile.get('college_rec_pg', 0) or 0
        ypr = profile.get('college_ypr', 0) or 0
        rec_td_pg = profile.get('college_rec_td_pg', 0) or 0
        forty = profile.get('combine_40')
        weight = profile.get('combine_weight') or profile.get('weight_lbs')

        scores = {'RECEIVING_TE': 0, 'ATHLETIC_TE': 0, 'BLOCKING_TE': 0}

        if rec_pg >= 3.5:
            scores['RECEIVING_TE'] += 3
        elif rec_pg >= 2:
            scores['RECEIVING_TE'] += 1

        if rec_pg < 2:
            scores['BLOCKING_TE'] += 2

        if forty and forty <= 4.55:
            scores['ATHLETIC_TE'] += 3
        if ypr and ypr >= 14:
            scores['ATHLETIC_TE'] += 1

        if weight and weight >= 255:
            scores['BLOCKING_TE'] += 1

        sorted_arch = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_arch[0][0], scores

    return 'UNKNOWN', {}


def load_mock_draft():
    """Load mock draft capital for 2026 prospects."""
    path = os.path.join(RAW_DIR, "mock_draft_2026.csv")
    if not os.path.exists(path):
        print("  ⚠️  No mock draft file found. Run: python -m data_collection.parse_mock_draft")
        return {}

    lookup = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row['espn_id']] = {
                'mock_pick': safe_float(row.get('mock_pick')),
                'mock_round': safe_float(row.get('mock_round')),
                'is_estimate': row.get('is_estimate', '') == 'True',
            }
    return lookup


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 70)
    print("BUILDING PLAYER PROFILES")
    print("=" * 70)

    # Load all data
    print("\n📂 Loading raw data...")
    roster = load_csv("nfl_roster_players.csv")
    details = load_csv("player_details.csv")
    college_stats = load_csv("college/college_stats.csv")
    college_gp = load_csv("college/college_games_played.csv")
    nfl_stats = load_csv("nfl/nfl_stats.csv")
    prospect_stats = load_csv("college/prospects_2026_college_stats.csv")
    prospect_gp = load_csv("college/prospects_2026_games_played.csv")
    prospect_ids = load_csv("prospects_2026_ids.csv")
    combine_historic = load_csv("college/historiccombinedata.csv")
    combine_2026 = load_csv("combine_2026_fantasy.csv")

    # Build detail lookup
    detail_lookup = {r['espn_id']: r for r in details}

    # Build combine lookups
    print("📊 Building combine lookups...")
    historic_combine = build_combine_lookup(combine_historic)
    prospect_combine = build_2026_combine_lookup(combine_2026)
    print(f"   Historic combine entries (fantasy pos): {len(historic_combine)}")
    print(f"   2026 combine entries: {len(prospect_combine)}")

    # Load mock draft
    print("📊 Loading mock draft capital...")
    mock_draft = load_mock_draft()
    print(f"   Mock draft entries: {len(mock_draft)}")

    # Build college profiles for NFL players
    print("📊 Building NFL player college profiles...")
    nfl_college_profiles = build_college_profiles(college_stats, college_gp, is_prospect=False)
    print(f"   NFL players with college profiles: {len(nfl_college_profiles)}")

    # Build NFL outcomes
    print("📊 Building NFL outcomes...")
    nfl_outcomes = build_nfl_outcomes(nfl_stats)
    print(f"   NFL players with outcomes: {len(nfl_outcomes)}")

    # Build 2026 prospect profiles
    print("📊 Building 2026 prospect profiles...")
    prospect_profiles = build_college_profiles(prospect_stats, prospect_gp, is_prospect=True)
    print(f"   2026 prospects with profiles: {len(prospect_profiles)}")

    # Merge everything into final profiles
    print("\n📊 Merging all data...")
    all_profiles = []

    # NFL players
    for pid, college_profile in nfl_college_profiles.items():
        detail = detail_lookup.get(pid, {})
        outcome = nfl_outcomes.get(pid, {})

        profile = {**college_profile}

        # Add detail info
        profile['height_inches'] = parse_height_inches(detail.get('height', ''))
        profile['weight_lbs'] = parse_weight_lbs(detail.get('weight', ''))
        profile['draft_year'] = safe_float(detail.get('draft_year'))
        profile['draft_round'] = safe_float(detail.get('draft_round'))
        profile['draft_pick'] = safe_float(detail.get('draft_pick'))
        profile['college'] = detail.get('college', '')

        # Add combine data
        name_lower = profile.get('name', '').strip().lower()
        college_lower = profile.get('college', '').strip().lower()
        combine = historic_combine.get((name_lower, college_lower), {})
        for k, v in combine.items():
            profile[k] = v

        # If no combine match, use roster height/weight
        if not profile.get('combine_height') and profile.get('height_inches'):
            profile['combine_height'] = profile['height_inches']
        if not profile.get('combine_weight') and profile.get('weight_lbs'):
            profile['combine_weight'] = profile['weight_lbs']

        # Add NFL outcomes
        for k, v in outcome.items():
            profile[k] = v

        # Classify archetype
        archetype, arch_scores = classify_archetype(profile)
        profile['archetype'] = archetype
        profile['archetype_scores'] = json.dumps(arch_scores)

        all_profiles.append(profile)

    # 2026 prospects
    for pid, college_profile in prospect_profiles.items():
        profile = {**college_profile}

        # Add combine data
        name_lower = profile.get('name', '').strip().lower()
        combine = prospect_combine.get(name_lower, {})
        for k, v in combine.items():
            profile[k] = v

        # Height/weight from prospect list
        prospect_info = next((p for p in prospect_ids if p['espn_id'] == pid), {})
        if not profile.get('combine_height'):
            profile['combine_height'] = parse_height_inches(prospect_info.get('height', ''))
            profile['height_inches'] = profile['combine_height']
        if not profile.get('combine_weight'):
            profile['combine_weight'] = parse_weight_lbs(prospect_info.get('weight', ''))
            profile['weight_lbs'] = profile['combine_weight']

        if not profile.get('college') and not profile.get('school'):
            profile['school'] = prospect_info.get('school', '')

        # ── Mock draft capital ──
        mock = mock_draft.get(pid, {})
        if mock:
            profile['draft_pick'] = mock.get('mock_pick')
            profile['draft_round'] = mock.get('mock_round')
            profile['draft_capital_is_mock'] = True
            profile['draft_capital_is_estimate'] = mock.get('is_estimate', False)
        else:
            # No mock data — assign UDFA range
            profile['draft_pick'] = 224  # UDFA proxy
            profile['draft_round'] = 7
            profile['draft_capital_is_mock'] = True
            profile['draft_capital_is_estimate'] = True

        # No NFL outcomes for prospects
        profile['nfl_seasons_played'] = None
        profile['rookie_ppr_ppg'] = None
        profile['best_season_ppr_ppg'] = None
        profile['peak_3yr_ppr_ppg'] = None
        profile['dynasty_ppg'] = None

        # Classify archetype
        archetype, arch_scores = classify_archetype(profile)
        profile['archetype'] = archetype
        profile['archetype_scores'] = json.dumps(arch_scores)

        all_profiles.append(profile)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "player_profiles.json")
    with open(output_path, 'w') as f:
        json.dump(all_profiles, f, indent=2, default=str)

    # Also save as CSV for inspection
    csv_path = os.path.join(PROCESSED_DIR, "player_profiles.csv")
    if all_profiles:
        all_keys = []
        seen = set()
        for p in all_profiles:
            for k in p.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_profiles)

    # Summary
    nfl_count = sum(1 for p in all_profiles if not p.get('is_prospect'))
    prospect_count = sum(1 for p in all_profiles if p.get('is_prospect'))
    with_combine = sum(1 for p in all_profiles if p.get('combine_40'))
    with_outcomes = sum(1 for p in all_profiles if p.get('rookie_ppr_ppg') is not None)
    with_draft = sum(1 for p in all_profiles if p.get('draft_pick') is not None)
    with_mock = sum(1 for p in all_profiles if p.get('draft_capital_is_mock'))

    print(f"\n{'=' * 70}")
    print(f"✅ PROFILES BUILT")
    print(f"   Total: {len(all_profiles)}")
    print(f"   NFL players: {nfl_count}")
    print(f"   2026 prospects: {prospect_count}")
    print(f"   With combine 40-yard: {with_combine}")
    print(f"   With NFL outcomes: {with_outcomes}")
    print(f"   With draft capital: {with_draft}")
    print(f"   With mock draft capital: {with_mock}")
    print(f"\n   Archetype breakdown:")
    from collections import Counter
    archetypes = Counter(p.get('archetype') for p in all_profiles)
    for arch, count in archetypes.most_common():
        print(f"     {arch}: {count}")
    print(f"\n   Saved: {output_path}")
    print(f"   Saved: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()