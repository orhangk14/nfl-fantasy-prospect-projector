# modeling/similarity.py
"""
Historical player comparison engine.
Finds the most similar historical NFL players to each 2026 prospect
using weighted cosine similarity across production, measurables,
and draft capital.

v2: Reweighted to emphasize college efficiency/dominance over raw
combine numbers. Captures guys like Puka Nacua / JSN who had elite
efficiency but not necessarily elite combine measurables.

Run standalone test: python -m modeling.similarity
"""

import json
import os
import math
from collections import defaultdict

PROCESSED_DIR = "data/processed"


# ─── Feature definitions per position ──────────────────────────────────────
# Categories: 'production', 'peak', 'efficiency', 'measurable', 'draft'

COMMON_MEASURABLES = [
    ('combine_height', 'measurable', True),
    ('combine_weight', 'measurable', True),
    ('combine_40', 'measurable', False),
    ('combine_vert', 'measurable', True),
    ('combine_broad', 'measurable', True),
    ('combine_shuttle', 'measurable', False),
    ('combine_3cone', 'measurable', False),
]

DRAFT_FEATURES = [
    ('draft_round', 'draft', False),
    ('draft_pick', 'draft', False),
]

WR_FEATURES = COMMON_MEASURABLES + DRAFT_FEATURES + [
    # Career production (weighted avg)
    ('college_ppr_ppg', 'production', True),
    ('college_rec_pg', 'production', True),
    ('college_rec_yds_pg', 'production', True),
    ('college_rec_td_pg', 'production', True),
    ('college_rush_yds_pg', 'production', True),
    ('total_college_games', 'production', True),
    ('productive_seasons', 'production', True),
    # Peak season (best single year — catches Ja'Marr Chase / Puka types)
    ('peak_ppr_ppg', 'peak', True),
    ('peak_rec_pg', 'peak', True),
    ('peak_rec_yds_pg', 'peak', True),
    ('peak_rec_td_pg', 'peak', True),
    ('peak_ypr', 'peak', True),
    ('peak_rec_yds', 'peak', True),
    ('peak_td_rate', 'peak', True),
    # Efficiency / context
    ('college_ypr', 'efficiency', True),
    ('rec_yds_per_team_game', 'efficiency', True),
    ('college_td_rate', 'efficiency', True),
    ('breakout_ratio', 'efficiency', True),
    ('last_season_ppr_ppg', 'efficiency', True),
    # v3: Peak dominance & draft interactions
    ('peak_to_career_ratio', 'peak', True),
    ('peak_to_last_ratio', 'peak', True),
    ('peak_season_share', 'peak', False),       # lower = more consistent (good)
    ('draft_pick_log', 'draft', False),
    ('draft_x_peak_ppg', 'draft', True),
    ('draft_x_college_ppg', 'draft', True),
    ('snap_share_proxy', 'draft', True),
    ('opportunity_adjusted_peak', 'draft', True),
    ('trend_direction', 'efficiency', True),
    ('peak_recency', 'efficiency', False),       # lower = more recent peak (good)
]

RB_FEATURES = COMMON_MEASURABLES + DRAFT_FEATURES + [
    # Career production
    ('college_ppr_ppg', 'production', True),
    ('college_rush_att_pg', 'production', True),
    ('college_rush_yds_pg', 'production', True),
    ('college_rush_td_pg', 'production', True),
    ('college_rec_pg', 'production', True),
    ('college_rec_yds_pg', 'production', True),
    ('total_college_games', 'production', True),
    ('productive_seasons', 'production', True),
    # Peak season
    ('peak_ppr_ppg', 'peak', True),
    ('peak_rush_yds_pg', 'peak', True),
    ('peak_rush_td_pg', 'peak', True),
    ('peak_rush_att_pg', 'peak', True),
    ('peak_ypc', 'peak', True),
    ('peak_rush_yds', 'peak', True),
    ('peak_td_rate', 'peak', True),
    # Efficiency
    ('college_ypc', 'efficiency', True),
    ('college_ypr', 'efficiency', True),
    ('rush_yds_per_team_game', 'efficiency', True),
    ('college_td_rate', 'efficiency', True),
    ('breakout_ratio', 'efficiency', True),
    ('last_season_ppr_ppg', 'efficiency', True),
    # v3: Peak dominance & draft interactions
    ('peak_to_career_ratio', 'peak', True),
    ('peak_to_last_ratio', 'peak', True),
    ('peak_season_share', 'peak', False),
    ('draft_pick_log', 'draft', False),
    ('draft_x_peak_ppg', 'draft', True),
    ('draft_x_college_ppg', 'draft', True),
    ('snap_share_proxy', 'draft', True),
    ('opportunity_adjusted_peak', 'draft', True),
    ('trend_direction', 'efficiency', True),
    ('peak_recency', 'efficiency', False),
]

QB_FEATURES = COMMON_MEASURABLES + DRAFT_FEATURES + [
    # Career production
    ('college_ppr_ppg', 'production', True),
    ('college_pass_yds_pg', 'production', True),
    ('college_pass_td_pg', 'production', True),
    ('college_rush_yds_pg', 'production', True),
    ('total_college_games', 'production', True),
    ('productive_seasons', 'production', True),
    # Peak season
    ('peak_ppr_ppg', 'peak', True),
    ('peak_pass_yds_pg', 'peak', True),
    ('peak_pass_td_pg', 'peak', True),
    ('peak_cmp_pct', 'peak', True),
    ('peak_ypa', 'peak', True),
    ('peak_qb_rating', 'peak', True),
    ('peak_rush_yds_pg_qb', 'peak', True),
    # Efficiency
    ('college_cmp_pct', 'efficiency', True),
    ('college_ypa', 'efficiency', True),
    ('college_qb_rating', 'efficiency', True),
    ('college_td_int_ratio', 'efficiency', True),
    ('breakout_ratio', 'efficiency', True),
    ('last_season_ppr_ppg', 'efficiency', True),
    # v3: Peak dominance & draft interactions
    ('peak_to_career_ratio', 'peak', True),
    ('peak_to_last_ratio', 'peak', True),
    ('peak_season_share', 'peak', False),
    ('draft_pick_log', 'draft', False),
    ('draft_x_peak_ppg', 'draft', True),
    ('draft_x_college_ppg', 'draft', True),
    ('snap_share_proxy', 'draft', True),
    ('opportunity_adjusted_peak', 'draft', True),
    ('trend_direction', 'efficiency', True),
    ('peak_recency', 'efficiency', False),
    ('qb_rushing_value', 'efficiency', True),
]

TE_FEATURES = COMMON_MEASURABLES + DRAFT_FEATURES + [
    # Career production
    ('college_ppr_ppg', 'production', True),
    ('college_rec_pg', 'production', True),
    ('college_rec_yds_pg', 'production', True),
    ('college_rec_td_pg', 'production', True),
    ('total_college_games', 'production', True),
    ('productive_seasons', 'production', True),
    # Peak season
    ('peak_ppr_ppg', 'peak', True),
    ('peak_rec_pg', 'peak', True),
    ('peak_rec_yds_pg', 'peak', True),
    ('peak_rec_td_pg', 'peak', True),
    ('peak_ypr', 'peak', True),
    ('peak_rec_yds', 'peak', True),
    ('peak_td_rate', 'peak', True),
    # Efficiency
    ('college_ypr', 'efficiency', True),
    ('rec_yds_per_team_game', 'efficiency', True),
    ('college_td_rate', 'efficiency', True),
    ('breakout_ratio', 'efficiency', True),
    ('last_season_ppr_ppg', 'efficiency', True),
    # v3: Peak dominance & draft interactions
    ('peak_to_career_ratio', 'peak', True),
    ('peak_to_last_ratio', 'peak', True),
    ('peak_season_share', 'peak', False),
    ('draft_pick_log', 'draft', False),
    ('draft_x_peak_ppg', 'draft', True),
    ('draft_x_college_ppg', 'draft', True),
    ('snap_share_proxy', 'draft', True),
    ('opportunity_adjusted_peak', 'draft', True),
    ('trend_direction', 'efficiency', True),
    ('peak_recency', 'efficiency', False),
]

POS_FEATURES = {
    'WR': WR_FEATURES,
    'RB': RB_FEATURES,
    'QB': QB_FEATURES,
    'TE': TE_FEATURES,
}

# ─── Category weights ──────────────────────────────────────────────────────
# Peak season gets significant weight — one dominant year is very telling.
# Draft capital stays high — NFL teams are usually right + opportunity.
#
# Production (career avg): 15%
# Peak season: 25%
# Efficiency: 10%
# Draft capital: 35%
# Measurables: 15%

# v3: Reweighted — peak dominance features make peak category stronger,
# draft interaction features capture opportunity premium better.
CATEGORY_WEIGHTS = {
    'production': 0.10,
    'peak': 0.25,
    'efficiency': 0.10,
    'draft': 0.40,
    'measurable': 0.15,
}
def load_profiles():
    path = os.path.join(PROCESSED_DIR, "player_profiles.json")
    with open(path, 'r') as f:
        return json.load(f)


def enrich_profiles(profiles):
    """
    Add derived efficiency/dominance features.
    
    v3: Enhanced with peak dominance, draft capital interactions,
    opportunity proxies, and trajectory features.
    
    Key insights:
    - Peak season matters WAY more than career avg (Ja'Marr Chase, JSN)
    - Draft capital = opportunity (snaps), not just talent signal
    - Recency matters: was the player trending up or down?
    """
    import math

    for p in profiles:
        pos = p.get('position', '')

        # ── Yards per team game (dominance proxy) ──
        total_games = p.get('total_college_games') or 0
        total_seasons = max(p.get('total_college_seasons', 1) or 1, 1)
        avg_games_per_season = total_games / total_seasons if total_seasons > 0 else 1

        if avg_games_per_season > 0:
            peak_rec_yds = p.get('peak_rec_yds') or 0
            peak_rush_yds = p.get('peak_rush_yds') or 0
            p['rec_yds_per_team_game'] = peak_rec_yds / max(avg_games_per_season, 1)
            p['rush_yds_per_team_game'] = peak_rush_yds / max(avg_games_per_season, 1)
        else:
            p['rec_yds_per_team_game'] = None
            p['rush_yds_per_team_game'] = None

        # ── TD rate (career) ──
        rec_pg = p.get('college_rec_pg') or 0
        rush_att_pg = p.get('college_rush_att_pg') or 0
        rec_td_pg = p.get('college_rec_td_pg') or 0
        rush_td_pg = p.get('college_rush_td_pg') or 0

        total_touches_pg = rec_pg + rush_att_pg
        total_td_pg = rec_td_pg + rush_td_pg

        if total_touches_pg > 0:
            p['college_td_rate'] = total_td_pg / total_touches_pg
        else:
            p['college_td_rate'] = None

        # ── TD:INT ratio for QBs ──
        if pos == 'QB':
            if p.get('college_qb_rating') and p['college_qb_rating'] > 0:
                p['college_td_int_ratio'] = p['college_qb_rating'] / 50.0
            else:
                p['college_td_int_ratio'] = None

        # ── Breakout age (simplified) ──
        seasons = p.get('total_college_seasons', 0) or 0
        if seasons > 0:
            college_ppg = p.get('college_ppr_ppg', 0) or 0
            peak_ppg = p.get('peak_ppr_ppg', 0) or 0
            if peak_ppg > 0 and college_ppg > 0:
                dominance = peak_ppg / college_ppg
                if dominance > 1.3:
                    p['breakout_age'] = min(seasons, 4)
                else:
                    p['breakout_age'] = max(1, seasons - 1)
            else:
                p['breakout_age'] = seasons
        else:
            p['breakout_age'] = None

        # ── Breakout ratio (if not already set by build_features) ──
        if p.get('breakout_ratio') is None:
            career_ppg = p.get('college_ppr_ppg', 0) or 0
            peak_ppg = p.get('peak_ppr_ppg', 0) or 0
            if career_ppg > 0:
                p['breakout_ratio'] = peak_ppg / career_ppg
            else:
                p['breakout_ratio'] = None

        # ══════════════════════════════════════════════════════════════
        # v3 NEW FEATURES: Peak Dominance, Draft Interactions, Opportunity
        # ══════════════════════════════════════════════════════════════

        peak_ppg = p.get('peak_ppr_ppg', 0) or 0
        career_ppg = p.get('college_ppr_ppg', 0) or 0
        last_ppg = p.get('last_season_ppr_ppg', 0) or 0
        draft_pick = p.get('draft_pick')
        draft_round = p.get('draft_round')

        # ── Peak Dominance Features ──────────────────────────────────
        # How much better was peak year vs career avg?
        # JSN: 25.8 / 15.1 = 1.71x — peak was WAY above average
        # Bowers: 19.5 / 18.4 = 1.06x — consistently elite
        if career_ppg > 0:
            p['peak_to_career_ratio'] = peak_ppg / career_ppg
        else:
            p['peak_to_career_ratio'] = None

        # Peak vs last season — catches declining players
        # If peak == last, player was peaking when drafted (good sign)
        # If peak >> last, player may have peaked early or was injured
        if last_ppg > 0:
            p['peak_to_last_ratio'] = peak_ppg / last_ppg
        else:
            p['peak_to_last_ratio'] = None

        # Peak season share — what % of total career PPR came in best year?
        # High share + few seasons = spike player (risky)
        # High share + many seasons = one breakout year surrounded by meh
        peak_gp = p.get('peak_gp') or 1
        peak_total = peak_ppg * peak_gp
        career_total_college = sum([
            (p.get('college_ppr_ppg', 0) or 0) * (p.get('total_college_games', 0) or 1)
        ])
        # Better estimate using per-game * games
        if total_games > 0 and career_ppg > 0:
            estimated_career_total = career_ppg * total_games
            if estimated_career_total > 0:
                p['peak_season_share'] = peak_total / estimated_career_total
            else:
                p['peak_season_share'] = None
        else:
            p['peak_season_share'] = None

        # ── Draft Capital Features ───────────────────────────────────
        # Log transform of draft pick — compresses 1-224 into ~0-5.4
        # Pick 1 = 0, Pick 10 = 2.3, Pick 32 = 3.5, Pick 100 = 4.6, Pick 224 = 5.4
        if draft_pick is not None:
            try:
                pick_val = float(draft_pick)
                if pick_val > 0:
                    p['draft_pick_log'] = math.log(pick_val)
                else:
                    p['draft_pick_log'] = 0.0
            except (ValueError, TypeError):
                p['draft_pick_log'] = None
        else:
            p['draft_pick_log'] = None

        # Draft × Peak interaction — high pick + elite peak = best signal
        # Uses inverse log so higher pick (lower number) = higher value
        if p.get('draft_pick_log') is not None and peak_ppg > 0:
            inv_log = 1.0 / (1.0 + p['draft_pick_log'])
            p['draft_x_peak_ppg'] = inv_log * peak_ppg
            p['draft_x_college_ppg'] = inv_log * career_ppg if career_ppg > 0 else None
        else:
            p['draft_x_peak_ppg'] = None
            p['draft_x_college_ppg'] = None

        # ── Opportunity Proxy ────────────────────────────────────────
        # Expected snap share / volume multiplier based on draft position
        # R1 picks get ~80% snaps year 1, late rounders get ~30-40%
        if draft_round is not None:
            try:
                rd = float(draft_round)
                if rd <= 1:
                    p['snap_share_proxy'] = 0.80
                elif rd <= 2:
                    p['snap_share_proxy'] = 0.65
                elif rd <= 3:
                    p['snap_share_proxy'] = 0.50
                elif rd <= 4:
                    p['snap_share_proxy'] = 0.40
                elif rd <= 5:
                    p['snap_share_proxy'] = 0.30
                else:
                    p['snap_share_proxy'] = 0.20
            except (ValueError, TypeError):
                p['snap_share_proxy'] = None
        else:
            p['snap_share_proxy'] = None

        # Volume-adjusted peak: what would peak production look like
        # scaled to expected NFL opportunity?
        if p.get('snap_share_proxy') and peak_ppg > 0:
            p['opportunity_adjusted_peak'] = peak_ppg * p['snap_share_proxy']
        else:
            p['opportunity_adjusted_peak'] = None

        # ── Recency / Trajectory ─────────────────────────────────────
        # Was the player trending UP or DOWN entering the draft?
        if career_ppg > 0 and last_ppg > 0:
            p['trend_direction'] = last_ppg / career_ppg
        else:
            p['trend_direction'] = None

        # Peak recency — how long ago was the peak?
        # 0 = peak was most recent season (ideal)
        # 1+ = peak was further back (JSN had peak in 2021, drafted 2023 → 2)
        peak_year = p.get('peak_season_year')
        last_year = p.get('last_season_year')
        if peak_year and last_year:
            try:
                p['peak_recency'] = int(float(last_year)) - int(float(peak_year))
            except (ValueError, TypeError):
                p['peak_recency'] = None
        else:
            p['peak_recency'] = None

        # ── QB-specific: rushing value interaction ───────────────────
        if pos == 'QB':
            qb_rush = p.get('college_rush_yds_pg') or p.get('college_rush_yds_pg_qb') or 0
            if p.get('snap_share_proxy') and qb_rush > 0:
                p['qb_rushing_value'] = qb_rush * p['snap_share_proxy']
            else:
                p['qb_rushing_value'] = None

    return profiles

def split_profiles(profiles):
    nfl_players = []
    prospects = []
    for p in profiles:
        if p.get('is_prospect'):
            prospects.append(p)
        elif p.get('rookie_ppr_ppg') is not None:
            nfl_players.append(p)
    return nfl_players, prospects


def compute_feature_stats(players, features):
    stats = {}
    for feat_key, _, _ in features:
        values = [p.get(feat_key) for p in players if p.get(feat_key) is not None]
        if len(values) < 2:
            stats[feat_key] = (0, 1)
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1
        stats[feat_key] = (mean, std)
    return stats


def normalize_value(value, mean, std):
    if std == 0:
        return 0
    return (value - mean) / std


def compute_similarity(prospect, nfl_player, features, feat_stats):
    """Weighted cosine + closeness similarity."""
    cat_features = defaultdict(list)
    for feat_key, cat, higher_is_better in features:
        cat_features[cat].append((feat_key, higher_is_better))

    cat_scores = {}
    feature_comparison = {}

    for cat, feat_list in cat_features.items():
        dot = 0
        mag_a = 0
        mag_b = 0
        valid_count = 0
        euclidean_sum = 0

        for feat_key, higher_is_better in feat_list:
            val_a = prospect.get(feat_key)
            val_b = nfl_player.get(feat_key)

            if val_a is None or val_b is None:
                continue

            mean, std = feat_stats.get(feat_key, (0, 1))
            norm_a = normalize_value(val_a, mean, std)
            norm_b = normalize_value(val_b, mean, std)

            feature_comparison[feat_key] = {
                'prospect': val_a,
                'comp': val_b,
                'diff_pct': abs(val_a - val_b) / abs(val_b) * 100 if val_b != 0 else 0,
            }

            dot += norm_a * norm_b
            mag_a += norm_a ** 2
            mag_b += norm_b ** 2
            euclidean_sum += (norm_a - norm_b) ** 2
            valid_count += 1

        if valid_count == 0 or mag_a == 0 or mag_b == 0:
            cat_scores[cat] = 0
        else:
            cosine = dot / (math.sqrt(mag_a) * math.sqrt(mag_b))
            euclidean = math.sqrt(euclidean_sum)
            closeness = 1 / (1 + euclidean / max(valid_count, 1))
            # Blend: 60% cosine, 40% closeness
            cat_scores[cat] = 0.6 * max(cosine, 0) + 0.4 * closeness

    # Weighted combination
    total_weight = 0
    weighted_score = 0
    for cat, weight in CATEGORY_WEIGHTS.items():
        if cat in cat_scores:
            weighted_score += weight * cat_scores[cat]
            total_weight += weight

    final_score = weighted_score / total_weight if total_weight > 0 else 0

    return final_score, feature_comparison, cat_scores


def find_comparisons(prospect, nfl_players, n=10):
    """Find the top-N most similar NFL players to a prospect."""
    pos = prospect.get('position', '')
    features = POS_FEATURES.get(pos, [])
    if not features:
        return []

    pos_players = [p for p in nfl_players if p.get('position') == pos]
    if not pos_players:
        return []

    all_pool = pos_players + [prospect]
    feat_stats = compute_feature_stats(all_pool, features)

    comparisons = []
    for nfl_p in pos_players:
        score, feat_comp, cat_scores = compute_similarity(
            prospect, nfl_p, features, feat_stats
        )
        comparisons.append({
            'espn_id': nfl_p.get('espn_id'),
            'name': nfl_p.get('name'),
            'position': nfl_p.get('position'),
            'archetype': nfl_p.get('archetype'),
            'college': nfl_p.get('college', ''),
            'draft_year': nfl_p.get('draft_year'),
            'draft_round': nfl_p.get('draft_round'),
            'draft_pick': nfl_p.get('draft_pick'),
            'similarity_score': round(score, 4),
            'category_scores': {k: round(v, 4) for k, v in cat_scores.items()},
            'feature_comparison': feat_comp,
            'rookie_ppr_ppg': nfl_p.get('rookie_ppr_ppg'),
            'rookie_ppr_total': nfl_p.get('rookie_ppr_total'),
            'best_season_ppr_ppg': nfl_p.get('best_season_ppr_ppg'),
            'peak_3yr_ppr_ppg': nfl_p.get('peak_3yr_ppr_ppg'),
            'dynasty_ppg': nfl_p.get('dynasty_ppg'),
            'nfl_seasons_played': nfl_p.get('nfl_seasons_played'),
            'career_ppr_total': nfl_p.get('career_ppr_total'),
        })

    comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
    return comparisons[:n]


def build_all_comparisons(n=10):
    """Build comparison results for all 2026 prospects."""
    profiles = load_profiles()
    profiles = enrich_profiles(profiles)
    nfl_players, prospects = split_profiles(profiles)

    print(f"NFL players with outcomes: {len(nfl_players)}")
    print(f"2026 prospects: {len(prospects)}")

    results = {}
    for prospect in prospects:
        pid = prospect.get('espn_id')
        comps = find_comparisons(prospect, nfl_players, n=n)

        results[pid] = {
            'prospect': {
                'espn_id': pid,
                'name': prospect.get('name'),
                'position': prospect.get('position'),
                'school': prospect.get('school', ''),
                'archetype': prospect.get('archetype'),
                'archetype_scores': prospect.get('archetype_scores'),
                'combine_height': prospect.get('combine_height'),
                'combine_weight': prospect.get('combine_weight'),
                'combine_40': prospect.get('combine_40'),
                'combine_vert': prospect.get('combine_vert'),
                'combine_broad': prospect.get('combine_broad'),
                'college_ppr_ppg': prospect.get('college_ppr_ppg'),
                'college_rec_pg': prospect.get('college_rec_pg'),
                'college_rec_yds_pg': prospect.get('college_rec_yds_pg'),
                'college_rush_yds_pg': prospect.get('college_rush_yds_pg'),
                'college_ypc': prospect.get('college_ypc'),
                'college_ypr': prospect.get('college_ypr'),
                'college_pass_yds_pg': prospect.get('college_pass_yds_pg'),
                'college_pass_td_pg': prospect.get('college_pass_td_pg'),
                'college_cmp_pct': prospect.get('college_cmp_pct'),
                'college_ypa': prospect.get('college_ypa'),
                'peak_ppr_ppg': prospect.get('peak_ppr_ppg'),
                'peak_rec_pg': prospect.get('peak_rec_pg'),
                'peak_rec_yds_pg': prospect.get('peak_rec_yds_pg'),
                'peak_rec_td_pg': prospect.get('peak_rec_td_pg'),
                'peak_ypr': prospect.get('peak_ypr'),
                'peak_rush_yds_pg': prospect.get('peak_rush_yds_pg'),
                'peak_pass_yds_pg': prospect.get('peak_pass_yds_pg'),
                'peak_pass_td_pg': prospect.get('peak_pass_td_pg'),
                'total_college_games': prospect.get('total_college_games'),
                'big_board_rank': prospect.get('big_board_rank'),
                'college_td_rate': prospect.get('college_td_rate'),
                'rec_yds_per_team_game': prospect.get('rec_yds_per_team_game'),
                'breakout_age': prospect.get('breakout_age'),
                'breakout_ratio': prospect.get('breakout_ratio'),
                'last_season_ppr_ppg': prospect.get('last_season_ppr_ppg'),
                # Draft capital
                'draft_round': prospect.get('draft_round'),
                'draft_pick': prospect.get('draft_pick'),
                'draft_capital_is_mock': prospect.get('draft_capital_is_mock'),
            },
            'comparisons': comps,
        }

    return results
    """Build comparison results for all 2026 prospects."""
    profiles = load_profiles()

    # Enrich with derived features
    profiles = enrich_profiles(profiles)

    nfl_players, prospects = split_profiles(profiles)

    print(f"NFL players with outcomes: {len(nfl_players)}")
    print(f"2026 prospects: {len(prospects)}")

    results = {}
    for prospect in prospects:
        pid = prospect.get('espn_id')
        comps = find_comparisons(prospect, nfl_players, n=n)

        results[pid] = {
            'prospect': {
                'espn_id': pid,
                'name': prospect.get('name'),
                'position': prospect.get('position'),
                'school': prospect.get('school', ''),
                'archetype': prospect.get('archetype'),
                'archetype_scores': prospect.get('archetype_scores'),
                'combine_height': prospect.get('combine_height'),
                'combine_weight': prospect.get('combine_weight'),
                'combine_40': prospect.get('combine_40'),
                'combine_vert': prospect.get('combine_vert'),
                'combine_broad': prospect.get('combine_broad'),
                'college_ppr_ppg': prospect.get('college_ppr_ppg'),
                'college_rec_pg': prospect.get('college_rec_pg'),
                'college_rec_yds_pg': prospect.get('college_rec_yds_pg'),
                'college_rush_yds_pg': prospect.get('college_rush_yds_pg'),
                'college_ypc': prospect.get('college_ypc'),
                'college_ypr': prospect.get('college_ypr'),
                'college_pass_yds_pg': prospect.get('college_pass_yds_pg'),
                'college_pass_td_pg': prospect.get('college_pass_td_pg'),
                'college_cmp_pct': prospect.get('college_cmp_pct'),
                'college_ypa': prospect.get('college_ypa'),
                'peak_ppr_ppg': prospect.get('peak_ppr_ppg'),
                'total_college_games': prospect.get('total_college_games'),
                'big_board_rank': prospect.get('big_board_rank'),
                # New efficiency features
                'college_td_rate': prospect.get('college_td_rate'),
                'rec_yds_per_team_game': prospect.get('rec_yds_per_team_game'),
                'breakout_age': prospect.get('breakout_age'),
            },
            'comparisons': comps,
        }

    return results


def main():
    results = build_all_comparisons(n=10)

    output_path = os.path.join(PROCESSED_DIR, "prospect_comparisons.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved comparisons to {output_path}")

    for pid, data in list(results.items())[:5]:
        prospect = data['prospect']
        print(f"\n{'─' * 60}")
        print(f"🏈 {prospect['name']} ({prospect['position']}) - {prospect['school']}")
        print(f"   Archetype: {prospect['archetype']}")
        print(f"   TD Rate: {prospect.get('college_td_rate', 'N/A')}")
        print(f"   Breakout Age: {prospect.get('breakout_age', 'N/A')}")
        print(f"   Top comps:")
        for i, comp in enumerate(data['comparisons'][:5], 1):
            cats = comp.get('category_scores', {})
            cat_str = " | ".join(f"{k}: {v:.0%}" for k, v in cats.items())
            print(f"   {i}. {comp['name']} ({comp['similarity_score']:.3f}) "
                  f"- Rookie: {comp.get('rookie_ppr_ppg', 0):.1f}, "
                  f"Dynasty: {comp.get('dynasty_ppg', 0):.1f} "
                  f"[{cat_str}]")


if __name__ == "__main__":
    main()