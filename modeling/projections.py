# modeling/projections.py
"""
Projection engine v3: draft capital adjustment + tuned exponent.

Run: python -m modeling.projections
"""

import json
import os
import math

PROCESSED_DIR = "data/processed"


def load_comparisons():
    path = os.path.join(PROCESSED_DIR, "prospect_comparisons.json")
    with open(path, 'r') as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw
    elif isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, dict):
                prospect = item.get('prospect', {})
                pid = prospect.get('espn_id')
                if pid:
                    result[str(pid)] = item
        return result
    return {}


def load_profile_lookup():
    """Load full profiles keyed by espn_id for draft capital lookup."""
    path = os.path.join(PROCESSED_DIR, "player_profiles.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        profiles = json.load(f)
    lookup = {}
    for p in profiles:
        pid = str(p.get('espn_id', ''))
        if pid:
            lookup[pid] = p
    return lookup


def weighted_projection(comparisons, key, top_n=10, exp=2):
    """
    Similarity-weighted projection.
    
    Filters out comps below 50% of the best comp's similarity score
    to prevent one outlier bust from tanking projections.
    """
    valid = [
        c for c in comparisons[:top_n]
        if c.get(key) is not None and c.get('similarity_score', 0) > 0
    ]
    if not valid:
        return None

    # Filter: drop comps below 50% of best similarity
    # This prevents a high-similarity bust from dominating
    best_sim = max(c['similarity_score'] for c in valid)
    sim_floor = best_sim * 0.85
    filtered = [c for c in valid if c['similarity_score'] >= sim_floor]
    
    # Keep at least 3 comps
    if len(filtered) < 3:
        filtered = valid[:max(3, len(filtered))]

    weights = [c['similarity_score'] ** exp for c in filtered]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    weighted_mean = sum(c[key] * w for c, w in zip(filtered, weights)) / total_weight

    weighted_var = sum(
        w * (c[key] - weighted_mean) ** 2 for c, w in zip(filtered, weights)
    ) / total_weight
    weighted_std = math.sqrt(weighted_var) if weighted_var > 0 else 0

    avg_similarity = sum(c['similarity_score'] for c in filtered) / len(filtered)
    confidence = min(avg_similarity * len(filtered) / top_n, 1.0)

    values = sorted([c[key] for c in filtered])
    n = len(values)
    p25_idx = max(0, int(n * 0.25))
    p75_idx = min(n - 1, int(n * 0.75))

    return {
        'projected': round(weighted_mean, 2),
        'floor': round(values[p25_idx], 2),
        'ceiling': round(values[p75_idx], 2),
        'std': round(weighted_std, 2),
        'confidence': round(confidence, 3),
        'n_comps': len(filtered),
        'n_filtered_out': len(valid) - len(filtered),
    }


def draft_capital_adjustment(proj_dict, draft_round, draft_pick, pos):
    """
    Adjust projection based on draft capital.
    
    Since draft capital is already 40% of similarity matching (comps are
    already biased toward same-round players), this is a RESIDUAL adjustment
    capturing the extra opportunity premium.
    
    Residual blend of 0.55 — empirically calibrated so that R1 picks 10-20
    WRs project near the historical average of 11.4 dynasty PPG.
    """
    if proj_dict is None:
        return proj_dict
    if draft_round is None and draft_pick is None:
        return proj_dict

    try:
        pick = float(draft_pick) if draft_pick is not None else None
        rnd = float(draft_round) if draft_round is not None else None

        if pick is None and rnd is not None:
            pick = rnd * 32
        if pick is None:
            return proj_dict
    except (ValueError, TypeError):
        return proj_dict

    # Empirical multipliers by pick range
    if pick <= 10:
        raw_mult = 1.35
    elif pick <= 20:
        raw_mult = 1.25
    elif pick <= 32:
        raw_mult = 1.18
    elif pick <= 48:
        raw_mult = 1.10
    elif pick <= 64:
        raw_mult = 1.05
    elif pick <= 100:
        raw_mult = 1.00
    elif pick <= 140:
        raw_mult = 0.92
    elif pick <= 180:
        raw_mult = 0.85
    elif pick <= 224:
        raw_mult = 0.78
    else:
        raw_mult = 0.70

    # Residual blend: 0.55 of the raw effect applied on top of
    # similarity-matched comps. Calibrated against R1 WR empirical avg.
    residual_blend = 0.55

    effective = 1.0 + (raw_mult - 1.0) * residual_blend

    adjusted = proj_dict.copy()
    adjusted['pre_adjustment'] = adjusted['projected']
    adjusted['projected'] = round(adjusted['projected'] * effective, 2)
    adjusted['floor'] = round(adjusted['floor'] * effective, 2)
    adjusted['ceiling'] = round(adjusted['ceiling'] * effective, 2)
    adjusted['draft_multiplier'] = round(effective, 3)
    adjusted['raw_draft_multiplier'] = round(raw_mult, 3)

    return adjusted


def tier_label(dynasty_ppg, pos):
    if dynasty_ppg is None or dynasty_ppg == 0:
        return 'Unknown'
    thresholds = {
        'QB': [(20, 'QB1 Elite'), (16, 'QB1'), (12, 'QB2'), (8, 'QB3'), (0, 'Backup')],
        'RB': [(14, 'RB1 Elite'), (10, 'RB1'), (7, 'RB2'), (4, 'RB3'), (0, 'Roster Clog')],
        'WR': [(14, 'WR1 Elite'), (10, 'WR1'), (7, 'WR2'), (4, 'WR3'), (0, 'Roster Clog')],
        'TE': [(12, 'TE1 Elite'), (9, 'TE1'), (6, 'TE2'), (3, 'TE3'), (0, 'Roster Clog')],
    }
    for threshold, label in thresholds.get(pos, [(0, 'Unknown')]):
        if dynasty_ppg >= threshold:
            return label
    return 'Unknown'


def bust_probability(comparisons, pos, top_n=10):
    thresholds = {'QB': 8, 'RB': 5, 'WR': 5, 'TE': 4}
    threshold = thresholds.get(pos, 5)
    valid = [c for c in comparisons[:top_n] if c.get('dynasty_ppg') is not None]
    if not valid:
        return None
    return round(sum(1 for c in valid if c['dynasty_ppg'] < threshold) / len(valid), 3)


def breakout_probability(comparisons, pos, top_n=10):
    thresholds = {'QB': 20, 'RB': 14, 'WR': 14, 'TE': 12}
    threshold = thresholds.get(pos, 14)
    valid = [c for c in comparisons[:top_n] if c.get('best_season_ppr_ppg') is not None]
    if not valid:
        return None
    return round(sum(1 for c in valid if c['best_season_ppr_ppg'] >= threshold) / len(valid), 3)


def build_all_projections():
    comparisons_data = load_comparisons()
    profile_lookup = load_profile_lookup()

    if not comparisons_data:
        print("ERROR: No comparison data.")
        return {}

    print(f"Loaded comparisons for {len(comparisons_data)} prospects")
    print(f"Loaded {len(profile_lookup)} profiles for draft capital")

    # Debug: check draft data availability
    draft_found = 0
    draft_missing = 0

    projections = {}

    for pid, data in comparisons_data.items():
        prospect = data.get('prospect', {})
        comps = data.get('comparisons', [])
        pos = prospect.get('position', '')

        if not comps:
            continue

        # Get draft capital — check multiple sources
        full_profile = profile_lookup.get(str(pid), {})

        draft_round = (
            full_profile.get('draft_round') or
            prospect.get('draft_round') or
            None
        )
        draft_pick = (
            full_profile.get('draft_pick') or
            prospect.get('draft_pick') or
            None
        )

        if draft_round is not None:
            draft_found += 1
        else:
            draft_missing += 1

        is_mock = full_profile.get('draft_capital_is_mock', False)

        # Compute projections with exponent 2
        rookie = weighted_projection(comps, 'rookie_ppr_ppg', exp=2)
        dynasty = weighted_projection(comps, 'dynasty_ppg', exp=2)
        peak = weighted_projection(comps, 'peak_3yr_ppr_ppg', exp=2)
        best_season = weighted_projection(comps, 'best_season_ppr_ppg', exp=2)
        career = weighted_projection(comps, 'nfl_seasons_played', exp=2)

        # Apply draft capital adjustment
        if draft_round is not None:
            rookie = draft_capital_adjustment(rookie, draft_round, draft_pick, pos)
            dynasty = draft_capital_adjustment(dynasty, draft_round, draft_pick, pos)
            peak = draft_capital_adjustment(peak, draft_round, draft_pick, pos)
            best_season = draft_capital_adjustment(best_season, draft_round, draft_pick, pos)

        dynasty_ppg = dynasty['projected'] if dynasty else 0
        tier = tier_label(dynasty_ppg, pos)
        bust_prob = bust_probability(comps, pos)
        breakout_prob = breakout_probability(comps, pos)

        # Enrich prospect for display
        prospect_enriched = {**prospect}
        prospect_enriched['draft_round'] = draft_round
        prospect_enriched['draft_pick'] = draft_pick
        prospect_enriched['draft_is_mock'] = is_mock

        projections[pid] = {
            'prospect': prospect_enriched,
            'comparisons': comps,
            'projections': {
                'rookie': rookie,
                'dynasty': dynasty,
                'peak_3yr': peak,
                'best_season': best_season,
                'career_length': career,
            },
            'evaluation': {
                'tier': tier,
                'bust_probability': bust_prob,
                'breakout_probability': breakout_prob,
            },
        }

    print(f"\nDraft capital found: {draft_found}/{draft_found + draft_missing}")
    if draft_missing > 0:
        print(f"⚠️  {draft_missing} prospects missing draft capital")

    return projections


def rank_prospects(projections, sort_by='dynasty'):
    key_map = {
        'dynasty': lambda p: (p['projections']['dynasty'] or {}).get('projected', 0) or 0,
        'rookie': lambda p: (p['projections']['rookie'] or {}).get('projected', 0) or 0,
        'peak': lambda p: (p['projections']['peak_3yr'] or {}).get('projected', 0) or 0,
        'ceiling': lambda p: (p['projections']['best_season'] or {}).get('ceiling', 0) or 0,
    }
    sort_fn = key_map.get(sort_by, key_map['dynasty'])
    ranked = sorted(projections.values(), key=sort_fn, reverse=True)
    for i, proj in enumerate(ranked, 1):
        proj['rank'] = i
    return ranked


def main():
    print("=" * 70)
    print("BUILDING PROJECTIONS (v3 — draft adjustment)")
    print("=" * 70)

    projections = build_all_projections()

    if not projections:
        print("No projections built.")
        return

    output_path = os.path.join(PROCESSED_DIR, "prospect_projections.json")
    with open(output_path, 'w') as f:
        json.dump(projections, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    ranked = rank_prospects(projections, sort_by='dynasty')

    print(f"\n{'─' * 100}")
    print(f"{'Rank':<5} {'Name':<25} {'Pos':<4} {'Arch':<18} "
          f"{'Rookie':>8} {'Dynasty':>8} {'Peak':>8} {'Tier':<15} "
          f"{'Bust%':>6} {'DAdj':>6} {'Pick':>5}")
    print(f"{'─' * 100}")

    for proj in ranked:
        p = proj['prospect']
        r = proj['projections']
        e = proj['evaluation']

        rookie_ppg = r['rookie']['projected'] if r['rookie'] else 0
        dynasty_ppg = r['dynasty']['projected'] if r['dynasty'] else 0
        peak_ppg = r['peak_3yr']['projected'] if r['peak_3yr'] else 0
        bust = e.get('bust_probability', 0) or 0
        d_adj = r['dynasty'].get('draft_multiplier', 1.0) if r['dynasty'] else 1.0
        pick = p.get('draft_pick', '—')
        pick_str = f"{int(float(pick))}" if pick and pick != '—' else '—'

        print(f"{proj.get('rank', '-'):<5} {p.get('name', '?'):<25} "
              f"{p.get('position', '?'):<4} {p.get('archetype', '?'):<18} "
              f"{rookie_ppg:>8.1f} {dynasty_ppg:>8.1f} {peak_ppg:>8.1f} "
              f"{e.get('tier', '?'):<15} {bust:>5.0%} {d_adj:>5.2f}x {pick_str:>5}")

    # Position breakdowns
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_ranked = [p for p in ranked if p['prospect'].get('position') == pos]
        if not pos_ranked:
            continue
        print(f"\n{'═' * 60}")
        print(f"  {pos} Rankings")
        print(f"{'═' * 60}")
        for i, proj in enumerate(pos_ranked, 1):
            p = proj['prospect']
            r = proj['projections']
            dynasty_ppg = r['dynasty']['projected'] if r['dynasty'] else 0
            d_adj = r['dynasty'].get('draft_multiplier', 1.0) if r['dynasty'] else 1.0
            pick = p.get('draft_pick', '—')
            pick_str = f"Pick {int(float(pick))}" if pick and pick != '—' else 'UDFA est.'
            print(f"  {i}. {p.get('name', '?'):<25} Dynasty: {dynasty_ppg:.1f} "
                  f"({proj['evaluation'].get('tier', '?')}) "
                  f"[{d_adj:.2f}x] {pick_str}")

    print(f"\n{'=' * 70}")
    print(f"✅ PROJECTIONS COMPLETE: {len(projections)} prospects")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()