# modeling/backtest.py
"""
Backtesting framework: evaluates the projection model against
the 2022, 2023, and 2024 draft classes.

For each class, we:
1. Treat drafted players as "prospects" (use only pre-NFL data)
2. Compare them against the historical pool BEFORE their draft year
3. Generate projections
4. Compare projections to actual NFL outcomes
5. Compute accuracy metrics

v2: Now properly enriches profiles with efficiency features and
    uses the updated category weights.

Run: python -m modeling.backtest
"""

import json
import os
import math
from collections import defaultdict
from modeling.similarity import (
    POS_FEATURES, CATEGORY_WEIGHTS, compute_feature_stats,
    compute_similarity, load_profiles, enrich_profiles
)

PROCESSED_DIR = "data/processed"

BACKTEST_YEARS = [2022, 2023, 2024]


def split_by_draft_year(profiles, test_year):
    """
    Split profiles into:
    - historical: NFL players drafted BEFORE test_year with outcomes
    - test_prospects: players drafted IN test_year (treat as prospects)
    """
    historical = []
    test_prospects = []

    for p in profiles:
        if p.get('is_prospect'):
            continue

        draft_year = p.get('draft_year')
        has_outcomes = p.get('rookie_ppr_ppg') is not None

        if draft_year is None:
            if has_outcomes:
                historical.append(p)
            continue

        try:
            dy = int(float(draft_year))
        except (ValueError, TypeError):
            if has_outcomes:
                historical.append(p)
            continue

        if dy < test_year and has_outcomes:
            historical.append(p)
        elif dy == test_year and has_outcomes:
            test_prospects.append(p)

    return historical, test_prospects


def find_backtest_comparisons(prospect, historical_pool, n=10):
    """Find comps for a backtest prospect from the historical pool."""
    pos = prospect.get('position', '')
    features = POS_FEATURES.get(pos, [])
    if not features:
        return []

    pos_players = [p for p in historical_pool if p.get('position') == pos]
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
            'name': nfl_p.get('name'),
            'position': nfl_p.get('position'),
            'archetype': nfl_p.get('archetype'),
            'draft_round': nfl_p.get('draft_round'),
            'draft_pick': nfl_p.get('draft_pick'),
            'similarity_score': score,
            'category_scores': cat_scores,
            'feature_comparison': feat_comp,
            'rookie_ppr_ppg': nfl_p.get('rookie_ppr_ppg'),
            'dynasty_ppg': nfl_p.get('dynasty_ppg'),
            'best_season_ppr_ppg': nfl_p.get('best_season_ppr_ppg'),
            'peak_3yr_ppr_ppg': nfl_p.get('peak_3yr_ppr_ppg'),
            'nfl_seasons_played': nfl_p.get('nfl_seasons_played'),
        })

    comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
    return comparisons[:n]


def weighted_projection(comparisons, key, top_n=10):
    """Same projection logic as projections.py."""
    valid = [
        c for c in comparisons[:top_n]
        if c.get(key) is not None and c.get('similarity_score', 0) > 0
    ]
    if not valid:
        return None

    weights = [c['similarity_score'] ** 3 for c in valid]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    weighted_mean = sum(c[key] * w for c, w in zip(valid, weights)) / total_weight
    return round(weighted_mean, 2)


def compute_accuracy(predictions):
    """
    Compute accuracy metrics across all predictions.
    predictions: list of (projected, actual, name, pos)
    """
    valid = [(p, a, n, ps) for p, a, n, ps in predictions
             if p is not None and a is not None]

    if not valid:
        return {}

    errors = [p - a for p, a, _, _ in valid]
    abs_errors = [abs(e) for e in errors]
    projs = [p for p, _, _, _ in valid]
    actuals = [a for _, a, _, _ in valid]

    n = len(errors)
    mae = sum(abs_errors) / n
    bias = sum(errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

    # Correlation
    if n >= 3:
        mean_p = sum(projs) / n
        mean_a = sum(actuals) / n
        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(projs, actuals)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in projs) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)
        correlation = cov / (std_p * std_a) if std_p > 0 and std_a > 0 else 0
    else:
        correlation = 0

    within_2 = sum(1 for e in abs_errors if e <= 2) / n
    within_5 = sum(1 for e in abs_errors if e <= 5) / n

    sorted_abs = sorted(abs_errors)

    return {
        'n': n,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'bias': round(bias, 2),
        'correlation': round(correlation, 3),
        'within_2_ppg': round(within_2, 3),
        'within_5_ppg': round(within_5, 3),
        'median_abs_error': round(sorted_abs[n // 2], 2),
    }


def analyze_misses(results, threshold=7.0):
    """
    Analyze the worst misses to identify systematic issues.
    Returns categorized miss patterns.
    """
    big_misses = [
        r for r in results
        if r.get('rookie_error') is not None and abs(r['rookie_error']) >= threshold
    ]

    over_projections = [r for r in big_misses if r['rookie_error'] > 0]
    under_projections = [r for r in big_misses if r['rookie_error'] < 0]

    patterns = {
        'total_big_misses': len(big_misses),
        'over_projected': len(over_projections),
        'under_projected': len(under_projections),
        'over_proj_details': [],
        'under_proj_details': [],
    }

    for r in over_projections:
        patterns['over_proj_details'].append({
            'name': r['name'],
            'position': r['position'],
            'error': r['rookie_error'],
            'comps': r.get('top_comps', []),
            'reason_guess': _guess_over_reason(r),
        })

    for r in under_projections:
        patterns['under_proj_details'].append({
            'name': r['name'],
            'position': r['position'],
            'error': r['rookie_error'],
            'comps': r.get('top_comps', []),
            'reason_guess': _guess_under_reason(r),
        })

    return patterns


def _guess_over_reason(r):
    """Guess why we over-projected a player."""
    name = r['name'].lower()
    pos = r['position']
    error = r['rookie_error']

    # Injured players
    injured_keywords = ['metchie', 'jameson williams', 'hooker', 'brooks', 'jonathon']
    for kw in injured_keywords:
        if kw in name:
            return 'INJURY — missed games rookie year'

    if pos == 'WR' and error > 5:
        return 'WR — likely buried on depth chart or slow adjustment'
    if pos == 'RB' and error > 5:
        return 'RB — likely lost job or committee'
    if pos == 'QB':
        return 'QB — likely backup, didnt start'

    return 'UNKNOWN — possibly situation/opportunity'


def _guess_under_reason(r):
    """Guess why we under-projected a player."""
    pos = r['position']
    error = abs(r['rookie_error'])

    if pos == 'WR' and error > 8:
        return 'WR BREAKOUT — likely had elite efficiency not captured by college volume'
    if pos == 'RB' and error > 8:
        return 'RB BREAKOUT — likely got unexpected opportunity/volume'
    if pos == 'TE' and error > 8:
        return 'TE BREAKOUT — rare immediate impact TE, model underweights ceiling'
    if pos == 'QB' and error > 8:
        return 'QB BREAKOUT — unexpected starter, model comped to backups'

    return 'BREAKOUT — exceeded comp pool ceiling'


def run_backtest(test_year, profiles, top_n=10):
    """Run backtest for a single draft year."""
    historical, test_prospects = split_by_draft_year(profiles, test_year)

    print(f"\n  Historical pool: {len(historical)} players")
    print(f"  Test prospects (class of {test_year}): {len(test_prospects)}")

    # Verify enrichment worked
    sample = historical[:5] if historical else []
    enriched_count = sum(1 for p in historical if p.get('college_td_rate') is not None)
    print(f"  Historical with efficiency features: {enriched_count}/{len(historical)}")

    rookie_predictions = []
    dynasty_predictions = []
    results = []

    for prospect in test_prospects:
        name = prospect.get('name', '?')
        pos = prospect.get('position', '?')

        comps = find_backtest_comparisons(prospect, historical, n=top_n)

        if not comps:
            continue

        proj_rookie = weighted_projection(comps, 'rookie_ppr_ppg', top_n)
        proj_dynasty = weighted_projection(comps, 'dynasty_ppg', top_n)

        actual_rookie = prospect.get('rookie_ppr_ppg')
        actual_dynasty = prospect.get('dynasty_ppg')

        rookie_predictions.append((proj_rookie, actual_rookie, name, pos))
        dynasty_predictions.append((proj_dynasty, actual_dynasty, name, pos))

        top_comp_names = [c['name'] for c in comps[:3]]
        top_comp_scores = [
            {k: round(v, 3) for k, v in c.get('category_scores', {}).items()}
            for c in comps[:3]
        ]

        results.append({
            'name': name,
            'position': pos,
            'archetype': prospect.get('archetype', '?'),
            'draft_round': prospect.get('draft_round'),
            'draft_pick': prospect.get('draft_pick'),
            'proj_rookie_ppg': proj_rookie,
            'actual_rookie_ppg': actual_rookie,
            'rookie_error': round(proj_rookie - actual_rookie, 2) if proj_rookie and actual_rookie else None,
            'proj_dynasty_ppg': proj_dynasty,
            'actual_dynasty_ppg': actual_dynasty,
            'dynasty_error': round(proj_dynasty - actual_dynasty, 2) if proj_dynasty and actual_dynasty else None,
            'top_comps': top_comp_names,
            'top_comp_cat_scores': top_comp_scores,
            'avg_similarity': round(
                sum(c['similarity_score'] for c in comps[:5]) / min(5, len(comps)), 3
            ) if comps else 0,
            # Efficiency features for diagnosis
            'college_td_rate': prospect.get('college_td_rate'),
            'college_ypr': prospect.get('college_ypr'),
            'college_ypc': prospect.get('college_ypc'),
            'breakout_age': prospect.get('breakout_age'),
            'rec_yds_per_team_game': prospect.get('rec_yds_per_team_game'),
        })

    rookie_accuracy = compute_accuracy(rookie_predictions)
    dynasty_accuracy = compute_accuracy(dynasty_predictions)
    miss_analysis = analyze_misses(results)

    return {
        'year': test_year,
        'n_historical': len(historical),
        'n_test': len(test_prospects),
        'results': results,
        'rookie_accuracy': rookie_accuracy,
        'dynasty_accuracy': dynasty_accuracy,
        'miss_analysis': miss_analysis,
    }


def run_all_backtests():
    """Run backtests across all test years."""
    profiles = load_profiles()

    # CRITICAL: Enrich profiles with efficiency features
    profiles = enrich_profiles(profiles)

    # Remove 2026 prospects for backtesting
    profiles = [p for p in profiles if not p.get('is_prospect')]

    print("=" * 70)
    print("BACKTESTING MODEL (v2 — with efficiency features)")
    print("=" * 70)
    print(f"Total NFL profiles: {len(profiles)}")

    # Verify enrichment
    with_td_rate = sum(1 for p in profiles if p.get('college_td_rate') is not None)
    with_breakout = sum(1 for p in profiles if p.get('breakout_age') is not None)
    with_rec_dom = sum(1 for p in profiles if p.get('rec_yds_per_team_game') is not None)
    print(f"With TD rate: {with_td_rate}")
    print(f"With breakout age: {with_breakout}")
    print(f"With rec dominance: {with_rec_dom}")

    all_results = {}
    all_rookie_preds = []
    all_dynasty_preds = []

    for year in BACKTEST_YEARS:
        print(f"\n{'─' * 70}")
        print(f"📅 BACKTEST: {year} Draft Class")
        print(f"{'─' * 70}")

        bt = run_backtest(year, profiles)
        all_results[year] = bt

        for r in bt['results']:
            if r['proj_rookie_ppg'] is not None and r['actual_rookie_ppg'] is not None:
                all_rookie_preds.append(
                    (r['proj_rookie_ppg'], r['actual_rookie_ppg'], r['name'], r['position'])
                )
            if r['proj_dynasty_ppg'] is not None and r['actual_dynasty_ppg'] is not None:
                all_dynasty_preds.append(
                    (r['proj_dynasty_ppg'], r['actual_dynasty_ppg'], r['name'], r['position'])
                )

        # Print year results
        print(f"\n  Rookie PPG Accuracy:")
        for k, v in bt['rookie_accuracy'].items():
            print(f"    {k}: {v}")

        print(f"\n  Dynasty PPG Accuracy:")
        for k, v in bt['dynasty_accuracy'].items():
            print(f"    {k}: {v}")

        # Top hits and misses
        sorted_by_error = sorted(
            [r for r in bt['results'] if r.get('rookie_error') is not None],
            key=lambda x: abs(x['rookie_error'])
        )

        if sorted_by_error:
            print(f"\n  🎯 Best predictions (rookie):")
            for r in sorted_by_error[:5]:
                cats = r.get('top_comp_cat_scores', [{}])[0] if r.get('top_comp_cat_scores') else {}
                cat_str = " | ".join(f"{k}: {v:.0%}" for k, v in cats.items()) if cats else ''
                print(f"    {r['name']:<25} Proj: {r['proj_rookie_ppg']:>6.1f}  "
                      f"Actual: {r['actual_rookie_ppg']:>6.1f}  "
                      f"Error: {r['rookie_error']:>+6.1f}  "
                      f"Comps: {r['top_comps']}")
                if cat_str:
                    print(f"    {'':25} [{cat_str}]")

            print(f"\n  ❌ Worst predictions (rookie):")
            for r in sorted_by_error[-5:]:
                cats = r.get('top_comp_cat_scores', [{}])[0] if r.get('top_comp_cat_scores') else {}
                cat_str = " | ".join(f"{k}: {v:.0%}" for k, v in cats.items()) if cats else ''
                print(f"    {r['name']:<25} Proj: {r['proj_rookie_ppg']:>6.1f}  "
                      f"Actual: {r['actual_rookie_ppg']:>6.1f}  "
                      f"Error: {r['rookie_error']:>+6.1f}  "
                      f"Comps: {r['top_comps']}")
                if cat_str:
                    print(f"    {'':25} [{cat_str}]")

        # Miss analysis
        miss = bt['miss_analysis']
        if miss['total_big_misses'] > 0:
            print(f"\n  📊 Miss Analysis (errors > 7 PPG):")
            print(f"    Over-projected: {miss['over_projected']}")
            print(f"    Under-projected: {miss['under_projected']}")

            if miss['over_proj_details']:
                print(f"\n    Over-projections:")
                for d in miss['over_proj_details']:
                    print(f"      {d['name']:<25} Error: {d['error']:>+6.1f}  "
                          f"Reason: {d['reason_guess']}")

            if miss['under_proj_details']:
                print(f"\n    Under-projections:")
                for d in miss['under_proj_details']:
                    print(f"      {d['name']:<25} Error: {d['error']:>+6.1f}  "
                          f"Reason: {d['reason_guess']}")

    # Aggregate accuracy
    print(f"\n{'═' * 70}")
    print("AGGREGATE ACCURACY (ALL BACKTEST YEARS)")
    print(f"{'═' * 70}")

    agg_rookie = compute_accuracy(all_rookie_preds)
    agg_dynasty = compute_accuracy(all_dynasty_preds)

    print(f"\n  Rookie PPG (n={agg_rookie.get('n', 0)}):")
    for k, v in agg_rookie.items():
        print(f"    {k}: {v}")

    print(f"\n  Dynasty PPG (n={agg_dynasty.get('n', 0)}):")
    for k, v in agg_dynasty.items():
        print(f"    {k}: {v}")

    # Per-position accuracy
    print(f"\n{'─' * 70}")
    print("PER-POSITION ROOKIE ACCURACY")
    print(f"{'─' * 70}")

    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_preds = [(p, a, n, ps) for p, a, n, ps in all_rookie_preds if ps == pos]
        if pos_preds:
            pos_acc = compute_accuracy(pos_preds)
            within5 = pos_acc.get('within_5_ppg', 0)
            print(f"\n  {pos} (n={pos_acc.get('n', 0)}):")
            print(f"    MAE: {pos_acc.get('mae', '?')}  "
                  f"Correlation: {pos_acc.get('correlation', '?')}  "
                  f"Within 5 PPG: {within5:.0%}")

    # Category weight analysis
    print(f"\n{'─' * 70}")
    print("CATEGORY WEIGHTS IN USE")
    print(f"{'─' * 70}")
    for cat, weight in CATEGORY_WEIGHTS.items():
        print(f"  {cat}: {weight:.0%}")

    # Spot check: Puka Nacua efficiency features
    print(f"\n{'─' * 70}")
    print("SPOT CHECK: Key Under-Projections — Efficiency Features")
    print(f"{'─' * 70}")

    spotlight_names = ['puka nacua', 'sam laporta', 'bucky irving', 'tank dell',
                       'brock bowers', 'dameon pierce']

    for year, bt in all_results.items():
        for r in bt['results']:
            if r['name'].lower() in spotlight_names:
                print(f"\n  {r['name']} ({year}):")
                print(f"    Proj: {r['proj_rookie_ppg']:.1f}  "
                      f"Actual: {r['actual_rookie_ppg']:.1f}  "
                      f"Error: {r['rookie_error']:+.1f}")
                td_r = r.get('college_td_rate')
                ypr_r = r.get('college_ypr')
                dom_r = r.get('rec_yds_per_team_game')
                ba_r = r.get('breakout_age')
                print(f"    TD Rate: {f'{td_r:.3f}' if td_r else 'N/A'}")
                print(f"    YPR: {f'{ypr_r:.1f}' if ypr_r else 'N/A'}")
                print(f"    Rec Dominance: {f'{dom_r:.1f}' if dom_r else 'N/A'}")
                print(f"    Breakout Age: {f'{ba_r:.1f}' if ba_r else 'N/A'}")
                print(f"    Draft: Rd {r.get('draft_round', '?')}, Pick {r.get('draft_pick', '?')}")
                print(f"    Comps: {r['top_comps']}")
                print(f"    Comp cat scores: {r.get('top_comp_cat_scores', [])[:1]}")

    # Save results
    output = {
        'backtest_years': BACKTEST_YEARS,
        'category_weights': CATEGORY_WEIGHTS,
        'per_year': {},
        'aggregate': {
            'rookie_accuracy': agg_rookie,
            'dynasty_accuracy': agg_dynasty,
        },
    }
    for year, bt in all_results.items():
        output['per_year'][year] = {
            'n_historical': bt['n_historical'],
            'n_test': bt['n_test'],
            'rookie_accuracy': bt['rookie_accuracy'],
            'dynasty_accuracy': bt['dynasty_accuracy'],
            'miss_analysis': bt['miss_analysis'],
            'results': bt['results'],
        }

    output_path = os.path.join(PROCESSED_DIR, "backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Saved: {output_path}")

    return output


def main():
    run_all_backtests()


if __name__ == "__main__":
    main()