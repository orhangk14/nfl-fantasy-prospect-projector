# modeling/tune.py
"""
Auto-tuner: optimizes category weights and model parameters
using backtest results across 2022-2024 draft classes.

Uses grid search over weight combinations to minimize MAE
and maximize correlation.

Run: python -m modeling.tune
"""

import json
import os
import math
import itertools
from collections import defaultdict
from modeling.similarity import (
    POS_FEATURES, compute_feature_stats, compute_similarity,
    load_profiles, enrich_profiles
)

PROCESSED_DIR = "data/processed"
BACKTEST_YEARS = [2022, 2023, 2024]


def split_by_draft_year(profiles, test_year):
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


def find_comps_with_weights(prospect, pool, features, cat_weights, n=10):
    """Find comps using custom category weights."""
    pos = prospect.get('position', '')
    pos_players = [p for p in pool if p.get('position') == pos]
    if not pos_players:
        return []

    all_pool = pos_players + [prospect]
    feat_stats = compute_feature_stats(all_pool, features)

    comparisons = []
    for nfl_p in pos_players:
        score, _, cat_scores = compute_similarity_custom(
            prospect, nfl_p, features, feat_stats, cat_weights
        )
        comparisons.append({
            'similarity_score': score,
            'rookie_ppr_ppg': nfl_p.get('rookie_ppr_ppg'),
            'dynasty_ppg': nfl_p.get('dynasty_ppg'),
        })

    comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
    return comparisons[:n]


def compute_similarity_custom(prospect, nfl_player, features, feat_stats, cat_weights):
    """Similarity with custom category weights."""
    cat_features = defaultdict(list)
    for feat_key, cat, higher_is_better in features:
        cat_features[cat].append((feat_key, higher_is_better))

    cat_scores = {}

    for cat, feat_list in cat_features.items():
        dot = 0
        mag_a = 0
        mag_b = 0
        euclidean_sum = 0
        valid_count = 0

        for feat_key, _ in feat_list:
            val_a = prospect.get(feat_key)
            val_b = nfl_player.get(feat_key)
            if val_a is None or val_b is None:
                continue

            mean, std = feat_stats.get(feat_key, (0, 1))
            if std == 0:
                std = 1
            norm_a = (val_a - mean) / std
            norm_b = (val_b - mean) / std

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
            cat_scores[cat] = 0.6 * max(cosine, 0) + 0.4 * closeness

    total_weight = 0
    weighted_score = 0
    for cat, weight in cat_weights.items():
        if cat in cat_scores:
            weighted_score += weight * cat_scores[cat]
            total_weight += weight

    final_score = weighted_score / total_weight if total_weight > 0 else 0
    return final_score, None, cat_scores


def weighted_projection(comparisons, key, top_n=10, exp=3):
    """Projection with configurable exponent."""
    valid = [
        c for c in comparisons[:top_n]
        if c.get(key) is not None and c.get('similarity_score', 0) > 0
    ]
    if not valid:
        return None

    weights = [c['similarity_score'] ** exp for c in valid]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    return sum(c[key] * w for c, w in zip(valid, weights)) / total_weight


def draft_capital_adjustment(projection, draft_round, draft_pick, pos):
    """
    Adjust projection based on draft capital.
    High picks get opportunity premium (more targets/carries/snaps).
    This captures the Tate/Lemon effect — first rounders get volume.
    """
    if projection is None or draft_round is None:
        return projection

    try:
        rnd = float(draft_round)
        pick = float(draft_pick) if draft_pick else rnd * 32
    except (ValueError, TypeError):
        return projection

    # Opportunity multiplier based on draft position
    # Round 1: 1.15-1.25x, Round 2: 1.05-1.10x, Round 3: 1.0x
    # Round 4+: 0.90-0.95x (less guaranteed opportunity)
    if pick <= 10:
        multiplier = 1.25
    elif pick <= 20:
        multiplier = 1.20
    elif pick <= 32:
        multiplier = 1.15
    elif pick <= 48:
        multiplier = 1.10
    elif pick <= 64:
        multiplier = 1.05
    elif pick <= 100:
        multiplier = 1.00
    elif pick <= 140:
        multiplier = 0.95
    elif pick <= 180:
        multiplier = 0.92
    else:
        multiplier = 0.88

    # Position-specific scaling
    # RBs especially benefit from opportunity (bellcow effect)
    # QBs benefit hugely (starter vs bench)
    pos_scale = {
        'QB': 1.3,   # starting vs backup is massive
        'RB': 1.2,   # volume is king for RBs
        'WR': 1.0,   # WRs can emerge regardless
        'TE': 0.9,   # TEs slower to develop regardless of capital
    }

    scale = pos_scale.get(pos, 1.0)

    # Blend: partial adjustment (don't go full multiplier)
    # Effective multiplier = 1 + (multiplier - 1) * scale * blend_factor
    blend = 0.6  # 60% of the adjustment
    effective = 1 + (multiplier - 1) * scale * blend

    return projection * effective


def evaluate_config(profiles, cat_weights, use_draft_adj=False,
                    draft_adj_blend=0.6, exp=3, top_n=10):
    """
    Evaluate a configuration across all backtest years.
    Returns aggregate metrics.
    """
    all_rookie_preds = []
    all_dynasty_preds = []

    for year in BACKTEST_YEARS:
        historical, test_prospects = split_by_draft_year(profiles, year)

        for prospect in test_prospects:
            pos = prospect.get('position', '')
            features = POS_FEATURES.get(pos, [])
            if not features:
                continue

            comps = find_comps_with_weights(
                prospect, historical, features, cat_weights, n=top_n
            )
            if not comps:
                continue

            proj_rookie = weighted_projection(comps, 'rookie_ppr_ppg', top_n, exp)
            proj_dynasty = weighted_projection(comps, 'dynasty_ppg', top_n, exp)

            if use_draft_adj:
                proj_rookie = draft_capital_adjustment(
                    proj_rookie,
                    prospect.get('draft_round'),
                    prospect.get('draft_pick'),
                    pos
                )
                proj_dynasty = draft_capital_adjustment(
                    proj_dynasty,
                    prospect.get('draft_round'),
                    prospect.get('draft_pick'),
                    pos
                )

            actual_rookie = prospect.get('rookie_ppr_ppg')
            actual_dynasty = prospect.get('dynasty_ppg')

            if proj_rookie is not None and actual_rookie is not None:
                all_rookie_preds.append((proj_rookie, actual_rookie,
                                        prospect.get('name', ''), pos))
            if proj_dynasty is not None and actual_dynasty is not None:
                all_dynasty_preds.append((proj_dynasty, actual_dynasty,
                                         prospect.get('name', ''), pos))

    rookie_metrics = compute_metrics(all_rookie_preds)
    dynasty_metrics = compute_metrics(all_dynasty_preds)

    return rookie_metrics, dynasty_metrics, all_rookie_preds, all_dynasty_preds


def compute_metrics(predictions):
    valid = [(p, a) for p, a, _, _ in predictions if p is not None and a is not None]
    if len(valid) < 5:
        return {'mae': 99, 'correlation': 0, 'within_5': 0, 'rmse': 99, 'n': 0}

    errors = [p - a for p, a in valid]
    abs_errors = [abs(e) for e in errors]
    projs = [p for p, _ in valid]
    actuals = [a for _, a in valid]
    n = len(errors)

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

    mean_p = sum(projs) / n
    mean_a = sum(actuals) / n
    cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(projs, actuals)) / n
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in projs) / n)
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)
    correlation = cov / (std_p * std_a) if std_p > 0 and std_a > 0 else 0

    within_5 = sum(1 for e in abs_errors if e <= 5) / n

    return {
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
        'correlation': round(correlation, 3),
        'within_5': round(within_5, 3),
        'n': n,
    }


def combined_score(rookie_metrics, dynasty_metrics):
    """
    Combined optimization score.
    We want: low MAE, high correlation, high within_5.
    Weight rookie more since it's more directly testable.
    """
    r = rookie_metrics
    d = dynasty_metrics

    # Normalize: lower MAE is better, higher correlation is better
    score = (
        -0.35 * r.get('mae', 10) +
         0.25 * r.get('correlation', 0) * 10 +
         0.15 * r.get('within_5', 0) * 10 +
        -0.10 * d.get('mae', 10) +
         0.10 * d.get('correlation', 0) * 10 +
         0.05 * d.get('within_5', 0) * 10
    )
    return round(score, 4)


def main():
    print("=" * 70)
    print("AUTO-TUNING MODEL PARAMETERS")
    print("=" * 70)

    profiles = load_profiles()
    profiles = enrich_profiles(profiles)
    profiles = [p for p in profiles if not p.get('is_prospect')]

    print(f"Total NFL profiles: {len(profiles)}")

    # ─── Grid search over category weights ──────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 1: Category Weight Optimization")
    print(f"{'─' * 70}")

    # Generate weight combinations (must sum to 1.0)
    # Search in increments of 0.05
    weight_options = [i / 20 for i in range(1, 12)]  # 0.05 to 0.55

    best_config = None
    best_score = -999
    results = []
    tested = 0

    # Smart grid: 5 categories now
    for prod_w in [0.10, 0.15, 0.20]:
        for peak_w in [0.15, 0.20, 0.25]:
            for eff_w in [0.05, 0.10, 0.15]:
                for draft_w in [0.30, 0.35, 0.40]:
                    meas_w = round(1.0 - prod_w - peak_w - eff_w - draft_w, 2)
                    if meas_w < 0.05 or meas_w > 0.25:
                        continue

                    cat_weights = {
                        'production': prod_w,
                        'peak': peak_w,
                        'efficiency': eff_w,
                        'draft': draft_w,
                        'measurable': meas_w,
                    }

                rookie_m, dynasty_m, _, _ = evaluate_config(
                    profiles, cat_weights, use_draft_adj=False
                )

                score = combined_score(rookie_m, dynasty_m)
                tested += 1

                results.append({
                    'weights': cat_weights.copy(),
                    'rookie': rookie_m,
                    'dynasty': dynasty_m,
                    'score': score,
                    'draft_adj': False,
                })

                if score > best_score:
                    best_score = score
                    best_config = {
                        'weights': cat_weights.copy(),
                        'rookie': rookie_m,
                        'dynasty': dynasty_m,
                        'score': score,
                    }

                if tested % 25 == 0:
                    print(f"  Tested {tested} configs... "
                          f"Best so far: score={best_score:.4f}")

    print(f"\n  Total configs tested: {tested}")
    print(f"\n  🏆 BEST WEIGHTS (no draft adjustment):")
    print(f"     {best_config['weights']}")
    print(f"     Score: {best_config['score']:.4f}")
    print(f"     Rookie — MAE: {best_config['rookie']['mae']:.2f}  "
          f"Corr: {best_config['rookie']['correlation']:.3f}  "
          f"Within 5: {best_config['rookie']['within_5']:.0%}")
    print(f"     Dynasty — MAE: {best_config['dynasty']['mae']:.2f}  "
          f"Corr: {best_config['dynasty']['correlation']:.3f}  "
          f"Within 5: {best_config['dynasty']['within_5']:.0%}")

    # ─── Phase 2: Test draft capital adjustment on top ──────────────
    print(f"\n{'─' * 70}")
    print("PHASE 2: Draft Capital Opportunity Adjustment")
    print(f"{'─' * 70}")

    best_weights = best_config['weights']

    adj_results = []
    for use_adj in [False, True]:
        rookie_m, dynasty_m, preds, _ = evaluate_config(
            profiles, best_weights, use_draft_adj=use_adj
        )
        score = combined_score(rookie_m, dynasty_m)
        adj_results.append({
            'draft_adj': use_adj,
            'rookie': rookie_m,
            'dynasty': dynasty_m,
            'score': score,
        })

        label = "WITH" if use_adj else "WITHOUT"
        print(f"\n  {label} draft capital adjustment:")
        print(f"    Score: {score:.4f}")
        print(f"    Rookie — MAE: {rookie_m['mae']:.2f}  "
              f"Corr: {rookie_m['correlation']:.3f}  "
              f"Within 5: {rookie_m['within_5']:.0%}")
        print(f"    Dynasty — MAE: {dynasty_m['mae']:.2f}  "
              f"Corr: {dynasty_m['correlation']:.3f}  "
              f"Within 5: {dynasty_m['within_5']:.0%}")

        # Spot check key players
        if use_adj:
            print(f"\n    Spot check (with adjustment):")
            spotlight = ['puka nacua', 'brock bowers', 'bucky irving',
                         'tank dell', 'sam laporta', 'carnell tate']
            for proj, actual, name, pos in preds:
                if name.lower() in spotlight:
                    err = proj - actual if proj and actual else None
                    print(f"      {name:<25} Proj: {proj:>6.1f}  "
                          f"Actual: {actual:>6.1f}  "
                          f"Error: {err:>+6.1f}" if err else
                          f"      {name:<25} Proj: {proj}  Actual: {actual}")

    # ─── Phase 3: Exponent tuning ──────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PHASE 3: Similarity Exponent Tuning")
    print(f"{'─' * 70}")

    best_adj = adj_results[0]['score'] < adj_results[1]['score']

    for exp in [2, 3, 4, 5]:
        rookie_m, dynasty_m, _, _ = evaluate_config(
            profiles, best_weights, use_draft_adj=best_adj, exp=exp
        )
        score = combined_score(rookie_m, dynasty_m)
        print(f"  Exponent {exp}: Score={score:.4f}  "
              f"Rookie MAE={rookie_m['mae']:.2f}  "
              f"Corr={rookie_m['correlation']:.3f}")

    # ─── Final recommendation ──────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("FINAL RECOMMENDATION")
    print(f"{'═' * 70}")

    # Sort all results
    all_results = results + adj_results
    top_10 = sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]

    print(f"\n  Top 10 configurations:")
    for i, cfg in enumerate(top_10, 1):
        w = cfg.get('weights', best_weights)
        adj = cfg.get('draft_adj', False)
        print(f"  {i}. Score: {cfg['score']:.4f}  "
              f"Weights: P={w['production']:.0%} E={w['efficiency']:.0%} "
              f"D={w['draft']:.0%} M={w['measurable']:.0%}  "
              f"DraftAdj: {adj}  "
              f"Rookie MAE: {cfg['rookie']['mae']:.2f}  "
              f"Corr: {cfg['rookie']['correlation']:.3f}")

    # Save recommendation
    recommendation = {
        'best_weights': best_config['weights'],
        'use_draft_adjustment': best_adj,
        'best_score': best_score,
        'metrics': {
            'rookie': best_config['rookie'],
            'dynasty': best_config['dynasty'],
        },
        'top_10_configs': [
            {
                'weights': cfg.get('weights', best_weights),
                'draft_adj': cfg.get('draft_adj', False),
                'score': cfg['score'],
                'rookie_mae': cfg['rookie']['mae'],
                'rookie_corr': cfg['rookie']['correlation'],
            }
            for cfg in top_10
        ],
    }

    output_path = os.path.join(PROCESSED_DIR, "tuning_results.json")
    with open(output_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    print(f"\n  ✅ Saved: {output_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()