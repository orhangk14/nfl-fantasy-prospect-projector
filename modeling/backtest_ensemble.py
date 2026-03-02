# modeling/backtest_ensemble.py
"""
Ensemble backtest + grid search optimizer.

Tests every combination of:
  - Ensemble weights (similarity vs RF vs GB in 0.05 increments)
  - Similarity exponent (2, 3)
  - Similarity comp floor (0.50, 0.65, 0.85 of best comp)
  - Draft capital adjustment blend (0.0, 0.35, 0.55, 0.75)

Evaluates honestly using Leave-One-Year-Out across 2022-2024.
Finds the optimal ensemble configuration and saves it.

Run: python -m modeling.backtest_ensemble
"""

import json
import os
import math
import time
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

from modeling.similarity import (
    POS_FEATURES, compute_feature_stats, compute_similarity,
    load_profiles, enrich_profiles
)
from modeling.ml_models import MLProjectionEngine, TARGETS, build_training_data

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
BACKTEST_YEARS = [2022, 2023, 2024]


# ─── Helpers ──────────────────────────────────────────────────────────────

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


def find_comps(prospect, pool, n=10):
    pos = prospect.get('position', '')
    features = POS_FEATURES.get(pos, [])
    if not features:
        return []
    pos_players = [p for p in pool if p.get('position') == pos]
    if not pos_players:
        return []
    all_pool = pos_players + [prospect]
    feat_stats = compute_feature_stats(all_pool, features)
    comparisons = []
    for nfl_p in pos_players:
        score, _, cat_scores = compute_similarity(
            prospect, nfl_p, features, feat_stats
        )
        comparisons.append({
            'similarity_score': score,
            'rookie_ppr_ppg': nfl_p.get('rookie_ppr_ppg'),
            'dynasty_ppg': nfl_p.get('dynasty_ppg'),
            'peak_3yr_ppr_ppg': nfl_p.get('peak_3yr_ppr_ppg'),
            'best_season_ppr_ppg': nfl_p.get('best_season_ppr_ppg'),
        })
    comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
    return comparisons[:n]


def sim_weighted_projection(comparisons, key, top_n=10, exp=2, sim_floor_pct=0.85):
    valid = [
        c for c in comparisons[:top_n]
        if c.get(key) is not None and c.get('similarity_score', 0) > 0
    ]
    if not valid:
        return None

    # Apply similarity floor filtering
    best_sim = max(c['similarity_score'] for c in valid)
    floor = best_sim * sim_floor_pct
    filtered = [c for c in valid if c['similarity_score'] >= floor]
    if len(filtered) < 3:
        filtered = valid[:max(3, len(filtered))]

    weights = [c['similarity_score'] ** exp for c in filtered]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    return sum(c[key] * w for c, w in zip(filtered, weights)) / total_weight


def draft_capital_adjustment(projection, draft_round, draft_pick, pos, blend=0.55):
    if projection is None or draft_round is None:
        return projection
    try:
        pick = float(draft_pick) if draft_pick else float(draft_round) * 32
    except (ValueError, TypeError):
        return projection

    if pick <= 10:
        raw = 1.35
    elif pick <= 20:
        raw = 1.25
    elif pick <= 32:
        raw = 1.18
    elif pick <= 48:
        raw = 1.10
    elif pick <= 64:
        raw = 1.05
    elif pick <= 100:
        raw = 1.00
    elif pick <= 140:
        raw = 0.92
    elif pick <= 180:
        raw = 0.85
    elif pick <= 224:
        raw = 0.78
    else:
        raw = 0.70

    effective = 1.0 + (raw - 1.0) * blend
    return projection * effective


def compute_metrics(predictions):
    valid = [(p, a, n, ps) for p, a, n, ps in predictions
             if p is not None and a is not None]
    if len(valid) < 5:
        return {'mae': 99, 'rmse': 99, 'correlation': 0, 'within_5': 0, 'bias': 0, 'n': 0}

    errors = [p - a for p, a, _, _ in valid]
    abs_errors = [abs(e) for e in errors]
    projs = [p for p, _, _, _ in valid]
    actuals = [a for _, a, _, _ in valid]
    n = len(errors)

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    bias = sum(errors) / n
    within_5 = sum(1 for e in abs_errors if e <= 5) / n

    mean_p = sum(projs) / n
    mean_a = sum(actuals) / n
    cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(projs, actuals)) / n
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in projs) / n)
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)
    correlation = cov / (std_p * std_a) if std_p > 0 and std_a > 0 else 0

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'correlation': round(correlation, 4),
        'within_5': round(within_5, 4),
        'bias': round(bias, 4),
        'n': n,
    }


def combined_score(rookie_m, dynasty_m):
    """
    Optimization objective. Weighted blend of metrics.
    Prioritizes: low MAE > high correlation > high within_5.
    Rookie weighted slightly more than dynasty (more testable).
    """
    return (
        -0.30 * rookie_m.get('mae', 10)
        + 0.20 * rookie_m.get('correlation', 0) * 10
        + 0.10 * rookie_m.get('within_5', 0) * 10
        - 0.20 * dynasty_m.get('mae', 10)
        + 0.15 * dynasty_m.get('correlation', 0) * 10
        + 0.05 * dynasty_m.get('within_5', 0) * 10
    )


# ─── Precompute ML predictions per year ──────────────────────────────────

def precompute_ml_predictions(profiles):
    """
    For each test year, train ML models on prior years, predict test year.
    Returns: dict of (year, player_name, target) -> {model_name: prediction}
    """
    print("  Precomputing ML predictions for all backtest years...")
    ml_cache = {}

    for test_year in BACKTEST_YEARS:
        train_profiles = []
        test_profiles = []

        for p in profiles:
            if p.get('is_prospect'):
                continue
            dy = p.get('draft_year')
            has_outcomes = p.get('rookie_ppr_ppg') is not None
            if dy is None:
                if has_outcomes:
                    train_profiles.append(p)
                continue
            try:
                dy = int(float(dy))
            except (ValueError, TypeError):
                if has_outcomes:
                    train_profiles.append(p)
                continue
            if dy < test_year and has_outcomes:
                train_profiles.append(p)
            elif dy == test_year and has_outcomes:
                test_profiles.append(p)

        # Train ML engine on historical data only
        engine = MLProjectionEngine()
        engine.train(train_profiles, targets=['rookie_ppr_ppg', 'dynasty_ppg'],
                     positions=['QB', 'RB', 'WR', 'TE'])

        # Predict each test player
        for prospect in test_profiles:
            name = prospect.get('name', '?')
            pos = prospect.get('position', '')

            for target in ['rookie_ppr_ppg', 'dynasty_ppg']:
                preds = engine.predict(prospect, pos, target)
                # Only keep RF and GB
                filtered_preds = {
                    k: v for k, v in preds.items()
                    if k in ('random_forest', 'gradient_boosting')
                }
                ml_cache[(test_year, name, target)] = filtered_preds

        print(f"    {test_year}: trained on {len(train_profiles)}, "
              f"predicted {len(test_profiles)} prospects")

    return ml_cache


# ─── Precompute similarity comps per year ─────────────────────────────────

def precompute_similarity_comps(profiles):
    """
    For each test year, compute similarity comps for all test prospects.
    Returns: dict of (year, player_name) -> comps list
    """
    print("  Precomputing similarity comps for all backtest years...")
    sim_cache = {}

    for test_year in BACKTEST_YEARS:
        historical, test_prospects = split_by_draft_year(profiles, test_year)

        for prospect in test_prospects:
            name = prospect.get('name', '?')
            comps = find_comps(prospect, historical, n=15)  # get extra for filtering experiments
            sim_cache[(test_year, name)] = comps

        print(f"    {test_year}: {len(historical)} historical, "
              f"{len(test_prospects)} test prospects")

    return sim_cache


# ─── Grid search evaluator ────────────────────────────────────────────────

def evaluate_ensemble_config(profiles, sim_cache, ml_cache, config):
    """
    Evaluate a single ensemble configuration across all backtest years.

    config = {
        'sim_weight': float,
        'rf_weight': float,
        'gb_weight': float,
        'sim_exp': int,
        'sim_floor_pct': float,
        'draft_adj_blend': float,
    }
    """
    sim_w = config['sim_weight']
    rf_w = config['rf_weight']
    gb_w = config['gb_weight']
    sim_exp = config['sim_exp']
    sim_floor = config['sim_floor_pct']
    draft_blend = config['draft_adj_blend']

    all_rookie_preds = []
    all_dynasty_preds = []

    for test_year in BACKTEST_YEARS:
        _, test_prospects = split_by_draft_year(profiles, test_year)

        for prospect in test_prospects:
            name = prospect.get('name', '?')
            pos = prospect.get('position', '')
            actual_rookie = prospect.get('rookie_ppr_ppg')
            actual_dynasty = prospect.get('dynasty_ppg')

            comps = sim_cache.get((test_year, name), [])
            ml_preds_rookie = ml_cache.get((test_year, name, 'rookie_ppr_ppg'), {})
            ml_preds_dynasty = ml_cache.get((test_year, name, 'dynasty_ppg'), {})

            for target, actual, ml_preds in [
                ('rookie_ppr_ppg', actual_rookie, ml_preds_rookie),
                ('dynasty_ppg', actual_dynasty, ml_preds_dynasty),
            ]:
                if actual is None:
                    continue

                # Similarity projection
                sim_proj = sim_weighted_projection(
                    comps, target, top_n=10, exp=sim_exp, sim_floor_pct=sim_floor
                )

                # Apply draft capital adjustment to similarity
                if sim_proj is not None and draft_blend > 0:
                    sim_proj = draft_capital_adjustment(
                        sim_proj,
                        prospect.get('draft_round'),
                        prospect.get('draft_pick'),
                        pos,
                        blend=draft_blend,
                    )

                # Blend
                all_preds = {}
                weights = {}
                if sim_proj is not None:
                    all_preds['similarity'] = sim_proj
                    weights['similarity'] = sim_w
                if 'random_forest' in ml_preds:
                    all_preds['random_forest'] = ml_preds['random_forest']
                    weights['random_forest'] = rf_w
                if 'gradient_boosting' in ml_preds:
                    all_preds['gradient_boosting'] = ml_preds['gradient_boosting']
                    weights['gradient_boosting'] = gb_w

                if not all_preds:
                    continue

                total_w = sum(weights.values())
                if total_w == 0:
                    continue

                blended = sum(
                    all_preds[k] * (weights[k] / total_w)
                    for k in all_preds
                )

                if target == 'rookie_ppr_ppg':
                    all_rookie_preds.append((blended, actual, name, pos))
                else:
                    all_dynasty_preds.append((blended, actual, name, pos))

    rookie_m = compute_metrics(all_rookie_preds)
    dynasty_m = compute_metrics(all_dynasty_preds)
    score = combined_score(rookie_m, dynasty_m)

    return rookie_m, dynasty_m, score


# ─── Main grid search ────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("ENSEMBLE BACKTEST + GRID SEARCH OPTIMIZER")
    print("=" * 80)
    print(f"Backtest years: {BACKTEST_YEARS}")
    print()

    # Load and enrich profiles
    profiles = load_profiles()
    profiles = enrich_profiles(profiles)
    profiles = [p for p in profiles if not p.get('is_prospect')]
    print(f"Total NFL profiles: {len(profiles)}")

    # ── Phase 1: Precompute everything ──────────────────────────
    print(f"\n{'─' * 80}")
    print("PHASE 1: Precomputing (this takes a few minutes)")
    print(f"{'─' * 80}")

    t0 = time.time()
    sim_cache = precompute_similarity_comps(profiles)
    ml_cache = precompute_ml_predictions(profiles)
    precompute_time = time.time() - t0
    print(f"\n  Precompute time: {precompute_time:.1f}s")
    print(f"  Similarity cache: {len(sim_cache)} entries")
    print(f"  ML cache: {len(ml_cache)} entries")

    # ── Phase 2: Grid search ────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("PHASE 2: Grid Search Over Ensemble Configurations")
    print(f"{'─' * 80}")

    # Weight grid: sim + rf + gb = 1.0, in 0.05 increments
    weight_steps = [i * 0.05 for i in range(1, 19)]  # 0.05 to 0.90

    sim_exponents = [2, 3]
    sim_floors = [0.50, 0.65, 0.85]
    draft_adj_blends = [0.0, 0.35, 0.55, 0.75]

    configs_to_test = []
    for sim_w in weight_steps:
        for rf_w in weight_steps:
            gb_w = round(1.0 - sim_w - rf_w, 2)
            if gb_w < 0.05 or gb_w > 0.90:
                continue
            # Skip silly configs where one model is >80%
            if sim_w > 0.80 or rf_w > 0.80 or gb_w > 0.80:
                continue
            for sim_exp in sim_exponents:
                for sim_floor in sim_floors:
                    for draft_blend in draft_adj_blends:
                        configs_to_test.append({
                            'sim_weight': sim_w,
                            'rf_weight': rf_w,
                            'gb_weight': gb_w,
                            'sim_exp': sim_exp,
                            'sim_floor_pct': sim_floor,
                            'draft_adj_blend': draft_blend,
                        })

    total_configs = len(configs_to_test)
    print(f"  Total configurations to test: {total_configs}")

    t1 = time.time()
    results = []
    best_score = -999
    best_config = None
    best_metrics = None

    for i, config in enumerate(configs_to_test):
        rookie_m, dynasty_m, score = evaluate_ensemble_config(
            profiles, sim_cache, ml_cache, config
        )

        results.append({
            'config': config,
            'rookie': rookie_m,
            'dynasty': dynasty_m,
            'score': score,
        })

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = {'rookie': rookie_m, 'dynasty': dynasty_m}

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            remaining = (total_configs - i - 1) / rate
            print(f"  [{i+1}/{total_configs}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining | "
                  f"Best: score={best_score:.4f} "
                  f"Sim={best_config['sim_weight']:.0%} "
                  f"RF={best_config['rf_weight']:.0%} "
                  f"GB={best_config['gb_weight']:.0%}")

    search_time = time.time() - t1
    print(f"\n  Grid search time: {search_time:.1f}s ({total_configs} configs)")

    # ── Phase 3: Results ────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("RESULTS")
    print(f"{'═' * 80}")

    # Sort all results
    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  🏆 OPTIMAL ENSEMBLE CONFIGURATION:")
    print(f"  {'─' * 60}")
    print(f"  Similarity weight:    {best_config['sim_weight']:.0%}")
    print(f"  Random Forest weight: {best_config['rf_weight']:.0%}")
    print(f"  Gradient Boost weight:{best_config['gb_weight']:.0%}")
    print(f"  Similarity exponent:  {best_config['sim_exp']}")
    print(f"  Similarity comp floor:{best_config['sim_floor_pct']:.0%}")
    print(f"  Draft adj blend:      {best_config['draft_adj_blend']}")
    print(f"  Combined score:       {best_score:.4f}")
    print()
    print(f"  Rookie PPG:")
    for k, v in best_metrics['rookie'].items():
        print(f"    {k}: {v}")
    print(f"\n  Dynasty PPG:")
    for k, v in best_metrics['dynasty'].items():
        print(f"    {k}: {v}")

    # Top 20
    print(f"\n  {'─' * 80}")
    print(f"  TOP 20 CONFIGURATIONS")
    print(f"  {'─' * 80}")
    print(f"  {'Rank':<5} {'Score':>7} {'Sim':>5} {'RF':>5} {'GB':>5} "
          f"{'Exp':>4} {'Floor':>6} {'DAdj':>5} "
          f"{'R_MAE':>7} {'R_Corr':>7} {'R_W5':>6} "
          f"{'D_MAE':>7} {'D_Corr':>7} {'D_W5':>6}")

    for i, r in enumerate(results[:20], 1):
        c = r['config']
        rm = r['rookie']
        dm = r['dynasty']
        print(f"  {i:<5} {r['score']:>7.4f} "
              f"{c['sim_weight']:>5.0%} {c['rf_weight']:>5.0%} {c['gb_weight']:>5.0%} "
              f"{c['sim_exp']:>4} {c['sim_floor_pct']:>5.0%} {c['draft_adj_blend']:>5.2f} "
              f"{rm['mae']:>7.3f} {rm['correlation']:>7.3f} {rm['within_5']:>6.1%} "
              f"{dm['mae']:>7.3f} {dm['correlation']:>7.3f} {dm['within_5']:>6.1%}")

    # ── Comparisons to baselines ────────────────────────────────
    print(f"\n  {'─' * 80}")
    print(f"  BASELINE COMPARISONS")
    print(f"  {'─' * 80}")

    baselines = [
        {'sim_weight': 1.0, 'rf_weight': 0.0, 'gb_weight': 0.0,
         'sim_exp': 2, 'sim_floor_pct': 0.85, 'draft_adj_blend': 0.55,
         'label': 'Similarity Only (current production)'},
        {'sim_weight': 0.0, 'rf_weight': 1.0, 'gb_weight': 0.0,
         'sim_exp': 2, 'sim_floor_pct': 0.85, 'draft_adj_blend': 0.0,
         'label': 'Random Forest Only'},
        {'sim_weight': 0.0, 'rf_weight': 0.0, 'gb_weight': 1.0,
         'sim_exp': 2, 'sim_floor_pct': 0.85, 'draft_adj_blend': 0.0,
         'label': 'Gradient Boosting Only'},
        {'sim_weight': 0.35, 'rf_weight': 0.35, 'gb_weight': 0.30,
         'sim_exp': 2, 'sim_floor_pct': 0.85, 'draft_adj_blend': 0.55,
         'label': 'Current Ensemble (Sim 35/RF 35/GB 30)'},
    ]

    for baseline in baselines:
        label = baseline.pop('label')
        rm, dm, sc = evaluate_ensemble_config(profiles, sim_cache, ml_cache, baseline)
        print(f"\n  {label}:")
        print(f"    Score: {sc:.4f}  "
              f"Rookie MAE: {rm['mae']:.3f}  Corr: {rm['correlation']:.3f}  W5: {rm['within_5']:.1%}  "
              f"Dynasty MAE: {dm['mae']:.3f}  Corr: {dm['correlation']:.3f}  W5: {dm['within_5']:.1%}")

    # ── Sensitivity analysis ────────────────────────────────────
    print(f"\n  {'─' * 80}")
    print(f"  SENSITIVITY: How stable is the optimum?")
    print(f"  {'─' * 80}")

    # Check how much the top configs differ
    top_5_scores = [r['score'] for r in results[:5]]
    top_20_scores = [r['score'] for r in results[:20]]
    print(f"  Top 5 score range:  {min(top_5_scores):.4f} — {max(top_5_scores):.4f} "
          f"(spread: {max(top_5_scores) - min(top_5_scores):.4f})")
    print(f"  Top 20 score range: {min(top_20_scores):.4f} — {max(top_20_scores):.4f} "
          f"(spread: {max(top_20_scores) - min(top_20_scores):.4f})")

    # Check if top configs cluster around similar weights
    top_10_sim_w = [r['config']['sim_weight'] for r in results[:10]]
    top_10_rf_w = [r['config']['rf_weight'] for r in results[:10]]
    top_10_gb_w = [r['config']['gb_weight'] for r in results[:10]]
    print(f"\n  Top 10 weight ranges:")
    print(f"    Similarity: {min(top_10_sim_w):.0%} — {max(top_10_sim_w):.0%} "
          f"(avg: {sum(top_10_sim_w)/10:.0%})")
    print(f"    RF:         {min(top_10_rf_w):.0%} — {max(top_10_rf_w):.0%} "
          f"(avg: {sum(top_10_rf_w)/10:.0%})")
    print(f"    GB:         {min(top_10_gb_w):.0%} — {max(top_10_gb_w):.0%} "
          f"(avg: {sum(top_10_gb_w)/10:.0%})")

    # ── Per-position breakdown of optimal config ────────────────
    print(f"\n  {'─' * 80}")
    print(f"  PER-POSITION BREAKDOWN (optimal config)")
    print(f"  {'─' * 80}")

    # Re-run optimal config collecting per-player predictions
    all_preds_by_pos = defaultdict(list)

    for test_year in BACKTEST_YEARS:
        _, test_prospects = split_by_draft_year(profiles, test_year)
        for prospect in test_prospects:
            name = prospect.get('name', '?')
            pos = prospect.get('position', '')
            actual = prospect.get('rookie_ppr_ppg')
            if actual is None:
                continue

            comps = sim_cache.get((test_year, name), [])
            ml_preds = ml_cache.get((test_year, name, 'rookie_ppr_ppg'), {})

            sim_proj = sim_weighted_projection(
                comps, 'rookie_ppr_ppg', top_n=10,
                exp=best_config['sim_exp'],
                sim_floor_pct=best_config['sim_floor_pct']
            )
            if sim_proj is not None and best_config['draft_adj_blend'] > 0:
                sim_proj = draft_capital_adjustment(
                    sim_proj, prospect.get('draft_round'),
                    prospect.get('draft_pick'), pos,
                    blend=best_config['draft_adj_blend']
                )

            all_p = {}
            weights = {}
            if sim_proj is not None:
                all_p['similarity'] = sim_proj
                weights['similarity'] = best_config['sim_weight']
            if 'random_forest' in ml_preds:
                all_p['random_forest'] = ml_preds['random_forest']
                weights['random_forest'] = best_config['rf_weight']
            if 'gradient_boosting' in ml_preds:
                all_p['gradient_boosting'] = ml_preds['gradient_boosting']
                weights['gradient_boosting'] = best_config['gb_weight']

            if not all_p:
                continue
            tw = sum(weights.values())
            if tw == 0:
                continue
            blended = sum(all_p[k] * (weights[k] / tw) for k in all_p)
            all_preds_by_pos[pos].append((blended, actual, name, pos))

    for pos in ['QB', 'RB', 'WR', 'TE']:
        preds = all_preds_by_pos.get(pos, [])
        if preds:
            m = compute_metrics(preds)
            print(f"\n  {pos} (n={m['n']}):")
            print(f"    MAE: {m['mae']:.3f}  Corr: {m['correlation']:.3f}  "
                  f"Within 5: {m['within_5']:.1%}  Bias: {m['bias']:+.3f}")

    # ── Save results ────────────────────────────────────────────
    output = {
        'optimal_config': best_config,
        'optimal_score': best_score,
        'optimal_metrics': best_metrics,
        'total_configs_tested': total_configs,
        'precompute_time_s': round(precompute_time, 1),
        'search_time_s': round(search_time, 1),
        'top_20': [
            {
                'rank': i + 1,
                'config': r['config'],
                'score': r['score'],
                'rookie_mae': r['rookie']['mae'],
                'rookie_corr': r['rookie']['correlation'],
                'rookie_within_5': r['rookie']['within_5'],
                'dynasty_mae': r['dynasty']['mae'],
                'dynasty_corr': r['dynasty']['correlation'],
                'dynasty_within_5': r['dynasty']['within_5'],
            }
            for i, r in enumerate(results[:20])
        ],
        'sensitivity': {
            'top_10_sim_weight_range': [min(top_10_sim_w), max(top_10_sim_w)],
            'top_10_rf_weight_range': [min(top_10_rf_w), max(top_10_rf_w)],
            'top_10_gb_weight_range': [min(top_10_gb_w), max(top_10_gb_w)],
        },
    }

    output_path = os.path.join(PROCESSED_DIR, "ensemble_optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Saved: {output_path}")

    # ── Final recommendation ────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("RECOMMENDATION")
    print(f"{'═' * 80}")
    print(f"\n  Update modeling/ensemble.py with:")
    print(f"    ENSEMBLE_WEIGHTS = {{")
    print(f"        'similarity': {best_config['sim_weight']},")
    print(f"        'random_forest': {best_config['rf_weight']},")
    print(f"        'gradient_boosting': {best_config['gb_weight']},")
    print(f"    }}")
    print(f"\n  In modeling/projections.py weighted_projection():")
    print(f"    exp={best_config['sim_exp']}")
    print(f"    sim_floor = best_sim * {best_config['sim_floor_pct']}")
    print(f"\n  In modeling/projections.py draft_capital_adjustment():")
    print(f"    residual_blend = {best_config['draft_adj_blend']}")
    print(f"\n  Then re-run:")
    print(f"    python -m modeling.ensemble")
    print(f"    streamlit run app.py")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()