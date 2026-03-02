# modeling/ensemble.py
"""
Ensemble projection engine: blends similarity-based projections
with ML model predictions for improved accuracy.

This replaces the projection step — run AFTER similarity.py and ml_models.py.

Ensemble weights (calibrated via backtest 2022-2024, n=198):
  - Similarity v3:      35%  (proven baseline, provides interpretable comps)
  - Random Forest:      35%  (best MAE: 3.29 rookie, 3.16 dynasty)
  - Gradient Boosting:  30%  (best dynasty correlation: 0.621)
  - KNN: excluded (worst on every metric — 3.60 MAE, 0.510 corr)

Run: python -m modeling.ensemble
"""

import json
import os
import math
from modeling.ml_models import MLProjectionEngine, TARGETS
from modeling.projections import (
    load_comparisons, load_profile_lookup,
    weighted_projection, draft_capital_adjustment,
    tier_label, bust_probability, breakout_probability
)
from modeling.similarity import load_profiles, enrich_profiles

PROCESSED_DIR = "data/processed"

# Ensemble weights — KNN excluded, RF promoted
# Calibrated against backtest 2022-2024 (n=198)
ENSEMBLE_WEIGHTS = {
    'similarity': 0.35,
    'random_forest': 0.35,
    'gradient_boosting': 0.30,
}

# Models to actually use in ensemble (excludes KNN)
ACTIVE_ML_MODELS = ['random_forest', 'gradient_boosting']


def blend_predictions(similarity_proj, ml_preds, weights=None):
    """
    Blend similarity projection with ML predictions.

    similarity_proj: float (from weighted_projection + draft adjustment)
    ml_preds: dict of model_name -> float (only active models used)
    weights: dict of source -> weight

    Returns: blended float prediction, detail dict
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    all_preds = {}
    if similarity_proj is not None:
        all_preds['similarity'] = similarity_proj

    # Only include active ML models
    if ml_preds:
        for model_name in ACTIVE_ML_MODELS:
            if model_name in ml_preds:
                all_preds[model_name] = ml_preds[model_name]

    if not all_preds:
        return None, {}

    # Normalize weights to available models
    available = {k: v for k, v in weights.items() if k in all_preds}
    total_w = sum(available.values())
    if total_w == 0:
        return None, {}

    blended = sum(all_preds[k] * (w / total_w) for k, w in available.items())

    details = {
        'blended': round(blended, 2),
        'individual': {k: round(v, 2) for k, v in all_preds.items()},
        'weights_used': {k: round(v / total_w, 3) for k, v in available.items()},
    }

    return round(blended, 2), details


def build_ensemble_projections():
    """
    Build final projections using ensemble of similarity + ML.
    """
    # Load similarity comparisons
    comparisons_data = load_comparisons()
    profile_lookup = load_profile_lookup()

    if not comparisons_data:
        print("ERROR: No comparison data. Run similarity.py first.")
        return {}

    # Load ML engine
    engine = MLProjectionEngine()
    ml_available = engine.load()

    if not ml_available:
        print("⚠️  No ML models found. Run ml_models.py first.")
        print("   Falling back to similarity-only projections.")

    # Load full profiles for ML prediction
    profiles = load_profiles()
    profiles = enrich_profiles(profiles)
    prospect_lookup = {
        str(p.get('espn_id')): p
        for p in profiles if p.get('is_prospect')
    }

    print(f"Loaded comparisons for {len(comparisons_data)} prospects")
    print(f"ML models available: {ml_available}")
    print(f"Active ML models: {ACTIVE_ML_MODELS}")
    print(f"Ensemble weights: {ENSEMBLE_WEIGHTS}")

    # Target mapping: our target names -> projection keys
    target_map = {
        'rookie_ppr_ppg': 'rookie',
        'dynasty_ppg': 'dynasty',
        'peak_3yr_ppr_ppg': 'peak_3yr',
        'best_season_ppr_ppg': 'best_season',
    }

    projections = {}

    for pid, data in comparisons_data.items():
        prospect_info = data.get('prospect', {})
        comps = data.get('comparisons', [])
        pos = prospect_info.get('position', '')

        if not comps:
            continue

        # Get full prospect profile for ML
        full_prospect = prospect_lookup.get(str(pid), prospect_info)

        # Draft capital info
        full_profile = profile_lookup.get(str(pid), {})
        draft_round = (
            full_profile.get('draft_round') or
            prospect_info.get('draft_round') or None
        )
        draft_pick = (
            full_profile.get('draft_pick') or
            prospect_info.get('draft_pick') or None
        )
        is_mock = full_profile.get('draft_capital_is_mock', False)

        proj_results = {}
        ensemble_details = {}

        for target, proj_key in target_map.items():
            # 1. Similarity-based projection
            sim_proj = weighted_projection(comps, target, exp=2)
            sim_value = None

            if sim_proj is not None:
                # Apply draft capital adjustment to similarity projection
                if draft_round is not None:
                    sim_proj = draft_capital_adjustment(
                        sim_proj, draft_round, draft_pick, pos
                    )
                sim_value = sim_proj['projected']

            # 2. ML predictions (all models, filtering happens in blend)
            ml_preds = {}
            if ml_available:
                ml_preds = engine.predict(full_prospect, pos, target)

            # 3. Ensemble blend (only uses active models)
            blended, details = blend_predictions(sim_value, ml_preds)

            # Build final projection dict
            if sim_proj is not None:
                final_proj = sim_proj.copy()
                if blended is not None:
                    final_proj['projected'] = blended
                    final_proj['similarity_projected'] = sim_value
                    final_proj['ml_predictions'] = {
                        k: v for k, v in ml_preds.items()
                        if k in ACTIVE_ML_MODELS
                    }
                    final_proj['ensemble_weights'] = details.get('weights_used', {})
                proj_results[proj_key] = final_proj
                ensemble_details[proj_key] = details
            elif blended is not None:
                # ML-only (no similarity comps — shouldn't happen but safety)
                proj_results[proj_key] = {
                    'projected': blended,
                    'ml_predictions': {
                        k: v for k, v in ml_preds.items()
                        if k in ACTIVE_ML_MODELS
                    },
                    'ensemble_weights': details.get('weights_used', {}),
                }
                ensemble_details[proj_key] = details

        # Career length (similarity only, not an ML target)
        career_proj = weighted_projection(comps, 'nfl_seasons_played', exp=2)
        if career_proj:
            proj_results['career_length'] = career_proj

        # Evaluation
        dynasty_ppg = proj_results.get('dynasty', {}).get('projected', 0) or 0
        tier = tier_label(dynasty_ppg, pos)
        bust_prob = bust_probability(comps, pos)
        breakout_prob = breakout_probability(comps, pos)

        # Enrich prospect info
        prospect_enriched = {**prospect_info}
        prospect_enriched['draft_round'] = draft_round
        prospect_enriched['draft_pick'] = draft_pick
        prospect_enriched['draft_is_mock'] = is_mock

        projections[pid] = {
            'prospect': prospect_enriched,
            'comparisons': comps,
            'projections': proj_results,
            'evaluation': {
                'tier': tier,
                'bust_probability': bust_prob,
                'breakout_probability': breakout_prob,
            },
            'ensemble_details': ensemble_details,
        }

    return projections


def main():
    print("=" * 70)
    print("BUILDING ENSEMBLE PROJECTIONS (Similarity + RF + GB)")
    print("=" * 70)
    print(f"Weights: Similarity={ENSEMBLE_WEIGHTS['similarity']:.0%}  "
          f"RF={ENSEMBLE_WEIGHTS['random_forest']:.0%}  "
          f"GB={ENSEMBLE_WEIGHTS['gradient_boosting']:.0%}")
    print(f"KNN: excluded (worst backtest performance)")
    print()

    projections = build_ensemble_projections()

    if not projections:
        print("No projections built.")
        return

    # Save
    output_path = os.path.join(PROCESSED_DIR, "prospect_projections.json")
    with open(output_path, 'w') as f:
        json.dump(projections, f, indent=2, default=str)
    print(f"\n💾 Saved: {output_path}")

    # Print rankings
    ranked = sorted(
        projections.values(),
        key=lambda p: (p['projections'].get('dynasty', {}).get('projected', 0) or 0),
        reverse=True
    )

    print(f"\n{'─' * 100}")
    print(f"{'Rank':<5} {'Name':<25} {'Pos':<4} {'Tier':<15} "
          f"{'Dynasty':>8} {'Sim':>7} {'RF':>7} {'GB':>7} "
          f"{'Rookie':>8} {'Peak':>8}")
    print(f"{'─' * 100}")

    for i, proj in enumerate(ranked, 1):
        p = proj['prospect']
        r = proj['projections']
        e = proj['evaluation']
        ed = proj.get('ensemble_details', {}).get('dynasty', {})

        dynasty_ppg = r.get('dynasty', {}).get('projected', 0)
        rookie_ppg = r.get('rookie', {}).get('projected', 0)
        peak_ppg = r.get('peak_3yr', {}).get('projected', 0)

        indiv = ed.get('individual', {})
        sim_val = indiv.get('similarity', 0)
        rf_val = indiv.get('random_forest', 0)
        gb_val = indiv.get('gradient_boosting', 0)

        print(f"{i:<5} {p.get('name', '?'):<25} {p.get('position', '?'):<4} "
              f"{e.get('tier', '?'):<15} "
              f"{dynasty_ppg:>8.1f} {sim_val:>7.1f} {rf_val:>7.1f} "
              f"{gb_val:>7.1f} "
              f"{rookie_ppg:>8.1f} {peak_ppg:>8.1f}")

    print(f"\n{'=' * 70}")
    print(f"✅ ENSEMBLE PROJECTIONS COMPLETE: {len(projections)} prospects")
    print(f"   Method: 35% Similarity + 35% Random Forest + 30% Gradient Boosting")
    print(f"   KNN excluded (backtest MAE: 3.60 vs RF 3.29)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()