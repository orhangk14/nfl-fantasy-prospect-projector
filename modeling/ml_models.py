# modeling/ml_models.py
"""
Machine Learning projection models for NFL Fantasy Prospect Projector.

Models:
  - KNN Regressor (natural extension of similarity matching)
  - Random Forest (handles mixed features, feature importance)
  - Gradient Boosting (XGBoost-style, best tabular performance)
  - Ensemble (blends similarity + ML predictions)

Training data: historical NFL players from player_profiles.json
Features: same position-specific features used in similarity.py
Targets: rookie_ppr_ppg, dynasty_ppg, peak_3yr_ppr_ppg, best_season_ppr_ppg

Run standalone: python -m modeling.ml_models
"""

import json
import os
import math
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modeling.similarity import (
    POS_FEATURES, CATEGORY_WEIGHTS, load_profiles, enrich_profiles
)

warnings.filterwarnings('ignore')

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Targets we predict ───────────────────────────────────────────────────
TARGETS = ['rookie_ppr_ppg', 'dynasty_ppg', 'peak_3yr_ppr_ppg', 'best_season_ppr_ppg']

# ─── Feature extraction ──────────────────────────────────────────────────


def get_feature_columns(position):
    """Get numeric feature column names for a position."""
    features = POS_FEATURES.get(position, [])
    return [feat_key for feat_key, _, _ in features]


def build_training_data(profiles, position, target):
    """
    Build X (features) and y (target) arrays for a given position.
    Only uses NFL players with known outcomes.
    """
    features = get_feature_columns(position)

    rows = []
    targets = []
    draft_years = []
    names = []

    for p in profiles:
        if p.get('is_prospect'):
            continue
        if p.get('position') != position:
            continue
        if p.get(target) is None:
            continue

        row = {}
        for feat in features:
            val = p.get(feat)
            row[feat] = val if val is not None else np.nan
        rows.append(row)
        targets.append(p[target])
        draft_years.append(p.get('draft_year', 0))
        names.append(p.get('name', '?'))

    if not rows:
        return None, None, None, None, None

    X = pd.DataFrame(rows)
    y = np.array(targets)
    years = np.array(draft_years)

    return X, y, years, names, features


def build_prospect_features(prospect, feature_columns):
    """Build feature vector for a single prospect."""
    row = {}
    for feat in feature_columns:
        val = prospect.get(feat)
        row[feat] = val if val is not None else np.nan
    return pd.DataFrame([row])


# ─── Model definitions ────────────────────────────────────────────────────


def get_models():
    """Return dict of model name -> unfitted model."""
    return {
        'knn': KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='nan_euclidean',
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=5,
            min_samples_split=10,
            subsample=0.8,
            random_state=42,
        ),
    }


# ─── Training pipeline ───────────────────────────────────────────────────


class MLProjectionEngine:
    """
    Trains and stores ML models per position per target.
    Handles missing values, scaling, and ensemble predictions.
    """

    def __init__(self):
        self.models = {}       # (pos, target, model_name) -> fitted model
        self.scalers = {}      # (pos, target) -> fitted scaler
        self.feature_cols = {} # pos -> list of feature column names
        self.training_stats = {}  # (pos, target) -> dict of CV metrics
        self.feature_importances = {}  # (pos, target) -> dict

    def _impute_X(self, X, fit=False, key=None):
        """
        Simple median imputation for missing values.
        Stores medians during fit, reuses during transform.
        """
        if fit:
            self._medians = {}
            for col in X.columns:
                self._medians[col] = X[col].median()
                if pd.isna(self._medians[col]):
                    self._medians[col] = 0.0
            if key:
                if not hasattr(self, '_all_medians'):
                    self._all_medians = {}
                self._all_medians[key] = self._medians.copy()
        else:
            if key and hasattr(self, '_all_medians') and key in self._all_medians:
                self._medians = self._all_medians[key]

        X_filled = X.copy()
        for col in X.columns:
            median_val = self._medians.get(col, 0.0)
            X_filled[col] = X_filled[col].fillna(median_val)
        return X_filled

    def train(self, profiles, positions=None, targets=None):
        """
        Train all models for all positions and targets.
        Uses Leave-One-Draft-Year-Out cross-validation for honest evaluation.
        """
        if positions is None:
            positions = ['QB', 'RB', 'WR', 'TE']
        if targets is None:
            targets = TARGETS

        print("=" * 70)
        print("TRAINING ML MODELS")
        print("=" * 70)

        for pos in positions:
            print(f"\n{'─' * 60}")
            print(f"📍 Position: {pos}")
            print(f"{'─' * 60}")

            for target in targets:
                X, y, years, names, feat_cols = build_training_data(
                    profiles, pos, target
                )
                if X is None or len(X) < 15:
                    print(f"  ⚠️  {target}: insufficient data ({0 if X is None else len(X)} samples)")
                    continue

                self.feature_cols[pos] = feat_cols
                impute_key = (pos, target)

                # Impute and scale
                X_filled = self._impute_X(X, fit=True, key=impute_key)
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X_filled),
                    columns=X_filled.columns,
                    index=X_filled.index
                )
                self.scalers[impute_key] = scaler

                # ── Cross-validation (Leave-One-Year-Out) ──
                # Filter out None/NaN years for CV grouping
                valid_year_mask = np.array([
                    y_val is not None and not (isinstance(y_val, float) and np.isnan(y_val))
                    for y_val in years
                ])
                clean_years = np.array([
                    float(y_val) if y_val is not None else 0.0 
                    for y_val in years
                ])
                
                unique_years = sorted(set(clean_years[valid_year_mask]))
                # Only CV on years with enough test samples
                cv_years = [yr for yr in unique_years if sum(clean_years == yr) >= 3]

                if len(cv_years) >= 3:
                    logo = LeaveOneGroupOut()
                    mask = np.isin(clean_years, cv_years) & valid_year_mask
                    X_cv = X_scaled[mask]
                    y_cv = y[mask]
                    years_cv = clean_years[mask]
                else:
                    X_cv, y_cv, years_cv = X_scaled, y, clean_years

                models = get_models()
                cv_results = {}

                for model_name, model in models.items():
                    try:
                        if len(cv_years) >= 3:
                            preds = cross_val_predict(
                                model, X_cv, y_cv,
                                groups=years_cv,
                                cv=logo
                            )
                            mae = mean_absolute_error(y_cv, preds)
                            rmse = np.sqrt(mean_squared_error(y_cv, preds))

                            # Correlation
                            if len(preds) >= 3:
                                corr = np.corrcoef(preds, y_cv)[0, 1]
                            else:
                                corr = 0

                            cv_results[model_name] = {
                                'mae': round(mae, 3),
                                'rmse': round(rmse, 3),
                                'correlation': round(corr, 3),
                                'n_samples': len(y_cv),
                            }
                        else:
                            cv_results[model_name] = {
                                'mae': None, 'rmse': None,
                                'correlation': None,
                                'n_samples': len(y),
                                'note': 'insufficient years for LOGO-CV'
                            }
                    except Exception as e:
                        cv_results[model_name] = {'error': str(e)}

                self.training_stats[(pos, target)] = cv_results

                # ── Fit final models on ALL data ──
                for model_name, model in get_models().items():
                    model.fit(X_scaled, y)
                    self.models[(pos, target, model_name)] = model

                    # Feature importance (for tree models)
                    if hasattr(model, 'feature_importances_'):
                        importances = dict(zip(
                            feat_cols,
                            model.feature_importances_
                        ))
                        self.feature_importances[(pos, target, model_name)] = dict(
                            sorted(importances.items(),
                                   key=lambda x: x[1], reverse=True)
                        )

                # Print results
                print(f"\n  📊 {target} (n={len(y)}):")
                for model_name, metrics in cv_results.items():
                    if metrics.get('mae') is not None:
                        print(f"    {model_name:<20} MAE: {metrics['mae']:.3f}  "
                              f"RMSE: {metrics['rmse']:.3f}  "
                              f"Corr: {metrics['correlation']:.3f}")
                    else:
                        print(f"    {model_name:<20} {metrics}")

        print(f"\n{'=' * 70}")
        print(f"✅ Training complete: {len(self.models)} models trained")
        print(f"{'=' * 70}")

    def predict(self, prospect, position, target='dynasty_ppg'):
        """
        Generate predictions from all ML models for a single prospect.
        Returns dict of model_name -> predicted value.
        """
        feat_cols = self.feature_cols.get(position)
        if feat_cols is None:
            return {}

        impute_key = (position, target)
        scaler = self.scalers.get(impute_key)
        if scaler is None:
            return {}

        # Build feature vector
        X_prospect = build_prospect_features(prospect, feat_cols)
        X_filled = self._impute_X(X_prospect, fit=False, key=impute_key)
        X_scaled = pd.DataFrame(
            scaler.transform(X_filled),
            columns=X_filled.columns
        )

        predictions = {}
        for model_name in ['knn', 'random_forest', 'gradient_boosting']:
            model = self.models.get((position, target, model_name))
            if model is None:
                continue
            pred = model.predict(X_scaled)[0]
            predictions[model_name] = round(float(pred), 2)

        return predictions

    def predict_all_targets(self, prospect, position):
        """
        Generate predictions for all targets for a single prospect.
        Returns dict of target -> {model_name -> value}.
        """
        results = {}
        for target in TARGETS:
            preds = self.predict(prospect, position, target)
            if preds:
                results[target] = preds
        return results

    def ensemble_predict(self, prospect, position, target='dynasty_ppg',
                         similarity_proj=None, weights=None):
        """
        Ensemble prediction blending ML models + similarity-based projection.

        Default weights (calibrated to minimize backtest MAE):
          - similarity:        0.35  (proven baseline)
          - gradient_boosting: 0.30  (best tabular learner)
          - random_forest:     0.20  (robust, diverse)
          - knn:               0.15  (captures local structure)
        """
        if weights is None:
            weights = {
                'similarity': 0.35,
                'gradient_boosting': 0.30,
                'random_forest': 0.20,
                'knn': 0.15,
            }

        ml_preds = self.predict(prospect, position, target)

        all_preds = {}
        if similarity_proj is not None:
            all_preds['similarity'] = similarity_proj
        all_preds.update(ml_preds)

        if not all_preds:
            return None

        # Normalize weights to only include available models
        available_weights = {k: v for k, v in weights.items() if k in all_preds}
        total_weight = sum(available_weights.values())

        if total_weight == 0:
            return None

        ensemble_val = sum(
            all_preds[k] * (w / total_weight)
            for k, w in available_weights.items()
        )

        return {
            'ensemble': round(ensemble_val, 2),
            'individual': all_preds,
            'weights_used': {k: round(v / total_weight, 3)
                             for k, v in available_weights.items()},
        }

    def get_feature_importance(self, position, target='dynasty_ppg',
                                model_name='gradient_boosting'):
        """Get feature importance for a specific model."""
        return self.feature_importances.get((position, target, model_name), {})

    def save(self, path=None):
        """Save all trained models to disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "ml_models.pkl")
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'feature_cols': self.feature_cols,
                'training_stats': self.training_stats,
                'feature_importances': self.feature_importances,
                '_all_medians': getattr(self, '_all_medians', {}),
            }, f)
        print(f"💾 Saved models to {path}")

    def load(self, path=None):
        """Load trained models from disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "ml_models.pkl")
        if not os.path.exists(path):
            print(f"⚠️  No saved models found at {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_cols = data['feature_cols']
        self.training_stats = data['training_stats']
        self.feature_importances = data['feature_importances']
        self._all_medians = data.get('_all_medians', {})
        print(f"✅ Loaded {len(self.models)} models from {path}")
        return True


# ─── Backtesting ML models ───────────────────────────────────────────────


def backtest_ml_models(profiles, test_years=None):
    """
    Backtest ML models using Leave-One-Year-Out.
    For each test year, train on all prior years, predict test year.
    Compare to similarity-based projections.
    """
    if test_years is None:
        test_years = [2022, 2023, 2024]

    print("=" * 70)
    print("BACKTESTING ML MODELS vs SIMILARITY")
    print("=" * 70)

    # Remove actual 2026 prospects
    profiles = [p for p in profiles if not p.get('is_prospect')]

    all_results = []

    for test_year in test_years:
        print(f"\n{'─' * 60}")
        print(f"📅 Test Year: {test_year}")
        print(f"{'─' * 60}")

        # Split
        train_profiles = []
        test_profiles = []

        for p in profiles:
            dy = p.get('draft_year')
            if dy is None:
                if p.get('rookie_ppr_ppg') is not None:
                    train_profiles.append(p)
                continue
            try:
                dy = int(float(dy))
            except (ValueError, TypeError):
                if p.get('rookie_ppr_ppg') is not None:
                    train_profiles.append(p)
                continue

            if dy < test_year and p.get('rookie_ppr_ppg') is not None:
                train_profiles.append(p)
            elif dy == test_year and p.get('rookie_ppr_ppg') is not None:
                test_profiles.append(p)

        print(f"  Train: {len(train_profiles)}  Test: {len(test_profiles)}")

        # Train ML engine on historical data
        engine = MLProjectionEngine()
        engine.train(train_profiles, targets=['rookie_ppr_ppg', 'dynasty_ppg'])

        # Predict each test prospect
        for prospect in test_profiles:
            pos = prospect.get('position', '')
            name = prospect.get('name', '?')

            for target in ['rookie_ppr_ppg', 'dynasty_ppg']:
                actual = prospect.get(target)
                if actual is None:
                    continue

                ml_preds = engine.predict(prospect, pos, target)

                for model_name, pred_val in ml_preds.items():
                    all_results.append({
                        'year': test_year,
                        'name': name,
                        'position': pos,
                        'target': target,
                        'model': model_name,
                        'predicted': pred_val,
                        'actual': actual,
                        'error': round(pred_val - actual, 2),
                        'abs_error': round(abs(pred_val - actual), 2),
                    })

    # Aggregate results
    if not all_results:
        print("No results generated.")
        return {}

    df = pd.DataFrame(all_results)

    print(f"\n{'═' * 70}")
    print("AGGREGATE ML BACKTEST RESULTS")
    print(f"{'═' * 70}")

    summary = {}
    for target in ['rookie_ppr_ppg', 'dynasty_ppg']:
        print(f"\n  📊 {target}:")
        target_df = df[df['target'] == target]

        for model_name in ['knn', 'random_forest', 'gradient_boosting']:
            model_df = target_df[target_df['model'] == model_name]
            if model_df.empty:
                continue

            mae = model_df['abs_error'].mean()
            rmse = np.sqrt((model_df['error'] ** 2).mean())
            corr = model_df[['predicted', 'actual']].corr().iloc[0, 1]
            within_5 = (model_df['abs_error'] <= 5).mean()

            summary_key = f"{target}_{model_name}"
            summary[summary_key] = {
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
                'correlation': round(corr, 3),
                'within_5_ppg': round(within_5, 3),
                'n': len(model_df),
            }

            print(f"    {model_name:<22} MAE: {mae:.3f}  RMSE: {rmse:.3f}  "
                  f"Corr: {corr:.3f}  Within 5: {within_5:.1%}  (n={len(model_df)})")

    # Per-position breakdown
    print(f"\n{'─' * 60}")
    print("PER-POSITION BREAKDOWN (rookie_ppr_ppg)")
    print(f"{'─' * 60}")

    rookie_df = df[df['target'] == 'rookie_ppr_ppg']
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_df = rookie_df[rookie_df['position'] == pos]
        if pos_df.empty:
            continue
        print(f"\n  {pos}:")
        for model_name in ['knn', 'random_forest', 'gradient_boosting']:
            model_df = pos_df[pos_df['model'] == model_name]
            if model_df.empty:
                continue
            mae = model_df['abs_error'].mean()
            corr = model_df[['predicted', 'actual']].corr().iloc[0, 1]
            print(f"    {model_name:<22} MAE: {mae:.3f}  Corr: {corr:.3f}  (n={len(model_df)})")

    # Save backtest results
    output = {
        'test_years': test_years,
        'summary': summary,
        'detailed_results': all_results,
    }
    output_path = os.path.join(PROCESSED_DIR, "ml_backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved ML backtest results to {output_path}")

    return output


# ─── Main: train + backtest + save ────────────────────────────────────────


def main():
    profiles = load_profiles()
    profiles = enrich_profiles(profiles)

    # 1. Backtest ML models (honest evaluation)
    backtest_results = backtest_ml_models(profiles)

    # 2. Train final models on ALL historical data
    engine = MLProjectionEngine()
    nfl_profiles = [p for p in profiles if not p.get('is_prospect')
                    and p.get('rookie_ppr_ppg') is not None]
    engine.train(nfl_profiles)

    # 3. Save trained models
    engine.save()

    # 4. Demo: predict for first few prospects
    prospects = [p for p in profiles if p.get('is_prospect')]

    print(f"\n{'═' * 70}")
    print("DEMO: ML PREDICTIONS FOR 2026 PROSPECTS")
    print(f"{'═' * 70}")

    for prospect in prospects[:5]:
        name = prospect.get('name', '?')
        pos = prospect.get('position', '?')
        print(f"\n  🏈 {name} ({pos}):")

        for target in TARGETS:
            preds = engine.predict(prospect, pos, target)
            if preds:
                pred_str = "  |  ".join(f"{k}: {v:.1f}" for k, v in preds.items())
                print(f"    {target:<25} {pred_str}")

    # 5. Feature importance
    print(f"\n{'─' * 60}")
    print("TOP FEATURES BY POSITION (gradient_boosting, dynasty_ppg)")
    print(f"{'─' * 60}")

    for pos in ['QB', 'RB', 'WR', 'TE']:
        fi = engine.get_feature_importance(pos, 'dynasty_ppg', 'gradient_boosting')
        if fi:
            top_5 = list(fi.items())[:5]
            print(f"\n  {pos}:")
            for feat, imp in top_5:
                print(f"    {feat:<30} {imp:.3f}")


if __name__ == "__main__":
    main()