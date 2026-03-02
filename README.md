# NFL Fantasy Prospect Projector

Predicts NFL fantasy football outcomes for incoming prospects using historical player similarity matching, machine learning models, college production profiles, combine measurables, and draft capital.

## Overview

The system works in 6 stages:

1. **Data Collection** - Scrape/parse college stats, NFL stats, combine data, and mock drafts
2. **Feature Engineering** - Build unified player profiles from raw data
3. **Similarity Matching** - Find the most similar historical NFL players to each prospect
4. **Projections** - Generate fantasy projections (3 methods available)
5. **Backtesting** - Validate accuracy against known NFL outcomes
6. **Dashboard** - Interactive Streamlit app with 7 analysis tabs

## Setup

    git clone <repo-url>
    cd nfl-fantasy-prospect-projector
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Execution Order

Run these in order. Each step depends on the previous one.

### Step 1: Data Collection

    python3 -m data_collection.scrape_all
    python3 -m data_collection.scrape_2026_prospects
    python3 -m data_collection.parse_2026_combine
    python3 -m data_collection.parse_mock_draft

Raw data is saved to data/raw/. Only re-run scrapers to refresh data.

### Step 2: Build Features

    python3 -m feature_engineering.build_features

Output: data/processed/player_profiles.json and .csv

### Step 3: Similarity Matching

    python3 -m modeling.similarity

Output: data/processed/prospect_comparisons.json

### Step 4: Generate Projections

This is where you choose your projection method. Three options are available, from simplest to most accurate.

#### Option A: Similarity Only (baseline)

Uses weighted similarity matching with draft capital adjustment. Simplest method, no ML dependencies beyond base install.

    python3 -m modeling.projections

#### Option B: ML Models (train + evaluate)

Trains Random Forest, Gradient Boosting, and KNN models. Runs backtesting to evaluate accuracy. Saves trained models to models/ml_models.pkl. Does NOT generate final projections by itself — use Option C to use these models.

    python3 -m modeling.ml_models

#### Option C: Ensemble — RECOMMENDED

Blends similarity-based projections with ML model predictions using optimized weights (30% Similarity, 65% Random Forest, 5% Gradient Boosting). KNN is excluded — it was the worst performer on every metric during backtesting.

Requires Step 3 (similarity) and Option B (ml_models) to be run first.

    python3 -m modeling.similarity
    python3 -m modeling.ml_models
    python3 -m modeling.ensemble

All three options write to the same file (data/processed/prospect_projections.json), so whichever you run last is what the Streamlit app displays.

### Step 5 (Optional): Tune and Backtest

Similarity weight tuning:

    python3 -m modeling.tune

Similarity-only backtesting:

    python3 -m modeling.backtest

Ensemble grid search optimization (recommended — finds the optimal blend of similarity + RF + GB):

    python3 -m modeling.backtest_ensemble

The ensemble optimizer precomputes all similarity comps and ML predictions, then grid searches ~3,888 configurations across:
- Ensemble weights (similarity vs RF vs GB in 0.05 increments, summing to 1.0)
- Similarity exponent (2 vs 3)
- Similarity comp floor threshold (0.50, 0.65, 0.85)
- Draft capital adjustment blend (0.0, 0.35, 0.55, 0.75)

Precompute takes ~40s, grid search takes ~10s. It evaluates each config using leave-one-year-out across 2022-2024, compares against baselines, runs sensitivity analysis, and prints exactly what to update in ensemble.py.

### Step 6: Launch App

    streamlit run app.py

## Quick Start — Recommended (if raw data already exists)

This runs the full pipeline using the best available method:

    python3 -m feature_engineering.build_features
    python3 -m modeling.similarity
    python3 -m modeling.ml_models
    python3 -m modeling.ensemble
    streamlit run app.py

## Quick Start — Lightweight (no ML, similarity only)

If you want to skip ML model training (faster, no scikit-learn required for projections):

    python3 -m feature_engineering.build_features
    python3 -m modeling.similarity
    python3 -m modeling.projections
    streamlit run app.py

## Quick Start — Full Optimization

Run everything including the ensemble grid search to find the optimal weights:

    python3 -m feature_engineering.build_features
    python3 -m modeling.similarity
    python3 -m modeling.ml_models
    python3 -m modeling.backtest_ensemble
    # Update ensemble.py with recommended weights from output
    python3 -m modeling.ensemble
    streamlit run app.py

## Model Performance

### Ensemble Grid Search Results (3,888 configs tested)

The optimal configuration was found via exhaustive grid search with leave-one-year-out backtesting across the 2022, 2023, and 2024 draft classes (n=198 players).

#### Optimal Ensemble Configuration

    Parameter               Value
    Similarity weight       30%
    Random Forest weight    65%
    Gradient Boosting wt    5%
    KNN weight              excluded
    Similarity exponent     2
    Similarity comp floor   50% of best comp
    Draft capital blend     0.75

#### Optimal Ensemble Accuracy (Backtest 2022-2024, n=198)

    Metric          Rookie PPG      Dynasty PPG
    MAE             3.259           3.120
    RMSE            4.142           3.942
    Correlation     0.615           0.633
    Within 5 PPG    81.8%           82.3%
    Bias            +0.109          +0.556

#### Per-Position Breakdown (Optimal Ensemble)

    Position    MAE     Corr    Within 5    Bias     n
    QB          4.901   0.444   60.0%       -1.039   25
    RB          3.522   0.514   78.8%       +0.539   52
    WR          3.149   0.559   83.3%       +0.507   78
    TE          2.187   0.533   95.3%       -0.466   43

#### All Methods Compared

    Method                          Rookie MAE  Rookie Corr  Dynasty MAE  Dynasty Corr  Score
    Ensemble (30/65/5) OPTIMAL      3.259       0.615        3.120        0.633         1.808
    Random Forest Only              3.286       0.607        3.157        0.614         1.746
    Previous Ensemble (35/35/30)    3.330       0.589        3.128        0.636         1.707
    Gradient Boosting Only          3.502       0.540        3.201        0.621         1.462
    Similarity Only                 3.584       0.556        3.424        0.605         1.405
    KNN Only                        3.602       0.510        3.344        0.540         excluded

#### Improvements from Original Similarity Baseline

    Metric          Original (Sim v1)   Optimal Ensemble    Improvement
    Rookie MAE      3.61                3.259               -9.7%
    Rookie Corr     0.533               0.615               +15.4%
    Dynasty MAE     3.50                3.120               -10.9%
    Dynasty Corr    0.574               0.633               +10.3%
    Within 5 PPG    75.8%               82.3%               +6.5%

#### Sensitivity Analysis

The optimum is stable. The top 5 configurations scored within 0.0023 of each other, and the top 20 within 0.0106. The top 10 configs all clustered around:
- Similarity: 25-30%
- Random Forest: 55-70%
- Gradient Boosting: 5-15%

This confirms RF is the strongest single predictor, similarity adds value as a stabilizer, and GB contributes marginally. The optimization surface is smooth, not noisy.

#### Top 5 Configurations

    Rank  Sim    RF    GB   Exp  Floor  DAdj   R_MAE   R_Corr   D_MAE   D_Corr  Score
    1     30%   65%    5%   2    50%    0.75   3.259   0.615    3.120   0.633   1.808
    2     30%   60%   10%   2    50%    0.75   3.264   0.613    3.116   0.635   1.807
    3     30%   65%    5%   2    65%    0.75   3.260   0.613    3.125   0.632   1.806
    4     25%   65%   10%   2    50%    0.75   3.260   0.613    3.115   0.634   1.806
    5     25%   70%    5%   2    50%    0.75   3.256   0.614    3.121   0.632   1.805

### Notable Correct Predictions

- Drake Maye: Projected 14.7, Actual 14.2 (Error: +0.5)
- Jaxon Smith-Njigba: Projected 9.0, Actual 8.8 (Error: +0.2)
- Wan Dale Robinson: Projected 8.0, Actual 8.6 (Error: -0.6)

### Known Limitations (Honest Assessment)

The model still cannot predict:

- Injuries (Jameson Williams, Jonathon Brooks)
- Generational outliers (Puka Nacua, Brock Bowers)
- Unexpected opportunity (Joe Milton III getting a starting job)
- Busted situations (Malachi Corley buried on depth chart)

These are fundamentally unpredictable from college data alone. The model is honest about uncertainty through confidence intervals and bust probabilities.

## Similarity Weights

Optimized via grid search backtesting:

    Category        Weight
    Draft Capital   40%
    Peak Season     25%
    Production      10%
    Measurables     15%
    Efficiency      10%

## Draft Capital Adjustment

Applied as a residual adjustment on top of similarity matching (which already weights draft capital at 40%). Optimal blend factor is 0.75 (determined by ensemble grid search).

    Pick Range    Raw Mult    Effective (0.75 blend)
    1-10          1.35x       1.26x
    11-20         1.25x       1.19x
    21-32         1.18x       1.14x
    33-48         1.10x       1.08x
    49-64         1.05x       1.04x
    65-100        1.00x       1.00x
    101-140       0.92x       0.94x
    141-180       0.85x       0.89x
    181-224       0.78x       0.84x
    UDFA          0.70x       0.78x

## App Features

- **Big Board** - Sortable/filterable prospect rankings with tier and risk indicators
- **Deep Dive** - Individual prospect analysis with historical comps and feature comparison
- **Position Rankings** - Per-position breakdowns with top value, ceiling, and safety insights
- **Compare** - Head-to-head prospect comparison with shared comp analysis
- **Build Custom Prospect** - Input any college stats + measurables to generate projections
- **Historical Classes** - Browse past draft classes, see hits/busts, round-by-round breakdowns, archetype performance, and cross-class rankings
- **Find My Player** - Pick any NFL player from the historical database and find which 2026 prospects most closely match their pre-NFL profile using live similarity computation

## Project Structure

    app.py                          <- Slim router, delegates to tab modules
    tabs/
    ├── __init__.py
    ├── helpers.py                  <- Shared rendering functions
    ├── big_board.py                <- Tab 1: Big Board
    ├── deep_dive.py                <- Tab 2: Deep Dive
    ├── position_rankings.py        <- Tab 3: Position Rankings
    ├── compare.py                  <- Tab 4: Head-to-Head Compare
    ├── custom_prospect.py          <- Tab 5: Build Custom Prospect
    ├── historical_class.py         <- Tab 6: Historical Draft Classes
    └── find_my_player.py           <- Tab 7: Find My Player
    modeling/
    ├── similarity.py               <- Similarity matching engine (v3)
    ├── projections.py              <- Similarity-only projections (Option A)
    ├── ml_models.py                <- ML model training + evaluation (Option B)
    ├── ensemble.py                 <- Ensemble projections (Option C, recommended)
    ├── backtest.py                 <- Similarity-only backtesting
    ├── backtest_ensemble.py        <- Ensemble grid search optimizer (~3,888 configs)
    └── tune.py                     <- Similarity weight optimization
    models/
    └── ml_models.pkl               <- Trained ML models (generated by ml_models.py)
    data_collection/
    ├── scrape_all.py
    ├── scrape_2026_prospects.py
    ├── parse_2026_combine.py
    └── parse_mock_draft.py
    feature_engineering/
    └── build_features.py
    data/
    ├── raw/                        <- Scraped source data
    └── processed/                  <- Built profiles and projections

## Pipeline Architecture

    [Raw Data] → build_features → [player_profiles.json]
                                        │
                                        ├── similarity.py → [prospect_comparisons.json]
                                        │       │
                                        │       ├── projections.py → [prospect_projections.json] (Option A)
                                        │       │
                                        │       └── ensemble.py → [prospect_projections.json] (Option C)
                                        │               ▲
                                        │               │
                                        └── ml_models.py → [models/ml_models.pkl] (Option B)

    [prospect_projections.json] → streamlit app (reads whichever method wrote last)

    Optional optimization:
    backtest_ensemble.py → tests 3,888 configs → recommends optimal weights for ensemble.py

## Scoring System (PPR)

- Passing: 0.04 pts/yard, 4 pts/TD, -2 pts/INT
- Rushing: 0.1 pts/yard, 6 pts/TD
- Receiving: 1 pt/reception, 0.1 pts/yard, 6 pts/TD

## Key Concepts

- **Dynasty PPG** - Average fantasy PPG over first 5 NFL seasons
- **Peak 3-Year** - Best 3-year rolling average PPG
- **Ceiling PPG** - Single best season PPG projection
- **Breakout Ratio** - Peak PPG / career avg PPG (>1.3 = one dominant spike year)
- **Archetype** - Player style classification (X_OUTSIDE, SLOT, WORKHORSE, DUAL_THREAT, POCKET_PASSER, etc.)
- **Bust Probability** - % of top-10 historical comps whose dynasty PPG fell below positional replacement level (QB <8, RB <5, WR <5, TE <4)
- **Breakout Probability** - % of top-10 historical comps who produced at least one elite season (QB ≥20, RB ≥14, WR ≥14, TE ≥12 PPG)
- **Ensemble Projection** - Blended prediction: 30% similarity matching + 65% Random Forest + 5% Gradient Boosting. KNN excluded. Weights optimized via grid search over 3,888 configurations.
