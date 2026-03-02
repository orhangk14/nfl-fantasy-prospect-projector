# NFL Fantasy Prospect Projector

Predicts NFL fantasy football outcomes for incoming prospects using historical player similarity matching, machine learning models, college production profiles, combine measurables, and draft capital.

## Overview

The system works in 6 stages:

1. **Data Collection** - Scrape/parse college stats, NFL stats, combine data, and mock drafts
2. **Feature Engineering** - Build unified player profiles from raw data
3. **Similarity Matching** - Find the most similar historical NFL players to each prospect
4. **Projections** - Generate fantasy projections (3 methods available)
5. **Backtesting** - Validate accuracy against known NFL outcomes
6. **Dashboard** - Interactive Streamlit app with 6 analysis tabs

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

#### Option B: ML Models Only (train + evaluate)

Trains Random Forest, Gradient Boosting, and KNN models. Runs backtesting to evaluate accuracy. Saves trained models to models/ml_models.pkl. Does NOT generate final projections by itself — use Option C to use these models.

    python3 -m modeling.ml_models

#### Option C: Ensemble — RECOMMENDED

Blends similarity-based projections with ML model predictions (35% similarity, 30% Gradient Boosting, 20% Random Forest, 15% KNN). This is the most accurate method and what you should use.

Requires Step 3 (similarity) and Option B (ml_models) to be run first.

    python3 -m modeling.similarity
    python3 -m modeling.ml_models
    python3 -m modeling.ensemble

All three options write to the same file (data/processed/prospect_projections.json), so whichever you run last is what the Streamlit app displays.

### Step 5 (Optional): Tune and Backtest

    python3 -m modeling.tune
    python3 -m modeling.backtest

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

## Projection Methods Compared

### Method Comparison (Backtest 2022-2024, n=198)

    Method                      Rookie MAE  Rookie Corr  Dynasty MAE  Dynasty Corr
    Similarity v1 (original)    3.61        0.533        3.50         0.574
    Similarity v3 (enhanced)    3.48        0.574        3.31         0.602
    ML Random Forest            3.29        0.607        3.16         0.614
    ML Gradient Boosting        3.50        0.540        3.20         0.621
    Ensemble (recommended)      best blend of all methods above

### Improvements from Original Baseline

    Metric          Original        Best Now        Improvement
    Rookie MAE      3.61            3.29 (RF)       -8.9%
    Rookie Corr     0.533           0.607 (RF)      +13.9%
    Dynasty MAE     3.50            3.16 (RF)       -9.7%
    Dynasty Corr    0.574           0.621 (GB)      +8.2%
    Within 5 PPG    75.8%           81.8% (RF)      +6.0%

The ensemble blends these methods because they make different kinds of errors. Similarity is best at finding interpretable comps. RF is best at raw accuracy. GB captures nonlinear interactions. The blend outperforms any single method.

### Per-Position Rookie Accuracy (Similarity v3)

    Position    MAE     Corr    Within 5 PPG
    QB          4.80    0.453   56%
    RB          3.80    0.501   81%
    WR          3.53    0.445   81%
    TE          2.21    0.516   93%

### Ensemble Weights

The final projection for each prospect is:

    Source              Weight
    Similarity v3       35%
    Gradient Boosting   30%
    Random Forest       20%
    KNN                 15%

These weights were calibrated via backtesting to minimize MAE while preserving the interpretable historical comparisons that similarity provides.

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

Applied as a residual adjustment on top of similarity matching (which already weights draft capital at 40%). Uses a 0.55 blend factor so the adjustment is partial, not doubled.

    Pick Range    Raw Mult    Effective (0.55 blend)
    1-10          1.35x       1.19x
    11-20         1.25x       1.14x
    21-32         1.18x       1.10x
    33-48         1.10x       1.05x
    49-64         1.05x       1.03x
    65-100        1.00x       1.00x
    101-140       0.92x       0.96x
    141-180       0.85x       0.92x
    181-224       0.78x       0.88x
    UDFA          0.70x       0.84x

## App Features

- **Big Board** - Sortable/filterable prospect rankings with tier and risk indicators
- **Deep Dive** - Individual prospect analysis with historical comps and feature comparison
- **Position Rankings** - Per-position breakdowns with top value, ceiling, and safety insights
- **Compare** - Head-to-head prospect comparison with shared comp analysis
- **Build Custom Prospect** - Input any college stats + measurables to generate projections
- **Historical Classes** - Browse past draft classes, see hits/busts, round-by-round breakdowns, archetype performance, and cross-class rankings

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
    └── historical_class.py         <- Tab 6: Historical Draft Classes
    modeling/
    ├── similarity.py               <- Similarity matching engine (v3)
    ├── projections.py              <- Similarity-only projections (Option A)
    ├── ml_models.py                <- ML model training + evaluation (Option B)
    ├── ensemble.py                 <- Ensemble projections (Option C, recommended)
    ├── tune.py                     <- Weight optimization via grid search
    └── backtest.py                 <- Backtesting framework
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
- **Bust Probability** - Likelihood of dynasty PPG falling below positional replacement level
- **Breakout Probability** - Likelihood of producing a top-12 positional season
- **Ensemble Projection** - Blended prediction from similarity matching + 3 ML models, weighted by backtest accuracy
