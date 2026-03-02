# NFL Fantasy Prospect Projector

Predicts NFL fantasy football outcomes for incoming prospects using historical player similarity matching, machine learning models, college production profiles, combine measurables, and draft capital.

## Overview

The system works in 6 stages:

1. **Data Collection** - Scrape/parse college stats, NFL stats, combine data, and mock drafts
2. **Feature Engineering** - Build unified player profiles from raw data
3. **Similarity Matching** - Find the most similar historical NFL players to each prospect
4. **Projections** - Generate fantasy projections weighted by similarity scores
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

### Step 4: Projections

    python3 -m modeling.projections

Output: data/processed/prospect_projections.json

### Step 5 (Optional): Tune and Backtest

    python3 -m modeling.tune
    python3 -m modeling.backtest

### Step 6: Launch App

    streamlit run app.py

## Quick Start (if raw data already exists)

    python3 -m feature_engineering.build_features
    python3 -m modeling.similarity
    python3 -m modeling.projections
    streamlit run app.py

## Model Performance

### Method Comparison (Backtest 2022-2024, n=198)

    Method                      Rookie MAE  Rookie Corr  Dynasty MAE  Dynasty Corr
    Similarity v1 (original)    3.61        0.533        3.50         0.574
    Similarity v3 (enhanced)    3.48        0.574        3.31         0.602
    ML Random Forest            3.29        0.607        3.16         0.614
    ML Gradient Boosting        3.50        0.540        3.20         0.621

### Improvements from Original Baseline

    Metric          Original        Best Now        Improvement
    Rookie MAE      3.61            3.29 (RF)       -8.9%
    Rookie Corr     0.533           0.607 (RF)      +13.9%
    Dynasty MAE     3.50            3.16 (RF)       -9.7%
    Dynasty Corr    0.574           0.621 (GB)      +8.2%
    Within 5 PPG    75.8%           81.8% (RF)      +6.0%

### Per-Position Rookie Accuracy (Similarity v3)

    Position    MAE     Corr    Within 5 PPG
    QB          4.80    0.453   56%
    RB          3.80    0.501   81%
    WR          3.53    0.445   81%
    TE          2.21    0.516   93%

### Notable Correct Predictions

- Drake Maye: Projected 14.7, Actual 14.2 (Error: +0.5)
- Jaxon Smith-Njigba: Projected 9.0, Actual 8.8 (Error: +0.2)
- Wan Dale Robinson: Projected 8.0, Actual 8.6 (Error: -0.6)

### Known Limitations (Honest Assessment)

The model still cannot predict:

- Injuries (Jameson Williams, Jonathon Brooks)
- Generational outliers (Puka Nacua, Brock Bowers)
- Unexpected opportunity/ Small sample (Joe Milton III getting a starting job)
- Busted situations (Malachi Corley buried on depth chart)

These are fundamentally unpredictable from college data alone — and that is okay. The model is honest about uncertainty through confidence intervals and bust probabilities.

## Similarity Weights

Optimized via grid search backtesting:

    Category        Weight
    Draft Capital   40%
    Peak Season     20%
    Production      15%
    Measurables     15%
    Efficiency      10%

## Draft Capital Adjustment

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

    app.py                          <- Slim router
    tabs/
    ├── __init__.py
    ├── helpers.py                  <- Shared rendering functions
    ├── big_board.py                <- Tab 1: Big Board
    ├── deep_dive.py                <- Tab 2: Deep Dive
    ├── position_rankings.py        <- Tab 3: Position Rankings
    ├── compare.py                  <- Tab 4: Head-to-Head Compare
    ├── custom_prospect.py          <- Tab 5: Build Custom Prospect
    └── historical_class.py         <- Tab 6: Historical Draft Classes
    data_collection/
    ├── scrape_all.py
    ├── scrape_2026_prospects.py
    ├── parse_2026_combine.py
    └── parse_mock_draft.py
    feature_engineering/
    └── build_features.py
    modeling/
    ├── similarity.py
    ├── projections.py
    ├── tune.py
    └── backtest.py
    data/
    ├── raw/                        <- Scraped source data
    └── processed/                  <- Built profiles and projections

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
