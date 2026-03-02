# NFL Fantasy Prospect Projector

Predicts NFL fantasy football outcomes for incoming prospects using historical player similarity matching, college production profiles, combine measurables, and draft capital.

## Overview

The system works in 5 stages:

1. Data Collection - Scrape/parse college stats, NFL stats, combine data, and mock drafts
2. Feature Engineering - Build unified player profiles from raw data
3. Similarity Matching - Find the most similar historical NFL players to each prospect
4. Projections - Generate fantasy projections weighted by similarity scores
5. Backtesting - Validate accuracy against known NFL outcomes

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

## Model Performance (Backtest 2022-2024, n=198)

### Aggregate Accuracy

    Metric          Rookie PPG    Dynasty PPG
    MAE             3.61          3.50
    RMSE            4.53          4.34
    Correlation     0.533         0.574
    Within 5 PPG    75.8%         80.3%
    Bias            +0.60         +1.15

### Per-Position Rookie Accuracy

    Position    MAE     Correlation    Within 5 PPG
    QB          5.04    0.380          56%
    RB          4.16    0.450          65%
    WR          3.61    0.446          78%
    TE          2.12    0.545          95%

### Notable Correct Predictions
- Drake Maye: Projected 14.7, Actual 14.2 (Error: +0.5)
- Jaxon Smith-Njigba: Projected 9.0, Actual 8.8 (Error: +0.2)
- Wan Dale Robinson: Projected 8.0, Actual 8.6 (Error: -0.6)

### Known Limitations
- Under-projects generational breakouts (Puka Nacua, Brock Bowers)
- Over-projects injured players (Jameson Williams, Jonathon Brooks)
- QB has highest error due to small sample + high variance

## Similarity Weights

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

- Big Board: Sortable/filterable prospect rankings
- Deep Dive: Individual prospect analysis with comps
- Position Rankings: Per-position breakdowns with insights
- Compare: Head-to-head prospect comparison
- Build Custom Prospect: Input any college stats + measurables to generate projections

## Scoring (PPR)

- Passing: 0.04 pts/yard, 4 pts/TD, -2 pts/INT
- Rushing: 0.1 pts/yard, 6 pts/TD
- Receiving: 1 pt/reception, 0.1 pts/yard, 6 pts/TD

## Key Concepts

- Dynasty PPG: Average fantasy PPG over first 5 NFL seasons
- Peak 3-Year: Best 3-year rolling average PPG
- Breakout Ratio: Peak PPG / career avg PPG (>1.3 = one dominant year)
- Archetype: Player style (X_OUTSIDE, SLOT, WORKHORSE, DUAL_THREAT, etc.)
