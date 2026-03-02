# app.py
"""
NFL Fantasy Prospect Projector — Streamlit Dashboard (Modular Router)

Run: streamlit run app.py
"""

import streamlit as st
import json
import os
import pandas as pd

PROCESSED_DIR = "data/processed"

# ─── Page Config & Global Styles ────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Fantasy Prospect Projector",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800; color: #1a1a2e;
        text-align: center; padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem; color: #666;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin: 0.5rem 0;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 2rem; }
    .archetype-badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; color: white; margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_projections():
    path = os.path.join(PROCESSED_DIR, "prospect_projections.json")
    if not os.path.exists(path):
        st.error(
            "Projections not found. Run the pipeline first:\n\n"
            "```\n"
            "python -m feature_engineering.build_features\n"
            "python -m modeling.similarity\n"
            "python -m modeling.projections\n"
            "```"
        )
        st.stop()
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_profiles():
    path = os.path.join(PROCESSED_DIR, "player_profiles.json")
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def build_board_df(projections):
    rows = []
    for pid, data in projections.items():
        p = data['prospect']
        r = data['projections']
        e = data['evaluation']

        row = {
            'espn_id': pid,
            'Name': p.get('name', ''),
            'Pos': p.get('position', ''),
            'School': p.get('school', ''),
            'Archetype': p.get('archetype', ''),
            'Height': p.get('combine_height'),
            'Weight': p.get('combine_weight'),
            '40-Yard': p.get('combine_40'),
            'College PPG': p.get('college_ppr_ppg'),
            'Peak College PPG': p.get('peak_ppr_ppg'),
            'Board Rank': p.get('big_board_rank'),
        }

        for proj_key, label in [('rookie', 'Rookie PPG'), ('dynasty', 'Dynasty PPG'),
                                 ('peak_3yr', 'Peak 3yr PPG'), ('best_season', 'Ceiling PPG')]:
            proj = r.get(proj_key)
            if proj and isinstance(proj, dict):
                row[label] = proj.get('projected')
                row[f'{label} Floor'] = proj.get('floor')
                row[f'{label} Ceil'] = proj.get('ceiling')
                row[f'{label} Conf'] = proj.get('confidence')
            else:
                row[label] = None

        career = r.get('career_length')
        row['Est Career Yrs'] = career.get('projected') if career and isinstance(career, dict) else None
        row['Tier'] = e.get('tier', '')
        row['Bust %'] = e.get('bust_probability')
        row['Breakout %'] = e.get('breakout_probability')
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar(df):
    with st.sidebar:
        st.title("🏈 Prospect Projector")
        st.caption("2026 NFL Draft — Fantasy Football")
        st.divider()

        st.subheader("🔍 Filters")
        pos_filter = st.multiselect(
            "Position", ['QB', 'RB', 'WR', 'TE'],
            default=['QB', 'RB', 'WR', 'TE'],
        )
        arch_options = sorted(df['Archetype'].dropna().unique().tolist())
        arch_filter = st.multiselect("Archetype", arch_options, default=arch_options)
        sort_by = st.selectbox(
            "Sort By",
            ['Dynasty PPG', 'Rookie PPG', 'Peak 3yr PPG', 'Ceiling PPG', 'Board Rank'],
            index=0,
        )

        st.divider()
        st.markdown("""
        **Model Weights:**
        - Draft Capital: 40%
        - Peak Season: 20%
        - Production: 15%
        - Measurables: 15%
        - Efficiency: 10%
        """)
        st.caption(f"Prospects: {len(df)} | Historical pool: ~548 NFL players")

    return pos_filter, arch_filter, sort_by


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    projections = load_projections()
    df = build_board_df(projections)

    pos_filter, arch_filter, sort_by = render_sidebar(df)

    # Apply filters
    filtered = df[
        df['Pos'].isin(pos_filter) & df['Archetype'].isin(arch_filter)
    ].copy()
    asc = sort_by == 'Board Rank'
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=asc, na_position='last')

    # Header
    st.markdown(
        '<div class="main-header">🏈 2026 NFL Fantasy Prospect Projector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Historical comparisons • Archetype analysis • '
        'Rookie & dynasty projections</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab_board, tab_detail, tab_pos, tab_compare, tab_custom, tab_hist = st.tabs([
        "📋 Big Board",
        "🔬 Deep Dive",
        "📊 Position Rankings",
        "⚖️ Compare",
        "🛠️ Build Custom Prospect",
        "📜 Historical Classes",
    ])

    # ── Tab 1 ───────────────────────────────────────────────────────
    with tab_board:
        from tabs.big_board import render as render_big_board
        render_big_board(filtered)

    # ── Tab 2 ───────────────────────────────────────────────────────
    with tab_detail:
        from tabs.deep_dive import render as render_deep_dive
        render_deep_dive(filtered, projections)

    # ── Tab 3 ───────────────────────────────────────────────────────
    with tab_pos:
        from tabs.position_rankings import render as render_pos_rankings
        render_pos_rankings(filtered, pos_filter)

    # ── Tab 4 ───────────────────────────────────────────────────────
    with tab_compare:
        from tabs.compare import render as render_compare
        render_compare(filtered, projections)

    # ── Tab 5 ───────────────────────────────────────────────────────
    with tab_custom:
        from tabs.custom_prospect import render as render_custom
        render_custom(load_profiles)

    # ── Tab 6 ───────────────────────────────────────────────────────
    with tab_hist:
        from tabs.historical_class import render as render_hist
        profiles = load_profiles()
        render_hist(profiles)


if __name__ == "__main__":
    main()