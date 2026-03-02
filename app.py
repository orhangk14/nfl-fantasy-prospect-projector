# app.py
"""
NFL Fantasy Prospect Projector — Streamlit Dashboard

Run: streamlit run app.py
"""

import streamlit as st
import json
import os
import math
import pandas as pd
from collections import defaultdict

PROCESSED_DIR = "data/processed"

st.set_page_config(
    page_title="NFL Fantasy Prospect Projector",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 2rem; }
    .archetype-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

ARCHETYPE_COLORS = {
    'X_OUTSIDE': '#e74c3c', 'SLOT': '#3498db', 'DEEP_THREAT': '#9b59b6',
    'YAC_GADGET': '#e67e22', 'WORKHORSE': '#2c3e50', 'SPEED_BACK': '#1abc9c',
    'RECEIVING_BACK': '#3498db', 'POWER_BACK': '#c0392b', 'POCKET_PASSER': '#2980b9',
    'DUAL_THREAT': '#8e44ad', 'SCRAMBLER': '#27ae60', 'RECEIVING_TE': '#2ecc71',
    'ATHLETIC_TE': '#f39c12', 'BLOCKING_TE': '#7f8c8d',
}


def get_tier_color(tier):
    if not tier or pd.isna(tier):
        return '#999'
    tier = str(tier)
    if 'Elite' in tier:
        return '#FFD700'
    if '1' in tier:
        return '#00C851'
    if '2' in tier:
        return '#33b5e5'
    if '3' in tier:
        return '#ff8800'
    return '#ff4444'


def fmt(val, decimals=1):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '—'
    return f"{val:.{decimals}f}"


def fmt_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '—'
    return f"{val:.0%}"


# ─── Data Loading ───────────────────────────────────────────────────────────
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


# ─── Rendering Helpers ──────────────────────────────────────────────────────
def render_projection_cards(projs):
    cols = st.columns(4)
    items = [
        ('Rookie PPG', projs.get('rookie')),
        ('Dynasty PPG', projs.get('dynasty')),
        ('Peak 3yr PPG', projs.get('peak_3yr')),
        ('Ceiling PPG', projs.get('best_season')),
    ]
    for col, (label, proj) in zip(cols, items):
        with col:
            if proj and isinstance(proj, dict) and proj.get('projected') is not None:
                val = proj['projected']
                floor = proj.get('floor', 0)
                ceiling = proj.get('ceiling', 0)
                conf = proj.get('confidence', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{label}</h3>
                    <h1>{val:.1f}</h1>
                    <small>Range: {floor:.1f} – {ceiling:.1f}</small><br>
                    <small>Confidence: {conf:.0%}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card" style="background:#999">
                    <h3>{label}</h3><h1>—</h1>
                    <small>Insufficient data</small>
                </div>
                """, unsafe_allow_html=True)


def render_eval_row(evaluation, projs):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tier = evaluation.get('tier', '—')
        color = get_tier_color(tier)
        st.markdown(f"**Dynasty Tier:** <span style='color:{color};font-weight:bold;"
                    f"font-size:1.2rem'>{tier}</span>", unsafe_allow_html=True)
    with c2:
        bust = evaluation.get('bust_probability')
        if bust is not None:
            bc = '#ff4444' if bust >= 0.5 else '#ff8800' if bust >= 0.3 else '#00C851'
            st.markdown(f"**Bust Risk:** <span style='color:{bc};font-weight:bold'>"
                        f"{bust:.0%}</span>", unsafe_allow_html=True)
    with c3:
        bo = evaluation.get('breakout_probability')
        if bo is not None:
            boc = '#FFD700' if bo >= 0.4 else '#00C851' if bo >= 0.2 else '#666'
            st.markdown(f"**Breakout Chance:** <span style='color:{boc};font-weight:bold'>"
                        f"{bo:.0%}</span>", unsafe_allow_html=True)
    with c4:
        career = projs.get('career_length')
        if career and isinstance(career, dict):
            st.markdown(f"**Est. Career:** <span style='font-weight:bold'>"
                        f"{career.get('projected', 0):.1f} yrs</span>", unsafe_allow_html=True)


def render_college_stats(prospect):
    pos = prospect.get('position', '')
    metrics_map = {
        'QB': [('PPR PPG', 'college_ppr_ppg'), ('Pass YPG', 'college_pass_yds_pg'),
               ('Pass TD/G', 'college_pass_td_pg'), ('Comp %', 'college_cmp_pct'),
               ('YPA', 'college_ypa'), ('Rush YPG', 'college_rush_yds_pg')],
        'RB': [('PPR PPG', 'college_ppr_ppg'), ('Rush ATT/G', 'college_rush_att_pg'),
               ('Rush YPG', 'college_rush_yds_pg'), ('YPC', 'college_ypc'),
               ('Rec/G', 'college_rec_pg'), ('Rec YPG', 'college_rec_yds_pg')],
        'WR': [('PPR PPG', 'college_ppr_ppg'), ('Rec/G', 'college_rec_pg'),
               ('Rec YPG', 'college_rec_yds_pg'), ('Rec TD/G', 'college_rec_td_pg'),
               ('YPR (aDOT proxy)', 'college_ypr'), ('Rush YPG', 'college_rush_yds_pg')],
        'TE': [('PPR PPG', 'college_ppr_ppg'), ('Rec/G', 'college_rec_pg'),
               ('Rec YPG', 'college_rec_yds_pg'), ('Rec TD/G', 'college_rec_td_pg'),
               ('YPR', 'college_ypr'), ('Games', 'total_college_games')],
    }
    metrics = metrics_map.get(pos, [])
    cols = st.columns(len(metrics))
    for col, (label, key) in zip(cols, metrics):
        with col:
            val = prospect.get(key)
            st.metric(label, fmt(val) if val is not None else '—')


def render_comp_row(i, comp):
    sim = comp.get('similarity_score', 0)
    sim_color = '#00C851' if sim >= 0.7 else '#33b5e5' if sim >= 0.5 else '#ff8800'
    arch = comp.get('archetype', 'N/A')
    arch_color = ARCHETYPE_COLORS.get(arch, '#666')

    draft_str = ''
    if comp.get('draft_round') and comp.get('draft_pick'):
        draft_str = f"Rd {int(comp['draft_round'])}, Pick {int(comp['draft_pick'])}"

    c1, c2, c3, c4, c5, c6 = st.columns([0.4, 3, 2, 1.5, 1.5, 1.5])
    with c1:
        st.markdown(f"**#{i}**")
    with c2:
        st.markdown(f"**{comp['name']}** <span style='font-size:0.8rem;color:#666'>"
                    f"{comp.get('college', '')}</span>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='archetype-badge' style='background:{arch_color};font-size:0.7rem'>"
            f"{arch.replace('_', ' ')}</span>"
            f"{'  📝 ' + draft_str if draft_str else ''}",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"<span style='color:{sim_color};font-weight:bold;font-size:1.1rem'>"
                    f"{sim * 100:.0f}% Match</span>", unsafe_allow_html=True)
        cats = comp.get('category_scores', {})
        st.caption(" | ".join(f"{k[:4]}: {v:.0%}" for k, v in cats.items()))
    with c4:
        st.metric("Rookie PPG", fmt(comp.get('rookie_ppr_ppg')))
    with c5:
        st.metric("Dynasty PPG", fmt(comp.get('dynasty_ppg')))
    with c6:
        st.metric("Peak 3yr", fmt(comp.get('peak_3yr_ppr_ppg')))


def render_compare_card(data):
    prospect = data['prospect']
    projs = data['projections']
    evaluation = data['evaluation']

    arch = prospect.get('archetype', 'N/A')
    arch_color = ARCHETYPE_COLORS.get(arch, '#666')

    st.markdown(f"### {prospect['name']}")
    st.markdown(f"**{prospect['position']}** — {prospect.get('school', 'N/A')}")
    st.markdown(f"<span class='archetype-badge' style='background:{arch_color}'>"
                f"{arch.replace('_', ' ')}</span>", unsafe_allow_html=True)

    ht = prospect.get('combine_height')
    wt = prospect.get('combine_weight')
    forty = prospect.get('combine_40')

    m1, m2, m3 = st.columns(3)
    with m1:
        if ht:
            st.metric("Height", f"{int(ht // 12)}'{ht % 12:.0f}\"")
        else:
            st.metric("Height", "—")
    with m2:
        st.metric("Weight", f"{int(wt)} lbs" if wt else "—")
    with m3:
        st.metric("40-Yard", f"{forty}s" if forty else "—")

    st.markdown("---")

    for proj_key, label in [('rookie', 'Rookie PPG'), ('dynasty', 'Dynasty PPG'),
                            ('peak_3yr', 'Peak 3yr'), ('best_season', 'Ceiling')]:
        proj = projs.get(proj_key)
        if proj and isinstance(proj, dict) and proj.get('projected') is not None:
            st.metric(label, f"{proj['projected']:.1f}",
                      delta=f"Floor {proj.get('floor', 0):.1f} / Ceil {proj.get('ceiling', 0):.1f}")

    tier = evaluation.get('tier', '—')
    color = get_tier_color(tier)
    st.markdown(f"**Tier:** <span style='color:{color};font-weight:bold'>{tier}</span>",
                unsafe_allow_html=True)

    bust = evaluation.get('bust_probability')
    bo = evaluation.get('breakout_probability')
    if bust is not None:
        st.markdown(f"**Bust:** {bust:.0%} | **Breakout:** {fmt_pct(bo)}")


# ─── Custom Prospect Engine ────────────────────────────────────────────────
def build_custom_profile(pos, stats, measurables, draft_pick, draft_round):
    profile = {
        'name': 'Custom Prospect',
        'position': pos,
        'is_prospect': True,
        'school': '',
    }

    for key, val in stats.items():
        if val is not None and val > 0:
            profile[key] = val

    if pos == 'QB':
        pass_yds = stats.get('college_pass_yds_pg', 0) or 0
        pass_td = stats.get('college_pass_td_pg', 0) or 0
        rush_yds = stats.get('college_rush_yds_pg', 0) or 0
        rush_td = stats.get('college_rush_td_pg', 0) or 0
        ints_pg = stats.get('college_int_pg', 0) or 0
        ppg = pass_yds * 0.04 + pass_td * 4 - ints_pg * 2 + rush_yds * 0.1 + rush_td * 6
    else:
        rec = stats.get('college_rec_pg', 0) or 0
        rec_yds = stats.get('college_rec_yds_pg', 0) or 0
        rec_td = stats.get('college_rec_td_pg', 0) or 0
        rush_yds = stats.get('college_rush_yds_pg', 0) or 0
        rush_td = stats.get('college_rush_td_pg', 0) or 0
        ppg = rec * 1.0 + rec_yds * 0.1 + rec_td * 6 + rush_yds * 0.1 + rush_td * 6

    profile['college_ppr_ppg'] = ppg
    profile['peak_ppr_ppg'] = ppg * 1.1

    if pos in ('WR', 'TE'):
        profile['peak_rec_pg'] = stats.get('college_rec_pg', 0) or 0
        profile['peak_rec_yds_pg'] = stats.get('college_rec_yds_pg', 0) or 0
        profile['peak_rec_td_pg'] = stats.get('college_rec_td_pg', 0) or 0
        profile['peak_ypr'] = stats.get('college_ypr', 0) or 0
        profile['peak_rec_yds'] = (stats.get('college_rec_yds_pg', 0) or 0) * 12
        profile['peak_rush_yds_pg'] = stats.get('college_rush_yds_pg', 0) or 0
        rec_pg = stats.get('college_rec_pg', 0) or 0
        rush_att_pg = stats.get('college_rush_att_pg', 0) or 0
        rec_td_pg = stats.get('college_rec_td_pg', 0) or 0
        rush_td_pg = stats.get('college_rush_td_pg', 0) or 0
        touches = rec_pg + rush_att_pg
        tds = rec_td_pg + rush_td_pg
        profile['peak_td_rate'] = tds / touches if touches > 0 else 0
        profile['college_td_rate'] = profile['peak_td_rate']
    elif pos == 'RB':
        profile['peak_rush_yds_pg'] = stats.get('college_rush_yds_pg', 0) or 0
        profile['peak_rush_td_pg'] = stats.get('college_rush_td_pg', 0) or 0
        profile['peak_rush_att_pg'] = stats.get('college_rush_att_pg', 0) or 0
        profile['peak_ypc'] = stats.get('college_ypc', 0) or 0
        profile['peak_rush_yds'] = (stats.get('college_rush_yds_pg', 0) or 0) * 12
        profile['peak_rec_pg'] = stats.get('college_rec_pg', 0) or 0
        profile['peak_rec_yds_pg'] = stats.get('college_rec_yds_pg', 0) or 0
        rush_att_pg = stats.get('college_rush_att_pg', 0) or 0
        rec_pg = stats.get('college_rec_pg', 0) or 0
        rush_td_pg = stats.get('college_rush_td_pg', 0) or 0
        rec_td_pg = stats.get('college_rec_td_pg', 0) or 0
        touches = rush_att_pg + rec_pg
        tds = rush_td_pg + rec_td_pg
        profile['peak_td_rate'] = tds / touches if touches > 0 else 0
        profile['college_td_rate'] = profile['peak_td_rate']
    elif pos == 'QB':
        profile['peak_pass_yds_pg'] = stats.get('college_pass_yds_pg', 0) or 0
        profile['peak_pass_td_pg'] = stats.get('college_pass_td_pg', 0) or 0
        profile['peak_cmp_pct'] = stats.get('college_cmp_pct', 0) or 0
        profile['peak_ypa'] = stats.get('college_ypa', 0) or 0
        profile['peak_qb_rating'] = stats.get('college_qb_rating', 0) or 0
        profile['peak_rush_yds_pg_qb'] = stats.get('college_rush_yds_pg', 0) or 0

    profile['total_college_games'] = stats.get('total_college_games', 36) or 36
    profile['total_college_seasons'] = 3
    profile['productive_seasons'] = 2
    profile['breakout_ratio'] = 1.1
    profile['last_season_ppr_ppg'] = ppg
    profile['last_season_rec_pg'] = stats.get('college_rec_pg', 0) or 0
    profile['last_season_rec_yds_pg'] = stats.get('college_rec_yds_pg', 0) or 0
    profile['rec_yds_per_team_game'] = (stats.get('college_rec_yds_pg', 0) or 0) * 12
    profile['rush_yds_per_team_game'] = (stats.get('college_rush_yds_pg', 0) or 0) * 12

    for key, val in measurables.items():
        if val is not None and val > 0:
            profile[key] = val

    profile['draft_pick'] = draft_pick
    profile['draft_round'] = draft_round

    return profile


def run_custom_similarity(prospect_profile, nfl_profiles, n=10):
    from modeling.similarity import (
        POS_FEATURES, compute_feature_stats,
        compute_similarity, enrich_profiles
    )

    pos = prospect_profile.get('position', '')
    features = POS_FEATURES.get(pos, [])
    if not features:
        return []

    nfl_profiles = enrich_profiles(list(nfl_profiles))
    pos_players = [p for p in nfl_profiles if p.get('position') == pos
                   and p.get('rookie_ppr_ppg') is not None]
    if not pos_players:
        return []

    all_pool = pos_players + [prospect_profile]
    feat_stats = compute_feature_stats(all_pool, features)

    comparisons = []
    for nfl_p in pos_players:
        score, feat_comp, cat_scores = compute_similarity(
            prospect_profile, nfl_p, features, feat_stats
        )
        comparisons.append({
            'name': nfl_p.get('name', '?'),
            'position': nfl_p.get('position', ''),
            'archetype': nfl_p.get('archetype', ''),
            'college': nfl_p.get('college', ''),
            'draft_round': nfl_p.get('draft_round'),
            'draft_pick': nfl_p.get('draft_pick'),
            'similarity_score': round(score, 4),
            'category_scores': {k: round(v, 4) for k, v in cat_scores.items()},
            'feature_comparison': feat_comp,
            'rookie_ppr_ppg': nfl_p.get('rookie_ppr_ppg'),
            'dynasty_ppg': nfl_p.get('dynasty_ppg'),
            'best_season_ppr_ppg': nfl_p.get('best_season_ppr_ppg'),
            'peak_3yr_ppr_ppg': nfl_p.get('peak_3yr_ppr_ppg'),
            'nfl_seasons_played': nfl_p.get('nfl_seasons_played'),
        })

    comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
    return comparisons[:n]


def run_custom_projections(comparisons, draft_pick, draft_round, pos):
    from modeling.projections import (
        weighted_projection, draft_capital_adjustment,
        tier_label, bust_probability, breakout_probability
    )

    rookie = weighted_projection(comparisons, 'rookie_ppr_ppg', exp=2)
    dynasty = weighted_projection(comparisons, 'dynasty_ppg', exp=2)
    peak = weighted_projection(comparisons, 'peak_3yr_ppr_ppg', exp=2)
    best_season = weighted_projection(comparisons, 'best_season_ppr_ppg', exp=2)
    career = weighted_projection(comparisons, 'nfl_seasons_played', exp=2)

    if draft_round is not None:
        rookie = draft_capital_adjustment(rookie, draft_round, draft_pick, pos)
        dynasty = draft_capital_adjustment(dynasty, draft_round, draft_pick, pos)
        peak = draft_capital_adjustment(peak, draft_round, draft_pick, pos)
        best_season = draft_capital_adjustment(best_season, draft_round, draft_pick, pos)

    dynasty_ppg = dynasty['projected'] if dynasty else 0
    tier = tier_label(dynasty_ppg, pos)
    bust_prob = bust_probability(comparisons, pos)
    breakout_prob = breakout_probability(comparisons, pos)

    return {
        'projections': {
            'rookie': rookie,
            'dynasty': dynasty,
            'peak_3yr': peak,
            'best_season': best_season,
            'career_length': career,
        },
        'evaluation': {
            'tier': tier,
            'bust_probability': bust_prob,
            'breakout_probability': breakout_prob,
        },
    }


# ─── Main App ──────────────────────────────────────────────────────────────
def main():
    projections = load_projections()
    df = build_board_df(projections)

    with st.sidebar:
        st.title("🏈 Prospect Projector")
        st.caption("2026 NFL Draft — Fantasy Football")
        st.divider()

        st.subheader("🔍 Filters")
        pos_filter = st.multiselect("Position", ['QB', 'RB', 'WR', 'TE'],
                                    default=['QB', 'RB', 'WR', 'TE'])
        arch_options = sorted(df['Archetype'].dropna().unique().tolist())
        arch_filter = st.multiselect("Archetype", arch_options, default=arch_options)
        sort_by = st.selectbox("Sort By",
                               ['Dynasty PPG', 'Rookie PPG', 'Peak 3yr PPG',
                                'Ceiling PPG', 'Board Rank'], index=0)

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

    filtered = df[df['Pos'].isin(pos_filter) & df['Archetype'].isin(arch_filter)].copy()
    asc = sort_by == 'Board Rank'
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=asc, na_position='last')

    st.markdown('<div class="main-header">🏈 2026 NFL Fantasy Prospect Projector</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Historical comparisons • Archetype analysis • '
                'Rookie & dynasty projections</div>', unsafe_allow_html=True)

    tab_board, tab_detail, tab_pos, tab_compare, tab_custom = st.tabs(
        ["📋 Big Board", "🔬 Deep Dive", "📊 Position Rankings",
         "⚖️ Compare", "🛠️ Build Custom Prospect"]
    )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: BIG BOARD
    # ═══════════════════════════════════════════════════════════════════
    with tab_board:
        st.subheader(f"Big Board — {len(filtered)} Prospects")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QBs", len(filtered[filtered['Pos'] == 'QB']))
        c2.metric("RBs", len(filtered[filtered['Pos'] == 'RB']))
        c3.metric("WRs", len(filtered[filtered['Pos'] == 'WR']))
        c4.metric("TEs", len(filtered[filtered['Pos'] == 'TE']))

        display_cols = ['Name', 'Pos', 'School', 'Archetype', 'Height', 'Weight',
                        '40-Yard', 'College PPG', 'Rookie PPG', 'Dynasty PPG',
                        'Peak 3yr PPG', 'Tier', 'Bust %', 'Breakout %']
        available = [c for c in display_cols if c in filtered.columns]
        show_df = filtered[available].reset_index(drop=True)
        show_df.index = show_df.index + 1
        show_df.index.name = 'Rank'

        num_fmt = {c: '{:.1f}' for c in ['College PPG', 'Rookie PPG', 'Dynasty PPG',
                                          'Peak 3yr PPG', 'Height', 'Weight', '40-Yard']
                   if c in show_df.columns}
        pct_fmt = {c: '{:.0%}' for c in ['Bust %', 'Breakout %'] if c in show_df.columns}
        all_fmt = {**num_fmt, **pct_fmt}

        def style_tier(val):
            if pd.isna(val) or val == '':
                return ''
            return f'color: {get_tier_color(str(val))}; font-weight: bold'

        def style_bust(val):
            if pd.isna(val):
                return ''
            if val >= 0.6:
                return 'color: #ff4444; font-weight: bold'
            if val >= 0.4:
                return 'color: #ff8800'
            return 'color: #00C851'

        def style_breakout(val):
            if pd.isna(val):
                return ''
            if val >= 0.4:
                return 'color: #FFD700; font-weight: bold'
            if val >= 0.2:
                return 'color: #00C851'
            return 'color: #666'

        styler = show_df.style.format(all_fmt, na_rep='—')
        if 'Tier' in show_df.columns:
            styler = styler.map(style_tier, subset=['Tier'])
        if 'Bust %' in show_df.columns:
            styler = styler.map(style_bust, subset=['Bust %'])
        if 'Breakout %' in show_df.columns:
            styler = styler.map(style_breakout, subset=['Breakout %'])

        st.dataframe(styler, use_container_width=True, height=650)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════
    with tab_detail:
        prospect_names = filtered.sort_values('Name')['Name'].tolist()
        if not prospect_names:
            st.warning("No prospects match your filters.")
        else:
            selected_name = st.selectbox("Select a prospect", prospect_names, index=0)
            sel_row = filtered[filtered['Name'] == selected_name].iloc[0]
            pid = sel_row['espn_id']
            pdata = projections.get(str(pid), projections.get(pid))

            if pdata is None:
                st.error("Prospect data not found.")
            else:
                prospect = pdata['prospect']
                comps = pdata['comparisons']
                projs = pdata['projections']
                evaluation = pdata['evaluation']

                h1, h2 = st.columns([1, 2])
                with h1:
                    arch = prospect.get('archetype', 'N/A')
                    arch_color = ARCHETYPE_COLORS.get(arch, '#666')

                    st.markdown(f"## {prospect['name']}")
                    st.markdown(f"**{prospect['position']}** — "
                                f"{prospect.get('school', 'N/A')}")
                    st.markdown(
                        f"<span class='archetype-badge' style='background:{arch_color}'>"
                        f"{arch.replace('_', ' ')}</span>", unsafe_allow_html=True)

                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    ht = prospect.get('combine_height')
                    with m1:
                        if ht:
                            st.metric("Height", f"{int(ht // 12)}'{ht % 12:.0f}\"")
                        else:
                            st.metric("Height", "—")
                    with m2:
                        wt = prospect.get('combine_weight')
                        st.metric("Weight", f"{int(wt)} lbs" if wt else "—")
                    with m3:
                        forty = prospect.get('combine_40')
                        st.metric("40-Yard", f"{forty}s" if forty else "—")

                    vert = prospect.get('combine_vert')
                    broad = prospect.get('combine_broad')
                    if vert or broad:
                        m4, m5 = st.columns(2)
                        with m4:
                            st.metric("Vert Leap", f"{vert}\"" if vert else "—")
                        with m5:
                            st.metric("Broad Jump", f"{broad}\"" if broad else "—")

                with h2:
                    st.markdown("### 📈 Projections")
                    render_projection_cards(projs)
                    render_eval_row(evaluation, projs)

                st.divider()

                st.markdown("### 🎓 College Production")
                render_college_stats(prospect)

                st.divider()

                st.markdown("### 🔍 Historical Comparisons")
                st.caption("Most similar NFL players based on college production, "
                           "measurables, and draft capital")

                if not comps:
                    st.info("No historical comparisons found.")
                else:
                    for i, comp in enumerate(comps[:10], 1):
                        render_comp_row(i, comp)
                        st.markdown("---")

                    st.markdown("### 📊 Comp Outcome Distributions")
                    comp_df = pd.DataFrame(comps[:10])

                    o1, o2 = st.columns(2)
                    with o1:
                        st.markdown("**Rookie Year PPG**")
                        if 'rookie_ppr_ppg' in comp_df.columns:
                            chart_data = comp_df[['name', 'rookie_ppr_ppg']].dropna()
                            chart_data.columns = ['Player', 'PPG']
                            st.bar_chart(chart_data.set_index('Player'), use_container_width=True)
                    with o2:
                        st.markdown("**Dynasty PPG**")
                        if 'dynasty_ppg' in comp_df.columns:
                            chart_data = comp_df[['name', 'dynasty_ppg']].dropna()
                            chart_data.columns = ['Player', 'PPG']
                            st.bar_chart(chart_data.set_index('Player'), use_container_width=True)

                    st.markdown("### 📋 Feature Comparison vs Top Comp")
                    if comps:
                        top_comp = comps[0]
                        feat_comp = top_comp.get('feature_comparison', {})
                        if feat_comp:
                            feat_rows = []
                            for fk, vals in feat_comp.items():
                                label = (fk.replace('college_', '').replace('combine_', '')
                                         .replace('_', ' ').title())
                                pv = vals.get('prospect')
                                cv = vals.get('comp')
                                feat_rows.append({
                                    'Feature': label,
                                    prospect['name']: fmt(pv),
                                    top_comp['name']: fmt(cv),
                                    'Diff %': f"{vals.get('diff_pct', 0):.0f}%",
                                })
                            st.dataframe(pd.DataFrame(feat_rows),
                                         use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: POSITION RANKINGS
    # ═══════════════════════════════════════════════════════════════════
    with tab_pos:
        st.subheader("Position Rankings")

        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos not in pos_filter:
                continue

            pos_df = filtered[filtered['Pos'] == pos].copy()
            if pos_df.empty:
                continue

            pos_df = pos_df.sort_values('Dynasty PPG', ascending=False, na_position='last')
            pos_df = pos_df.reset_index(drop=True)
            pos_df.index = pos_df.index + 1
            pos_df.index.name = 'Rank'

            st.markdown(f"### {pos} Rankings")

            pcols = ['Name', 'School', 'Archetype', 'Rookie PPG', 'Dynasty PPG',
                     'Peak 3yr PPG', 'Ceiling PPG', 'Tier', 'Bust %', 'Breakout %']
            pavail = [c for c in pcols if c in pos_df.columns]

            pfmt = {c: '{:.1f}' for c in ['Rookie PPG', 'Dynasty PPG', 'Peak 3yr PPG',
                                           'Ceiling PPG'] if c in pos_df.columns}
            pfmt.update({c: '{:.0%}' for c in ['Bust %', 'Breakout %'] if c in pos_df.columns})

            pstyler = pos_df[pavail].style.format(pfmt, na_rep='—')
            if 'Tier' in pos_df.columns:
                pstyler = pstyler.map(style_tier, subset=['Tier'])

            st.dataframe(pstyler, use_container_width=True)

            with st.expander(f"📊 {pos} Insights"):
                ic1, ic2, ic3 = st.columns(3)

                with ic1:
                    st.markdown("**Top Dynasty Value**")
                    for _, row in pos_df.nlargest(3, 'Dynasty PPG', 'all').iterrows():
                        st.markdown(f"- **{row['Name']}**: {fmt(row.get('Dynasty PPG'))} PPG")

                with ic2:
                    st.markdown("**Highest Ceiling**")
                    for _, row in pos_df.nlargest(3, 'Ceiling PPG', 'all').iterrows():
                        st.markdown(f"- **{row['Name']}**: {fmt(row.get('Ceiling PPG'))} PPG")

                with ic3:
                    st.markdown("**Safest Picks (Lowest Bust %)**")
                    safe = pos_df.dropna(subset=['Bust %'])
                    if not safe.empty:
                        for _, row in safe.nsmallest(3, 'Bust %').iterrows():
                            st.markdown(f"- **{row['Name']}**: {fmt_pct(row.get('Bust %'))} bust")

            st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: COMPARE
    # ═══════════════════════════════════════════════════════════════════
    with tab_compare:
        st.subheader("Head-to-Head Prospect Comparison")

        prospect_names = filtered.sort_values('Name')['Name'].tolist()
        if len(prospect_names) < 2:
            st.warning("Need at least 2 prospects to compare.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                name_a = st.selectbox("Prospect A", prospect_names, index=0, key='cmp_a')
            with cc2:
                name_b = st.selectbox("Prospect B", prospect_names,
                                      index=min(1, len(prospect_names) - 1), key='cmp_b')

            row_a = filtered[filtered['Name'] == name_a].iloc[0]
            row_b = filtered[filtered['Name'] == name_b].iloc[0]

            pid_a = row_a['espn_id']
            pid_b = row_b['espn_id']

            data_a = projections.get(str(pid_a), projections.get(pid_a))
            data_b = projections.get(str(pid_b), projections.get(pid_b))

            if data_a and data_b:
                st.divider()

                col_a, col_vs, col_b = st.columns([5, 1, 5])

                with col_a:
                    render_compare_card(data_a)
                with col_vs:
                    st.markdown("<div style='text-align:center;padding-top:8rem'>"
                                "<h1>VS</h1></div>", unsafe_allow_html=True)
                with col_b:
                    render_compare_card(data_b)

                st.divider()
                st.markdown("### 📊 Stat Comparison")

                compare_fields = [
                    ('Position', 'Pos'), ('School', 'School'), ('Archetype', 'Archetype'),
                    ('Height (in)', 'Height'), ('Weight (lbs)', 'Weight'),
                    ('40-Yard', '40-Yard'), ('College PPR PPG', 'College PPG'),
                    ('Peak College PPG', 'Peak College PPG'),
                    ('Rookie PPG Proj', 'Rookie PPG'), ('Dynasty PPG Proj', 'Dynasty PPG'),
                    ('Peak 3yr PPG Proj', 'Peak 3yr PPG'),
                    ('Ceiling PPG Proj', 'Ceiling PPG'), ('Tier', 'Tier'),
                    ('Bust %', 'Bust %'), ('Breakout %', 'Breakout %'),
                    ('Board Rank', 'Board Rank'),
                ]

                comp_rows = []
                for label, col_key in compare_fields:
                    va = row_a.get(col_key)
                    vb = row_b.get(col_key)

                    if col_key in ('Bust %', 'Breakout %'):
                        sa = fmt_pct(va)
                        sb = fmt_pct(vb)
                    elif isinstance(va, float) and not pd.isna(va):
                        sa = f"{va:.1f}"
                        sb = f"{vb:.1f}" if isinstance(vb, float) and not pd.isna(vb) else '—'
                    else:
                        sa = str(va) if va is not None and not (isinstance(va, float) and pd.isna(va)) else '—'
                        sb = str(vb) if vb is not None and not (isinstance(vb, float) and pd.isna(vb)) else '—'

                    edge = ''
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        if not pd.isna(va) and not pd.isna(vb):
                            higher_better = col_key not in ('Bust %', '40-Yard', 'Board Rank')
                            if va > vb:
                                edge = '⬅️' if higher_better else '➡️'
                            elif vb > va:
                                edge = '➡️' if higher_better else '⬅️'
                            else:
                                edge = '🟰'

                    comp_rows.append({
                        'Metric': label,
                        name_a: sa,
                        'Edge': edge,
                        name_b: sb,
                    })

                comp_table = pd.DataFrame(comp_rows)
                st.dataframe(comp_table, use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("### 🔗 Shared Historical Comps")

                comps_a = {c['name'] for c in data_a.get('comparisons', [])[:10]}
                comps_b = {c['name'] for c in data_b.get('comparisons', [])[:10]}
                shared = comps_a & comps_b

                if shared:
                    st.success(f"**{len(shared)} shared comparisons:** {', '.join(sorted(shared))}")
                else:
                    st.info("No shared top-10 comparisons — these are very different profiles.")

                uc1, uc2 = st.columns(2)
                with uc1:
                    unique_a = comps_a - comps_b
                    if unique_a:
                        st.markdown(f"**Unique to {name_a}:** {', '.join(sorted(unique_a))}")
                with uc2:
                    unique_b = comps_b - comps_a
                    if unique_b:
                        st.markdown(f"**Unique to {name_b}:** {', '.join(sorted(unique_b))}")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: BUILD CUSTOM PROSPECT
    # ═══════════════════════════════════════════════════════════════════
    with tab_custom:
        st.subheader("🛠️ Build Custom Prospect")
        st.caption("Input college stats, measurables, and draft capital to generate a projection. "
                   "Use this to test hypothetical prospects or players not in the database.")

        st.divider()

        custom_pos = st.selectbox("Position", ['WR', 'RB', 'QB', 'TE'], index=0, key='custom_pos')

        col_stats, col_meas, col_draft = st.columns(3)

        with col_stats:
            st.markdown("### 🎓 College Stats (Per Game)")
            stats = {}
            if custom_pos == 'WR':
                stats['college_rec_pg'] = st.number_input("Receptions/Game", 0.0, 15.0, 5.0, 0.5, key='c_rec')
                stats['college_rec_yds_pg'] = st.number_input("Rec Yards/Game", 0.0, 200.0, 70.0, 5.0, key='c_recyds')
                stats['college_rec_td_pg'] = st.number_input("Rec TD/Game", 0.0, 3.0, 0.5, 0.1, key='c_rectd')
                stats['college_ypr'] = st.number_input("Yards Per Reception", 0.0, 25.0, 14.0, 0.5, key='c_ypr')
                stats['college_rush_yds_pg'] = st.number_input("Rush Yards/Game", 0.0, 50.0, 2.0, 1.0, key='c_rushyds')
                stats['college_rush_td_pg'] = st.number_input("Rush TD/Game", 0.0, 2.0, 0.1, 0.1, key='c_rushtd')
                stats['college_rush_att_pg'] = st.number_input("Rush Att/Game", 0.0, 10.0, 0.5, 0.5, key='c_rushatt')
            elif custom_pos == 'RB':
                stats['college_rush_att_pg'] = st.number_input("Rush Att/Game", 0.0, 30.0, 15.0, 1.0, key='c_rushatt')
                stats['college_rush_yds_pg'] = st.number_input("Rush Yards/Game", 0.0, 200.0, 80.0, 5.0, key='c_rushyds')
                stats['college_rush_td_pg'] = st.number_input("Rush TD/Game", 0.0, 3.0, 0.8, 0.1, key='c_rushtd')
                stats['college_ypc'] = st.number_input("Yards Per Carry", 0.0, 12.0, 5.5, 0.1, key='c_ypc')
                stats['college_rec_pg'] = st.number_input("Receptions/Game", 0.0, 10.0, 2.0, 0.5, key='c_rec')
                stats['college_rec_yds_pg'] = st.number_input("Rec Yards/Game", 0.0, 100.0, 15.0, 5.0, key='c_recyds')
                stats['college_rec_td_pg'] = st.number_input("Rec TD/Game", 0.0, 2.0, 0.1, 0.1, key='c_rectd')
                stats['college_ypr'] = st.number_input("Yards Per Reception", 0.0, 25.0, 8.0, 0.5, key='c_ypr')
            elif custom_pos == 'QB':
                stats['college_pass_yds_pg'] = st.number_input("Pass Yards/Game", 0.0, 500.0, 250.0, 10.0, key='c_passyds')
                stats['college_pass_td_pg'] = st.number_input("Pass TD/Game", 0.0, 5.0, 2.0, 0.1, key='c_passtd')
                stats['college_cmp_pct'] = st.number_input("Completion %", 0.0, 100.0, 65.0, 1.0, key='c_cmp')
                stats['college_ypa'] = st.number_input("Yards Per Attempt", 0.0, 15.0, 8.0, 0.1, key='c_ypa')
                stats['college_qb_rating'] = st.number_input("QB Rating", 0.0, 200.0, 140.0, 5.0, key='c_qbr')
                stats['college_rush_yds_pg'] = st.number_input("Rush Yards/Game", 0.0, 100.0, 20.0, 5.0, key='c_rushyds')
                stats['college_rush_td_pg'] = st.number_input("Rush TD/Game", 0.0, 2.0, 0.3, 0.1, key='c_rushtd')
                stats['college_pass_att_pg'] = st.number_input("Pass Att/Game", 0.0, 60.0, 30.0, 1.0, key='c_passatt')
                stats['college_int_pg'] = st.number_input("INT/Game", 0.0, 3.0, 0.5, 0.1, key='c_int')
            elif custom_pos == 'TE':
                stats['college_rec_pg'] = st.number_input("Receptions/Game", 0.0, 10.0, 3.0, 0.5, key='c_rec')
                stats['college_rec_yds_pg'] = st.number_input("Rec Yards/Game", 0.0, 150.0, 40.0, 5.0, key='c_recyds')
                stats['college_rec_td_pg'] = st.number_input("Rec TD/Game", 0.0, 2.0, 0.3, 0.1, key='c_rectd')
                stats['college_ypr'] = st.number_input("Yards Per Reception", 0.0, 25.0, 12.0, 0.5, key='c_ypr')
                stats['college_rush_yds_pg'] = st.number_input("Rush Yards/Game", 0.0, 20.0, 0.0, 1.0, key='c_rushyds')
                stats['college_rush_att_pg'] = st.number_input("Rush Att/Game", 0.0, 5.0, 0.0, 0.5, key='c_rushatt')
                stats['college_rush_td_pg'] = st.number_input("Rush TD/Game", 0.0, 1.0, 0.0, 0.1, key='c_rushtd')

            stats['total_college_games'] = st.number_input("Total College Games", 1, 60, 36, 1, key='c_games')

        with col_meas:
            st.markdown("### 📏 Measurables")

            ht_defaults = {'WR': 72.0, 'RB': 70.0, 'QB': 75.0, 'TE': 77.0}
            wt_defaults = {'WR': 195.0, 'RB': 210.0, 'QB': 220.0, 'TE': 245.0}
            forty_defaults = {'WR': 4.48, 'RB': 4.50, 'QB': 4.70, 'TE': 4.60}

            measurables = {}
            measurables['combine_height'] = st.number_input(
                "Height (inches)", 64.0, 82.0, ht_defaults[custom_pos], 0.5, key='m_ht')
            measurables['combine_weight'] = st.number_input(
                "Weight (lbs)", 150.0, 280.0, wt_defaults[custom_pos], 5.0, key='m_wt')
            measurables['combine_40'] = st.number_input(
                "40-Yard Dash", 4.20, 5.20, forty_defaults[custom_pos], 0.01, key='m_40')
            measurables['combine_vert'] = st.number_input(
                "Vertical Leap (in)", 0.0, 48.0, 35.0, 0.5, key='m_vert')
            measurables['combine_broad'] = st.number_input(
                "Broad Jump (in)", 0.0, 145.0, 120.0, 1.0, key='m_broad')
            measurables['combine_shuttle'] = st.number_input(
                "Shuttle (seconds)", 0.0, 5.00, 4.30, 0.01, key='m_shuttle')
            measurables['combine_3cone'] = st.number_input(
                "3-Cone (seconds)", 0.0, 8.00, 7.00, 0.01, key='m_3cone')

            st.caption("Set to 0 if unknown — will be excluded from matching")

        with col_draft:
            st.markdown("### 📝 Draft Capital")

            draft_round = st.selectbox("Draft Round", [1, 2, 3, 4, 5, 6, 7], index=1, key='d_round')
            draft_pick = st.number_input("Overall Pick", 1, 260, draft_round * 32 - 16, 1, key='d_pick')

            st.markdown("---")
            st.markdown("**Computed College PPR PPG:**")
            if custom_pos == 'QB':
                pass_yds = stats.get('college_pass_yds_pg', 0) or 0
                pass_td = stats.get('college_pass_td_pg', 0) or 0
                rush_yds = stats.get('college_rush_yds_pg', 0) or 0
                rush_td = stats.get('college_rush_td_pg', 0) or 0
                ints_pg = stats.get('college_int_pg', 0) or 0
                preview_ppg = pass_yds * 0.04 + pass_td * 4 - ints_pg * 2 + rush_yds * 0.1 + rush_td * 6
            else:
                rec = stats.get('college_rec_pg', 0) or 0
                rec_yds = stats.get('college_rec_yds_pg', 0) or 0
                rec_td = stats.get('college_rec_td_pg', 0) or 0
                rush_yds = stats.get('college_rush_yds_pg', 0) or 0
                rush_td = stats.get('college_rush_td_pg', 0) or 0
                preview_ppg = rec * 1.0 + rec_yds * 0.1 + rec_td * 6 + rush_yds * 0.1 + rush_td * 6

            st.metric("College PPR PPG", f"{preview_ppg:.1f}")

        clean_measurables = {k: v for k, v in measurables.items() if v is not None and v > 0}

        st.divider()

        if st.button("🚀 Generate Projection", type="primary", use_container_width=True):
            with st.spinner("Running similarity matching against 500+ NFL players..."):
                all_profiles = load_profiles()
                nfl_profiles = [p for p in all_profiles if not p.get('is_prospect')
                                and p.get('rookie_ppr_ppg') is not None]

                custom_profile = build_custom_profile(
                    custom_pos, stats, clean_measurables,
                    float(draft_pick), float(draft_round)
                )

                comps = run_custom_similarity(custom_profile, nfl_profiles, n=10)

                if not comps:
                    st.error("No comparisons found. Try adjusting your inputs.")
                else:
                    results = run_custom_projections(
                        comps, float(draft_pick), float(draft_round), custom_pos
                    )
                    projs = results['projections']
                    evaluation = results['evaluation']

                    st.markdown("## 📈 Custom Prospect Projection")
                    render_projection_cards(projs)
                    st.markdown("")
                    render_eval_row(evaluation, projs)

                    st.divider()
                    st.markdown("### 🔍 Historical Comparisons")

                    for i, comp in enumerate(comps[:10], 1):
                        render_comp_row(i, comp)
                        st.markdown("---")

                    st.markdown("### 📊 Comp Outcome Distribution")
                    comp_df = pd.DataFrame(comps[:10])

                    o1, o2 = st.columns(2)
                    with o1:
                        st.markdown("**Rookie Year PPG**")
                        if 'rookie_ppr_ppg' in comp_df.columns:
                            chart_data = comp_df[['name', 'rookie_ppr_ppg']].dropna()
                            chart_data.columns = ['Player', 'PPG']
                            st.bar_chart(chart_data.set_index('Player'), use_container_width=True)
                    with o2:
                        st.markdown("**Dynasty PPG**")
                        if 'dynasty_ppg' in comp_df.columns:
                            chart_data = comp_df[['name', 'dynasty_ppg']].dropna()
                            chart_data.columns = ['Player', 'PPG']
                            st.bar_chart(chart_data.set_index('Player'), use_container_width=True)


if __name__ == "__main__":
    main()