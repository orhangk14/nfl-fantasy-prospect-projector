# tabs/helpers.py
"""
Shared rendering helpers for all tabs.
"""

import streamlit as st
import pandas as pd

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