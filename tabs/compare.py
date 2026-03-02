# tabs/compare.py
"""Tab 4: Compare — head-to-head prospect comparison."""

import streamlit as st
import pandas as pd
from tabs.helpers import fmt, fmt_pct, render_compare_card


def render(filtered, projections):
    st.subheader("Head-to-Head Prospect Comparison")

    prospect_names = filtered.sort_values('Name')['Name'].tolist()
    if len(prospect_names) < 2:
        st.warning("Need at least 2 prospects to compare.")
        return

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

    if not data_a or not data_b:
        st.error("Could not load data for selected prospects.")
        return

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