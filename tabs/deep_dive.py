# tabs/deep_dive.py
"""Tab 2: Deep Dive — individual prospect analysis with comps."""

import streamlit as st
import pandas as pd
from tabs.helpers import (
    ARCHETYPE_COLORS, fmt, render_projection_cards,
    render_eval_row, render_college_stats, render_comp_row
)


def render(filtered, projections):
    prospect_names = filtered.sort_values('Name')['Name'].tolist()
    if not prospect_names:
        st.warning("No prospects match your filters.")
        return

    selected_name = st.selectbox("Select a prospect", prospect_names, index=0)
    sel_row = filtered[filtered['Name'] == selected_name].iloc[0]
    pid = sel_row['espn_id']
    pdata = projections.get(str(pid), projections.get(pid))

    if pdata is None:
        st.error("Prospect data not found.")
        return

    prospect = pdata['prospect']
    comps = pdata['comparisons']
    projs = pdata['projections']
    evaluation = pdata['evaluation']

    h1, h2 = st.columns([1, 2])
    with h1:
        arch = prospect.get('archetype', 'N/A')
        arch_color = ARCHETYPE_COLORS.get(arch, '#666')

        st.markdown(f"## {prospect['name']}")
        st.markdown(f"**{prospect['position']}** — {prospect.get('school', 'N/A')}")
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