# tabs/position_rankings.py
"""Tab 3: Position Rankings — per-position breakdowns with insights."""

import streamlit as st
import pandas as pd
from tabs.helpers import fmt, fmt_pct, style_tier


def render(filtered, pos_filter):
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