# tabs/big_board.py
"""Tab 1: Big Board — sortable/filterable prospect rankings."""

import streamlit as st
import pandas as pd
from tabs.helpers import get_tier_color, style_tier, style_bust, style_breakout


def render(filtered):
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

    styler = show_df.style.format(all_fmt, na_rep='—')
    if 'Tier' in show_df.columns:
        styler = styler.map(style_tier, subset=['Tier'])
    if 'Bust %' in show_df.columns:
        styler = styler.map(style_bust, subset=['Bust %'])
    if 'Breakout %' in show_df.columns:
        styler = styler.map(style_breakout, subset=['Breakout %'])

    st.dataframe(styler, use_container_width=True, height=650)