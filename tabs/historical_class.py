# tabs/historical_class.py
"""Tab 6: Historical Draft Class — view past draft classes and how they performed."""

import streamlit as st
import pandas as pd
from tabs.helpers import fmt, fmt_pct, get_tier_color, style_tier, ARCHETYPE_COLORS


def render(profiles):
    st.subheader("📜 Historical Draft Class Explorer")
    st.caption("Select a draft year to see how that class performed in the NFL. "
               "Compare projected vs actual outcomes.")

    # Get available draft years
    nfl_players = [p for p in profiles if not p.get('is_prospect')
                   and p.get('rookie_ppr_ppg') is not None
                   and p.get('draft_year') is not None]

    if not nfl_players:
        st.warning("No historical player data available.")
        return

    years = sorted(set(
        int(float(p['draft_year'])) for p in nfl_players
        if p.get('draft_year') is not None
    ), reverse=True)

    if not years:
        st.warning("No draft year data available.")
        return

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_year = st.selectbox("Draft Year", years, index=0, key='hist_year')
    with c2:
        pos_filter = st.multiselect("Position", ['QB', 'RB', 'WR', 'TE'],
                                     default=['QB', 'RB', 'WR', 'TE'], key='hist_pos')
    with c3:
        sort_by = st.selectbox("Sort By",
                               ['Draft Pick', 'Rookie PPG', 'Dynasty PPG',
                                'Best Season PPG', 'College PPG'],
                               index=0, key='hist_sort')

    # Filter players for selected year
    class_players = [
        p for p in nfl_players
        if int(float(p['draft_year'])) == selected_year
        and p.get('position', '') in pos_filter
    ]

    if not class_players:
        st.info(f"No players found for the {selected_year} draft class with selected positions.")
        return

    # Build DataFrame
    rows = []
    for p in class_players:
        draft_rd = p.get('draft_round')
        draft_pk = p.get('draft_pick')

        row = {
            'Name': p.get('name', '?'),
            'Pos': p.get('position', '?'),
            'College': p.get('college', p.get('school', '?')),
            'Archetype': p.get('archetype', '?'),
            'Draft Pick': int(float(draft_pk)) if draft_pk else None,
            'Draft Rd': int(float(draft_rd)) if draft_rd else None,
            'College PPG': p.get('college_ppr_ppg'),
            'Peak College PPG': p.get('peak_ppr_ppg'),
            'Rookie PPG': p.get('rookie_ppr_ppg'),
            'Rookie GP': p.get('rookie_gp'),
            'Dynasty PPG': p.get('dynasty_ppg'),
            'Best Season PPG': p.get('best_season_ppr_ppg'),
            'Peak 3yr PPG': p.get('peak_3yr_ppr_ppg'),
            'NFL Seasons': p.get('nfl_seasons_played'),
            'Career PPR Total': p.get('career_ppr_total'),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort
    sort_map = {
        'Draft Pick': ('Draft Pick', True),
        'Rookie PPG': ('Rookie PPG', False),
        'Dynasty PPG': ('Dynasty PPG', False),
        'Best Season PPG': ('Best Season PPG', False),
        'College PPG': ('College PPG', False),
    }
    sort_col, sort_asc = sort_map.get(sort_by, ('Draft Pick', True))
    df = df.sort_values(sort_col, ascending=sort_asc, na_position='last')

    # Summary metrics
    st.divider()
    st.markdown(f"### {selected_year} Draft Class — {len(df)} Players")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Players", len(df))
    m2.metric("Avg Rookie PPG", fmt(df['Rookie PPG'].mean()))
    m3.metric("Avg Dynasty PPG", fmt(df['Dynasty PPG'].mean()))
    m4.metric("Best Rookie", fmt(df['Rookie PPG'].max()))
    m5.metric("Best Dynasty", fmt(df['Dynasty PPG'].max()))

    # Position breakdown
    st.divider()
    pos_cols = st.columns(4)
    for col, pos in zip(pos_cols, ['QB', 'RB', 'WR', 'TE']):
        with col:
            pos_df = df[df['Pos'] == pos]
            if not pos_df.empty:
                st.markdown(f"**{pos}s: {len(pos_df)}**")
                top = pos_df.nlargest(1, 'Dynasty PPG')
                if not top.empty:
                    r = top.iloc[0]
                    st.caption(f"Best: {r['Name']} ({fmt(r['Dynasty PPG'])} PPG)")
            else:
                st.markdown(f"**{pos}s: 0**")

    # Main table
    st.divider()

    display_cols = ['Name', 'Pos', 'College', 'Archetype', 'Draft Rd', 'Draft Pick',
                    'College PPG', 'Rookie PPG', 'Dynasty PPG', 'Best Season PPG',
                    'NFL Seasons', 'Career PPR Total']
    available = [c for c in display_cols if c in df.columns]
    show_df = df[available].reset_index(drop=True)
    show_df.index = show_df.index + 1
    show_df.index.name = '#'

    num_fmt = {c: '{:.1f}' for c in ['College PPG', 'Rookie PPG', 'Dynasty PPG',
                                      'Best Season PPG', 'Peak 3yr PPG',
                                      'Career PPR Total'] if c in show_df.columns}

    styler = show_df.style.format(num_fmt, na_rep='—')
    st.dataframe(styler, use_container_width=True, height=500)

    # Highlights
    st.divider()
    st.markdown("### 🏆 Class Highlights")

    h1, h2 = st.columns(2)

    with h1:
        st.markdown("**🌟 Hits (Dynasty PPG ≥ 12)**")
        hits = df[df['Dynasty PPG'] >= 12].sort_values('Dynasty PPG', ascending=False)
        if hits.empty:
            hits = df.nlargest(3, 'Dynasty PPG')
            st.caption("(Top 3 — no players hit 12+ dynasty PPG)")
        for _, r in hits.iterrows():
            pick_str = (f"Rd {int(r['Draft Rd'])}, Pick {int(r['Draft Pick'])}"
                        if pd.notna(r.get('Draft Pick')) else 'UDFA')
            st.markdown(f"- **{r['Name']}** ({r['Pos']}) — {pick_str} → "
                        f"Dynasty {fmt(r['Dynasty PPG'])} PPG, "
                        f"Best Season {fmt(r.get('Best Season PPG'))} PPG")

    with h2:
        st.markdown("**💔 Busts (Rd 1–2, Dynasty PPG < 8)**")
        early = df[(df['Draft Rd'].notna()) & (df['Draft Rd'] <= 2)]
        busts = early[early['Dynasty PPG'] < 8].sort_values('Draft Pick', ascending=True)
        if busts.empty:
            st.caption("No early-round busts in this class — solid drafting! ✅")
        else:
            for _, r in busts.iterrows():
                pick_str = (f"Rd {int(r['Draft Rd'])}, Pick {int(r['Draft Pick'])}"
                            if pd.notna(r.get('Draft Pick')) else '?')
                seasons = r.get('NFL Seasons')
                season_str = f", {int(seasons)} seasons" if pd.notna(seasons) else ""
                st.markdown(f"- **{r['Name']}** ({r['Pos']}) — {pick_str} → "
                            f"Dynasty {fmt(r['Dynasty PPG'])} PPG{season_str}")

    # Value Picks
    st.divider()
    st.markdown("### 💎 Value Picks")
    st.caption("Players drafted in rounds 3+ who exceeded 10 PPG dynasty value.")

    late = df[(df['Draft Rd'].notna()) & (df['Draft Rd'] >= 3)]
    values = late[late['Dynasty PPG'] >= 10].sort_values('Dynasty PPG', ascending=False)
    if values.empty:
        st.info("No late-round value picks found in this class with current thresholds.")
    else:
        for _, r in values.iterrows():
            pick_str = (f"Rd {int(r['Draft Rd'])}, Pick {int(r['Draft Pick'])}"
                        if pd.notna(r.get('Draft Pick')) else '?')
            arch = r.get('Archetype', '')
            arch_color = ARCHETYPE_COLORS.get(arch, '#666')
            st.markdown(
                f"- **{r['Name']}** ({r['Pos']}) — {pick_str} → "
                f"Dynasty {fmt(r['Dynasty PPG'])} PPG, "
                f"Best {fmt(r.get('Best Season PPG'))} PPG "
                f"<span class='archetype-badge' style='background:{arch_color};"
                f"font-size:0.7rem'>{arch.replace('_', ' ')}</span>",
                unsafe_allow_html=True)

    # Round-by-Round Breakdown
    st.divider()
    st.markdown("### 📊 Round-by-Round Breakdown")

    round_df = df[df['Draft Rd'].notna()].copy()
    if round_df.empty:
        st.info("No draft round data available.")
    else:
        round_agg = (
            round_df.groupby('Draft Rd')
            .agg(
                Players=('Name', 'count'),
                Avg_Rookie=('Rookie PPG', 'mean'),
                Avg_Dynasty=('Dynasty PPG', 'mean'),
                Avg_Best=('Best Season PPG', 'mean'),
                Avg_Seasons=('NFL Seasons', 'mean'),
            )
            .reset_index()
            .rename(columns={
                'Draft Rd': 'Round',
                'Avg_Rookie': 'Avg Rookie PPG',
                'Avg_Dynasty': 'Avg Dynasty PPG',
                'Avg_Best': 'Avg Best Season',
                'Avg_Seasons': 'Avg Seasons',
            })
        )
        round_agg['Round'] = round_agg['Round'].astype(int)
        round_agg = round_agg.sort_values('Round')

        rd_fmt = {c: '{:.1f}' for c in ['Avg Rookie PPG', 'Avg Dynasty PPG',
                                          'Avg Best Season', 'Avg Seasons']}
        st.dataframe(
            round_agg.style.format(rd_fmt, na_rep='—'),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart — avg dynasty PPG by round
        st.markdown("**Avg Dynasty PPG by Draft Round**")
        chart_data = round_agg[['Round', 'Avg Dynasty PPG']].copy()
        chart_data['Round'] = chart_data['Round'].apply(lambda x: f"Rd {x}")
        st.bar_chart(chart_data.set_index('Round'), use_container_width=True)

    # Archetype Performance
    st.divider()
    st.markdown("### 🧬 Archetype Performance")

    arch_df = df[df['Archetype'].notna() & (df['Archetype'] != '?')].copy()
    if arch_df.empty:
        st.info("No archetype data available for this class.")
    else:
        arch_agg = (
            arch_df.groupby('Archetype')
            .agg(
                Players=('Name', 'count'),
                Avg_Rookie=('Rookie PPG', 'mean'),
                Avg_Dynasty=('Dynasty PPG', 'mean'),
                Avg_Best=('Best Season PPG', 'mean'),
            )
            .reset_index()
            .rename(columns={
                'Avg_Rookie': 'Avg Rookie PPG',
                'Avg_Dynasty': 'Avg Dynasty PPG',
                'Avg_Best': 'Avg Best Season',
            })
            .sort_values('Avg Dynasty PPG', ascending=False)
        )

        arch_fmt = {c: '{:.1f}' for c in ['Avg Rookie PPG', 'Avg Dynasty PPG',
                                            'Avg Best Season']}
        st.dataframe(
            arch_agg.style.format(arch_fmt, na_rep='—'),
            use_container_width=True,
            hide_index=True,
        )

    # Cross-Class Comparison
    st.divider()
    st.markdown("### 📈 Cross-Class Comparison")
    st.caption("How does this class stack up against other years?")

    all_years_rows = []
    for yr in years:
        yr_players = [
            p for p in nfl_players
            if int(float(p['draft_year'])) == yr
            and p.get('position', '') in pos_filter
        ]
        if not yr_players:
            continue

        rookie_vals = [p['rookie_ppr_ppg'] for p in yr_players
                       if p.get('rookie_ppr_ppg') is not None]
        dynasty_vals = [p['dynasty_ppg'] for p in yr_players
                        if p.get('dynasty_ppg') is not None]
        best_vals = [p.get('best_season_ppr_ppg') for p in yr_players
                     if p.get('best_season_ppr_ppg') is not None]

        hit_count = sum(1 for v in dynasty_vals if v >= 12)

        all_years_rows.append({
            'Year': yr,
            'Players': len(yr_players),
            'Avg Rookie PPG': (sum(rookie_vals) / len(rookie_vals)) if rookie_vals else None,
            'Avg Dynasty PPG': (sum(dynasty_vals) / len(dynasty_vals)) if dynasty_vals else None,
            'Max Dynasty PPG': max(dynasty_vals) if dynasty_vals else None,
            'Max Best Season': max(best_vals) if best_vals else None,
            'Hits (12+ PPG)': hit_count,
            'Hit Rate': (hit_count / len(dynasty_vals)) if dynasty_vals else None,
        })

    if all_years_rows:
        all_years_df = pd.DataFrame(all_years_rows).sort_values('Year', ascending=False)

        # Highlight the selected year
        def _highlight_selected(row):
            if row['Year'] == selected_year:
                return ['background-color: rgba(102, 126, 234, 0.15)'] * len(row)
            return [''] * len(row)

        cross_fmt = {
            'Avg Rookie PPG': '{:.1f}',
            'Avg Dynasty PPG': '{:.1f}',
            'Max Dynasty PPG': '{:.1f}',
            'Max Best Season': '{:.1f}',
            'Hit Rate': '{:.0%}',
        }

        st.dataframe(
            all_years_df.style
            .format(cross_fmt, na_rep='—')
            .apply(_highlight_selected, axis=1),
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + len(all_years_df) * 35),
        )

        # Trend chart
        st.markdown("**Dynasty PPG Trend Across Classes**")
        trend_data = all_years_df[['Year', 'Avg Dynasty PPG']].dropna().copy()
        trend_data = trend_data.sort_values('Year')
        trend_data['Year'] = trend_data['Year'].astype(str)
        st.line_chart(trend_data.set_index('Year'), use_container_width=True)

        # Rank this class
        if not all_years_df.empty:
            rank_col = all_years_df.sort_values('Avg Dynasty PPG', ascending=False).reset_index(drop=True)
            rank_col['Rank'] = rank_col.index + 1
            sel_rank = rank_col[rank_col['Year'] == selected_year]
            if not sel_rank.empty:
                rank_num = sel_rank.iloc[0]['Rank']
                total = len(rank_col)
                avg_dyn = sel_rank.iloc[0].get('Avg Dynasty PPG')

                if rank_num <= 3:
                    emoji = "🥇" if rank_num == 1 else ("🥈" if rank_num == 2 else "🥉")
                else:
                    emoji = "📊"

                st.markdown(
                    f"{emoji} The **{selected_year}** class ranks **#{rank_num} of {total}** "
                    f"in average dynasty PPG ({fmt(avg_dyn)} PPG) among loaded classes."
                )