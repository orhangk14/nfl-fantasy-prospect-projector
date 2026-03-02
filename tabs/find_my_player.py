# tabs/find_my_player.py
"""Tab 7: Find My Player — pick an NFL player, find which 2026 prospects match them."""

import streamlit as st
import pandas as pd
from tabs.helpers import fmt, fmt_pct, ARCHETYPE_COLORS


def render(projections):
    st.subheader("🔎 Find My Player")
    st.caption("Pick an NFL player from the historical comp pool and see which "
               "2026 prospects are most similar to them.")

    # Build reverse lookup: for each NFL comp, which prospects matched them?
    # Structure: nfl_name -> [(prospect_name, prospect_data, similarity_score, comp_entry)]
    nfl_player_map = {}

    for pid, data in projections.items():
        prospect = data.get('prospect', {})
        comps = data.get('comparisons', [])

        for comp in comps:
            nfl_name = comp.get('name', '')
            if not nfl_name:
                continue

            if nfl_name not in nfl_player_map:
                nfl_player_map[nfl_name] = {
                    'info': comp,  # store comp info (position, archetype, etc)
                    'matches': [],
                }

            nfl_player_map[nfl_name]['matches'].append({
                'prospect_name': prospect.get('name', '?'),
                'prospect_pos': prospect.get('position', '?'),
                'prospect_school': prospect.get('school', ''),
                'prospect_archetype': prospect.get('archetype', ''),
                'similarity_score': comp.get('similarity_score', 0),
                'category_scores': comp.get('category_scores', {}),
                'prospect_data': data,
            })

    if not nfl_player_map:
        st.warning("No comparison data available.")
        return

    # Sort each NFL player's matches by similarity (highest first)
    for nfl_name in nfl_player_map:
        nfl_player_map[nfl_name]['matches'].sort(
            key=lambda x: x['similarity_score'], reverse=True
        )

    # Build selector options with position and college
    nfl_options = {}
    for nfl_name, data in nfl_player_map.items():
        info = data['info']
        pos = info.get('position', '?')
        college = info.get('college', '')
        n_matches = len(data['matches'])
        label = f"{nfl_name} ({pos}) — {college}" if college else f"{nfl_name} ({pos})"
        nfl_options[label] = nfl_name

    # Filters
    fc1, fc2 = st.columns([1, 3])
    with fc1:
        pos_filter = st.multiselect(
            "Filter by Position",
            ['QB', 'RB', 'WR', 'TE'],
            default=['QB', 'RB', 'WR', 'TE'],
            key='fmp_pos',
        )
    with fc2:
        # Filter options by position
        filtered_options = {
            label: name for label, name in nfl_options.items()
            if nfl_player_map[name]['info'].get('position', '') in pos_filter
        }

        if not filtered_options:
            st.info("No NFL players match the selected position filter.")
            return

        # Sort alphabetically
        sorted_labels = sorted(filtered_options.keys())
        selected_label = st.selectbox(
            "Select an NFL Player",
            sorted_labels,
            index=0,
            key='fmp_player',
        )

    selected_nfl_name = filtered_options[selected_label]
    nfl_data = nfl_player_map[selected_nfl_name]
    nfl_info = nfl_data['info']
    matches = nfl_data['matches']

    # ── NFL Player Card ─────────────────────────────────────────
    st.divider()

    arch = nfl_info.get('archetype', '')
    arch_color = ARCHETYPE_COLORS.get(arch, '#666')
    pos = nfl_info.get('position', '?')
    college = nfl_info.get('college', '')

    draft_rd = nfl_info.get('draft_round')
    draft_pk = nfl_info.get('draft_pick')
    draft_str = ''
    if draft_rd and draft_pk:
        try:
            draft_str = f"Round {int(float(draft_rd))}, Pick {int(float(draft_pk))}"
        except (ValueError, TypeError):
            pass

    pc1, pc2 = st.columns([1, 2])

    with pc1:
        st.markdown(f"## {selected_nfl_name}")
        st.markdown(f"**{pos}** — {college}")
        if arch:
            st.markdown(
                f"<span class='archetype-badge' style='background:{arch_color}'>"
                f"{arch.replace('_', ' ')}</span>",
                unsafe_allow_html=True,
            )
        if draft_str:
            st.markdown(f"📝 **Draft:** {draft_str}")

    with pc2:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Rookie PPG", fmt(nfl_info.get('rookie_ppr_ppg')))
        with m2:
            st.metric("Dynasty PPG", fmt(nfl_info.get('dynasty_ppg')))
        with m3:
            st.metric("Peak 3yr PPG", fmt(nfl_info.get('peak_3yr_ppr_ppg')))
        with m4:
            st.metric("Best Season", fmt(nfl_info.get('best_season_ppr_ppg')))

        seasons = nfl_info.get('nfl_seasons_played')
        if seasons:
            st.caption(f"NFL Seasons: {int(float(seasons))}")

    # ── Matching Prospects ──────────────────────────────────────
    st.divider()
    st.markdown(f"### 🎯 2026 Prospects Similar to {selected_nfl_name}")
    st.caption(f"{len(matches)} prospect(s) had {selected_nfl_name} appear in their top-10 comps. "
               f"Sorted by similarity score (highest = closest match).")

    if not matches:
        st.info("No 2026 prospects matched this player.")
        return

    # Summary table
    table_rows = []
    for match in matches:
        proj_data = match['prospect_data']
        projs = proj_data.get('projections', {})
        evaluation = proj_data.get('evaluation', {})

        rookie_ppg = projs.get('rookie', {}).get('projected')
        dynasty_ppg = projs.get('dynasty', {}).get('projected')
        peak_ppg = projs.get('peak_3yr', {}).get('projected')
        ceiling_ppg = projs.get('best_season', {}).get('projected')

        table_rows.append({
            'Prospect': match['prospect_name'],
            'Pos': match['prospect_pos'],
            'School': match['prospect_school'],
            'Archetype': match['prospect_archetype'],
            'Similarity': match['similarity_score'],
            'Rookie PPG': rookie_ppg,
            'Dynasty PPG': dynasty_ppg,
            'Peak 3yr': peak_ppg,
            'Ceiling': ceiling_ppg,
            'Tier': evaluation.get('tier', ''),
            'Bust %': evaluation.get('bust_probability'),
            'Breakout %': evaluation.get('breakout_probability'),
        })

    match_df = pd.DataFrame(table_rows)

    # Format and style
    num_fmt = {c: '{:.1f}' for c in ['Rookie PPG', 'Dynasty PPG', 'Peak 3yr', 'Ceiling']
               if c in match_df.columns}
    num_fmt['Similarity'] = '{:.1%}'
    pct_fmt = {c: '{:.0%}' for c in ['Bust %', 'Breakout %'] if c in match_df.columns}
    all_fmt = {**num_fmt, **pct_fmt}

    def _style_sim(val):
        if pd.isna(val):
            return ''
        if val >= 0.7:
            return 'color: #00C851; font-weight: bold'
        if val >= 0.5:
            return 'color: #33b5e5; font-weight: bold'
        return 'color: #ff8800'

    styler = match_df.style.format(all_fmt, na_rep='—')
    if 'Similarity' in match_df.columns:
        styler = styler.map(_style_sim, subset=['Similarity'])

    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Detailed Cards ──────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Detailed Prospect Comparisons")

    for i, match in enumerate(matches, 1):
        sim = match['similarity_score']
        sim_color = '#00C851' if sim >= 0.7 else '#33b5e5' if sim >= 0.5 else '#ff8800'
        p_arch = match['prospect_archetype']
        p_arch_color = ARCHETYPE_COLORS.get(p_arch, '#666')

        proj_data = match['prospect_data']
        projs = proj_data.get('projections', {})
        evaluation = proj_data.get('evaluation', {})

        with st.expander(
            f"#{i} — {match['prospect_name']} ({match['prospect_pos']}, "
            f"{match['prospect_school']}) — {sim:.0%} match",
            expanded=(i <= 3),
        ):
            ec1, ec2 = st.columns([1, 2])

            with ec1:
                st.markdown(
                    f"<span style='color:{sim_color};font-weight:bold;"
                    f"font-size:1.3rem'>{sim:.0%} Match</span>",
                    unsafe_allow_html=True,
                )
                if p_arch:
                    st.markdown(
                        f"<span class='archetype-badge' style='background:{p_arch_color}'>"
                        f"{p_arch.replace('_', ' ')}</span>",
                        unsafe_allow_html=True,
                    )

                # Category score breakdown
                cats = match.get('category_scores', {})
                if cats:
                    st.markdown("**Similarity Breakdown:**")
                    for cat, score in sorted(cats.items(), key=lambda x: x[1], reverse=True):
                        bar_pct = int(score * 100)
                        cat_label = cat.replace('_', ' ').title()
                        st.markdown(
                            f"<small>{cat_label}: <b>{score:.0%}</b></small> "
                            f"<div style='background:#eee;border-radius:4px;height:8px;width:100%'>"
                            f"<div style='background:{sim_color};border-radius:4px;"
                            f"height:8px;width:{bar_pct}%'></div></div>",
                            unsafe_allow_html=True,
                        )

            with ec2:
                st.markdown("**Projected Outcomes:**")
                pm1, pm2, pm3, pm4 = st.columns(4)
                with pm1:
                    st.metric("Rookie PPG",
                              fmt(projs.get('rookie', {}).get('projected')))
                with pm2:
                    st.metric("Dynasty PPG",
                              fmt(projs.get('dynasty', {}).get('projected')))
                with pm3:
                    st.metric("Peak 3yr",
                              fmt(projs.get('peak_3yr', {}).get('projected')))
                with pm4:
                    st.metric("Ceiling",
                              fmt(projs.get('best_season', {}).get('projected')))

                tier = evaluation.get('tier', '—')
                bust = evaluation.get('bust_probability')
                breakout = evaluation.get('breakout_probability')
                st.markdown(
                    f"**Tier:** {tier} · "
                    f"**Bust:** {fmt_pct(bust)} · "
                    f"**Breakout:** {fmt_pct(breakout)}"
                )

                # Show how this NFL player compares to prospect's OTHER comps
                other_comps = proj_data.get('comparisons', [])
                comp_names = [c['name'] for c in other_comps[:10]]
                if selected_nfl_name in comp_names:
                    rank = comp_names.index(selected_nfl_name) + 1
                    st.caption(
                        f"{selected_nfl_name} is comp #{rank} of 10 for "
                        f"{match['prospect_name']}"
                    )

            st.markdown("---")

    # ── Fun Section: Who is the BEST match? ─────────────────────
    if len(matches) >= 2:
        st.divider()
        best = matches[0]
        st.markdown(
            f"### 🏆 The Next {selected_nfl_name}?")
        st.markdown(
            f"Based on historical similarity matching, **{best['prospect_name']}** "
            f"({best['prospect_pos']}, {best['prospect_school']}) is the closest "
            f"2026 prospect to {selected_nfl_name}'s pre-NFL profile at "
            f"**{best['similarity_score']:.0%}** similarity."
        )

        if best['similarity_score'] >= 0.7:
            st.success("🔥 Very strong match — similar production, measurables, and draft capital.")
        elif best['similarity_score'] >= 0.5:
            st.info("📊 Solid match — meaningful similarities with some differences.")
        else:
            st.warning("⚠️ Moderate match — there are similarities but also significant gaps. "
                       "Take the comparison with a grain of salt.")