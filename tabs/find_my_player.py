# tabs/find_my_player.py
"""Tab 7: Find My Player — pick any NFL player, compute live similarity against all 2026 prospects."""

import streamlit as st
import pandas as pd
from tabs.helpers import fmt, fmt_pct, ARCHETYPE_COLORS, get_tier_color


def _build_nfl_roster(profiles):
    """Get all NFL players (non-prospects) with outcomes."""
    players = []
    for p in profiles:
        if p.get('is_prospect'):
            continue
        if p.get('rookie_ppr_ppg') is None:
            continue
        players.append(p)
    return players


def _build_prospect_list(profiles):
    """Get all 2026 prospects."""
    return [p for p in profiles if p.get('is_prospect')]


def _compute_similarities(nfl_player, prospects, pos_features, compute_feature_stats, compute_similarity):
    """
    Run similarity engine: compare one NFL player against all prospects
    of the same position. Returns sorted list of (prospect, score, cat_scores).
    """
    pos = nfl_player.get('position', '')
    features = pos_features.get(pos, [])
    if not features:
        return []

    pos_prospects = [p for p in prospects if p.get('position') == pos]
    if not pos_prospects:
        return []

    # Build stat pool from NFL player + all same-position prospects
    all_pool = pos_prospects + [nfl_player]
    feat_stats = compute_feature_stats(all_pool, features)

    results = []
    for prospect in pos_prospects:
        score, feat_comp, cat_scores = compute_similarity(
            nfl_player, prospect, features, feat_stats
        )
        results.append({
            'prospect': prospect,
            'similarity_score': round(score, 4),
            'category_scores': {k: round(v, 4) for k, v in cat_scores.items()},
            'feature_comparison': feat_comp,
        })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results


def render(projections, load_profiles_fn):
    st.subheader("🔎 Find My Player")
    st.caption("Pick any NFL player from the historical database and find which "
               "2026 prospects most closely match their pre-NFL profile.")

    # Load and enrich profiles
    from modeling.similarity import (
        POS_FEATURES, compute_feature_stats, compute_similarity, enrich_profiles
    )

    profiles = load_profiles_fn()
    profiles = enrich_profiles(profiles)

    nfl_players = _build_nfl_roster(profiles)
    prospects = _build_prospect_list(profiles)

    if not nfl_players:
        st.warning("No historical NFL player data available.")
        return
    if not prospects:
        st.warning("No 2026 prospect data available.")
        return

    # ── Filters & Selection ─────────────────────────────────────
    fc1, fc2 = st.columns([1, 3])

    with fc1:
        pos_filter = st.multiselect(
            "Position",
            ['QB', 'RB', 'WR', 'TE'],
            default=['QB', 'RB', 'WR', 'TE'],
            key='fmp_pos',
        )

    # Build selector options
    filtered_nfl = [p for p in nfl_players if p.get('position', '') in pos_filter]

    if not filtered_nfl:
        st.info("No NFL players match the selected position filter.")
        return

    # Build display labels sorted alphabetically
    player_labels = {}
    for p in filtered_nfl:
        name = p.get('name', '?')
        pos = p.get('position', '?')
        college = p.get('college', p.get('school', ''))
        dynasty = p.get('dynasty_ppg')
        dynasty_str = f" — {dynasty:.1f} dynasty PPG" if dynasty else ""
        label = f"{name} ({pos}, {college}){dynasty_str}"
        player_labels[label] = p

    sorted_labels = sorted(player_labels.keys())

    with fc2:
        # Try to default to a recognizable name
        default_idx = 0
        for i, label in enumerate(sorted_labels):
            if 'Ja\'Marr Chase' in label or 'Mike Evans' in label or 'Bucky Irving' in label:
                default_idx = i
                break

        selected_label = st.selectbox(
            "Select an NFL Player",
            sorted_labels,
            index=default_idx,
            key='fmp_player',
        )

    selected_nfl = player_labels[selected_label]
    nfl_name = selected_nfl.get('name', '?')
    nfl_pos = selected_nfl.get('position', '?')

    # ── NFL Player Card ─────────────────────────────────────────
    st.divider()

    arch = selected_nfl.get('archetype', '')
    arch_color = ARCHETYPE_COLORS.get(arch, '#666')
    college = selected_nfl.get('college', selected_nfl.get('school', ''))

    draft_rd = selected_nfl.get('draft_round')
    draft_pk = selected_nfl.get('draft_pick')
    draft_yr = selected_nfl.get('draft_year')
    draft_str = ''
    if draft_rd and draft_pk:
        try:
            draft_str = f"Round {int(float(draft_rd))}, Pick {int(float(draft_pk))}"
            if draft_yr:
                draft_str += f" ({int(float(draft_yr))})"
        except (ValueError, TypeError):
            pass

    pc1, pc2 = st.columns([1, 2])

    with pc1:
        st.markdown(f"## {nfl_name}")
        st.markdown(f"**{nfl_pos}** — {college}")
        if arch:
            st.markdown(
                f"<span class='archetype-badge' style='background:{arch_color}'>"
                f"{arch.replace('_', ' ')}</span>",
                unsafe_allow_html=True,
            )
        if draft_str:
            st.markdown(f"📝 **Draft:** {draft_str}")

        # Measurables
        ht = selected_nfl.get('combine_height')
        wt = selected_nfl.get('combine_weight')
        forty = selected_nfl.get('combine_40')
        meas_parts = []
        if ht:
            ft = int(ht // 12)
            inch = int(ht % 12)
            meas_parts.append(f"{ft}'{inch}\"")
        if wt:
            meas_parts.append(f"{int(wt)} lbs")
        if forty:
            meas_parts.append(f"{forty}s 40")
        if meas_parts:
            st.caption(" · ".join(meas_parts))

    with pc2:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Rookie PPG", fmt(selected_nfl.get('rookie_ppr_ppg')))
        with m2:
            st.metric("Dynasty PPG", fmt(selected_nfl.get('dynasty_ppg')))
        with m3:
            st.metric("Peak 3yr", fmt(selected_nfl.get('peak_3yr_ppr_ppg')))
        with m4:
            st.metric("Best Season", fmt(selected_nfl.get('best_season_ppr_ppg')))
        with m5:
            seasons = selected_nfl.get('nfl_seasons_played')
            st.metric("Seasons", int(float(seasons)) if seasons else '—')

        # College stats for context
        college_ppg = selected_nfl.get('college_ppr_ppg')
        peak_ppg = selected_nfl.get('peak_ppr_ppg')
        if college_ppg or peak_ppg:
            st.caption(
                f"College: {fmt(college_ppg)} PPG avg · {fmt(peak_ppg)} PPG peak"
            )

    # ── Run Live Similarity ─────────────────────────────────────
    st.divider()
    st.markdown(f"### 🎯 2026 Prospects Most Similar to {nfl_name}")

    with st.spinner(f"Computing similarity for {nfl_name} against all {nfl_pos} prospects..."):
        matches = _compute_similarities(
            selected_nfl, prospects,
            POS_FEATURES, compute_feature_stats, compute_similarity
        )

    if not matches:
        st.info(f"No {nfl_pos} prospects found in the 2026 class.")
        return

    # How many to show
    n_show = st.slider("Number of matches to show", 3, min(20, len(matches)),
                       min(10, len(matches)), key='fmp_n')

    top_matches = matches[:n_show]

    # ── Summary Table ───────────────────────────────────────────
    table_rows = []
    for match in top_matches:
        prospect = match['prospect']
        pid = str(prospect.get('espn_id', ''))

        # Get projection data if available
        proj_data = projections.get(pid, {})
        projs = proj_data.get('projections', {})
        evaluation = proj_data.get('evaluation', {})

        table_rows.append({
            'Prospect': prospect.get('name', '?'),
            'School': prospect.get('school', ''),
            'Archetype': prospect.get('archetype', ''),
            'Similarity': match['similarity_score'],
            'Rookie PPG': projs.get('rookie', {}).get('projected'),
            'Dynasty PPG': projs.get('dynasty', {}).get('projected'),
            'Peak 3yr': projs.get('peak_3yr', {}).get('projected'),
            'Ceiling': projs.get('best_season', {}).get('projected'),
            'Tier': evaluation.get('tier', ''),
            'Bust %': evaluation.get('bust_probability'),
            'Breakout %': evaluation.get('breakout_probability'),
        })

    match_df = pd.DataFrame(table_rows)

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
    if 'Tier' in match_df.columns:
        from tabs.helpers import style_tier
        styler = styler.map(style_tier, subset=['Tier'])

    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Detailed Cards ──────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Detailed Comparisons")

    for i, match in enumerate(top_matches, 1):
        prospect = match['prospect']
        sim = match['similarity_score']
        sim_color = '#00C851' if sim >= 0.7 else '#33b5e5' if sim >= 0.5 else '#ff8800'
        p_arch = prospect.get('archetype', '')
        p_arch_color = ARCHETYPE_COLORS.get(p_arch, '#666')
        p_name = prospect.get('name', '?')
        p_school = prospect.get('school', '')
        pid = str(prospect.get('espn_id', ''))

        proj_data = projections.get(pid, {})
        projs = proj_data.get('projections', {})
        evaluation = proj_data.get('evaluation', {})

        with st.expander(
            f"#{i} — {p_name} ({p_school}) — {sim:.0%} match",
            expanded=(i <= 3),
        ):
            ec1, ec2 = st.columns([1, 2])

            with ec1:
                st.markdown(
                    f"<span style='color:{sim_color};font-weight:bold;"
                    f"font-size:1.4rem'>{sim:.0%} Match</span>",
                    unsafe_allow_html=True,
                )
                if p_arch:
                    st.markdown(
                        f"<span class='archetype-badge' style='background:{p_arch_color}'>"
                        f"{p_arch.replace('_', ' ')}</span>",
                        unsafe_allow_html=True,
                    )

                # Category breakdown with visual bars
                cats = match.get('category_scores', {})
                if cats:
                    st.markdown("**Similarity Breakdown:**")
                    for cat, score in sorted(cats.items(), key=lambda x: x[1], reverse=True):
                        bar_pct = max(int(score * 100), 2)
                        cat_label = cat.replace('_', ' ').title()
                        st.markdown(
                            f"<small>{cat_label}: <b>{score:.0%}</b></small>"
                            f"<div style='background:#eee;border-radius:4px;height:8px;"
                            f"width:100%;margin-bottom:4px'>"
                            f"<div style='background:{sim_color};border-radius:4px;"
                            f"height:8px;width:{bar_pct}%'></div></div>",
                            unsafe_allow_html=True,
                        )

                # Prospect measurables
                p_ht = prospect.get('combine_height')
                p_wt = prospect.get('combine_weight')
                p_40 = prospect.get('combine_40')
                meas = []
                if p_ht:
                    meas.append(f"{int(p_ht // 12)}'{int(p_ht % 12)}\"")
                if p_wt:
                    meas.append(f"{int(p_wt)} lbs")
                if p_40:
                    meas.append(f"{p_40}s")
                if meas:
                    st.caption(" · ".join(meas))

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
                tier_color = get_tier_color(tier)
                bust = evaluation.get('bust_probability')
                breakout = evaluation.get('breakout_probability')
                st.markdown(
                    f"**Tier:** <span style='color:{tier_color};font-weight:bold'>"
                    f"{tier}</span> · "
                    f"**Bust:** {fmt_pct(bust)} · "
                    f"**Breakout:** {fmt_pct(breakout)}",
                    unsafe_allow_html=True,
                )

                # Feature comparison table
                feat_comp = match.get('feature_comparison', {})
                if feat_comp:
                    with st.popover("📊 Feature-by-Feature Comparison"):
                        feat_rows = []
                        for fk, vals in feat_comp.items():
                            label = (fk.replace('college_', '').replace('combine_', '')
                                     .replace('peak_', 'pk_').replace('_', ' ').title())
                            pv = vals.get('prospect')  # this is actually the NFL player
                            cv = vals.get('comp')       # this is the prospect
                            feat_rows.append({
                                'Feature': label,
                                nfl_name: fmt(pv),
                                p_name: fmt(cv),
                                'Diff %': f"{vals.get('diff_pct', 0):.0f}%",
                            })
                        if feat_rows:
                            st.dataframe(
                                pd.DataFrame(feat_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

    # ── "The Next ___" callout ──────────────────────────────────
    if top_matches:
        st.divider()
        best = top_matches[0]
        best_name = best['prospect'].get('name', '?')
        best_school = best['prospect'].get('school', '')
        best_sim = best['similarity_score']

        st.markdown(f"### 🏆 The Next {nfl_name}?")
        st.markdown(
            f"Based on live similarity matching across college production, "
            f"measurables, and draft capital, **{best_name}** ({best_school}) "
            f"is the closest 2026 prospect to {nfl_name}'s pre-NFL profile "
            f"at **{best_sim:.0%}** similarity."
        )

        if best_sim >= 0.7:
            st.success(
                "🔥 Very strong match — similar production, measurables, and draft capital."
            )
        elif best_sim >= 0.5:
            st.info(
                "📊 Solid match — meaningful similarities with some differences."
            )
        else:
            st.warning(
                "⚠️ Moderate match — there are similarities but also significant gaps. "
                "Take the comparison with a grain of salt."
            )

        # Compare career outcomes of NFL player to prospect's projection
        nfl_dynasty = selected_nfl.get('dynasty_ppg')
        pid = str(best['prospect'].get('espn_id', ''))
        proj_data = projections.get(pid, {})
        proj_dynasty = proj_data.get('projections', {}).get('dynasty', {}).get('projected')

        if nfl_dynasty and proj_dynasty:
            diff = proj_dynasty - nfl_dynasty
            direction = "higher" if diff > 0 else "lower"
            st.caption(
                f"{nfl_name}'s actual dynasty PPG: {nfl_dynasty:.1f} · "
                f"{best_name}'s projected dynasty PPG: {proj_dynasty:.1f} "
                f"({abs(diff):.1f} PPG {direction})"
            )