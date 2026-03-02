# tabs/custom_prospect.py
"""Tab 5: Build Custom Prospect — input stats to generate projections."""

import streamlit as st
import pandas as pd
from tabs.helpers import (
    render_projection_cards, render_eval_row, render_comp_row
)


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
            'rookie': rookie, 'dynasty': dynasty,
            'peak_3yr': peak, 'best_season': best_season,
            'career_length': career,
        },
        'evaluation': {
            'tier': tier, 'bust_probability': bust_prob,
            'breakout_probability': breakout_prob,
        },
    }


def render(load_profiles_fn):
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
        measurables['combine_height'] = st.number_input("Height (inches)", 64.0, 82.0, ht_defaults[custom_pos], 0.5, key='m_ht')
        measurables['combine_weight'] = st.number_input("Weight (lbs)", 150.0, 280.0, wt_defaults[custom_pos], 5.0, key='m_wt')
        measurables['combine_40'] = st.number_input("40-Yard Dash", 4.20, 5.20, forty_defaults[custom_pos], 0.01, key='m_40')
        measurables['combine_vert'] = st.number_input("Vertical Leap (in)", 0.0, 48.0, 35.0, 0.5, key='m_vert')
        measurables['combine_broad'] = st.number_input("Broad Jump (in)", 0.0, 145.0, 120.0, 1.0, key='m_broad')
        measurables['combine_shuttle'] = st.number_input("Shuttle (seconds)", 0.0, 5.00, 4.30, 0.01, key='m_shuttle')
        measurables['combine_3cone'] = st.number_input("3-Cone (seconds)", 0.0, 8.00, 7.00, 0.01, key='m_3cone')
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
            all_profiles = load_profiles_fn()
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