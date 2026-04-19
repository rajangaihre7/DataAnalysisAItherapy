"""
EDA — Section 6: Inter-Rater Reliability (ICC)
===============================================
Excludes Participant 102 (Therapist-only, no Parent pairs).

Sub-sections:
  6.1  Paired Observations Summary
  6.2/6.3  ICC Results Table (ICC3 — two-way mixed, single measures)
  6.4  Kappa for Non-Quantitative Items (Q12 weighted, Q16/Q17 Cohen's)
  6.5  ICC Visualisations (bar chart, forest plot, domain-grouped)
"""

import os
import sys
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    QUESTION_MAPPING,
    REVERSE_QUESTION_MAPPING,
    ITEM_DESC,
    QUANT_ITEMS,
)

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except ImportError:
    _HAS_PINGOUIN = False

# ── Constants ─────────────────────────────────────────────────────────────────

EXCLUDED_PARTICIPANT = 128

DOMAINS = {
    # Q1–Q6
    'Participation & Engagement':           ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'],
    # Q7–Q17 (quantitative items only: Q12/Q15/Q16/Q17 are non-numeric)
    'Observed Behaviour & Emotions':        ['Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14'],
    # Q18–Q26 (quantitative items only: no Q19)
    'Response to Story Theme / Learning':   ['Q18', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26'],
}

_INTERP_COLOURS = {
    'Poor':      '#d62728',
    'Fair':      '#ff7f0e',
    'Good':      '#2ca02c',
    'Excellent': '#1f77b4',
    'N/A':       '#aaaaaa',
}

_INTERP_BG = {
    'Poor':      'background-color: #ffd7d7',
    'Fair':      'background-color: #fff3cc',
    'Good':      'background-color: #d4edda',
    'Excellent': 'background-color: #cce5ff',
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def _get_quant_cols(df):
    """Return list of (column_name, q_label) for quantitative items present in df."""
    return [
        (col, q)
        for col, q in QUESTION_MAPPING.items()
        if q in QUANT_ITEMS and col in df.columns
    ]


def _icc_label(icc):
    if pd.isna(icc):
        return 'N/A'
    if icc < 0.40:
        return 'Poor'
    if icc < 0.60:
        return 'Fair'
    if icc < 0.75:
        return 'Good'
    return 'Excellent'


def _kappa_label(k):
    if pd.isna(k):
        return 'N/A'
    if k < 0.00:
        return 'Poor (< 0)'
    if k < 0.20:
        return 'Slight'
    if k < 0.40:
        return 'Fair'
    if k < 0.60:
        return 'Moderate'
    if k < 0.80:
        return 'Substantial'
    return 'Almost Perfect'


def _domain_for(q_label):
    for domain, items in DOMAINS.items():
        if q_label in items:
            return domain
    return 'Other'


# ── ICC computation ───────────────────────────────────────────────────────────

def _build_paired_long(df_ex, col):
    """
    Build long-format DataFrame for ICC:
      Participant_Session | Observer_Type | Score
    Only retains targets (participant × session) that have BOTH a T and P rating.
    Deduplicates by averaging if multiple rows exist per target × rater.
    """
    sub = df_ex[['Participant id', 'Session number', 'Submitted_by', col]].copy()
    sub['Score'] = pd.to_numeric(sub[col], errors='coerce')
    sub = sub.dropna(subset=['Score'])
    sub['Participant_Session'] = (
        sub['Participant id'].astype(str) + '_' + sub['Session number'].astype(str)
    )
    # Deduplicate: mean per target × observer
    sub = (
        sub.groupby(['Participant_Session', 'Submitted_by'])['Score']
        .mean()
        .reset_index()
        .rename(columns={'Submitted_by': 'Observer_Type'})
    )
    # Keep only targets with BOTH T and P
    pivot_check = sub.pivot_table(
        index='Participant_Session', columns='Observer_Type', values='Score', aggfunc='first'
    )
    missing = [c for c in ['T', 'P'] if c not in pivot_check.columns]
    if missing:
        return sub.iloc[0:0].reset_index(drop=True)  # empty — no pairs possible
    valid = pivot_check.dropna(subset=['T', 'P']).index
    return sub[sub['Participant_Session'].isin(valid)].reset_index(drop=True)


def _compute_icc(paired_df):
    """
    Compute ICC3 (two-way mixed, single measures) using pingouin.
    Returns (icc, ci_low, ci_high, pval, n_pairs, error_msg).
    """
    n_pairs = paired_df['Participant_Session'].nunique()
    if not _HAS_PINGOUIN:
        return np.nan, np.nan, np.nan, np.nan, n_pairs, 'pingouin not installed'
    if n_pairs < 3:
        return np.nan, np.nan, np.nan, np.nan, n_pairs, f'too few pairs ({n_pairs})'
    # Drop remaining NaN scores before passing to pingouin
    clean = paired_df.dropna(subset=['Score']).copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = pg.intraclass_corr(
                data=clean,
                targets='Participant_Session',
                raters='Observer_Type',
                ratings='Score',
            )
        # Pingouin version-safe lookup: try several possible Type strings for ICC3
        _icc3_candidates = ['ICC3', 'ICC3,1', 'ICC(3,1)', 'ICC3k']
        row = None
        for candidate in _icc3_candidates:
            match = result[result['Type'] == candidate]
            if not match.empty:
                row = match.iloc[0]
                break
        # Fallback: use ICC2 (two-way random) which is the next best
        if row is None:
            for candidate in ['ICC2', 'ICC2,1', 'ICC(2,1)']:
                match = result[result['Type'] == candidate]
                if not match.empty:
                    row = match.iloc[0]
                    break
        # Last resort: take whichever row has the highest single-measures ICC
        if row is None:
            if result.empty:
                return np.nan, np.nan, np.nan, np.nan, n_pairs, \
                    f'pingouin returned empty result; types={[]}'
            row = result.iloc[result['ICC'].abs().argmax()]
        # CI95% format varies by pingouin version:
        #   older  → 'CI95%' column holds a list/tuple [lo, hi]
        #   newer  → separate 'CI95%_lower' / 'CI95%_upper' columns
        all_cols = result.columns.tolist()
        if 'CI95%_lower' in all_cols and 'CI95%_upper' in all_cols:
            ci_lo, ci_hi = float(row['CI95%_lower']), float(row['CI95%_upper'])
        elif 'CI95%' in all_cols:
            ci = row['CI95%']
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
        else:
            lower_col = next((c for c in all_cols if 'lower' in c.lower()), None)
            upper_col = next((c for c in all_cols if 'upper' in c.lower()), None)
            ci_lo = float(row[lower_col]) if lower_col else np.nan
            ci_hi = float(row[upper_col]) if upper_col else np.nan
        return float(row['ICC']), ci_lo, ci_hi, float(row['pval']), n_pairs, ''
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, n_pairs, str(e)


def _build_icc_table(df_ex):
    """Compute ICC3 for all 21 quantitative items. df_ex must already exclude P102."""
    rows = []
    for col, q_label in _get_quant_cols(df_ex):
        paired = _build_paired_long(df_ex, col)
        icc, ci_lo, ci_hi, pval, n, err = _compute_icc(paired)
        rows.append({
            'Q':              q_label,
            'Description':    ITEM_DESC.get(q_label, q_label),
            'Domain':         _domain_for(q_label),
            'Paired N':       n,
            'ICC':            icc,
            'CI Lower':       ci_lo,
            'CI Upper':       ci_hi,
            'p-value':        pval,
            'Interpretation': _icc_label(icc),
            '_error':         err,
        })
    return pd.DataFrame(rows).sort_values('Q').reset_index(drop=True)


# ── Kappa computation ─────────────────────────────────────────────────────────

def _quadratic_weighted_kappa(a, b, n_cats=3):
    """Quadratic weighted kappa (fallback when sklearn not installed)."""
    a, b = np.array(a, dtype=int), np.array(b, dtype=int)
    cats = np.arange(n_cats)
    w = 1 - ((np.subtract.outer(cats, cats) / max(n_cats - 1, 1)) ** 2)
    obs = np.zeros((n_cats, n_cats))
    for ai, bi in zip(a, b):
        if 0 <= ai < n_cats and 0 <= bi < n_cats:
            obs[ai, bi] += 1
    n = obs.sum()
    if n == 0:
        return np.nan
    obs /= n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    po = (obs * w).sum()
    pe = (exp * w).sum()
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _simple_cohen_kappa(a, b):
    """Cohen's kappa (fallback when sklearn not installed)."""
    a, b = np.array(a, dtype=int), np.array(b, dtype=int)
    n = len(a)
    if n == 0:
        return np.nan
    classes = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in classes)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _pivot_for_kappa(df_ex, col, value_fn):
    """
    Shared helper: map column values via value_fn, pivot to T/P,
    return (pivot, value_counts_of_raw) or (None, {}) on failure.
    """
    sub = df_ex[['Participant id', 'Session number', 'Submitted_by', col]].dropna(
        subset=[col]
    ).copy()
    sub['mapped'] = sub[col].apply(value_fn)
    sub = sub.dropna(subset=['mapped'])
    value_counts = df_ex[col].value_counts().to_dict()
    pivot = sub.pivot_table(
        index=['Participant id', 'Session number'],
        columns='Submitted_by',
        values='mapped',
        aggfunc='first',
    )
    if 'T' not in pivot.columns or 'P' not in pivot.columns:
        return None, value_counts
    pivot = pivot.dropna(subset=['T', 'P'], how='any')
    if len(pivot) < 3:
        return None, value_counts
    return pivot, value_counts


def _compute_q12_kappa(df_ex):
    """Weighted (quadratic) kappa for Q12 (ordinal: Not Improved=0 / Same=1 / Improved=2)."""
    col = REVERSE_QUESTION_MAPPING.get('Q12', '')
    if col not in df_ex.columns:
        return np.nan, 0, {}

    def _ordinal(v):
        v = str(v).strip().lower()
        if any(x in v for x in ('not improv', 'worsen', 'negative', 'no improve', 'not improved')):
            return 0
        if any(x in v for x in ('same', 'no change', 'neutral', 'unchanged', 'stayed')):
            return 1
        if any(x in v for x in ('improv', 'better', 'positive', 'improved')):
            return 2
        return np.nan

    pivot, vc = _pivot_for_kappa(df_ex, col, _ordinal)
    if pivot is None:
        return np.nan, 0, vc

    t_vals = pivot['T'].astype(int).tolist()
    p_vals = pivot['P'].astype(int).tolist()
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(t_vals, p_vals, weights='quadratic')
    except ImportError:
        kappa = _quadratic_weighted_kappa(t_vals, p_vals, n_cats=3)
    return kappa, len(pivot), vc


def _compute_binary_kappa(df_ex, q_label):
    """Cohen's kappa for binary items Q16 or Q17 (Yes=1 / No=0)."""
    col = REVERSE_QUESTION_MAPPING.get(q_label, '')
    if col not in df_ex.columns:
        return np.nan, 0, {}

    def _bin(v):
        v = str(v).strip().lower()
        if v in ('yes', '1', 'true', 'y'):
            return 1
        if v in ('no', '0', 'false', 'n'):
            return 0
        return np.nan

    pivot, vc = _pivot_for_kappa(df_ex, col, _bin)
    if pivot is None:
        return np.nan, 0, vc

    t_vals = pivot['T'].astype(int).tolist()
    p_vals = pivot['P'].astype(int).tolist()
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(t_vals, p_vals)
    except ImportError:
        kappa = _simple_cohen_kappa(t_vals, p_vals)
    return kappa, len(pivot), vc


# ── Section renderers ─────────────────────────────────────────────────────────

def _section_6_1(df, df_ex):
    st.subheader("6.1  Paired Observations Summary")
    st.markdown(
        f"**Participant {EXCLUDED_PARTICIPANT} is excluded** — Therapist-only record "
        f"(no Parent submissions), so no T/P pairs can be formed.  \n"
        f"Pairing rule: one T score **and** one P score for the same Participant × Session."
    )

    # Per-participant observer breakdown
    all_pids = sorted(df['Participant id'].unique())
    t_cnt = df[df['Submitted_by'] == 'T'].groupby('Participant id').size()
    p_cnt = df[df['Submitted_by'] == 'P'].groupby('Participant id').size()
    pid_summary = pd.DataFrame({
        'Participant':    all_pids,
        'T records':      [int(t_cnt.get(pid, 0)) for pid in all_pids],
        'P records':      [int(p_cnt.get(pid, 0)) for pid in all_pids],
        'Included in ICC': ['✓' if pid != EXCLUDED_PARTICIPANT else '✗ (excluded)' for pid in all_pids],
    })
    st.dataframe(pid_summary, use_container_width=True)

    # Per-item pairing summary
    st.markdown("---")
    st.markdown("**Paired observations available per quantitative item:**")
    rows = []
    for col, q in _get_quant_cols(df_ex):
        paired = _build_paired_long(df_ex, col)
        rows.append({
            'Q':             q,
            'Description':   ITEM_DESC.get(q, q),
            'Domain':        _domain_for(q),
            'Paired (P×S)':  paired['Participant_Session'].nunique(),
        })
    pair_df = pd.DataFrame(rows).sort_values('Q').reset_index(drop=True)
    st.dataframe(pair_df, use_container_width=True)


def _section_6_3(icc_table):
    st.subheader("6.2 / 6.3  ICC Results Table")

    # Legend badges inline
    st.markdown(
        "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px'>"
        "<span style='background:#d62728;color:white;padding:3px 10px;border-radius:12px;font-size:13px'>Poor &lt; 0.40</span>"
        "<span style='background:#ff7f0e;color:white;padding:3px 10px;border-radius:12px;font-size:13px'>Fair 0.40–0.59</span>"
        "<span style='background:#2ca02c;color:white;padding:3px 10px;border-radius:12px;font-size:13px'>Good 0.60–0.74</span>"
        "<span style='background:#1f77b4;color:white;padding:3px 10px;border-radius:12px;font-size:13px'>Excellent ≥ 0.75</span>"
        "</div>"
        "<small>ICC3 — Two-way mixed model, single measures. Raters fixed: Therapist and Parent.</small>",
        unsafe_allow_html=True,
    )

    if not _HAS_PINGOUIN:
        st.error(
            "`pingouin` is not installed. ICC cannot be computed.  \n"
            "Run `pip install pingouin` in your virtual environment and restart the app."
        )
        return

    # Show any ICC computation errors
    errors = icc_table[icc_table['_error'].str.len() > 0][['Q', 'Description', 'Paired N', '_error']]
    if not errors.empty:
        with st.expander(f"⚠️ {len(errors)} item(s) could not be computed — click to see errors"):
            st.dataframe(errors.rename(columns={'_error': 'Error'}), use_container_width=True)

    # ── Summary metric cards ──────────────────────────────────────────────────
    counts = icc_table['Interpretation'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Excellent (≥ 0.75)", counts.get('Excellent', 0))
    c2.metric("Good (0.60–0.74)",   counts.get('Good',      0))
    c3.metric("Fair (0.40–0.59)",   counts.get('Fair',      0))
    c4.metric("Poor (< 0.40)",      counts.get('Poor',      0))
    st.markdown("---")

    # ── ICC bar chart (compact inline visual) ────────────────────────────────
    valid = icc_table.dropna(subset=['ICC']).copy()
    if not valid.empty:
        sdf = valid.sort_values('ICC', ascending=True).reset_index(drop=True)
        colours = [_INTERP_COLOURS[_icc_label(v)] for v in sdf['ICC']]
        y_labels = sdf['Q'] + ' — ' + sdf['Description'].str[:38]

        fig, ax = plt.subplots(figsize=(9, max(4, len(sdf) * 0.38)))
        bars = ax.barh(y_labels, sdf['ICC'], color=colours, edgecolor='white', height=0.68)

        for thresh, ls in [(0.40, '--'), (0.60, '-.'), (0.75, ':')]:
            ax.axvline(thresh, color='#555', linestyle=ls, linewidth=1.1, alpha=0.8)

        for bar, val in zip(bars, sdf['ICC']):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=7.5,
            )

        ax.set_xlim(0, 1.12)
        ax.set_xlabel('ICC3', fontsize=9)
        ax.set_title('ICC3 by Item (sorted ascending)', fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Full results table ────────────────────────────────────────────────────
    disp = icc_table.copy()
    disp['ICC']      = disp['ICC'].map(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
    disp['CI Lower'] = disp['CI Lower'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    disp['CI Upper'] = disp['CI Upper'].map(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    disp['95% CI']   = disp.apply(
        lambda r: f"[{r['CI Lower']}, {r['CI Upper']}]" if r['CI Lower'] else 'N/A', axis=1
    )
    disp['p-value']  = disp['p-value'].map(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    disp['Sig.']     = disp['p-value'].map(
        lambda x: '✓' if x not in ('N/A', '') and float(x) < 0.05 else ''
    )

    # Colour-coded Interpretation badge via cell styling
    _BADGE_CSS = {
        'Poor':      'background-color:#ffd7d7;color:#7b0000;font-weight:600',
        'Fair':      'background-color:#fff3cc;color:#7a5200;font-weight:600',
        'Good':      'background-color:#d4edda;color:#155724;font-weight:600',
        'Excellent': 'background-color:#cce5ff;color:#004085;font-weight:600',
    }

    def _style_table(df_s):
        styles = pd.DataFrame('', index=df_s.index, columns=df_s.columns)
        for i, interp in enumerate(df_s['Interpretation']):
            css = _BADGE_CSS.get(interp, '')
            if css:
                styles.loc[i, 'Interpretation'] = css
                styles.loc[i, 'ICC'] = css.replace('font-weight:600', 'font-weight:700')
        return styles

    out_cols = ['Q', 'Description', 'Domain', 'Paired N', 'ICC', '95% CI', 'p-value', 'Sig.', 'Interpretation']
    st.dataframe(
        disp[out_cols].style.apply(_style_table, axis=None),
        use_container_width=True,
        height=min(800, 45 + len(disp) * 38),
    )

    # ── Domain breakdown ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Domain averages**")
    domain_avg = (
        icc_table.dropna(subset=['ICC'])
        .groupby('Domain')['ICC']
        .agg(Mean='mean', Min='min', Max='max', N='count')
        .reset_index()
        .sort_values('Mean', ascending=False)
    )
    domain_avg['Interpretation'] = domain_avg['Mean'].apply(_icc_label)
    for c in ['Mean', 'Min', 'Max']:
        domain_avg[c] = domain_avg[c].map(lambda x: f"{x:.3f}")
    st.dataframe(
        domain_avg.rename(columns={'Mean': 'Mean ICC', 'Min': 'Min ICC',
                                   'Max': 'Max ICC', 'N': 'N items'}),
        use_container_width=True,
    )


def _section_6_4(df_ex):
    st.subheader("6.4  Kappa for Non-Quantitative Items")
    st.markdown(
        "**Q12** uses a 3-category ordinal response (Not Improved / Stayed Same / Improved) — "
        "**Quadratic Weighted Kappa** applied.  \n"
        "**Q16 / Q17** are binary Yes/No items — **Cohen's Kappa** applied.  \n\n"
        "Scale: < 0.20 Slight · 0.20–0.39 Fair · 0.40–0.59 Moderate · "
        "0.60–0.79 Substantial · ≥ 0.80 Almost Perfect"
    )

    k12, n12, vc12 = _compute_q12_kappa(df_ex)
    k16, n16, vc16 = _compute_binary_kappa(df_ex, 'Q16')
    k17, n17, vc17 = _compute_binary_kappa(df_ex, 'Q17')

    kappa_df = pd.DataFrame([
        {
            'Item':           'Q12',
            'Description':    ITEM_DESC.get('Q12', 'Retelling story at home impact'),
            'Kappa Type':     'Weighted (Quadratic)',
            'N pairs':        n12,
            'Kappa':          f"{k12:.3f}" if pd.notna(k12) else 'N/A',
            'Interpretation': _kappa_label(k12),
        },
        {
            'Item':           'Q16',
            'Description':    'Response time decreased from last session',
            'Kappa Type':     "Cohen's Kappa (binary)",
            'N pairs':        n16,
            'Kappa':          f"{k16:.3f}" if pd.notna(k16) else 'N/A',
            'Interpretation': _kappa_label(k16),
        },
        {
            'Item':           'Q17',
            'Description':    'Response time increased from last session',
            'Kappa Type':     "Cohen's Kappa (binary)",
            'N pairs':        n17,
            'Kappa':          f"{k17:.3f}" if pd.notna(k17) else 'N/A',
            'Interpretation': _kappa_label(k17),
        },
    ])
    st.dataframe(kappa_df, use_container_width=True)

    # Value distributions
    st.markdown("---")
    st.markdown("**Raw value distributions (T vs P) for non-quantitative items:**")

    for q_label, vc in [('Q12', vc12), ('Q16', vc16), ('Q17', vc17)]:
        col_name = REVERSE_QUESTION_MAPPING.get(q_label, '')
        if col_name and col_name in df_ex.columns and vc:
            with st.expander(f"{q_label} — value distribution"):
                dist = (
                    df_ex[[col_name, 'Submitted_by']]
                    .dropna(subset=[col_name])
                    .groupby([col_name, 'Submitted_by'])
                    .size()
                    .unstack(fill_value=0)
                )
                st.dataframe(dist, use_container_width=True)


def _section_6_5(icc_table):
    st.subheader("6.5  ICC Visualisations")

    if not _HAS_PINGOUIN:
        st.error(
            "`pingouin` is not installed.  \n"
            "Run `pip install pingouin` in your virtual environment and restart the app."
        )
        return

    valid = icc_table.dropna(subset=['ICC']).copy()
    if valid.empty:
        st.warning("No valid ICC values to plot.")
        return

    tab_bar, tab_forest, tab_domain = st.tabs(["Bar Chart", "Forest Plot", "Domain Summary"])

    # ── Bar chart ──────────────────────────────────────────────────────────────
    with tab_bar:
        sdf = valid.sort_values('ICC', ascending=True).reset_index(drop=True)
        colours = [_INTERP_COLOURS[_icc_label(v)] for v in sdf['ICC']]
        y_labels = sdf['Q'] + ' — ' + sdf['Description'].str[:35]

        fig, ax = plt.subplots(figsize=(10, max(5, len(sdf) * 0.42)))
        bars = ax.barh(y_labels, sdf['ICC'], color=colours, edgecolor='white', height=0.72)

        for thresh, ls, lbl in [
            (0.40, '--', 'Poor/Fair 0.40'),
            (0.60, '-.', 'Fair/Good 0.60'),
            (0.75, ':',  'Good/Excellent 0.75'),
        ]:
            ax.axvline(thresh, color='#555555', linestyle=ls, linewidth=1.2, alpha=0.8, label=lbl)

        for bar, val in zip(bars, sdf['ICC']):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=7.5,
            )

        ax.set_xlim(0, 1.08)
        ax.set_xlabel('ICC3 Value', fontsize=10)
        ax.set_title('ICC3 by Item — sorted ascending', fontsize=11, fontweight='bold')

        legend_patches = [
            mpatches.Patch(color='#d62728', label='Poor (< 0.40)'),
            mpatches.Patch(color='#ff7f0e', label='Fair (0.40–0.59)'),
            mpatches.Patch(color='#2ca02c', label='Good (0.60–0.74)'),
            mpatches.Patch(color='#1f77b4', label='Excellent (≥ 0.75)'),
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Forest plot ────────────────────────────────────────────────────────────
    with tab_forest:
        sdf = valid.sort_values('ICC', ascending=False).reset_index(drop=True)
        n = len(sdf)
        colours = [_INTERP_COLOURS[_icc_label(v)] for v in sdf['ICC']]
        y_labels = sdf['Q'] + ' — ' + sdf['Description'].str[:32]

        fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)))
        y_pos = np.arange(n)

        for i, row in sdf.iterrows():
            ci_lo = row['CI Lower'] if pd.notna(row['CI Lower']) else row['ICC']
            ci_hi = row['CI Upper'] if pd.notna(row['CI Upper']) else row['ICC']
            ax.plot([ci_lo, ci_hi], [i, i], color=colours[i], linewidth=2.8, solid_capstyle='round')
            ax.scatter(row['ICC'], i, color=colours[i], s=55, zorder=5)

        for thresh, ls, lbl in [
            (0.40, '--', '0.40'),
            (0.60, '-.', '0.60'),
            (0.75, ':',  '0.75'),
        ]:
            ax.axvline(thresh, color='#777777', linestyle=ls, linewidth=1.0, label=lbl)

        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlim(-0.15, 1.12)
        ax.set_xlabel('ICC3 with 95% Confidence Interval', fontsize=10)
        ax.set_title('Forest Plot — ICC3 with 95% CIs', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, title='Benchmarks')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Domain-grouped summary ─────────────────────────────────────────────────
    with tab_domain:
        domain_stats = (
            valid.groupby('Domain')['ICC']
            .agg(Mean='mean', Min='min', Max='max', N='count')
            .reset_index()
            .sort_values('Mean', ascending=False)
        )
        domain_stats['Interpretation'] = domain_stats['Mean'].apply(_icc_label)

        # Table
        fmt = domain_stats.copy()
        for c in ['Mean', 'Min', 'Max']:
            fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
        fmt = fmt.rename(columns={'Mean': 'Mean ICC', 'Min': 'Min ICC', 'Max': 'Max ICC', 'N': 'N items'})
        st.dataframe(fmt, use_container_width=True)

        # Bar chart
        sdf = domain_stats.sort_values('Mean', ascending=True).reset_index(drop=True)
        colours = [_INTERP_COLOURS[_icc_label(v)] for v in sdf['Mean']]

        fig, ax = plt.subplots(figsize=(9, max(4, len(sdf) * 0.7)))
        bars = ax.barh(sdf['Domain'], sdf['Mean'], color=colours, edgecolor='white', height=0.6)

        # Error bars (min–max)
        xerr_lo = (sdf['Mean'] - sdf['Min']).values
        xerr_hi = (sdf['Max'] - sdf['Mean']).values
        ax.errorbar(
            sdf['Mean'], np.arange(len(sdf)),
            xerr=[xerr_lo, xerr_hi],
            fmt='none', color='#333333', capsize=5, linewidth=1.8,
        )

        for thresh, ls in [(0.40, '--'), (0.60, '-.'), (0.75, ':')]:
            ax.axvline(thresh, color='#555555', linestyle=ls, linewidth=1.0, alpha=0.8)

        for bar, val in zip(bars, sdf['Mean']):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=8.5,
            )

        ax.set_xlim(0, 1.10)
        ax.set_xlabel('Mean ICC3  (error bars = min / max within domain)', fontsize=9)
        ax.set_title('Domain-Grouped ICC Summary', fontsize=11, fontweight='bold')

        legend_patches = [
            mpatches.Patch(color='#d62728', label='Poor (< 0.40)'),
            mpatches.Patch(color='#ff7f0e', label='Fair (0.40–0.59)'),
            mpatches.Patch(color='#2ca02c', label='Good (0.60–0.74)'),
            mpatches.Patch(color='#1f77b4', label='Excellent (≥ 0.75)'),
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def display(df, scale_map=None):
    st.header("Section 6 — Inter-Rater Reliability (ICC)")

    if not _HAS_PINGOUIN:
        st.warning(
            "⚠️ **`pingouin` is not installed.**  "
            "ICC computation requires pingouin.  \n"
            "Run `pip install pingouin` in your virtual environment and restart the app."
        )

    df_ex = df[df['Participant id'] != EXCLUDED_PARTICIPANT].copy()

    with st.spinner("Computing ICC for all 21 items — this may take a few seconds…"):
        icc_table = _build_icc_table(df_ex)

    tab1, tab2, tab3, tab4 = st.tabs([
        "6.1 Paired Observations",
        "6.2/6.3 ICC Results",
        "6.4 Kappa",
        "6.5 Visualisations",
    ])

    with tab1:
        _section_6_1(df, df_ex)
    with tab2:
        _section_6_3(icc_table)
    with tab3:
        _section_6_4(df_ex)
    with tab4:
        _section_6_5(icc_table)
