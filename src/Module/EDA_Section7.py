"""
EDA — Section 7: Correlation Analysis
======================================
Sub-sections:
  7.1  Inter-Item Correlation Matrix  — Pearson heatmap, all 21 items
  7.2  Top Correlations Table         — 10 strongest positive, 10 weakest
  7.3  Domain Coherence               — Mean within-domain correlation per domain
  7.4  Item-Total Correlation         — Each item vs mean of all others; flags < 0.30
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    QUESTION_MAPPING,
    ITEM_DESC,
    QUANT_ITEMS,
)

# ── Domain definitions (matching Section 6 user specification) ────────────────
CORR_DOMAINS = {
    'Participation & Engagement':          ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'],
    'Observed Behaviour & Emotions':       ['Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q13', 'Q14'],
    'Response to Story Theme / Learning':  ['Q18', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25'],
    'Social Impact':                       ['Q26'],
}

# ── Short axis labels — matching RQ9 style ───────────────────────────────────
SHORT_LABEL = {
    'Q1':  'Engagement',
    'Q2':  'Understanding',
    'Q3':  'Connection',
    'Q4':  'Verbal',
    'Q5':  'Attention',
    'Q6':  'Retelling',
    'Q7':  'Enjoyment',
    'Q8':  'Distress',
    'Q9':  'Initiation',
    'Q10': 'Repetition',
    'Q11': 'Creativity',
    'Q13': 'RelationImprove',
    'Q14': 'Expression',
    'Q18': 'StoryTheme',
    'Q20': 'ApplyLearning',
    'Q21': 'Confidence',
    'Q22': 'Generalise',
    'Q23': 'Recall',
    'Q24': 'Reflect',
    'Q25': 'LinkReal',
    'Q26': 'SocialImpact',
}

def _axis_label(q):
    """Return 'Q1: Engagement' style label."""
    return f"{q}: {SHORT_LABEL.get(q, q)}"



def _build_numeric_df(df):
    """
    Return a DataFrame with one column per QUANT_ITEM (labelled Q1, Q2…),
    coerced to numeric, in Q-label order.
    """
    cols_ordered = []
    for col, q in QUESTION_MAPPING.items():
        if q in QUANT_ITEMS and col in df.columns:
            cols_ordered.append((q, col))
    cols_ordered.sort(key=lambda x: (int(x[0][1:]) if x[0][1:].isdigit() else 99))

    num_df = pd.DataFrame()
    for q, col in cols_ordered:
        values = pd.to_numeric(df[col], errors='coerce')
        # Q8 (Distress) is reverse-coded: high raw score = bad; apply 4 - Q8 so high = good
        if q == 'Q8':
            values = 4 - values
        num_df[q] = values
    return num_df


def _domain_for(q_label):
    for domain, items in CORR_DOMAINS.items():
        if q_label in items:
            return domain
    return 'Other'


# ── Section renderers ─────────────────────────────────────────────────────────

def _section_7_1(num_df):
    st.subheader("7.1  Inter-Item Correlation Matrix")
    st.markdown(
        "Pearson correlation across all **21 quantitative items**. "
        "Pairs with fewer than 10 valid observations are shown as blank."
    )

    corr = num_df.corr(method='pearson', min_periods=10)

    # Rename columns/index to "Q1: Engagement" style
    axis_labels = [_axis_label(q) for q in corr.columns]
    corr_display = corr.copy()
    corr_display.columns = axis_labels
    corr_display.index   = axis_labels

    n = len(corr_display)
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')

    sns.heatmap(
        corr_display,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor='#cccccc',
        square=True,
        ax=ax,
        annot_kws={'size': 7.5, 'weight': 'bold'},
        cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
        mask=corr_display.isna(),
    )

    # Domain boundary lines (map boundaries back to index positions)
    q_list = list(corr.columns)
    domain_sizes = []
    seen = []
    for q in q_list:
        d = _domain_for(q)
        if not seen or seen[-1] != d:
            seen.append(d)
            domain_sizes.append(1)
        else:
            domain_sizes[-1] += 1

    boundary = 0
    for sz in domain_sizes[:-1]:
        boundary += sz
        ax.axhline(boundary, color='#111', linewidth=2.0)
        ax.axvline(boundary, color='#111', linewidth=2.0)

    ax.set_title("Inter-Item Pearson Correlation Matrix", fontsize=13, fontweight='bold', pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Legend for domain boundaries
    domain_names = list(CORR_DOMAINS.keys())
    st.markdown(
        " · ".join(
            f"**{d}**: {', '.join(_axis_label(q) for q in CORR_DOMAINS[d])}" for d in domain_names
        )
    )


def _section_7_2(num_df):
    st.subheader("7.2  Top Correlations Table")

    corr = num_df.corr(method='pearson', min_periods=10)

    # Extract upper-triangle pairs
    rows = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            qi, qj = cols[i], cols[j]
            r = corr.loc[qi, qj]
            if pd.isna(r):
                continue
            # t-stat and p-value
            n = num_df[[qi, qj]].dropna().shape[0]
            if n > 2:
                t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r ** 2, 1e-12))
                p = 2 * stats.t.sf(abs(t), df=n - 2)
            else:
                p = np.nan
            rows.append({
                'Item A': _axis_label(qi),
                'Item B': _axis_label(qj),
                'r':      r,
                'n':      n,
                'p':      p,
            })

    pair_df = pd.DataFrame(rows).sort_values('r', ascending=False).reset_index(drop=True)

    def _fmt(df_in):
        out = df_in.copy()
        out['r'] = out['r'].map(lambda x: f"{x:.3f}")
        out['p'] = out['p'].map(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        out['Sig.'] = df_in['p'].map(
            lambda x: '***' if pd.notna(x) and x < 0.001
            else ('**' if pd.notna(x) and x < 0.01
                  else ('*' if pd.notna(x) and x < 0.05 else ''))
        )
        return out[['Item A', 'Item B', 'r', 'n', 'p', 'Sig.']]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**10 Strongest Positive Correlations**")
        st.dataframe(_fmt(pair_df.head(10)), use_container_width=True)
    with col2:
        st.markdown("**10 Weakest / Most Negative Correlations**")
        st.dataframe(_fmt(pair_df.tail(10).iloc[::-1].reset_index(drop=True)), use_container_width=True)

    # Scatter for top pair — need original Q labels to index num_df
    if len(pair_df) > 0:
        top = pair_df.iloc[0]
        # Reverse axis label back to Q-label for num_df indexing
        rev = {v: k for k, v in SHORT_LABEL.items()}
        def _q(label):
            # label is "Q1: Engagement" → extract "Q1"
            return label.split(':')[0].strip()
        qa, qb = _q(top['Item A']), _q(top['Item B'])
        st.markdown(f"---\n**Scatter: {top['Item A']} vs {top['Item B']}** (r = {top['r']:.3f})")
        valid = num_df[[qa, qb]].dropna()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(valid[qa], valid[qb], alpha=0.45, s=25, color='#1f77b4', edgecolors='none')
        m, b = np.polyfit(valid[qa], valid[qb], 1)
        xr = np.linspace(valid[qa].min(), valid[qa].max(), 100)
        ax.plot(xr, m * xr + b, color='#d62728', linewidth=1.8)
        ax.set_xlabel(top['Item A'], fontsize=8)
        ax.set_ylabel(top['Item B'], fontsize=8)
        ax.set_title(f"Top correlation: {top['Item A']} × {top['Item B']}  (r = {top['r']:.3f})", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _section_7_3(num_df):
    st.subheader("7.3  Domain Coherence")
    st.markdown(
        "Mean pairwise Pearson correlation **within each domain** — "
        "higher values indicate items cluster together conceptually. "
        "A value ≥ 0.30 is generally acceptable internal consistency."
    )

    corr = num_df.corr(method='pearson', min_periods=10)
    domain_rows = []

    for domain, items in CORR_DOMAINS.items():
        present = [q for q in items if q in corr.columns]
        if len(present) < 2:
            domain_rows.append({
                'Domain': domain,
                'Items':  ', '.join(present),
                'N items': len(present),
                'Mean r': np.nan,
                'Min r':  np.nan,
                'Max r':  np.nan,
                'Coherence': 'N/A (too few)',
            })
            continue
        sub = corr.loc[present, present]
        # Upper triangle only
        rs = [sub.iloc[i, j]
              for i in range(len(present))
              for j in range(i + 1, len(present))
              if not np.isnan(sub.iloc[i, j])]
        mean_r = np.mean(rs) if rs else np.nan
        coherence = ('Strong (≥ 0.50)' if mean_r >= 0.50
                     else 'Acceptable (0.30–0.49)' if mean_r >= 0.30
                     else 'Weak (< 0.30)')
        domain_rows.append({
            'Domain':  domain,
            'Items':   ', '.join(present),
            'N items': len(present),
            'Mean r':  mean_r,
            'Min r':   min(rs) if rs else np.nan,
            'Max r':   max(rs) if rs else np.nan,
            'Coherence': coherence,
        })

    dom_df = pd.DataFrame(domain_rows)

    # Styled table
    def _style_coh(row):
        coh = row['Coherence']
        if 'Strong' in str(coh):
            bg = 'background-color:#d4edda;color:#155724'
        elif 'Acceptable' in str(coh):
            bg = 'background-color:#fff3cc;color:#7a5200'
        elif 'Weak' in str(coh):
            bg = 'background-color:#ffd7d7;color:#7b0000'
        else:
            bg = ''
        return [bg if c == 'Coherence' else '' for c in row.index]

    disp = dom_df.copy()
    for c in ['Mean r', 'Min r', 'Max r']:
        disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')
    st.dataframe(
        disp.style.apply(_style_coh, axis=1),
        use_container_width=True,
    )

    # Per-domain mini heatmaps
    st.markdown("---")
    st.markdown("**Within-domain correlation sub-matrices:**")
    domain_list = [d for d, items in CORR_DOMAINS.items()
                   if len([q for q in items if q in corr.columns]) >= 2]
    n_cols = min(2, len(domain_list))
    cols = st.columns(n_cols)

    for idx, domain in enumerate(domain_list):
        present = [q for q in CORR_DOMAINS[domain] if q in corr.columns]
        sub = corr.loc[present, present]
        n = len(present)
        fig, ax = plt.subplots(figsize=(max(3, n * 0.7), max(2.5, n * 0.65)))
        im = ax.imshow(sub.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_labels = [f"{q}\n{ITEM_DESC.get(q,'')[:14]}" for q in present]
        ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(short_labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                val = sub.values[i, j]
                if not np.isnan(val):
                    tc = 'white' if abs(val) > 0.65 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7, color=tc)
        ax.set_title(domain, fontsize=8, fontweight='bold', pad=6)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        cols[idx % n_cols].pyplot(fig)
        plt.close(fig)


def _section_7_4(num_df):
    st.subheader("7.4  Item-Total Correlation")
    st.markdown(
        "Each item is correlated (Pearson) with the **mean of all other items** "
        "(corrected item-total correlation). "
        "Items with r **< 0.30** may be poor contributors to a composite score."
    )

    items = [q for q in num_df.columns]
    rows = []

    for q in items:
        others = [o for o in items if o != q]
        this_item = num_df[q]
        rest_mean = num_df[others].mean(axis=1)
        valid = pd.concat([this_item, rest_mean], axis=1).dropna()
        if len(valid) < 5:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        rows.append({
            'Q':             q,
            'Description':   ITEM_DESC.get(q, q),
            'Domain':        _domain_for(q),
            'r (item-total)': r,
            'p-value':        p,
            'n':              len(valid),
            'Flag':           '⚠ Low (<0.30)' if pd.notna(r) and r < 0.30 else '',
        })

    it_df = pd.DataFrame(rows).sort_values('r (item-total)', ascending=False).reset_index(drop=True)

    # Styled table — apply styling on numeric it_df, format via styler
    def _style_it(row):
        r_val = row['r (item-total)']
        try:
            r_num = float(r_val)
        except (TypeError, ValueError):
            return [''] * len(row)
        if r_num < 0.30:
            bg = 'background-color:#ffd7d7;color:#7b0000'
        elif r_num < 0.50:
            bg = 'background-color:#fff3cc;color:#7a5200'
        else:
            bg = 'background-color:#d4edda;color:#155724'
        return [bg if c == 'r (item-total)' else '' for c in row.index]

    out_cols = ['Q', 'Description', 'Domain', 'r (item-total)', 'p-value', 'n', 'Flag']
    st.dataframe(
        it_df[out_cols].style
            .apply(_style_it, axis=1)
            .format({
                'r (item-total)': lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A',
                'p-value':        lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A',
            }),
        use_container_width=True,
        height=min(800, 45 + len(it_df) * 38),
    )

    # Dot-plot
    valid_plot = it_df.dropna(subset=['r (item-total)']).sort_values('r (item-total)', ascending=True)
    colours = ['#d62728' if r < 0.30 else '#ff7f0e' if r < 0.50 else '#2ca02c'
               for r in valid_plot['r (item-total)']]
    y_labels = valid_plot['Q'] + ' — ' + valid_plot['Description'].str[:35]

    fig, ax = plt.subplots(figsize=(8, max(5, len(valid_plot) * 0.38)))
    ax.barh(y_labels, valid_plot['r (item-total)'], color=colours, edgecolor='white', height=0.65)
    ax.axvline(0.30, color='#d62728', linestyle='--', linewidth=1.4, label='Threshold 0.30')
    ax.axvline(0.50, color='#ff7f0e', linestyle='-.', linewidth=1.2, label='0.50')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Corrected Item-Total r', fontsize=9)
    ax.set_title('Item-Total Correlation', fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

    legend_patches = [
        mpatches.Patch(color='#d62728', label='Low (< 0.30)'),
        mpatches.Patch(color='#ff7f0e', label='Moderate (0.30–0.49)'),
        mpatches.Patch(color='#2ca02c', label='Good (≥ 0.50)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Flag summary
    flagged = it_df[it_df['Flag'] != '']
    if not flagged.empty:
        st.warning(
            f"**{len(flagged)} item(s) below threshold (r < 0.30):** "
            + ', '.join(flagged['Q'].tolist())
        )
    else:
        st.success("All items exceed the r ≥ 0.30 item-total threshold.")


# ── Main entry point ──────────────────────────────────────────────────────────

def display(df, scale_map=None):
    st.header("Section 7 — Correlation Analysis")

    num_df = _build_numeric_df(df)
    if num_df.empty:
        st.error("No quantitative columns found in the dataset.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "7.1 Correlation Matrix",
        "7.2 Top Correlations",
        "7.3 Domain Coherence",
        "7.4 Item-Total",
    ])

    with tab1:
        _section_7_1(num_df)
    with tab2:
        _section_7_2(num_df)
    with tab3:
        _section_7_3(num_df)
    with tab4:
        _section_7_4(num_df)
