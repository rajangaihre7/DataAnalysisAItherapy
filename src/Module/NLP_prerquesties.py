"""
NLP Prerequisites — Part A: Build the Gold Analytical Dataset
=============================================================
Constructs the Gold-layer CSV from the Silver dataset:

  A1  Reverse-code Q8  (distress → positive polarity, consistent with all other items)
  A2  Min-Max normalise all 21 quantitative items to 0.0 – 1.0
  A3  Compute composite Engagement Score (row-mean of the 21 normalised items)
  A4  Save Gold dataset  →  Data/Gold/data_gold_analytical.csv
  A5  Quick verification: distribution · observer comparison ·
      session trajectory · autism level · item-total correlations
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import QUESTION_MAPPING, QUANT_ITEMS, SCALE_10_ITEMS

# ── Import Silver → Gold transformation logic ─────────────────────────────────
_src_dir = os.path.normpath(os.path.join(_current_dir, '..'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from silver_to_gold_transformation import (  # noqa: E402
    NORM_ITEMS,
    norm_label  as _norm_label,
    get_gold_path as _gold_path,
    build_gold_df,
)


# ── Cached wrapper (Streamlit cache stays in the UI layer) ────────────────────
@st.cache_data(show_spinner=False)
def _build_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper around build_gold_df for Streamlit."""
    return build_gold_df(df)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _meta(gold: pd.DataFrame, col: str):
    """Return a metadata column with stripped string values, or None."""
    for candidate in [col, col.strip()]:
        if candidate in gold.columns:
            return gold[candidate].astype(str).str.strip()
    return None


# ═════════════════════════════════════════════════════════════════════════════
# STEP RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def _render_a1(gold: pd.DataFrame):
    st.subheader("A1 — Reverse-Code Q8 (Distress)")

    st.markdown("""
**Why reverse-code Q8?**
All other items are coded *high = good* (e.g., high engagement, high creativity).
Q8 measures *"distress, boredom, or frustration"*, where **0 = no distress (good)** and
**4 = very frequent distress (bad)**.  
Leaving Q8 un-reversed would *pull down* the composite score for participants who are doing
well — introducing systematic error.

**Transform:**  `Q8_R = 4 − Q8`  
After reversal: 0 = very frequent distress (bad), 4 = no distress (good) ✓
""")

    if 'Q8_R' not in gold.columns:
        st.error("Q8_R column not found — check that Q8 exists in Silver data.")
        return

    q8_col = next((c for c, q in QUESTION_MAPPING.items() if q == 'Q8' and c in gold.columns), None)
    q8_orig = pd.to_numeric(gold[q8_col], errors='coerce') if q8_col else pd.Series(dtype=float)

    col1, col2, col3 = st.columns(3)
    col1.metric("Q8 original mean", f"{q8_orig.mean():.3f}", help="Low = low distress = good")
    col2.metric("Q8_R reversed mean", f"{gold['Q8_R'].mean():.3f}", help="High = low distress = good")
    col3.metric("Sum check (should be 4.000)", f"{(q8_orig.mean() + gold['Q8_R'].mean()):.3f}")

    # Side-by-side distributions
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, series, title, colour in [
        (axes[0], q8_orig,        "Q8 original (0=good)", '#d62728'),
        (axes[1], gold['Q8_R'],   "Q8_R reversed (4=good)", '#2ca02c'),
    ]:
        counts = series.dropna().value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=colour, alpha=0.7, edgecolor='white')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    fig.suptitle("Q8 Before and After Reverse-Coding", fontsize=10, y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_a2(gold: pd.DataFrame):
    st.subheader("A2 — Min-Max Normalisation (0.0 – 1.0)")

    st.markdown("""
Items use two different raw scales:

| Scale | Divisor | Items |
|-------|---------|-------|
| 0 – 4 | ÷ 4 | Q1–Q3, Q5–Q11, Q13–Q14, Q18, Q20–Q25 + **Q8_R** |
| 0 – 10 | ÷ 10 | Q4 (Verbal), Q26 (SocialImpact) |

**Q8_R_norm** (reversed distress) enters the composite — **Q8_norm is not created**.
""")

    present = [c for c in NORM_ITEMS if c in gold.columns]
    stats_rows = []
    for col in present:
        s = gold[col].dropna()
        stats_rows.append({
            'Norm column': col,
            'Label': _norm_label(col),
            'N': int(s.notna().sum()),
            'Min': round(float(s.min()), 3),
            'Max': round(float(s.max()), 3),
            'Mean': round(float(s.mean()), 3),
            'SD': round(float(s.std()), 3),
            'In range': '✓' if s.min() >= 0.0 and s.max() <= 1.0 else '✗ OUT OF BOUNDS',
        })
    st.dataframe(pd.DataFrame(stats_rows).set_index('Norm column'), use_container_width=True)
    out_of_bounds = [r for r in stats_rows if r['In range'] != '✓']
    if out_of_bounds:
        st.error(f"Out-of-bounds columns: {[r['Norm column'] for r in out_of_bounds]}")
    else:
        st.success(f"All {len(present)} of 21 normalised items verified: range 0.0 – 1.0 ✓")


def _render_a3(gold: pd.DataFrame):
    st.subheader("A3 — Composite Engagement Score")

    st.markdown("""
`Engagement_Score` = row-mean of available normalised items (max 21).  
Records missing ≥ 5 items (< 17 of 21 present) are flagged as **Low_Item_Count**.
""")

    es = gold['Engagement_Score'].dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean",   f"{es.mean():.4f}")
    c2.metric("SD",     f"{es.std():.4f}")
    c3.metric("Min",    f"{es.min():.4f}")
    c4.metric("Max",    f"{es.max():.4f}")
    c5.metric("Low item-count records", int(gold['Low_Item_Count'].sum()))

    st.caption(f"Valid scores: {int(gold['Engagement_Score'].notna().sum())} of {len(gold)} records")

    # Items available distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    cnt = gold['Items_Available'].value_counts().sort_index()
    ax.bar(cnt.index, cnt.values, color='#1f77b4', edgecolor='white', alpha=0.8)
    ax.axvline(17, color='#d62728', linestyle='--', linewidth=1.2, label='80% threshold (17)')
    ax.set_xlabel("Items available per record", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Distribution of items available per record", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_a4(gold: pd.DataFrame):
    st.subheader("A4 — Save Gold Dataset")

    path = _gold_path()
    new_cols = ['Q8_R'] + NORM_ITEMS + ['Engagement_Score', 'Items_Available', 'Low_Item_Count']
    present_new = [c for c in new_cols if c in gold.columns]

    st.markdown(f"""
**Output path:** `{path}`

**New columns added ({len(present_new)}):**  
{', '.join(f'`{c}`' for c in present_new)}
""")

    already_saved = os.path.exists(path)

    if already_saved:
        mtime = pd.Timestamp(os.path.getmtime(path), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"Gold dataset already exists (saved {mtime}). Click below to overwrite.")

    if st.button("💾  Save / Overwrite Gold Dataset", type="primary"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gold.to_csv(path, index=False)
        st.success(f"Saved → `{path}`")
        st.caption(f"Rows: {len(gold):,}   Columns: {len(gold.columns)}")
        # Clear cache so next load picks up new file
        st.cache_data.clear()

    if already_saved:
        try:
            saved = pd.read_csv(path, nrows=3)
            with st.expander("Preview first 3 rows of saved Gold CSV"):
                st.dataframe(saved[present_new].head(3), use_container_width=True)
        except Exception:
            pass


def _render_a5(gold: pd.DataFrame):
    st.subheader("A5 — Verification of Gold Dataset")

    es      = gold['Engagement_Score'].dropna()
    obs_col = 'Submitted_by'
    ses_col = 'Session number'
    aut_col = 'Autism Level'
    pid_col = 'Participant id'

    tabs = st.tabs([
        "Distribution",
        "Observer Type",
        "Session Trajectory",
        "Autism Level",
        "Item Correlations",
    ])

    # ── Tab 1: Distribution ───────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("**Engagement Score Distribution**")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

        # Histogram
        axes[0].hist(es, bins=20, color='#1f77b4', edgecolor='white', alpha=0.85)
        axes[0].axvline(es.mean(), color='#d62728', linestyle='--', linewidth=1.5,
                        label=f'Mean = {es.mean():.3f}')
        axes[0].axvline(es.median(), color='#ff7f0e', linestyle=':', linewidth=1.5,
                        label=f'Median = {es.median():.3f}')
        axes[0].set_xlabel('Engagement Score', fontsize=9)
        axes[0].set_ylabel('Count', fontsize=9)
        axes[0].set_title('Histogram', fontsize=10)
        axes[0].legend(fontsize=8)

        # KDE / box overlay
        sns.boxplot(x=es, ax=axes[1], color='#aec7e8', width=0.4,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
        axes[1].set_xlabel('Engagement Score', fontsize=9)
        axes[1].set_title('Box Plot', fontsize=10)

        fig.suptitle('Composite Engagement Score — Distribution', fontsize=11, y=1.01)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Descriptive summary
        desc = es.describe().rename({
            'count':'N','mean':'Mean','std':'SD','min':'Min',
            '25%':'Q1 (25th)','50%':'Median','75%':'Q3 (75th)','max':'Max'
        })
        st.dataframe(desc.to_frame('Engagement Score').T.round(4), use_container_width=True)

    # ── Tab 2: Observer type ──────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("**Engagement Score by Observer Type (T = Therapist · P = Parent)**")
        if obs_col not in gold.columns:
            st.warning(f"Column '{obs_col}' not found.")
        else:
            obs_data = gold[[obs_col, 'Engagement_Score']].dropna()
            obs_data[obs_col] = obs_data[obs_col].astype(str).str.strip()
            grp = obs_data.groupby(obs_col)['Engagement_Score'].agg(['mean','std','count']).reset_index()
            grp.columns = ['Observer','Mean','SD','N']
            grp['SE'] = grp['SD'] / np.sqrt(grp['N'])

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(grp['Observer'], grp['Mean'], yerr=grp['SE'],
                          color=['#1f77b4','#ff7f0e'][:len(grp)],
                          edgecolor='white', alpha=0.85, capsize=5)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Mean Engagement Score (± SE)', fontsize=9)
            ax.set_title('Observer Type Comparison', fontsize=10)
            for bar, row in zip(bars, grp.itertuples()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row.SE + 0.01,
                        f'{row.Mean:.3f}\n(n={row.N})', ha='center', fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(grp.set_index('Observer').round(4), use_container_width=True)

            # Independent t-test if T and P both exist
            types = grp['Observer'].tolist()
            if 'T' in types and 'P' in types:
                t_vals = obs_data[obs_data[obs_col]=='T']['Engagement_Score'].dropna()
                p_vals = obs_data[obs_data[obs_col]=='P']['Engagement_Score'].dropna()
                t_stat, p_val = stats.ttest_ind(t_vals, p_vals, equal_var=False)
                st.markdown(f"**Welch's t-test:**  t = {t_stat:.3f},  p = {p_val:.4f}"
                            + ("  *(significant at α=0.05)*" if p_val < 0.05 else "  *(not significant)*"))

    # ── Tab 3: Session trajectory ─────────────────────────────────────────────
    with tabs[2]:
        st.markdown("**Mean Engagement Score per Session — should show upward trend**")
        if ses_col not in gold.columns:
            st.warning(f"Column '{ses_col}' not found.")
        else:
            ses_data = gold[[ses_col, 'Engagement_Score']].dropna()
            ses_data[ses_col] = pd.to_numeric(ses_data[ses_col], errors='coerce')
            ses_data = ses_data.dropna(subset=[ses_col])
            ses_grp = ses_data.groupby(ses_col)['Engagement_Score'].agg(
                Mean='mean', SD='std', N='count').reset_index()
            ses_grp['SE'] = ses_grp['SD'] / np.sqrt(ses_grp['N'])

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(ses_grp[ses_col], ses_grp['Mean'], marker='o', linewidth=2,
                    color='#1f77b4', markersize=5, label='Mean')
            ax.fill_between(ses_grp[ses_col],
                             ses_grp['Mean'] - ses_grp['SE'],
                             ses_grp['Mean'] + ses_grp['SE'],
                             alpha=0.2, color='#1f77b4', label='± SE')
            # Trend line
            x = ses_grp[ses_col].values
            y = ses_grp['Mean'].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 1:
                m, b = np.polyfit(x[mask], y[mask], 1)
                ax.plot(x[mask], m * x[mask] + b, '--', color='#d62728',
                        linewidth=1.4, label=f'Trend (slope={m:+.4f})')
            ax.set_xlabel('Session', fontsize=9)
            ax.set_ylabel('Mean Engagement Score', fontsize=9)
            ax.set_title('Engagement Score Trajectory Across Sessions', fontsize=10)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(ses_grp.set_index(ses_col).round(4), use_container_width=True)

    # ── Tab 4: Autism level ───────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("**Mean Engagement Score by Autism Level**")
        if aut_col not in gold.columns:
            st.warning(f"Column '{aut_col}' not found.")
        else:
            aut_data = gold[[aut_col, 'Engagement_Score']].dropna()
            aut_data[aut_col] = pd.to_numeric(aut_data[aut_col], errors='coerce')
            aut_data = aut_data.dropna(subset=[aut_col])
            aut_grp = aut_data.groupby(aut_col)['Engagement_Score'].agg(
                Mean='mean', SD='std', N='count').reset_index()
            aut_grp['SE'] = aut_grp['SD'] / np.sqrt(aut_grp['N'])

            palette = ['#2ca02c','#ff7f0e','#d62728','#9467bd','#8c564b']
            fig, ax = plt.subplots(figsize=(max(5, len(aut_grp)*1.2), 4))
            bars = ax.bar(aut_grp[aut_col].astype(int).astype(str),
                          aut_grp['Mean'], yerr=aut_grp['SE'],
                          color=palette[:len(aut_grp)],
                          edgecolor='white', alpha=0.85, capsize=5)
            ax.set_xlabel('Autism Level', fontsize=9)
            ax.set_ylabel('Mean Engagement Score (± SE)', fontsize=9)
            ax.set_title('Engagement Score by Autism Level', fontsize=10)
            ax.set_ylim(0, 1)
            for bar, row in zip(bars, aut_grp.itertuples()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row.SE + 0.01,
                        f'{row.Mean:.3f}\n(n={row.N})', ha='center', fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(aut_grp.set_index(aut_col).round(4), use_container_width=True)

            # One-way ANOVA if ≥2 groups
            groups = [g['Engagement_Score'].dropna().values
                      for _, g in aut_data.groupby(aut_col) if len(g) > 1]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                st.markdown(f"**One-way ANOVA:**  F = {f_stat:.3f},  p = {p_val:.4f}"
                            + ("  *(significant at α=0.05)*" if p_val < 0.05 else "  *(not significant)*"))

    # ── Tab 5: Item-total correlations ────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
**Pearson r between Engagement_Score and each normalised item**

> This is an *item-total* check: how much does each item co-vary with the overall
> composite?  A well-constructed composite should show r ≥ 0.30 for all items.
""")
        present = [c for c in NORM_ITEMS if c in gold.columns]
        rows = []
        for col in present:
            pair = gold[[col, 'Engagement_Score']].dropna()
            if len(pair) < 3:
                continue
            r, p = stats.pearsonr(pair[col], pair['Engagement_Score'])
            rows.append({
                'Item': col,
                'Label': _norm_label(col),
                'r': round(r, 4),
                'p': round(p, 4),
                'Sig.': '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')),
                'Flag': '⚠ r < 0.30' if r < 0.30 else '',
            })

        if not rows:
            st.warning("No correlation data available.")
            return

        corr_df = pd.DataFrame(rows)

        def _style_r(val):
            if val >= 0.70: return 'background-color:#2ca02c;color:white'
            if val >= 0.50: return 'background-color:#98df8a'
            if val >= 0.30: return 'background-color:#dbedbc'
            return 'background-color:#f8d7da'

        styled = (
            corr_df.set_index('Item')
            .style
            .map(_style_r, subset=['r'])
            .format({'r': '{:.4f}', 'p': '{:.4f}'})
        )
        st.dataframe(styled, use_container_width=True)

        low_items = corr_df[corr_df['r'] < 0.30]
        if low_items.empty:
            st.success("All items correlate ≥ 0.30 with Engagement_Score ✓")
        else:
            st.warning(f"{len(low_items)} item(s) with r < 0.30: "
                       f"{', '.join(low_items['Label'].tolist())}")

        # Horizontal bar chart
        corr_df_sorted = corr_df.sort_values('r', ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(4, len(corr_df_sorted) * 0.4)))
        colours = ['#d62728' if r < 0.30 else ('#2ca02c' if r >= 0.70 else '#1f77b4')
                   for r in corr_df_sorted['r']]
        ax.barh(corr_df_sorted['Label'], corr_df_sorted['r'],
                color=colours, alpha=0.85, edgecolor='white')
        ax.axvline(0.30, color='#d62728', linestyle='--', linewidth=1.2, label='r = 0.30 threshold')
        ax.axvline(0.70, color='#2ca02c', linestyle='--', linewidth=1.2, label='r = 0.70 (strong)')
        ax.set_xlabel('Pearson r with Engagement_Score', fontsize=9)
        ax.set_title('Item-Total Correlations', fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def display(df: pd.DataFrame, scale_map=None):
    st.header("NLP Prerequisites — Part A: Gold Analytical Dataset")
    st.markdown("""
This section constructs the **Gold analytical dataset** used for all downstream
NLP and modelling work (LME, clustering, sentiment analysis).

| Step | Action |
|------|--------|
| **A1** | Reverse-code Q8 (distress) so all items are *high = good* |
| **A2** | Min-Max normalise all 21 items to 0.0 – 1.0 |
| **A3** | Compute composite **Engagement Score** (row-mean of 21 norm items) |
| **A4** | Save Gold CSV → `Data/Gold/data_gold_analytical.csv` |
| **A5** | Verify distribution, observer, session trajectory, autism level, item correlations |
""")
    st.divider()

    with st.spinner("Building Gold dataset …"):
        gold = _build_gold_df(df)

    tab_a1, tab_a2, tab_a3, tab_a4, tab_a5 = st.tabs([
        "A1 · Reverse Q8",
        "A2 · Normalise",
        "A3 · Engagement Score",
        "A4 · Save Gold",
        "A5 · Verification",
    ])

    with tab_a1:
        _render_a1(gold)
    with tab_a2:
        _render_a2(gold)
    with tab_a3:
        _render_a3(gold)
    with tab_a4:
        _render_a4(gold)
    with tab_a5:
        _render_a5(gold)
