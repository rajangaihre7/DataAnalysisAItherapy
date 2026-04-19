"""
NLP Sentiment Analysis — Part B  (Results Display)
====================================================
This module only DISPLAYS pre-computed results.

To generate all results, run ONCE from the project root (with venv activated):
    python src/run_nlp_pipeline.py

Results are read from  results/nlp/
    descriptive/
        sentiment_distribution.png
        sentiment_summary_stats.csv
        sentiment_by_observer.png  /  sentiment_by_observer.csv
        sentiment_by_session.png   /  sentiment_by_session.csv
        sentiment_by_autism.png    /  sentiment_by_autism.csv
        mismatch_neg_sent_pos_lik.csv
        mismatch_pos_sent_neg_lik.csv
    rq2/
        rq2_overall.csv
        rq2_by_observer.csv
        rq2_by_session.csv
        sentiment_item_correlations.csv
        sentiment_vs_engagement_scatter.png
        sentiment_item_correlation_bar.png
    trajectory/
        dual_trajectory_engagement_vs_sentiment.png
        trajectory_table.csv
    ground_truth_sample.csv
    pipeline_summary.csv
"""

import os
import sys

import streamlit as st
import pandas as pd

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir     = os.path.normpath(os.path.join(_current_dir, '..'))
_project_dir = os.path.normpath(os.path.join(_src_dir, '..'))

for p in [_current_dir, _src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nlp(sub: str = '', filename: str = '') -> str:
    """Absolute path inside results/nlp/[sub/]filename."""
    base = os.path.join(_project_dir, 'results', 'nlp')
    parts = [base]
    if sub:
        parts.append(sub)
    if filename:
        parts.append(filename)
    return os.path.join(*parts)


def _read(sub: str, filename: str) -> pd.DataFrame | None:
    path = _nlp(sub, filename)
    return pd.read_csv(path) if os.path.exists(path) else None


def _show_img(sub: str, filename: str, caption: str = ''):
    path = _nlp(sub, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: `{filename}` — run `python src/run_nlp_pipeline.py` first.")


def _pipeline_ran() -> bool:
    return os.path.exists(_nlp('', 'pipeline_summary.csv'))


# ════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ════════════════════════════════════════════════════════════════════════════

def _render_b2():
    st.subheader("B2 — Sentiment Score Overview")
    summary = _read('', 'pipeline_summary.csv')
    if summary is None:
        st.warning("No results yet — run `python src/run_nlp_pipeline.py` first.")
        return

    row = summary.set_index('Metric')['Value']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records",            row.get('Total records', '—'))
    c2.metric("Records with notes",       row.get('Records with notes', '—'))
    c3.metric("Sentiment_Score non-null", row.get('Sentiment_Score non-null', '—'))
    c4.metric("Sentiment_Score mean",     row.get('Sentiment_Score mean', '—'))
    st.dataframe(summary.set_index('Metric'), use_container_width=True)


def _render_b3():
    st.subheader("B3 — Descriptive Analysis")

    tabs = st.tabs([
        "Distribution",
        "Summary Statistics",
        "By Observer Type",
        "By Session",
        "By Autism Level",
        "Mismatch Analysis",
    ])

    with tabs[0]:
        _show_img('descriptive', 'sentiment_distribution.png',
                  'Sentiment Score — Histogram and Box Plot')

    with tabs[1]:
        df = _read('descriptive', 'sentiment_summary_stats.csv')
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("Summary stats not found — run the pipeline script.")

    with tabs[2]:
        _show_img('descriptive', 'sentiment_by_observer.png',
                  'Mean Sentiment Score by Observer Type (T = Therapist, P = Parent)')
        df = _read('descriptive', 'sentiment_by_observer.csv')
        if df is not None:
            st.dataframe(df.set_index(df.columns[0]).round(4), use_container_width=True)

        mw_path = _nlp('descriptive', 'observer_mannwhitney.txt')
        if os.path.exists(mw_path):
            st.markdown("**Mann-Whitney U test (T vs P):**")
            st.code(open(mw_path).read(), language=None)

    with tabs[3]:
        _show_img('descriptive', 'sentiment_by_session.png',
                  'Sentiment & Engagement trajectories per session')
        df = _read('descriptive', 'sentiment_by_session.csv')
        if df is not None:
            st.dataframe(df.round(4), use_container_width=True)

    with tabs[4]:
        _show_img('descriptive', 'sentiment_by_autism.png',
                  'Mean Sentiment Score by Autism Level')
        df = _read('descriptive', 'sentiment_by_autism.csv')
        if df is not None:
            st.dataframe(df.set_index(df.columns[0]).round(4), use_container_width=True)

    with tabs[5]:
        st.markdown("**Sentiment-Clinical Mismatch Analysis**")
        st.markdown("""
Records where NLP sentiment and Likert Engagement Score contradict each other.
These illustrate the *sentiment-clinical mismatch* hypothesis (Chapter 3).
""")
        neg = _read('descriptive', 'mismatch_neg_sent_pos_lik.csv')
        pos = _read('descriptive', 'mismatch_pos_sent_neg_lik.csv')

        c1, c2 = st.columns(2)
        c1.metric("NLP negative / Likert positive  (Sent<0.3, Eng>0.6)",
                  len(neg) if neg is not None else 0)
        c2.metric("NLP positive / Likert negative  (Sent>0.7, Eng<0.4)",
                  len(pos) if pos is not None else 0)

        st.markdown("##### NLP negative — Likert positive")
        if neg is not None and len(neg):
            st.dataframe(neg.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No mismatches in this direction.")

        st.markdown("##### NLP positive — Likert negative")
        if pos is not None and len(pos):
            st.dataframe(pos.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No mismatches in this direction.")


def _render_b4():
    st.subheader("B4 — Ground-Truth Validation Sample")
    st.markdown("""
The pipeline script exports a **stratified 50-note sample** to `results/nlp/ground_truth_sample.csv`.

**Instructions:**
1. Open the file and fill in the `Manual_Label` column for each row  
   — `1` = Positive clinical trajectory  
   — `0` = Neutral / Negative trajectory  
2. Save the file, then come back to **B5** to compute validation metrics.
""")

    path = _nlp('', 'ground_truth_sample.csv')
    if not os.path.exists(path):
        st.warning("Ground-truth sample not yet generated — run the pipeline script.")
        return

    gt = pd.read_csv(path)
    filled = int((gt['Manual_Label'].astype(str).str.strip() != '').sum())
    st.info(f"Sample size: **{len(gt)}**  |  Manual_Label filled: **{filled} / {len(gt)}**")
    st.dataframe(gt.head(10), use_container_width=True, hide_index=True)

    csv_bytes = gt.to_csv(index=False).encode()
    st.download_button(
        label="⬇  Download ground_truth_sample.csv",
        data=csv_bytes,
        file_name="ground_truth_sample.csv",
        mime="text/csv",
    )


def _render_b5():
    st.subheader("B5 — Validation Metrics (After Manual Coding)")

    saved_path = _nlp('', 'ground_truth_sample.csv')

    if not os.path.exists(saved_path):
        st.warning("Ground-truth sample not yet generated — run the pipeline script first.")
        return

    try:
        gt = pd.read_csv(saved_path, encoding='latin-1')
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    if 'Manual_Label' not in gt.columns or 'Model_Predicted_Label' not in gt.columns:
        st.error("CSV must contain both 'Manual_Label' and 'Model_Predicted_Label' columns.")
        return

    # ── Interactive labelling ─────────────────────────────────────────────────
    notes_col = next((c for c in gt.columns if 'note' in c.lower() or 'observation' in c.lower()), None)
    blank = gt['Manual_Label'].isna() | (gt['Manual_Label'].astype(str).str.strip() == '')

    if blank.any():
        st.info(
            f"**{blank.sum()} row(s) still need a Manual_Label.**  "
            f"Use the table below to assign labels (`1` = Positive, `0` = Neutral/Negative), "
            f"then click **Save labels**."
        )
        display_cols = ['Participant id', 'Session number', 'Submitted_by']
        if notes_col:
            display_cols.append(notes_col)
        display_cols += ['Sentiment_Score', 'Model_Predicted_Label', 'Manual_Label']
        display_cols = [c for c in display_cols if c in gt.columns]

        edited = st.data_editor(
            gt[display_cols],
            column_config={
                'Manual_Label': st.column_config.NumberColumn(
                    'Manual_Label', min_value=0, max_value=1, step=1,
                    help='Enter 1 (Positive) or 0 (Neutral/Negative)'
                )
            },
            use_container_width=True,
            hide_index=True,
            key='gt_editor',
        )
        if st.button("💾 Save labels to file"):
            # Write edited Manual_Label back to the full dataframe
            gt.loc[edited.index, 'Manual_Label'] = edited['Manual_Label'].values
            gt.to_csv(saved_path, index=False)
            st.success("Labels saved. The metrics will now appear below.")
            st.rerun()
        return

    # ── Compute metrics ───────────────────────────────────────────────────────
    try:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
        )
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        return

    y_true = gt['Manual_Label'].astype(int)
    y_pred = gt['Model_Predicted_Label'].astype(int)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = recall_score(   y_true, y_pred, average='macro', zero_division=0)
    f1        = f1_score(       y_true, y_pred, average='macro', zero_division=0)
    accuracy  = float((y_true == y_pred).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1",  f"{f1:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3.metric("Recall",    f"{recall:.3f}")
    c4.metric("Accuracy",  f"{accuracy:.3f}")

    report = classification_report(y_true, y_pred,
                                   target_names=['Negative/Neutral', 'Positive'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred: Neg/Neutral', 'Pred: Positive'],
                yticklabels=['True: Neg/Neutral', 'True: Positive'])
    ax.set_title('Confusion Matrix', fontsize=10)
    ax.set_ylabel('Manual Label (Ground Truth)', fontsize=9)
    ax.set_xlabel('Model Prediction', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # By observer
    obs_col = next((c for c in gt.columns if 'observer' in c.lower() or c == 'Submitted_by'), None)
    if obs_col:
        st.markdown("**F1 by Observer Type**")
        rows = []
        for obs in gt[obs_col].dropna().unique():
            sub = gt[gt[obs_col] == obs]
            if len(sub) > 2:
                obs_f1 = f1_score(sub['Manual_Label'].astype(int),
                                  sub['Model_Predicted_Label'].astype(int),
                                  average='macro', zero_division=0)
                rows.append({'Observer': obs, 'F1 (macro)': round(obs_f1, 3), 'N': len(sub)})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index('Observer'), use_container_width=True)


def _render_b6():
    st.subheader("B6 — Spearman Correlation: NLP Sentiment vs Engagement Score (RQ2)")
    st.markdown("""
**Research Question 2:** Does NLP-derived sentiment correlate with structured clinical assessments?  
Spearman ρ (non-parametric, robust to non-normality).
""")

    overall = _read('rq2', 'rq2_overall.csv')
    if overall is not None:
        row = overall.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman ρ (overall)", row.get('rho', '—'))
        c2.metric("p-value",              row.get('p',   '—'))
        c3.metric("n",                    row.get('N',   '—'))
        sig = "significant at α=0.05" if str(row.get('significant', '')).lower() == 'true' else "not significant"
        st.caption(sig)
    else:
        st.warning("RQ2 results not found — run the pipeline script.")

    tabs = st.tabs(["By Observer", "By Session", "By Item", "Scatter Plot", "Item Bar Chart"])

    with tabs[0]:
        df = _read('rq2', 'rq2_by_observer.csv')
        if df is not None:
            st.dataframe(df.set_index('Observer').round(3), use_container_width=True)
        else:
            st.info("No data.")

    with tabs[1]:
        df = _read('rq2', 'rq2_by_session.csv')
        if df is not None:
            st.dataframe(df.set_index('Session').round(3), use_container_width=True)
        else:
            st.info("No data.")

    with tabs[2]:
        df = _read('rq2', 'sentiment_item_correlations.csv')
        if df is not None:
            st.dataframe(df.set_index('Item').round(3), use_container_width=True)
        else:
            st.info("No data.")

    with tabs[3]:
        _show_img('rq2', 'sentiment_vs_engagement_scatter.png',
                  'RQ2: NLP Sentiment vs Structured Clinical Assessment')

    with tabs[4]:
        _show_img('rq2', 'sentiment_item_correlation_bar.png',
                  'Spearman ρ between Sentiment Score and each assessment item')


def _render_b7():
    st.subheader("B7 — Dual Trajectory: Engagement Score vs Sentiment Score Across Sessions")
    _show_img('trajectory', 'dual_trajectory_engagement_vs_sentiment.png',
              'Session-level mean Engagement Score (Likert) vs Sentiment Score (NLP)')
    df = _read('trajectory', 'trajectory_table.csv')
    if df is not None:
        st.dataframe(df.round(4), use_container_width=True)


def _render_b8():
    st.subheader("B8 — Final Gold Dataset")
    path = _nlp('', 'data_gold_with_sentiment.csv')
    if os.path.exists(path):
        mtime = pd.Timestamp(os.path.getmtime(path), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        st.success(f"Gold + Sentiment dataset saved at `{path}`  (last updated: {mtime})")
        try:
            preview = pd.read_csv(path, nrows=5)
            with st.expander("Preview first 5 rows"):
                st.dataframe(preview, use_container_width=True)
        except Exception:
            pass
    else:
        st.warning("Gold + Sentiment dataset not found — run the pipeline script.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def display(df, scale_map=None):
    st.header("NLP Sentiment Analysis Pipeline — Part B")

    st.markdown("""
All computations are performed **offline** by the standalone pipeline script.  
This dashboard only displays the pre-computed results.

> To generate or refresh results, run from the project root (venv activated):
> ```
> python src/run_nlp_pipeline.py
> ```

| Step | What is shown |
|------|---------------|
| **B2** | Sentiment score overview (counts, mean) |
| **B3** | Descriptive analysis — distribution, observer, session, autism level, mismatches |
| **B4** | Ground-truth validation sample (download for manual coding) |
| **B5** | Validation metrics after manual coding (upload coded CSV here) |
| **B6** | Spearman correlation — RQ2: sentiment vs engagement |
| **B7** | Dual trajectory plot — session-level comparison |
| **B8** | Final Gold dataset with Sentiment_Score |
""")

    if not _pipeline_ran():
        st.warning("Pipeline results not found. Run `python src/run_nlp_pipeline.py` to generate them.")

    st.divider()

    tab_b2, tab_b3, tab_b4, tab_b5, tab_b6, tab_b7, tab_b8 = st.tabs([
        "B2 · Overview",
        "B3 · Descriptive",
        "B4 · Validation Export",
        "B5 · Validation Metrics",
        "B6 · RQ2 Correlation",
        "B7 · Dual Trajectory",
        "B8 · Gold Dataset",
    ])

    with tab_b2: _render_b2()
    with tab_b3: _render_b3()
    with tab_b4: _render_b4()
    with tab_b5: _render_b5()
    with tab_b6: _render_b6()
    with tab_b7: _render_b7()
    with tab_b8: _render_b8()
