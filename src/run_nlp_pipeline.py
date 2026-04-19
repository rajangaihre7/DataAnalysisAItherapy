"""
NLP Sentiment Analysis Pipeline — Standalone Runner
====================================================
Run this script ONCE (outside Streamlit) to compute all NLP results.
The Streamlit dashboard will then load and display the saved outputs.

Usage (from the project root, with venv activated):
    python src/run_nlp_pipeline.py

Outputs written to  results/nlp/
    descriptive/
        sentiment_distribution.png
        sentiment_by_observer.png
        sentiment_by_session.png
        sentiment_by_autism.png
        mismatch_neg_sent_pos_lik.csv
        mismatch_pos_sent_neg_lik.csv
    rq2/
        sentiment_vs_engagement_scatter.png
        sentiment_item_correlation_bar.png
        sentiment_item_correlations.csv
    trajectory/
        dual_trajectory_engagement_vs_sentiment.png
        trajectory_table.csv
    ground_truth_sample.csv
    pipeline_summary.csv
    data_gold_with_sentiment.csv   (Gold + Sentiment_Score)
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, skew as _skew

# ── Path setup ─────────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_dir  = os.path.join(_script_dir, 'Module')
_project_dir = os.path.normpath(os.path.join(_script_dir, '..'))

for p in [_script_dir, _module_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from questionnaire_mapping import NOTES_COLUMNS, METADATA_COLUMNS
from silver_to_gold_transformation import (
    build_gold_df,
    NORM_ITEMS,
    SCALE_4_ITEMS,
    SCALE_10_ITEMS,
    norm_label,
    get_gold_path,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_NOTES_COL    = NOTES_COLUMNS.get('additional_notes', 'Additional_notes_observations')
_SESSION_COL  = METADATA_COLUMNS.get('session',        'Session number')
_OBSERVER_COL = METADATA_COLUMNS.get('observer',       'Submitted_by')
_PART_COL     = METADATA_COLUMNS.get('participant_id', 'Participant id')
_AUTISM_COL   = 'Autism Level'


def _results_dir(sub: str = '') -> str:
    base = os.path.join(_project_dir, 'results', 'nlp')
    path = os.path.join(base, sub) if sub else base
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# B1 — Load DistilBERT
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print("\n[B1] Loading DistilBERT model …")
    from transformers import pipeline as hf_pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"     Device: {'GPU' if device == 0 else 'CPU'}")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        top_k=None,
        device=device,
        truncation=True,
        max_length=128,
    )
    test = pipe("The participant showed good engagement today")[0]
    pos  = next(r['score'] for r in test if r['label'] == 'POSITIVE')
    print(f"     Test sentence POSITIVE score: {pos:.4f}")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# B2 — Run Sentiment Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment(gold: pd.DataFrame, pipe) -> pd.DataFrame:
    print("\n[B2] Running sentiment analysis on observation notes …")
    gold = gold.copy()
    scores = []
    total = len(gold)
    for i, (idx, row) in enumerate(gold.iterrows()):
        text = str(row.get(_NOTES_COL, '')) if pd.notna(row.get(_NOTES_COL)) else ''
        if text.strip():
            try:
                result = pipe(text)[0]
                pos = next((r['score'] for r in result if r['label'] == 'POSITIVE'), np.nan)
                scores.append(pos)
            except Exception as e:
                print(f"     Error at index {idx}: {e}")
                scores.append(np.nan)
        else:
            scores.append(np.nan)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"     Progress: {i+1}/{total}", end='\r')

    gold['Sentiment_Score'] = scores
    ss = gold['Sentiment_Score'].dropna()
    print(f"\n     Non-null: {len(ss)}  |  Mean: {ss.mean():.4f}  |  SD: {ss.std():.4f}  |  "
          f"Min: {ss.min():.4f}  |  Max: {ss.max():.4f}")
    return gold


# ══════════════════════════════════════════════════════════════════════════════
# B3 — Descriptive Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_descriptive(gold: pd.DataFrame):
    print("\n[B3] Descriptive analysis …")
    ss = gold['Sentiment_Score'].dropna()
    out = _results_dir('descriptive')

    # ── Distribution ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ss, bins=25, color='#1f77b4', edgecolor='white', alpha=0.85)
    axes[0].axvline(ss.mean(),   color='#d62728', linestyle='--', lw=1.5, label=f'Mean = {ss.mean():.3f}')
    axes[0].axvline(ss.median(), color='#ff7f0e', linestyle=':',  lw=1.5, label=f'Median = {ss.median():.3f}')
    axes[0].set_xlabel('Sentiment Score'); axes[0].set_ylabel('Count')
    axes[0].set_title('Histogram'); axes[0].legend(fontsize=8)
    sns.boxplot(x=ss, ax=axes[1], color='#aec7e8', width=0.4,
                flierprops=dict(marker='o', markersize=3, alpha=0.5))
    axes[1].set_xlabel('Sentiment Score'); axes[1].set_title('Box Plot')
    fig.suptitle('Sentiment Score Distribution', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("     Saved: sentiment_distribution.png")

    # ── Summary stats CSV ─────────────────────────────────────────────────────
    summary = pd.DataFrame([{
        'N': int(len(ss)), 'Mean': round(ss.mean(), 4), 'SD': round(ss.std(), 4),
        'Median': round(ss.median(), 4), 'Min': round(ss.min(), 4),
        'Max': round(ss.max(), 4), 'Skewness': round(float(_skew(ss)), 4),
    }])
    summary.to_csv(os.path.join(out, 'sentiment_summary_stats.csv'), index=False)
    print("     Saved: sentiment_summary_stats.csv")

    # ── By Observer ───────────────────────────────────────────────────────────
    if _OBSERVER_COL in gold.columns:
        obs_data = gold[[_OBSERVER_COL, 'Sentiment_Score']].dropna()
        obs_data[_OBSERVER_COL] = obs_data[_OBSERVER_COL].astype(str).str.strip()
        grp = obs_data.groupby(_OBSERVER_COL)['Sentiment_Score'].agg(
            N='count', Mean='mean', SD='std').reset_index()
        grp['SE'] = grp['SD'] / np.sqrt(grp['N'])
        grp.to_csv(os.path.join(out, 'sentiment_by_observer.csv'), index=False)

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(grp[_OBSERVER_COL], grp['Mean'], yerr=grp['SE'],
                      color=['#1f77b4', '#ff7f0e'][:len(grp)],
                      edgecolor='white', alpha=0.85, capsize=5)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Mean Sentiment Score (± SE)')
        ax.set_title('Sentiment Score by Observer Type')
        for bar, row in zip(bars, grp.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + row.SE + 0.01,
                    f'{row.Mean:.3f}\n(n={row.N})', ha='center', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'sentiment_by_observer.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("     Saved: sentiment_by_observer.png")

        types = grp[_OBSERVER_COL].tolist()
        if 'T' in types and 'P' in types:
            t_vals = obs_data[obs_data[_OBSERVER_COL] == 'T']['Sentiment_Score'].dropna()
            p_vals = obs_data[obs_data[_OBSERVER_COL] == 'P']['Sentiment_Score'].dropna()
            u_stat, p_val = stats.mannwhitneyu(t_vals, p_vals, alternative='two-sided')
            print(f"     Mann-Whitney U: U={u_stat:.1f}, p={p_val:.4f} "
                  f"({'significant' if p_val < 0.05 else 'not significant'} at α=0.05)")
            with open(os.path.join(out, 'observer_mannwhitney.txt'), 'w') as f:
                f.write(f"Mann-Whitney U test: T vs P observer sentiment scores\n")
                f.write(f"U = {u_stat:.1f},  p = {p_val:.4f}\n")
                f.write("Significant at α=0.05\n" if p_val < 0.05 else "Not significant\n")

    # ── By Session ────────────────────────────────────────────────────────────
    if _SESSION_COL in gold.columns:
        ses_data = gold[[_SESSION_COL, 'Sentiment_Score', 'Engagement_Score']].copy()
        ses_data[_SESSION_COL] = pd.to_numeric(ses_data[_SESSION_COL], errors='coerce')
        ses_data = ses_data.dropna(subset=[_SESSION_COL])
        sent_grp = ses_data.dropna(subset=['Sentiment_Score']).groupby(_SESSION_COL)['Sentiment_Score'].mean()
        eng_grp  = ses_data.dropna(subset=['Engagement_Score']).groupby(_SESSION_COL)['Engagement_Score'].mean()

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(eng_grp.index, eng_grp.values, 'b-o', lw=2, label='Engagement Score (Likert)')
        ax1.set_xlabel('Session'); ax1.set_ylabel('Mean Engagement Score (0-1)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue'); ax1.set_ylim(0, 1)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax2 = ax1.twinx()
        ax2.plot(sent_grp.index, sent_grp.values, 'r-s', lw=2, label='Sentiment Score (NLP)')
        ax2.set_ylabel('Mean Sentiment Score (0-1)', color='red')
        ax2.tick_params(axis='y', labelcolor='red'); ax2.set_ylim(0, 1)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.2)
        fig.suptitle('Sentiment & Engagement by Session', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'sentiment_by_session.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("     Saved: sentiment_by_session.png")

        tbl = pd.concat([sent_grp.rename('Mean Sentiment'),
                         eng_grp.rename('Mean Engagement')], axis=1).round(4)
        tbl.to_csv(os.path.join(out, 'sentiment_by_session.csv'))

    # ── By Autism Level ───────────────────────────────────────────────────────
    if _AUTISM_COL in gold.columns:
        aut_data = gold[[_AUTISM_COL, 'Sentiment_Score']].copy()
        aut_data[_AUTISM_COL] = pd.to_numeric(aut_data[_AUTISM_COL], errors='coerce')
        aut_data = aut_data.dropna()
        grp = aut_data.groupby(_AUTISM_COL)['Sentiment_Score'].agg(
            N='count', Mean='mean', SD='std').reset_index()
        grp['SE'] = grp['SD'] / np.sqrt(grp['N'])
        grp.to_csv(os.path.join(out, 'sentiment_by_autism.csv'), index=False)

        palette = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        fig, ax = plt.subplots(figsize=(max(5, len(grp) * 1.2), 4))
        bars = ax.bar(grp[_AUTISM_COL].astype(int).astype(str), grp['Mean'], yerr=grp['SE'],
                      color=palette[:len(grp)], edgecolor='white', alpha=0.85, capsize=5)
        ax.set_ylim(0, 1); ax.set_xlabel('Autism Level')
        ax.set_ylabel('Mean Sentiment Score (± SE)')
        ax.set_title('Sentiment Score by Autism Level')
        for bar, row in zip(bars, grp.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + row.SE + 0.01,
                    f'{row.Mean:.3f}\n(n={row.N})', ha='center', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'sentiment_by_autism.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("     Saved: sentiment_by_autism.png")

    # ── Mismatch Analysis ─────────────────────────────────────────────────────
    if 'Engagement_Score' in gold.columns:
        both = gold.dropna(subset=['Sentiment_Score', 'Engagement_Score', _NOTES_COL])
        neg_sent_pos_lik = both[(both['Sentiment_Score'] < 0.3) & (both['Engagement_Score'] > 0.6)]
        pos_sent_neg_lik = both[(both['Sentiment_Score'] > 0.7) & (both['Engagement_Score'] < 0.4)]
        disp_cols = [c for c in [_PART_COL, _SESSION_COL, _OBSERVER_COL,
                                  'Sentiment_Score', 'Engagement_Score', _NOTES_COL]
                     if c in gold.columns]
        if len(neg_sent_pos_lik):
            neg_sent_pos_lik[disp_cols].to_csv(
                os.path.join(out, 'mismatch_neg_sent_pos_lik.csv'), index=False)
        if len(pos_sent_neg_lik):
            pos_sent_neg_lik[disp_cols].to_csv(
                os.path.join(out, 'mismatch_pos_sent_neg_lik.csv'), index=False)
        print(f"     Mismatches — NLP neg/Likert pos: {len(neg_sent_pos_lik)} | "
              f"NLP pos/Likert neg: {len(pos_sent_neg_lik)}")


# ══════════════════════════════════════════════════════════════════════════════
# B4 — Ground-Truth Sample Export
# ══════════════════════════════════════════════════════════════════════════════

def run_ground_truth_export(gold: pd.DataFrame):
    print("\n[B4] Generating ground-truth validation sample …")
    notes_df    = gold.dropna(subset=['Sentiment_Score', _NOTES_COL]).copy()
    n_available = len(notes_df)
    sample_size = min(50, n_available)

    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        if (_OBSERVER_COL in notes_df.columns and
                notes_df[_OBSERVER_COL].nunique() > 1 and n_available > sample_size):
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
            for _, idx in splitter.split(notes_df, notes_df[_OBSERVER_COL].astype(str)):
                sample = notes_df.iloc[idx]
        else:
            sample = notes_df.sample(n=sample_size, random_state=42)
    except ImportError:
        sample = notes_df.sample(n=sample_size, random_state=42)

    export_cols = [c for c in [_PART_COL, _SESSION_COL, _OBSERVER_COL,
                                _NOTES_COL, 'Sentiment_Score'] if c in sample.columns]
    gt = sample[export_cols].copy()
    gt['Model_Predicted_Label'] = (gt['Sentiment_Score'] > 0.5).astype(int)
    gt['Manual_Label'] = ''

    save_path = os.path.join(_results_dir(), 'ground_truth_sample.csv')
    gt.to_csv(save_path, index=False)
    print(f"     Saved: ground_truth_sample.csv  ({len(gt)} notes)")
    if _OBSERVER_COL in gt.columns:
        for obs in gt[_OBSERVER_COL].dropna().unique():
            print(f"       {obs}: {(gt[_OBSERVER_COL] == obs).sum()} notes")


# ══════════════════════════════════════════════════════════════════════════════
# B6 — Spearman Correlation (RQ2)
# ══════════════════════════════════════════════════════════════════════════════

def run_rq2_correlation(gold: pd.DataFrame):
    print("\n[B6] Spearman correlation — RQ2 …")
    out  = _results_dir('rq2')
    both = gold.dropna(subset=['Sentiment_Score', 'Engagement_Score'])
    print(f"     Records with both scores: {len(both)}")

    rho, pval = spearmanr(both['Sentiment_Score'], both['Engagement_Score'])
    print(f"     Overall: ρ = {rho:.3f},  p = {pval:.4f}  "
          f"({'significant' if pval < 0.05 else 'not significant'} at α=0.05)")

    # Per observer
    obs_rows = []
    if _OBSERVER_COL in both.columns:
        for obs in both[_OBSERVER_COL].dropna().unique():
            sub = both[both[_OBSERVER_COL].astype(str).str.strip() == obs]
            if len(sub) > 10:
                r, p = spearmanr(sub['Sentiment_Score'], sub['Engagement_Score'])
                obs_rows.append({'Observer': obs, 'rho': round(r, 3), 'p': round(p, 4), 'N': len(sub)})
                print(f"     {obs} observer: ρ = {r:.3f},  p = {p:.4f},  n = {len(sub)}")
    pd.DataFrame(obs_rows).to_csv(os.path.join(out, 'rq2_by_observer.csv'), index=False)

    # Per session
    sess_rows = []
    if _SESSION_COL in both.columns:
        both_s = both.copy()
        both_s[_SESSION_COL] = pd.to_numeric(both_s[_SESSION_COL], errors='coerce')
        for sess in sorted(both_s[_SESSION_COL].dropna().unique()):
            sub = both_s[both_s[_SESSION_COL] == sess]
            if len(sub) > 5:
                r, p = spearmanr(sub['Sentiment_Score'], sub['Engagement_Score'])
                sess_rows.append({'Session': int(sess), 'rho': round(r, 3), 'p': round(p, 4), 'N': len(sub)})
                print(f"     Session {int(sess)}: ρ = {r:.3f},  p = {p:.4f},  n = {len(sub)}")
    pd.DataFrame(sess_rows).to_csv(os.path.join(out, 'rq2_by_session.csv'), index=False)

    # Per item
    item_rows = []
    for q in SCALE_4_ITEMS + SCALE_10_ITEMS:
        norm_col = f'{q}_norm'
        if norm_col in both.columns:
            sub = both.dropna(subset=[norm_col])
            if len(sub) > 10:
                r, p = spearmanr(sub['Sentiment_Score'], sub[norm_col])
                item_rows.append({
                    'Item': q, 'Label': norm_label(norm_col),
                    'Spearman_rho': round(r, 3), 'p_value': round(p, 4), 'N': len(sub),
                    'Significant': 'Yes' if p < 0.05 else 'No',
                })
    item_df = pd.DataFrame(item_rows)
    item_df.to_csv(os.path.join(out, 'sentiment_item_correlations.csv'), index=False)
    print(f"     Saved: sentiment_item_correlations.csv  ({len(item_df)} items)")

    # Summary row with overall result
    pd.DataFrame([{'rho': round(rho, 3), 'p': round(pval, 4), 'N': len(both),
                   'significant': pval < 0.05}]).to_csv(
        os.path.join(out, 'rq2_overall.csv'), index=False)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {'T': 'coral', 'P': 'steelblue'}
    if _OBSERVER_COL in both.columns:
        for obs, colour in palette.items():
            sub = both[both[_OBSERVER_COL].astype(str).str.strip() == obs]
            label = 'Therapist' if obs == 'T' else 'Parent'
            ax.scatter(sub['Sentiment_Score'], sub['Engagement_Score'],
                       alpha=0.5, s=30, color=colour, label=label)
    else:
        ax.scatter(both['Sentiment_Score'], both['Engagement_Score'], alpha=0.5, s=30)
    z = np.polyfit(both['Sentiment_Score'], both['Engagement_Score'], 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'k--', alpha=0.5, label=f'Overall: ρ={rho:.3f}')
    ax.set_xlabel('NLP Sentiment Score (DistilBERT)', fontsize=10)
    ax.set_ylabel('Composite Engagement Score (Normalised)', fontsize=10)
    ax.set_title('RQ2: NLP Sentiment vs Structured Clinical Assessment', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'sentiment_vs_engagement_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("     Saved: sentiment_vs_engagement_scatter.png")

    # Bar chart of item correlations
    if item_rows:
        sorted_df = item_df.sort_values('Spearman_rho', ascending=True)
        colours   = ['#2ca02c' if p < 0.05 else '#aec7e8' for p in sorted_df['p_value']]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(sorted_df['Label'], sorted_df['Spearman_rho'],
                color=colours, edgecolor='black', alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Spearman ρ with Sentiment Score", fontsize=9)
        ax.set_title('Correlation Between NLP Sentiment and Individual Assessment Items', fontsize=10)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#2ca02c', label='p < 0.05'),
                            Patch(color='#aec7e8', label='p ≥ 0.05')], loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'sentiment_item_correlation_bar.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("     Saved: sentiment_item_correlation_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# B7 — Dual Trajectory
# ══════════════════════════════════════════════════════════════════════════════

def run_dual_trajectory(gold: pd.DataFrame):
    print("\n[B7] Dual trajectory plot …")
    out = _results_dir('trajectory')

    if _SESSION_COL not in gold.columns:
        print("     Skipped: session column not found.")
        return

    df_s = gold.copy()
    df_s[_SESSION_COL] = pd.to_numeric(df_s[_SESSION_COL], errors='coerce')
    df_s = df_s.dropna(subset=[_SESSION_COL])

    eng_by_sess  = df_s.dropna(subset=['Engagement_Score']).groupby(_SESSION_COL)['Engagement_Score'].mean()
    sent_by_sess = df_s.dropna(subset=['Sentiment_Score']).groupby(_SESSION_COL)['Sentiment_Score'].mean() \
                   if 'Sentiment_Score' in df_s.columns else pd.Series(dtype=float)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(eng_by_sess.index, eng_by_sess.values, 'b-o', lw=2.5, markersize=6,
             label='Engagement Score (Likert)')
    ax1.set_xlabel('Session Number', fontsize=10)
    ax1.set_ylabel('Mean Engagement Score (0-1)', color='blue', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='blue'); ax1.set_ylim(0, 1)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax1.grid(True, alpha=0.2)
    ax2 = ax1.twinx()
    if not sent_by_sess.empty:
        ax2.plot(sent_by_sess.index, sent_by_sess.values, 'r-s', lw=2.5, markersize=6,
                 label='Sentiment Score (NLP)')
    ax2.set_ylabel('Mean Sentiment Score (0-1)', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red'); ax2.set_ylim(0, 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
    fig.suptitle('Dual Trajectory: Structured Assessment vs NLP Sentiment Across Sessions',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'dual_trajectory_engagement_vs_sentiment.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("     Saved: dual_trajectory_engagement_vs_sentiment.png")

    tbl = pd.concat([
        eng_by_sess.rename('Mean Engagement Score'),
        sent_by_sess.rename('Mean Sentiment Score') if not sent_by_sess.empty else pd.Series(dtype=float),
    ], axis=1).round(4)
    tbl.to_csv(os.path.join(out, 'trajectory_table.csv'))
    print("     Saved: trajectory_table.csv")


# ══════════════════════════════════════════════════════════════════════════════
# B8 — Save Gold + Sentiment
# ══════════════════════════════════════════════════════════════════════════════

def save_gold_with_sentiment(gold: pd.DataFrame):
    print("\n[B8] Saving Gold dataset with Sentiment_Score …")
    path = os.path.join(_results_dir(), 'data_gold_with_sentiment.csv')
    gold.to_csv(path, index=False)
    print(f"     Saved: data_gold_with_sentiment.csv  "
          f"(Rows: {len(gold)}, Columns: {len(gold.columns)})")

    # NOTE: We intentionally do NOT overwrite Data/Gold/data_gold_analytical.csv
    # to avoid file-lock conflicts with the running Streamlit app.

    # Pipeline summary
    es = gold['Engagement_Score'].dropna()
    ss = gold['Sentiment_Score'].dropna() if 'Sentiment_Score' in gold.columns else pd.Series(dtype=float)
    summary = pd.DataFrame([
        {'Metric': 'Total records',              'Value': len(gold)},
        {'Metric': 'Records with notes',         'Value': int(gold[_NOTES_COL].notna().sum())},
        {'Metric': 'Sentiment_Score non-null',   'Value': int(ss.notna().sum())},
        {'Metric': 'Sentiment_Score mean',       'Value': round(ss.mean(), 4) if not ss.empty else None},
        {'Metric': 'Engagement_Score non-null',  'Value': int(es.notna().sum())},
        {'Metric': 'Engagement_Score mean',      'Value': round(es.mean(), 4)},
    ])
    summary.to_csv(os.path.join(_results_dir(), 'pipeline_summary.csv'), index=False)
    print("     Saved: pipeline_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NLP Sentiment Analysis Pipeline — Part B")
    print("=" * 60)

    # Load Silver data
    silver_path = os.path.join(_project_dir, 'Data', 'Silver', 'data_silver_cleaned.csv')
    print(f"\nLoading Silver data from: {silver_path}")
    df = pd.read_csv(silver_path, encoding='latin-1')
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Build Gold layer
    print("\nBuilding Gold analytical dataset …")
    gold = build_gold_df(df)
    print(f"  Gold: {len(gold)} rows, {len(gold.columns)} columns")

    # B1 + B2
    pipe = load_model()
    gold = run_sentiment(gold, pipe)

    # B3
    run_descriptive(gold)

    # B4
    run_ground_truth_export(gold)

    # B6
    run_rq2_correlation(gold)

    # B7
    run_dual_trajectory(gold)

    # B8
    save_gold_with_sentiment(gold)

    print("\n" + "=" * 60)
    print("  Pipeline complete.  All results saved to results/nlp/")
    print("  Open the Streamlit dashboard to view results.")
    print("=" * 60)


if __name__ == '__main__':
    main()
