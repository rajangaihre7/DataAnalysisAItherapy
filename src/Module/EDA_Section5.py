"""
EDA — Section 5: Observer Type Comparison (T vs P)
====================================================
Sub-sections:
  5.1  Overall Comparison — all 21 items: T mean, P mean, diff, Cohen's d, Wilcoxon p
  5.2  Session-by-Session for Q1 — convergence pattern table + plot
  5.3  Divergence by Item — bar chart sorted by |Cohen's d|
  5.4  T vs P Scatter — T mean (x) vs P mean (y) per item with diagonal reference line
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from scipy import stats

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    QUESTION_MAPPING,
    ITEM_DESC,
    QUANT_ITEMS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_quant_cols(df):
    """Return list of (full_question_col, q_label) for quant items present in df."""
    return [
        (col, q_label)
        for col, q_label in QUESTION_MAPPING.items()
        if q_label in QUANT_ITEMS and col in df.columns
    ]


def _cohens_d(a, b):
    """Cohen's d = (mean_b - mean_a) / pooled SD."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_sd == 0:
        return np.nan
    return (b.mean() - a.mean()) / pooled_sd


def _wilcoxon_p(a, b):
    """
    Wilcoxon signed-rank test on paired residuals.
    Uses Mann-Whitney U when lengths differ (unpaired fallback).
    """
    a, b = a.dropna(), b.dropna()
    if len(a) < 4 or len(b) < 4:
        return np.nan
    try:
        if len(a) == len(b):
            _, p = stats.wilcoxon(a.values, b.values, zero_method='wilcox')
        else:
            _, p = stats.mannwhitneyu(a.values, b.values, alternative='two-sided')
        return p
    except Exception:
        return np.nan


def _effect_label(d):
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return f"{d:.2f} (neg)"
    elif ad < 0.5:
        return f"{d:.2f} (small)"
    elif ad < 0.8:
        return f"{d:.2f} (med)"
    else:
        return f"{d:.2f} (large)"


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)


# ── 5.1: Overall Comparison ───────────────────────────────────────────────────

def _section_5_1(df, df_t, df_p):
    st.markdown("### 5.1 Overall Observer Comparison — All 21 Items")
    st.markdown(
        "*T Mean / P Mean / Δ (P − T) / Cohen's d / Wilcoxon (or Mann-Whitney U) p-value*"
    )

    rows = []
    for col, q_label in _get_quant_cols(df):
        desc = ITEM_DESC.get(q_label, q_label)
        t_vals = df_t[col]
        p_vals = df_p[col]
        t_mean = t_vals.dropna().mean()
        p_mean = p_vals.dropna().mean()
        diff   = p_mean - t_mean if not (np.isnan(t_mean) or np.isnan(p_mean)) else np.nan
        d      = _cohens_d(t_vals, p_vals)
        pval   = _wilcoxon_p(t_vals, p_vals)

        rows.append({
            "Item":        q_label,
            "Description": desc,
            "T N":         int(t_vals.dropna().count()),
            "T Mean":      round(t_mean, 2) if not np.isnan(t_mean) else "N/A",
            "T SD":        round(t_vals.dropna().std(), 2) if t_vals.dropna().count() > 1 else "N/A",
            "P N":         int(p_vals.dropna().count()),
            "P Mean":      round(p_mean, 2) if not np.isnan(p_mean) else "N/A",
            "P SD":        round(p_vals.dropna().std(), 2) if p_vals.dropna().count() > 1 else "N/A",
            "Δ (P−T)":     round(diff, 2) if not np.isnan(diff) else "N/A",
            "Cohen's d":   _effect_label(d),
            "p-value":     f"{pval:.4f}" if not np.isnan(pval) else "N/A",
            "Sig (p<.05)": "✓" if (not np.isnan(pval) and pval < 0.05) else "",
        })

    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    n_sig = result_df["Sig (p<.05)"].eq("✓").sum()
    st.markdown(
        f"**{n_sig} of {len(result_df)} items** show a statistically significant "
        "T vs P difference (p < .05)."
    )
    st.markdown(
        "**Effect size guide:** neg = |d| < 0.2 · small = 0.2–0.5 · med = 0.5–0.8 · large ≥ 0.8"
    )
    return result_df


# ── 5.2: Session-by-Session for Q1 ───────────────────────────────────────────

def _section_5_2(df, df_t, df_p):
    st.markdown("### 5.2 Session-by-Session Comparison — Q1 (Engagement)")
    st.markdown("*T Mean vs P Mean per session — shows convergence / divergence pattern*")

    q1_col = next(
        (col for col, q_label in QUESTION_MAPPING.items() if q_label == "Q1" and col in df.columns),
        None,
    )
    if q1_col is None:
        st.warning("Q1 column not found in data.")
        return

    session_col = "Session number"
    sessions = sorted(df[session_col].dropna().unique())

    rows = []
    t_means, p_means = [], []
    for s in sessions:
        t_vals = df_t.loc[df_t[session_col] == s, q1_col].dropna()
        p_vals = df_p.loc[df_p[session_col] == s, q1_col].dropna()
        t_m = t_vals.mean() if len(t_vals) > 0 else np.nan
        p_m = p_vals.mean() if len(p_vals) > 0 else np.nan
        diff = round(p_m - t_m, 2) if not (np.isnan(t_m) or np.isnan(p_m)) else np.nan
        pval = _wilcoxon_p(t_vals, p_vals)
        rows.append({
            "Session": int(s),
            "T N":     int(len(t_vals)),
            "T Mean":  round(t_m, 2) if not np.isnan(t_m) else "N/A",
            "P N":     int(len(p_vals)),
            "P Mean":  round(p_m, 2) if not np.isnan(p_m) else "N/A",
            "Diff (P−T)": diff if not np.isnan(diff) else "N/A",
            "p-value": f"{pval:.4f}" if not np.isnan(pval) else "N/A",
        })
        t_means.append(t_m)
        p_means.append(p_m)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Line plot
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    ax.plot(sessions, t_means, marker='o', linewidth=2.3, markersize=8,
            color='#0066CC', label='Therapist (T)',
            markeredgewidth=1.5, markeredgecolor='white')
    ax.plot(sessions, p_means, marker='s', linewidth=2.3, markersize=8,
            color='#CC0000', linestyle='--', label='Parent (P)',
            markeredgewidth=1.5, markeredgecolor='white')

    # Shade gap between lines
    valid = [(s, t, p) for s, t, p in zip(sessions, t_means, p_means)
             if not (np.isnan(t) or np.isnan(p))]
    if valid:
        sv, tv, pv = zip(*valid)
        ax.fill_between(sv, tv, pv, alpha=0.10, color='#888888')

    _style_ax(ax, "Q1 Engagement — T vs P Mean per Session",
              xlabel="Session", ylabel="Mean Score (0–4)")
    ax.legend(fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 5.3: Divergence by Item ───────────────────────────────────────────────────

def _section_5_3(df, df_t, df_p):
    st.markdown("### 5.3 Divergence by Item — |Cohen's d| Sorted")
    st.markdown("*Items sorted by absolute effect size; blue = T > P, red = P > T*")

    items, labels, d_vals, colors = [], [], [], []
    for col, q_label in _get_quant_cols(df):
        d = _cohens_d(df_t[col], df_p[col])
        if np.isnan(d):
            continue
        items.append(q_label)
        labels.append(ITEM_DESC.get(q_label, q_label))
        d_vals.append(d)
        colors.append('#CC0000' if d > 0 else '#0066CC')

    if not items:
        st.info("No valid Cohen's d values to display.")
        return

    # Sort by |d| descending
    order = np.argsort(np.abs(d_vals))[::-1]
    items_s  = [items[i]  for i in order]
    labels_s = [labels[i] for i in order]
    d_s      = [d_vals[i] for i in order]
    colors_s = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(items_s) * 0.42)))
    fig.patch.set_facecolor('white')

    y_pos = range(len(items_s))
    bars = ax.barh(y_pos, d_s, color=colors_s, edgecolor='white', linewidth=0.5, height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{lbl}" for lbl in labels_s],
        fontsize=9,
    )
    ax.axvline(0, color='#333333', linewidth=1.0)
    ax.axvline(0.2,  color='grey', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axvline(-0.2, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axvline(0.5,  color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(-0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(0.8,  color='grey', linewidth=0.8, linestyle='-.',  alpha=0.6)
    ax.axvline(-0.8, color='grey', linewidth=0.8, linestyle='-.',  alpha=0.6)

    for bar, d_val in zip(bars, d_s):
        xoff = 0.03 if d_val >= 0 else -0.03
        ha   = 'left' if d_val >= 0 else 'right'
        ax.text(d_val + xoff, bar.get_y() + bar.get_height() / 2,
                f"{d_val:+.2f}", va='center', ha=ha, fontsize=8.5, fontweight='bold',
                color='#222222')

    blue_patch = mpatches.Patch(color='#0066CC', label="T > P (d < 0)")
    red_patch  = mpatches.Patch(color='#CC0000', label="P > T (d > 0)")
    ax.legend(handles=[blue_patch, red_patch], fontsize=9, frameon=True)

    _style_ax(ax, "Observer Divergence per Item (|Cohen's d|, P − T direction)",
              xlabel="Cohen's d  (P − T)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 5.4: T vs P Scatter ───────────────────────────────────────────────────────

def _section_5_4(df, df_t, df_p):
    st.markdown("### 5.4 T vs P Mean Score Scatter Plot")
    st.markdown(
        "*Each dot = one item. Points above the diagonal → parents rate higher; "
        "below → therapists rate higher.*"
    )

    x_vals, y_vals, item_labels = [], [], []
    for col, q_label in _get_quant_cols(df):
        t_m = df_t[col].dropna().mean()
        p_m = df_p[col].dropna().mean()
        if np.isnan(t_m) or np.isnan(p_m):
            continue
        x_vals.append(t_m)
        y_vals.append(p_m)
        item_labels.append(q_label)

    if not x_vals:
        st.info("No data to display.")
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')

    # Diagonal reference line
    all_vals = np.concatenate([x_vals, y_vals])
    lim_min = max(0, all_vals.min() - 0.3)
    lim_max = all_vals.max() + 0.3
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='#888888', linewidth=1.4, linestyle='--', label='T = P (no difference)')

    # Colour by difference direction
    point_colors = ['#CC0000' if p > t else '#0066CC'
                    for t, p in zip(x_vals, y_vals)]
    ax.scatter(x_vals, y_vals, c=point_colors, s=90, zorder=3,
               edgecolors='white', linewidths=0.8)

    # Annotate each point with Q label
    for x, y, lbl in zip(x_vals, y_vals, item_labels):
        ax.annotate(lbl, (x, y), fontsize=8.5, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points', color='#222222')

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    blue_patch = mpatches.Patch(color='#0066CC', label="T > P")
    red_patch  = mpatches.Patch(color='#CC0000', label="P > T")
    ax.legend(handles=[blue_patch, red_patch,
                        plt.Line2D([0], [0], color='#888888', linestyle='--', label='T = P')],
              fontsize=9, frameon=True)

    _style_ax(ax, "T Mean vs P Mean per Item",
              xlabel="Therapist (T) Mean Score",
              ylabel="Parent (P) Mean Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Main Display Function ─────────────────────────────────────────────────────

def display(df, scale_map=None):
    """Section 5 entry point."""
    st.markdown("## 5. Observer Type Comparison (T vs P)")
    st.markdown("---")

    observer_col = "Submitted_by"
    df_t = df[df[observer_col].str.upper().str.strip() == "T"].copy()
    df_p = df[df[observer_col].str.upper().str.strip() == "P"].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Therapist (T) Records", len(df_t))
    col2.metric("Parent (P) Records",    len(df_p))
    col3.metric("Total Records",         len(df))
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "5.1 Overall Comparison",
        "5.2 Q1 Session-by-Session",
        "5.3 Divergence by Item",
        "5.4 T vs P Scatter",
    ])

    with tab1:
        _section_5_1(df, df_t, df_p)
    with tab2:
        _section_5_2(df, df_t, df_p)
    with tab3:
        _section_5_3(df, df_t, df_p)
    with tab4:
        _section_5_4(df, df_t, df_p)
