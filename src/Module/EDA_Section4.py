"""
EDA — Section 4: Longitudinal Trajectories (Raw Scales)
=========================================================
Sub-sections:
  4.1  Overall Trends — 6 key items, mean per session with 95% CI
  4.2  By Observer Type (T vs P) — 6 key items, separate lines
  4.3  By Autism Level — 6 key items, Level 1 / 2 / 3 lines
  4.4  By Age Group — 3 items (Q1, Q9, Q22), age-band lines
  4.5  Engagement Success Rate — mean per session trajectory
  4.6  Response Time — mean per session from response_time_cleaned.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys
from scipy import stats

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    QUESTION_MAPPING,
    ITEM_DESC,
    KEY_ITEMS,
)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = ['#0066CC', '#CC0000', '#009900', '#FF8800', '#8800CC', '#00AAAA']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

KEY_ITEM_COLS = {q_label: col for col, q_label in QUESTION_MAPPING.items()
                 if q_label in KEY_ITEMS}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ci95(series):
    """95% CI half-width for a series."""
    valid = series.dropna()
    n = len(valid)
    if n < 2:
        return 0.0
    se = valid.std(ddof=1) / np.sqrt(n)
    return se * stats.t.ppf(0.975, df=n - 1)


def _style_ax(ax, title, xlabel="Session", ylabel="Mean Score"):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)


def _get_age_group(age):
    if pd.isna(age):
        return None
    age = int(age)
    if 8 <= age <= 14:
        return "8-14"
    elif 15 <= age <= 19:
        return "15-19"
    elif 20 <= age <= 26:
        return "20-26"
    return None


# ── 4.1: Overall Trends ───────────────────────────────────────────────────────

def _section_4_1(df):
    st.markdown("### 4.1 Overall Trends — 6 Key Items")
    st.markdown("*Mean score per session (sessions 2-9) across all observers, with 95% CI shading*")

    session_col = "Session number"
    sessions = sorted(df[session_col].dropna().unique())

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')

    for idx, q_label in enumerate(KEY_ITEMS):
        col = KEY_ITEM_COLS.get(q_label)
        if col is None or col not in df.columns:
            continue

        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        desc = ITEM_DESC.get(q_label, q_label)

        means, cis, sess_list = [], [], []
        for s in sessions:
            vals = df.loc[df[session_col] == s, col].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                cis.append(_ci95(vals))
                sess_list.append(s)

        means = np.array(means)
        cis = np.array(cis)

        ax.plot(sess_list, means, marker=marker, linewidth=2.2, markersize=8,
                label=f"{q_label}: {desc}", color=color,
                markeredgewidth=1.5, markeredgecolor='white')
        ax.fill_between(sess_list, means - cis, means + cis, alpha=0.12, color=color)

    _style_ax(ax, "Overall Longitudinal Trends — 6 Key Items (All Observers)")
    ax.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 4.2: By Observer Type ─────────────────────────────────────────────────────

def _section_4_2(df):
    st.markdown("### 4.2 Trends by Observer Type (T vs P)")
    st.markdown("*Same 6 key items — solid = Therapist, dashed = Parent*")

    session_col = "Session number"
    observer_col = "Submitted_by"
    sessions = sorted(df[session_col].dropna().unique())

    df_t = df[df[observer_col].str.upper().str.strip() == "T"]
    df_p = df[df[observer_col].str.upper().str.strip() == "P"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for idx, q_label in enumerate(KEY_ITEMS):
        col = KEY_ITEM_COLS.get(q_label)
        ax = axes[idx]
        if col is None or col not in df.columns:
            ax.set_visible(False)
            continue

        desc = ITEM_DESC.get(q_label, q_label)

        for (subset, label, ls, color) in [
            (df_t, "Therapist (T)", '-',  '#0066CC'),
            (df_p, "Parent (P)",    '--', '#CC0000'),
        ]:
            means, cis, sess_list = [], [], []
            for s in sessions:
                vals = subset.loc[subset[session_col] == s, col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    cis.append(_ci95(vals))
                    sess_list.append(s)
            if not means:
                continue
            means = np.array(means)
            cis = np.array(cis)
            ax.plot(sess_list, means, marker='o', linewidth=2.0, linestyle=ls,
                    markersize=6, label=label, color=color,
                    markeredgewidth=1.2, markeredgecolor='white')
            ax.fill_between(sess_list, means - cis, means + cis, alpha=0.12, color=color)

        _style_ax(ax, f"{q_label}: {desc}")
        ax.legend(fontsize=8, frameon=True, framealpha=0.9)

    plt.suptitle("Longitudinal Trends by Observer Type (T vs P)", fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 4.3: By Autism Level ──────────────────────────────────────────────────────

def _section_4_3(df):
    st.markdown("### 4.3 Trends by Autism Level")
    st.markdown("*6 key items — separate lines per autism level (1 / 2 / 3)*")

    session_col = "Session number"
    autism_col  = "Autism Level"
    sessions    = sorted(df[session_col].dropna().unique())
    levels      = sorted(df[autism_col].dropna().unique())
    level_colors = {1: '#0066CC', 2: '#CC0000', 3: '#009900'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for idx, q_label in enumerate(KEY_ITEMS):
        col = KEY_ITEM_COLS.get(q_label)
        ax = axes[idx]
        if col is None or col not in df.columns:
            ax.set_visible(False)
            continue

        desc = ITEM_DESC.get(q_label, q_label)

        for level in levels:
            subset = df[df[autism_col] == level]
            color = level_colors.get(level, COLORS[int(level) % len(COLORS)])
            means, cis, sess_list = [], [], []
            for s in sessions:
                vals = subset.loc[subset[session_col] == s, col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    cis.append(_ci95(vals))
                    sess_list.append(s)
            if not means:
                continue
            means = np.array(means)
            cis = np.array(cis)
            ax.plot(sess_list, means, marker='o', linewidth=2.0, markersize=6,
                    label=f"Level {int(level)}", color=color,
                    markeredgewidth=1.2, markeredgecolor='white')
            ax.fill_between(sess_list, means - cis, means + cis, alpha=0.12, color=color)

        _style_ax(ax, f"{q_label}: {desc}")
        ax.legend(fontsize=8, frameon=True, framealpha=0.9)

    plt.suptitle("Longitudinal Trends by Autism Level", fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 4.4: By Age Group ─────────────────────────────────────────────────────────

def _section_4_4(df):
    st.markdown("### 4.4 Trends by Age Group")
    st.markdown("*3 items (Q1, Q9, Q22) — lines for age bands 8-14, 15-19, 20-26*")

    session_col = "Session number"
    age_col     = "Age"
    age_groups  = ["8-14", "15-19", "20-26"]
    age_colors  = {"8-14": '#0066CC', "15-19": '#CC0000', "20-26": '#009900'}
    items_44    = ["Q1", "Q9", "Q22"]

    df = df.copy()
    df['_age_group'] = df[age_col].apply(_get_age_group)
    sessions = sorted(df[session_col].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor('white')

    for idx, q_label in enumerate(items_44):
        col = KEY_ITEM_COLS.get(q_label)
        ax = axes[idx]
        if col is None or col not in df.columns:
            ax.set_visible(False)
            continue

        desc = ITEM_DESC.get(q_label, q_label)

        for grp in age_groups:
            subset = df[df['_age_group'] == grp]
            if subset.empty:
                continue
            color = age_colors[grp]
            means, cis, sess_list = [], [], []
            for s in sessions:
                vals = subset.loc[subset[session_col] == s, col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    cis.append(_ci95(vals))
                    sess_list.append(s)
            if not means:
                continue
            means = np.array(means)
            cis = np.array(cis)
            ax.plot(sess_list, means, marker='o', linewidth=2.0, markersize=7,
                    label=grp, color=color,
                    markeredgewidth=1.2, markeredgecolor='white')
            ax.fill_between(sess_list, means - cis, means + cis, alpha=0.12, color=color)

        _style_ax(ax, f"{q_label}: {desc}")
        ax.legend(fontsize=9, frameon=True, framealpha=0.9)

    plt.suptitle("Longitudinal Trends by Age Group", fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 4.5: Engagement Success Rate ─────────────────────────────────────────────

def _section_4_5(df):
    st.markdown("### 4.5 Engagement Success Rate Trajectory")
    st.markdown("*Mean Success_percentage per session with 95% CI (values 0–1 shown as 0–100%)*")

    session_col = "Session number"
    success_col = "Success_percentage"

    if success_col not in df.columns:
        st.warning(f"Column '{success_col}' not found in data.")
        return

    df_valid = df[[session_col, success_col]].dropna()
    df_valid[success_col] = pd.to_numeric(df_valid[success_col], errors='coerce') * 100
    sessions = sorted(df_valid[session_col].unique())

    means, cis = [], []
    for s in sessions:
        vals = df_valid.loc[df_valid[session_col] == s, success_col].dropna()
        means.append(vals.mean())
        cis.append(_ci95(vals))

    means = np.array(means)
    cis   = np.array(cis)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    ax.plot(sessions, means, marker='o', linewidth=2.5, markersize=9,
            color='#0066CC', markeredgewidth=1.8, markeredgecolor='white',
            label='Mean Success %')
    ax.fill_between(sessions, means - cis, means + cis, alpha=0.15, color='#0066CC')

    for s, m in zip(sessions, means):
        ax.annotate(f'{m:.1f}%', xy=(s, m), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', color='#0066CC')

    _style_ax(ax, "Engagement Success Rate per Session", ylabel="Success Rate (%)")
    ax.set_ylim(0, min(110, means.max() + 20))
    ax.legend(fontsize=10, frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 4.6: Response Time ────────────────────────────────────────────────────────

def _section_4_6():
    st.markdown("### 4.6 Response Time Trajectory")
    st.markdown("*Mean response time (seconds) per session — expected decreasing trend as participants improve*")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rt_path = os.path.join(script_dir, "..", "..", "Data", "Silver", "response_time_cleaned.csv")

    if not os.path.exists(rt_path):
        st.warning("response_time_cleaned.csv not found in Data/Silver/.")
        return

    rt_df = pd.read_csv(rt_path, encoding='cp1252')
    rt_df.columns = rt_df.columns.str.strip()

    session_col = "Session number"
    rt_col      = "response_seconds"

    if rt_col not in rt_df.columns:
        st.warning(f"Column '{rt_col}' not found in response time data.")
        return

    rt_df[rt_col] = pd.to_numeric(rt_df[rt_col], errors='coerce')
    rt_valid = rt_df[[session_col, rt_col]].dropna()
    sessions = sorted(rt_valid[session_col].unique())

    means, cis = [], []
    for s in sessions:
        vals = rt_valid.loc[rt_valid[session_col] == s, rt_col].dropna()
        means.append(vals.mean())
        cis.append(_ci95(vals))

    means = np.array(means)
    cis   = np.array(cis)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')

    ax.plot(sessions, means, marker='o', linewidth=2.5, markersize=9,
            color='#CC0000', markeredgewidth=1.8, markeredgecolor='white',
            label='Mean Response Time')
    ax.fill_between(sessions, means - cis, means + cis, alpha=0.15, color='#CC0000')

    # Trend line (linear regression)
    if len(sessions) >= 3:
        slope, intercept, r, p, _ = stats.linregress(sessions, means)
        trend_vals = [intercept + slope * s for s in sessions]
        ax.plot(sessions, trend_vals, linestyle='--', linewidth=1.6,
                color='#666666', label=f'Trend (slope={slope:.1f} s/session)')

    for s, m in zip(sessions, means):
        ax.annotate(f'{m:.0f}s', xy=(s, m), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', color='#CC0000')

    _style_ax(ax, "Mean Response Time per Session (Seconds)", ylabel="Response Time (seconds)")
    ax.legend(fontsize=10, frameon=True)
    ax.invert_yaxis()   # lower is better, so invert to show improvement going up visually
    ax.set_ylabel("Response Time (seconds) — lower = faster", fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Mini stats table
    rt_by_session = (rt_valid.groupby(session_col)[rt_col]
                     .agg(N='count', Mean='mean', SD='std', Median='median',
                          Min='min', Max='max')
                     .reset_index()
                     .rename(columns={session_col: 'Session'}))
    rt_by_session = rt_by_session.round(1)
    st.markdown("**Response Time Summary by Session**")
    st.dataframe(rt_by_session, use_container_width=True, hide_index=True)


# ── Main Display Function ─────────────────────────────────────────────────────

def display(df, scale_map=None):
    """Section 4 entry point."""
    st.markdown("## 4. Longitudinal Trajectories (Raw Scales)")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "4.1 Overall Trends",
        "4.2 By Observer",
        "4.3 By Autism Level",
        "4.4 By Age Group",
        "4.5 Success Rate",
        "4.6 Response Time",
    ])

    with tab1:
        _section_4_1(df)
    with tab2:
        _section_4_2(df)
    with tab3:
        _section_4_3(df)
    with tab4:
        _section_4_4(df)
    with tab5:
        _section_4_5(df)
    with tab6:
        _section_4_6()
