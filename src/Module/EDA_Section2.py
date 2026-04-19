"""
EDA — Section 2: Missing Data Profiling
========================================
Comprehensive analysis of missing data patterns across items, observers, and sessions.

Sub-sections:
  2.1  Per-Item Missing Summary (21 Q items + Q12, Q15, Q16, Q17 + Additional_Notes)
  2.2  Missing by Observer Type (T vs P split)
  2.3  Missing by Session (totals across all items)
  2.4  Missing Data Heatmap (items × sessions, split by T/P)
  2.5  Notes Availability (by observer, word count distribution)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
import seaborn as sns

# Add current directory to path for dynamic module loading
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    get_all_items_for_analysis, 
    get_session_column, 
    get_observer_column, 
    get_additional_notes_column,
    QUANT_ITEMS,
    QUESTION_MAPPING,
    ITEM_DESC
)


# ── Tufte Style Helpers ───────────────────────────────────────────────────────

# Muted, purposeful palette — avoids saturated colors
TUFTE_PALETTE = [
    "#4878CF",   # steel blue   (primary)
    "#D65F5F",   # muted red    (contrast / flag)
    "#6ACC65",   # sage green
    "#B47CC7",   # muted purple
    "#C4AD66",   # warm tan
    "#77BEDB",   # sky blue
    "#4a4a4a",   # dark grey
]
T_BLUE   = TUFTE_PALETTE[0]
T_RED    = TUFTE_PALETTE[1]
T_GREEN  = TUFTE_PALETTE[2]
T_PURPLE = TUFTE_PALETTE[3]
T_TAN    = TUFTE_PALETTE[4]


def _tufte_ax(ax, keep_left=True, keep_bottom=True):
    """Apply Tufte's data-ink principles to a Matplotlib Axes."""
    for spine in ax.spines.values():
        spine.set_visible(False)

    if keep_bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("#cccccc")

    if keep_left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["left"].set_color("#cccccc")

    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.set_tick_params(labelsize=9, labelcolor="#555555")
    ax.xaxis.set_tick_params(labelsize=10, labelcolor="#333333")
    ax.set_facecolor("white")


# ── Helper Functions ──────────────────────────────────────────────────────────

def _load_silver(df):
    """Load silver CSV or use passed DataFrame."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    silver_path = os.path.join(script_dir, "..", "..", "Data", "Silver", "data_silver_cleaned.csv")
    if os.path.exists(silver_path):
        try:
            full = pd.read_csv(silver_path, encoding="cp1252")
            full.columns = full.columns.str.strip()
        except Exception:
            full = df.copy()
    else:
        full = df.copy()
    
    full.columns = full.columns.str.strip()
    return full



# ── 2.1: Per-Item Missing Summary ─────────────────────────────────────────────

def _section_2_1(df):
    """Display per-item missing data summary."""
    st.markdown("### 2.1 Per-Item Missing Summary")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    items_to_check = full_questions + ([notes["additional_notes"]] if notes.get("additional_notes") else [])
    
    rows = []
    for col in items_to_check:
        if col not in df.columns:
            continue
        valid_n = df[col].notna().sum()
        missing_n = df[col].isna().sum()
        missing_pct = (missing_n / len(df) * 100) if len(df) > 0 else 0
        
        rows.append({
            "Item": col,
            "Valid N": int(valid_n),
            "Missing N": int(missing_n),
            "Missing %": f"{missing_pct:.1f}%"
        })
    
    missing_df = pd.DataFrame(rows)
    st.dataframe(missing_df, use_container_width=True, height=400)
    
    # Summary stats
    total_missing = missing_df["Missing N"].sum()
    max_missing_pct = float(missing_df["Missing %"].str.rstrip("%").astype(float).max())
    max_missing_item = missing_df.loc[missing_df["Missing %"].str.rstrip("%").astype(float).idxmax(), "Item"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Missing Cells", int(total_missing))
    col2.metric("Max Missing %", f"{max_missing_pct:.1f}%")
    col3.metric("Item with Most Missing", max_missing_item)


# ── 2.2: Missing by Observer Type ─────────────────────────────────────────────

def _section_2_2(df):
    """Split missing data by observer type (T vs P)."""
    st.markdown("### 2.2 Missing by Observer Type (T vs P)")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    items_to_check = full_questions
    observer_col = metadata["observer"]
    
    # Split by observer
    df_t = df[df[observer_col].str.upper().str.strip() == "T"]
    df_p = df[df[observer_col].str.upper().str.strip() == "P"]
    
    rows = []
    for col in items_to_check:
        if col not in df.columns:
            continue
        t_missing = df_t[col].isna().sum()
        p_missing = df_p[col].isna().sum()
        
        t_pct = (t_missing / len(df_t) * 100) if len(df_t) > 0 else 0
        p_pct = (p_missing / len(df_p) * 100) if len(df_p) > 0 else 0
        
        rows.append({
            "Item": col,
            "T Missing": int(t_missing),
            "T %": f"{t_pct:.1f}%",
            "P Missing": int(p_missing),
            "P %": f"{p_pct:.1f}%"
        })
    
    observer_df = pd.DataFrame(rows)
    st.dataframe(observer_df, use_container_width=True, height=400)
    
    # Highlight items where parents omit more
    st.markdown("#### Items with Higher Missing % from Parents (P > T)")
    observer_df["T_pct_num"] = observer_df["T %"].str.rstrip("%").astype(float)
    observer_df["P_pct_num"] = observer_df["P %"].str.rstrip("%").astype(float)
    higher_p = observer_df[observer_df["P_pct_num"] > observer_df["T_pct_num"]][["Item", "T %", "P %"]]
    
    if len(higher_p) > 0:
        st.dataframe(higher_p, use_container_width=True)
    else:
        st.info("No items have higher missing % from parents than therapists.")


# ── 2.3: Missing by Session ───────────────────────────────────────────────────

def _section_2_3(df):
    """Total missing values per session."""
    st.markdown("### 2.3 Missing by Session")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    items_to_check = full_questions
    session_col = metadata["session"]
    
    df_clean = df.dropna(subset=[session_col]).copy()
    df_clean[session_col] = df_clean[session_col].astype(int)
    
    session_missing = []
    for session in sorted(df_clean[session_col].unique()):
        df_sess = df_clean[df_clean[session_col] == session]
        total_cells = len(df_sess) * len(items_to_check)
        missing_cells = sum(df_sess[col].isna().sum() for col in items_to_check if col in df.columns)
        
        session_missing.append({
            "Session": int(session),
            "Records": len(df_sess),
            "Total Cells": total_cells,
            "Missing Count": missing_cells,
            "Missing %": f"{missing_cells / total_cells * 100:.1f}%" if total_cells > 0 else "0%"
        })
    
    session_df = pd.DataFrame(session_missing)
    st.dataframe(session_df, use_container_width=True, height=300)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    
    sessions = session_df["Session"].values
    missing_pcts = session_df["Missing %"].str.rstrip("%").astype(float).values
    
    bars = ax.bar(sessions, missing_pcts, color=T_BLUE, edgecolor="none", width=0.6)
    for bar, pct in zip(bars, missing_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, color="#222222")
    
    ax.set_xlabel("Session Number", fontsize=11, color="#555555")
    ax.set_ylabel("Missing %", fontsize=11, color="#555555")
    ax.set_title("Missing Data % by Session", fontsize=12, pad=10, color="#333333")
    ax.set_ylim(0, missing_pcts.max() + 5)
    _tufte_ax(ax)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── 2.4: Missing Data Heatmap ─────────────────────────────────────────────────

def _section_2_4(df):
    """Missing data heatmap: items × sessions, split by observer type."""
    st.markdown("### 2.4 Missing Data Heatmap (Items × Sessions)")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    items_to_check = full_questions
    session_col = metadata["session"]
    observer_col = metadata["observer"]
    
    df_clean = df.dropna(subset=[session_col]).copy()
    df_clean[session_col] = df_clean[session_col].astype(int)
    df_clean["Observer"] = df_clean[observer_col].str.upper().str.strip()
    
    # Create heatmap data: rows=items, cols=sessions, values=missing%
    sessions = sorted(df_clean[session_col].unique())
    heatmap_data_t = []
    heatmap_data_p = []
    
    for item in items_to_check:
        if item not in df.columns:
            continue
        row_t = []
        row_p = []
        
        for session in sessions:
            df_sess = df_clean[df_clean[session_col] == session]
            df_sess_t = df_sess[df_sess["Observer"] == "T"]
            df_sess_p = df_sess[df_sess["Observer"] == "P"]
            
            missing_t = df_sess_t[item].isna().sum()
            missing_p = df_sess_p[item].isna().sum()
            
            pct_t = (missing_t / len(df_sess_t) * 100) if len(df_sess_t) > 0 else 0
            pct_p = (missing_p / len(df_sess_p) * 100) if len(df_sess_p) > 0 else 0
            
            row_t.append(pct_t)
            row_p.append(pct_p)
        
        heatmap_data_t.append(row_t)
        heatmap_data_p.append(row_p)
    
    # Build dataframes with valid items
    valid_items = [item for item in items_to_check if item in df.columns]
    
    # Map column names to Q labels and then to item descriptions
    item_labels = []
    for item in valid_items:
        q_label = QUESTION_MAPPING.get(item, item)
        item_desc = ITEM_DESC.get(q_label, q_label)
        item_labels.append(item_desc)
    
    heatmap_df_t = pd.DataFrame(heatmap_data_t, index=item_labels,
                                columns=[f"S{int(s)}" for s in sessions])
    heatmap_df_p = pd.DataFrame(heatmap_data_p, index=item_labels,
                                columns=[f"S{int(s)}" for s in sessions])
    
    # Display heatmaps
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Therapist (T) — Missing %")
        fig, ax = plt.subplots(figsize=(10, 12))
        fig.patch.set_facecolor("white")
        sns.heatmap(heatmap_df_t, annot=True, fmt=".0f", cmap="RdYlGn_r", 
                    cbar_kws={"label": "Missing %"}, ax=ax, vmin=0, vmax=100,
                    linewidths=0.5, linecolor="#e0e0e0")
        ax.set_title("Therapist Missing Data % (Items × Sessions)", fontsize=12, pad=10, color="#333333")
        ax.set_xlabel("Session", fontsize=11, color="#555555")
        ax.set_ylabel("Item", fontsize=11, color="#555555")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("#### Parent (P) — Missing %")
        fig, ax = plt.subplots(figsize=(10, 12))
        fig.patch.set_facecolor("white")
        sns.heatmap(heatmap_df_p, annot=True, fmt=".0f", cmap="RdYlGn_r",
                    cbar_kws={"label": "Missing %"}, ax=ax, vmin=0, vmax=100,
                    linewidths=0.5, linecolor="#e0e0e0")
        ax.set_title("Parent Missing Data % (Items × Sessions)", fontsize=12, pad=10, color="#333333")
        ax.set_xlabel("Session", fontsize=11, color="#555555")
        ax.set_ylabel("Item", fontsize=11, color="#555555")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── 2.5: Notes Availability ──────────────────────────────────────────────────

def _section_2_5(df):
    """Notes availability by observer and word count distribution."""
    st.markdown("### 2.5 Notes Availability & Word Count Distribution")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    notes_col = notes.get("additional_notes")
    observer_col = metadata["observer"]
    
    if not notes_col:
        st.warning("No notes column found in data.")
        return
    
    df_clean = df.dropna(subset=[notes_col]).copy()
    df_clean["Observer"] = df_clean[observer_col].str.upper().str.strip()
    df_clean["Word_Count"] = df_clean[notes_col].astype(str).str.split().apply(len)
    
    # By observer
    t_notes = df_clean[df_clean["Observer"] == "T"]
    p_notes = df_clean[df_clean["Observer"] == "P"]
    
    st.markdown("#### Notes Count by Observer")
    notes_summary = pd.DataFrame({
        "Observer": ["Therapist (T)", "Parent (P)"],
        "Notes Count": [len(t_notes), len(p_notes)],
        "Total Records": [len(df[df[observer_col].str.upper().str.strip() == "T"]),
                         len(df[df[observer_col].str.upper().str.strip() == "P"])],
    })
    notes_summary["Coverage %"] = (notes_summary["Notes Count"] / notes_summary["Total Records"] * 100).apply(lambda x: f"{x:.1f}%")
    st.dataframe(notes_summary, use_container_width=False)
    
    # Word count statistics — handle empty arrays
    st.markdown("#### Word Count Statistics")
    t_wc = t_notes["Word_Count"].values
    p_wc = p_notes["Word_Count"].values
    
    # Helper function to safely compute stats
    def safe_stats(arr):
        if len(arr) == 0:
            return ["N/A", "N/A", "N/A", "N/A", "N/A"]
        return [
            f"{arr.mean():.1f}",
            f"{np.median(arr):.1f}",
            f"{int(arr.min())}",
            f"{int(arr.max())}",
            f"{arr.std():.1f}"
        ]
    
    wc_stats = pd.DataFrame({
        "Metric": ["Mean", "Median", "Min", "Max", "SD"],
        "Therapist (T)": safe_stats(t_wc),
        "Parent (P)": safe_stats(p_wc)
    })
    st.dataframe(wc_stats, use_container_width=False)
    
    # Visualize word count distribution — skip if no data
    if len(t_wc) == 0 and len(p_wc) == 0:
        st.info("No notes found to visualize word count distribution.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    
    # Therapist histogram
    ax1 = axes[0]
    if len(t_wc) > 0:
        ax1.hist(t_wc, bins=30, color=T_BLUE, edgecolor="white", alpha=0.8)
        ax1.axvline(t_wc.mean(), color=T_RED, linestyle="--", linewidth=2, label=f"Mean: {t_wc.mean():.1f}")
        ax1.set_xlabel("Word Count", fontsize=11, color="#555555")
    ax1.set_ylabel("Frequency", fontsize=11, color="#555555")
    ax1.set_title(f"Therapist Notes Word Count (n={len(t_wc)})", fontsize=12, pad=10, color="#333333")
    ax1.legend(fontsize=10, frameon=False)
    _tufte_ax(ax1)
    
    # Parent histogram
    ax2 = axes[1]
    if len(p_wc) > 0:
        ax2.hist(p_wc, bins=30, color=T_GREEN, edgecolor="white", alpha=0.8)
        ax2.axvline(p_wc.mean(), color=T_RED, linestyle="--", linewidth=2, label=f"Mean: {p_wc.mean():.1f}")
        ax2.set_xlabel("Word Count", fontsize=11, color="#555555")
        ax2.legend(fontsize=10, frameon=False)
    ax2.set_ylabel("Frequency", fontsize=11, color="#555555")
    ax2.set_title(f"Parent Notes Word Count (n={len(p_wc)})", fontsize=12, pad=10, color="#333333")
    _tufte_ax(ax2)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def display(df, scale_map=None):
    """Section 2: Missing Data Profiling."""
    st.markdown("## Section 2 — Missing Data Profiling")
    
    full = _load_silver(df)
    
    _section_2_1(full)
    st.markdown("---")
    _section_2_2(full)
    st.markdown("---")
    _section_2_3(full)
    st.markdown("---")
    _section_2_4(full)
    st.markdown("---")
    _section_2_5(full)
