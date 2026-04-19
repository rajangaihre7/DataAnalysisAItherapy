"""
EDA — Section 3: Descriptive Statistics & Distribution Analysis
================================================================
Sub-sections:
  3.1  Overall Summary Table (21 quantitative items)
  3.2  By Observer Type (T vs P) with Cohen's d
  3.3  Distribution Visualisations (boxplots, violin, histograms)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy import stats

# Add current directory to path for dynamic module loading
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from questionnaire_mapping import (
    get_all_items_for_analysis,
    QUANT_ITEMS,
    QUESTION_MAPPING,
    ITEM_DESC,
    KEY_ITEMS,
    is_reverse_coded,
    get_observer_column
)


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_q_label(full_question_text):
    """Map full question text to Q label."""
    return QUESTION_MAPPING.get(full_question_text, None)


def compute_summary_stats(series):
    """Compute summary statistics for a numeric series."""
    valid = series.dropna()
    if len(valid) == 0:
        return {
            'N': 0, 'Mean': np.nan, 'SD': np.nan, 'Median': np.nan,
            'Min': np.nan, 'Max': np.nan, 'Skew': np.nan, 'Kurt': np.nan
        }
    
    return {
        'N': len(valid),
        'Mean': valid.mean(),
        'SD': valid.std(),
        'Median': valid.median(),
        'Min': valid.min(),
        'Max': valid.max(),
        'Skew': valid.skew(),
        'Kurt': valid.kurtosis()
    }


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    
    mean_diff = g2.mean() - g1.mean()
    pooled_sd = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
    
    if pooled_sd == 0:
        return np.nan
    
    return mean_diff / pooled_sd


# ── 3.1: Overall Summary Table ────────────────────────────────────────────────

def _section_3_1(df):
    """Display overall summary statistics for 21 quantitative items."""
    st.markdown("### 3.1 Overall Summary Statistics (21 Items)")
    st.markdown("*Across all observers and sessions*")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    
    # Get only quantitative items that exist in the dataframe
    quant_cols = [q for q in full_questions if get_q_label(q) in QUANT_ITEMS and q in df.columns]
    
    rows = []
    for col in quant_cols:
        q_label = get_q_label(col)
        item_desc = ITEM_DESC.get(q_label, q_label)
        
        stats_dict = compute_summary_stats(df[col])
        
        rows.append({
            'Item': q_label,
            'Description': item_desc,
            'N': int(stats_dict['N']) if stats_dict['N'] > 0 else 0,
            'Mean': round(stats_dict['Mean'], 2) if not np.isnan(stats_dict['Mean']) else 'N/A',
            'SD': round(stats_dict['SD'], 2) if not np.isnan(stats_dict['SD']) else 'N/A',
            'Median': round(stats_dict['Median'], 2) if not np.isnan(stats_dict['Median']) else 'N/A',
            'Min': int(stats_dict['Min']) if not np.isnan(stats_dict['Min']) else 'N/A',
            'Max': int(stats_dict['Max']) if not np.isnan(stats_dict['Max']) else 'N/A',
            'Skew': round(stats_dict['Skew'], 2) if not np.isnan(stats_dict['Skew']) else 'N/A',
            'Kurt': round(stats_dict['Kurt'], 2) if not np.isnan(stats_dict['Kurt']) else 'N/A'
        })
    
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Items", len(summary_df))
    col2.metric("Total Observations", int(summary_df['N'].sum()))
    col3.metric("Mean Missingness", f"{(1 - summary_df['N'].sum() / (len(summary_df) * len(df))) * 100:.1f}%")


# ── 3.2: By Observer Type ─────────────────────────────────────────────────────

def _section_3_2(df):
    """Display summary statistics split by observer type (T vs P)."""
    st.markdown("### 3.2 Summary by Observer Type (T vs P)")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    observer_col = metadata["observer"]
    
    # Get only quantitative items
    quant_cols = [q for q in full_questions if get_q_label(q) in QUANT_ITEMS and q in df.columns]
    
    df_t = df[df[observer_col].str.upper().str.strip() == "T"]
    df_p = df[df[observer_col].str.upper().str.strip() == "P"]
    
    # Build comparison table
    rows = []
    for col in quant_cols:
        q_label = get_q_label(col)
        item_desc = ITEM_DESC.get(q_label, q_label)
        
        stats_t = compute_summary_stats(df_t[col])
        stats_p = compute_summary_stats(df_p[col])
        d = cohens_d(df_t[col], df_p[col])
        
        # Interpret effect size
        if np.isnan(d):
            d_interp = "N/A"
        elif abs(d) < 0.2:
            d_interp = f"{d:.2f} (negligible)"
        elif abs(d) < 0.5:
            d_interp = f"{d:.2f} (small)"
        elif abs(d) < 0.8:
            d_interp = f"{d:.2f} (medium)"
        else:
            d_interp = f"{d:.2f} (large)"
        
        rows.append({
            'Item': q_label,
            'Description': item_desc,
            'T_N': int(stats_t['N']) if stats_t['N'] > 0 else 0,
            'T_Mean': round(stats_t['Mean'], 2) if not np.isnan(stats_t['Mean']) else 'N/A',
            'T_SD': round(stats_t['SD'], 2) if not np.isnan(stats_t['SD']) else 'N/A',
            'P_N': int(stats_p['N']) if stats_p['N'] > 0 else 0,
            'P_Mean': round(stats_p['Mean'], 2) if not np.isnan(stats_p['Mean']) else 'N/A',
            'P_SD': round(stats_p['SD'], 2) if not np.isnan(stats_p['SD']) else 'N/A',
            'Δ Mean (P-T)': round(stats_p['Mean'] - stats_t['Mean'], 2) if not (np.isnan(stats_p['Mean']) or np.isnan(stats_t['Mean'])) else 'N/A',
            "Cohen's d": d_interp
        })
    
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("**Effect Size Interpretation:**")
    st.markdown("- **Negligible**: |d| < 0.2 | **Small**: 0.2 ≤ |d| < 0.5 | **Medium**: 0.5 ≤ |d| < 0.8 | **Large**: |d| ≥ 0.8")


# ── 3.3: Distribution Visualisations ──────────────────────────────────────────

def _section_3_3(df):
    """Display distribution visualisations: boxplots, violin, histograms."""
    st.markdown("### 3.3 Distribution Visualisations")
    
    full_questions, q_labels, metadata, notes = get_all_items_for_analysis(df)
    observer_col = metadata["observer"]
    
    # Get only quantitative items
    quant_cols = [q for q in full_questions if get_q_label(q) in QUANT_ITEMS and q in df.columns]
    quant_labels = [get_q_label(q) for q in quant_cols]
    
    # Prepare data for visualization
    viz_data = []
    for col, label in zip(quant_cols, quant_labels):
        temp = df[[col, observer_col]].dropna()
        temp['Item'] = label
        temp['Observer'] = temp[observer_col].str.upper().str.strip()
        temp['Value'] = temp[col]
        viz_data.append(temp[['Item', 'Observer', 'Value']])
    
    viz_df = pd.concat(viz_data, ignore_index=True)
    
    # ── Tab 1: Boxplots ───────────────────────────────────────────────────
    st.markdown("#### Boxplots: All 21 Items (T vs P)")

    
    # Split into two groups for better readability
    n_items = len(quant_labels)
    mid_point = (n_items + 1) // 2
    
    items_group1 = quant_labels[:mid_point]
    items_group2 = quant_labels[mid_point:]
    
    # Group 1
    viz_df_g1 = viz_df[viz_df['Item'].isin(items_group1)]
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    fig1.patch.set_facecolor('white')
    
    sns.boxplot(data=viz_df_g1, x='Item', y='Value', hue='Observer', ax=ax1, palette=['#4878CF', '#D65F5F'])
    ax1.set_title(f'Distribution of Items 1-{mid_point} by Observer Type', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Item', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Group 2
    if len(items_group2) > 0:
        viz_df_g2 = viz_df[viz_df['Item'].isin(items_group2)]
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        fig2.patch.set_facecolor('white')
        
        sns.boxplot(data=viz_df_g2, x='Item', y='Value', hue='Observer', ax=ax2, palette=['#4878CF', '#D65F5F'])
        ax2.set_title(f'Distribution of Items {mid_point+1}-{n_items} by Observer Type', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xlabel('Item', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    
    # ── Tab 2: Violin Plots (6 Key Items) ─────────────────────────────────
    st.markdown("#### Violin Plots: 6 Key Items Split by Observer")
    
    key_item_cols = [q for q in quant_cols if get_q_label(q) in KEY_ITEMS]
    key_item_labels = [get_q_label(q) for q in key_item_cols]
    
    if len(key_item_labels) > 0:
        # Create mapping from Q label to description
        label_to_desc = {label: ITEM_DESC.get(label, label) for label in key_item_labels}
        
        viz_df_key = viz_df[viz_df['Item'].isin(key_item_labels)].copy()
        viz_df_key['Item'] = viz_df_key['Item'].map(label_to_desc)
        
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        fig3.patch.set_facecolor('white')
        
        sns.violinplot(data=viz_df_key, x='Item', y='Value', hue='Observer', ax=ax3, palette=['#4878CF', '#D65F5F'], split=True)
        ax3.set_title('Distribution of 6 Key Items by Observer Type (Violin Plot)', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xlabel('Item Description', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    
    # ── Tab 3: Histogram Grid (4×6) ───────────────────────────────────────
    st.markdown("#### Histogram Grid: Distribution of All 21 Items")
    
    fig4, axes = plt.subplots(4, 6, figsize=(16, 10))
    fig4.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(zip(quant_cols, quant_labels)):
        ax = axes[idx]
        data_valid = df[col].dropna()
        
        ax.hist(data_valid, bins=15, color='#4878CF', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Score', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide the extra subplot if not divisible by 24
    for idx in range(len(quant_labels), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)


# ── Main Display Function ─────────────────────────────────────────────────────

def display(df, scale_map=None):
    """Section 3 entry point."""
    st.markdown("## 3. Descriptive Statistics & Distribution Analysis")
    st.markdown("---")
    
    # Create tabs for each subsection
    tab1, tab2, tab3 = st.tabs(["3.1 Overall Summary", "3.2 By Observer Type", "3.3 Distributions"])
    
    with tab1:
        _section_3_1(df)
    
    with tab2:
        _section_3_2(df)
    
    with tab3:
        _section_3_3(df)
