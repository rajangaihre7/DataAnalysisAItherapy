import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display(df, scale_map=None):
    """Display RQ7: Personalization & Verbal Participation Correlation"""
    st.subheader("RQ7: Personalization Impact on Verbal Participation")
    
    # Find columns dynamically
    q2_col = None
    q4_col = None
    
    for col in df.columns:
        if 'personalised' in col.lower():
            q2_col = col
        if 'verbal' in col.lower() and 'participation' in col.lower():
            q4_col = col
    
    if not q2_col or not q4_col:
        st.error("Required columns not found in dataset.")
        return
    
    # Display the questions being analyzed
    st.markdown("**Q2 (Personalization):** " + q2_col)
    st.markdown("**Q4 (Verbal Participation):** " + q4_col)
    
    # Convert to numeric and drop nulls
    df[q2_col] = pd.to_numeric(df[q2_col], errors='coerce')
    df[q4_col] = pd.to_numeric(df[q4_col], errors='coerce')
    
    clean_df = df[[q2_col, q4_col]].dropna()
    
    if len(clean_df) == 0:
        st.info("No valid data pairs available for correlation analysis.")
        return
    
    # Calculate Pearson correlation
    correlation = clean_df[q2_col].corr(clean_df[q4_col])
    
    # Interpret correlation
    if correlation > 0.7:
        interpretation = "Strong positive correlation"
        color_interpret = "#27AE60"  # Green
    elif correlation > 0.4:
        interpretation = "Moderate positive correlation"
        color_interpret = "#F39C12"  # Orange
    elif correlation > 0:
        interpretation = "Weak positive correlation"
        color_interpret = "#3498DB"  # Blue
    else:
        interpretation = "No or negative correlation"
        color_interpret = "#E74C3C"  # Red
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Valid Data Pairs", len(clean_df))
    col2.metric("Correlation Coefficient", f"{correlation:.4f}")
    col3.metric("Interpretation", interpretation)
    
    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Scatter plot
    ax.scatter(clean_df[q2_col], clean_df[q4_col], 
              alpha=0.6, s=100, color='#3498DB', edgecolors='black', linewidth=0.5,
              label=f'Data points (n={len(clean_df)})')
    
    # Regression line
    z = np.polyfit(clean_df[q2_col], clean_df[q4_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(clean_df[q2_col].min(), clean_df[q2_col].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, label=f'Trend line (r={correlation:.4f})')
    
    # Grid and styling
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set scale with labels on x-axis, numeric range 0-10 on y-axis
    scale_labels = ['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully']
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(scale_labels, fontsize=9)
    ax.set_yticks(range(0, 11))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 10)
    
    ax.set_xlabel('Personalization Understanding (Q2)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Verbal Participation Score (Q4)', fontsize=12, fontweight='bold')
    ax.set_title('Correlation: Personalization Understanding vs. Verbal Participation',
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', frameon=False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display scale reference
    st.markdown("**Scale Reference:**")
    scale_df = pd.DataFrame({
        'Score': [0, 1, 2, 3, 4],
        'Label': ['Not at all', 'Slightly', 'Moderately', 'Very', 'Fully']
    })
    st.table(scale_df)
    
    # Add spacing
    st.write("")
   
    
    # Violin Plot: Q4 distribution by Q2 level
    st.markdown("---")
    st.markdown("#### Verbal Participation Distribution by Personalization Level")
    st.write("*How Q4 (verbal participation) varies across different Q2 (personalization) levels*")
    st.write("")
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    fig3.patch.set_facecolor('white')
    
    # Prepare data for violin plot
    plot_data = []
    q2_levels = sorted(clean_df[q2_col].unique())
    
    for level in q2_levels:
        q4_values = clean_df[clean_df[q2_col] == level][q4_col].values
        plot_data.append(q4_values)
    
    # Create violin plot
    parts = ax3.violinplot(plot_data, positions=q2_levels, widths=0.7, 
                           showmeans=True, showmedians=True)
    
    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style mean/median lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('darkred')
    parts['cmedians'].set_linewidth(2)
    
    # Grid and styling
    ax3.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax3.set_axisbelow(True)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['bottom'].set_linewidth(1.5)
    
    ax3.set_xlabel('Personalization Understanding Level (Q2)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Verbal Participation Score (Q4)', fontsize=12, fontweight='bold')
    ax3.set_title('Verbal Participation Distribution Across Personalization Levels',
                 fontsize=13, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                   Line2D([0], [0], color='darkred', lw=2)]
    ax3.legend(custom_lines, ['Mean', 'Median'], fontsize=10, loc='upper left', frameon=False)
    
    # Set x-axis with scale labels
    scale_labels = ['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully']
    ax3.set_xticks(q2_levels)
    ax3.set_xticklabels([scale_labels[int(x)] if int(x) < len(scale_labels) else f'{int(x)}' for x in q2_levels], fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personalization Understanding (Q2)")
        st.write(f"**Mean:** {clean_df[q2_col].mean():.2f}")
        st.write(f"**Std Dev:** {clean_df[q2_col].std():.2f}")
        st.write(f"**Min:** {clean_df[q2_col].min():.0f}")
        st.write(f"**Max:** {clean_df[q2_col].max():.0f}")
    
    with col2:
        st.subheader("Verbal Participation Score (Q4)")
        st.write(f"**Mean:** {clean_df[q4_col].mean():.2f}")
        st.write(f"**Std Dev:** {clean_df[q4_col].std():.2f}")
        st.write(f"**Min:** {clean_df[q4_col].min():.0f}")
        st.write(f"**Max:** {clean_df[q4_col].max():.0f}")
    
    # Insight
    st.markdown("---")
    st.markdown("### Key Insight")
    if correlation > 0.7:
        st.success(f"✓ **Strong Evidence:** The strong positive correlation (r={correlation:.4f}) indicates that "
                   f"when participants better understand the personalization of the story, they significantly "
                   f"increase their verbal participation. This strongly supports the hypothesis that "
                   f"personalization drives communication engagement.")
    elif correlation > 0.4:
        st.info(f"✓ **Moderate Evidence:** The moderate positive correlation (r={correlation:.4f}) suggests that "
                f"personalization moderately increases verbal participation. Other factors may also influence "
                f"communication engagement.")
    else:
        st.warning(f"⚠ **Limited Evidence:** The low correlation (r={correlation:.4f}) suggests that "
                   f"personalization alone may not be the primary driver of verbal participation.")
