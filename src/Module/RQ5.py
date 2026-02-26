import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ5: Self-Initiated Social Interaction"""
    st.subheader("RQ5: Self-Initiated Social Interaction")
    q9 = "Did the participant initiate interactions related to the story?"
    st.markdown("**Q9:** " + q9)

    # Trend over sessions
    trend = df.groupby('Session number')[q9].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.plot(trend['Session number'], trend[q9], marker='o', linewidth=2.5,
            markersize=9, color='#1976D2', markerfacecolor='#1976D2', markeredgewidth=2, markeredgecolor='white')

    # Annotate values above line
    for _, row in trend.iterrows():
        ax.text(row['Session number'], row[q9] + 0.15, f"{row[q9]:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1976D2')

    # Styling and scale
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(ticks=[0,1,2,3,4])
    if scale_map:
        ax.set_yticklabels([f"{k}: {v}" for k,v in sorted(scale_map.items())], fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Initiation Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Self-Initiated Social Interaction', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)

    # Contextual evidence: show up to 6 comment examples from Comment_Q9
    comment_col = 'Comment_Q9'
    if comment_col in df.columns:
        examples = df[comment_col].dropna().astype(str).str.strip()
        examples = examples[examples != '']
        examples = examples.unique()[:6]
        if len(examples) > 0:
            st.markdown('**Contextual Examples (from Comment_Q9):**')
            for ex in examples:
                st.write('- ' + ex)
        else:
            st.info('No contextual comments available for Q9.')
    else:
        st.info('No comment column found for contextual evidence.')
