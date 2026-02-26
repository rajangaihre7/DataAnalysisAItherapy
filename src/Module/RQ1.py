import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ1: Emotional Growth Analysis"""
    st.subheader("RQ1: Growth in Engagement & Connection")
    
    q1 = "How engaged was the participant during today's storytelling session?"
    q3 = "Did the participant demonstrate emotional connection?"
    
    # Display the questions being analyzed
    st.markdown("**Q1 (Engagement):** " + q1)
    st.markdown("**Q3 (Connection):** " + q3)
    
    trend = df.groupby('Session number')[[q1, q3]].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Plot lines with markers - cleaner style
    ax.plot(trend['Session number'], trend[q1], marker='o', linewidth=2.5, 
            markersize=10, label='Engagement (Q1)', color='#0066CC', markerfacecolor='#0066CC', markeredgewidth=2, markeredgecolor='white')
    ax.plot(trend['Session number'], trend[q3], marker='s', linewidth=2.5, 
            markersize=10, label='Connection (Q3)', color='#CC0000', markerfacecolor='#CC0000', markeredgewidth=2, markeredgecolor='white')
    
    # Add value annotations with better spacing
    for idx, row in trend.iterrows():
        value_q1 = row[q1]
        ax.annotate(f'{value_q1:.2f}', 
                   xy=(row['Session number'], value_q1),
                   xytext=(0, 12), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   color='#0066CC', family='sans-serif')
        
        value_q3 = row[q3]
        ax.annotate(f'{value_q3:.2f}', 
                   xy=(row['Session number'], value_q3),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   color='#CC0000', family='sans-serif')
    
    # Add grid - subtle
    ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Set scale and styling
    ax.set_ylim(-0.8, 4.8)
    ax.set_yticks(ticks=[0,1,2,3,4])
    
    # Dynamic scale labels
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully'], fontsize=10)
    
    # Clean axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Participant Engagement and Emotional Connection Across Sessions', 
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display scale reference
    if scale_map:
        st.markdown("**Scale Reference:**")
        scale_df = pd.DataFrame(list(scale_map.items()), columns=['Score', 'Label'])
        st.table(scale_df)
