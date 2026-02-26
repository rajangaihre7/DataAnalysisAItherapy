import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ2: Distress Reduction Analysis"""
    st.subheader("RQ2: Does Distress Decrease Over Time?")
    
    q8 = "Did the participant exhibit distress, boredom, or frustration?"
    st.markdown("**Q8:** " + q8)
    
    trend = df.groupby('Session number')[q8].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    ax.plot(trend['Session number'], trend[q8], marker='o', color='#D32F2F', linewidth=2.5, 
            markersize=10, markerfacecolor='#D32F2F', markeredgewidth=2, markeredgecolor='white')
    
    # Add value annotations
    for idx, row in trend.iterrows():
        value = row[q8]
        ax.annotate(f'{value:.2f}', 
                   xy=(row['Session number'], value),
                   xytext=(0, 12), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   color='#D32F2F', family='sans-serif')
    
    # Add grid - subtle
    ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.8, 4.8)
    ax.set_yticks(ticks=[0,1,2,3,4])
    
    # Dynamic scale labels
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Calm', '1: Slightly', '2: Moderately', '3: Very', '4: High Distress'], fontsize=10)
    
    # Clean axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distress Level', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Participant Distress Across Sessions', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display scale reference
    if scale_map:
        st.markdown("**Scale Reference:**")
        scale_df = pd.DataFrame(list(scale_map.items()), columns=['Score', 'Label'])
        st.table(scale_df)
