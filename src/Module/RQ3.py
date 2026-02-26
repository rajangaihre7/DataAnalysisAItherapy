import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ3: Real-World Generalization Analysis"""
    st.subheader("RQ3: Skill Transfer to Daily Life")
    
    q22 = "Did the participant generalise the behaviour outside the story?"
    q25 = "Did the participant link the story to real-life experiences?"
    
    st.markdown("**Q22 (Generalization):** " + q22)
    st.markdown("**Q25 (Linking to Life):** " + q25)
    
    trend = df.groupby('Session number')[[q22, q25]].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    ax.plot(trend['Session number'], trend[q22], marker='o', label='Generalization (Q22)', linewidth=2.5, 
            markersize=10, color='#2E7D32', markerfacecolor='#2E7D32', markeredgewidth=2, markeredgecolor='white')
    ax.plot(trend['Session number'], trend[q25], marker='s', label='Linking to Life (Q25)', linewidth=2.5, 
            markersize=10, color='#6A1B9A', markerfacecolor='#6A1B9A', markeredgewidth=2, markeredgecolor='white')
    
    # Add value annotations
    for idx, row in trend.iterrows():
        ax.annotate(f'{row[q22]:.2f}', 
                   xy=(row['Session number'], row[q22]),
                   xytext=(0, -20), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   color='#2E7D32', family='sans-serif')
        ax.annotate(f'{row[q25]:.2f}', 
                   xy=(row['Session number'], row[q25]),
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   color='#6A1B9A', family='sans-serif')
    
    # Add grid - subtle
    ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.8, 4.8)
    ax.set_yticks(ticks=[0,1,2,3,4])
    
    # Dynamic scale labels
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    
    # Clean axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Behavioral Generalization and Real-Life Application', 
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display scale reference
    if scale_map:
        st.markdown("**Scale Reference:**")
        scale_df = pd.DataFrame(list(scale_map.items()), columns=['Score', 'Label'])
        st.table(scale_df)
