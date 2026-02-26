import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ4: Family Relationship Impact"""
    st.subheader("RQ4: Family Relationship Impact")
    q13 = "How much did the relationship between a participant and carer/parent improved?"
    st.markdown("**Q13:** " + q13)
    
    # create separate trends for therapist (T) and parent (P)
    # assume Submitted_by column contains 'T' or 'P'
    base = df[['Session number', 'Submitted_by', q13]].copy()
    base = base[base['Submitted_by'].isin(['T','P'])]
    
    trend_T = base[base['Submitted_by']=='T'].groupby('Session number')[q13].mean().reset_index()
    trend_P = base[base['Submitted_by']=='P'].groupby('Session number')[q13].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # plot therapist line
    if not trend_T.empty:
        ax.plot(trend_T['Session number'], trend_T[q13], marker='o', linewidth=2.5, 
                markersize=10, color='#FF8C00', markerfacecolor='#FF8C00', markeredgewidth=2, markeredgecolor='white',
                label='Therapist (T)')
        for idx, row in trend_T.iterrows():
            ax.text(row['Session number'], row[q13] - 0.3, f"{row[q13]:.2f}",
                   ha='center', va='top', fontsize=10, color='#FF8C00')
    # plot parent line
    if not trend_P.empty:
        ax.plot(trend_P['Session number'], trend_P[q13], marker='s', linewidth=2.5, 
                markersize=10, color='#006400', markerfacecolor='#006400', markeredgewidth=2, markeredgecolor='white',
                label='Parent (P)')
        for idx, row in trend_P.iterrows():
            ax.text(row['Session number'], row[q13] + 0.2, f"{row[q13]:.2f}",
                   ha='center', va='bottom', fontsize=10, color='#006400')
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # use 0-4 scale like other modules
    ax.set_ylim(-0.5, 4.5)
    ax.set_yticks(ticks=[0,1,2,3,4])
    # dynamic labels
    if scale_map:
        labels = [f"{k}: {v}" for k,v in sorted(scale_map.items())]
        ax.set_yticklabels(labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)
    
    # clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Parent/Carer Relationship Improvement',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    if scale_map:
        st.markdown("**Scale Reference:**")
        scale_df = pd.DataFrame(list(scale_map.items()), columns=['Score', 'Label'])
        st.table(scale_df)
