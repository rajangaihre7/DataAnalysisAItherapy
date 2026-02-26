import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ6 Overall Social Behaviour Impact"""
    st.subheader("RQ6: Overall Social Behaviour Improvement")
    q26 = "How much different scenarios stories impact overall social behaviour ?"
    st.markdown("**Q26:** " + q26)
    
    trend = df.groupby('Session number')[q26].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    bars = ax.bar(trend['Session number'], trend[q26], color='#1565C0', edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value annotations on top of bars
    for idx, (session, value) in enumerate(zip(trend['Session number'], trend[q26])):
        ax.text(session, value + 0.3, f'{value:.2f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               color='#1565C0', family='sans-serif')
    
    # Add grid - subtle, horizontal only
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.5, 11)
    ax.set_yticks(ticks=[0,2,4,6,8,10])
    ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=10)
    
    # Clean axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Social Impact Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Trends in Overall Social Behavior Impact', 
                fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add spacing between figures
    st.write("")
    st.write("")
    
    # --- SECOND FIGURE: Breakdown by Autism Level ---
    st.markdown("---")
    st.markdown("#### Analysis by Autism Level")
    st.write("")
    
    # Prepare data by autism level
    if 'Autism Level' in df.columns:
        autism_trend = df.groupby(['Session number', 'Autism Level'])[q26].mean().reset_index()
        
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        fig2.patch.set_facecolor('white')
        
        # Get unique autism levels and assign colors
        autism_levels = sorted(autism_trend['Autism Level'].dropna().unique())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        color_map = {level: colors[i % len(colors)] for i, level in enumerate(autism_levels)}
        
        # Plot line for each autism level
        for level in autism_levels:
            level_data = autism_trend[autism_trend['Autism Level'] == level]
            ax2.plot(level_data['Session number'], level_data[q26], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f'Autism Level: {level}', 
                    color=color_map[level])
            
            # Add value annotations for each point
            for _, row in level_data.iterrows():
                ax2.text(row['Session number'], row[q26] + 0.3, f'{round(row[q26], 4):.2f}',
                        ha='center', va='bottom', fontsize=8, color=color_map[level])
        
        ax2.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax2.set_axisbelow(True)
        ax2.set_ylim(-0.5, 11)
        ax2.set_yticks(ticks=[0,2,4,6,8,10])
        ax2.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=10)
        
        # Clean axes
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(1.5)
        ax2.spines['bottom'].set_linewidth(1.5)
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Social Impact Score', fontsize=12, fontweight='bold')
        ax2.set_title('Social Behavior Impact by Autism Level Across Sessions', 
                     fontsize=13, fontweight='bold', pad=20)
        ax2.legend(fontsize=10, loc='upper left', frameon=False)
        
        fig2.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("Autism Level column not found for breakdown analysis.")

