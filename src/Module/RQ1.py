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
    
    
    # ============ AGE GROUP ANALYSIS: 8-14 YEARS ============
    st.divider()
    st.subheader("Age-Group Analysis: 8-14 Years (Session-wise Improvement)")
    st.markdown("**Engagement and Emotional Connection Trends by Gender (Age 8-14)**")
    
    # Filter data for age group 8-14
    age_filtered_df = df[(df['Age'] >= 8) & (df['Age'] <= 14)].copy()
    
    if len(age_filtered_df) > 0:
        # Calculate trends by gender and session
        male_trend = age_filtered_df[age_filtered_df['Gender'] == 'Male'].groupby('Session number')[[q1, q3]].mean().reset_index()
        female_trend = age_filtered_df[age_filtered_df['Gender'] == 'Female'].groupby('Session number')[[q1, q3]].mean().reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Plot 1: Engagement (Q1) trends
        if len(male_trend) > 0:
            ax1.plot(male_trend['Session number'], male_trend[q1], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male engagement
            for idx, row in male_trend.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend) > 0:
            ax1.plot(female_trend['Session number'], female_trend[q1], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female engagement
            for idx, row in female_trend.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement (Q1) Improvement Across Sessions\nAge 8-14 by Gender', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(-0.8, 4.8)
        ax1.set_yticks([0, 1, 2, 3, 4])
        ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Emotional Connection (Q3) trends
        if len(male_trend) > 0:
            ax2.plot(male_trend['Session number'], male_trend[q3], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male connection
            for idx, row in male_trend.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend) > 0:
            ax2.plot(female_trend['Session number'], female_trend[q3], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female connection
            for idx, row in female_trend.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Connection Score', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection (Q3) Improvement Across Sessions\nAge 8-14 by Gender', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(-0.8, 4.8)
        ax2.set_yticks([0, 1, 2, 3, 4])
        ax2.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics table
        st.markdown("**Summary Statistics for Age 8-14**")
        
        summary_data = []
        for gender in ['Male', 'Female']:
            gender_df = age_filtered_df[age_filtered_df['Gender'] == gender]
            if len(gender_df) > 0:
                summary_data.append({
                    'Gender': gender,
                    'Participants': len(gender_df['Participant id'].unique()),
                    'Sessions': len(gender_df),
                    'Avg Engagement': f"{gender_df[q1].mean():.2f}",
                    'Avg Connection': f"{gender_df[q3].mean():.2f}",
                    'Engagement Trend': f"{gender_df.groupby('Session number')[q1].mean().iloc[-1] - gender_df.groupby('Session number')[q1].mean().iloc[0]:.2f}" if len(gender_df.groupby('Session number')[q1].mean()) > 1 else "N/A"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for age group 8-14")

    # Display scale reference
    if scale_map:
        st.markdown("**Scale Reference:**")
        scale_df = pd.DataFrame(list(scale_map.items()), columns=['Score', 'Label'])
        st.table(scale_df)    # Display scale reference
 