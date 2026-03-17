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
    # ============ PARENT VS THERAPIST PERSPECTIVE ANALYSIS ============
    st.divider()
    st.subheader("Parent vs Therapist Perspective Analysis")
    st.markdown("**Engagement and Emotional Connection: Comparing Parent and Therapist Observations**")
    
    # Map Submitted_by to standardize parent/therapist labels (P/C already mapped to P in Silver data)
    df_perspective = df.copy()
    df_perspective['Perspective'] = df_perspective['Submitted_by'].map({
        'T': 'Therapist',
        'P': 'Parent'
    }).fillna('Other')
    
    # Filter to only Parent and Therapist data
    df_perspective = df_perspective[df_perspective['Perspective'].isin(['Parent', 'Therapist'])].copy()
    
    if len(df_perspective) > 0:
        # Calculate trends by perspective and session
        parent_trend = df_perspective[df_perspective['Perspective'] == 'Parent'].groupby('Session number')[[q1, q3]].mean().reset_index()
        therapist_trend = df_perspective[df_perspective['Perspective'] == 'Therapist'].groupby('Session number')[[q1, q3]].mean().reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Plot 1: Engagement (Q1) trends by perspective
        if len(parent_trend) > 0:
            ax1.plot(parent_trend['Session number'], parent_trend[q1], marker='o', linewidth=2.5, 
                    markersize=10, label='Parent', color='#9b59b6', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for parent engagement
            for idx, row in parent_trend.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#9b59b6', family='sans-serif')
        
        if len(therapist_trend) > 0:
            ax1.plot(therapist_trend['Session number'], therapist_trend[q1], marker='s', linewidth=2.5, 
                    markersize=10, label='Therapist', color='#e67e22', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for therapist engagement
            for idx, row in therapist_trend.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e67e22', family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement Across Sessions:\nParent vs Therapist Perspective', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(-0.8, 4.8)
        ax1.set_yticks([0, 1, 2, 3, 4])
        if scale_map:
            scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
            ax1.set_yticklabels(scale_labels, fontsize=10)
        else:
            ax1.set_yticklabels(['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully'], fontsize=10)
        ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Emotional Connection (Q3) trends by perspective
        if len(parent_trend) > 0:
            ax2.plot(parent_trend['Session number'], parent_trend[q3], marker='o', linewidth=2.5, 
                    markersize=10, label='Parent', color='#9b59b6', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for parent connection
            for idx, row in parent_trend.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#9b59b6', family='sans-serif')
        
        if len(therapist_trend) > 0:
            ax2.plot(therapist_trend['Session number'], therapist_trend[q3], marker='s', linewidth=2.5, 
                    markersize=10, label='Therapist', color='#e67e22', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for therapist connection
            for idx, row in therapist_trend.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e67e22', family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Connection Score', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection Across Sessions:\nParent vs Therapist Perspective', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(-0.8, 4.8)
        ax2.set_yticks([0, 1, 2, 3, 4])
        if scale_map:
            scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
            ax2.set_yticklabels(scale_labels, fontsize=10)
        else:
            ax2.set_yticklabels(['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully'], fontsize=10)
        ax2.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics table for perspective comparison
        st.markdown("**Summary Statistics: Parent vs Therapist Perspective**")
        
        perspective_data = []
        for perspective in ['Parent', 'Therapist']:
            perspective_df = df_perspective[df_perspective['Perspective'] == perspective]
            if len(perspective_df) > 0:
                perspective_data.append({
                    'Perspective': perspective,
                    'Total Observations': len(perspective_df),
                    'Unique Participants': len(perspective_df['Participant id'].unique()),
                    'Avg Engagement': f"{perspective_df[q1].mean():.2f}",
                    'Engagement Std Dev': f"{perspective_df[q1].std():.2f}",
                    'Avg Connection': f"{perspective_df[q3].mean():.2f}",
                    'Connection Std Dev': f"{perspective_df[q3].std():.2f}"
                })
        
        perspective_comparison_df = pd.DataFrame(perspective_data)
        st.dataframe(perspective_comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No Parent or Therapist data available for perspective analysis")

    # ============ AUTISM LEVEL ANALYSIS ============
    st.divider()
    st.subheader("Autism Level Analysis")
    st.markdown("**Engagement and Emotional Connection Trends by Autism Level**")
    
    # Get unique autism levels and filter data
    autism_levels = sorted(df['Autism Level'].dropna().unique())
    
    if len(autism_levels) > 0:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Define colors for different autism levels
        colors_map = {
            1: '#3498db',  # Blue
            2: '#e74c3c',  # Red
            3: '#2ecc71',  # Green
            4: '#f39c12',  # Orange
            5: '#9b59b6'   # Purple
        }
        
        # Plot 1: Engagement (Q1) trends by autism level
        for autism_level in autism_levels:
            autism_trend = df[df['Autism Level'] == autism_level].groupby('Session number')[[q1, q3]].mean().reset_index()
            
            if len(autism_trend) > 0:
                color = colors_map.get(autism_level, '#95a5a6')
                ax1.plot(autism_trend['Session number'], autism_trend[q1], marker='o', linewidth=2.5, 
                        markersize=10, label=f'Level {int(autism_level)}', color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations
                for idx, row in autism_trend.iterrows():
                    # Alternate annotation position to avoid overlap
                    xytext_offset = (0, 12) if idx % 2 == 0 else (0, -15)
                    ax1.annotate(f'{row[q1]:.2f}', 
                               xy=(row['Session number'], row[q1]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement Across Sessions by Autism Level', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(-0.8, 4.8)
        ax1.set_yticks([0, 1, 2, 3, 4])
        if scale_map:
            scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
            ax1.set_yticklabels(scale_labels, fontsize=10)
        else:
            ax1.set_yticklabels(['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully'], fontsize=10)
        ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=10, loc='best', frameon=True, shadow=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Emotional Connection (Q3) trends by autism level
        for autism_level in autism_levels:
            autism_trend = df[df['Autism Level'] == autism_level].groupby('Session number')[[q1, q3]].mean().reset_index()
            
            if len(autism_trend) > 0:
                color = colors_map.get(autism_level, '#95a5a6')
                ax2.plot(autism_trend['Session number'], autism_trend[q3], marker='o', linewidth=2.5, 
                        markersize=10, label=f'Level {int(autism_level)}', color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations
                for idx, row in autism_trend.iterrows():
                    # Alternate annotation position to avoid overlap
                    xytext_offset = (0, 12) if idx % 2 == 0 else (0, -15)
                    ax2.annotate(f'{row[q3]:.2f}', 
                               xy=(row['Session number'], row[q3]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Connection Score', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection Across Sessions by Autism Level', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(-0.8, 4.8)
        ax2.set_yticks([0, 1, 2, 3, 4])
        if scale_map:
            scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
            ax2.set_yticklabels(scale_labels, fontsize=10)
        else:
            ax2.set_yticklabels(['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully'], fontsize=10)
        ax2.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=10, loc='best', frameon=True, shadow=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics table for autism level comparison
        st.markdown("**Summary Statistics: Engagement and Connection by Autism Level**")
        
        autism_data = []
        for autism_level in autism_levels:
            autism_df = df[df['Autism Level'] == autism_level]
            if len(autism_df) > 0:
                autism_data.append({
                    'Autism Level': f'Level {int(autism_level)}',
                    'Total Observations': len(autism_df),
                    'Unique Participants': len(autism_df['Participant id'].unique()),
                    'Avg Engagement': f"{autism_df[q1].mean():.2f}",
                    'Engagement Std Dev': f"{autism_df[q1].std():.2f}",
                    'Avg Connection': f"{autism_df[q3].mean():.2f}",
                    'Connection Std Dev': f"{autism_df[q3].std():.2f}"
                })
        
        autism_comparison_df = pd.DataFrame(autism_data)
        st.dataframe(autism_comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No Autism Level data available for analysis")    
    
    # ============ AGE GROUP ANALYSIS: 8-14 YEARS ============
    st.divider()
    st.subheader("Age-Group Analysis: 8-14 Years")
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
        ax1.set_title('Engagement Improvement Across Sessions\nAge 8-14 by Gender', fontsize=12, fontweight='bold', pad=15)
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
        ax2.set_title('Emotional Connection Improvement Across Sessions\nAge 8-14 by Gender', fontsize=12, fontweight='bold', pad=15)
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
                    'Engagement Std Dev': f"{gender_df[q1].std():.2f}",
                    'Avg Connection': f"{gender_df[q3].mean():.2f}",
                    'Connection Std Dev': f"{gender_df[q3].std():.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for age group 8-14")

    # ============ AGE GROUP ANALYSIS: 15-19 YEARS ============
    st.divider()
    st.subheader("Age-Group Analysis: 15-19 Years")
    st.markdown("**Engagement and Emotional Connection Trends (Age 15-19)**")
    
    # Filter data for age group 15-19
    age_filtered_df_15_19 = df[(df['Age'] >= 15) & (df['Age'] <= 19)].copy()
    
    if len(age_filtered_df_15_19) > 0:
        # Calculate trends by gender and session
        male_trend_15_19 = age_filtered_df_15_19[age_filtered_df_15_19['Gender'] == 'Male'].groupby('Session number')[[q1, q3]].mean().reset_index()
        female_trend_15_19 = age_filtered_df_15_19[age_filtered_df_15_19['Gender'] == 'Female'].groupby('Session number')[[q1, q3]].mean().reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Plot 1: Engagement (Q1) trends
        if len(male_trend_15_19) > 0:
            ax1.plot(male_trend_15_19['Session number'], male_trend_15_19[q1], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male engagement
            for idx, row in male_trend_15_19.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend_15_19) > 0:
            ax1.plot(female_trend_15_19['Session number'], female_trend_15_19[q1], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female engagement
            for idx, row in female_trend_15_19.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement Improvement Across Sessions\n Age 15-19', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(-0.8, 4.8)
        ax1.set_yticks([0, 1, 2, 3, 4])
        ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Emotional Connection (Q3) trends
        if len(male_trend_15_19) > 0:
            ax2.plot(male_trend_15_19['Session number'], male_trend_15_19[q3], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male connection
            for idx, row in male_trend_15_19.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend_15_19) > 0:
            ax2.plot(female_trend_15_19['Session number'], female_trend_15_19[q3], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female connection
            for idx, row in female_trend_15_19.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Connection Score', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection Improvement Across Sessions\nAge 15-19', fontsize=12, fontweight='bold', pad=15)
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
        st.markdown("**Summary Statistics for Age 15-19**")
        
        summary_data_15_19 = []
        for gender in ['Male', 'Female']:
            gender_df = age_filtered_df_15_19[age_filtered_df_15_19['Gender'] == gender]
            if len(gender_df) > 0:
                summary_data_15_19.append({
                    'Gender': gender,
                    'Participants': len(gender_df['Participant id'].unique()),
                    'Sessions': len(gender_df),
                    'Avg Engagement': f"{gender_df[q1].mean():.2f}",
                    'Engagement Std Dev': f"{gender_df[q1].std():.2f}",
                    'Avg Connection': f"{gender_df[q3].mean():.2f}",
                    'Connection Std Dev': f"{gender_df[q3].std():.2f}"
                })
        
        summary_df_15_19 = pd.DataFrame(summary_data_15_19)
        st.dataframe(summary_df_15_19, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for age group 15-19")

    # ============ AGE GROUP ANALYSIS: 20-26 YEARS ============
    st.divider()
    st.subheader("Age-Group Analysis: 20-26 Years")
    st.markdown("**Engagement and Emotional Connection Trends(Age 20-26)**")
    
    # Filter data for age group 20-26
    age_filtered_df_20_26 = df[(df['Age'] >= 20) & (df['Age'] <= 26)].copy()
    
    if len(age_filtered_df_20_26) > 0:
        # Calculate trends by gender and session
        male_trend_20_26 = age_filtered_df_20_26[age_filtered_df_20_26['Gender'] == 'Male'].groupby('Session number')[[q1, q3]].mean().reset_index()
        female_trend_20_26 = age_filtered_df_20_26[age_filtered_df_20_26['Gender'] == 'Female'].groupby('Session number')[[q1, q3]].mean().reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Plot 1: Engagement (Q1) trends
        if len(male_trend_20_26) > 0:
            ax1.plot(male_trend_20_26['Session number'], male_trend_20_26[q1], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male engagement
            for idx, row in male_trend_20_26.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend_20_26) > 0:
            ax1.plot(female_trend_20_26['Session number'], female_trend_20_26[q1], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female engagement
            for idx, row in female_trend_20_26.iterrows():
                ax1.annotate(f'{row[q1]:.2f}', 
                           xy=(row['Session number'], row[q1]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement Improvement Across Sessions\nAge 20-26', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(-0.8, 4.8)
        ax1.set_yticks([0, 1, 2, 3, 4])
        ax1.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=11, loc='best', frameon=True, shadow=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Emotional Connection (Q3) trends
        if len(male_trend_20_26) > 0:
            ax2.plot(male_trend_20_26['Session number'], male_trend_20_26[q3], marker='o', linewidth=2.5, 
                    markersize=10, label='Male', color='#3498db', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for male connection
            for idx, row in male_trend_20_26.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#3498db', family='sans-serif')
        
        if len(female_trend_20_26) > 0:
            ax2.plot(female_trend_20_26['Session number'], female_trend_20_26[q3], marker='s', linewidth=2.5, 
                    markersize=10, label='Female', color='#e74c3c', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for female connection
            for idx, row in female_trend_20_26.iterrows():
                ax2.annotate(f'{row[q3]:.2f}', 
                           xy=(row['Session number'], row[q3]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e74c3c', family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Connection Score', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection Improvement Across Sessions\nAge 20-26 by Gender', fontsize=12, fontweight='bold', pad=15)
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
        st.markdown("**Summary Statistics for Age 20-26**")
        
        summary_data_20_26 = []
        for gender in ['Male', 'Female']:
            gender_df = age_filtered_df_20_26[age_filtered_df_20_26['Gender'] == gender]
            if len(gender_df) > 0:
                summary_data_20_26.append({
                    'Gender': gender,
                    'Participants': len(gender_df['Participant id'].unique()),
                    'Sessions': len(gender_df),
                    'Avg Engagement': f"{gender_df[q1].mean():.2f}",
                    'Engagement Std Dev': f"{gender_df[q1].std():.2f}",
                    'Avg Connection': f"{gender_df[q3].mean():.2f}",
                    'Connection Std Dev': f"{gender_df[q3].std():.2f}"
                })
        
        summary_df_20_26 = pd.DataFrame(summary_data_20_26)
        st.dataframe(summary_df_20_26, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available for age group 20-26")

    # ============ COMPARATIVE ANALYSIS ACROSS ALL AGE GROUPS ============
    st.divider()
    st.subheader("Comparative Analysis Across Age Groups")
    st.markdown("**Overall Engagement and Emotional Connection Comparison**")
    
    # Aggregate statistics across all age groups
    all_age_groups = [
        {'range': '8-14', 'df': age_filtered_df},
        {'range': '15-19', 'df': age_filtered_df_15_19},
        {'range': '20-26', 'df': age_filtered_df_20_26}
    ]
    
    comparative_data = []
    for age_group_info in all_age_groups:
        age_group_df = age_group_info['df']
        if len(age_group_df) > 0:
            comparative_data.append({
                'Age Group': age_group_info['range'],
                'Total Participants': len(age_group_df['Participant id'].unique()),
                'Total Sessions': len(age_group_df),
                'Avg Engagement Score': f"{age_group_df[q1].mean():.2f}",
                'Avg Connection Score': f"{age_group_df[q3].mean():.2f}",
                'Engagement Std Dev': f"{age_group_df[q1].std():.2f}",
                'Connection Std Dev': f"{age_group_df[q3].std():.2f}"
            })
    
    comparative_df = pd.DataFrame(comparative_data)
    st.dataframe(comparative_df, use_container_width=True, hide_index=True)
    
    # Visualization: Compare average scores across age groups
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    age_groups_labels = []
    engagement_scores = []
    connection_scores = []
    
    for age_group_info in all_age_groups:
        age_group_df = age_group_info['df']
        if len(age_group_df) > 0:
            age_groups_labels.append(age_group_info['range'])
            engagement_scores.append(age_group_df[q1].mean())
            connection_scores.append(age_group_df[q3].mean())
    
    if age_groups_labels:
        # Engagement comparison
        colors_eng = ['#3498db', '#e74c3c', '#2ecc71']
        bars1 = ax1.bar(age_groups_labels, engagement_scores, color=colors_eng[:len(age_groups_labels)], edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Average Engagement Score', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax1.set_title('Engagement Scores by Age Group', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylim(0, 4)
        ax1.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontweight='bold', fontsize=11)
        
        # Connection comparison
        colors_conn = ['#9b59b6', '#f39c12', '#1abc9c']
        bars2 = ax2.bar(age_groups_labels, connection_scores, color=colors_conn[:len(age_groups_labels)], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Average Connection Score', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax2.set_title('Emotional Connection Scores by Age Group', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(0, 4)
        ax2.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.set_axisbelow(True)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)