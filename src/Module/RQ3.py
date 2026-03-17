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
    ax.set_title('Trends in Behavioral Generalization and Real-Life Application', 
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============ PARENT VS THERAPIST PERSPECTIVE ANALYSIS ============
    st.divider()
    st.subheader("Parent vs Therapist Perspective Analysis")
    st.markdown("**Generalization and Real-Life Linking: Comparing Parent and Therapist Observations**")
    
    # Map Submitted_by to standardize parent/therapist labels
    df_perspective = df.copy()
    df_perspective['Perspective'] = df_perspective['Submitted_by'].map({
        'T': 'Therapist',
        'P': 'Parent'
    }).fillna('Other')
    
    # Filter to only Parent and Therapist data
    df_perspective = df_perspective[df_perspective['Perspective'].isin(['Parent', 'Therapist'])].copy()
    
    if len(df_perspective) > 0:
        # Calculate trends by perspective and session
        parent_trend = df_perspective[df_perspective['Perspective'] == 'Parent'].groupby('Session number')[[q22, q25]].mean().reset_index()
        therapist_trend = df_perspective[df_perspective['Perspective'] == 'Therapist'].groupby('Session number')[[q22, q25]].mean().reset_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Plot 1: Generalization (Q22) trends by perspective
        if len(parent_trend) > 0:
            ax1.plot(parent_trend['Session number'], parent_trend[q22], marker='o', linewidth=2.5, 
                    markersize=10, label='Parent', color='#9b59b6', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for parent generalization
            for idx, row in parent_trend.iterrows():
                ax1.annotate(f'{row[q22]:.2f}', 
                           xy=(row['Session number'], row[q22]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#9b59b6', family='sans-serif')
        
        if len(therapist_trend) > 0:
            ax1.plot(therapist_trend['Session number'], therapist_trend[q22], marker='s', linewidth=2.5, 
                    markersize=10, label='Therapist', color='#e67e22', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for therapist generalization
            for idx, row in therapist_trend.iterrows():
                ax1.annotate(f'{row[q22]:.2f}', 
                           xy=(row['Session number'], row[q22]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e67e22', family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Generalization Score', fontsize=12, fontweight='bold')
        ax1.set_title('Skill Generalization Across Sessions:\nParent vs Therapist Perspective', fontsize=12, fontweight='bold', pad=15)
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
        
        # Plot 2: Real-Life Linking (Q25) trends by perspective
        if len(parent_trend) > 0:
            ax2.plot(parent_trend['Session number'], parent_trend[q25], marker='o', linewidth=2.5, 
                    markersize=10, label='Parent', color='#9b59b6', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for parent real-life linking
            for idx, row in parent_trend.iterrows():
                ax2.annotate(f'{row[q25]:.2f}', 
                           xy=(row['Session number'], row[q25]),
                           xytext=(0, 25), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#9b59b6', family='sans-serif')
        
        if len(therapist_trend) > 0:
            ax2.plot(therapist_trend['Session number'], therapist_trend[q25], marker='s', linewidth=2.5, 
                    markersize=10, label='Therapist', color='#e67e22', markeredgewidth=2, markeredgecolor='white')
            # Add annotations for therapist real-life linking
            for idx, row in therapist_trend.iterrows():
                ax2.annotate(f'{row[q25]:.2f}', 
                           xy=(row['Session number'], row[q25]),
                           xytext=(0, -28), textcoords='offset points',
                           ha='center', fontsize=10, fontweight='bold',
                           color='#e67e22', family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Real-Life Linking Score', fontsize=12, fontweight='bold')
        ax2.set_title('Real-Life Experience Linking Across Sessions:\nParent vs Therapist Perspective', fontsize=12, fontweight='bold', pad=15)
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
        
    
    # ============ AUTISM LEVEL ANALYSIS ============
    st.divider()
    st.subheader("Autism Level Analysis")
    st.markdown("**Generalization and Real-Life Linking Trends by Autism Level**")
    
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
        
        # Plot 1: Generalization (Q22) trends by autism level
        for autism_level in autism_levels:
            autism_trend = df[df['Autism Level'] == autism_level].groupby('Session number')[[q22, q25]].mean().reset_index()
            
            if len(autism_trend) > 0:
                color = colors_map.get(autism_level, '#95a5a6')
                ax1.plot(autism_trend['Session number'], autism_trend[q22], marker='o', linewidth=2.5, 
                        markersize=10, label=f'Level {int(autism_level)}', color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations - Level 1 above, Level 2 below, others alternating
                for idx, row in autism_trend.iterrows():
                    if autism_level == 1:
                        xytext_offset = (0, 15)  # Above for Level 1
                    elif autism_level >= 2:
                        xytext_offset = (0, -20)  # Below for Level 2
                    else:
                        xytext_offset = (0, 12) if idx % 2 == 0 else (0, -15)  # Alternate for others
                    ax1.annotate(f'{row[q22]:.2f}', 
                               xy=(row['Session number'], row[q22]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Generalization Score', fontsize=12, fontweight='bold')
        ax1.set_title('Skill Generalization Across Sessions by Autism Level', fontsize=12, fontweight='bold', pad=15)
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
        
        # Plot 2: Real-Life Linking (Q25) trends by autism level
        for autism_level in autism_levels:
            autism_trend = df[df['Autism Level'] == autism_level].groupby('Session number')[[q22, q25]].mean().reset_index()
            
            if len(autism_trend) > 0:
                color = colors_map.get(autism_level, '#95a5a6')
                ax2.plot(autism_trend['Session number'], autism_trend[q25], marker='o', linewidth=2.5, 
                        markersize=10, label=f'Level {int(autism_level)}', color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations - Level 1 above, Level 2 below, others alternating
                for idx, row in autism_trend.iterrows():
                    if autism_level == 1:
                        xytext_offset = (0, 15)  # Above for Level 1
                    elif autism_level == 2:
                        xytext_offset = (0, -20)  # Below for Level 2
                    else:
                        xytext_offset = (0, 12) if idx % 2 == 0 else (0, -15)  # Alternate for others
                    ax2.annotate(f'{row[q25]:.2f}', 
                               xy=(row['Session number'], row[q25]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Real-Life Linking Score', fontsize=12, fontweight='bold')
        ax2.set_title('Real-Life Experience Linking Across Sessions by Autism Level', fontsize=12, fontweight='bold', pad=15)
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
    else:
        st.warning("No Autism Level data available for analysis")
    
    # ============ AGE GROUP ANALYSIS ============
    st.divider()
    st.subheader("Age Group Analysis")
    st.markdown("**Generalization and Real-Life Linking Trends by Age Group**")
    
    # Define age groups
    age_bins = [8, 15, 20, 27]
    age_labels = ['8-14', '15-19', '20-26']
    df_age = df.copy()
    df_age['Age Group'] = pd.cut(df_age['Age'], bins=age_bins, labels=age_labels, right=False)
    
    age_groups = sorted(df_age['Age Group'].dropna().unique())
    
    if len(age_groups) > 0:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Define colors for different age groups
        age_colors_map = {
            '8-14': '#3498db',      # Blue
            '15-19': '#e74c3c',     # Red
            '20-26': '#2ecc71'      # Green
        }
        
        # Plot 1: Generalization (Q22) trends by age group
        for age_group in age_groups:
            age_trend = df_age[df_age['Age Group'] == age_group].groupby('Session number')[[q22, q25]].mean().reset_index()
            
            if len(age_trend) > 0:
                color = age_colors_map.get(age_group, '#95a5a6')
                ax1.plot(age_trend['Session number'], age_trend[q22], marker='o', linewidth=2.5, 
                        markersize=10, label=age_group, color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations - specific positioning per age group
                for idx, row in age_trend.iterrows():
                    if age_group == '15-19':
                        xytext_offset = (0, 15)  # Above for 15-19
                    else:  # 8-14 and 20-26
                        xytext_offset = (0, -20)  # Below for 8-14 and 20-26
                    ax1.annotate(f'{row[q22]:.2f}', 
                               xy=(row['Session number'], row[q22]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax1.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Generalization Score', fontsize=12, fontweight='bold')
        ax1.set_title('Skill Generalization Across Sessions by Age Group', fontsize=12, fontweight='bold', pad=15)
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
        
        # Plot 2: Real-Life Linking (Q25) trends by age group
        for age_group in age_groups:
            age_trend = df_age[df_age['Age Group'] == age_group].groupby('Session number')[[q22, q25]].mean().reset_index()
            
            if len(age_trend) > 0:
                color = age_colors_map.get(age_group, '#95a5a6')
                ax2.plot(age_trend['Session number'], age_trend[q25], marker='o', linewidth=2.5, 
                        markersize=10, label=age_group, color=color, markeredgewidth=2, markeredgecolor='white')
                
                # Add annotations - specific positioning per age group
                for idx, row in age_trend.iterrows():
                    if age_group == '15-19':
                        xytext_offset = (0, 15)  # Above for 15-19
                    else:  # 8-14 and 20-26
                        xytext_offset = (0, -20)  # Below for 8-14 and 20-26
                    ax2.annotate(f'{row[q25]:.2f}', 
                               xy=(row['Session number'], row[q25]),
                               xytext=xytext_offset, textcoords='offset points',
                               ha='center', fontsize=9, fontweight='bold',
                               color=color, family='sans-serif')
        
        ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Real-Life Linking Score', fontsize=12, fontweight='bold')
        ax2.set_title('Real-Life Experience Linking Across Sessions by Age Group', fontsize=12, fontweight='bold', pad=15)
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
        
        # Statistics table for age group comparison
        st.markdown("**Summary Statistics: Generalization and Real-Life Linking by Age Group**")
        
        age_group_data = []
        for age_group in age_groups:
            age_group_df = df_age[df_age['Age Group'] == age_group]
            if len(age_group_df) > 0:
                age_group_data.append({
                    'Age Group': age_group,
                    'Total Observations': len(age_group_df),
                    'Unique Participants': len(age_group_df['Participant id'].unique()),
                    'Avg Generalization': f"{age_group_df[q22].mean():.2f}",
                    'Generalization Std Dev': f"{age_group_df[q22].std():.2f}",
                    'Avg Real-Life Linking': f"{age_group_df[q25].mean():.2f}",
                    'Real-Life Linking Std Dev': f"{age_group_df[q25].std():.2f}"
                })
        
        age_group_comparison_df = pd.DataFrame(age_group_data)
        st.dataframe(age_group_comparison_df, use_container_width=True, hide_index=True)
        
        # Box plot for distribution comparison
        st.markdown("**Distribution Analysis: Score Variation Across Age Groups**")
        
        fig, (ax_box1, ax_box2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        # Define scale labels for y-axis
        scale_labels_list = ['0: Not at all', '1: Slightly', '2: Moderately', '3: Very', '4: Fully']
        
        # Box plot for Q22 using seaborn for better control
        sns.boxplot(data=df_age, x='Age Group', y=q22, ax=ax_box1, palette=['#3498db', '#e74c3c', '#2ecc71'])
        ax_box1.set_title('Generalization (Q22) Distribution by Age Group', fontsize=12, fontweight='bold')
        ax_box1.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax_box1.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax_box1.set_ylim(-0.5, 4.5)
        ax_box1.set_yticks([0, 1, 2, 3, 4])
        ax_box1.set_yticklabels(scale_labels_list, fontsize=9)
        ax_box1.grid(True, alpha=0.3, axis='y')
        
        # Box plot for Q25 using seaborn for better control
        sns.boxplot(data=df_age, x='Age Group', y=q25, ax=ax_box2, palette=['#3498db', '#e74c3c', '#2ecc71'])
        ax_box2.set_title('Real-Life Linking (Q25) Distribution by Age Group', fontsize=12, fontweight='bold')
        ax_box2.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax_box2.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax_box2.set_ylim(-0.5, 4.5)
        ax_box2.set_yticks([0, 1, 2, 3, 4])
        ax_box2.set_yticklabels(scale_labels_list, fontsize=9)
        ax_box2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No Age Group data available for analysis")