import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def categorize_age_groups(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return None
    age = int(age)
    if age <= 14:
        return '8-14'
    elif age <= 19:
        return '15-19'
    else:
        return '20-26'

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

    # ============ AGE GROUP ANALYSIS ============
    st.divider()
    st.markdown("**Age Group Analysis**")
    
    # Prepare data with age groups
    df_with_age = df[["Session number", q9, "Age"]].copy()
    df_with_age = df_with_age.dropna(subset=[q9])
    df_with_age['Age Group'] = df_with_age['Age'].apply(categorize_age_groups)
    df_with_age = df_with_age.dropna(subset=['Age Group'])
    
    # Create age group trend analysis
    age_groups_order = ['8-14', '15-19', '20-26']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Define colors for each age group
    colors = {'8-14': '#FF6B6B', '15-19': '#87CEEB', '20-26': '#9B59B6'}
    
    # Plot trend for each age group
    for age_group in age_groups_order:
        age_data = df_with_age[df_with_age['Age Group'] == age_group]
        if len(age_data) > 0:
            trend_age = age_data.groupby('Session number')[q9].mean().reset_index()
            if not trend_age.empty:
                ax.plot(trend_age['Session number'], trend_age[q9], 
                       marker='o', linewidth=2.5, markersize=10, 
                       color=colors[age_group], markerfacecolor=colors[age_group], 
                       markeredgewidth=2, markeredgecolor='white',
                       label=f'{age_group} years (n={len(age_data)})')
                
                # Add value annotations
                for idx, row in trend_age.iterrows():
                    if age_group == '8-14':
                        offset = -0.15
                        va = 'top'
                    elif age_group == '15-19':
                        offset = 0.08
                        va = 'bottom'
                    else:  # 20-26
                        offset = 0.15
                        va = 'bottom'
                    ax.text(row['Session number'], row[q9] + offset, 
                           f"{row[q9]:.2f}", ha='center', va=va, 
                           fontsize=9, color=colors[age_group])
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(ticks=[0,1,2,3,4])
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Initiation Score', fontsize=12, fontweight='bold')
    ax.set_title('Self-Initiated Interaction Trends Across Age Groups',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    # ============ AGE GROUP 8-14 BREAKDOWN BY GENDER ============
    st.markdown("**Age Group 8-14 Breakdown: Male vs Female**")
    
    # Prepare data for age group 8-14 with gender
    df_age_8_14 = df_with_age[df_with_age['Age Group'] == '8-14'].copy()
    df_age_8_14['Gender'] = df.loc[df_age_8_14.index, 'Gender']
    df_age_8_14 = df_age_8_14.dropna(subset=['Gender'])
    
    if len(df_age_8_14) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        genders = sorted(df_age_8_14['Gender'].unique())
        gender_colors = {'Male': '#2ecc71', 'Female': '#f39c12'}
        
        # Plot trend for each gender
        for gender in genders:
            gender_data = df_age_8_14[df_age_8_14['Gender'] == gender]
            if len(gender_data) > 0:
                trend_gender = gender_data.groupby('Session number')[q9].mean().reset_index()
                if not trend_gender.empty:
                    marker = 'o' if gender == 'Male' else 's'
                    ax.plot(trend_gender['Session number'], trend_gender[q9], 
                           marker=marker, linewidth=2.5, markersize=10, 
                           color=gender_colors[gender], markerfacecolor=gender_colors[gender], 
                           markeredgewidth=2, markeredgecolor='white',
                           label=f'{gender} (n={len(gender_data)})')
                    
                    # Add value annotations
                    for idx, row in trend_gender.iterrows():
                        offset = 0.15 if gender == 'Male' else -0.15
                        va = 'bottom' if gender == 'Male' else 'top'
                        ax.text(row['Session number'], row[q9] + offset, 
                               f"{row[q9]:.2f}", ha='center', va=va, 
                               fontsize=9, color=gender_colors[gender], fontweight='bold')
        
        ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        ax.set_ylim(-0.3, 4.3)
        ax.set_yticks(ticks=[0,1,2,3,4])
        if scale_map:
            scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
            ax.set_yticklabels(scale_labels, fontsize=10)
        else:
            ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Initiation Score', fontsize=12, fontweight='bold')
        ax.set_title('Self-Initiated Interaction Trends: Age Group 8-14 (Male vs Female)', 
                    fontsize=13, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11, loc='upper left', frameon=False)
        plt.tight_layout()
        st.pyplot(fig)

    # ============ AUTISM LEVEL ANALYSIS ============
    st.divider()
    st.markdown("**Autism Level Analysis**")
    
    # Prepare data with autism level
    df_with_autism = df[["Session number", q9, "Autism Level"]].copy()
    df_with_autism = df_with_autism.dropna(subset=[q9, "Autism Level"])
    
    # Create autism level trend analysis
    autism_levels = sorted(df_with_autism["Autism Level"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Define colors for each autism level
    colors_autism = {1: '#E74C3C', 2: '#3498DB'}
    labels_autism = {1: 'Level 1', 2: 'Level 2'}
    
    # Plot trend for each autism level
    for autism_level in autism_levels:
        autism_data = df_with_autism[df_with_autism["Autism Level"] == autism_level]
        if len(autism_data) > 0:
            trend_autism = autism_data.groupby('Session number')[q9].mean().reset_index()
            if not trend_autism.empty:
                ax.plot(trend_autism['Session number'], trend_autism[q9], 
                       marker='o', linewidth=2.5, markersize=10, 
                       color=colors_autism[autism_level], markerfacecolor=colors_autism[autism_level], 
                       markeredgewidth=2, markeredgecolor='white',
                       label=f'{labels_autism[autism_level]} (n={len(autism_data)})')
                
                # Add value annotations
                for idx, row in trend_autism.iterrows():
                    offset = -0.2 if autism_level == 1 else 0.2
                    va = 'top' if autism_level == 1 else 'bottom'
                    ax.text(row['Session number'], row[q9] + offset, 
                           f"{row[q9]:.2f}", ha='center', va=va, 
                           fontsize=9, color=colors_autism[autism_level])
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(ticks=[0,1,2,3,4])
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Initiation Score', fontsize=12, fontweight='bold')
    ax.set_title('Self-Initiated Interaction Trends by Autism Level',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    # ============ THERAPIST vs PARENT PERSPECTIVE ============
    st.divider()
    st.markdown("**Comparative Analysis: Therapist vs Parent Perspective**")
    
    # Prepare data with submitted_by (therapist vs parent)
    df_with_perspective = df[["Session number", q9, "Submitted_by"]].copy()
    df_with_perspective = df_with_perspective.dropna(subset=[q9, "Submitted_by"])
    
    trend_T = df_with_perspective[df_with_perspective['Submitted_by']=='T'].groupby('Session number')[q9].mean().reset_index()
    trend_P = df_with_perspective[df_with_perspective['Submitted_by']=='P'].groupby('Session number')[q9].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Plot therapist line
    if not trend_T.empty:
        ax.plot(trend_T['Session number'], trend_T[q9], marker='o', linewidth=2.5, 
                markersize=10, color='#FF8C00', markerfacecolor='#FF8C00', markeredgewidth=2, markeredgecolor='white',
                label=f'Therapist (T) (n={len(df_with_perspective[df_with_perspective["Submitted_by"]=="T"])})')
        for idx, row in trend_T.iterrows():
            ax.text(row['Session number'], row[q9] - 0.15, f"{row[q9]:.2f}",
                   ha='center', va='top', fontsize=10, color='#FF8C00', fontweight='bold')
    
    # Plot parent line
    if not trend_P.empty:
        ax.plot(trend_P['Session number'], trend_P[q9], marker='s', linewidth=2.5, 
                markersize=10, color='#006400', markerfacecolor='#006400', markeredgewidth=2, markeredgecolor='white',
                label=f'Parent (P) (n={len(df_with_perspective[df_with_perspective["Submitted_by"]=="P"])})')
        for idx, row in trend_P.iterrows():
            ax.text(row['Session number'], row[q9] + 0.25, f"{row[q9]:.2f}",
                   ha='center', va='bottom', fontsize=10, color='#006400', fontweight='bold')
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(ticks=[0,1,2,3,4])
    if scale_map:
        scale_labels = [f"{k}: {v}" for k, v in sorted(scale_map.items())]
        ax.set_yticklabels(scale_labels, fontsize=10)
    else:
        ax.set_yticklabels(['0: Not at all','1: Slightly','2: Moderately','3: Very','4: Fully'], fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Initiation Score', fontsize=12, fontweight='bold')
    ax.set_title('Self-Initiated Interaction Trends: Therapist vs Parent Perspective',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', frameon=False)
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
