import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ8 Overall Social Behaviour Impact"""
    st.subheader("RQ8: Overall Social Behaviour Improvement")
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
    
    # Add spacing between figures
    st.write("")
    st.write("")
    
    # --- THIRD FIGURE: Breakdown by Parent vs Therapist Perspective ---
    st.markdown("---")
    st.markdown("#### Analysis by Parent vs Therapist Perspective")
    st.write("")
    
    if 'Submitted_by' in df.columns:
        # Create perspective mapping
        perspective_map = {'T': 'Therapist', 'P': 'Parent'}
        df_copy = df.copy()
        df_copy['Perspective'] = df_copy['Submitted_by'].map(perspective_map)
        
        # Prepare data by perspective
        perspective_trend = df_copy.groupby(['Session number', 'Perspective'])[q26].mean().reset_index()
        
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        fig3.patch.set_facecolor('white')
        
        # Get unique perspectives
        perspectives = sorted(perspective_trend['Perspective'].dropna().unique())
        colors_perspective = ['#FF6B6B', '#4ECDC4']  # Red for Parent, Teal for Therapist
        color_map_perspective = {persp: colors_perspective[i] for i, persp in enumerate(perspectives)}
        
        # Plot line for each perspective
        for perspective in perspectives:
            persp_data = perspective_trend[perspective_trend['Perspective'] == perspective]
            ax3.plot(persp_data['Session number'], persp_data[q26], 
                    marker='o', linewidth=2.5, markersize=10, 
                    label=perspective, 
                    color=color_map_perspective[perspective])
            
            # Add value annotations for each point
            for _, row in persp_data.iterrows():
                # Position annotations below for Therapist, above for Parent
                if perspective == 'Therapist':
                    ax3.text(row['Session number'], row[q26] - 0.8, f'{round(row[q26], 2):.2f}',
                            ha='center', va='top', fontsize=9, fontweight='bold',
                            color=color_map_perspective[perspective])
                else:  # Parent
                    ax3.text(row['Session number'], row[q26] + 0.3, f'{round(row[q26], 2):.2f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color=color_map_perspective[perspective])
        
        ax3.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax3.set_axisbelow(True)
        ax3.set_ylim(-0.5, 11)
        ax3.set_yticks(ticks=[0,2,4,6,8,10])
        ax3.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=10)
        
        # Clean axes
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.spines['bottom'].set_linewidth(1.5)
        
        ax3.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Social Impact Score', fontsize=12, fontweight='bold')
        ax3.set_title('Social Behavior Impact: Parent vs Therapist Perspective Across Sessions', 
                     fontsize=13, fontweight='bold', pad=20)
        ax3.legend(fontsize=11, loc='upper left', frameon=False)
        
        fig3.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Add spacing
        st.write("")
        st.write("")
        
        # Summary statistics by perspective
        st.markdown("**Summary Statistics by Perspective:**")
        summary_stats = df_copy.groupby('Perspective')[q26].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(summary_stats, use_container_width=True)
    else:
        st.info("Submitted_by column not found for parent/therapist perspective analysis.")
    
    # Add spacing between figures
    st.write("")
    st.write("")
    
    # --- FOURTH FIGURE: Analysis by Age Group ---
    st.divider()
    st.subheader("Age Group Analysis")
    st.markdown("**Social Behavior Impact Trends by Age Group**")
    
    if 'Age' in df.columns:
        # Define age groups
        age_bins = [0, 8, 15, 20, 27, 36, 100]
        age_labels = ['<8', '8-14', '15-19', '20-26', '27-35', '36+']
        df_age = df.copy()
        df_age['Age_Group'] = pd.cut(df_age['Age'], bins=age_bins, labels=age_labels, right=False)
        
        # Get unique age groups
        age_groups = sorted(df_age['Age_Group'].dropna().unique())
        
        if len(age_groups) > 0:
            # Create visualization
            fig4, ax4 = plt.subplots(figsize=(14, 7))
            fig4.patch.set_facecolor('white')
            
            # Define colors for age groups
            colors_age = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F8B195']
            color_map_age = {age_group: colors_age[i % len(colors_age)] for i, age_group in enumerate(age_groups)}
            
            # Plot line for each age group
            for age_group in age_groups:
                age_trend = df_age[df_age['Age_Group'] == age_group].groupby('Session number')[q26].mean().reset_index()
                
                if len(age_trend) > 0:
                    ax4.plot(age_trend['Session number'], age_trend[q26], 
                            marker='o', linewidth=2, markersize=8, 
                            label=f'Age {age_group}', 
                            color=color_map_age[age_group])
                    
                    # Add value annotations
                    for _, row in age_trend.iterrows():
                        # Position annotations below for age group 8-14, above for others
                        if str(age_group) == '8-14':
                            ax4.text(row['Session number'], row[q26] - 0.6, f'{round(row[q26], 2):.2f}',
                                    ha='center', va='top', fontsize=8, color=color_map_age[age_group])
                        elif str(age_group) == '15-19':
                            ax4.text(row['Session number'], row[q26] + 0.5, f'{round(row[q26], 2):.2f}',
                                    ha='center', va='bottom', fontsize=8, color=color_map_age[age_group])
                        else:
                            ax4.text(row['Session number'], row[q26] + 0.2, f'{round(row[q26], 2):.2f}',
                                    ha='center', va='bottom', fontsize=8, color=color_map_age[age_group])
            
            ax4.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax4.set_axisbelow(True)
            ax4.set_ylim(-0.5, 11)
            ax4.set_yticks(ticks=[0,2,4,6,8,10])
            ax4.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=10)
            
            # Clean axes
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['left'].set_linewidth(1.5)
            ax4.spines['bottom'].set_linewidth(1.5)
            
            ax4.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Social Impact Score', fontsize=12, fontweight='bold')
            ax4.set_title('Social Behavior Impact by Age Group Across Sessions', 
                         fontsize=13, fontweight='bold', pad=20)
            ax4.legend(fontsize=10, loc='best', frameon=True)
            
            fig4.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95)
            plt.tight_layout()
            st.pyplot(fig4)
            
            # Statistics table for age group comparison
            st.markdown("**Summary Statistics by Age Group:**")
            
            age_data = []
            for age_group in age_groups:
                age_df = df_age[df_age['Age_Group'] == age_group]
                if len(age_df) > 0:
                    age_data.append({
                        'Age Group': f'{age_group}',
                        'Total Observations': len(age_df),
                        'Unique Participants': len(age_df['Participant id'].unique()),
                        'Avg Score': f"{age_df[q26].mean():.2f}",
                        'Std Dev': f"{age_df[q26].std():.2f}",
                        'Min': f"{age_df[q26].min():.2f}",
                        'Max': f"{age_df[q26].max():.2f}"
                    })
            
            age_comparison_df = pd.DataFrame(age_data)
            st.dataframe(age_comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("No age group data available for analysis.")
    else:
        st.info("Age column not found for age group analysis.")
    
    # Add spacing between figures
    st.write("")
    st.write("")
    
    # --- FIFTH FIGURE: Analysis by Gender ---
    st.divider()
    st.subheader("Gender Analysis")
    st.markdown("**Social Behavior Impact Trends by Gender**")
    
    if 'Gender' in df.columns:
        gender_groups = df['Gender'].dropna().unique()
        
        if len(gender_groups) > 0:
            # Create visualization
            fig5, ax5 = plt.subplots(figsize=(12, 7))
            fig5.patch.set_facecolor('white')
            
            # Define colors for gender
            colors_gender = ['#3498db', '#e74c3c']  # Blue for Male, Red for Female
            gender_color_map = {'Male': '#3498db', 'Female': '#e74c3c', 'Other': '#95a5a6'}
            
            # Plot line for each gender
            for idx, gender in enumerate(sorted(gender_groups)):
                gender_trend = df[df['Gender'] == gender].groupby('Session number')[q26].mean().reset_index()
                
                if len(gender_trend) > 0:
                    color = gender_color_map.get(gender, colors_gender[idx % len(colors_gender)])
                    ax5.plot(gender_trend['Session number'], gender_trend[q26], 
                            marker='o', linewidth=2.5, markersize=10, 
                            label=gender, 
                            color=color)
                    
                    # Add value annotations - Male above, Female below
                    for jdx, row in gender_trend.iterrows():
                        if gender == 'Male':
                            # Male annotation above
                            ax5.text(row['Session number'], row[q26] + 0.3, f'{round(row[q26], 2):.2f}',
                                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                                    color=color)
                        else:
                            # Other genders (Female, Other) annotation below
                            ax5.text(row['Session number'], row[q26] - 0.8, f'{round(row[q26], 2):.2f}',
                                    ha='center', va='top', fontsize=9, fontweight='bold',
                                    color=color)
            
            ax5.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax5.set_axisbelow(True)
            ax5.set_ylim(-0.5, 11)
            ax5.set_yticks(ticks=[0,2,4,6,8,10])
            ax5.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=10)
            
            # Clean axes
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['left'].set_linewidth(1.5)
            ax5.spines['bottom'].set_linewidth(1.5)
            
            ax5.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Social Impact Score', fontsize=12, fontweight='bold')
            ax5.set_title('Social Behavior Impact by Gender Across Sessions', 
                         fontsize=13, fontweight='bold', pad=20)
            ax5.legend(fontsize=11, loc='best', frameon=True)
            
            fig5.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95)
            plt.tight_layout()
            st.pyplot(fig5)
            
            # Statistics table for gender comparison
            st.markdown("**Summary Statistics by Gender:**")
            
            gender_data = []
            for gender in sorted(gender_groups):
                gender_df = df[df['Gender'] == gender]
                if len(gender_df) > 0:
                    gender_data.append({
                        'Gender': gender,
                        'Total Observations': len(gender_df),
                        'Unique Participants': len(gender_df['Participant id'].unique()),
                        'Avg Score': f"{gender_df[q26].mean():.2f}",
                        'Std Dev': f"{gender_df[q26].std():.2f}",
                        'Min': f"{gender_df[q26].min():.2f}",
                        'Max': f"{gender_df[q26].max():.2f}"
                    })
            
            gender_comparison_df = pd.DataFrame(gender_data)
            st.dataframe(gender_comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("No gender data available for analysis.")
    else:
        st.info("Gender column not found for gender analysis.")

