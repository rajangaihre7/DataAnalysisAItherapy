import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def display(df, scale_map=None):
    """Display RQ6: Response Time Trends"""
    st.subheader("RQ6: Response Time Improvement")
    response_col = "What is the response time of the participant? (in minutes roughly)"
    st.markdown("**Q15:** " + response_col)
    
    # Check if response_seconds column exists, if not parse it
    if 'response_seconds' not in df.columns:
        import numpy as np
        import re
        
        def parse_response_time(val):
            if pd.isna(val):
                return np.nan
            val = str(val).lower().strip()
            if any(x in val for x in ['null', 'nothing', 'not response']):
                return np.nan
            # Range handling
            range_match = re.findall(r'(\d+)\s*[-to]+\s*(\d+)\s*(min|minute|minutes|sec|second|seconds)?', val)
            if range_match:
                start, end, unit = range_match[0]
                avg = (float(start) + float(end)) / 2
                if unit and 'min' in unit:
                    return avg * 60
                return avg
            # Minutes
            min_match = re.search(r'(\d+(?:\.\d+)?)\s*(min|minute|minutes)', val)
            if min_match:
                return float(min_match.group(1)) * 60
            # Seconds
            sec_match = re.search(r'(\d+(?:\.\d+)?)\s*(sec|second|seconds)', val)
            if sec_match:
                return float(sec_match.group(1))
            return np.nan
        
        if response_col in df.columns:
            df['response_seconds'] = df[response_col].apply(parse_response_time)
    
    # Calculate average response time per session
    if 'response_seconds' in df.columns:
        trend = df[df['response_seconds'].notna()].groupby('Session number')['response_seconds'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        # Line plot with markers
        ax.plot(trend['Session number'], trend['response_seconds'],
                marker='o', linewidth=2.5, markersize=9,
                color='#FF6B35', markerfacecolor='#FF6B35', markeredgewidth=2, markeredgecolor='white')
        
        # Annotate values above line
        for _, row in trend.iterrows():
            seconds = row['response_seconds']
            ax.text(row['Session number'], seconds + 5, f'{seconds:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FF6B35')
        
        # Styling
        ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(trend['response_seconds']) + 20)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Longitudinal Trends in Response Time Across Sessions',
                     fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add spacing
        st.write("")
        st.write("")
        
        # Breakdown by Autism Level
        st.markdown("---")
        st.markdown("#### Analysis by Autism Level")
        st.write("")
        
        if 'Autism Level' in df.columns:
            autism_trend = df[df['response_seconds'].notna()].groupby(['Session number', 'Autism Level'])['response_seconds'].mean().reset_index()
            
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            fig2.patch.set_facecolor('white')
            
            # Get unique autism levels
            autism_levels = sorted(autism_trend['Autism Level'].dropna().unique())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            color_map = {level: colors[i % len(colors)] for i, level in enumerate(autism_levels)}
            
            # Plot for each level
            for level in autism_levels:
                level_data = autism_trend[autism_trend['Autism Level'] == level]
                ax2.plot(level_data['Session number'], level_data['response_seconds'],
                        marker='o', linewidth=2, markersize=8,
                        label=f'Autism Level: {level}',
                        color=color_map[level])
                
                # Add annotations - red below, green above
                for _, row in level_data.iterrows():
                    seconds = row['response_seconds']
                    color = color_map[level]
                    # Red annotations below, green annotations above
                    if color == '#FF6B6B':  # Red
                        offset = -5
                        va = 'top'
                    elif color in ['#4ECDC4', '#98D8C8']:  # Green/Teal
                        offset = 5
                        va = 'bottom'
                    else:  # Other colors below
                        offset = -5
                        va = 'top'
                    
                    ax2.text(row['Session number'], seconds + offset, f'{seconds:.2f}',
                            ha='center', va=va, fontsize=8, color=color, fontweight='bold')
            
            ax2.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax2.set_axisbelow(True)
            ax2.set_ylim(0, max(autism_trend['response_seconds']) + 20)
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.5)
            ax2.spines['bottom'].set_linewidth(1.5)
            
            ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax2.set_title('Response Time Trends by Autism Level',
                         fontsize=13, fontweight='bold', pad=20)
            ax2.legend(fontsize=10, loc='upper right', frameon=False)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Add spacing
            st.write("")
            st.write("")
            
            # Third chart: Bar chart of average response time by Autism Level
            st.markdown("---")
            st.markdown("#### Average Response Time by Autism Level")
            st.write("")
            
            avg_by_level = df[df['response_seconds'].notna()].groupby('Autism Level')['response_seconds'].mean().reset_index()
            avg_by_level = avg_by_level.sort_values('Autism Level')
            
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            fig3.patch.set_facecolor('white')
            
            bars = ax3.bar(avg_by_level['Autism Level'].astype(str), avg_by_level['response_seconds'],
                          color='#4ECDC4', edgecolor='black', linewidth=1.5, width=0.6)
            
            # Add value annotations on bars
            for idx, (level, value) in enumerate(zip(avg_by_level['Autism Level'], avg_by_level['response_seconds'])):
                ax3.text(idx, value + 3, f'{value:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color='#4ECDC4', family='sans-serif')
            
            # Styling
            ax3.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax3.set_axisbelow(True)
            ax3.set_ylim(0, max(avg_by_level['response_seconds']) + 15)
            
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_linewidth(1.5)
            ax3.spines['bottom'].set_linewidth(1.5)
            
            ax3.set_xlabel('Autism Level', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
            ax3.set_title('Average Response Time by Autism Level',
                         fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig3)
    else:
        st.info("Response time data not available for trend analysis.")
    
    # --- AGE GROUP ANALYSIS ---
    st.divider()
    st.subheader("Response Time Analysis by Age Group")
    st.markdown("**Response Time Trends for Age Groups: 8-14, 15-19, 20-26**")
    
    if 'Age' in df.columns and 'response_seconds' in df.columns:
        import numpy as np
        
        # Define age groups
        age_bins = [0, 8, 15, 20, 27, 100]
        age_labels = ['<8', '8-14', '15-19', '20-26', '27+']
        df_age = df.copy()
        df_age['Age_Group'] = pd.cut(df_age['Age'], bins=age_bins, labels=age_labels, right=False)
        
        age_groups_to_plot = ['8-14', '15-19', '20-26']
        df_age_filtered = df_age[df_age['Age_Group'].isin(age_groups_to_plot) & df_age['response_seconds'].notna()]
        
        # Age Group 8-14: Annotation ABOVE the line
        if len(df_age_filtered) > 0:
            age_8_14_data = df_age_filtered[df_age_filtered['Age_Group'] == '8-14'].groupby('Session number')['response_seconds'].mean().reset_index()
            
            if len(age_8_14_data) > 0:
                st.markdown("#### Age Group 8-14")
                fig_8_14, ax_8_14 = plt.subplots(figsize=(12, 7))
                fig_8_14.patch.set_facecolor('white')
                
                ax_8_14.plot(age_8_14_data['Session number'], age_8_14_data['response_seconds'],
                            marker='o', linewidth=2.5, markersize=10, color='#FF6B35',
                            markerfacecolor='#FF6B35', markeredgewidth=2, markeredgecolor='white')
                
                # Annotations ABOVE the line
                for _, row in age_8_14_data.iterrows():
                    ax_8_14.text(row['Session number'], row['response_seconds'] + 5, f'{row["response_seconds"]:.2f}',
                                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FF6B35')
                
                ax_8_14.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.7, color='gray')
                ax_8_14.set_axisbelow(True)
                ax_8_14.spines['top'].set_visible(False)
                ax_8_14.spines['right'].set_visible(False)
                ax_8_14.spines['left'].set_linewidth(1.5)
                ax_8_14.spines['bottom'].set_linewidth(1.5)
                
                max_val = age_8_14_data['response_seconds'].max()
                ax_8_14.set_ylim(0, max_val + 20)
                
                ax_8_14.set_xlabel('Session Number', fontsize=12, fontweight='bold')
                ax_8_14.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
                ax_8_14.set_title('Response Time Trends - Age Group 8-14', fontsize=13, fontweight='bold', pad=20)
                
                plt.tight_layout()
                st.pyplot(fig_8_14)
                st.write("")
        
        # Age Group 15-19: Annotation BELOW the line
        age_15_19_data = df_age_filtered[df_age_filtered['Age_Group'] == '15-19'].groupby('Session number')['response_seconds'].mean().reset_index()
        
        if len(age_15_19_data) > 0:
            st.markdown("#### Age Group 15-19")
            fig_15_19, ax_15_19 = plt.subplots(figsize=(12, 7))
            fig_15_19.patch.set_facecolor('white')
            
            ax_15_19.plot(age_15_19_data['Session number'], age_15_19_data['response_seconds'],
                         marker='o', linewidth=2.5, markersize=10, color='#4ECDC4',
                         markerfacecolor='#4ECDC4', markeredgewidth=2, markeredgecolor='white')
            
            # Annotations BELOW the line
            for _, row in age_15_19_data.iterrows():
                ax_15_19.text(row['Session number'], row['response_seconds'] - 5, f'{row["response_seconds"]:.2f}',
                             ha='center', va='top', fontsize=10, fontweight='bold', color='#4ECDC4')
            
            ax_15_19.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.7, color='gray')
            ax_15_19.set_axisbelow(True)
            ax_15_19.spines['top'].set_visible(False)
            ax_15_19.spines['right'].set_visible(False)
            ax_15_19.spines['left'].set_linewidth(1.5)
            ax_15_19.spines['bottom'].set_linewidth(1.5)
            
            max_val = age_15_19_data['response_seconds'].max()
            ax_15_19.set_ylim(0, max_val + 20)
            
            ax_15_19.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax_15_19.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax_15_19.set_title('Response Time Trends - Age Group 15-19', fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig_15_19)
            st.write("")
        
        # Age Group 20-26: Annotation IN THE MIDDLE of the line
        age_20_26_data = df_age_filtered[df_age_filtered['Age_Group'] == '20-26'].groupby('Session number')['response_seconds'].mean().reset_index()
        
        if len(age_20_26_data) > 0:
            st.markdown("#### Age Group 20-26")
            fig_20_26, ax_20_26 = plt.subplots(figsize=(12, 7))
            fig_20_26.patch.set_facecolor('white')
            
            ax_20_26.plot(age_20_26_data['Session number'], age_20_26_data['response_seconds'],
                         marker='o', linewidth=2.5, markersize=10, color='#45B7D1',
                         markerfacecolor='#45B7D1', markeredgewidth=2, markeredgecolor='white')
            
            # Annotations ABOVE the line
            for _, row in age_20_26_data.iterrows():
                ax_20_26.text(row['Session number'], row['response_seconds'] + 5, f'{row["response_seconds"]:.2f}',
                             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#45B7D1')
            
            ax_20_26.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.7, color='gray')
            ax_20_26.set_axisbelow(True)
            ax_20_26.spines['top'].set_visible(False)
            ax_20_26.spines['right'].set_visible(False)
            ax_20_26.spines['left'].set_linewidth(1.5)
            ax_20_26.spines['bottom'].set_linewidth(1.5)
            
            max_val = age_20_26_data['response_seconds'].max()
            ax_20_26.set_ylim(0, max_val + 20)
            
            ax_20_26.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax_20_26.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax_20_26.set_title('Response Time Trends - Age Group 20-26', fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig_20_26)
    else:
        st.info("Age or response time data not available for age group analysis.")
    
    # --- GENDER ANALYSIS ---
    st.divider()
    st.subheader("Response Time Analysis by Gender")
    st.markdown("**Response Time Trends for Male and Female**")
    
    if 'Gender' in df.columns and 'response_seconds' in df.columns:
        df_gender = df[df['response_seconds'].notna()].copy()
        
        # Get data for both male and female
        male_data = df_gender[df_gender['Gender'] == 'Male'].groupby('Session number')['response_seconds'].mean().reset_index()
        female_data = df_gender[df_gender['Gender'] == 'Female'].groupby('Session number')['response_seconds'].mean().reset_index()
        
        if len(male_data) > 0 or len(female_data) > 0:
            fig_gender, ax_gender = plt.subplots(figsize=(12, 7))
            fig_gender.patch.set_facecolor('white')
            
            # Plot Male line
            if len(male_data) > 0:
                ax_gender.plot(male_data['Session number'], male_data['response_seconds'],
                              marker='o', linewidth=2.5, markersize=10, color='#2980B9',
                              markerfacecolor='#2980B9', markeredgewidth=2, markeredgecolor='white',
                              label='Male')
                
                # Annotations ABOVE the line for Male
                for _, row in male_data.iterrows():
                    ax_gender.text(row['Session number'], row['response_seconds'] + 5, f'{row["response_seconds"]:.2f}',
                                  ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980B9')
            
            # Plot Female line
            if len(female_data) > 0:
                ax_gender.plot(female_data['Session number'], female_data['response_seconds'],
                              marker='s', linewidth=2.5, markersize=10, color='#C0392B',
                              markerfacecolor='#C0392B', markeredgewidth=2, markeredgecolor='white',
                              label='Female')
                
                # Annotations BELOW the line for Female
                for _, row in female_data.iterrows():
                    ax_gender.text(row['Session number'], row['response_seconds'] - 2, f'{row["response_seconds"]:.2f}',
                                  ha='center', va='top', fontsize=10, fontweight='bold', color='#C0392B')
            
            ax_gender.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.7, color='gray')
            ax_gender.set_axisbelow(True)
            ax_gender.spines['top'].set_visible(False)
            ax_gender.spines['right'].set_visible(False)
            ax_gender.spines['left'].set_linewidth(1.5)
            ax_gender.spines['bottom'].set_linewidth(1.5)
            
            # Dynamic y-axis based on both datasets
            all_values = []
            if len(male_data) > 0:
                all_values.extend(male_data['response_seconds'].values)
            if len(female_data) > 0:
                all_values.extend(female_data['response_seconds'].values)
            
            max_val = max(all_values) if all_values else 50
            ax_gender.set_ylim(0, max_val + 20)
            
            ax_gender.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax_gender.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax_gender.set_title('Response Time Trends - Male vs Female', fontsize=13, fontweight='bold', pad=20)
            ax_gender.legend(fontsize=11, loc='upper right', frameon=False)
            
            plt.tight_layout()
            st.pyplot(fig_gender)
    else:
        st.info("Gender or response time data not available for gender analysis.")
