import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_age_group(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if 8 <= age <= 14:
        return "8-14"
    elif 15 <= age <= 19:
        return "15-19"
    elif 20 <= age <= 26:
        return "20-26"
    elif 27 <= age <= 35:
        return "27-35"
    else:
        return "36+"

def display(df, scale_map=None):
    """Display Descriptive Analysis"""
    st.subheader("Descriptive Analysis of Participants")
    
    # Get unique participants only
    unique_df = df.drop_duplicates(subset=['Participant id']).copy()
    
    st.markdown("**Analysis based on participants: " + str(len(unique_df)) + " participants**")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Gender Distribution", "Age Distribution", "Autism Level"])
    
    # ============ TAB 1: GENDER DISTRIBUTION ============
    with tab1:
        st.markdown("### Gender Distribution")
        
        # Count gender
        gender_counts = unique_df['Gender'].value_counts()
        
        # Display statistics
        '''
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Participants", len(unique_df))
        with col2:
            st.metric("Male Participants", gender_counts.get('Male', 0))
        with col3:
            st.metric("Female Participants", gender_counts.get('Female', 0))
        '''    
        # Visualization 1: Pie Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')
        
        # Pie chart
        colors = ['#3498db', '#e74c3c']
        wedges, texts, autotexts = ax1.pie(gender_counts, 
                                             labels=gender_counts.index, 
                                             autopct='%1.1f%%',
                                             colors=colors,
                                             startangle=90,
                                             textprops={'fontsize': 11, 'weight': 'bold'})
        ax1.set_title('Gender Distribution of Participants', fontsize=13, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Bar chart
        gender_counts.plot(kind='bar', ax=ax2, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('Count of Participants by Gender', fontsize=13, fontweight='bold', pad=20)
        ax2.set_xlabel('Gender', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Participants', fontsize=11, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(gender_counts.items()):
            ax2.text(i, val + 0.5, str(val), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Table
        st.markdown("**Gender Count Table**")
        gender_table = pd.DataFrame({
            'Gender': gender_counts.index,
            'Count': gender_counts.values,
            'Percentage': (gender_counts.values / len(unique_df) * 100).round(2).astype(str) + '%'
        })
        st.dataframe(gender_table, use_container_width=True, hide_index=True)
    
    # ============ TAB 2: AGE DISTRIBUTION ============
    with tab2:
        st.markdown("### Age Distribution")
        
        # Create age groups
        unique_df['Age_Group'] = unique_df['Age'].apply(get_age_group)
        age_group_order = ['1-8', '9-14', '15-19', '20-26', '27-35', '36+', 'Unknown']
        
        # Count age groups
        age_counts = unique_df['Age_Group'].value_counts().reindex(age_group_order, fill_value=0)
        age_counts = age_counts[age_counts > 0]  # Remove empty groups
        '''
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Participants", len(unique_df))
        with col2:
            st.metric("Avg Age", f"{unique_df['Age'].mean():.1f}")
        with col3:
            st.metric("Min Age", int(unique_df['Age'].min()))
        with col4:
            st.metric("Max Age", int(unique_df['Age'].max()))
        '''
        # Visualization: Age Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')
        
        # Bar chart by age group
        colors_age = sns.color_palette("husl", len(age_counts))
        age_counts.plot(kind='bar', ax=ax1, color=colors_age, edgecolor='black', linewidth=1.5)
        ax1.set_title('Participants by Age Group', fontsize=13, fontweight='bold', pad=20)
        ax1.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Participants', fontsize=11, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, val in enumerate(age_counts.values):
            ax1.text(i, val + 0.3, str(val), ha='center', fontsize=10, fontweight='bold')
        
        # Age distribution by Gender across age groups
        gender_age_dist = unique_df.groupby(['Age_Group', 'Gender']).size().unstack(fill_value=0)
        gender_age_dist = gender_age_dist.reindex(age_group_order, fill_value=0)
        gender_age_dist = gender_age_dist.loc[gender_age_dist.index.isin(age_counts.index)]
        
        x = np.arange(len(gender_age_dist.index))
        width = 0.35
        
        if 'Male' in gender_age_dist.columns:
            ax2.bar(x - width/2, gender_age_dist['Male'], width, label='Male', color='#3498db', edgecolor='black', linewidth=1)
        if 'Female' in gender_age_dist.columns:
            ax2.bar(x + width/2, gender_age_dist['Female'], width, label='Female', color='#e74c3c', edgecolor='black', linewidth=1)
        
        ax2.set_title('Age Distribution by Gender', fontsize=13, fontweight='bold', pad=20)
        ax2.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Participants', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(gender_age_dist.index)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        
        # Add value labels on bars
        for i, (label, row) in enumerate(gender_age_dist.iterrows()):
            if 'Male' in gender_age_dist.columns and row['Male'] > 0:
                ax2.text(i - width/2, row['Male'] + 0.2, str(int(row['Male'])), ha='center', fontsize=9, fontweight='bold')
            if 'Female' in gender_age_dist.columns and row['Female'] > 0:
                ax2.text(i + width/2, row['Female'] + 0.2, str(int(row['Female'])), ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Table
        st.markdown("**Age Group Count Table**")
        age_table = pd.DataFrame({
            'Age Group': age_counts.index,
            'Count': age_counts.values,
            'Percentage': (age_counts.values / len(unique_df) * 100).round(2).astype(str) + '%'
        })
        st.dataframe(age_table, use_container_width=True, hide_index=True)
    
    # ============ TAB 3: AUTISM LEVEL ============
    with tab3:
        st.markdown("### Autism Severity Level Distribution")
        
        # Count autism levels
        autism_counts = unique_df['Autism Level'].value_counts().sort_index()
        
        '''
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Participants", len(unique_df))
        with col2:
            st.metric("Avg Autism Level", f"{unique_df['Autism Level'].mean():.2f}")
        with col3:
            st.metric("Most Common Level", str(autism_counts.idxmax()))
        '''
        # Visualization: Autism Level Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')
        
        # Bar chart
        colors_autism = sns.color_palette("RdYlGn_r", len(autism_counts))
        autism_counts.plot(kind='bar', ax=ax1, color=colors_autism, edgecolor='black', linewidth=1.5)
        ax1.set_title('Autism Severity Level Distribution', fontsize=13, fontweight='bold', pad=20)
        ax1.set_xlabel('Autism Level', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Participants', fontsize=11, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, val in enumerate(autism_counts.values):
            ax1.text(i, val + 0.3, str(val), ha='center', fontsize=10, fontweight='bold')
        
        # Pie chart
        colors_pie = sns.color_palette("RdYlGn_r", len(autism_counts))
        wedges, texts, autotexts = ax2.pie(autism_counts, 
                                             labels=[f'Level {int(x)}' for x in autism_counts.index], 
                                             autopct='%1.1f%%',
                                             colors=colors_pie,
                                             startangle=90,
                                             textprops={'fontsize': 10, 'weight': 'bold'})
        ax2.set_title('Autism Level Proportion', fontsize=13, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Table
        st.markdown("**Autism Level Count Table**")
        autism_table = pd.DataFrame({
            'Autism Level': autism_counts.index,
            'Count': autism_counts.values,
            'Percentage': (autism_counts.values / len(unique_df) * 100).round(2).astype(str) + '%'
        })
        st.dataframe(autism_table, use_container_width=True, hide_index=True)
    
    # ============ SUMMARY SECTION ============
    st.divider()
    st.markdown("### Summary Statistics")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown("**Gender Breakdown**")
        for gender, count in gender_counts.items():
            pct = (count / len(unique_df) * 100)
            st.text(f"{gender}: {count} ({pct:.1f}%)")
    
    with summary_col2:
        st.markdown("**Age Statistics**")
        st.text(f"Mean: {unique_df['Age'].mean():.2f}")
        st.text(f"Median: {unique_df['Age'].median():.2f}")
        st.text(f"Std Dev: {unique_df['Age'].std():.2f}")
        st.text(f"Range: {unique_df['Age'].min():.0f}-{unique_df['Age'].max():.0f}")
    
    with summary_col3:
        st.markdown("**Autism Level Stats**")
        st.text(f"Mean: {unique_df['Autism Level'].mean():.2f}")
        st.text(f"Median: {unique_df['Autism Level'].median():.2f}")
        st.text(f"Mode: {unique_df['Autism Level'].mode().values[0]}")
        st.text(f"Range: {unique_df['Autism Level'].min():.0f}-{unique_df['Autism Level'].max():.0f}")
