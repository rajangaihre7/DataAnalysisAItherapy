import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def display(df, scale_map=None):
    """Display RQ9: Correlation Heatmap of All Questions"""
    st.subheader("RQ9: Correlation Analysis - all Questions (Q1-Q25)")
    st.markdown("**Correlation heatmap showing relationships between all survey questions**")
    
    # Define all question columns with short names (Q1 to Q25)
    questions_mapping = {
        "How engaged was the participant during today's storytelling session?": "Q1: Engagement",
        "How well did the participant understand that this story is personalised for them?": "Q2: Understanding",
        "Did the participant demonstrate emotional connection?": "Q3: Connection",
        "How would you rate the participant's verbal participation?": "Q4: Verbal",
        "Did the participant maintain attention throughout the session?": "Q5: Attention",
        "How likely a participant was to retell and use the stories in his/her daily conversation or interaction with\ntheir peers, carers, parents or therapists?": "Q6: Retelling",
        "Did the participant show any signs of enjoyment during the session?": "Q7: Enjoyment",
        "Did the participant exhibit distress, boredom, or frustration? ": "Q8: Distress",
        "Did the participant initiate interactions related to the story?": "Q9: Initiation",
        "Did the participant repeat the story or any content from the story?": "Q10: Repetition",
        "Did the participant want to or try to creatively change the story?": "Q11: Creativity",
        "How much did the relationship between a participant and carer/parent improved?": "Q13: RelationImprove",
        "Did the participant express their feelings about the story (verbally or otherwise)?": "Q14: Expression",
       # "What is the response time of the participant? (in minutes roughly)": "Q15: ResponseTime",
        "Did the participant apply the learning during the session?": "Q18: Apply learning",
        "Did the participant generalise the behaviour outside the story?": "Q19: Generalize",
        "Did the participant recall or refer to a previous story or theme?": "Q20: Recall",
        "Did the participant reflect on or comment about the theme after the story ended?": "Q21: Reflect",
        "Did the participant link the story to real-life experiences?": "Q22: LinkReal",
        "How much different scenarios stories impact overall social behaviour ?": "Q23: SocialImpact"
    }
    
    # Filter available questions that exist in the dataframe
    available_questions = [q for q in questions_mapping.keys() if q in df.columns]
    available_short_names = [questions_mapping[q] for q in available_questions]
    
    if len(available_questions) == 0:
        st.error("No questions found in the dataset")
        return
    
    # Convert all question columns to numeric to avoid errors
    df_numeric = df.copy()
    for question in available_questions:
        df_numeric[question] = pd.to_numeric(df_numeric[question], errors='coerce')
    
    st.info(f"Found {len(available_questions)} questions for correlation analysis")
    
    # --- CORRELATION HEATMAP ---
    st.markdown("---")
    st.markdown("#### Correlation Heatmap - All Questions")
    st.write("Shows correlation coefficients between all survey questions (values range from -1 to 1)")
    st.write("")
    
    # Create correlation matrix
    correlation_matrix = df_numeric[available_questions].corr()
    
    # Rename columns and index with short names
    correlation_matrix.columns = available_short_names
    correlation_matrix.index = available_short_names
    
    # Create large heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax, 
                linewidths=0.5, linecolor='gray', center=0, vmin=-1, vmax=1,
                square=True, annot_kws={'size': 9, 'weight': 'bold'},
                cbar=True)
    
    ax.set_title('Correlation Matrix - All Survey Questions (Q1-Q23)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add spacing
    st.write("")
    st.write("")
    
    # --- Strongest Correlations ---
    st.markdown("---")
    st.markdown("#### Strongest Positive and Negative Correlations")
    
    # Get upper triangle of correlation matrix to avoid duplicates
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_pairs.append({
                'Question 1': correlation_matrix.columns[i],
                'Question 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    
    # Top 10 positive correlations
    st.markdown("**Top 10 Strongest Positive Correlations:**")
    top_positive = corr_df.nlargest(10, 'Correlation')[['Question 1', 'Question 2', 'Correlation']]
    st.dataframe(top_positive, use_container_width=True, hide_index=True)
    
    st.write("")
    
    # Top 10 negative correlations
    st.markdown("**Top 10 Strongest Negative Correlations:**")
    top_negative = corr_df.nsmallest(10, 'Correlation')[['Question 1', 'Question 2', 'Correlation']]
    st.dataframe(top_negative, use_container_width=True, hide_index=True)
    
    # Add spacing
    st.write("")
    st.write("")
    
    # --- Summary Statistics ---
    st.markdown("---")
    st.markdown("#### Summary Statistics for All Questions")
    
    summary_stats = []
    for i, question in enumerate(available_questions):
        summary_stats.append({
            'Question': available_short_names[i],
            'Mean': f'{df_numeric[question].mean():.2f}',
            'Std Dev': f'{df_numeric[question].std():.2f}',
            'Min': f'{df_numeric[question].min():.2f}',
            'Max': f'{df_numeric[question].max():.2f}',
            'Count': len(df_numeric[question].dropna())
        })
    
    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

