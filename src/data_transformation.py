"""
Data Transformation Module
Handles all data cleaning and transformation from Bronze to Silver layer
"""
import pandas as pd
import os


def clean_submitted_by(df):
    """
    Map P/C to P for consistent parent perspective tracking
    
    Args:
        df: DataFrame with Submitted_by column
        
    Returns:
        DataFrame with cleaned Submitted_by values
    """
    df = df.copy()
    df['Submitted_by'] = df['Submitted_by'].str.upper().str.strip()
    df['Submitted_by'] = df['Submitted_by'].replace('P/C', 'P')
    return df


def clean_column_names(df):
    """
    Remove hidden spaces and standardize column names
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip()
    return df


def convert_numeric_columns(df):
    """
    Convert scale columns to numeric format
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with numeric conversions
    """
    df = df.copy()
    
    target_cols = [
        "How engaged was the participant during today's storytelling session?",
        "Did the participant demonstrate emotional connection?",
        "Did the participant exhibit distress, boredom, or frustration?",
        "Did the participant initiate interactions related to the story?",
        "Did the participant generalise the behaviour outside the story?",
        "Did the participant link the story to real-life experiences?",
        "How much different scenarios stories impact overall social behaviour ?",
        "How much did the relationship between a participant and carer/parent improved?",
        "Did the response time decrease from the last session?",
        "Did the response time increase from the last session?"
    ]
    
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_and_transform_silver_data(bronze_path):
    """
    Load Bronze data and apply all transformations
    
    Args:
        bronze_path: Path to Bronze CSV file
        
    Returns:
        Cleaned DataFrame ready for analysis
    """
    try:
        # Load Bronze data
        df = pd.read_csv(bronze_path, encoding='cp1252')
        
        # Apply transformations
        df = clean_column_names(df)
        df = clean_submitted_by(df)
        df = convert_numeric_columns(df)
        
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def save_to_silver(df, silver_path):
    """
    Save cleaned data to Silver layer
    
    Args:
        df: Cleaned DataFrame
        silver_path: Path to save Silver CSV file
        
    Returns:
        Success/error message
    """
    try:
        df.to_csv(silver_path, index=False, encoding='cp1252')
        return f"Data successfully saved to {silver_path}"
    except Exception as e:
        return f"Error saving data: {str(e)}"


def get_scale_map():
    """
    Return the standard 0-4 scale mapping
    
    Returns:
        Dictionary with scale mapping
    """
    return {
        0: 'Not at all',
        1: 'Slightly',
        2: 'Moderately',
        3: 'Very',
        4: 'Fully'
    }
