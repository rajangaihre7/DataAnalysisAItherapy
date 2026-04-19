"""
Data Transformation Module
Handles all data cleaning and transformation from Bronze to Silver layer
"""
import pandas as pd
import os

# ── Scale mappings ──────────────────────────────────────────────────────────
# Standard scale: Not at all=0, Slightly=1, Moderately=2, Very=3, Fully=4
# Used for: engagement, emotional connection, understanding, attention,
#           retelling likelihood, enjoyment, relationship improved
_STANDARD_MAP = {
    'Not at all': 0, 'Nor at all': 0,
    'Slightly': 1,   'Rarely': 1,
    'Moderately': 2, 'Modereately': 2,
    'Very': 3,       'Vey': 3,
    'Fully': 4,
    'Occasionally': 2, 'Often': 3, 'Very Frequently': 4,
}

# Frequency scale: Not at all=0, Rarely=1, Occasionally=2, Often=3, Very Frequently=4
# Used for: distress, initiation, repetition, creativity, expressing feelings
_FREQUENCY_MAP = {
    'Not at all': 0,   ' Not at all': 0,
    'Rarely': 1,       'Slightly': 1,
    'Occasionally': 2, 'Occassionally': 2, 'Occasssionally': 2,
    'Occassionnally': 2, 'Ocassinally': 2,  'Ocassionally': 2,
    'Occasinally': 2,  'occassionally': 2,  'Moderately': 2,
    'Often': 3,
    'Very Frequently': 4, 'Very frequently': 4,  'very frequently': 4,
    'Very  Frequently': 4, 'Very  frequently': 4,
    'Very': 3,
    '0': 0,
}

# Behaviour scale: No=0, Slightly=1, Sometimes=2, Often=3, Very Often=4
# Used for: generalise, link real-life, recall, reflect, apply learning,
#           understand theme, confidence
_BEHAVIOUR_MAP = {
    'No': 0,         'no': 0,
    'Slightly': 1,
    'Sometimes': 2,  'Somtimes': 2,   'Sometiems': 2, 'sometimes': 2,
    'Often': 3,      'often': 3,       ' Often': 3,    'Oftne': 3,
    'Very Often': 4, 'Very often': 4,  'Vey often': 4,
    'Not at all': 0, 'Rarely': 1, 'Occasionally': 2, 'Very Frequently': 4,
}

# Yes/No scale: No=0, Yes=1
_YESNO_MAP = {
    'No': 0, 'no': 0,
    'Yes': 1, 'yes': 1,
}

# Strings treated as missing / not applicable
_NULL_STRINGS = {'NULL', 'Null', 'null', 'No data', 'no data', 'No Data',
                 'nan', '', 'Attentive listening', 'Verbal Communication'}


def replace_null_strings(df):
    """
    Strip whitespace from all text cells and replace known NULL-like strings
    with NaN so they are treated as missing values in analysis.
    """
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(list(_NULL_STRINGS), pd.NA)
    return df


def clean_submitted_by(df):
    """
    Standardise Submitted_by values:
      - uppercase + strip whitespace
      - P/C  → P   (parent/carer joint entries counted as parent)
      - p    → P   (lowercase typo)
    """
    df = df.copy()
    df['Submitted_by'] = df['Submitted_by'].astype(str).str.upper().str.strip()
    df['Submitted_by'] = df['Submitted_by'].replace({'P/C': 'P', 'P': 'P'})
    return df


def clean_column_names(df):
    """
    Remove hidden leading/trailing spaces from column names.
    """
    df.columns = df.columns.str.strip()
    return df


def map_scale_columns_to_numeric(df):
    """
    Map all text-scale survey columns to numeric values using predefined
    dictionaries. Typos and case variants are corrected inline via the maps.

    Scale groups
    ─────────────────────────────────────────────────────────
    Standard  (0-4): Not at all / Slightly / Moderately / Very / Fully
    Frequency (0-4): Not at all / Rarely / Occasionally / Often / Very Frequently
    Behaviour (0-4): No / Slightly / Sometimes / Often / Very Often
    Yes/No    (0-1): No / Yes
    Numeric        : already numeric in Bronze – just cast to float
    """
    df = df.copy()

    def _apply(series, mapping):
        return (series.astype(str).str.strip()
                      .map(lambda v: mapping.get(v, pd.NA)))

    # ── Standard scale ──────────────────────────────────────
    standard_cols = [
        "How engaged was the participant during today's storytelling session?",
        "How well did the participant understand that this story is personalised for them?",
        "Did the participant demonstrate emotional connection?",
        "Did the participant maintain attention throughout the session?",
        ("How likely a participant was to retell and use the stories in his/her daily "
         "conversation or interaction with\ntheir peers, carers, parents or therapists?"),
        "Did the participant show any signs of enjoyment during the session?",
        "How much did the relationship between a participant and carer/parent improved?",
    ]
    for col in standard_cols:
        if col in df.columns:
            df[col] = _apply(df[col], _STANDARD_MAP)

    # ── Frequency scale ─────────────────────────────────────
    frequency_cols = [
        "Did the participant exhibit distress, boredom, or frustration?",
        "Did the participant initiate interactions related to the story?",
        "Did the participant repeat the story or any content from the story?",
        "Did the participant want to or try to creatively change the story?",
        "Did the participant express their feelings about the story (verbally or otherwise)?",
    ]
    for col in frequency_cols:
        if col in df.columns:
            df[col] = _apply(df[col], _FREQUENCY_MAP)

    # ── Behaviour scale ─────────────────────────────────────
    behaviour_cols = [
        "To what extent did the participant understand the story theme?",
        "Did the participant apply the learning during the session?",
        ("Do you feel that the participant is confident and has the potential to be able "
         "to apply this story in the real world after a session?"),
        "Did the participant generalise the behaviour outside the story?",
        "Did the participant recall or refer to a previous story or theme?",
        "Did the participant reflect on or comment about the theme after the story ended?",
        "Did the participant link the story to real-life experiences?",
    ]
    for col in behaviour_cols:
        if col in df.columns:
            df[col] = _apply(df[col], _BEHAVIOUR_MAP)

    # ── Yes / No scale ───────────────────────────────────────
    yesno_cols = [
        "Did the response time decrease from the last session?",
        "Did the response time increase from the last session?",
    ]
    for col in yesno_cols:
        if col in df.columns:
            df[col] = _apply(df[col], _YESNO_MAP)

    # ── Already numeric – cast only ─────────────────────────
    numeric_cols = [
        "How would you rate the participant\u2019s verbal participation?",
        "How much different scenarios stories impact overall social behaviour ?",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_and_transform_silver_data(bronze_path):
    """
    Load Bronze data and apply all transformations for the Silver layer:
      1. Strip column-name whitespace
      2. Replace NULL strings / No data → NaN
      3. Standardise Submitted_by (P/C, p → P)
      4. Map all text scale columns to numeric (0-4 / 0-1 / 0-10)
    """
    try:
        df = pd.read_csv(bronze_path, encoding='cp1252')

        df = clean_column_names(df)
        df = replace_null_strings(df)
        df = clean_submitted_by(df)
        df = map_scale_columns_to_numeric(df)

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


def run_bronze_to_silver_pipeline(bronze_path=None, silver_path=None):
    """
    Full Bronze to Silver pipeline:
    Loads Bronze raw data, applies all transformations, and saves to Silver layer.

    Args:
        bronze_path: Path to Bronze CSV. Defaults to Data/Bronze/data_bronze_numeric_format_data.csv
        silver_path: Path to save Silver CSV. Defaults to Data/Silver/data_silver_cleaned.csv

    Returns:
        Tuple of (DataFrame or None, status message)
    """
    # Resolve default paths relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if bronze_path is None:
        bronze_path = os.path.join(base_dir, "Data", "Bronze", "data_bronze_raw_data.csv")

    if silver_path is None:
        silver_path = os.path.join(base_dir, "Data", "Silver", "data_silver_cleaned.csv")

    # Step 1: Validate Bronze file exists
    if not os.path.exists(bronze_path):
        return None, f"Error: Bronze file not found at '{bronze_path}'"

    print(f"[Pipeline] Loading Bronze data from: {bronze_path}")

    # Step 2: Load and transform
    df, error_msg = load_and_transform_silver_data(bronze_path)
    if error_msg:
        return None, error_msg

    print(f"[Pipeline] Transformations applied. Rows: {len(df)}, Columns: {len(df.columns)}")

    # Step 3: Ensure Silver directory exists
    os.makedirs(os.path.dirname(silver_path), exist_ok=True)

    # Step 4: Save to Silver
    status = save_to_silver(df, silver_path)
    print(f"[Pipeline] {status}")

    return df, status


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


if __name__ == "__main__":
    df, status = run_bronze_to_silver_pipeline()
    if df is not None:
        print(f"\nSilver layer generated successfully.")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    else:
        print(f"\nPipeline failed: {status}")
