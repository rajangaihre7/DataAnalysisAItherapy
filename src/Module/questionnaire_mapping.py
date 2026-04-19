"""
Questionnaire Column Mapping
============================
Maps full original question text to Q1-Q26 labels.
Provides utilities for accessing questionnaire columns, notes, and metadata across all sections.
"""

import pandas as pd
import os


# ══════════════════════════════════════════════════════════════════════════════
# FULL QUESTION TEXT → Q LABEL MAPPING
# These are the EXACT column names in data_silver_cleaned.csv
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_MAPPING = {
    # Full question text → Q label
    "How engaged was the participant during today's storytelling session?": "Q1",
    "How well did the participant understand that this story is personalised for them?": "Q2",
    "Did the participant demonstrate emotional connection?": "Q3",
    "How would you rate the participant’s verbal participation?": "Q4",
    "Did the participant maintain attention throughout the session?": "Q5",
    "How likely a participant was to retell and use the stories in his/her daily conversation or interaction with\ntheir peers, carers, parents or therapists?": "Q6",
    "Did the participant show any signs of enjoyment during the session?": "Q7",
    "Did the participant exhibit distress, boredom, or frustration?": "Q8",
    "Did the participant initiate interactions related to the story?": "Q9",
    "Did the participant repeat the story or any content from the story?": "Q10",
    "Did the participant want to or try to creatively change the story?": "Q11",
    "How did retelling the stories at home impact the carer/participant relationship?": "Q12",
    "How much did the relationship between a participant and carer/parent improved?": "Q13",
    "Did the participant express their feelings about the story (verbally or otherwise)?": "Q14",
    "What is the response time of the participant? (in minutes roughly)": "Q15",
    "Did the response time decrease from the last session?": "Q16",
    "Did the response time increase from the last session?": "Q17",
    "To what extent did the participant understand the story theme?": "Q18",
    "Did the participant apply the learning during the session?": "Q20",
    "Do you feel that the participant is confident and has the potential to be able to apply this story in the real world after a session?": "Q21",
    "Did the participant generalise the behaviour outside the story?": "Q22",
    "Did the participant recall or refer to a previous story or theme?": "Q23",
    "Did the participant reflect on or comment about the theme after the story ended?": "Q24",
    "Did the participant link the story to real-life experiences?": "Q25",
    "How much different scenarios stories impact overall social behaviour ?": "Q26",
}

# Reverse mapping: Q label → full question text
REVERSE_QUESTION_MAPPING = {v: k for k, v in QUESTION_MAPPING.items()}

# Metadata column names (these ARE already renamed in Silver CSV)
METADATA_COLUMNS = {
    "session": "Session number",
    "observer": "Submitted_by",
    "participant_id": "Participant id",
    "date": "Session date",
    "theme": "Theme of Today's Story",
}

# Notes/comments columns
NOTES_COLUMNS = {
    "additional_notes": "Additional_notes_observations",
}

# Q column index mapping (positions in the Silver CSV) — used for data access
Q_IDX = {
    16:'Q1', 18:'Q2', 20:'Q3', 22:'Q4', 24:'Q5', 26:'Q6',
    28:'Q7', 30:'Q8', 32:'Q9', 34:'Q10', 36:'Q11',
    40:'Q13', 42:'Q14', 51:'Q18', 53:'Q20', 55:'Q21',
    57:'Q22', 59:'Q23', 61:'Q24', 63:'Q25', 65:'Q26'
}

# Items on 0-10 scale (rest are 0-4)
SCALE_10_ITEMS = ['Q4', 'Q26']

# All quantitative items for analysis (21 items)
QUANT_ITEMS = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11',
               'Q13','Q14','Q18','Q20','Q21','Q22','Q23','Q24','Q25','Q26']

# 6 key items for detailed trajectory plots
KEY_ITEMS = ['Q1', 'Q3', 'Q5', 'Q7', 'Q9', 'Q22']

# Non-quantitative items (excluded from numeric analysis)
NON_QUANT_ITEMS = {
    'Q12': 'Categorical: Improved/Stayed same/Not Improved',
    'Q15': 'Free-text response time',
    'Q16': 'Binary: Yes/No (response time decreased)',
    'Q17': 'Binary: Yes/No (response time increased)',
    'Q19': 'Not used in analysis'
}

# 21 item descriptions
ITEM_DESC = {
    'Q1': 'Engagement during storytelling',
    'Q2': 'Understanding personalised story',
    'Q3': 'Emotional connection',
    'Q4': 'Verbal participation',
    'Q5': 'Maintained attention',
    'Q6': 'Likelihood of retelling',
    'Q7': 'Signs of enjoyment',
    'Q8': 'Distress/boredom/frustration',
    'Q9': 'Initiated story interactions',
    'Q10': 'Repeated story content',
    'Q11': 'Creative changes to story',
    'Q12': 'Retelling a story at home impact',
    'Q13': 'Relationship improvement',
    'Q14': 'Expressed feelings about story',
    'Q15': 'Response time',
    'Q16': 'Binary: Yes/No (response time decreased)',
    'Q17': 'Binary: Yes/No (response time increased)',
    'Q18': 'Understanding story theme',
    'Q20': 'Applied learning during session',
    'Q21': 'Confidence to apply in real world',
    'Q22': 'Generalised behaviour outside story',
    'Q23': 'Recalled previous story/theme',
    'Q24': 'Reflected on theme after session',
    'Q25': 'Linked story to real-life',
    'Q26': 'Social behaviour impact'
}

# ══════════════════════════════════════════════════════════════════════════════
# ACCESSOR FUNCTIONS FOR QUESTION TEXT & LABELS
# ══════════════════════════════════════════════════════════════════════════════

def get_full_question_text(q_label):
    """Get full question text for a Q label."""
    return REVERSE_QUESTION_MAPPING.get(q_label, q_label)


def get_q_label(full_text):
    """Get Q label for full question text."""
    return QUESTION_MAPPING.get(full_text, None)


def get_available_question_columns(df):
    """Get all question columns from df (mapped to Q labels) that exist in the dataframe."""
    available = {}
    for full_text, q_label in QUESTION_MAPPING.items():
        if full_text in df.columns:
            available[full_text] = q_label
    return available


def get_available_q_labels(df):
    """Get list of Q labels for questions that exist in df."""
    return [QUESTION_MAPPING[q] for q in QUESTION_MAPPING if q in df.columns]


# ══════════════════════════════════════════════════════════════════════════════
# ACCESSOR FUNCTIONS FOR ITEM METADATA
# ══════════════════════════════════════════════════════════════════════════════

# Reverse-coded items (where lower score = poor outcome)
REVERSE_CODED_ITEMS = ['Q8']  # Q8: Distress/boredom/frustration (REVERSE)

def get_item_description(q_item):
    """Get human-readable description for a Q item."""
    return ITEM_DESC.get(q_item, q_item)


def is_scale_10(q_item):
    """Check if item uses 0-10 scale (vs 0-4)."""
    return q_item in SCALE_10_ITEMS


def is_key_item(q_item):
    """Check if item is in the 6 key items for trajectory analysis."""
    return q_item in KEY_ITEMS


def is_reverse_coded(q_item):
    """Check if item needs reverse coding (lower = worse)."""
    return q_item in REVERSE_CODED_ITEMS


def is_quantitative(q_item):
    """Check if item is one of the 21 quantitative items for analysis."""
    return q_item in QUANT_ITEMS


def get_all_descriptions():
    """Get full mapping of all items and descriptions."""
    return ITEM_DESC.copy()


def get_key_items_descriptions():
    """Get descriptions for only the 6 key trajectory items."""
    return {item: ITEM_DESC[item] for item in KEY_ITEMS if item in ITEM_DESC}


# ══════════════════════════════════════════════════════════════════════════════
# ACCESSOR FUNCTIONS FOR COMMENT COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def get_all_comment_columns():
    """Get all per-item comment column names."""
    return [f"Comment_{item}" for item in QUANT_ITEMS]


# ══════════════════════════════════════════════════════════════════════════════
# ACCESSOR FUNCTIONS FOR METADATA COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def get_session_column():
    """Get session column name."""
    return METADATA_COLUMNS["session"]


def get_observer_column():
    """Get observer/submitted_by column name."""
    return METADATA_COLUMNS["observer"]


def get_participant_id_column():
    """Get participant ID column name."""
    return METADATA_COLUMNS["participant_id"]


def get_date_column():
    """Get session date column name."""
    return METADATA_COLUMNS["date"]


def get_theme_column():
    """Get theme column name."""
    return METADATA_COLUMNS["theme"]


def get_additional_notes_column():
    """Get additional notes column name."""
    return NOTES_COLUMNS["additional_notes"]


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ACCESSOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_all_items_for_analysis(df):
    """
    Get all columns needed for analysis (Q items + notes).
    Returns tuple: (full_question_columns, q_labels, metadata_dict, notes_columns)
    """
    # Get available question columns (full text)
    available_questions = [q for q in QUESTION_MAPPING.keys() if q in df.columns]
    q_labels = [QUESTION_MAPPING[q] for q in available_questions]
    
    # Metadata
    metadata = {
        "session": get_session_column(),
        "observer": get_observer_column(),
        "participant_id": get_participant_id_column(),
        "date": get_date_column(),
        "theme": get_theme_column(),
    }
    
    # Notes columns
    notes = {
        
        "additional_notes": get_additional_notes_column() if get_additional_notes_column() in df.columns else None,
    }
    
    return available_questions, q_labels, metadata, notes
