"""
Silver to Gold Transformation Module
=====================================
Handles all transformations from the Silver cleaned dataset to the
Gold analytical layer:

  Step 1  Reverse-code Q8  (distress → positive polarity, consistent with all other items)
  Step 2  Min-Max normalise all 21 quantitative items to 0.0 – 1.0
  Step 3  Compute composite Engagement Score (row-mean of 21 normalised items)
  Step 4  Save Gold dataset  →  Data/Gold/data_gold_analytical.csv
"""

import os
import sys

import pandas as pd
import numpy as np

# ── Ensure questionnaire_mapping is importable ────────────────────────────────
_src_dir    = os.path.dirname(os.path.abspath(__file__))
_module_dir = os.path.join(_src_dir, 'Module')
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

from questionnaire_mapping import QUESTION_MAPPING  # noqa: E402

# ── Item lists ─────────────────────────────────────────────────────────────────
# Q8 is reverse-coded (Q8_R) before normalising; all other items taken directly.
SCALE_4_ITEMS  = [
    'Q1', 'Q2', 'Q3', 'Q5', 'Q6', 'Q7', 'Q8_R',
    'Q9', 'Q10', 'Q11', 'Q13', 'Q14',
    'Q18', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25',
]  # 19 items on 0-4 scale (Q8_R replaces raw Q8)

SCALE_10_ITEMS = ['Q4', 'Q26']  # 2 items on 0-10 scale

NORM_ITEMS = [f'{q}_norm' for q in SCALE_4_ITEMS + SCALE_10_ITEMS]  # 21 normalised columns

# ── Display labels for normalised columns ──────────────────────────────────────
ITEM_LABELS = {
    'Q1':   'Engagement',
    'Q2':   'Understanding',
    'Q3':   'Connection',
    'Q4':   'Verbal',
    'Q5':   'Attention',
    'Q6':   'Retelling',
    'Q7':   'Enjoyment',
    'Q8_R': 'Distress(rev)',
    'Q9':   'Initiation',
    'Q10':  'Repetition',
    'Q11':  'Creativity',
    'Q13':  'RelationImprove',
    'Q14':  'Expression',
    'Q18':  'StoryTheme',
    'Q20':  'ApplyLearning',
    'Q21':  'Confidence',
    'Q22':  'Generalise',
    'Q23':  'Recall',
    'Q24':  'Reflect',
    'Q25':  'LinkReal',
    'Q26':  'SocialImpact',
}


def norm_label(norm_col: str) -> str:
    """Convert a normalised column name to a readable label.

    Example: 'Q8_R_norm' → 'Q8_R: Distress(rev)'
    """
    q = norm_col.replace('_norm', '')
    return f"{q}: {ITEM_LABELS.get(q, q)}"


def get_gold_path() -> str:
    """Return the default output path for the Gold CSV."""
    base = os.path.normpath(os.path.join(_src_dir, '..'))
    return os.path.join(base, 'Data', 'Gold', 'data_gold_analytical.csv')


# ══════════════════════════════════════════════════════════════════════════════
# CORE TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════════

def build_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a Silver-layer DataFrame into the Gold analytical DataFrame.

    Steps
    -----
    1. Reverse-code Q8 → Q8_R  (4 − raw, so 4 = no distress = good)
    2. Min-Max normalise:
       - 0-4 items (incl. Q8_R) divided by 4.0
       - 0-10 items (Q4, Q26)   divided by 10.0
    3. Compute Engagement_Score = row-mean of all 21 normalised items
       - Items_Available : how many of the 21 items are non-null per row
       - Low_Item_Count  : True when fewer than 17 items are available

    Parameters
    ----------
    df : pd.DataFrame
        Silver cleaned dataset (column names are the original survey question text).

    Returns
    -------
    pd.DataFrame
        Gold dataset — original columns plus all derived columns.
    """
    gold = df.copy()

    # Build Q-label → raw column-name lookup
    q_to_col = {q: col for col, q in QUESTION_MAPPING.items() if col in gold.columns}

    # ── Step 1: Reverse-code Q8 ───────────────────────────────────────────────
    q8_col = q_to_col.get('Q8')
    if q8_col:
        gold['Q8_R'] = 4 - pd.to_numeric(gold[q8_col], errors='coerce')
    else:
        gold['Q8_R'] = np.nan

    # ── Step 2: Normalise ─────────────────────────────────────────────────────
    for q in SCALE_4_ITEMS:
        if q == 'Q8_R':
            gold['Q8_R_norm'] = gold['Q8_R'] / 4.0
        else:
            col = q_to_col.get(q)
            gold[f'{q}_norm'] = (
                pd.to_numeric(gold[col], errors='coerce') / 4.0
                if col else np.nan
            )

    for q in SCALE_10_ITEMS:
        col = q_to_col.get(q)
        gold[f'{q}_norm'] = (
            pd.to_numeric(gold[col], errors='coerce') / 10.0
            if col else np.nan
        )

    # ── Step 3: Composite Engagement Score ───────────────────────────────────
    present = [c for c in NORM_ITEMS if c in gold.columns]
    gold['Engagement_Score'] = gold[present].mean(axis=1)
    gold['Items_Available']  = gold[present].notna().sum(axis=1)
    gold['Low_Item_Count']   = gold['Items_Available'] < 17

    return gold


def save_gold_df(gold: pd.DataFrame, path: str = None) -> str:
    """
    Save the Gold DataFrame to CSV.

    Parameters
    ----------
    gold : pd.DataFrame
        Gold analytical dataset produced by build_gold_df().
    path : str, optional
        Output file path.  Defaults to Data/Gold/data_gold_analytical.csv.

    Returns
    -------
    str
        Success or error message.
    """
    if path is None:
        path = get_gold_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gold.to_csv(path, index=False)
        return f"Saved → {path}  (Rows: {len(gold):,}   Columns: {len(gold.columns)})"
    except Exception as e:
        return f"Error saving Gold dataset: {str(e)}"


def run_silver_to_gold_pipeline(silver_path: str = None, gold_path_out: str = None):
    """
    Full Silver → Gold pipeline:
    Loads the Silver cleaned CSV, applies all transformations, and saves
    the Gold analytical CSV.

    Parameters
    ----------
    silver_path : str, optional
        Path to Silver CSV.  Defaults to Data/Silver/data_silver_cleaned.csv.
    gold_path_out : str, optional
        Output path for Gold CSV.  Defaults to Data/Gold/data_gold_analytical.csv.

    Returns
    -------
    tuple[pd.DataFrame | None, str]
        (gold_df, status_message)
    """
    if silver_path is None:
        base = os.path.normpath(os.path.join(_src_dir, '..'))
        silver_path = os.path.join(base, 'Data', 'Silver', 'data_silver_cleaned.csv')

    try:
        df = pd.read_csv(silver_path)
    except Exception as e:
        return None, f"Error loading Silver data: {str(e)}"

    gold = build_gold_df(df)
    msg  = save_gold_df(gold, gold_path_out)
    return gold, msg
