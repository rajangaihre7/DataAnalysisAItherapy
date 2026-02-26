import streamlit as st
import pandas as pd
import os
import importlib.util
import sys

# --- 1. PAGE CONFIG (Must be the very first Streamlit command) ---
st.set_page_config(page_title="AI Therapy Dashboard", layout="wide")

# --- 2. FAIL-SAFE DATA LOADING ---
@st.cache_data
def load_data():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "Data", "Bronze", "data_bronze_numeric_format_data.csv")
    
    # Check if file exists to prevent black screen crash
    if not os.path.exists(file_path):
        return None, f"Error: '{file_path}' not found."
    
    try:
        # Try loading with cp1252 (Windows) encoding which handles the 0x92 byte
        df = pd.read_csv(file_path, encoding='cp1252')
        
        # CLEANING: Remove hidden spaces from column names (e.g., 'Q8 ' -> 'Q8')
        df.columns = df.columns.str.strip()
        
        # Define our mapping for the 0-4 scale
        scale_map = {0: 'Not at all', 1: 'Slightly', 2: 'Moderately', 3: 'Very', 4: 'Fully'}
        
        # List of columns to convert to numbers
        target_cols = [
            "How engaged was the participant during today's storytelling session?",
            "Did the participant demonstrate emotional connection?",
            "Did the participant exhibit distress, boredom, or frustration?",
            "Did the participant initiate interactions related to the story?",
            "Did the participant generalise the behaviour outside the story?",
            "Did the participant link the story to real-life experiences?",
            "How much different scenarios stories impact overall social behaviour ?",
            "How much did the relationship between a participant and carer/parent improved?"
        ]
        
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df, scale_map
    except Exception as e:
        return None, f"Data Load Error: {str(e)}"

# --- 3. DYNAMIC MODULE LOADER ---
def load_module(module_name):
    """Dynamically load a module from the Module folder"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, "Module", f"{module_name}.py")
    
    if not os.path.exists(module_path):
        return None
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- 4. RUN LOAD ---
df, status = load_data()

# --- 5. THE UI ---
st.title("📊 Autism AI Therapy Analysis")

if df is None:
    st.error(status)
    st.info("Please ensure the CSV file is in the same folder as this app.py script.")
else:
    # Sidebar Navigation
    st.sidebar.header("Select Research Question")
    rq_options = {
        # "Welcome / Overview": None,
        "RQ1: Emotional Growth": "RQ1",
        "RQ2: Distress Reduction": "RQ2",
        "RQ3: Real-World Generalization": "RQ3",
        "RQ4: Family Relationship Impact": "RQ4",
        "RQ5: Self-Initiated Social Interaction": "RQ5",
        "RQ6: Social Behaviour Impact": "RQ9",
    }
    
    selected = st.sidebar.radio("Analysis View", list(rq_options.keys()))
    rq_module = rq_options[selected]

    if selected == "Welcome / Overview":
        st.subheader("Project Summary")
        st.write("Welcome to the AI Storytelling Research Dashboard.")
        st.write(f"**Total Records Analyzed:** {len(df)}")
        st.success("Data loaded successfully. Select a Research Question from the sidebar to begin.")
        
        # Quick summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Engagement", round(df["How engaged was the participant during today's storytelling session?"].mean(), 2))
        c2.metric("Min Distress", round(df["Did the participant exhibit distress, boredom, or frustration?"].min(), 2))
        c3.metric("Max Social Impact", round(df["How much different scenarios stories impact overall social behaviour ?"].max(), 2))
    
    else:
        # Dynamically load and display the selected RQ module
        module = load_module(rq_module)
        if module and hasattr(module, 'display'):
            module.display(df, status)
        else:
            st.error(f"Could not load module for {selected}")