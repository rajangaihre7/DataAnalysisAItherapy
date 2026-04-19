import streamlit as st
import pandas as pd
import os
import importlib.util
import sys
import time
from data_transformation import get_scale_map

# --- 1. PAGE CONFIG (Must be the very first Streamlit command) ---
st.set_page_config(page_title="AI Therapy Dashboard", layout="wide")

# --- 2. FAIL-SAFE DATA LOADING WITH AUTO-CACHE INVALIDATION ---
@st.cache_data
def load_data_cached(file_mod_time):
    """Cached data loading that invalidates when file changes"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    silver_path = os.path.join(script_dir, "..", "Data", "Silver", "data_silver_cleaned.csv")
    
    # Check if file exists to prevent black screen crash
    if not os.path.exists(silver_path):
        return None, f"Error: '{silver_path}' not found."
    
    try:
        # Load Silver data directly (already cleaned)
        df = pd.read_csv(silver_path, encoding='cp1252')
        df.columns = df.columns.str.strip()
        
        # Get scale mapping
        scale_map = get_scale_map()
            
        return df, scale_map
    except Exception as e:
        return None, f"Data Load Error: {str(e)}"

def load_data():
    """Wrapper function that checks file modification time"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    silver_path = os.path.join(script_dir, "..", "Data", "Silver", "data_silver_cleaned.csv")
    
    # Get file modification time to invalidate cache when file changes
    if os.path.exists(silver_path):
        file_mod_time = os.path.getmtime(silver_path)
    else:
        file_mod_time = 0
    
    return load_data_cached(file_mod_time)

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
        "Exploratory Data Analysis (EDA)": "EDA",
        "Descriptive Analysis": "Descriptive_Analysis",
        "RQ1: Emotional Growth": "RQ1",
        "RQ2: Distress Reduction": "RQ2",
        "RQ3: Real-World Generalization": "RQ3",
        "RQ4: Family Relationship Impact": "RQ4",
        "RQ5: Self-Initiated Social Interaction": "RQ5",
        "RQ6: Response Time Improvement": "RQ6",
        "RQ7: Personalization & Verbal Participation": "RQ7",
        "RQ8: Social Behaviour Impact": "RQ8",
        "RQ9: Comprehensive Heatmap Analysis": "RQ9",
        "NLP Prerequisites: Gold Dataset": "NLP_prerquesties",
        "NLP Part B: Sentiment Analysis": "NLP_SentimentAnalysis",
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