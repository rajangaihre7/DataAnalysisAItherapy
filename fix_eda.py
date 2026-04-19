content = '''"""
EDA - Exploratory Data Analysis (main entry point)
====================================================
Organises the EDA into five top-level sections, each in its own file:

  EDA_Section1.py  - Section 1: Dataset Overview & Demographics  (complete)
  EDA_Section2.py  - Section 2: Missing Data Profiling           (complete)
  EDA_Section3.py  - Section 3: Descriptive Statistics           (complete)
  EDA_Section4.py  - Section 4: Longitudinal Trajectories        (complete)
  EDA_Section5.py  - Section 5: Observer Type Comparison         (complete)

This file is the top-level router only - it loads each section module
using importlib (same pattern as app.py) and delegates rendering to it.
"""

import os
import importlib.util
import streamlit as st


def _load_section(name):
    """Load a section module by filename (no extension) from the same directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"{name}.py")
    if not os.path.exists(path):
        return None
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def display(df, scale_map=None):
    """Main EDA entry point - renders five section tabs."""
    st.header("Exploratory Data Analysis (EDA)")

    sec1, sec2, sec3, sec4, sec5 = st.tabs([
        "1. Overview & Demographics",
        "2. Missing Data Profiling",
        "3. Descriptive Statistics",
        "4. Longitudinal Trajectories",
        "5. Observer Comparison",
    ])

    with sec1:
        m = _load_section("EDA_Section1")
        if m:
            m.display(df, scale_map)

    with sec2:
        m = _load_section("EDA_Section2")
        if m:
            m.display(df, scale_map)

    with sec3:
        m = _load_section("EDA_Section3")
        if m:
            m.display(df, scale_map)

    with sec4:
        m = _load_section("EDA_Section4")
        if m:
            m.display(df, scale_map)

    with sec5:
        m = _load_section("EDA_Section5")
        if m:
            m.display(df, scale_map)
'''

with open('src/Module/EDA.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('EDA.py rewritten successfully')
