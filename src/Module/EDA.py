"""
EDA - Exploratory Data Analysis (main entry point)
====================================================
Organises the EDA into five top-level sections, each in its own file:

  EDA_Section1.py  - Section 1: Dataset Overview & Demographics  (complete)
  EDA_Section2.py  - Section 2: Missing Data Profiling           (complete)
  EDA_Section3.py  - Section 3: Descriptive Statistics           (complete)
  EDA_Section4.py  - Section 4: Longitudinal Trajectories        (complete)
  EDA_Section5.py  - Section 5: Observer Type Comparison         (complete)
  EDA_Section6.py  - Section 6: Inter-Rater Reliability (ICC)    (complete)
  EDA_Section7.py  - Section 7: Correlation Analysis              (complete)

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

    sec1, sec2, sec3, sec4, sec5, sec6, sec7 = st.tabs([
        "1. Overview & Demographics",
        "2. Missing Data Profiling",
        "3. Descriptive Statistics",
        "4. Longitudinal Trajectories",
        "5. Observer Comparison",
        "6. Inter-Rater Reliability",
        "7. Correlation Analysis",
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

    with sec6:
        m = _load_section("EDA_Section6")
        if m:
            m.display(df, scale_map)

    with sec7:
        m = _load_section("EDA_Section7")
        if m:
            m.display(df, scale_map)
