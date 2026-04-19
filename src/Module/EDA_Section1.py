"""
EDA — Section 1: Dataset Overview and Demographics
===================================================
Sub-sections
  1.1  Dataset Structure
  1.2  Demographics Table
  1.3  Cross-tabulations
  1.4  Visualisations (6 charts, each in a 1×2 subplot grid)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


# ── Tufte Style Helpers ───────────────────────────────────────────────────────

# Muted, purposeful palette — avoids saturated colors
TUFTE_PALETTE = [
    "#4878CF",   # steel blue   (primary)
    "#D65F5F",   # muted red    (contrast / flag)
    "#6ACC65",   # sage green
    "#B47CC7",   # muted purple
    "#C4AD66",   # warm tan
    "#77BEDB",   # sky blue
    "#4a4a4a",   # dark grey
]
T_BLUE   = TUFTE_PALETTE[0]
T_RED    = TUFTE_PALETTE[1]
T_GREEN  = TUFTE_PALETTE[2]
T_PURPLE = TUFTE_PALETTE[3]
T_TAN    = TUFTE_PALETTE[4]


def _tufte_ax(ax, keep_left=True, keep_bottom=True):
    """
    Apply Tufte's data-ink principles to a Matplotlib Axes:
      • Remove top, right (and optionally left/bottom) spines
      • Lighten remaining axis lines to near-invisible grey
      • Remove tick marks (data labels carry the information)
      • No gridlines — direct annotations serve as guides
      • White figure / axes background
    """
    for spine in ax.spines.values():
        spine.set_visible(False)

    if keep_bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("#cccccc")

    if keep_left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["left"].set_color("#cccccc")

    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.set_tick_params(labelsize=9, labelcolor="#555555")
    ax.xaxis.set_tick_params(labelsize=10, labelcolor="#333333")
    ax.set_facecolor("white")


def _label_bars(ax, bars, fmt="{}", pad_frac=0.015, fontsize=11, color="#333333"):
    """Place a value label above each bar using data coordinates."""
    ylim_top = ax.get_ylim()[1]
    pad = ylim_top * pad_frac
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad,
                fmt.format(int(h)),
                ha="center", va="bottom",
                fontsize=fontsize, color=color,
                fontweight="normal",
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _age_group(age):
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if 8 <= age <= 14:
        return "8-14"
    elif 15 <= age <= 19:
        return "15-19"
    elif 20 <= age <= 26:
        return "20-26"
    return "Other"


def _load_silver(df):
    """Return (full_df, participants_df).
    Tries to load silver CSV; falls back to the df passed from app.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    silver_path = os.path.join(script_dir, "..", "..", "Data", "Silver", "data_silver_cleaned.csv")
    if os.path.exists(silver_path):
        try:
            full = pd.read_csv(silver_path, encoding="cp1252")
            full.columns = full.columns.str.strip()
        except Exception:
            full = df.copy()
    else:
        full = df.copy()

    full.columns = full.columns.str.strip()
    participants = full.drop_duplicates(subset=["Participant id"]).copy()
    participants["Age_Group"] = participants["Age"].apply(_age_group)
    return full, participants


# ── Dataset Structure ─────────────────────────────────────────────────────────

def _show_dataset_structure(full, participants):
    """Display dataset overview metrics and table."""
    st.markdown("### Dataset Structure")

    total_records   = len(full)
    n_participants  = participants["Participant id"].nunique()
    sessions        = sorted(full["Session number"].dropna().unique().astype(int))
    n_sessions      = len(sessions)
    sess_range      = f"{min(sessions)}–{max(sessions)}"

    submitted       = full["Submitted_by"].str.upper().str.strip()
    t_count         = int((submitted == "T").sum())
    p_count         = int((submitted == "P").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",  total_records)
    col2.metric("Participants",   n_participants)
    col3.metric("Sessions",       f"{n_sessions} ({sess_range})")
    col4.metric("Observer split", f"T={t_count}  |  P={p_count}")

    st.markdown(
        f"""
| Property | Value |
|---|---|
| Total records | **{total_records}** |
| Participants | **{n_participants}** |
| Sessions | **{n_sessions}** (sessions {sess_range}) |
| Therapist observations (T) | **{t_count}** |
| Parent observations (P) | **{p_count}** |
        """
    )


# ── Demographics Table ────────────────────────────────────────────────────────

def _show_demographics_table(participants):
    """Display comprehensive demographics table."""
    st.markdown("### Demographics Table  *(Participant Level — n=32)*")

    # ── Age ──
    age = participants["Age"].dropna()
    age_stats = pd.DataFrame({
        "Statistic": ["Min", "Max", "Mean", "SD", "Median"],
        "Value": [
            f"{int(age.min())}",
            f"{int(age.max())}",
            f"{age.mean():.2f}",
            f"{age.std():.2f}",
            f"{age.median():.1f}",
        ]
    })

    # ── Gender ──
    gender_counts = participants["Gender"].value_counts()
    n = len(participants)
    gender_rows = []
    for g, cnt in gender_counts.items():
        gender_rows.append({"Category": g, "Count": cnt, "Percent": f"{cnt/n*100:.1f}%"})
    gender_df = pd.DataFrame(gender_rows)

    # ── Autism Level ──
    autism_counts = participants["Autism Level"].value_counts().sort_index()
    autism_rows = []
    for lvl, cnt in autism_counts.items():
        flag = " ⚠️ (Participant 129)" if lvl == 6 else ""
        autism_rows.append({"Autism Level": f"Level {int(lvl)}{flag}", "Count": cnt})
    autism_df = pd.DataFrame(autism_rows)

    # ── Severity Level ──
    severity_counts = participants["Level of Severity"].value_counts().sort_index()
    severity_df = pd.DataFrame({
        "Severity Level": [f"Level {int(l)}" for l in severity_counts.index],
        "Count": severity_counts.values
    })

    # ── Co-existing Disability ──
    coexist_counts = participants["Any co-existing disabiltiy diagnosis"].value_counts()
    coexist_rows = []
    for val, cnt in coexist_counts.items():
        coexist_rows.append({"Co-existing Disability": val, "Count": cnt,
                              "Percent": f"{cnt/n*100:.1f}%"})
    coexist_df = pd.DataFrame(coexist_rows)

    # ── Stimming ──
    stimming_counts = participants["Stimming behaviour Identified?"].value_counts()
    stimming_rows = []
    for val, cnt in stimming_counts.items():
        stimming_rows.append({"Stimming": val, "Count": cnt,
                               "Percent": f"{cnt/n*100:.1f}%"})
    stimming_df = pd.DataFrame(stimming_rows)

    # ── Age Group ──
    agegroup_counts = participants["Age_Group"].value_counts().reindex(
        ["8-14", "15-19", "20-26"], fill_value=0
    )
    agegroup_df = pd.DataFrame({
        "Age Group": agegroup_counts.index,
        "Count": agegroup_counts.values
    })

    # ── Layout ──
    st.markdown("#### Age Statistics")
    st.dataframe(age_stats, use_container_width=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Gender")
        st.dataframe(gender_df, use_container_width=True)

        st.markdown("#### Co-existing Disability")
        st.dataframe(coexist_df, use_container_width=True)

    with c2:
        st.markdown("#### Stimming")
        st.dataframe(stimming_df, use_container_width=True)

        st.markdown("#### Age Group")
        st.dataframe(agegroup_df, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Autism Level")
        st.dataframe(autism_df, use_container_width=True)
    with c4:
        st.markdown("#### Severity Level")
        st.dataframe(severity_df, use_container_width=True)


# ── Cross-tabulations ─────────────────────────────────────────────────────────

def _show_cross_tabulations(participants):
    """Display cross-tabulation tables."""
    st.markdown("### Cross-tabulations")

    order_ag = ["8-14", "15-19", "20-26"]

    ct1 = pd.crosstab(
        participants["Age_Group"],
        participants["Autism Level"],
    )
    ct1 = ct1.reindex(order_ag, fill_value=0)
    ct1.columns = [f"Level {int(c)}" for c in ct1.columns]
    ct1.index.name = "Age Group"

    ct2 = pd.crosstab(
        participants["Age_Group"],
        participants["Gender"],
    )
    ct2 = ct2.reindex(order_ag, fill_value=0)
    ct2.index.name = "Age Group"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Age Group × Autism Level")
        st.dataframe(ct1, use_container_width=True)
    with c2:
        st.markdown("#### Age Group × Gender")
        st.dataframe(ct2, use_container_width=True)



def _render_visualizations(participants):
    """
    Render all 6 visualizations on a single page.
    Each chart uses an internal subplot grid (1×2 or 1×n) following Tufte's
    data-ink principles.  All charts are tuned for maximum clarity:
      • Font sizes 12–14 pt throughout
      • Descriptive panel subtitles (not just "Count")
      • Explicit n= in suptitle
      • Direct value labels on every bar (count + %)
      • Sufficient y-axis headroom so labels never clip

    Chart layouts
    ------------------
    1. Gender                     : [horiz count + %]        | [vert proportion %]
    2. Age Distribution            : [per-age histogram 7-26] | [age-group totals]
    3. Age Distribution by Gender  : [grouped bar AG×Gender]  | [stacked histogram]
    4. Autism Level                : [vert count bar]         | [horiz ranked + %]
    5. Autism Level × Age Group    : [small multiples — one panel per level]
    6. Severity Level              : [vert count bar]         | [horiz ranked + %]
    """

    n = len(participants)
    AGE_RANGE = range(7, 27)
    ORDER_AG = ["8-14", "15-19", "20-26"]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Gender Distribution
    # ─────────────────────────────────────────────────────────────────────────
    gender_counts = participants["Gender"].value_counts()
    colors_g = [T_BLUE, T_RED][: len(gender_counts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    # Left — horizontal count bar (count outside, % inside)
    hbars = ax1.barh(
        gender_counts.index, gender_counts.values,
        color=colors_g, height=0.5, edgecolor="none",
    )
    for bar in hbars:
        w = bar.get_width()
        mid_y = bar.get_y() + bar.get_height() / 2
        ax1.text(w + 0.4, mid_y, str(int(w)),
                 va="center", ha="left", fontsize=12, color="#222222")
        ax1.text(w / 2, mid_y, f"{w / n * 100:.0f}%",
                 va="center", ha="center", fontsize=11, color="white")
    ax1.set_xlim(0, gender_counts.max() + 6)
    ax1.set_xlabel("Number of Participants", fontsize=11, color="#555555")
    ax1.set_title("Count & Percentage per Gender", fontsize=12, pad=10, color="#333333")
    ax1.tick_params(axis="y", length=0)
    ax1.set_yticks(range(len(gender_counts)))
    ax1.set_yticklabels(gender_counts.index, fontsize=12)
    _tufte_ax(ax1)

    # Right — vertical proportion bar
    vbars = ax2.bar(
        gender_counts.index, gender_counts.values,
        color=colors_g, edgecolor="none", width=0.45,
    )
    for bar, val in zip(vbars, gender_counts.values):
        pct = val / n * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{int(val)}  ({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, color="#222222",
        )
    ax2.set_ylim(0, gender_counts.max() + 7)
    ax2.set_xlabel("Gender", fontsize=11, color="#555555")
    ax2.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax2.set_title("Proportion by Gender", fontsize=12, pad=10, color="#333333")
    ax2.tick_params(axis="x", labelsize=12)
    _tufte_ax(ax2)

    fig.suptitle(f"Gender Distribution  (n = {n})", fontsize=14, y=1.02, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Age Distribution
    # ─────────────────────────────────────────────────────────────────────────
    ages = participants["Age"].dropna().astype(int)
    age_val_counts = ages.value_counts().reindex(AGE_RANGE, fill_value=0)
    agegroup_counts = (
        participants["Age_Group"]
        .value_counts()
        .reindex(ORDER_AG, fill_value=0)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # Left — per-age histogram (7–26) with colour per group
    for age_val, count in age_val_counts.items():
        if 8 <= age_val <= 14:
            c = T_BLUE
        elif 15 <= age_val <= 19:
            c = T_GREEN
        elif 20 <= age_val <= 26:
            c = T_TAN
        else:
            c = "#cccccc"
        ax1.bar(age_val, count, width=0.78, color=c, edgecolor="white", linewidth=0.4)
        if count > 0:
            ax1.text(age_val, count + 0.1, str(count),
                     ha="center", va="bottom", fontsize=9, color="#444444")

    for boundary in [14.5, 19.5]:
        ax1.axvline(boundary, color="#bbbbbb", linestyle=":", linewidth=1)

    grp_label_y = age_val_counts.max() + 0.6
    ax1.text(11,  grp_label_y, "8–14",  ha="center", fontsize=10, color=T_BLUE,  fontweight="bold")
    ax1.text(17,  grp_label_y, "15–19", ha="center", fontsize=10, color=T_GREEN, fontweight="bold")
    ax1.text(23,  grp_label_y, "20–26", ha="center", fontsize=10, color=T_TAN,   fontweight="bold")

    ax1.set_xlim(6.5, 26.5)
    ax1.set_xticks(list(AGE_RANGE))
    ax1.set_xticklabels([str(a) for a in AGE_RANGE], fontsize=9, rotation=45)
    ax1.set_ylim(0, age_val_counts.max() + 3)
    ax1.set_xlabel("Age (years)", fontsize=11, color="#555555")
    ax1.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax1.set_title("Individual Age — Full Range (7–26 yrs)", fontsize=12, pad=10, color="#333333")
    _tufte_ax(ax1)

    # Right — age-group aggregated bar
    ag_colors = [T_BLUE, T_GREEN, T_TAN]
    ag_bars = ax2.bar(
        agegroup_counts.index, agegroup_counts.values,
        color=ag_colors[: len(agegroup_counts)], edgecolor="none", width=0.5,
    )
    for bar, val in zip(ag_bars, agegroup_counts.values):
        pct = val / n * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.4,
            f"{int(val)}  ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=11, color="#222222",
        )
    ax2.set_ylim(0, agegroup_counts.max() + 6)
    ax2.set_xlabel("Age Group", fontsize=11, color="#555555")
    ax2.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax2.set_title("Totals by Age Group", fontsize=12, pad=10, color="#333333")
    ax2.tick_params(axis="x", labelsize=12)
    _tufte_ax(ax2)

    fig.suptitle(f"Age Distribution  (n = {n})", fontsize=14, y=1.02, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Age Distribution by Gender
    # ─────────────────────────────────────────────────────────────────────────

    genders       = [g for g in ["Male", "Female"] if g in participants["Gender"].values]
    gender_colors = {"Male": T_BLUE, "Female": T_RED}

    ag_gender = (
        participants.groupby(["Age_Group", "Gender"])
        .size()
        .unstack(fill_value=0)
        .reindex(ORDER_AG, fill_value=0)
    )
    age_gender_ct = (
        participants.dropna(subset=["Age"])
        .assign(Age=lambda d: d["Age"].astype(int))
        .groupby(["Age", "Gender"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(AGE_RANGE), fill_value=0)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # Left — grouped bar: Age Group × Gender
    x      = np.arange(len(ORDER_AG))
    bw     = 0.35
    offset = {"Male": -bw / 2, "Female": bw / 2}

    for gender in genders:
        vals = ag_gender[gender].values if gender in ag_gender.columns else np.zeros(len(ORDER_AG))
        bars = ax1.bar(
            x + offset[gender], vals,
            width=bw, color=gender_colors[gender],
            edgecolor="none", label=gender,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.2,
                    str(int(val)),
                    ha="center", va="bottom", fontsize=11, color="#222222",
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels(ORDER_AG, fontsize=12)
    ax1.set_ylim(0, ag_gender.values.max() + 5)
    ax1.set_xlabel("Age Group", fontsize=11, color="#555555")
    ax1.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax1.set_title("Count by Age Group & Gender", fontsize=12, pad=10, color="#333333")
    ax1.legend(title="Gender", fontsize=11, title_fontsize=11, frameon=False, loc="upper right")
    _tufte_ax(ax1)

    # Right — per-age stacked histogram coloured by gender
    bottom   = np.zeros(len(AGE_RANGE))
    age_vals = list(AGE_RANGE)
    for gender in genders:
        col_data = (
            age_gender_ct[gender].values
            if gender in age_gender_ct.columns
            else np.zeros(len(AGE_RANGE))
        )
        ax2.bar(
            age_vals, col_data,
            bottom=bottom, width=0.78,
            color=gender_colors[gender], edgecolor="white", linewidth=0.4,
            label=gender,
        )
        bottom = bottom + col_data

    totals = age_gender_ct.sum(axis=1).reindex(list(AGE_RANGE), fill_value=0)
    for age_val, total in zip(age_vals, totals):
        if total > 0:
            ax2.text(age_val, total + 0.1, str(int(total)),
                     ha="center", va="bottom", fontsize=9, color="#444444")

    for boundary in [14.5, 19.5]:
        ax2.axvline(boundary, color="#bbbbbb", linestyle=":", linewidth=1)

    ax2.set_xlim(6.5, 26.5)
    ax2.set_xticks(age_vals)
    ax2.set_xticklabels([str(a) for a in age_vals], fontsize=9, rotation=45)
    ax2.set_ylim(0, totals.max() + 3)
    ax2.set_xlabel("Age (years)", fontsize=11, color="#555555")
    ax2.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax2.set_title("Stacked Age Histogram by Gender (7–26 yrs)", fontsize=12, pad=10, color="#333333")
    ax2.legend(title="Gender", fontsize=11, title_fontsize=11, frameon=False, loc="upper right")
    _tufte_ax(ax2)

    fig.suptitle(f"Age Distribution by Gender  (n = {n})", fontsize=14, y=1.02, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Autism Level Distribution
    # ─────────────────────────────────────────────────────────────────────────
    autism_counts  = participants["Autism Level"].value_counts().sort_index()
    labels_al      = [f"Level {int(l)}" for l in autism_counts.index]
    bar_colors_al  = [T_RED if l == 6 else T_BLUE for l in autism_counts.index]
    footnote       = "† Level 6: Participant 129 (flagged outlier)" if 6 in autism_counts.index else ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Left — vertical count bar
    vbars = ax1.bar(labels_al, autism_counts.values,
                    color=bar_colors_al, edgecolor="none", width=0.5)
    for bar, val in zip(vbars, autism_counts.values):
        pct = val / n * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.4,
            f"{int(val)}  ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=11, color="#222222",
        )
    if footnote:
        ax1.annotate(footnote, xy=(0, 0), xycoords="axes fraction",
                     xytext=(0, -0.22), textcoords="axes fraction",
                     fontsize=9, color=T_RED, ha="left")
    ax1.set_ylim(0, autism_counts.max() + 6)
    ax1.set_xlabel("Autism Level", fontsize=11, color="#555555")
    ax1.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax1.set_title("Count per Autism Level", fontsize=12, pad=10, color="#333333")
    ax1.tick_params(axis="x", labelsize=11)
    _tufte_ax(ax1)

    # Right — horizontal ranked bar
    al_sorted     = autism_counts.sort_values(ascending=True)
    labels_al_h   = [f"Level {int(l)}" for l in al_sorted.index]
    hbar_cols_al  = [T_RED if l == 6 else T_BLUE for l in al_sorted.index]
    hbars = ax2.barh(labels_al_h, al_sorted.values,
                     color=hbar_cols_al, height=0.5, edgecolor="none")
    for bar in hbars:
        w = bar.get_width()
        ax2.text(
            w + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w)}  ({w / n * 100:.0f}%)",
            va="center", ha="left", fontsize=11, color="#222222",
        )
    ax2.set_xlim(0, al_sorted.max() + 8)
    ax2.set_xlabel("Number of Participants", fontsize=11, color="#555555")
    ax2.set_title("Ranked by Frequency (with %)", fontsize=12, pad=10, color="#333333")
    ax2.tick_params(axis="y", length=0, labelsize=11)
    _tufte_ax(ax2)

    fig.suptitle(f"Autism Level Distribution  (n = {n})", fontsize=14, y=1.02, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Autism Level by Age Group — small multiples
    # ─────────────────────────────────────────────────────────────────────────
    ct = pd.crosstab(participants["Age_Group"], participants["Autism Level"])
    ct = ct.reindex(ORDER_AG, fill_value=0)
    autism_levels = sorted(ct.columns)
    n_levels      = len(autism_levels)
    level_colors  = [T_RED if l == 6 else TUFTE_PALETTE[i]
                     for i, l in enumerate(autism_levels)]

    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 5), sharey=True)
    fig.patch.set_facecolor("white")
    if n_levels == 1:
        axes = [axes]

    y_max = ct.values.max() + 4
    for ax, level, col in zip(axes, autism_levels, level_colors):
        vals = ct[level].values
        bars = ax.bar(ORDER_AG, vals, color=col, edgecolor="none", width=0.55)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.2, str(int(val)),
                    ha="center", va="bottom", fontsize=12, color="#222222",
                )
        flag = "  †" if level == 6 else ""
        ax.set_title(f"Autism Level {int(level)}{flag}", fontsize=12, pad=10, color="#333333")
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Age Group", fontsize=11, color="#555555")
        ax.tick_params(axis="x", labelsize=11)
        _tufte_ax(ax, keep_left=(ax is axes[0]))

    axes[0].set_ylabel("Number of Participants", fontsize=11, color="#555555")
    if 6 in autism_levels:
        fig.text(0.01, -0.04,
                 "† Level 6: Participant 129 (flagged outlier)",
                 fontsize=9, color=T_RED, ha="left")

    fig.suptitle(f"Autism Level Distribution by Age Group  (n = {n})",
                 fontsize=14, y=1.03, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Severity Level Distribution
    # ─────────────────────────────────────────────────────────────────────────
    severity_counts = participants["Level of Severity"].value_counts().sort_index()
    labels_sev      = [f"Level {int(l)}" for l in severity_counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # Left — vertical count bar
    vbars = ax1.bar(labels_sev, severity_counts.values,
                    color=T_PURPLE, edgecolor="none", width=0.5)
    for bar, val in zip(vbars, severity_counts.values):
        pct = val / n * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.4,
            f"{int(val)}  ({pct:.0f}%)",
            ha="center", va="bottom", fontsize=11, color="#222222",
        )
    ax1.set_ylim(0, severity_counts.max() + 6)
    ax1.set_xlabel("Severity Level", fontsize=11, color="#555555")
    ax1.set_ylabel("Number of Participants", fontsize=11, color="#555555")
    ax1.set_title("Count per Severity Level", fontsize=12, pad=10, color="#333333")
    ax1.tick_params(axis="x", labelsize=11)
    _tufte_ax(ax1)

    # Right — horizontal ranked bar
    sev_sorted   = severity_counts.sort_values(ascending=True)
    labels_sev_h = [f"Level {int(l)}" for l in sev_sorted.index]
    hbars = ax2.barh(labels_sev_h, sev_sorted.values,
                     color=T_PURPLE, height=0.5, edgecolor="none")
    for bar in hbars:
        w = bar.get_width()
        ax2.text(
            w + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w)}  ({w / n * 100:.0f}%)",
            va="center", ha="left", fontsize=11, color="#222222",
        )
    ax2.set_xlim(0, sev_sorted.max() + 8)
    ax2.set_xlabel("Number of Participants", fontsize=11, color="#555555")
    ax2.set_title("Ranked by Frequency (with %)", fontsize=12, pad=10, color="#333333")
    ax2.tick_params(axis="y", length=0, labelsize=11)
    _tufte_ax(ax2)

    fig.suptitle(f"Severity Level Distribution  (n = {n})", fontsize=14, y=1.02, color="#111111", fontweight="bold")
    plt.tight_layout(pad=1.8)
    st.pyplot(fig)
    plt.close(fig)


def display(df, scale_map=None):
    """Section 1: All content on a single page."""
    st.markdown("## Section 1 — Dataset Overview & Demographics")

    full, participants = _load_silver(df)
    
    # Show all content in sequence on one page
    _show_dataset_structure(full, participants)
    st.markdown("---")
    _show_demographics_table(participants)
    st.markdown("---")
    _show_cross_tabulations(participants)
    st.markdown("---")
    st.markdown("### Visualisations")
    _render_visualizations(participants)
