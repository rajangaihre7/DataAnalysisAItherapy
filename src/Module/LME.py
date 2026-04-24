"""
LME (Linear Mixed-Effects) Analysis — Display Module
=====================================================
Displays pre-computed LME modelling results in the Streamlit dashboard.

To generate results, run ONCE from the project root (with venv activated):
    python src/run_lme_pipeline.py

Results are read from  results/lme/
    model_comparison_table.csv
    final_model_coefficients.csv
    null_model_variance.csv
    lme_summary.csv
    model_diagnostics.png
    random_effects_distribution.png
    forest_plot_fixed_effects.png
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np

# ── Path helpers ───────────────────────────────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir     = os.path.normpath(os.path.join(_current_dir, '..'))
_project_dir = os.path.normpath(os.path.join(_src_dir,     '..'))

for p in [_current_dir, _src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _lme(filename: str = '') -> str:
    """Absolute path inside  results/lme/[filename]."""
    base = os.path.join(_project_dir, 'results', 'lme')
    return os.path.join(base, filename) if filename else base


def _read(filename: str) -> pd.DataFrame | None:
    path = _lme(filename)
    return pd.read_csv(path) if os.path.exists(path) else None


def _show_img(filename: str, caption: str = ''):
    path = _lme(filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(
            f"Image `{filename}` not found — "
            "run `python src/run_lme_pipeline.py` first."
        )


def _pipeline_ran() -> bool:
    return os.path.exists(_lme('lme_summary.csv'))


# ════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ════════════════════════════════════════════════════════════════════════════

def _render_overview():
    st.subheader("Overview & ICC (Null Model)")

    summary = _read('lme_summary.csv')
    if summary is None:
        st.warning("No results yet — run `python src/run_lme_pipeline.py` first.")
        return

    row = summary.set_index('Metric')['Value']

    # ── Key metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full dataset (N)",    row.get('N_Full',       '—'))
    c2.metric("Sentiment subset (N)", row.get('N_Sentiment', '—'))
    c3.metric("ICC",                  row.get('ICC',          '—'))
    c4.metric("Best model selected",  row.get('Best_Model',   '—'))

    st.markdown("---")

    # ── Null model variance decomposition ──────────────────────────────────
    st.markdown("### Null Model — Variance Decomposition")

    null_var = _read('null_model_variance.csv')
    if null_var is not None:
        st.dataframe(null_var.set_index('Component').rename(
            columns={'Value': 'Estimate'}),
            use_container_width=True
        )

    icc_val = row.get('ICC', None)
    try:
        icc_float = float(icc_val)
        if icc_float > 0.05:
            st.success(
                f"ICC = {icc_float:.4f} ({icc_float * 100:.1f}% of variance between "
                f"participants) — Mixed-effects modelling is justified (ICC > 0.05)."
            )
        else:
            st.warning(
                f"ICC = {icc_float:.4f} — Limited between-participant variance "
                "(ICC ≤ 0.05). Interpret results with caution."
            )
    except (TypeError, ValueError):
        pass

    st.markdown("---")
    st.markdown("### Full LME Summary Metrics")
    st.dataframe(summary.set_index('Metric'), use_container_width=True)


def _render_model_comparison():
    st.subheader("Model Comparison Table")

    comp = _read('model_comparison_table.csv')
    if comp is None:
        st.warning("No results yet — run `python src/run_lme_pipeline.py` first.")
        return

    st.markdown("""
The table below compares all fitted models.  
Models 1–2 use the **full dataset**; Models 3–4 use the **sentiment subset** (records with a valid Sentiment Score).
""")

    # Colour best AIC row
    numeric_cols = ['AIC', 'BIC', 'LogLik']
    for col in numeric_cols:
        if col in comp.columns:
            comp[col] = pd.to_numeric(comp[col], errors='coerce')

    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ── AIC bar chart ──────────────────────────────────────────────────────
    aic_data = comp[comp['AIC'].notna()][['Model', 'AIC']].copy()
    if not aic_data.empty:
        st.markdown("#### AIC by Model (lower is better)")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        colours = ['#1f77b4' if v != aic_data['AIC'].min() else '#d62728'
                   for v in aic_data['AIC']]
        ax.bar(aic_data['Model'], aic_data['AIC'], color=colours, edgecolor='black')
        ax.set_ylabel('AIC')
        ax.set_xlabel('Model')
        ax.set_title('AIC Comparison (red = lowest / best)')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── LRT summary ────────────────────────────────────────────────────────
    st.markdown("#### Likelihood Ratio Tests")
    lrt_rows = comp[comp['LRT_vs'] != '—'][['Model', 'LRT_vs', 'LRT_chi2', 'LRT_p']]
    if not lrt_rows.empty:
        st.dataframe(lrt_rows, use_container_width=True, hide_index=True)


def _render_coefficients():
    st.subheader("Final Model — Fixed Effect Coefficients")

    coef = _read('final_model_coefficients.csv')
    if coef is None:
        st.warning("No results yet — run `python src/run_lme_pipeline.py` first.")
        return

    summary = _read('lme_summary.csv')
    if summary is not None:
        best = summary.set_index('Metric')['Value'].get('Best_Model', 'Best model')
        st.info(f"Displaying coefficients for **{best}**.")

    # ── Coefficient table ──────────────────────────────────────────────────
    st.dataframe(coef, use_container_width=True, hide_index=True)

    # ── Plain-language interpretation ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Plain-Language Interpretation")

    _PARAM_LABELS = {
        'Intercept':                       'Intercept (Therapist, Level 1, Session 2 baseline)',
        'Session_Centred':                 'Session (per additional session)',
        'Observer_Numeric':                'Observer type (Parent vs Therapist)',
        'Autism_Numeric':                  'Autism Level (2 vs 1)',
        'Sentiment_Score':                 'NLP Sentiment Score',
        'Session_Centred:Observer_Numeric': 'Session × Observer interaction',
    }

    coef_dict = coef.set_index('Parameter').to_dict('index')
    for param, label in _PARAM_LABELS.items():
        if param not in coef_dict:
            continue
        row_c = coef_dict[param]
        est  = float(row_c['Estimate'])
        ci_l = float(row_c['CI_Lower'])
        ci_u = float(row_c['CI_Upper'])
        p    = float(row_c['p_value'])
        sig  = row_c['Significance']

        direction = "increase" if est > 0 else "decrease"
        sig_str   = "**significant**" if p < 0.05 else "*not significant*"

        if param == 'Intercept':
            st.markdown(
                f"**{label}:** β = {est:.4f}  \n"
                f"Expected Engagement Score for a therapist-rated, Autism Level 1 "
                f"participant at Session 2."
            )
        elif param == 'Session_Centred':
            st.markdown(
                f"**{label}:** β = {est:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], "
                f"p = {p:.4f} {sig}  \n"
                f"Each additional session is associated with a {abs(est):.4f} "
                f"{direction} in Engagement Score ({sig_str})."
            )
        elif param == 'Observer_Numeric':
            rater = "Parents" if est > 0 else "Therapists"
            st.markdown(
                f"**{label}:** β = {est:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], "
                f"p = {p:.4f} {sig}  \n"
                f"{rater} rate engagement {abs(est):.4f} points {'higher' if est > 0 else 'lower'} "
                f"than the other observer ({sig_str})."
            )
        elif param == 'Autism_Numeric':
            st.markdown(
                f"**{label}:** β = {est:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], "
                f"p = {p:.4f} {sig}  \n"
                f"Level 2 participants score {abs(est):.4f} points {'higher' if est > 0 else 'lower'} "
                f"than Level 1 ({sig_str})."
            )
        elif param == 'Sentiment_Score':
            st.markdown(
                f"**{label} (RQ2):** β = {est:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], "
                f"p = {p:.4f} {sig}  \n"
                f"A 0.1 increase in NLP sentiment probability is associated with a "
                f"{est * 0.1:.4f} {direction} in Engagement Score ({sig_str})."
            )
        elif param == 'Session_Centred:Observer_Numeric':
            direction_obs = "steeper" if est > 0 else "flatter"
            st.markdown(
                f"**{label}:** β = {est:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], "
                f"p = {p:.4f} {sig}  \n"
                f"Parents show a {direction_obs} engagement trajectory across sessions "
                f"compared to therapists ({sig_str})."
            )


def _render_diagnostics():
    st.subheader("Model Diagnostics")

    summary = _read('lme_summary.csv')
    if summary is not None:
        row = summary.set_index('Metric')['Value']
        sw_w = row.get('Shapiro_W', '—')
        sw_p = row.get('Shapiro_p', '—')
        c1, c2 = st.columns(2)
        c1.metric("Shapiro-Wilk W", sw_w)
        c2.metric("Shapiro-Wilk p", sw_p)

        try:
            if float(str(sw_p)) > 0.05:
                st.success("Residuals are approximately normally distributed (p > 0.05).")
            else:
                st.info(
                    "Residuals deviate from normality (p ≤ 0.05).  \n"
                    "With large samples, minor deviations are expected and often negligible "
                    "— inspect the Q-Q plot visually."
                )
        except (ValueError, TypeError):
            pass

    st.markdown("---")
    st.markdown("### Residual Diagnostic Plots")
    _show_img('model_diagnostics.png', 'Q-Q Plot · Residuals vs Fitted · Residual Distribution')

    st.markdown("---")
    st.markdown("### Participant-Level Random Intercepts")
    _show_img('random_effects_distribution.png',
              'Distribution of random intercepts (u_i) across all 32 participants')

    if summary is not None:
        row = summary.set_index('Metric')['Value']
        st.markdown("**Random Effects Summary**")
        col1, col2 = st.columns(2)
        col1.metric("Random Intercept Variance (σ²_u)",
                    row.get('Random_Intercept_Var', '—'))
        col2.metric("Residual Variance (σ²)",
                    row.get('Residual_Var', '—'))


def _render_forest():
    st.subheader("Forest Plot — Fixed Effects")

    st.markdown("""
Each point shows the estimated coefficient (β) with a 95% confidence interval.  
A CI that does not cross zero (red dashed line) indicates a statistically significant effect.
""")
    _show_img('forest_plot_fixed_effects.png', 'Fixed effect estimates with 95% confidence intervals')


def _render_rq_summary():
    st.subheader("Research Question Summary")

    summary = _read('lme_summary.csv')
    if summary is None:
        st.warning("No results yet — run `python src/run_lme_pipeline.py` first.")
        return

    row = summary.set_index('Metric')['Value']

    # ── ICC ────────────────────────────────────────────────────────────────
    with st.expander("Null Model — ICC & Variance Partitioning", expanded=True):
        try:
            icc = float(str(row.get('ICC', 'nan')))
            pct = float(str(row.get('ICC_pct', str(icc * 100)).replace('%', '')))
        except ValueError:
            icc, pct = float('nan'), float('nan')

        st.markdown(f"""
**ICC = {icc:.4f}** ({pct:.1f}% of total variance attributable to between-participant differences)

- ICC > 0.05 justifies a mixed-effects structure.
- This confirms that therapy engagement varies meaningfully across participants,
  independent of session or observer effects.
""")

    # ── Main effects ───────────────────────────────────────────────────────
    with st.expander("Main Effects (Session, Observer, Autism Level)", expanded=True):
        lrt_2 = row.get('M2_LRT_p', '—')
        st.markdown(f"""
**LRT (Model 2 vs Null): p = {lrt_2}**

- Session, observer type, and autism level were entered as fixed effects.
- The LRT assesses whether their combined inclusion significantly improves model fit
  over the intercept-only (null) model.
""")

    # ── Sentiment (RQ2) ────────────────────────────────────────────────────
    with st.expander("RQ2 — Does NLP Sentiment Predict Engagement?", expanded=True):
        sb    = row.get('Sentiment_Beta', 'n/a')
        sp    = row.get('Sentiment_p',    'n/a')
        lrt_3 = row.get('M3_LRT_p',       'n/a')

        try:
            sp_float = float(str(sp))
            sig_sent = sp_float < 0.05
            sent_label = ("YES — NLP-derived sentiment is a **significant** predictor of engagement "
                          "after controlling for session, observer type, and autism level."
                          if sig_sent else
                          "NO — Sentiment does not significantly predict engagement after controls.  \n"
                          "Any bivariate correlation may reflect shared temporal trends.")
        except (ValueError, TypeError):
            sent_label = "Result unavailable — sentiment data may not have been processed."

        st.markdown(f"""
**Sentiment_Score β = {sb}**, p = {sp}  
LRT (Model 3 vs Model 2): p = {lrt_3}

{sent_label}
""")

    # ── Interaction ────────────────────────────────────────────────────────
    with st.expander("Session × Observer Interaction", expanded=True):
        lrt_4 = row.get('Interaction_LRT_p', 'n/a')
        try:
            lrt_4_float = float(str(lrt_4))
            int_sig = lrt_4_float < 0.05
            int_label = ("The interaction **is significant** — therapists and parents observe "
                         "different engagement trajectories across sessions."
                         if int_sig else
                         "The interaction is **not significant** — no evidence that "
                         "engagement trajectories differ between observer types.")
        except (ValueError, TypeError):
            int_label = "Result unavailable."

        st.markdown(f"""
**LRT (Model 4 vs Model 3): p = {lrt_4}**

{int_label}
""")

    # ── Best model ─────────────────────────────────────────────────────────
    with st.expander("Best Model Selected", expanded=True):
        best  = row.get('Best_Model', '—')
        b_aic = row.get('Best_AIC',   '—')
        b_bic = row.get('Best_BIC',   '—')
        st.markdown(f"""
**{best}** was selected based on LRT significance and AIC comparison.

| Metric | Value |
|--------|-------|
| AIC    | {b_aic} |
| BIC    | {b_bic} |
""")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def display(df=None, scale_map=None):
    """Entry point called by app.py — df and scale_map are not used here."""
    st.subheader("LME: Linear Mixed-Effects Modelling")

    st.markdown("""
**Dataset:** Gold analytical dataset · 512 records · 32 ASD therapy participants  
**Dependent variable:** `Engagement_Score` (composite mean of 20 normalised Likert items, 0–1)  
**Grouping variable:** `Participant_ID` (random intercept)  
**Fixed effects:** Session, Observer Type, Autism Level, NLP Sentiment Score (where available)

> To run or re-run the analysis:  
> `python src/run_lme_pipeline.py`  (from the project root with venv active)
""")

    if not _pipeline_ran():
        st.warning(
            "LME results have not been generated yet.  \n"
            "Run `python src/run_lme_pipeline.py` from the project root to compute all models."
        )
        st.info(
            "**What the pipeline produces:**  \n"
            "- **Model 1** — Null (unconditional means) → ICC  \n"
            "- **Model 2** — Main effects: Session + Observer + Autism Level  \n"
            "- **Model 3** — Main effects + NLP Sentiment Score (RQ2)  \n"
            "- **Model 4** — Main effects + Sentiment + Session × Observer interaction  \n"
            "- Likelihood Ratio Tests, AIC/BIC comparison, and diagnostic plots"
        )
        return

    tabs = st.tabs([
        "Overview & ICC",
        "Model Comparison",
        "Coefficients",
        "Diagnostics",
        "Forest Plot",
        "RQ Summary",
    ])

    with tabs[0]:
        _render_overview()

    with tabs[1]:
        _render_model_comparison()

    with tabs[2]:
        _render_coefficients()

    with tabs[3]:
        _render_diagnostics()

    with tabs[4]:
        _render_forest()

    with tabs[5]:
        _render_rq_summary()
