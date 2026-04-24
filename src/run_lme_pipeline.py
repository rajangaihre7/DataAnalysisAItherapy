"""
Linear Mixed-Effects (LME) Modelling Pipeline — Standalone Runner
==================================================================
Run this script ONCE (outside Streamlit) to compute all LME results.
The Streamlit dashboard will then load and display the saved outputs.

Usage (from the project root, with venv activated):
    python src/run_lme_pipeline.py

Outputs written to  results/lme/
    model_comparison_table.csv
    final_model_coefficients.csv
    null_model_variance.csv
    model_diagnostics.png
    random_effects_distribution.png
    forest_plot_fixed_effects.png
    lme_summary.csv
"""

import os
import sys

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for special chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# ── Path setup ─────────────────────────────────────────────────────────────────
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.normpath(os.path.join(_script_dir, '..'))


def _results_dir(sub: str = '') -> str:
    base = os.path.join(_project_dir, 'results', 'lme')
    path = os.path.join(base, sub) if sub else base
    os.makedirs(path, exist_ok=True)
    return path


def _out(filename: str) -> str:
    return os.path.join(_results_dir(), filename)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load Gold + Sentiment dataset, falling back to Gold-only if needed."""
    sent_path = os.path.join(_project_dir, 'results', 'nlp', 'data_gold_with_sentiment.csv')
    gold_path = os.path.join(_project_dir, 'Data', 'Gold', 'data_gold_analytical.csv')

    if os.path.exists(sent_path):
        print(f"[LME] Loading Gold+Sentiment dataset: {sent_path}")
        df = pd.read_csv(sent_path, low_memory=False)
    elif os.path.exists(gold_path):
        print(f"[LME] Sentiment dataset not found — loading Gold only: {gold_path}")
        df = pd.read_csv(gold_path, low_memory=False)
    else:
        raise FileNotFoundError(
            "Neither data_gold_with_sentiment.csv nor data_gold_analytical.csv found.\n"
            "Run `python src/run_nlp_pipeline.py` first."
        )

    # ── Standardise column names ────────────────────────────────────────────
    rename_map = {
        'Participant id':   'Participant_ID',
        'Session number':   'Session_Number',
        'Submitted_by':     'Observer_Type',
        'Autism Level':     'Autism_Level',
        'Level of Severity': 'Severity_Level',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    print(f"[LME] Loaded: {len(df)} records, "
          f"{df['Participant_ID'].nunique()} participants, "
          f"Sessions {sorted(df['Session_Number'].unique())}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def prepare_variables(df: pd.DataFrame):
    """Centre session, encode categoricals, split into analysis datasets."""

    # Session_Centred: 2→0, 3→1, … 9→7  (intercept = Session 2 baseline)
    df['Session_Centred'] = df['Session_Number'] - 2

    # Observer_Numeric: T=0 (reference), P=1
    df['Observer_Numeric'] = (df['Observer_Type'] == 'P').astype(int)

    # Autism_Numeric: Level 1=0 (reference), Level 2=1
    df['Autism_Numeric'] = (df['Autism_Level'] == 2).astype(int)

    # Full dataset — all records with Engagement_Score
    df_full = df.dropna(subset=['Engagement_Score']).copy()

    # Sentiment dataset — records with both scores
    has_sentiment = 'Sentiment_Score' in df.columns
    df_sent = (
        df.dropna(subset=['Engagement_Score', 'Sentiment_Score']).copy()
        if has_sentiment else pd.DataFrame()
    )

    print(f"\n[LME] Full dataset (no Sentiment):   {len(df_full)} records")
    if has_sentiment:
        print(f"[LME] Sentiment dataset:              {len(df_sent)} records")
    else:
        print("[LME] Sentiment_Score not found — Models 3 and 4 will be skipped.")

    # Descriptive summary
    print(f"\n  Engagement_Score: mean={df_full['Engagement_Score'].mean():.4f}, "
          f"SD={df_full['Engagement_Score'].std():.4f}")
    for obs in ['T', 'P']:
        d = df_full[df_full['Observer_Type'] == obs]['Engagement_Score']
        print(f"  Observer {obs}: mean={d.mean():.4f}, SD={d.std():.4f}, n={len(d)}")

    return df_full, df_sent, has_sentiment


# ══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lrt(result_full, result_reduced, extra_params: int):
    lr_stat = 2 * (result_full.llf - result_reduced.llf)
    p_val   = stats.chi2.sf(lr_stat, extra_params)
    return lr_stat, p_val


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: MODEL 1 — NULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_model1(df_full: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODEL 1: NULL MODEL (Unconditional Means)")
    print("=" * 60)

    m = smf.mixedlm("Engagement_Score ~ 1", data=df_full,
                    groups=df_full["Participant_ID"])
    r = m.fit(reml=False)
    print(r.summary())

    var_between = r.cov_re.iloc[0, 0]
    var_within  = r.scale
    icc         = var_between / (var_between + var_within)

    print(f"\n  Between-participant variance (sigma2_u): {var_between:.6f}")
    print(f"  Within-participant variance  (sigma2):   {var_within:.6f}")
    print(f"  ICC = {icc:.4f} ({icc * 100:.1f}%)")
    if icc > 0.05:
        print("  ICC > 0.05 — Mixed-effects modelling is justified.")
    else:
        print("  WARNING: ICC ≤ 0.05 — Limited between-participant variance.")

    return r, icc, var_between, var_within


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: MODEL 2 — MAIN EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

def fit_model2(df_full: pd.DataFrame, result1):
    print("\n" + "=" * 60)
    print("MODEL 2: MAIN EFFECTS (Session + Observer + Autism Level)")
    print("=" * 60)

    m = smf.mixedlm(
        "Engagement_Score ~ Session_Centred + Observer_Numeric + Autism_Numeric",
        data=df_full, groups=df_full["Participant_ID"]
    )
    r = m.fit(reml=False)
    print(r.summary())

    lr_stat, lr_p = _lrt(r, result1, extra_params=3)
    print(f"\n  LRT vs Model 1: χ²={lr_stat:.4f}, p={lr_p:.6f} "
          f"({'significant' if lr_p < 0.05 else 'not significant'})")

    return r, lr_stat, lr_p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MODEL 3 — ADD SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def fit_model3(df_sent: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODEL 3: MAIN EFFECTS + SENTIMENT SCORE")
    print(f"(N = {len(df_sent)} records with Sentiment Score)")
    print("=" * 60)

    # Refit Model 2 on the same sentiment subset for a fair LRT
    m2s = smf.mixedlm(
        "Engagement_Score ~ Session_Centred + Observer_Numeric + Autism_Numeric",
        data=df_sent, groups=df_sent["Participant_ID"]
    )
    r2s = m2s.fit(reml=False)

    m3 = smf.mixedlm(
        "Engagement_Score ~ Session_Centred + Observer_Numeric + Autism_Numeric + Sentiment_Score",
        data=df_sent, groups=df_sent["Participant_ID"]
    )
    r3 = m3.fit(reml=False)
    print(r3.summary())

    lr_stat, lr_p = _lrt(r3, r2s, extra_params=1)
    print(f"\n  LRT vs Model 2 (sent. subset): χ²={lr_stat:.4f}, p={lr_p:.6f} "
          f"({'significant' if lr_p < 0.05 else 'not significant'})")

    sent_coef = r3.fe_params.get('Sentiment_Score', np.nan)
    sent_p    = r3.pvalues.get('Sentiment_Score', np.nan)
    sent_ci   = (r3.conf_int().loc['Sentiment_Score'].tolist()
                 if 'Sentiment_Score' in r3.conf_int().index else [np.nan, np.nan])

    print(f"\n  RQ2 — Sentiment_Score β={sent_coef:.4f}, "
          f"CI=[{sent_ci[0]:.4f}, {sent_ci[1]:.4f}], p={sent_p:.6f}")

    return r2s, r3, lr_stat, lr_p, sent_coef, sent_p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: MODEL 4 — INTERACTION
# ══════════════════════════════════════════════════════════════════════════════

def fit_model4(df_sent: pd.DataFrame, result3):
    print("\n" + "=" * 60)
    print("MODEL 4: INTERACTION (Session × Observer)")
    print("=" * 60)

    m4 = smf.mixedlm(
        "Engagement_Score ~ Session_Centred + Observer_Numeric + Autism_Numeric "
        "+ Sentiment_Score + Session_Centred:Observer_Numeric",
        data=df_sent, groups=df_sent["Participant_ID"]
    )
    r4 = m4.fit(reml=False)
    print(r4.summary())

    lr_stat, lr_p = _lrt(r4, result3, extra_params=1)
    print(f"\n  LRT vs Model 3: χ²={lr_stat:.4f}, p={lr_p:.6f} "
          f"({'significant' if lr_p < 0.05 else 'not significant'})")

    int_param = 'Session_Centred:Observer_Numeric'
    int_coef  = r4.fe_params.get(int_param, np.nan)
    int_p     = r4.pvalues.get(int_param, np.nan)

    if not np.isnan(int_coef):
        direction = "steeper" if int_coef > 0 else "flatter"
        print(f"  Interaction: β={int_coef:.4f}, p={int_p:.6f} "
              f"— parents show {direction} trajectory than therapists")

    return r4, lr_stat, lr_p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison_table(results: dict) -> pd.DataFrame:
    rows = []

    def _row(name, n, fe, r, lrt_vs, lrt_p):
        return {
            'Model':         name,
            'N':             n,
            'Fixed_Effects': fe,
            'AIC':           round(r.aic, 2),
            'BIC':           round(r.bic, 2),
            'LogLik':        round(r.llf, 2),
            'LRT_vs':        lrt_vs,
            'LRT_chi2':      '—' if lrt_p == '—' else f"{results.get(f'{name}_lrt_chi2', '—')}",
            'LRT_p':         lrt_p,
        }

    r1    = results['r1']
    r2    = results['r2']
    r2s   = results.get('r2s')
    r3    = results.get('r3')
    r4    = results.get('r4')
    n_full = results['n_full']
    n_sent = results.get('n_sent', 0)

    rows.append({
        'Model': 'M1: Null', 'N': n_full,
        'Fixed_Effects': 'Intercept only',
        'AIC': round(r1.aic, 2), 'BIC': round(r1.bic, 2), 'LogLik': round(r1.llf, 2),
        'LRT_vs': '—', 'LRT_chi2': '—', 'LRT_p': '—',
    })
    rows.append({
        'Model': 'M2: Main Effects', 'N': n_full,
        'Fixed_Effects': 'Session + Observer + Autism',
        'AIC': round(r2.aic, 2), 'BIC': round(r2.bic, 2), 'LogLik': round(r2.llf, 2),
        'LRT_vs': 'vs M1',
        'LRT_chi2': round(results['lr_2v1_chi2'], 4),
        'LRT_p': round(results['lr_2v1_p'], 6),
    })

    if r2s is not None:
        rows.append({
            'Model': 'M2: Main (sent. subset)', 'N': n_sent,
            'Fixed_Effects': 'Session + Observer + Autism',
            'AIC': round(r2s.aic, 2), 'BIC': round(r2s.bic, 2), 'LogLik': round(r2s.llf, 2),
            'LRT_vs': '—', 'LRT_chi2': '—', 'LRT_p': '—',
        })

    if r3 is not None:
        rows.append({
            'Model': 'M3: + Sentiment', 'N': n_sent,
            'Fixed_Effects': '+ Sentiment Score',
            'AIC': round(r3.aic, 2), 'BIC': round(r3.bic, 2), 'LogLik': round(r3.llf, 2),
            'LRT_vs': 'vs M2(sub)',
            'LRT_chi2': round(results['lr_3v2_chi2'], 4),
            'LRT_p': round(results['lr_3v2_p'], 6),
        })

    if r4 is not None:
        rows.append({
            'Model': 'M4: + Interaction', 'N': n_sent,
            'Fixed_Effects': '+ Session × Observer',
            'AIC': round(r4.aic, 2), 'BIC': round(r4.bic, 2), 'LogLik': round(r4.llf, 2),
            'LRT_vs': 'vs M3',
            'LRT_chi2': round(results['lr_4v3_chi2'], 4),
            'LRT_p': round(results['lr_4v3_p'], 6),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(_out('model_comparison_table.csv'), index=False)
    print(f"\n[LME] Saved: model_comparison_table.csv")
    print(comparison.to_string(index=False))
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: SAVE FINAL MODEL COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════

def save_coefficients(best_model, best_name: str) -> pd.DataFrame:
    coefs  = best_model.fe_params
    ci     = best_model.conf_int().loc[coefs.index]   # fixed effects only
    pvals  = best_model.pvalues.loc[coefs.index]      # fixed effects only
    # bse includes the random-effect variance row — restrict to fixed effects
    se     = best_model.bse.loc[coefs.index]

    sig_labels = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                  for p in pvals.values]

    coef_table = pd.DataFrame({
        'Parameter':    coefs.index,
        'Estimate':     coefs.values.round(6),
        'Std_Error':    se.values.round(6),
        'CI_Lower':     ci.iloc[:, 0].values.round(6),
        'CI_Upper':     ci.iloc[:, 1].values.round(6),
        'z_value':      (coefs / se).values.round(4),
        'p_value':      pvals.values.round(6),
        'Significance': sig_labels,
    })

    coef_table.to_csv(_out('final_model_coefficients.csv'), index=False)
    print(f"\n[LME] Saved: final_model_coefficients.csv ({best_name})")
    print(coef_table.to_string(index=False))
    return coef_table


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: DIAGNOSTICS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def save_diagnostics(best_model):
    residuals = best_model.resid
    fitted    = best_model.fittedvalues

    # Shapiro-Wilk (max 5000 samples)
    sample = residuals if len(residuals) <= 5000 else residuals.sample(5000, random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    print(f"\n[LME] Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")

    # Diagnostic figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LME Model Diagnostics', fontsize=14, fontweight='bold')

    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title('Q-Q Plot of Residuals')
    axes[0].get_lines()[0].set(markerfacecolor='steelblue', markersize=3, alpha=0.6)

    axes[1].scatter(fitted, residuals, alpha=0.3, s=10, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Fitted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs Fitted')

    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Residual Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')

    plt.tight_layout()
    plt.savefig(_out('model_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[LME] Saved: model_diagnostics.png")

    # Random effects distribution
    re_values = [v.iloc[0] for v in best_model.random_effects.values()]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(re_values, bins=15, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Random Intercept Value (u_i)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Participant-Level Random Intercepts (N={len(re_values)})')
    plt.tight_layout()
    plt.savefig(_out('random_effects_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[LME] Saved: random_effects_distribution.png")

    return shapiro_stat, shapiro_p


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: FOREST PLOT
# ══════════════════════════════════════════════════════════════════════════════

def save_forest_plot(coef_table: pd.DataFrame):
    _PARAM_LABELS = {
        'Session_Centred':               'Session (per session)',
        'Observer_Numeric':              'Observer (Parent vs Therapist)',
        'Autism_Numeric':                'Autism Level (2 vs 1)',
        'Sentiment_Score':               'NLP Sentiment Score',
        'Session_Centred:Observer_Numeric': 'Session × Observer',
    }

    params = coef_table[coef_table['Parameter'] != 'Intercept'].copy()
    params = params.sort_values('Estimate').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(4, len(params) * 1.2 + 1)))
    y_pos = range(len(params))

    ax.errorbar(
        params['Estimate'], list(y_pos),
        xerr=[params['Estimate'] - params['CI_Lower'],
              params['CI_Upper'] - params['Estimate']],
        fmt='o', color='navy', ecolor='gray', capsize=4, markersize=8, zorder=3
    )
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Null effect (β=0)')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([_PARAM_LABELS.get(p, p) for p in params['Parameter']])
    ax.set_xlabel('Coefficient Estimate (β) with 95% CI')
    ax.set_title('Fixed Effects: Forest Plot')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()

    x_max = params['CI_Upper'].max()
    for i, row in params.iterrows():
        ax.annotate(
            row['Significance'],
            (x_max + abs(x_max) * 0.05, i),
            va='center', fontsize=10
        )

    plt.tight_layout()
    plt.savefig(_out('forest_plot_fixed_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[LME] Saved: forest_plot_fixed_effects.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: SAVE SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(results: dict, shapiro_stat: float, shapiro_p: float,
                 best_name: str, best_model):
    icc        = results['icc']
    r1         = results['r1']
    n_full     = results['n_full']
    lr_2v1_p   = results['lr_2v1_p']
    sent_coef  = results.get('sent_coef', np.nan)
    sent_p     = results.get('sent_p', np.nan)
    lr_3v2_p   = results.get('lr_3v2_p', np.nan)
    lr_4v3_p   = results.get('lr_4v3_p', np.nan)
    n_sent     = results.get('n_sent', 0)

    rows = [
        ('Best_Model',           best_name),
        ('N_Full',               n_full),
        ('N_Sentiment',          n_sent),
        ('ICC',                  round(icc, 4)),
        ('ICC_pct',              f"{icc * 100:.1f}%"),
        ('Var_Between',          round(results['var_between'], 6)),
        ('Var_Within',           round(results['var_within'], 6)),
        ('M1_AIC',               round(r1.aic, 2)),
        ('M2_LRT_p',             round(lr_2v1_p, 6)),
        ('Sentiment_Beta',       round(sent_coef, 4) if not np.isnan(sent_coef) else 'n/a'),
        ('Sentiment_p',          round(sent_p, 6) if not np.isnan(sent_p) else 'n/a'),
        ('M3_LRT_p',             round(lr_3v2_p, 6) if not np.isnan(lr_3v2_p) else 'n/a'),
        ('Interaction_LRT_p',    round(lr_4v3_p, 6) if not np.isnan(lr_4v3_p) else 'n/a'),
        ('Best_AIC',             round(best_model.aic, 2)),
        ('Best_BIC',             round(best_model.bic, 2)),
        ('Shapiro_W',            round(shapiro_stat, 4)),
        ('Shapiro_p',            round(shapiro_p, 6)),
        ('Random_Intercept_Var', round(best_model.cov_re.iloc[0, 0], 6)),
        ('Residual_Var',         round(best_model.scale, 6)),
    ]

    summary_df = pd.DataFrame(rows, columns=['Metric', 'Value'])
    summary_df.to_csv(_out('lme_summary.csv'), index=False)
    print(f"\n[LME] Saved: lme_summary.csv")

    # Null model variance breakdown
    null_var = pd.DataFrame({
        'Component': ['Between-participant (sigma2_u)', 'Within-participant (sigma2)', 'ICC'],
        'Value': [
            round(results['var_between'], 6),
            round(results['var_within'],  6),
            round(icc, 4),
        ]
    })
    null_var.to_csv(_out('null_model_variance.csv'), index=False)
    print("[LME] Saved: null_model_variance.csv")

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("LME MODELLING PIPELINE")
    print("=" * 60)

    df = load_data()
    df_full, df_sent, has_sentiment = prepare_variables(df)

    # ── Fit models ─────────────────────────────────────────────────────────
    r1, icc, var_between, var_within = fit_model1(df_full)
    r2, lr_2v1_chi2, lr_2v1_p        = fit_model2(df_full, r1)

    results = {
        'r1':           r1,
        'r2':           r2,
        'icc':          icc,
        'var_between':  var_between,
        'var_within':   var_within,
        'n_full':       len(df_full),
        'lr_2v1_chi2':  lr_2v1_chi2,
        'lr_2v1_p':     lr_2v1_p,
    }

    best_model = r2
    best_name  = 'Model 2'

    if has_sentiment and len(df_sent) > 0:
        r2s, r3, lr_3v2_chi2, lr_3v2_p, sent_coef, sent_p = fit_model3(df_sent)
        r4, lr_4v3_chi2, lr_4v3_p = fit_model4(df_sent, r3)

        results.update({
            'r2s':          r2s,
            'r3':           r3,
            'r4':           r4,
            'n_sent':       len(df_sent),
            'lr_3v2_chi2':  lr_3v2_chi2,
            'lr_3v2_p':     lr_3v2_p,
            'lr_4v3_chi2':  lr_4v3_chi2,
            'lr_4v3_p':     lr_4v3_p,
            'sent_coef':    sent_coef,
            'sent_p':       sent_p,
        })

        if lr_4v3_p < 0.05:
            best_model, best_name = r4, 'Model 4'
        else:
            best_model, best_name = r3, 'Model 3'

    print(f"\n[LME] Best model selected: {best_name}")

    # ── Save outputs ────────────────────────────────────────────────────────
    save_comparison_table(results)
    coef_table = save_coefficients(best_model, best_name)
    shapiro_stat, shapiro_p = save_diagnostics(best_model)
    save_forest_plot(coef_table)
    save_summary(results, shapiro_stat, shapiro_p, best_name, best_model)

    # ── Chapter 4 summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY FOR CHAPTER 4")
    print("=" * 60)
    print(f"""
  1. NULL MODEL:
     ICC = {icc:.4f} ({icc * 100:.1f}% of variance between participants)

  2. MAIN EFFECTS (N = {len(df_full)}):
     LRT vs Null: χ²={lr_2v1_chi2:.4f}, p={lr_2v1_p:.6f}

  3. SENTIMENT MODEL (N = {results.get('n_sent', 'n/a')}):
     Sentiment β = {results.get('sent_coef', float('nan')):.4f},
     p = {results.get('sent_p', float('nan')):.6f}

  4. INTERACTION (Session × Observer):
     LRT p = {results.get('lr_4v3_p', float('nan')):.6f}

  5. BEST MODEL: {best_name}
     AIC = {best_model.aic:.2f}, BIC = {best_model.bic:.2f}
""")
    print("[LME] Pipeline complete. Results saved to results/lme/")


if __name__ == '__main__':
    main()
