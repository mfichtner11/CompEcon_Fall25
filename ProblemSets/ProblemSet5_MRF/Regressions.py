"""
Regression Analysis Module for NSCH Developmental Screening Study
================================================================

This module implements three econometric models to analyze the impact of the 2017 
Bright Futures 4th Edition guidelines on developmental screening rates using the 
National Survey of Children's Health (NSCH) dataset.

Models Implemented:
1. Basic OLS: Simple linear regression with demographic controls
2. State Fixed Effects: Controls for unobserved state-level heterogeneity  
3. Regression Discontinuity: Exploits sharp policy discontinuity at 2017 cutoff

The analysis focuses on children aged 9-35 months across survey years 2016-2021,
examining how the January 2017 policy change affected screening compliance.

Author: Marli Fichtner
Course: Computational Economics Fall 2025
Dataset: NSCH 2016-2021 (cleaned and analysis-ready)
Policy Context: AAP Bright Futures Guidelines Implementation
Statistical Software: Python (statsmodels, stargazer)
"""

# Import packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm
import os
from stargazer.stargazer import Stargazer

def load_analysis_data():
    """
    Load the cleaned and analysis-ready NSCH dataset.
    
    This function loads the preprocessed NSCH dataset that has been cleaned,
    filtered to the target population (9-35 months), and enhanced with 
    analysis variables created by the data cleaning pipeline.
    
    Returns
    -------
    pd.DataFrame
        Analysis-ready dataset containing:
        - DEV_SCREEN: Binary developmental screening outcome (0/1)
        - POST2017: Policy treatment indicator (1 if year ≥ 2017)
        - Insurance indicators: PRIVATE_INSURANCE, PUBLIC_INSURANCE
        - Demographics: FEMALE, BLACK, OTHER_RACE, HIGH_EDUCATION, URBAN
        - Controls: AGE_MONTHS, STATE
        
    Raises
    ------
    FileNotFoundError
        If the cleaned dataset is not found in expected location
        
    Notes
    -----
    Expected file location: ../cleaned_data/nsch_analysis_ready.csv
    This file should be created by running DataCleaning_PS5_Fichtner.py first.
    
    Examples
    --------
    >>> df = load_analysis_data()
    >>> print(f"Loaded {len(df):,} observations with {len(df.columns)} variables")
    Loaded 20,171 observations with 15 variables
    """
    # Set directory paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "cleaned_data", "nsch_analysis_ready.csv")
    
    # Load the preprocessed dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded: {len(df):,} rows")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find cleaned dataset at {csv_path}. "
                              f"Please run DataCleaning_PS5_Fichtner.py first.")

def estimate_basic_ols(df):
    """
    Estimate Model 1: Basic OLS regression with demographic controls.
    
    This model provides baseline estimates of the policy effect without controlling
    for unobserved state-level factors. It includes standard demographic and 
    socioeconomic controls commonly used in health services research.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset with required variables
        
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model with heteroskedasticity-robust standard errors
        
    Notes
    -----
    Model Specification:
    DEV_SCREEN = β₀ + β₁POST2017 + β₂PRIVATE_INS + β₃PUBLIC_INS + 
                 β₄FEMALE + β₅BLACK + β₆OTHER_RACE + β₇HIGH_ED + 
                 β₈URBAN + β₉AGE_MONTHS + ε
    
    Key Coefficient:
    - β₁ (POST2017): Average treatment effect of 2017 policy change
    
    Standard Errors:
    - HC1 (MacKinnon-White) heteroskedasticity-robust standard errors
    
    Interpretation:
    - Coefficients represent percentage point changes in screening probability
    - POST2017 coefficient is the main policy parameter of interest
    
    Examples
    --------
    >>> model1 = estimate_basic_ols(df)
    >>> print(f"Policy effect: {model1.params['POST2017']:.4f}")
    Policy effect: 0.0442
    """
    print("="*60)
    print("MODEL 1: BASIC OLS")
    print("="*60)
    
    # Estimate basic OLS with demographic controls
    formula = ('DEV_SCREEN ~ POST2017 + PRIVATE_INSURANCE + PUBLIC_INSURANCE + '
              'FEMALE + BLACK + OTHER_RACE + HIGH_EDUCATION + URBAN + AGE_MONTHS')
    
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC1')
    print(model.summary())
    
    return model

def estimate_state_fixed_effects(df):
    """
    Estimate Model 2: OLS with state fixed effects.
    
    This model controls for time-invariant unobserved heterogeneity across states
    that might confound the policy effect. State fixed effects absorb differences
    in healthcare infrastructure, provider density, and state-level policies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset with STATE variable for fixed effects
        
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted model with state fixed effects and robust standard errors
        
    Notes
    -----
    Model Specification:
    DEV_SCREEN = β₀ + β₁POST2017 + β₂PRIVATE_INS + β₃PUBLIC_INS + 
                 β₄FEMALE + β₅BLACK + β₆OTHER_RACE + β₇HIGH_ED + 
                 β₈URBAN + β₉AGE_MONTHS + γₛ + ε
    
    Where γₛ are state-specific intercepts (fixed effects)
    
    Identification:
    - Policy effect identified from within-state variation over time
    - Controls for state-level confounders (Medicaid expansion, provider supply)
    - More credible causal identification than basic OLS
    
    Advantages:
    - Eliminates bias from time-invariant state characteristics
    - Controls for state healthcare policy environments
    - Reduces omitted variable bias
    
    Limitations:
    - Cannot identify effects of time-invariant state policies
    - Requires sufficient within-state variation
    - May reduce precision due to multicollinearity
    
    Examples
    --------
    >>> model2 = estimate_state_fixed_effects(df)
    >>> print(f"Policy effect with state FE: {model2.params['POST2017']:.4f}")
    Policy effect with state FE: 0.0421
    """
    print("\n" + "="*60)
    print("MODEL 2: STATE FIXED EFFECTS")
    print("="*60)
    
    # Estimate model with state fixed effects
    formula = ('DEV_SCREEN ~ POST2017 + PRIVATE_INSURANCE + PUBLIC_INSURANCE + '
              'FEMALE + BLACK + OTHER_RACE + HIGH_EDUCATION + URBAN + AGE_MONTHS + C(STATE)')
    
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC1')
    print(model.summary())
    
    return model

def estimate_regression_discontinuity(df):
    """
    Estimate Model 3: Regression Discontinuity design around 2017 policy change.
    
    This model exploits the sharp discontinuity in policy implementation at
    January 2017 to identify causal effects of the Bright Futures guidelines.
    The running variable is survey year, with treatment assignment determined
    by the 2017 cutoff.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset that will be modified to include RD variables
        
    Returns
    -------
    tuple
        (fitted_model, enhanced_dataframe) where:
        - fitted_model: statsmodels regression results
        - enhanced_dataframe: df with added RD variables
        
    Notes
    -----
    Model Specification:
    DEV_SCREEN = β₀ + β₁POST2017 + β₂YEAR_CENTERED + β₃(POST2017×YEAR_CENTERED) + 
                 β₄PRIVATE_INS + β₅PUBLIC_INS + β₆FEMALE + β₇BLACK + β₈OTHER_RACE + 
                 β₉HIGH_ED + β₁₀URBAN + β₁₁AGE_MONTHS + ε
    
    Where:
    - POST2017 = 1 if survey year ≥ 2017 (treatment indicator)
    - YEAR_CENTERED = (YEAR - 2017) (running variable centered at cutoff)
    - POST2017×YEAR_CENTERED = interaction allowing different slopes
    
    Key Parameters:
    - β₁: RD treatment effect (jump at 2017 cutoff)
    - β₂: Pre-treatment trend slope
    - β₃: Change in slope post-treatment
    
    Identification Strategy:
    - Sharp RD: Treatment assignment purely determined by year ≥ 2017
    - Local randomization around cutoff
    - Compares outcomes just before vs just after 2017
    
    RD Assumptions:
    1. **Continuity**: No other policies/shocks at 2017 cutoff
    2. **No Manipulation**: Survey timing not manipulated around cutoff
    3. **Local Randomization**: Observations near cutoff are comparable
    
    Examples
    --------
    >>> model3, df_enhanced = estimate_regression_discontinuity(df)
    >>> rd_effect = model3.params['POST2017']
    >>> print(f"RD treatment effect: {rd_effect:.4f}")
    RD treatment effect: 0.0445
    """
    print("\n" + "="*60)
    print("MODEL 3: REGRESSION DISCONTINUITY")
    print("="*60)
    
    # Create RD variables
    df['YEAR_CENTERED'] = df['YEAR'] - 2017  # Center running variable at cutoff
    df['RD_INTERACTION'] = df['POST2017'] * df['YEAR_CENTERED']  # Different slopes
    
    print(f"RD Setup:")
    print(f"  Cutoff year: 2017")
    print(f"  Pre-treatment years: {sorted(df[df['POST2017']==0]['YEAR'].unique())}")
    print(f"  Post-treatment years: {sorted(df[df['POST2017']==1]['YEAR'].unique())}")
    print(f"  Sample size: {len(df):,} observations")
    
    # Estimate RD model
    formula = ('DEV_SCREEN ~ POST2017 + YEAR_CENTERED + RD_INTERACTION + '
              'PRIVATE_INSURANCE + PUBLIC_INSURANCE + '
              'MALE + BLACK + OTHER_RACE + HIGH_EDUCATION + URBAN + AGE_MONTHS')
    
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC1')
    print(model.summary())
    
    # RD-specific diagnostics
    print(f"\nRD Diagnostics:")
    rd_effect = model.params['POST2017']
    rd_se = model.bse['POST2017']
    print(f"  Treatment effect at cutoff: {rd_effect:.4f} ({rd_se:.4f})")
    print(f"  Pre-treatment slope: {model.params['YEAR_CENTERED']:.4f}")
    if 'RD_INTERACTION' in model.params:
        print(f"  Change in slope: {model.params['RD_INTERACTION']:.4f}")
    
    return model, df

def create_rd_visualization(df, output_dir):
    """
    Create RD visualization plot showing discontinuity at 2017 cutoff.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with RD variables
    output_dir : str
        Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate mean screening rates by year
    yearly_means = df.groupby('YEAR')['DEV_SCREEN'].agg(['mean', 'count']).reset_index()
    yearly_means.columns = ['YEAR', 'screening_rate', 'n']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot points with size proportional to sample size
    plt.scatter(yearly_means['YEAR'], yearly_means['screening_rate'], 
               s=yearly_means['n']/50, alpha=0.7, color='blue')
    
    # Add vertical line at cutoff
    plt.axvline(x=2017, linestyle='--', color='red', alpha=0.8, linewidth=2,
               label='2017 Policy Implementation')
    
    # Fit separate trend lines
    pre_2017 = yearly_means[yearly_means['YEAR'] < 2017]
    post_2017 = yearly_means[yearly_means['YEAR'] >= 2017]
    
    if len(pre_2017) > 1:
        z_pre = np.polyfit(pre_2017['YEAR'], pre_2017['screening_rate'], 1)
        p_pre = np.poly1d(z_pre)
        plt.plot(pre_2017['YEAR'], p_pre(pre_2017['YEAR']), "r--", alpha=0.8, 
                linewidth=2, label='Pre-2017 trend')
    
    if len(post_2017) > 1:
        z_post = np.polyfit(post_2017['YEAR'], post_2017['screening_rate'], 1)
        p_post = np.poly1d(z_post)
        plt.plot(post_2017['YEAR'], p_post(post_2017['YEAR']), "g--", alpha=0.8, 
                linewidth=2, label='Post-2017 trend')
    
    plt.xlabel('Survey Year')
    plt.ylabel('Developmental Screening Rate')
    plt.title('Regression Discontinuity: Screening Rates Around 2017 Policy Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "rd_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"RD visualization saved to: {output_dir}/rd_visualization.png")

def create_regression_table(models, output_dir):
    """
    Create clean regression table without individual state fixed effects.
    
    This function creates a publication-ready LaTeX table showing results from
    all three models while hiding the individual state coefficients from the
    fixed effects model to maintain readability.
    
    Parameters
    ----------
    models : list
        List of fitted regression models [OLS, State_FE, RD]
    output_dir : str
        Directory to save the LaTeX table file
        
    Returns
    -------
    str or None
        LaTeX table string if successful, None if failed
        
    Notes
    -----
    The function:
    1. Uses Stargazer to create professional LaTeX tables
    2. Hides individual state coefficients (C(STATE)[T.2], etc.)
    3. Keeps the "State Fixed Effects: Yes/No" line for clarity
    4. Uses professional variable labels
    5. Falls back to manual table creation if Stargazer fails
    """
    print("\n" + "="*60)
    print("CREATING LATEX TABLE")
    print("="*60)
    
    try:
        # Initialize stargazer
        stargazer = Stargazer(models)
        stargazer.title("Regression Results: Impact of 2017 Policy on Developmental Screening")
        stargazer.custom_columns(["OLS", "State FE", "Regression Discontinuity"], [1, 1, 1])
        
        # Hide individual state fixed effects coefficients
        # Get all coefficient names from the state FE model (model 2)
        if len(models) > 1 and models[1] is not None:
            state_fe_coeffs = [param for param in models[1].params.index 
                             if param.startswith('C(STATE)')]
            
            # Remove state fixed effects coefficients from table
            if state_fe_coeffs:
                stargazer.remove_covariates(state_fe_coeffs)
                print(f"✓ Hiding {len(state_fe_coeffs)} individual state coefficients")
        
        # Add informational line about state fixed effects
        stargazer.add_line('State Fixed Effects', ['No', 'Yes', 'No'])
        
        # Professional variable labels
        stargazer.rename_covariates({
            'POST2017': 'Post-2017 Policy', 
            'PRIVATE_INSURANCE': 'Private Insurance', 
            'PUBLIC_INSURANCE': 'Public Insurance',
            'YEAR_CENTERED': 'Year Centered',
            'RD_INTERACTION': 'Post-2017 × Year',
            'FEMALE': 'Female',
            'MALE': 'Male',
            'BLACK': 'Black',
            'OTHER_RACE': 'Other Race',
            'HIGH_EDUCATION': 'High Education',
            'URBAN': 'Urban',
            'AGE_MONTHS': 'Age (Months)',
            'Intercept': 'Constant'
        })
        
        # Additional formatting options
        stargazer.significance_levels([0.1, 0.05, 0.01])
        stargazer.show_degrees_of_freedom(False)
        stargazer.show_model_numbers(True)
        
        # Generate and save table
        latex_table = stargazer.render_latex()
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "regression_table.tex"), 'w') as f:
            f.write(latex_table)
        
        print("✓ LaTeX table saved to: regression_table.tex")
        print("✓ Individual state coefficients excluded from publication table")
        return latex_table
        
    except Exception as e:
        print(f"❌ Error creating Stargazer table: {e}")
        print("Attempting manual table creation as fallback...")
        
        # Fallback: Create simple manual table
        return create_simple_manual_table(models, output_dir)

def create_simple_manual_table(models, output_dir):
    """
    Create a simple LaTeX table manually as fallback when Stargazer fails.
    
    Parameters
    ----------
    models : list
        List of fitted regression models
    output_dir : str
        Directory to save the table
        
    Returns
    -------
    str
        LaTeX table string
    """
    print("Creating manual LaTeX table...")
    
    # Extract key coefficients (excluding individual state FE)
    key_vars = ['POST2017', 'PRIVATE_INSURANCE', 'PUBLIC_INSURANCE', 
                'FEMALE', 'MALE', 'BLACK', 'OTHER_RACE', 'HIGH_EDUCATION', 
                'URBAN', 'AGE_MONTHS', 'YEAR_CENTERED', 'RD_INTERACTION', 'Intercept']
    
    var_labels = {
        'POST2017': 'Post-2017 Policy',
        'PRIVATE_INSURANCE': 'Private Insurance',
        'PUBLIC_INSURANCE': 'Public Insurance', 
        'FEMALE': 'Female',
        'MALE': 'Male',
        'BLACK': 'Black',
        'OTHER_RACE': 'Other Race',
        'HIGH_EDUCATION': 'High Education',
        'URBAN': 'Urban',
        'AGE_MONTHS': 'Age (Months)',
        'YEAR_CENTERED': 'Year Centered',
        'RD_INTERACTION': 'Post-2017 × Year',
        'Intercept': 'Constant'
    }
    
    # Build table content
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Regression Results: Impact of 2017 Policy on Developmental Screening}",
        "\\begin{tabular}{lccc}",
        "\\hline\\hline",
        " & (1) OLS & (2) State FE & (3) RD Design \\\\",
        "\\hline",
    ]
    
    # Add coefficient rows (excluding individual state FE)
    for var in key_vars:
        if any(var in model.params.index for model in models if model is not None):
            label = var_labels.get(var, var)
            coef_row = f"{label} & "
            se_row = " & "
            
            for model in models:
                if model is not None and var in model.params.index:
                    coef = model.params[var]
                    se = model.bse[var] 
                    pval = model.pvalues[var]
                    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                    
                    coef_row += f"{coef:.3f}{stars} & "
                    se_row += f"({se:.3f}) & "
                else:
                    coef_row += " & "
                    se_row += " & "
            
            # Clean up trailing &
            coef_row = coef_row.rstrip(' & ') + " \\\\"
            se_row = se_row.rstrip(' & ') + " \\\\"
            
            table_lines.extend([coef_row, se_row])
    
    # Add model statistics
    table_lines.extend([
        "\\hline",
        "State Fixed Effects & No & Yes & No \\\\",
    ])
    
    # Add N and R-squared
    n_row = "Observations & "
    r2_row = "R-squared & "
    
    for model in models:
        if model is not None:
            n_row += f"{int(model.nobs):,} & "
            r2_row += f"{model.rsquared:.3f} & "
        else:
            n_row += " & "
            r2_row += " & "
    
    n_row = n_row.rstrip(' & ') + " \\\\"
    r2_row = r2_row.rstrip(' & ') + " \\\\"
    
    table_lines.extend([n_row, r2_row])
    
    # Close table
    table_lines.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item \\textit{Notes:} Heteroskedasticity-robust standard errors in parentheses.",
        "\\item State fixed effects included in Model 2 but individual coefficients not reported.",
        "\\item * p$<$0.10, ** p$<$0.05, *** p$<$0.01",
        "\\end{tablenotes}",
        "\\end{table}"
    ])
    
    # Save table
    table_content = '\n'.join(table_lines)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "regression_table.tex"), 'w') as f:
        f.write(table_content)
    
    print("✓ Manual LaTeX table created successfully")
    print("✓ Individual state coefficients excluded from table")
    return table_content

def summarize_policy_effects(models):
    """
    Extract and summarize key policy effects from all models.
    
    This function provides a clear summary of the main findings by extracting
    the POST2017 coefficients from each model and presenting them in an
    easily interpretable format for policy analysis.
    
    Parameters
    ----------
    models : list
        List of fitted regression models [OLS, FE, RD]
        
    Returns
    -------
    None
        Prints formatted summary of policy effects
        
    Notes
    -----
    The summary includes:
    - Point estimates for POST2017 coefficient
    - Standard errors (heteroskedasticity-robust)
    - P-values for statistical significance
    - Percentage point interpretation
    
    Model Comparison:
    - Basic OLS: Baseline estimate (potential confounding)
    - State FE: Controls for state-level heterogeneity
    - RD: Most credible causal estimate (under assumptions)
    
    Examples
    --------
    >>> summarize_policy_effects([model1, model2, model3])
    
    KEY POLICY EFFECTS (POST2017 COEFFICIENTS)
    ============================================================
    
    Basic OLS:
      Coefficient: 0.0442
      Std Error: 0.0078
      P-value: 0.000
      Interpretation: 4.4 percentage point increase in screening
    """
    print("\n" + "="*60)
    print("KEY POLICY EFFECTS (POST2017 COEFFICIENTS)")
    print("="*60)
    
    model_names = ['Basic OLS', 'State FE', 'RD Design']
    
    for i, (name, model) in enumerate(zip(model_names, models)):
        if model is not None and 'POST2017' in model.params.index:
            coef = model.params['POST2017']
            se = model.bse['POST2017']
            pval = model.pvalues['POST2017']
            
            print(f"\n{name}:")
            print(f"  Coefficient: {coef:.4f}")
            print(f"  Std Error: {se:.4f}")
            print(f"  P-value: {pval:.3f}")
            print(f"  Significance: {'***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else 'Not significant'}")
            print(f"  Interpretation: {coef*100:.1f} percentage point change in screening rate")
            
            # Add RD-specific output for model 3
            if i == 2 and 'YEAR_CENTERED' in model.params.index:  # RD model
                year_coef = model.params['YEAR_CENTERED']
                rd_int_coef = model.params.get('RD_INTERACTION', 0)
                print(f"  Pre-treatment trend: {year_coef:.4f}")
                print(f"  Slope change post-2017: {rd_int_coef:.4f}")
        else:
            print(f"\n{name}: Model estimation failed")

def main():
    """
    Execute the complete regression analysis pipeline with RD design.
    
    This is the main function that orchestrates the entire econometric analysis:
    1. Loads the cleaned NSCH dataset
    2. Estimates all three regression models (OLS, State FE, RD)
    3. Creates RD visualization
    4. Creates publication-ready output tables
    5. Summarizes key policy findings
    
    The function handles the complete workflow from data loading to final output
    generation, making it easy to reproduce the entire analysis with a single
    function call.
    
    Returns
    -------
    tuple
        (models_list, enhanced_dataframe) where:
        - models_list: List of fitted regression models
        - enhanced_dataframe: Dataset with RD variables added
        
    Raises
    ------
    FileNotFoundError
        If the cleaned dataset is not available
    Exception
        If any of the regression models fail to converge
        
    Notes
    -----
    Expected Runtime: ~30-60 seconds depending on system
    Memory Usage: ~200MB for full NSCH dataset
    Output Files Generated:
    - regression_table.tex (LaTeX table)
    - rd_visualization.png (RD plot)
    - Console output with all model summaries
    
    Examples
    --------
    >>> models, df = main()
    Data loaded: 20,171 rows
    
    ============================================================
    MODEL 1: BASIC OLS
    ============================================================
    [Model output...]
    
    Regression analysis completed!
    """
    print("=" * 70)
    print("🏥 NSCH DEVELOPMENTAL SCREENING ANALYSIS")
    print("📊 Regression Analysis Pipeline")
    print("=" * 70)
    
    # Load analysis-ready dataset
    df = load_analysis_data()
    
    # Estimate all three models
    print("\n🔄 ESTIMATING REGRESSION MODELS...")
    print("-" * 40)
    
    model1 = estimate_basic_ols(df)
    model2 = estimate_state_fixed_effects(df)
    model3, df_enhanced = estimate_regression_discontinuity(df)
    
    # Set up output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(current_dir, "results")
    
    # Create RD visualization
    print("\n📈 CREATING RD VISUALIZATION...")
    create_rd_visualization(df_enhanced, results_dir)
    
    # Create publication table (with state FE coefficients hidden)
    print("\n📋 CREATING PUBLICATION TABLE...")
    models = [model1, model2, model3]
    create_regression_table(models, results_dir)
    
    # Summarize key findings
    summarize_policy_effects(models)
    
    # Final summary
    print(f"\n" + "="*70)
    print(f"✅ REGRESSION ANALYSIS COMPLETED!")
    print(f"📊 All 3 models estimated successfully")
    print(f"📄 Results saved to: {results_dir}")
    print(f"📋 Publication table excludes individual state coefficients")
    print("="*70)
    
    return models, df_enhanced

# Execute analysis when script is run directly
if __name__ == "__main__":
    models, df = main()
