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
    """Create clean regression table without individual state fixed effects."""
    print("\n" + "="*60)
    print("CREATING LATEX TABLE")
    print("="*60)
    
    try:
        # Initialize stargazer
        stargazer = Stargazer(models)
        stargazer.title("Regression Results: Developmental Screening")
        stargazer.custom_columns(["OLS", "State FE", "Regression Discontinuity"], [1, 1, 1])
        
        # Hide individual state fixed effects (keep only the note)
        stargazer.add_line('State Fixed Effects', ['No', 'Yes', 'No'])
        
        # Professional variable labels
        stargazer.rename_covariates({
            'POST2017': 'Post-2017', 
            'PRIVATE_INSURANCE': 'Private Insurance', 
            'PUBLIC_INSURANCE': 'Public Insurance',
            'YEAR_CENTERED': 'Year Centered',
            'RD_INTERACTION': 'RD Interaction',
            'FEMALE': 'Female',
            'BLACK': 'Black',
            'OTHER_RACE': 'Other Race',
            'HIGH_EDUCATION': 'High Education',
            'URBAN': 'Urban',
            'AGE_MONTHS': 'Age (Months)'
        })
        
        # Generate and save table
        latex_table = stargazer.render_latex()
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "regression_table.tex"), 'w') as f:
            f.write(latex_table)
        
        print("LaTeX table saved!")
        return latex_table
        
    except Exception as e:
        print(f"Error creating LaTeX table: {e}")
        return None

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
    
    model_names = ['Basic OLS', 'State FE', 'RD Design']  # Updated model name
    
    for i, (name, model) in enumerate(zip(model_names, models)):
        if 'POST2017' in model.params.index:
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
    # Load analysis-ready dataset
    df = load_analysis_data()
    
    # Estimate all three models
    print("Estimating regression models...")
    model1 = estimate_basic_ols(df)
    model2 = estimate_state_fixed_effects(df)
    model3, df_enhanced = estimate_regression_discontinuity(df)  # Using RD
    
    # Create RD visualization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(current_dir, "results")
    create_rd_visualization(df_enhanced, results_dir)
    
    # Create publication table
    models = [model1, model2, model3]
    create_regression_table(models, results_dir)
    
    # Summarize key findings
    summarize_policy_effects(models)
    
    print("\nRegression analysis completed!")
    return models, df_enhanced

# Execute analysis when script is run directly
if __name__ == "__main__":
    models, df = main()
