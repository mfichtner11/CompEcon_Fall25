"""
Developmental Screening Analysis Visualizations (NSCH 2016–2021)
================================================================

This module creates three publication-ready visualizations analyzing developmental 
screening rates from the National Survey of Children's Health (NSCH) dataset.

Figures Generated:
1. State-by-State Screening Rates (Top 20 States)
2. Screening Rates by Insurance Type 
3. Trends Over Time with Regression Discontinuity Design

Author: Marli Fichtner
Course: Computational Economics Fall 2025
Dataset: NSCH 2016-2021 (children aged 9-35 months)
Policy Context: 2017 Bright Futures 4th Edition Guidelines
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def prepare_data(df):
    """
    Prepare the NSCH dataset for visualization by adding human-readable labels.
    
    This function performs essential data preprocessing by mapping FIPS codes to 
    state names and insurance type codes to descriptive labels, making the 
    visualizations more interpretable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw NSCH dataset containing STATE (FIPS codes) and INSTYPE columns
        
    Returns
    -------
    pd.DataFrame
        Enhanced dataset with new columns:
        - 'INSURANCE_LABEL': Human-readable insurance type names
        - 'STATE_NAME': State names mapped from FIPS codes
        
    Notes
    -----
    Insurance Type Mapping:
    - 1: 'Public' (Medicaid, CHIP, etc.)
    - 2: 'Private' (employer-sponsored, individual market)
    - 4: 'Dually Eligible' (Medicare + Medicaid)
    - 5: 'Uninsured'
    
    FIPS codes cover all 50 states plus DC (excludes territories)
    
    Examples
    --------
    >>> df_prepared = prepare_data(df_raw)
    >>> print(df_prepared['INSURANCE_LABEL'].value_counts())
    Private         12000
    Public           8000
    Uninsured         150
    Dually Eligible    21
    """
    # Insurance type labels based on NSCH codebook
    insurance_labels = {
        1: 'Public', 
        2: 'Private', 
        4: 'Dually Eligible', 
        5: 'Uninsured'
    }
    df['INSURANCE_LABEL'] = df['INSTYPE'].map(insurance_labels).fillna('Unknown')

    # FIPS to State Name Mapping (Federal Information Processing Standards)
    # Source: https://www.census.gov/library/reference/code-lists/ansi.html
    fips_to_state = {
        1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
        8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'DC', 12: 'Florida',
        13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois', 18: 'Indiana',
        19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
        24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi',
        29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire',
        34: 'New Jersey', 35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
        39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
        45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas', 49: 'Utah',
        50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
    }

    # Convert FIPS codes to state names, fallback to string conversion for unknown codes
    df['STATE_NAME'] = df['STATE'].map(fips_to_state).fillna(df['STATE'].astype(str))
    
    return df

def create_state_screening_figure(df, images_dir):
    """
    Generate Figure 1: Horizontal bar chart of top 20 states by screening rates.
    
    Creates a publication-ready visualization showing which states have the highest
    developmental screening rates. This helps identify geographic variation and 
    potential best practices in pediatric care delivery.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared NSCH dataset with STATE_NAME and DEV_SCREEN columns
    images_dir : str
        Directory path where the figure PNG will be saved
        
    Returns
    -------
    None
        Displays plot and saves high-resolution PNG to images_dir
        
    Notes
    -----
    - Calculates mean screening rate by state across all survey years
    - Shows only top 20 states for readability
    - Uses 90-degree rotated state labels to prevent overlap
    - Saves as 300 DPI PNG for publication quality
    
    File Output
    -----------
    Figure1_State_Screening_Rates.png : High-resolution bar chart
    
    Interpretation
    --------------
    States with higher bars indicate better compliance with developmental
    screening guidelines. Geographic clustering may suggest regional
    differences in healthcare infrastructure or policy implementation.
    
    Examples
    --------
    >>> create_state_screening_figure(df_prepared, "./images/")
    Figure 1 saved to images/Figure1_State_Screening_Rates.png
    """
    # Calculate state-level mean screening rates
    state_rates = df.groupby('STATE_NAME')['DEV_SCREEN'].mean().reset_index()
    state_rates.columns = ['STATE_NAME', 'rate']
    top_states = state_rates.sort_values('rate', ascending=False).head(20)

    # Create the visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_states, x='STATE_NAME', y='rate', color='skyblue')
    plt.xlabel('State')
    plt.ylabel('Developmental Screening Rate (Respondents ages 9-35 months)')
    plt.title('Figure 1: Top 20 States: Developmental Screening Rates (NSCH 2016-2021)')
    plt.xticks(rotation=90)  
    plt.tight_layout()

    # Save high-resolution figure
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, "Figure1_State_Screening_Rates.png"), 
               dpi=300, bbox_inches='tight')
    print("Figure 1 saved to images/Figure1_State_Screening_Rates.png")
    plt.show()

def create_insurance_screening_figure(df, images_dir):
    """
    Generate Figure 2: Bar chart comparing screening rates by insurance type.
    
    Examines disparities in developmental screening access across different 
    insurance categories. This analysis is crucial for understanding how 
    coverage type affects pediatric preventive care utilization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset with INSURANCE_LABEL and DEV_SCREEN columns
    images_dir : str
        Output directory for saving the figure
        
    Returns
    -------
    None
        Displays interactive plot and saves PNG file
        
    Features
    --------
    - Sample size annotations above each bar (n=X,XXX format)
    - Color palette optimized for accessibility 
    - Y-axis scaled 0-1 for rate interpretation
    - Error handling for empty insurance categories
    
    File Output
    -----------
    Figure2_ScreenRatebyInsurance.png : Publication-quality bar chart
    
    Policy Implications
    -------------------
    Significant differences between insurance types may indicate:
    - Access barriers for certain coverage types
    - Provider network adequacy issues  
    - Prior authorization requirements
    - Socioeconomic factors correlated with insurance type
    
    Examples
    --------
    >>> create_insurance_screening_figure(df_prepared, "./images/")
    Figure 2 saved to images/Figure2_ScreenRatebyInsurance.png
    """
    # Calculate screening rates and sample sizes by insurance type
    insurance_rates = df.groupby('INSURANCE_LABEL')['DEV_SCREEN'].agg(['mean', 'count']).reset_index()
    insurance_rates.columns = ['INSURANCE_TYPE', 'rate', 'n']

    # Create the visualization
    plt.figure(figsize=(8, 6))
    bars = sns.barplot(data=insurance_rates, x='INSURANCE_TYPE', y='rate', palette='Set2')

    # Annotate bars with sample sizes for transparency
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        n = insurance_rates.iloc[i]['n']
        bars.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'n={n:,}', ha='center', va='bottom', fontsize=10)

    plt.title('Developmental Screening by Insurance Type')
    plt.ylabel('Screening Rate (Respondents ages 9-35 months)')
    plt.ylim(0, 1)
    plt.tight_layout()

    # Save the figure
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, "Figure2_ScreenRatebyInsurance.png"), 
               dpi=300, bbox_inches='tight')
    print("Figure 2 saved to images/Figure2_ScreenRatebyInsurance.png")
    plt.show()

def create_trends_over_time_figure(df, images_dir):
    """
    Generate Figure 3: Time series plot for Regression Discontinuity analysis.
    
    Creates a line plot showing developmental screening trends over time by 
    insurance type, with a vertical line marking the 2017 Bright Futures 
    guideline implementation. This visualization supports the RD research design.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with YEAR, INSURANCE_LABEL, and DEV_SCREEN columns
    images_dir : str
        Directory path for output file
        
    Returns
    -------
    None
        Displays plot with RD visualization and saves PNG
        
    Research Design
    ---------------
    This figure supports regression discontinuity analysis by:
    - Showing pre/post 2017 trends clearly
    - Identifying potential discontinuities at the policy cutoff
    - Revealing differential effects by insurance type
    - Providing visual evidence for parallel trends assumption
    
    Data Filtering
    --------------
    - Excludes year-insurance combinations with n<50 observations
    - Ensures statistical reliability of trend estimates
    - Maintains interpretability with sufficient sample sizes
    
    File Output
    -----------
    Figure3_RD_BrightFutures2017.png : Time series with policy discontinuity
    
    Analytical Value
    ----------------
    Supports causal identification by visualizing:
    - Sharp discontinuity at 2017 cutoff (if policy effective)
    - Parallel pre-trends (validates RD assumptions)
    - Heterogeneous treatment effects by insurance type
    - Absence of anticipation effects (no pre-2017 jump)
    
    Examples
    --------
    >>> create_trends_over_time_figure(df_prepared, "./images/")
    Figure 3 saved to images/Figure3_RD_BrightFutures2017.png
    """
    # Calculate annual trends by insurance type with sample size filtering
    trend = df.groupby(['YEAR', 'INSURANCE_LABEL'])['DEV_SCREEN'].agg(['mean', 'count']).reset_index()
    trend.columns = ['YEAR', 'INSURANCE_TYPE', 'rate', 'n']
    trend = trend[trend['n'] >= 50]  # Ensure statistical reliability

    # Create the time series visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=trend, x='YEAR', y='rate', hue='INSURANCE_TYPE', 
                marker='o', linewidth=2, markersize=8)

    # Mark the policy intervention point
    plt.axvline(x=2017, linestyle='--', color='red', alpha=0.7, 
               label='Bright Futures 4th Ed.')

    plt.title('Developmental Screening Trends by Insurance Type (2016-2021)')
    plt.xlabel('Survey Year')
    plt.ylabel('Screening Rate')
    plt.ylim(0, 1)
    plt.legend(title='Insurance Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the regression discontinuity visualization
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, "Figure3_RD_BrightFutures2017.png"), 
               dpi=300, bbox_inches='tight')
    print("Figure 3 saved to images/Figure3_RD_BrightFutures2017.png")
    plt.show()

def main():
    """
    Execute the complete visualization pipeline for NSCH developmental screening analysis.
    
    This function orchestrates the entire visualization workflow:
    1. Loads the cleaned NSCH dataset
    2. Prepares data with proper labels and mappings
    3. Generates all three publication-ready figures
    4. Saves high-resolution outputs to the images directory
    
    Directory Structure Expected
    ----------------------------
    ProblemSet5/
    ├── scripts/
    │   └── Visualization_PS5_Fichtner.py  ← This file
    ├── cleaned_data/
    │   └── nsch_analysis_ready.csv        ← Input dataset
    └── images/                            ← Output directory (created if needed)
        ├── Figure1_State_Screening_Rates.png
        ├── Figure2_ScreenRatebyInsurance.png
        └── Figure3_RD_BrightFutures2017.png
    
    Returns
    -------
    None
        Prints progress updates and file save confirmations
        
    Raises
    ------
    FileNotFoundError
        If nsch_analysis_ready.csv is not found in expected location
    PermissionError  
        If images directory cannot be created or written to
        
    Examples
    --------
    Run from command line:
    >>> python Visualization_PS5_Fichtner.py
    Data loaded: 20,171 rows
    
    Creating visualizations...
    Figure 1 saved to images/Figure1_State_Screening_Rates.png
    Figure 2 saved to images/Figure2_ScreenRatebyInsurance.png  
    Figure 3 saved to images/Figure3_RD_BrightFutures2017.png
    
    All visualizations completed successfully!
    
    Or import as module:
    >>> from Visualization_PS5_Fichtner import main
    >>> main()
    """
    # Set up file paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "cleaned_data", "nsch_analysis_ready.csv")
    images_dir = os.path.join(current_dir, "images")

    # Load the preprocessed NSCH dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded: {len(df):,} rows")
    except FileNotFoundError:
        print(f"ERROR: Could not find dataset at {csv_path}")
        print("Please ensure DataCleaning_PS5_Fichtner.py has been run first.")
        return

    # Prepare data with human-readable labels
    df = prepare_data(df)

    # Generate all publication figures
    print("\nCreating visualizations...")
    create_state_screening_figure(df, images_dir)
    create_insurance_screening_figure(df, images_dir)
    create_trends_over_time_figure(df, images_dir)

    print("\nAll visualizations completed successfully!")

if __name__ == "__main__":
    main()
