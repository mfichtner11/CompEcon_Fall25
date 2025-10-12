#Create variables
"""
Variable Creation Module for NSCH Developmental Screening Analysis
=================================================================

This module creates and transforms variables from the National Survey of Children's 
Health (NSCH) dataset for developmental screening analysis. It implements the official 
HRSA methodology for computing developmental screening indicators and creates 
additional variables needed for econometric analysis.

Key Variables Created:
- POST2017: Policy treatment indicator (Bright Futures 4th Edition)
- DEV_SCREEN: Binary developmental screening outcome (HRSA methodology)
- TARGET_AGE: Age group indicator (9-35 months)
- AGE_MONTHS: Precise age in months

Author: Marli Fichtner
Course: Computational Economics Fall 2025
Dataset: NSCH 2016-2021
Policy Context: 2017 Bright Futures Guidelines Implementation
"""

import pandas as pd
import numpy as np
import os
import sys

def create_variables(df):
    """
    Create analysis variables from raw NSCH data following official methodology.
    
    This function transforms the raw NSCH dataset by creating key variables needed
    for developmental screening analysis. It handles flexible column matching to
    accommodate different NSCH data versions and implements the official HRSA
    methodology for developmental screening indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw NSCH dataset with original variable names (e.g., 'year', 'sc_sex', etc.)
        Must contain survey years 2016-2021 and children's health data
        
    Returns
    -------
    pd.DataFrame
        Enhanced dataset with additional analysis variables:
        - YEAR: Survey year (integer)
        - POST2017: Treatment indicator (1 if year ≥ 2017, 0 otherwise)
        - INSTYPE: Insurance type code (1=Public, 2=Private, 4=Dual, 5=Uninsured)
        - STATE: State FIPS code
        - SEX: Child's sex (1=Male, 2=Female)
        - RACEETH: Race/ethnicity code
        - HEDUC: Household education level
        - METRO: Metropolitan area indicator
        - AGE_MONTHS: Child's age in months (estimated)
        - TARGET_AGE: 1 if child is 9-35 months old, 0 otherwise
        - DEV_SCREEN: Binary screening outcome (1=screened, 0=not screened)
    """
    
    print("Creating analysis variables from raw NSCH data...")
    print(f"Input dataset: {len(df):,} rows, {len(df.columns)} columns")
    
    # Column name mapping for flexible matching across NSCH versions
    col_map = {col.lower(): col for col in df.columns}
    
    def find_cols(patterns):
        """Find columns matching any of the specified patterns."""
        matches = []
        for pattern in patterns:
            matches.extend([col for col_lower, col in col_map.items() if pattern in col_lower])
        return matches[:1]  # Return first match only
        
    # 1. Create temporal variables for policy analysis
    if 'year' in df.columns:
        df['YEAR'] = df['year'].astype(int)
        df['POST2017'] = (df['YEAR'] >= 2017).astype(int)
        print(f"   ✓ Created YEAR and POST2017 (policy treatment indicator)")
    else:
        print(f"   ✗ WARNING: 'year' column not found")
        df['YEAR'] = np.nan
        df['POST2017'] = np.nan
    
    # 2. Map raw NSCH variables to analysis variables
    variable_mappings = {
        'INSTYPE': ['instype'],           # Insurance type
        'STATE': ['fipsst'],              # State FIPS code  
        'SEX': ['sc_sex'],                # Child's sex
        'RACEETH': ['sc_racer'],          # Race/ethnicity
        'HEDUC': ['higrade'],             # Household education
        'METRO': ['metro_yn'],            # Metropolitan status
    }
    
    for var_name, patterns in variable_mappings.items():
        cols = find_cols(patterns)
        if cols:
            df[var_name] = df[cols[0]]
            print(f"   ✓ Created {var_name} from {cols[0]}")
        else:
            df[var_name] = np.nan
            print(f"   ⚠ WARNING: Could not find variable for {var_name}")
        
    # 3. Create precise age variables for target population identification
    if 'sc_age_years' in df.columns:
        df['AGE_YEARS'] = df['sc_age_years']
        
        # Handle children under 9 months separately
        if 'sc_age_lt9' in df.columns:
            df['AGE_LT9'] = df['sc_age_lt9']
            # Estimate months: 4 months for <9 month olds, years*12+6 for others
            df['AGE_MONTHS'] = np.where(df['AGE_LT9'] == 1, 
                                       4,  # Conservative estimate for <9 month group
                                       df['AGE_YEARS'] * 12 + 6)  # Midpoint of year
        else:
            df['AGE_MONTHS'] = df['AGE_YEARS'] * 12 + 6
            
        print(f"   ✓ Created AGE_MONTHS (estimated from years with midpoint assumption)")
        
        # Create target age group indicator (AAP screening guidelines: 9, 18, 30 months)
        df["TARGET_AGE"] = np.where((df["AGE_MONTHS"] >= 9) & (df["AGE_MONTHS"] <= 35), 1, 0)
        print(f"   ✓ Created TARGET_AGE (1 if 9-35 months, 0 if outside range)")
        
        # Show age distribution
        target_count = df["TARGET_AGE"].sum()
        print(f"   → {target_count:,} children in target age range (9-35 months)")
    else:
        df['AGE_MONTHS'] = np.nan
        df['TARGET_AGE'] = np.nan
        print(f"   ⚠ WARNING: Could not find age variables (sc_age_years)")
        
    # 4. Create developmental screening variables using official HRSA methodology
    df = compute_dev_screening_variables(df)
    
    print(f"Enhanced dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df

def compute_dev_screening_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute developmental screening indicators using official HRSA NSCH methodology.
    """

    df = df.copy()
    n = len(df)
    
    print("   Computing developmental screening variables...")
    
    # Column name mapping for flexible matching across NSCH data versions
    col_map = {col.lower(): col for col in df.columns}
    
    def find_cols(patterns):
        """Find columns matching any of the specified patterns."""
        matches = []
        for pattern in patterns:
            matches.extend([col for col_lower, col in col_map.items() if pattern in col_lower])
        return matches[:1]  # Return first match only
    
    # Locate K6Q variables using flexible pattern matching
    k6q12_cols = find_cols(['k6q12'])    # Developmental delay diagnosis
    k6q13a_cols = find_cols(['k6q13a'])  # 9-23 month screening question A
    k6q13b_cols = find_cols(['k6q13b'])  # 9-23 month screening question B
    k6q14a_cols = find_cols(['k6q14a'])  # 24-35 month screening question A
    k6q14b_cols = find_cols(['k6q14b'])  # 24-35 month screening question B
    
    # Validate that all required K6Q variables are present
    required_cols = {
        'K6Q12': k6q12_cols,   
        'K6Q13A': k6q13a_cols, 
        'K6Q13B': k6q13b_cols, 
        'K6Q14A': k6q14a_cols, 
        'K6Q14B': k6q14b_cols  
    }
    
    missing_cols = [col for col, matches in required_cols.items() if not matches]
    
    if missing_cols:
        print(f"   ⚠ WARNING: Missing K6Q columns: {missing_cols}")
        available_k6q = [col for col in df.columns if 'k6q' in col.lower()]
        print(f"   → Available K6Q columns: {available_k6q}")
        # Create placeholder DEV_SCREEN variable with missing values
        df['DEV_SCREEN'] = np.nan
        df['DevScrnng'] = np.nan
        df['both_9to23'] = np.nan
        df['both_24to35'] = np.nan
        print("   ✗ Created DEV_SCREEN with missing values due to missing K6Q columns")
        return df
    
    # Extract actual column names for use in logic
    k6q12_col = k6q12_cols[0]
    k6q13a_col = k6q13a_cols[0]
    k6q13b_col = k6q13b_cols[0]
    k6q14a_col = k6q14a_cols[0]
    k6q14b_col = k6q14b_cols[0]
    
    # Initialize age-specific screening indicators
    both_9to23 = np.full(n, np.nan, dtype=object)   
    both_24to35 = np.full(n, np.nan, dtype=object)  

    # Helper function to check for positive responses or missing applicable data
    in_1_or_M = lambda s: (s == 1) | (s == ".M")

    # Check if we have the age variables needed
    if 'AGE_YEARS' not in df.columns:
        print("   ⚠ WARNING: AGE_YEARS not found, cannot compute age-specific screening")
        df['DEV_SCREEN'] = np.nan
        return df
        
    # Create AGE_LT9 if it doesn't exist
    if 'AGE_LT9' not in df.columns:
        df['AGE_LT9'] = np.where(df['AGE_YEARS'] < 1, 1, 0)

    # ========== BOTH_9TO23: Screening Logic for 9-23 Month Age Group ==========
    # Not screened if K6Q12=No and age<2, or if either screening question=No
    both_9to23[(df[k6q12_col] == 2) & (df["AGE_YEARS"] < 2)] = 2
    both_9to23[(df[k6q13a_col] == 2) | (df[k6q13b_col] == 2)] = 2
    
    # Screened if both screening questions are Yes or Missing-but-applicable
    both_9to23[in_1_or_M(df[k6q13a_col]) & in_1_or_M(df[k6q13b_col])] = 1
    
    # Missing data if both screening questions are missing
    both_9to23[(df[k6q13a_col] == ".M") & (df[k6q13b_col] == ".M")] = ".M"
    
    # Logically skipped if outside target age range
    both_9to23[(df["AGE_YEARS"] > 1) & (df["AGE_YEARS"] < 6)] = ".L"
    both_9to23[df["AGE_LT9"] == 1] = ".L"  
    
    # Not applicable if too old
    both_9to23[df["AGE_YEARS"] > 5] = ".N"

    # ========== BOTH_24TO35: Screening Logic for 24-35 Month Age Group ==========
    # Not screened if K6Q12=No and age=2, or if either screening question=No
    both_24to35[(df[k6q12_col] == 2) & (df["AGE_YEARS"] == 2)] = 2
    both_24to35[(df[k6q14a_col] == 2) | (df[k6q14b_col] == 2)] = 2
    
    # Screened if both screening questions are Yes or Missing-but-applicable
    both_24to35[in_1_or_M(df[k6q14a_col]) & in_1_or_M(df[k6q14b_col])] = 1
    
    # Missing data if both screening questions are missing
    both_24to35[(df[k6q14a_col] == ".M") & (df[k6q14b_col] == ".M")] = ".M"
    
    # Logically skipped if outside target age range
    both_24to35[(df["AGE_YEARS"] < 2) | ((df["AGE_YEARS"] > 2) & (df["AGE_YEARS"] < 6))] = ".L"
    
    # Not applicable if too old
    both_24to35[df["AGE_YEARS"] > 5] = ".N"

    # Store intermediate age-specific indicators
    df["both_9to23"] = both_9to23
    df["both_24to35"] = both_24to35

    # ========== DEVSCRNNG: Overall Developmental Screening Status ==========
    DevScrnng = np.full(n, np.nan, dtype=object)

    # Screened if either age group was screened
    DevScrnng[(df["both_9to23"] == 1) | (df["both_24to35"] == 1)] = 1
    
    # Not screened if either age group was not screened
    DevScrnng[(df["both_9to23"] == 2) | (df["both_24to35"] == 2)] = 2
    
    # Missing if any relevant questions have missing data
    DevScrnng[(df["both_9to23"] == ".M") | (df["both_24to35"] == ".M")] = ".M"
    DevScrnng[df[k6q12_col] == ".M"] = ".M"
    
    # Logically skipped for certain age groups
    DevScrnng[(df["AGE_LT9"] == 1) | ((df["AGE_YEARS"] > 2) & (df["AGE_YEARS"] < 6))] = ".L"
    
    # Not applicable if too old
    DevScrnng[df["AGE_YEARS"] > 5] = ".N"

    df["DevScrnng"] = DevScrnng
    
    # ========== DEV_SCREEN: Binary Analysis Variable ==========
    df['DEV_SCREEN'] = np.where(df['DevScrnng'] == 1, 1,        # Screened = 1
                               np.where(df['DevScrnng'] == 2, 0,  # Not screened = 0
                                       np.nan))                   # Missing/NA = NaN
    
    print(f"   ✓ Created DEV_SCREEN using HRSA methodology")
    print(f"   → K6Q variables used: {k6q12_col}, {k6q13a_col}, {k6q13b_col}, {k6q14a_col}, {k6q14b_col}")
    
    # Quality check: Report screening rates
    screening_rate = df['DEV_SCREEN'].mean()
    valid_responses = df['DEV_SCREEN'].notna().sum()
    if valid_responses > 0:
        print(f"   → Screening rate: {screening_rate:.3f} ({valid_responses:,} valid responses)")
    else:
        print(f"   ⚠ No valid screening responses found")
    
    return df

def main():
    """
    Main function to test variable creation if run as standalone script.
    """
    print("Variable Creation Module - Standalone Test")
    print("=" * 50)
    
    # Try to load data from the expected location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Look for the combined data file
    input_files = [
        os.path.join(script_dir, "cleaned_data", "nsch_all_years_combined.dta"),
        os.path.join(script_dir, "cleaned_data", "nsch_all_years_combined.csv"),
    ]
    
    df = None
    for file_path in input_files:
        if os.path.exists(file_path):
            print(f"Loading data from: {file_path}")
            try:
                if file_path.endswith('.dta'):
                    import pyreadstat
                    df, meta = pyreadstat.read_dta(file_path)
                else:
                    df = pd.read_csv(file_path)
                print(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
                break
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    if df is None:
        print("ERROR: No data file found. Expected files:")
        for file_path in input_files:
            print(f"  - {file_path}")
        print("\nPlease run ImportandCombine.R first to create the data file.")
        sys.exit(1)
    
    # Show available columns
    print(f"\nAvailable columns in dataset:")
    print(f"  Total: {len(df.columns)} columns")
    key_cols = [col for col in df.columns if any(x in col.lower() for x in ['year', 'age', 'k6q', 'sex', 'race'])]
    print(f"  Key columns: {key_cols}")
    
    # Create variables
    print(f"\nCreating analysis variables...")
    df_enhanced = create_variables(df)
    
    # Show results
    new_vars = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"\nNew variables created: {new_vars}")
    
    # Show some basic statistics
    if 'DEV_SCREEN' in df_enhanced.columns:
        screening_stats = df_enhanced['DEV_SCREEN'].value_counts(dropna=False)
        print(f"\nDevelopmental screening distribution:")
        print(screening_stats)
    
    if 'POST2017' in df_enhanced.columns:
        policy_stats = df_enhanced['POST2017'].value_counts()
        print(f"\nPolicy period distribution:")
        print("Pre-2017:", policy_stats.get(0, 0))
        print("Post-2017:", policy_stats.get(1, 0))
    
    print(f"\nVariable creation completed successfully!")
    return df_enhanced

if __name__ == "__main__":
    df_result = main()




