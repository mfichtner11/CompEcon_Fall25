# Problem Set 5 - Data Cleaning Script

import pandas as pd
import numpy as np
import os
import pyreadstat
from CreateVariables import create_variables  

# Set directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))

def import_combined_dta():
    """Import the combined .dta file created in R
    """
    file_path = os.path.join(script_dir, "cleaned_data", "nsch_all_years_combined.dta")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    try:
        df, meta = pyreadstat.read_dta(file_path)
        print(f"SUCCESS: Imported {df.shape[0]:,} rows, {df.shape[1]} columns")
        if 'year' in df.columns:
            print(f"   Years: {sorted(df['year'].unique())}")
        return df, meta
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return None

def regression_ready_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression-ready indicator variables from categorical variables.
    Indicators are 1 if condition is met, 0 otherwise.
    Indicators include:
       1) Race: White, Black, Other Race
       2) Sex: Female only, equal to 1 if female
       3) Insurance: Private, Public, Uninsured
       4) Education: Low (high school or less), High (some college+)

    """
    print("Creating indicator variables:")
    
    # Race/ethnicity indicators
    df['WHITE'] = np.where(df['RACEETH'] == 1, 1, 0)
    df['BLACK'] = np.where(df['RACEETH'] == 2, 1, 0)
    df['OTHER_RACE'] = np.where(df['RACEETH'] == 3, 1, 0)
    print("   Created race/ethnicity indicators")

    # Sex indicators
    df['FEMALE'] = np.where(df['SEX'] == 2, 1, 0)
    df['MALE'] = np.where(df['SEX'] == 1, 1, 0)
    print("   Created sex indicators")

    # Insurance indicators (CORRECTED)
    df['PRIVATE_INSURANCE'] = np.where(df['INSTYPE'] == 2, 1, 0)
    df['PUBLIC_INSURANCE'] = np.where(df['INSTYPE'] == 1, 1, 0)
    df['UNINSURED'] = np.where((df['INSTYPE'] == 4) | (df['INSTYPE'] == 5), 1, 0)
    print("   Created insurance indicators")

    # Education indicators
    df['LOW_EDUCATION'] = np.where(df['HEDUC'] <= 2, 1, 0)  # High school or less
    df['HIGH_EDUCATION'] = np.where(df['HEDUC'] >= 3, 1, 0)  # Some college+
    print("   Created education indicators")

    # Urban/rural
    df['RURAL'] = np.where(df['METRO'] == 2, 1, 0)
    df['URBAN'] = np.where(df['METRO'] == 1, 1, 0)
    print("   Created urban/rural indicators")
    
    return df

def clean_data(df):
    """
    Uses precursor file: CreateVariables.py
    Creates analysis-ready dataset with all variables and indicators
    """ 
    
    print(f"\nStarting data cleaning...")
    print(f"Input: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Create all variables
    print("Creating analysis variables:")
    df = create_variables(df)
    
    # Create indicator variables
    df = regression_ready_variables(df)
    
    # Filter to target population (ages 9-35 months)
    if not df['AGE_MONTHS'].isna().all():
        mask = (df['AGE_MONTHS'] >= 9) & (df['AGE_MONTHS'] <= 35)
        df = df[mask].copy()
        print(f"Filtered to ages 9-35 months: {len(df):,} rows")
    
    # Select final variables (including new indicators)
    final_vars = [
        # Core variables
        'YEAR', 'POST2017', 'STATE', 'DEV_SCREEN', 'AGE_MONTHS', 'TARGET_AGE', 'INSTYPE',
        # Indicator variables
        'WHITE', 'BLACK', 'OTHER_RACE',
        'MALE', 'FEMALE', 
        'PRIVATE_INSURANCE', 'PUBLIC_INSURANCE', 'UNINSURED',
        'LOW_EDUCATION', 'HIGH_EDUCATION',
        'URBAN', 'RURAL'
    ]
    
    # Keep only variables that exist and have some data
    available_vars = [var for var in final_vars 
                     if var in df.columns and not df[var].isna().all()]
    
    df_final = df[available_vars].copy()
    
    # Remove rows missing the outcome variable
    if 'DEV_SCREEN' in df_final.columns:
        df_final = df_final.dropna(subset=['DEV_SCREEN'])
        print(f"Removed rows missing DEV_SCREEN: {len(df_final):,} rows remain")
    
    # Summary
    print(f"\nFINAL DATASET:")
    print(f"   Rows: {len(df_final):,}")
    print(f"   Variables: {len(df_final.columns)}")
    print(f"   Available variables: {list(df_final.columns)}")
    
    if 'DEV_SCREEN' in df_final.columns:
        screening_rate = df_final['DEV_SCREEN'].mean()
        print(f"   Screening rate: {screening_rate:.1%}")
    
    if 'YEAR' in df_final.columns:
        year_counts = df_final['YEAR'].value_counts().sort_index()
        print(f"   Sample by year:")
        for year, count in year_counts.items():
            print(f"     {year}: {count:,}")
    
    # Show insurance distribution
    if 'INSTYPE' in df_final.columns:
        print(f"   Insurance distribution:")
        insurance_counts = df_final['INSTYPE'].value_counts().sort_index()
        for ins_type, count in insurance_counts.items():
            print(f"     INSTYPE {ins_type}: {count:,}")
    
    return df_final  

def main():
    """Main execution pipeline"""
    print("NSCH DEVELOPMENTAL SCREENING ANALYSIS")
    print("Data Processing Pipeline")
    
    # Import data
    result = import_combined_dta()
    if not result:
        return
    
    df, meta = result
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save outputs
    output_files = {
        'csv': os.path.join(current_dir, "cleaned_data", "nsch_analysis_ready.csv"),
        'dta': os.path.join(current_dir, "cleaned_data", "nsch_analysis_ready.dta")
    }
    
    # Save CSV
    df_clean.to_csv(output_files['csv'], index=False)
    print(f"CSV saved: {output_files['csv']}")
    
    # Save Stata file
    try:
        df_clean.to_stata(output_files['dta'], write_index=False)
        print(f"Stata saved: {output_files['dta']}")
    except Exception as e:
        print(f"WARNING: Stata save failed: {e}")
    
    # Display sample
    print(f"\nSAMPLE DATA:")
    print(df_clean.head())
    
    print(f"\nPipeline completed successfully!")
    return df_clean

if __name__ == "__main__":
    df_final = main()