# Unit Test q2
# Marli Fichtner

from pathlib import Path
import pytest
import pandas as pd
from ProblemSet4_Fichtner_code2 import clean_data   # adjust filename if needed

# Load the real dataset once for all tests
@pytest.fixture(scope="module")
def raw_psid_df():
    path = Path("/Users/marlifirillo/Comp_Econ/CompEconPrivate/PSID_data.dta")
    if not path.exists():
        pytest.skip(f"PSID_data.dta not found at {path}. Skipping tests.")
    return pd.read_stata(path)
def test_selection_criteria(raw_psid_df):
    """(a) All observations in cleaned data should meet selection rules."""
    df = clean_data(raw_psid_df)

    # Check each rule with plain assert
    # Not testing "head" because all data is heads of households
    assert df['age'].between(25, 60).all()
    assert (df['wage'] > 7).all()
    assert set(df['year']).issubset({1971, 1980, 1990, 2000})

def test_indicator_variables(raw_psid_df):
    """(b) Indicator variables should be binary and mutually consistent."""
    df = clean_data(raw_psid_df)
    dummies = df[['black', 'hispanic', 'other']]

    # Only 0 or 1 allowed
    for col in dummies.columns:
        assert set(dummies[col].dropna().unique()).issubset({0, 1})

    # Sum must be 0 (White) or 1 (exactly one non-White)
    sum_ind = dummies.sum(axis=1)
    assert set(sum_ind.unique()).issubset({0, 1})
