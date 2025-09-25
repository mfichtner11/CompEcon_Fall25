import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="module")
def raw_psid_df():
    path = Path("/Users/marlifirillo/Comp_Econ/CompEconPrivate/PSID_data.dta")
    if not path.exists():
        pytest.skip(f"PSID_data.dta not found at {path}. Skipping tests.")
    return pd.read_stata(path)
