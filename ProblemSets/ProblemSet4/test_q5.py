import numpy as np
import pandas as pd
import ProblemSet4_Fichtner_code2 as ps4   

def test_mle_vs_ols():
    # Make a small fake dataset
    n = 100
    rng = np.random.default_rng(0)
    hyrsed = rng.integers(8, 20, size=n)
    age = rng.integers(25, 60, size=n)
    age_sq = age ** 2
    black = rng.binomial(1, 0.2, size=n)
    hispanic = rng.binomial(1, 0.1, size=n)
    other = rng.binomial(1, 0.05, size=n)

    # True coefficients
    beta_true = np.array([0.5, 0.08, 0.02, -0.0002, -0.1, -0.05, -0.02])
    X = np.column_stack([np.ones(n), hyrsed, age, age_sq, black, hispanic, other])
    y = X @ beta_true + rng.normal(0, 0.3, size=n)

    # Run OLS and MLE
    beta_ols = ps4.ols_numpy(X, y)
    mle = ps4.mle_gaussian(X, y, method="L-BFGS-B")
    beta_mle = mle["beta"]

    # Test: coefficients should match closely
    assert np.allclose(beta_mle, beta_ols, atol=1e-6)
