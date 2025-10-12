# Problem Set 4 — MLE-only script
# Marli Fichtner

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---- Paths ----
base_dir = r'/Users/marlifirillo/Comp_Econ/CompEcon_Fall25'
opt_dir = os.path.join(base_dir, 'Optimization')
output_path = os.path.join(base_dir, 'ProblemSets')
data_path = os.path.join(opt_dir, 'PSID_data.dta')
os.makedirs(output_path, exist_ok=True)

# ------------------------------------------------------------
# Import
# ------------------------------------------------------------
def import_data(file_path: str) -> pd.DataFrame:
    """
    Import PSID data from Stata file."""
    return pd.read_stata(file_path)

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply selection rules:
      - age 25–60
      - male only (hsex==1)
      - wage > 7 (hlabinc/hannhrs)
      - years in {1971, 1980, 1990, 2000}

    Create variables for regression:
      - ln_wage
      - age_sq
      - race dummies: black, hispanic, other
    """
    clean_df = df.copy()

    # wage
    clean_df["wage"] = clean_df["hlabinc"] / clean_df["hannhrs"]

    # filters
    clean_df = clean_df[
        (clean_df["age"].between(25, 60)) &
        (clean_df["hsex"] == 1) &
        (clean_df["wage"] > 7) &
        (clean_df["year"].astype(int).isin([1971, 1980, 1990, 2000]))
    ].copy()

    clean_df["year"] = clean_df["year"].astype(int)

    # race dummies
    clean_df["black"] = (clean_df["hrace"] == 2).astype(int)
    clean_df["hispanic"] = (clean_df["hrace"] == 3).astype(int)
    clean_df["other"] = (~clean_df["hrace"].isin([1, 2, 3])).astype(int)

    # derived variables
    clean_df["ln_wage"] = np.log(clean_df["wage"])
    clean_df["age_sq"] = clean_df["age"] ** 2

    # return tidy
    relevant_vars = ["year", "hsex", "wage", "age", "hyrsed",
                     "black", "hispanic", "other", "ln_wage", "age_sq"]
    return clean_df[relevant_vars].dropna()

# ------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------
PARAM_ORDER = ["Intercept", "hyrsed", "age", "age_sq", "black", "hispanic", "other"]

def build_X(sub: pd.DataFrame) -> np.ndarray:
    """ Build design matrix X from subset dataframe.
    """
    return np.column_stack([
        np.ones(len(sub)),
        sub["hyrsed"].to_numpy(float),
        sub["age"].to_numpy(float),
        sub["age_sq"].to_numpy(float),
        sub["black"].to_numpy(float),
        sub["hispanic"].to_numpy(float),
        sub["other"].to_numpy(float),
    ])

# ------------------------------------------------------------
# NLL + gradient
# ------------------------------------------------------------
def nll_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    """ Compute negative log-likelihood and its gradient for Gaussian linear model.
    """
    n, k = X.shape
    beta = theta[:k]
    lsig = theta[k]

    r = y - X @ beta
    RSS = float(r @ r)
    e2 = np.exp(-2.0 * lsig)

    nll = n * lsig + 0.5 * e2 * RSS
    grad_beta = -e2 * (X.T @ r)
    grad_lsig = n - e2 * RSS
    grad = np.concatenate([grad_beta, np.array([grad_lsig])])
    return nll, grad

# ------------------------------------------------------------
# MLE wrapper
# ------------------------------------------------------------
def mle_gaussian(X: np.ndarray, y: np.ndarray, method="L-BFGS-B"):
    """ MLE estimation for Gaussian linear model using scipy.optimize.minimize.
    """
    n, k = X.shape
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ols
    sigma_init = np.sqrt(max((resid @ resid) / n, 1e-12))
    theta0 = np.concatenate([beta_ols, [np.log(sigma_init)]])

    bounds = [(-np.inf, np.inf)] * k + [(np.log(1e-12), np.inf)]

    res = minimize(
        fun=lambda th: nll_and_grad(th, X, y)[0],
        x0=theta0,
        jac=lambda th: nll_and_grad(th, X, y)[1],
        method=method,
        bounds=bounds if method in ("L-BFGS-B", "SLSQP") else None,
        options=dict(maxiter=50_000, ftol=1e-12)
    )

    th = res.x
    beta_hat = th[:k]
    sigma_hat = float(np.exp(th[k]))

    r = y - X @ beta_hat
    RSS = float(r @ r)
    ll = -0.5*n*np.log(2*np.pi) - n*np.log(sigma_hat) - 0.5*(RSS/(sigma_hat**2))

    return dict(success=res.success, beta=beta_hat, sigma=sigma_hat, ll=ll)

# ------------------------------------------------------------
# Run estimation by year
# ------------------------------------------------------------
def run_mle_years(cleaned: pd.DataFrame, years=(1971, 1980, 1990, 2000)):
    """ Run MLE for each year and collect results in DataFrame.
    """
    results = []
    for t in years:
        sub = cleaned.loc[cleaned["year"] == t, :]
        if sub.empty:
            print(f"[{t}] skipped (no obs)")
            continue
        y = sub["ln_wage"].to_numpy(float)
        X = build_X(sub)
        mle = mle_gaussian(X, y)

        for i, name in enumerate(PARAM_ORDER):
            results.append({
                "year": t,
                "param": name,
                "beta": mle["beta"][i],
                "sigma": mle["sigma"],
                "logLik": mle["ll"],
                "success": mle["success"]
            })
    return pd.DataFrame(results)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    """ Main execution: import, clean, run MLE, save results."""
    PSID_df = import_data(data_path)
    clean_PSID = clean_data(PSID_df)
    results = run_mle_years(clean_PSID)
    results_path = os.path.join(output_path, "ps4_mle_results.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved results → {results_path}")
