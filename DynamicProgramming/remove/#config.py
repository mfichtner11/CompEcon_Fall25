#config.py


#--------Packages
from pathlib import Path

### Below from SMM notes
import pandas as pd
import matplotlib.pyplot as plt
import dask
from dask import delayed
from dask.distributed import Client
import scipy.optimize as opt
import scipy.stats as stats
#import plotly.express as px

#---------relative path configuration

# Root = folder containing config.py

from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd()

SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

print("Setup successful")
print(f"ROOT: {ROOT}")
print(f"SCRIPTS: {SCRIPTS}")
print(f"RESULTS: {RESULTS}")

#--------CE Fixed Parameters

beta = 0.95      # discount factor
delta = 0.15     # depreciation rate
p = 1.0          # price of capital

#--------SMM starting parameters

alpha_start = 0.70
gamma_start = 0.13
rho_start = 0.10
sigma_start = 0.89
phi0_start = 0.01

theta0 = [
    alpha_start,
    gamma_start,
    rho_start,
    sigma_start,
    phi0_start
]


# -------------------------
# Target moments: Table 3
# -------------------------

target_moments = [
    0.03,   # a1
    0.24,   # a2
    0.40,   # serial correlation of I/K
    0.25,   # std(pi/K)
    3.00,   # average Q
    0.25    # fraction externally financed
]

