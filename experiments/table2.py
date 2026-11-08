"""
Reproducing table 2 from our paper
We expect to see the sample moments of daily sp500 returns
"""
import numpy as np
import pandas as pd
from scipy import stats

# defining the quantil
alpha = 0.01

df = pd.read_csv("data/sp500.csv", index_col="date", parse_dates=True)
y_all = df["ret"]

periods = {
    "Full": y_all,
    "In-sample": y_all.loc[:"2005-12-30"],
    "Out-of-sample": y_all.loc["2006-01-03":],
}

def acf1(x):
    """
    Computing the lag-1 autocorrelation (how much we are correlated with the anteriorobservation).
    """
    x = np.asarray(x)
    # print(np.corrcoef(x[:-1], x[1:]))
    return np.corrcoef(x[:-1], x[1:])[0, 1]

def ccf21(y):
    """
    Computing the correlation of the current return and the future squared (leverage effect).
    """
    y = np.asarray(y)
    return np.corrcoef(y[:-1], y[1:] ** 2)[0, 1]

rows = {}
for name, s in periods.items():
    y = s.values
    n = len(y)
    jb = stats.jarque_bera(y)
    band = stats.norm.ppf(1-alpha / 2) / np.sqrt(n)
    # band = 2.576 / np.sqrt(n)
    r2, r21 = acf1(y ** 2), ccf21(y)
    rows[name] = {
        "Sample size": n,
        "Mean": y.mean(),
        "Median": np.median(y),
        "Maximum": y.max(),
        "Minimum": y.min(),
        "Variance": y.var(ddof=1),
        "Skewness": stats.skew(y),
        "Kurtosis": stats.kurtosis(y, fisher=False),
        "Jarque-Bera": jb.statistic,
        "rho2(1)": r2,
        "rho2(1) signif 1%": abs(r2) > band,
        "rho21(1)": r21,
        "rho21(1) signif 1%": abs(r21) > band,
    }

out = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(out)
out.to_csv("results/table2_reproduced.csv")
