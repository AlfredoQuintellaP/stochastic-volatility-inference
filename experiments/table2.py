"""
Reproducing table 2 from our paper
We expect to see the sample moments of daily sp500 returns
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# defining the quantil
SIGNIF = 0.01
MAX_LAG = 20

df = pd.read_csv("data/sp500.csv", index_col="date", parse_dates=True)
y_all = df["ret"]

periods = {
    "Full": y_all,
    "In-sample": y_all.loc[:"2005-12-30"],
    "Out-of-sample": y_all.loc["2006-01-03":],
}

def acf(x, lag=1):
    """
    Computing the lag-"lag" autocorrelation (how much we are correlated with the anteriorobservation).
    """
    x = np.asarray(x)
    # print(np.corrcoef(x[:-1], x[1:]))
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def ccf2(y, lag=1):
    """
    Computing the correlation of the current return and the future squared (leverage effect).
    """
    y = np.asarray(y)
    return np.corrcoef(y[:-lag], y[lag:] ** 2)[0, 1]

if __name__ == "__main__":
    rows = {}
    stems = {}

    for name, s in periods.items():
        y = s.values
        n = len(y)
        jb = stats.jarque_bera(y)
        band = stats.norm.ppf(1 - SIGNIF / 2) / np.sqrt(n)

        lags = np.arange(1, MAX_LAG + 1)
        rho2 = np.array([acf(y ** 2, l) for l in lags])
        rho21 = np.array([ccf2(y, l) for l in lags])
        stems[name] = pd.DataFrame(
            {"rho2": rho2, "signif_rho2": np.abs(rho2) > band,
             "rho21": rho21, "signif_rho21": np.abs(rho21) > band},
            index=pd.Index(lags, name="lag"),
        )

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
            "Jarque-Bera p-value": jb.pvalue,
            "rho2(1)": rho2[0],
            "rho2(1) signif": abs(rho2[0]) > band,
            "rho21(1)": rho21[0],
            "rho21(1) signif": abs(rho21[0]) > band,
        }

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(out)
    out.to_csv("results/table2_reproduced.csv")
