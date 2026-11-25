"""
Descriptive statistics of the sp500 returns (section 5.1 of the paper)
Reproducing: table 2 (sample moments), figure 5(a) (daily returns) and figure 5(b) (sample and plug-in acf and ccf)
Tables are saved in results/tables and images in results/images
"""
import sys
sys.path.append(".")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.data_utils import load_periods, load_returns
from src.stats_utils import acf, ccf2, sample_curves, skewness_kurtosis_pvalues
from src.theory import theoretical_acf, theoretical_ccf

# defining the quantil
SIGNIF = 0.01
MAX_LAG = 20

# end of the in-sample period and of each out-of-sample subperiod (figure 5a)
SPLIT_DATES = ["2005-12-30", "2007-12-31", "2009-12-31", "2012-12-31"]

def table2():
    """
    Table 2: sample moments of daily sp500 returns (full, in-sample and out-of-sample).
    Output: results/tables/table2_sample_moments_sp500.csv
    """
    periods = load_periods()
    rows = {}

    for name, s in periods.items():
        y = s.values
        n = len(y)
        jb = stats.jarque_bera(y)
        band = stats.norm.ppf(1 - SIGNIF / 2) / np.sqrt(n)
        S, K, p_S, p_K = skewness_kurtosis_pvalues(y)
        r2, r21 = acf(y ** 2), ccf2(y)

        rows[name] = {
            "Sample size": n,
            "Mean": y.mean(),
            "Median": np.median(y),
            "Maximum": y.max(),
            "Minimum": y.min(),
            "Variance": y.var(ddof=1),
            "Skewness": S,
            "Skewness p-value": p_S,
            "Kurtosis": K,
            "Kurtosis p-value": p_K,
            "Jarque-Bera": jb.statistic,
            "Jarque-Bera p-value": jb.pvalue,
            "rho2(1)": r2,
            "rho2(1) signif": abs(r2) > band,
            "rho21(1)": r21,
            "rho21(1) signif": abs(r21) > band,
        }

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(out)
    out.to_csv("results/tables/table2_sample_moments_sp500.csv")

def figure5a():
    """
    Figure 5(a): daily sp500 returns from January 2, 1996 to September 18, 2015, with the in-sample / out-of-sample
    split and the four out-of-sample subperiods (2006-2007, 2008-2009, 2010-2012 and 2013-2015).
    Output: results/images/figure5a_sp500_daily_returns.png
    """
    y_all = load_returns()

    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(y_all.index, y_all.values, color="black", linewidth=0.5)
    for d in SPLIT_DATES:
        plt.axvline(pd.to_datetime(d), color="red", linestyle="-", linewidth=1.2)

    plt.ylabel(r"$y_t$ (% return)", fontsize=11)
    plt.xlim(y_all.index[0], y_all.index[-1])
    plt.ylim(-12, 15)
    plt.title("Daily S&P 500 returns from January 2, 1996 to September 18, 2015", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/images/figure5a_sp500_daily_returns.png")
    plt.close()

def figure5b(results=None):
    """
    Figure 5(b): sample acf of squared returns (left) and ccf between returns and future squared returns (right),
    in-sample, for the GED models (top) and the Gaussian models (bottom).
    results is the dictionary of the estimation step, {"TGA-SV-G": {"params": {...}}, ...}. When it is given, the
    plug-in theoretical curves of the estimated models are drawn on top of the sample stems (red).
    Outputs: results/images/figure5b_sample_vs_plugin_acf_ccf.png
             results/tables/figure5b_sample_acf_ccf_in_sample.csv
    """
    y = load_periods()["In-sample"].values
    n = len(y)
    band = stats.norm.ppf(1 - SIGNIF / 2) / np.sqrt(n)
    lags, rho2, rho21 = sample_curves(y, max_lag=MAX_LAG)

    stems = pd.DataFrame(
        {"rho2": rho2, "signif_rho2": np.abs(rho2) > band,
         "rho21": rho21, "signif_rho21": np.abs(rho21) > band},
        index=pd.Index(lags, name="lag"),
    )
    print(stems.round(3))
    stems.to_csv("results/tables/figure5b_sample_acf_ccf_in_sample.csv")

    styles = [
        ("TGA-SV", dict(color="black", ls="-", lw=1.4)),
        ("E-SV",   dict(color="purple", ls="--", lw=1.3)),
        ("A-ARSV", dict(color="black", ls="None", marker=".", ms=5)),
        ("ARSV",   dict(color="blue", ls="-.", lw=1.2)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for row_idx, suffix in enumerate(["G", "N"]):
        axes[row_idx, 0].stem(lags, rho2, linefmt="r-", markerfmt=" ", basefmt="k-")
        axes[row_idx, 1].stem(lags, rho21, linefmt="r-", markerfmt=" ", basefmt="k-")

        if results is not None:
            for base, st in styles:
                m = f"{base}-{suffix}"
                p = results[m]["params"]
                sig = float(np.sqrt(p["sigma_eta2"]))
                r2 = theoretical_acf(lags, p["phi"], sig, c=2, nu=p["nu"],
                                     alpha=p["alpha"], gamma1=p["gamma1"], gamma2=p["gamma2"])
                r21 = theoretical_ccf(lags, p["phi"], sig, c=2, nu=p["nu"],
                                      alpha=p["alpha"], gamma1=p["gamma1"], gamma2=p["gamma2"])
                axes[row_idx, 0].plot(lags, r2, label=m, **st)
                axes[row_idx, 1].plot(lags, r21, label=m, **st)
            axes[row_idx, 0].legend(fontsize=8)

        axes[row_idx, 0].set_ylim(0, 0.4)
        axes[row_idx, 0].set_xlim(0, MAX_LAG)
        axes[row_idx, 1].set_ylim(-0.2, 0.0)
        axes[row_idx, 1].set_xlim(0, MAX_LAG)
        axes[row_idx, 0].set_xlabel(r"$\tau$")
        axes[row_idx, 1].set_xlabel(r"$\tau$")

    axes[0, 0].set_title("GED — ACF of squares")
    axes[0, 1].set_title("GED — CCF")
    axes[1, 0].set_title("Gaussian — ACF of squares")
    axes[1, 1].set_title("Gaussian — CCF")
    plt.tight_layout()
    plt.savefig("results/images/figure5b_sample_vs_plugin_acf_ccf.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/images", exist_ok=True)

    table2()
    figure5a()
    figure5b()   # i have to correct this later passing the real results...
