"""
Loading the S&P 500 returns and splitting them in the periods used in the paper
"""
import pandas as pd

def load_returns(path="data/sp500.csv"):
    """
    Reading the daily returns y_t = 100 * diff(log P_t) saved by download_data.py.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df["ret"]

def load_periods(path="data/sp500.csv"):
    """
    Splitting the returns in full sample, in-sample (until 2005-12-30) and out-of-sample (from 2006-01-03).
    """
    y_all = load_returns(path)
    return {
        "Full": y_all,
        "In-sample": y_all.loc[:"2005-12-30"],
        "Out-of-sample": y_all.loc["2006-01-03":],
    }
