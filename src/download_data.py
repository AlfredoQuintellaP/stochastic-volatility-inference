"""
Download the data and compute the return
return as y_t = 100 * diff(log P_t)
"""
import numpy as np
import pandas as pd
import yfinance as yf

START = "1995-12-29"
END = "2015-09-19"

def main():
    raw = yf.download("^GSPC", start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    px = raw["Adj Close"].dropna()
    ret = 100 * np.log(px).diff().dropna()
    df = pd.DataFrame({"adj_close": px, "ret": ret}).dropna()
    df.index.name = "date"
    df.to_csv("data/sp500.csv")
    print(f"Saved data/sp500.csv: {len(df)} returns, from {df.index[0].date()} to {df.index[-1].date()}")

if __name__ == "__main__":
    main()
