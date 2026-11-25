"""
Sample statistics of the returns (used in table 2 and in the sample part of figure 5b)
"""
import numpy as np
from scipy import stats

def acf(x, lag=1):
    """
    Computing the lag-"lag" autocorrelation (how much we are correlated with the anterior observation).
    """
    x = np.asarray(x)
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def ccf2(y, lag=1):
    """
    Computing the correlation of the current return and the future squared (leverage effect).
    """
    y = np.asarray(y)
    return np.corrcoef(y[:-lag], y[lag:] ** 2)[0, 1]

def sample_curves(y, max_lag=20):
    """
    Computing rho_2(tau) and rho_21(tau) for tau = 1, ..., max_lag.
    """
    lags = np.arange(1, max_lag + 1)
    rho2 = np.array([acf(y ** 2, l) for l in lags])
    rho21 = np.array([ccf2(y, l) for l in lags])
    return lags, rho2, rho21

def skewness_kurtosis_pvalues(y):
    """
    Computing the p-values of the skewness and kurtosis tests using the asymptotic variances
    of Premaratne & Bera (2016), as in the table 2 of the paper (footnote 12).
    Returns skewness, kurtosis, p-value of skewness and p-value of kurtosis.
    """
    y = np.asarray(y)
    T = len(y)
    y_c = y - y.mean()
    m2 = np.mean(y_c ** 2)
    m3 = np.mean(y_c ** 3)
    m4 = np.mean(y_c ** 4)
    m5 = np.mean(y_c ** 5)
    m6 = np.mean(y_c ** 6)
    m8 = np.mean(y_c ** 8)

    S = m3 / (m2 ** 1.5)
    K = m4 / (m2 ** 2.0)

    var_S = (m6 - 6.0 * m2 * m4 + 9.0 * (m2 ** 3) - 2.0 * S * m5 + 4.0 * (S ** 2) * (m2 ** 3)) / (T * (m2 ** 3))
    var_K = (
        (m8 / (m2 ** 4))
        - 4.0 * (m6 * m4) / (m2 ** 5)
        + 4.0 * (m4 ** 3) / (m2 ** 6)
        - (m4 ** 2) / (m2 ** 4)
        + 16.0 * (m3 ** 2 * m4) / (m2 ** 5)
        - 8.0 * (m3 * m5) / (m2 ** 4)
        + 16.0 * (m3 ** 2) / (m2 ** 3)
    ) / T

    # the max() keeps the variance at least as large as the one under normality (6/T and 24/T)
    z_S = S / np.sqrt(max(var_S, 6.0 / T))
    p_S = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_S)))

    z_K = (K - 3.0) / np.sqrt(max(var_K, 24.0 / T))
    p_K = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_K)))
    return S, K, p_S, p_K
