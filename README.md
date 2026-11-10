# ABC-filter ML for asymmetric SV models (SP500)
Reproduction of Mao, Czellar, Ruiz & Veiga (2016) + comparison with other estimation methods.

-------------------------------

## Basic structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── sp500.csv             # daily S&P 500 adjusted close and returns
├── src/
│   ├── __init__.py
│   └── download_data.py      # data acquisition (Yahoo Finance)
├── experiments/
│   └── table2.py     # Table 2 and Figure 5 (sample part)
├── results/
│   └── table2_reproduced.csv
```

The project is organized in six folders:

- `data/`: datasets, normally saved as `.csv`.
- `src/`: reusable code (data acquisition now; the model, filters and estimators later, e.g. `ged.py`, `model.py`, `abc_filter.py`).
- `experiments/`: one script per table or figure reproduced from the paper, named after what it reproduces.
- `results/`: outputs of the experiments (tables, figures, saved arrays).

### Data

S&P 500 index (`^GSPC`) adjusted close from Yahoo Finance, from 2 January 1996 to 18 September 2015. Returns are computed as y_t = 100 · Δlog(P_t). The in-sample period ends on 30 December 2005 and the out-of-sample period starts on 3 January 2006. Yahoo occasionally revises historical data, so small differences with respect to the paper are possible.

-------------------------------

We will organize a basic roadmap to have a start (can and will definitly change over time). 


## Roadmap

### 1. Data and descriptive statistics (Table 2, Figure 5 sample part)
- [x] Obtain S&P 500 returns
- [x] Split in-sample / out-of-sample
- [x] Sample moments (mean, median, max, min, variance, skewness, kurtosis)
- [x] Jarque-Bera test
- [ ] Skewness / kurtosis p-values (Premaratne-Bera correction)
- [x] ACF of y^2 and CCF of (y_t, y_{t+tau}^2), lags 1-20

### 2. Model and simulator
- [ ] Standardized GED(nu): density, sampler, E|eps|
- [ ] Asymmetry function f(eps; theta) (TGA-SV)
- [ ] TGA-SV simulator
- [ ] Nested models: E-SV, A-ARSV, ARSV

### 3. Theoretical moments
- [ ] P(a, phi) infinite products
- [ ] Variance and kurtosis, ACF and CCF (plug-in)
- [ ] Validate against long simulated series

### 4. ABC filter
- [ ] State-observation sampling, kernel and bandwidth
- [ ] Continuous resampling (Malik-Pitt), common random numbers
- [ ] Figure 4 (standard vs continuous log-likelihood)

### 5. Estimation and reproduction
- [ ] ML optimizer
- [ ] Table 1 (Monte Carlo)
- [ ] Table 3 (S&P 500, 8 models)
- [ ] Figure 5(b) with plug-in curves

### 6. Out-of-sample forecasting
- [ ] One-step-ahead VaR (1%, 5%)
- [ ] VaR evaluation: DQ test, PQL, SPA (Table 4)
- [ ] Squared-return quantiles and Kupiec (Table 5)

### 7. Comparison with alternative methods
- [ ] Bootstrap particle filter + ML
- [ ] PMMH
- [ ] Comparative experiments (variability, cost, bias/RMSE)

### 8. Deliverables (deadline: 10 January)
- [ ] Notebook, slides, clean repository, viva preparation## . Comparative experiments
