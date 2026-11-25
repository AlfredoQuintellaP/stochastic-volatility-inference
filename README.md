# ABC-filter ML for asymmetric SV models (SP500)
Reproduction of Mao, Czellar, Ruiz & Veiga (2016) + comparison with other estimation methods.

The main objective of the paper is to propose a model to the return volatility and a method to estimate their parameters. As the paper does, we are going to reproduce some results using the S&P 500 results.

## The problem, the method and the whys

The model assumes that volatility exists, but no one sees it. Volatility is like a measure of how nervous the market is: how much the prices fluctuate. What we get as data is just the return, a single number each day. A given return doesn't necessarily say much about the volatility, a high return, for example, can come from a high volatility or a rare draw with low volatility.
In SV models the volatility is a hidden variable that runs by itself and has its own noise. We can only infer it by the returns, just like we can only see the mood of someone by their actions. This is the main difference between SV and models like GARCH, where the volatility is exaclty determined by past returns.

We want the model to capture three main things (as observed in the S&P500 returns):
- Volatile days tend to be followed by volatile days (volatility persistence or clustering).
- Extreme price movements occur more frequently than expected under a normal distribution (fat tails).
- Price drops tend to be followed by higher volatility than price increases of the same magnitude (the leverage effect).

A model in which volatility is hidden, persistent over time and reacts to the previous day's return can reproduce these three facts. The price of this flexibility is that, since volatility is never observed, estimating the model is hard.

-------------------------------

## Basic structure of the project

```
.
├── README.md
├── requirements.txt
├── data
│   └── sp500.csv
├── experiments
│   ├── descriptive_statistics.py
│   └── theorical_properties.py
├── results
│   ├── images
│   │   ├── figure1_rho2_rho21_vs_alpha_gamma1.png
│   │   ├── figure2_theoretical_acf_ccf_by_nu.png
│   │   ├── figure3_volatility_response_surfaces.png
│   │   ├── figure5a_sp500_daily_returns.png
│   │   ├── figure5b_sample_acf_ccf_in_sample.png
│   │   └── figure5b_sample_vs_plugin_acf_ccf.png
│   └── tables
│       ├── figure5b_sample_acf_ccf_in_sample.csv
│       ├── table2_reproduced.csv
│       └── table2_sample_moments_sp500.csv
└── src
    ├── data_utils.py
    ├── download_data.py
    ├── __init__.py
    ├── stats_utils.py
    └── theory.py

```

-------------------------------

# Reproducing the results

