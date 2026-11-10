# ABC-filter ML for asymmetric SV models (SP500)
Reproduction of Mao, Czellar, Ruiz & Veiga (2016) + comparison with other estimation methods.

We will organize a basic roadmap to have a start (can and will definitly change over time). 

## (DONE) Basic data obtetion and calculations

The main check here is the table 2 of the article. We expect equal results to check that we got correct results and used the correct dates.

### 1. Obtain S&P 500 returns

### 2. Basic calculations

- **Objective:** Split the data into in-sample and out-of-sample periods. Use the in-sample period to estimate model parameters and the out-of-sample period only to evaluate forecasts, without reestimating parameters. Compute the mean, median, max and min, variance, skewness and kurtosis for the in-sample, out-of-sample and for the full sample periods.

### 3. Jarque--Bera test and skewness/kurtosis p-values

### 4. Autocorrelations of squared returns and cross-correlations between returns and future squared returns

### 5. ACF of squared returns and CCF of returns with future squared returns

-------------------------------

## Model and Simulator

-------------------------------

## Theorical Moments

-------------------------------

## ABC Filter (basic central concept of the paper)

-------------------------------

## Estimation and Reproduction of Results (main basic goal)

-------------------------------

## Out-of-Sample Forecasting

### . One-step-ahead VaR at the 1% and 5%

### . Evaluate VaR forecasts

-------------------------------

## Comparison with Alternative Methods

### . Bootstrap particle filter with maximum likelihood (core objective I think)

### . Particle marginal Metropolis--Hastings

### . Comparative experiments
