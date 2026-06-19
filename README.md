# Forex Currency Predictor

An end-to-end machine learning project that forecasts foreign exchange rates for 22 currencies against the USD, deployed as an interactive Streamlit web app and packaged with Docker.

## Project Overview

This project compares multiple forecasting approaches on real-world daily forex data (2000–2019) and selects the best-performing model for each currency individually.

## Dataset

- 5,217 daily exchange rate records
- 22 currencies vs USD (AUD, EUR, GBP, JPY, INR, and more)
- Source: historical forex rates dataset
- Cleaned and preprocessed: date parsing, missing value handling via forward/backward fill

## Methodology

Three forecasting approaches were trained and evaluated per currency, using the last 60 days as a test set:

1. **Naive baseline** — predicts tomorrow's rate as today's last known rate
2. **Prophet** — Facebook's trend/seasonality forecasting model
3. **XGBoost** — gradient boosting using lag features (last 5 days)

### Key finding

Across all 22 currencies, **XGBoost consistently outperformed both Prophet and the naive baseline**, often by a wide margin (e.g. EUR: 0.21% MAPE vs 1.22% naive vs 4.30% Prophet).

Interestingly, the naive baseline beat Prophet on every currency tested — a well-documented effect in financial forecasting, since exchange rates closely resemble a random walk and trend/seasonality-based models often add noise rather than signal. Lag-based feature engineering (XGBoost) proved far more effective at capturing short-term dependencies.

| Model | Behavior |
|---|---|
| Naive | Simple, surprisingly strong baseline |
| Prophet | Underperformed — forex lacks strong seasonality |
| XGBoost | Best performer for all 22 currencies |

## Project Structure