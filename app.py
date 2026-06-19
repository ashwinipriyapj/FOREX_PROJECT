import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forex Predictor", layout="centered")

st.title("💱 Forex Currency Predictor")
st.write("Forecast future exchange rates using trained XGBoost models")

# Load last known values
with open("models/last_values.json", "r") as f:
    last_values = json.load(f)

currency_names = {
    "AUD": "Australian Dollar", "EUR": "Euro", "NZD": "New Zealand Dollar",
    "GBP": "British Pound", "BRL": "Brazilian Real", "CAD": "Canadian Dollar",
    "CNY": "Chinese Yuan", "HKD": "Hong Kong Dollar", "INR": "Indian Rupee",
    "KRW": "Korean Won", "MXN": "Mexican Peso", "ZAR": "South African Rand",
    "SGD": "Singapore Dollar", "DKK": "Danish Krone", "JPY": "Japanese Yen",
    "MYR": "Malaysian Ringgit", "NOK": "Norwegian Krone", "SEK": "Swedish Krona",
    "LKR": "Sri Lankan Rupee", "CHF": "Swiss Franc", "TWD": "Taiwan Dollar",
    "THB": "Thai Baht"
}

# Sidebar controls
st.sidebar.header("Settings")
selected_code = st.sidebar.selectbox(
    "Select Currency",
    options=list(currency_names.keys()),
    format_func=lambda x: f"{x} - {currency_names[x]}"
)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)


def make_forecast(currency_code, days, n_lags=5):
    """Dynamically loads the correct model and generates forecast"""
    model = joblib.load(f"models/{currency_code}_model.pkl")
    history = list(last_values[currency_code])

    predictions = []
    for _ in range(days):
        lag_features = np.array(history[-n_lags:][::-1]).reshape(1, -1)
        pred = model.predict(lag_features)[0]
        predictions.append(pred)
        history.append(pred)

    return predictions


if st.sidebar.button("Generate Forecast"):
    predictions = make_forecast(selected_code, forecast_days)

    st.subheader(f"{forecast_days}-Day Forecast for {currency_names[selected_code]} ({selected_code}/USD)")

    forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates.strftime("%Y-%m-%d"),
        "Predicted Rate": [round(p, 4) for p in predictions]
    })

    st.dataframe(forecast_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    history_vals = last_values[selected_code]
    ax.plot(range(len(history_vals)), history_vals, label="Recent History", marker="o")
    ax.plot(range(len(history_vals) - 1, len(history_vals) + forecast_days),
            [history_vals[-1]] + predictions, label="Forecast", marker="o", linestyle="--", color="red")
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"{selected_code}/USD Rate")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("👈 Select a currency and click 'Generate Forecast' to see predictions")