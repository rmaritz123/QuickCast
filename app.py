import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="QuickCast", layout="wide")

# Sidebar
st.sidebar.title("QuickCast")
st.sidebar.markdown("Forecast-as-a-Service")
page = st.sidebar.radio("Navigate", ["Home", "SKU Zoom", "Help & FAQ"])

# Session state
if "data" not in st.session_state:
    st.session_state.data = None
if "forecast_combined" not in st.session_state:
    st.session_state.forecast_combined = None
if "kpis" not in st.session_state:
    st.session_state.kpis = None

# ---------------- Forecasting helpers ----------------

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def aggregate_series(sku_df, output_granularity):
    df = sku_df.copy()
    # detect date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("No date-like column found.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # detect quantity column
    qty_col = next((c for c in df.columns if c.lower() in ["quantity", "qty", "value"]), None)
    if qty_col is None:
        raise ValueError("No quantity column found.")

    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    if output_granularity == "Weekly":
        rule = "W-MON"
        prophet_freq = "W"
    elif output_granularity == "Monthly":
        rule = "MS"
        prophet_freq = "MS"
    else:
        raise ValueError("Unsupported output granularity.")

    agg = df.resample(rule)[qty_col].sum().reset_index()
    agg = agg.rename(columns={date_col: "ds", qty_col: "y"})
    return agg, prophet_freq

def evaluate_model(train, test, model_name, prophet_freq):
    try:
        if model_name == "Naive":
            pred = pd.Series([train["y"].iloc[-1]] * len(test), index=test["ds"])
        elif model_name == "Moving Average":
            window = 3
            if len(train["y"]) < window:
                val = train["y"].iloc[-1]
            else:
                val = train["y"].rolling(window).mean().iloc[-1]
            pred = pd.Series([val] * len(test), index=test["ds"])
        elif model_name == "ETS":
            model = ExponentialSmoothing(train["y"], trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            pred = pd.Series(fit.forecast(len(test)), index=test["ds"])
        elif model_name == "ARIMA":
            model = ARIMA(train["y"], order=(1, 1, 1))
            fit = model.fit()
            pred = pd.Series(fit.forecast(steps=len(test)), index=test["ds"])
        elif model_name == "Prophet":
            prophet = Prophet()
            prophet.fit(train)
            future = prophet.make_future_dataframe(periods=len(test), freq=prophet_freq)
            forecast_df = prophet.predict(future)
            pred = forecast_df.set_index("ds").loc[test["ds"]]["yhat"]
        else:
            return None, None

        y_true = test["y"]
        y_pred = pred
        model_mape = safe_mape(y_true.values, y_pred.values)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_mae = mean_absolute_error(y_true, y_pred)
        bias = (y_pred - y_true).mean()
        kpis
