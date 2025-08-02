
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_all_models(sku_df, forecast_periods, output_freq):
    # Parse dates
    sku_df["Date"] = pd.to_datetime(sku_df["Date"])

    # Aggregate to output frequency
    if output_freq == "W":
        sku_df["Period"] = sku_df["Date"].dt.to_period("W").dt.start_time
    elif output_freq == "M":
        sku_df["Period"] = sku_df["Date"].dt.to_period("M").dt.to_timestamp()
    else:
        return None, None, "Invalid frequency"

    df = sku_df.groupby("Period").agg({"Quantity": "sum"}).reset_index()
    df = df.rename(columns={"Period": "ds", "Quantity": "y"})

    if len(df) < forecast_periods + 3:
        return None, None, "Insufficient data after aggregation"

    train = df[:-forecast_periods]
    test = df[-forecast_periods:]
    results = {}
    kpis = []

    # Naive
    try:
        naive_forecast = pd.Series([train["y"].iloc[-1]] * forecast_periods, index=test["ds"])
        kpis.append(("Naive", mape(test["y"], naive_forecast), mean_squared_error(test["y"], naive_forecast, squared=False), mean_absolute_error(test["y"], naive_forecast), (naive_forecast - test["y"]).mean()))
        results["Naive"] = naive_forecast
    except:
        pass

    # Moving Average
    try:
        ma_forecast = pd.Series([train["y"].rolling(3).mean().iloc[-1]] * forecast_periods, index=test["ds"])
        kpis.append(("Moving Average", mape(test["y"], ma_forecast), mean_squared_error(test["y"], ma_forecast, squared=False), mean_absolute_error(test["y"], ma_forecast), (ma_forecast - test["y"]).mean()))
        results["Moving Average"] = ma_forecast
    except:
        pass

    # ETS
    try:
        model_ets = ExponentialSmoothing(train["y"], seasonal=None, trend="add", initialization_method="estimated")
        fit_ets = model_ets.fit()
        ets_forecast = pd.Series(fit_ets.forecast(forecast_periods), index=test["ds"])
        kpis.append(("ETS", mape(test["y"], ets_forecast), mean_squared_error(test["y"], ets_forecast, squared=False), mean_absolute_error(test["y"], ets_forecast), (ets_forecast - test["y"]).mean()))
        results["ETS"] = ets_forecast
    except:
        pass

    # ARIMA
    try:
        model_arima = ARIMA(train["y"], order=(1, 1, 1))
        fit_arima = model_arima.fit()
        arima_forecast = pd.Series(fit_arima.forecast(steps=forecast_periods), index=test["ds"])
        kpis.append(("ARIMA", mape(test["y"], arima_forecast), mean_squared_error(test["y"], arima_forecast, squared=False), mean_absolute_error(test["y"], arima_forecast), (arima_forecast - test["y"]).mean()))
        results["ARIMA"] = arima_forecast
    except:
        pass

    # Prophet
    try:
        model_prophet = Prophet()
        train_prophet = train.rename(columns={"ds": "ds", "y": "y"})
        model_prophet.fit(train_prophet)
        future = model_prophet.make_future_dataframe(periods=forecast_periods, freq=output_freq)
        forecast = model_prophet.predict(future)
        prophet_forecast = forecast.set_index("ds").loc[test["ds"]]["yhat"]
        kpis.append(("Prophet", mape(test["y"], prophet_forecast), mean_squared_error(test["y"], prophet_forecast, squared=False), mean_absolute_error(test["y"], prophet_forecast), (prophet_forecast - test["y"]).mean()))
        results["Prophet"] = prophet_forecast
    except:
        pass

    kpi_df = pd.DataFrame(kpis, columns=["Model", "MAPE", "RMSE", "MAE", "Bias"])

    if kpi_df.empty:
        return None, None, "All models failed for this SKU."

    best_model = kpi_df.sort_values("MAPE").iloc[0]["Model"]
    forecast_values = results[best_model]

    forecast_result = pd.DataFrame({
        "SKU": sku_df["SKU"].iloc[0],
        "Date": test["ds"],
        "Forecast/Actual": "Forecast",
        "Forecast Method": best_model,
        "Quantity": forecast_values.values
    })

    return forecast_result, kpi_df, None
