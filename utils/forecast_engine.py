
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    # avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

def aggregate_series(sku_df, output_granularity):
    # detect date column
    date_col = next((c for c in sku_df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("No date column found in SKU data.")
    sku_df = sku_df.copy()
    sku_df[date_col] = pd.to_datetime(sku_df[date_col])
    # detect quantity column
    qty_col = next((c for c in ["Quantity", "Qty", "Value"] if c in sku_df.columns), None)
    if qty_col is None:
        raise ValueError("No quantity column found in SKU data.")
    # set index
    sku_df = sku_df.sort_values(date_col)
    sku_df.set_index(date_col, inplace=True)

    # determine rule
    if output_granularity in ("W", "Weekly"):
        rule = "W-MON"
        prophet_freq = "W"
    elif output_granularity in ("M", "Monthly"):
        rule = "MS"
        prophet_freq = "MS"
    else:
        raise ValueError("Invalid output granularity: expected Weekly or Monthly.")

    agg = sku_df.resample(rule)[qty_col].sum().reset_index()
    agg = agg.rename(columns={date_col: "ds", qty_col: "y"})
    return agg, prophet_freq

def evaluate_model(train, test, model_name, forecast_periods, prophet_freq):
    results = {}
    kpis = []
    try:
        # Naive
        if model_name == "Naive":
            last = train["y"].iloc[-1]
            pred_test = pd.Series([last] * len(test), index=test["ds"])
        elif model_name == "Moving Average":
            window = 3
            ma = train["y"].rolling(window).mean().iloc[-1]
            if np.isnan(ma):
                raise ValueError("Not enough data for moving average")
            pred_test = pd.Series([ma] * len(test), index=test["ds"])
        elif model_name == "ETS":
            model = ExponentialSmoothing(train["y"], trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            pred_test = pd.Series(fit.forecast(len(test)), index=test["ds"])
        elif model_name == "ARIMA":
            model = ARIMA(train["y"], order=(1, 1, 1))
            fit = model.fit()
            pred_test = pd.Series(fit.forecast(steps=len(test)), index=test["ds"])
        elif model_name == "Prophet":
            prophet = Prophet()
            prophet.fit(train)
            future = prophet.make_future_dataframe(periods=len(test), freq=prophet_freq)
            forecast = prophet.predict(future)
            # align on test ds
            pred_test = forecast.set_index("ds").loc[test["ds"]]["yhat"]
        else:
            return None, None  # unknown model

        # KPI calculation
        y_true = test["y"].values
        y_pred = pred_test.values
        model_mape = mape(test["y"].values, pred_test.values)
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_mae = mean_absolute_error(y_true, y_pred)
        bias = (pred_test - test["y"]).mean()
        kpis = {"Model": model_name, "MAPE": model_mape, "RMSE": model_rmse, "MAE": model_mae, "Bias": bias}
        results = pred_test
        return results, kpis
    except Exception as e:
        # return failure without raising to allow other models to run
        return None, None

def fit_best_and_forecast_full(df, best_model, forecast_periods, prophet_freq):
    # df includes full aggregated history
    # Generate forecast for next periods
    try:
        if best_model == "Naive":
            last = df["y"].iloc[-1]
            future_index = pd.date_range(start=df["ds"].iloc[-1], periods=forecast_periods + 1, freq=prophet_freq)[1:]
            forecast_future = pd.Series([last] * forecast_periods, index=future_index)
        elif best_model == "Moving Average":
            window = 3
            ma = df["y"].rolling(window).mean().iloc[-1]
            future_index = pd.date_range(start=df["ds"].iloc[-1], periods=forecast_periods + 1, freq=prophet_freq)[1:]
            forecast_future = pd.Series([ma] * forecast_periods, index=future_index)
        elif best_model == "ETS":
            model = ExponentialSmoothing(df["y"], trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            forecast_future = pd.Series(fit.forecast(forecast_periods), index=pd.date_range(start=df["ds"].iloc[-1], periods=forecast_periods + 1, freq=prophet_freq)[1:])
        elif best_model == "ARIMA":
            model = ARIMA(df["y"], order=(1, 1, 1))
            fit = model.fit()
            forecast_future = pd.Series(fit.forecast(steps=forecast_periods), index=pd.date_range(start=df["ds"].iloc[-1], periods=forecast_periods + 1, freq=prophet_freq)[1:])
        elif best_model == "Prophet":
            prophet = Prophet()
            prophet.fit(df)
            future = prophet.make_future_dataframe(periods=forecast_periods, freq=prophet_freq)
            forecast = prophet.predict(future)
            forecast_future = forecast.set_index("ds").iloc[-forecast_periods:]["yhat"]
        else:
            return None
        return forecast_future
    except Exception:
        return None

def run_all_models(sku_df, forecast_periods, output_granularity):
    try:
        agg_df, prophet_freq = aggregate_series(sku_df, output_granularity)

        if len(agg_df) < forecast_periods + 3:
            return None, None, "Insufficient data after aggregation"

        # Backtest portion
        train = agg_df[:-forecast_periods]
        test = agg_df[-forecast_periods:]

        model_names = ["Naive", "Moving Average", "ETS", "ARIMA", "Prophet"]
        kpi_list = []
        model_predictions = {}
        for model_name in model_names:
            pred_test, kpis = evaluate_model(train, test, model_name, forecast_periods, prophet_freq)
            if pred_test is not None:
                kpi_list.append(kpis)
                model_predictions[model_name] = pred_test

        if not kpi_list:
            return None, None, "All models failed for this SKU."

        kpi_df = pd.DataFrame(kpi_list)
        # select best based on MAPE (lower is better)
        best_model = kpi_df.sort_values("MAPE").iloc[0]["Model"]

        # Forecast future using best model on full history
        forecast_future = fit_best_and_forecast_full(agg_df, best_model, forecast_periods, prophet_freq)
        if forecast_future is None:
            return None, kpi_df, "Failed generating future forecast"

        # Build forecast result DataFrame
        future_dates = forecast_future.index
        forecast_result = pd.DataFrame({
            "SKU": sku_df.iloc[0][next((c for c in sku_df.columns if "sku" in c.lower()), "SKU")],
            "Date": future_dates,
            "Forecast/Actual": "Forecast",
            "Forecast Method": best_model,
            "Quantity": forecast_future.values
        })

        return forecast_result, kpi_df, None
    except Exception as e:
        return None, None, str(e)
