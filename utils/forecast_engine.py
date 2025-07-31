
import pandas as pd
import numpy as np

def run_forecast_for_all_skus(df, input_freq, output_freq, horizon_text):
    skus = df["SKU"].unique()
    forecasts = {}
    kpis = {}
    for sku in skus:
        sku_data = df[df["SKU"] == sku]
        # Placeholder logic for demo
        forecast = sku_data.tail(3).copy()
        forecast["Forecast/Actual"] = "Forecast"
        forecast["Forecast Method"] = "Prophet"
        forecasts[sku] = forecast
        kpis[sku] = {
            "Model": ["Prophet"],
            "MAPE": [6.5],
            "RMSE": [12.4],
            "MAE": [8.9],
            "Bias": [0.2]
        }
    return forecasts, kpis
