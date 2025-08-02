
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np

def preprocess_and_aggregate(sku_df, input_granularity, output_granularity):
    sku_df['Date'] = pd.to_datetime(sku_df['Date'])
    sku_df = sku_df.sort_values('Date')
    sku_df.set_index('Date', inplace=True)

    if output_granularity == 'Weekly':
        rule = 'W-MON'
    elif output_granularity == 'Monthly':
        rule = 'MS'
    else:
        rule = 'D'

    agg_df = sku_df.resample(rule).sum().reset_index()
    agg_df.rename(columns={'Date': 'ds', 'Quantity': 'y'}, inplace=True)
    return agg_df

def evaluate_forecast(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return mape, rmse, mae

def run_all_models(sku_df, forecast_periods, output_granularity):
    try:
        df = preprocess_and_aggregate(sku_df, None, output_granularity)

        if df.shape[0] < 6:
            raise ValueError("Not enough data after aggregation.")

        # Prophet model
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_periods, freq='W' if output_granularity == 'Weekly' else 'MS')
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat']].tail(forecast_periods)
        forecast_df['Model'] = 'Prophet'

        # KPIs (using last few periods from training)
        y_true = df['y'].values[-forecast_periods:]
        y_pred = model.predict(df[['ds']])['yhat'].values[-forecast_periods:]
        mape, rmse, mae = evaluate_forecast(y_true, y_pred)
        kpi_df = pd.DataFrame([{'Model': 'Prophet', 'MAPE': mape, 'RMSE': rmse, 'MAE': mae}])

        return forecast_df, kpi_df, None

    except Exception as e:
        return None, None, str(e)
