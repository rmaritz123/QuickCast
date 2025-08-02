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
        kpis = {
            "Model": model_name,
            "MAPE": model_mape,
            "RMSE": model_rmse,
            "MAE": model_mae,
            "Bias": bias
        }
        return pred, kpis
    except Exception:
        return None, None

def forecast_future(train_full, best_model, prophet_freq, forecast_periods):
    try:
        if best_model == "Naive":
            last = train_full["y"].iloc[-1]
            if prophet_freq == "W":
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="W-MON")[1:]
            else:
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="MS")[1:]
            forecast = pd.Series([last] * forecast_periods, index=future_index)
        elif best_model == "Moving Average":
            window = 3
            if len(train_full["y"]) < window:
                val = train_full["y"].iloc[-1]
            else:
                val = train_full["y"].rolling(window).mean().iloc[-1]
            if prophet_freq == "W":
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="W-MON")[1:]
            else:
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="MS")[1:]
            forecast = pd.Series([val] * forecast_periods, index=future_index)
        elif best_model == "ETS":
            model = ExponentialSmoothing(train_full["y"], trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            if prophet_freq == "W":
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="W-MON")[1:]
            else:
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="MS")[1:]
            forecast = pd.Series(fit.forecast(forecast_periods), index=future_index)
        elif best_model == "ARIMA":
            model = ARIMA(train_full["y"], order=(1, 1, 1))
            fit = model.fit()
            if prophet_freq == "W":
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="W-MON")[1:]
            else:
                future_index = pd.date_range(start=train_full["ds"].iloc[-1], periods=forecast_periods+1, freq="MS")[1:]
            forecast = pd.Series(fit.forecast(steps=forecast_periods), index=future_index)
        elif best_model == "Prophet":
            prophet = Prophet()
            prophet.fit(train_full)
            future = prophet.make_future_dataframe(periods=forecast_periods, freq=prophet_freq)
            forecast_df = prophet.predict(future)
            forecast = forecast_df.set_index("ds").tail(forecast_periods)["yhat"]
        else:
            return None
        return forecast
    except Exception:
        return None

def run_all_models_selfcontained(sku_df, forecast_periods, output_granularity):
    try:
        agg_df, prophet_freq = aggregate_series(sku_df, output_granularity)
        if len(agg_df) < forecast_periods + 3:
            return None, None, "Not enough data after aggregation"

        train = agg_df[:-forecast_periods]
        test = agg_df[-forecast_periods:]

        model_names = ["Naive", "Moving Average", "ETS", "ARIMA", "Prophet"]
        kpi_list = []
        preds = {}
        for name in model_names:
            pred_test, kpis = evaluate_model(train, test, name, prophet_freq)
            if pred_test is not None:
                kpi_list.append(kpis)
                preds[name] = pred_test

        if not kpi_list:
            return None, None, "All models failed"

        kpi_df = pd.DataFrame(kpi_list)
        best_row = kpi_df.sort_values("MAPE").iloc[0]
        best_model = best_row["Model"]
        future_forecast = forecast_future(agg_df, best_model, prophet_freq, forecast_periods)
        if future_forecast is None:
            return None, kpi_df, "Future forecast failed"
        result = pd.DataFrame({
            "ds": future_forecast.index,
            "yhat": future_forecast.values,
            "Model": best_model
        })
        return result, kpi_df, None
    except Exception as e:
        return None, None, str(e)

# ---------------- UI ----------------
def aggregate_history(sku_df, output_granularity):
    df = sku_df.copy()
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    qty_col = next((c for c in df.columns if c.lower() in ["quantity", "qty", "value"]), None)
    if qty_col is None:
        return None
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)
    if output_granularity == "Weekly":
        rule = "W-MON"
    elif output_granularity == "Monthly":
        rule = "MS"
    else:
        rule = "D"
    agg = df.resample(rule).sum().reset_index()
    agg.rename(columns={date_col: "Date"}, inplace=True)
    agg = agg.rename(columns={qty_col: "Quantity"})
    return agg

# --- Home Page ---
if page == "Home":
    st.title("QuickCast: Forecast-as-a-Service")
    st.markdown("## Step 1: Accept Terms & Conditions")
    terms = st.checkbox(
        "I confirm that the data I’m uploading is anonymized and that I accept the QuickCast Terms & Conditions."
    )
    if not terms:
        st.warning("You must accept the Terms & Conditions to proceed.")
        st.stop()

    st.markdown("## Step 2: Upload Data")
    uploaded = st.file_uploader("Upload Excel or CSV with columns: SKU, Date (or Invoice Date), Quantity", type=["xlsx", "csv"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.data = df
            st.success("File uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read upload: {e}")
            st.stop()

    if st.session_state.data is not None:
        st.markdown("## Step 3: Forecast Settings")
        _ = st.selectbox("1️⃣ Granularity of uploaded data (informational)", ["Daily", "Weekly", "Monthly"])
        output_granularity = st.selectbox("2️⃣ Output forecast granularity", ["Weekly", "Monthly"])
        horizon = st.selectbox("3️⃣ Forecast horizon", ["3 months", "6 months", "9 months"])

        if output_granularity == "Weekly":
            horizon_map = {"3 months": 12, "6 months": 24, "9 months": 36}
        else:
            horizon_map = {"3 months": 3, "6 months": 6, "9 months": 9}
        forecast_periods = horizon_map[horizon]

        if st.button("Run Forecast"):
            data = st.session_state.data
            if "SKU" not in data.columns:
                st.error("Input file must contain a 'SKU' column.")
                st.stop()

            skus = data["SKU"].unique()
            all_output_rows = []
            kpi_frames = []
            best_model_summary = []

            for sku in skus:
                sku_df = data[data["SKU"] == sku].copy()
                forecast_result, kpi_df, error = run_all_models_selfcontained(sku_df, forecast_periods, output_granularity)
                if error:
                    st.warning(f"Skipping SKU {sku}: {error}")
                    continue

                # Historical aggregation
                hist = aggregate_history(sku_df, output_granularity)
                if hist is not None:
                    for _, row in hist.iterrows():
                        date = row["Date"]
                        qty = row["Quantity"]
                        week_num = int(pd.to_datetime(date).isocalendar().week)
                        month_name = pd.to_datetime(date).strftime("%B")
                        all_output_rows.append({
                            "SKU": sku,
                            "Date": date,
                            "Week Number": week_num,
                            "Month Name": month_name,
                            "Forecast/Actual": "Actual",
                            "Forecast Method": "-",
                            "Quantity": qty
                        })

                # Forecast rows
                if forecast_result is not None and "ds" in forecast_result.columns:
                    method = forecast_result["Model"].iloc[0] if "Model" in forecast_result.columns else "Prophet"
                    for _, prow in forecast_result.iterrows():
                        date = prow["ds"]
                        qty = prow.get("yhat", np.nan)
                        week_num = int(pd.to_datetime(date).isocalendar().week)
                        month_name = pd.to_datetime(date).strftime("%B")
                        all_output_rows.append({
                            "SKU": sku,
                            "Date": date,
                            "Week Number": week_num,
                            "Month Name": month_name,
                            "Forecast/Actual": "Forecast",
                            "Forecast Method": method,
                            "Quantity": qty
                        })
                else:
                    st.warning(f"Unexpected forecast structure for SKU {sku}; skipping forecast rows.")

                # KPI summary
                if kpi_df is not None:
                    kpi_df["SKU"] = sku
                    kpi_frames.append(kpi_df)
                    if "MAPE" in kpi_df.columns:
                        best = kpi_df.sort_values("MAPE").iloc[0]
                        best_model_summary.append({
                            "SKU": sku,
                            "Best Model": best["Model"],
                            "MAPE": best["MAPE"],
                            "RMSE": best.get("RMSE", np.nan),
                            "MAE": best.get("MAE", np.nan)
                        })

            if all_output_rows:
                final_df = pd.DataFrame(all_output_rows)
                final_df = final_df.sort_values(["SKU", "Date"])
                st.session_state.forecast_combined = final_df

                if kpi_frames:
                    combined_kpis = pd.concat(kpi_frames, ignore_index=True)
                    st.session_state.kpis = combined_kpis

                if best_model_summary:
                    summary_df = pd.DataFrame(best_model_summary)
                    st.markdown("### Forecast Summary (Best Model per SKU)")
                    st.dataframe(summary_df)

                ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
                filename = f"QuickCast_Forecast_{ts}.xlsx"
                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    final_df.to_excel(writer, index=False, sheet_name="Forecasts")
                with open(filename, "rb") as f:
                    st.download_button("Download Full Forecast Output", f, file_name=filename)
            else:
                st.error("No forecast data generated for any SKU.")

# --- SKU Zoom ---
elif page == "SKU Zoom":
    st.title("🔍 SKU Zoom")
    if st.session_state.forecast_combined is None:
        st.warning("Run a forecast first on the Home page.")
        st.stop()

    df_all = st.session_state.forecast_combined
    skus = df_all["SKU"].unique()
    selected = st.selectbox("Select SKU", skus)
    df_sku = df_all[df_all["SKU"] == selected]

    pivot = df_sku.pivot(index="Date", columns="Forecast/Actual", values="Quantity")
    st.markdown(f"### SKU: {selected}")
    st.bar_chart(pivot)

    if st.session_state.kpis is not None:
        kpi_df = st.session_state.kpis
        sku_kpis = kpi_df[kpi_df["SKU"] == selected]
        if not sku_kpis.empty:
            st.markdown("### KPI Comparison")
            st.dataframe(sku_kpis)

# --- Help & FAQ ---
elif page == "Help & FAQ":
    st.title("❓ Help & FAQ")
    st.markdown(
        """
### 🧾 Accepted File Formats
- Excel (.xlsx)
- CSV (.csv)

### 📂 Required Columns
- SKU
- Date or Invoice Date
- Quantity

### 🧮 Forecast Settings
1. Input granularity (informational)
2. Output granularity (Weekly or Monthly)
3. Forecast horizon (3, 6, or 9 months)

### 📈 Output
Downloads a combined Excel with historical (Actual) and forecasted values per SKU, with best model summary.

### ❓ Common Issues
- No date column found → ensure one of the columns contains 'date' in its name.
- No quantity column found → include 'Quantity', 'Qty', or 'Value'.
"""
    )
