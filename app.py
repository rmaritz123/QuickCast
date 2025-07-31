
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

st.set_page_config(page_title="QuickCast", layout="wide")

# Sidebar Navigation
st.sidebar.title("QuickCast")
st.sidebar.markdown("Forecast-as-a-Service")
page = st.sidebar.radio("Go to", ["Home", "SKU Zoom", "Help & FAQ"])

# Session State
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "data" not in st.session_state:
    st.session_state.data = None
if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = {}

# --- Forecast Engine ---
def run_simple_forecast(df, sku, freq, periods):
    df["Date"] = pd.to_datetime(df["Date"])
    sku_df = df[df["SKU"] == sku].copy()
    sku_df = sku_df[["Date", "Value"]]
    sku_df = sku_df.set_index("Date").sort_index()
    sku_df = sku_df.resample(freq).sum()

    result_df = sku_df.reset_index().rename(columns={"Date": "ds", "Value": "y"})
    if len(result_df) < 12:
        return None, "Not enough data points for forecasting."

    try:
        model = Prophet()
        model.fit(result_df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].tail(periods), None
    except Exception as e:
        return None, str(e)

# --- Home Page ---
if page == "Home":
    st.title("QuickCast: Forecast-as-a-Service")

    st.markdown("### Step 1: Accept Terms & Conditions")
    st.session_state.terms_accepted = st.checkbox(
        "I confirm that the data I’m uploading is anonymized and that I accept the QuickCast [Terms & Conditions](#).")

    if not st.session_state.terms_accepted:
        st.warning("You must accept the Terms & Conditions before proceeding.")
        st.stop()

    st.markdown("### Step 2: Upload Your Data File")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = [col.strip() for col in df.columns]
            df.rename(columns={df.columns[0]: "SKU", df.columns[1]: "Date", df.columns[2]: "Value"}, inplace=True)
            st.session_state.data = df
            st.session_state.uploaded = True
            st.success("✅ File uploaded successfully!")
            st.write("Preview of your data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    if st.session_state.uploaded:
        st.markdown("### Step 3: Choose Forecast Settings")

        input_freq = st.selectbox("1️⃣ What is the granularity of your uploaded data?", ["Daily", "Weekly", "Monthly"])
        output_freq = st.selectbox("2️⃣ What output forecast format do you want?", ["6 Days Ahead", "6 Weeks Ahead", "6 Months Ahead"])

        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        period_map = {
            "6 Days Ahead": ("D", 6),
            "6 Weeks Ahead": ("W", 6),
            "6 Months Ahead": ("M", 6)
        }

        if st.button("Run Forecast"):
            forecast_results = {}
            forecast_freq, periods = period_map[output_freq]
            for sku in st.session_state.data["SKU"].unique():
                forecast, error = run_simple_forecast(st.session_state.data, sku, freq_map[input_freq], periods)
                if forecast is not None:
                    forecast_results[sku] = forecast
                else:
                    forecast_results[sku] = pd.DataFrame({"Error": [error]})
            st.session_state.forecast_result = forecast_results
            st.success("✅ Forecast complete! Check the SKU Zoom tab.")
            st.write("Sample Forecast Output:")
            st.write(forecast_results[list(forecast_results.keys())[0]])

# --- SKU Zoom ---
elif page == "SKU Zoom":
    st.title("🔍 SKU Zoom")
    if not st.session_state.uploaded or not st.session_state.forecast_result:
        st.warning("Please upload data and run a forecast in the Home page first.")
        st.stop()

    sku_list = list(st.session_state.forecast_result.keys())
    selected_sku = st.selectbox("Select a SKU", sku_list)
    df_original = st.session_state.data[st.session_state.data["SKU"] == selected_sku]
    st.subheader("📊 Historical Data")
    st.line_chart(df_original.set_index("Date")["Value"])

    forecast_df = st.session_state.forecast_result[selected_sku]
    if "yhat" in forecast_df.columns:
        st.subheader("🔮 Forecast (Prophet)")
        st.line_chart(forecast_df.set_index("ds")["yhat"])
    else:
        st.warning(forecast_df.iloc[0]["Error"])

# --- Help Page ---
elif page == "Help & FAQ":
    st.title("❓ Help & FAQ")

    st.markdown(""" 
### 🧾 What file formats are accepted?
- Excel (.xlsx)
- CSV (.csv)

### 📅 What date formats are allowed?
We support formats like `dd-mm-yyyy`, `yyyy-mm-dd`, `dd/mm/yyyy`, `dd.mm.yyyy`

### 🧠 What is QuickCast?
QuickCast is a Forecast-as-a-Service tool designed for non-technical users to generate 6-period forecasts per SKU.
""")

    st.markdown("### 📂 Download Templates")
    try:
        with open("data/sample_input.xlsx", "rb") as f:
            st.download_button("Download Sample Template", data=f, file_name="quickcast_template.xlsx")
    except FileNotFoundError:
        st.info("Sample file not available yet.")
