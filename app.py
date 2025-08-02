import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.forecast_engine import run_all_models

st.set_page_config(page_title="QuickCast", layout="wide")

# Sidebar
st.sidebar.title("QuickCast")
st.sidebar.markdown("Forecast-as-a-Service")
page = st.sidebar.radio("Navigate", ["Home", "SKU Zoom", "Help & FAQ"])

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "forecast_combined" not in st.session_state:
    st.session_state.forecast_combined = None
if "kpis" not in st.session_state:
    st.session_state.kpis = None

# Utility: aggregate historical per SKU given output granularity
def aggregate_history(sku_df, output_granularity):
    df = sku_df.copy()
    # Detect date column
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        return None
    df[date_col] = pd.to_datetime(df[date_col])
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
    # Identify quantity column
    qty_col = next((c for c in agg.columns if c.lower() in ["quantity", "qty", "value"]), None)
    if qty_col is None:
        return None
    agg = agg.rename(columns={qty_col: "Quantity"})
    return agg

# Home Page
if page == "Home":
    st.title("QuickCast: Forecast-as-a-Service")
    st.markdown("## Step 1: Accept Terms & Conditions")
    terms = st.checkbox("I confirm that the data I’m uploading is anonymized and that I accept the QuickCast Terms & Conditions.")
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
        input_granularity = st.selectbox("1️⃣ Granularity of uploaded data (informational)", ["Daily", "Weekly", "Monthly"])
        output_granularity = st.selectbox("2️⃣ Output forecast granularity", ["Weekly", "Monthly"])
        horizon = st.selectbox("3️⃣ Forecast horizon", ["3 months", "6 months", "9 months"])

        # Determine number of forecast periods based on output granularity
        if output_granularity == "Weekly":
            horizon_map = {"3 months": 12, "6 months": 24, "9 months": 36}
        else:
            horizon_map = {"3 months": 3, "6 months": 6, "9 months": 9}
        forecast_periods = horizon_map[horizon]

        if st.button("Run Forec
