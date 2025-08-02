
import streamlit as st
import pandas as pd
from utils.forecast_engine import run_all_models
from utils.report_generator import generate_sku_report
from utils.copilot import generate_copilot_summary
from datetime import datetime

st.set_page_config(page_title="QuickCast", layout="wide")
st.sidebar.title("QuickCast")
page = st.sidebar.radio("Navigate", ["Home", "SKU Zoom", "Help"])

if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "kpis" not in st.session_state:
    st.session_state.kpis = {}
if "copilot" not in st.session_state:
    st.session_state.copilot = {}

if page == "Home":
    st.title("📊 QuickCast: Forecast-as-a-Service")

    st.markdown("### Step 1: Upload Your File")
    uploaded_file = st.file_uploader("Upload your .xlsx or .csv file", type=["xlsx", "csv"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("File uploaded successfully.")

    if st.session_state.data is not None:
        st.markdown("### Step 2: Forecast Settings")
        input_granularity = st.selectbox("Input data granularity", ["Daily", "Weekly", "Monthly"])
        output_granularity = st.selectbox("Output forecast granularity", ["Weekly", "Monthly"])
        horizon = st.selectbox("Forecast period", ["3 months", "6 months", "9 months"])

        if st.button("Run Forecast"):
            with st.spinner("Running forecasts..."):
                forecasts, kpis = run_forecast_for_all_skus(st.session_state.data, input_granularity, output_granularity, horizon)
                st.session_state.results = forecasts
                st.session_state.kpis = kpis
                st.session_state.copilot = {
                    sku: generate_copilot_summary(sku, df) for sku, df in forecasts.items()
                }
            st.success("Forecasts complete.")

elif page == "SKU Zoom":
    st.title("🔍 SKU Zoom")
    if not st.session_state.results:
        st.warning("Please run a forecast first.")
    else:
        sku = st.selectbox("Choose a SKU", list(st.session_state.results.keys()))
        st.dataframe(st.session_state.results[sku])
        st.markdown("**AI Copilot Insight:**")
        st.info(st.session_state.copilot.get(sku, "No insight available."))
        st.markdown("### Download Report")
        report = generate_sku_report(sku, st.session_state.results[sku], st.session_state.kpis[sku], st.session_state.copilot[sku])
        st.download_button("Download Report", report, file_name=f"{sku}_QuickCast_Report.xlsx")

elif page == "Help":
    st.title("❓ Help")
    st.markdown("Upload structured demand data with columns: `SKU`, `Date`, `Value`.")
