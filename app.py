
import streamlit as st
import pandas as pd
from utils.forecast_engine import run_all_models
from datetime import datetime

st.set_page_config(page_title="QuickCast", layout="wide")

# Sidebar Navigation
st.sidebar.title("QuickCast")
st.sidebar.markdown("Forecast-as-a-Service")
page = st.sidebar.radio("Go to", ["Home", "SKU Zoom", "Help & FAQ"])

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "data" not in st.session_state:
    st.session_state.data = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = {}
if "kpis" not in st.session_state:
    st.session_state.kpis = {}

# --- Home Page ---
if page == "Home":
    st.title("QuickCast: Forecast-as-a-Service")

    st.markdown("### Step 1: Accept Terms & Conditions")
    terms = st.checkbox(
        "I confirm that the data I’m uploading is anonymized and that I accept the QuickCast Terms & Conditions.")

    if not terms:
        st.warning("You must accept the Terms & Conditions before proceeding.")
        st.stop()

    st.markdown("### Step 2: Upload Your Data File")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
            df.columns = [col.strip() for col in df.columns]
            df.rename(columns={df.columns[0]: "SKU", df.columns[1]: "Date", df.columns[2]: "Value"}, inplace=True)
            st.session_state.data = df
            st.session_state.uploaded = True
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    if st.session_state.uploaded:
        st.markdown("### Step 3: Choose Forecast Settings")

        input_granularity = st.selectbox("1️⃣ What is the granularity of your uploaded data?", ["Daily", "Weekly", "Monthly"])
        output_granularity = st.selectbox("2️⃣ What output forecast format do you want?", ["Weekly", "Monthly"])
        horizon = st.selectbox("3️⃣ Forecast period", ["3 months", "6 months", "9 months"])

        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        horizon_map = {"3 months": 3, "6 months": 6, "9 months": 9}
        forecast_periods = horizon_map[horizon]
        freq = freq_map[output_granularity]

        if st.button("Run Forecast"):
            all_skus = st.session_state.data["SKU"].unique()
            combined_forecasts = []
            combined_kpis = []

            with st.spinner("Running forecasts for each SKU..."):
                for sku in all_skus:
                    sku_df = st.session_state.data[st.session_state.data["SKU"] == sku]
                    forecast_df, kpi_df, error = run_all_models(sku_df, forecast_periods, freq)

                    if forecast_df is not None:
                        combined_forecasts.append(forecast_df)
                        kpi_df["SKU"] = sku
                        combined_kpis.append(kpi_df)
                    else:
                        st.warning(f"Skipping SKU {sku}: {error}")

                if combined_forecasts:
                    full_forecast_df = pd.concat(combined_forecasts)
                    st.session_state.forecast_result = full_forecast_df

                    if combined_kpis:
                        full_kpi_df = pd.concat(combined_kpis)
                        st.session_state.kpis = full_kpi_df

                    # Save file
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
                    filename = f"QuickCast_Forecast_{timestamp}.xlsx"
                    full_forecast_df.to_excel(filename, index=False)
                    with open(filename, "rb") as f:
                        st.download_button("Download Forecast Results", f, file_name=filename)

# --- SKU Zoom ---
elif page == "SKU Zoom":
    st.title("🔍 SKU Zoom")
    if not st.session_state.uploaded or st.session_state.forecast_result is None:
        st.warning("Please upload data and run a forecast in the Home page first.")
        st.stop()

    sku_list = st.session_state.forecast_result["SKU"].unique()
    selected_sku = st.selectbox("Select a SKU", sku_list)
    df_zoom = st.session_state.forecast_result[st.session_state.forecast_result["SKU"] == selected_sku]

    st.bar_chart(df_zoom.set_index("Date")["Quantity"])

    st.markdown("### KPI Comparison")
    kpi_df = st.session_state.kpis
    kpi_selected = kpi_df[kpi_df["SKU"] == selected_sku]
    st.dataframe(kpi_selected)

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
