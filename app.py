
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="QuickCast", layout="wide")

# Sidebar Navigation
st.sidebar.title("QuickCast")
st.sidebar.markdown("Forecast-as-a-Service")
page = st.sidebar.radio("Go to", ["Home", "SKU Zoom", "Help & FAQ"])

# State initialization
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "data" not in st.session_state:
    st.session_state.data = None
if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False

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
        granularity = st.selectbox("Select data granularity", ["Daily", "Weekly", "Monthly"])
        horizon = st.selectbox("Select forecast horizon", ["Next 6 Days", "Next 6 Weeks", "Next 6 Months"])

        if st.button("Run Forecast"):
            st.info("🔮 Forecast engine not connected yet – placeholder only.")
            st.write("Once connected, you'll see summary tables and download links here.")

# --- SKU Zoom ---
elif page == "SKU Zoom":
    st.title("🔍 SKU Zoom")
    if not st.session_state.uploaded:
        st.warning("Please upload data in the Home page first.")
        st.stop()
    sku_list = st.session_state.data["Product Code"].unique()
    selected_sku = st.selectbox("Select a SKU", sku_list)
    df_sku = st.session_state.data[st.session_state.data["Product Code"] == selected_sku]
    st.write(f"Showing data for SKU: `{selected_sku}`")
    st.line_chart(df_sku.set_index("Date")["Value"])

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
