import sys
import os
from io import BytesIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

from feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    drop_na_rows
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Retail Sales Intelligence Platform")
st.caption("Weekly Store & Category Revenue Forecasting")
st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/weekly_store_category_revenue.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Analysis Filters")

stores = sorted(df["StoreID"].unique())
selected_store = st.sidebar.selectbox("Store", stores)

store_df = df[df["StoreID"] == selected_store]
categories = sorted(store_df["Category"].unique())
selected_category = st.sidebar.selectbox("Category", categories)

st.sidebar.divider()
st.sidebar.caption("Retail Forecast v2.1.0")

segment_df = store_df[store_df["Category"] == selected_category].copy()

# --------------------------------------------------
# DATA COVERAGE STATS (NEW)
# --------------------------------------------------
start_date = segment_df["Date"].min().strftime("%b %Y")
end_date = segment_df["Date"].max().strftime("%b %Y")
total_weeks = segment_df["Date"].nunique()

st.markdown(
    f"**Data Coverage:** {start_date} – {end_date}  |  "
    f"**Total Weeks:** {total_weeks}"
)

st.divider()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
segment_df = create_time_features(segment_df)
segment_df = create_lag_features(segment_df)
segment_df = create_rolling_features(segment_df)
segment_df = drop_na_rows(segment_df)

features = [
    "year",
    "week_of_year",
    "quarter",
    "lag_1",
    "lag_4",
    "lag_12",
    "rolling_mean_4",
    "rolling_mean_12",
]

model_path = f"models/weekly_model_{selected_store}_{selected_category}.pkl"

if not os.path.exists(model_path):
    st.warning("Model not available for this segment.")
    st.stop()

model = joblib.load(model_path)

segment_df["ML_Prediction"] = model.predict(segment_df[features])
segment_df["Baseline_Prediction"] = segment_df["lag_1"]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred))
    )

rf_mae = np.mean(np.abs(segment_df["Revenue"] - segment_df["ML_Prediction"]))
baseline_mae = np.mean(np.abs(segment_df["Revenue"] - segment_df["Baseline_Prediction"]))
rf_smape = smape(segment_df["Revenue"], segment_df["ML_Prediction"])

improvement = ((baseline_mae - rf_mae) / baseline_mae) * 100
weekly_savings = baseline_mae - rf_mae
annual_savings = weekly_savings * 52

# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------
st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Random Forest MAE", f"${rf_mae:,.0f}")
col2.metric("Baseline MAE", f"${baseline_mae:,.0f}")
col3.metric("Improvement", f"{improvement:.1f}%")
col4.metric("SMAPE", f"{rf_smape:.2f}%")

st.markdown(
    f"""
    <div style="
        background-color: #1f4d2b;
        padding: 16px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        margin-top: 10px;
    ">
        Estimated Financial Impact:
        <strong>${weekly_savings:,.0f}</strong> weekly savings
        (~<strong>${annual_savings:,.0f}</strong> annualized)
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# MODEL INFO PANEL (NEW)
# --------------------------------------------------
with st.expander("Model Information"):
    st.write("**Model Type:** Random Forest Regressor")
    st.write("**Forecast Granularity:** Weekly")
    st.write("**Forecast Horizon:** 4 Weeks Ahead")
    st.write("**Features Used:**")
    st.write(", ".join(features))
    st.write("**Training Scope:** Store & Category specific model")

st.divider()

# --------------------------------------------------
# TABS (UNCHANGED)
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Revenue Overview",
    "Model Performance",
    "Forecast"
])

# --------------------------------------------------
# TAB 1 — HISTORICAL
# --------------------------------------------------
with tab1:
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(segment_df["Date"], segment_df["Revenue"], label="Actual")
    ax1.set_title("Historical Revenue")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Revenue ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

# --------------------------------------------------
# TAB 2 — MODEL COMPARISON
# --------------------------------------------------
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(segment_df["Date"], segment_df["Revenue"], label="Actual")
    ax2.plot(segment_df["Date"], segment_df["ML_Prediction"], linestyle="--", label="Random Forest")
    ax2.plot(segment_df["Date"], segment_df["Baseline_Prediction"], linestyle=":", label="Baseline")
    ax2.set_title("Actual vs Model Comparison")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Revenue ($)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# --------------------------------------------------
# TAB 3 — FORECAST
# --------------------------------------------------
with tab3:
    last_row = segment_df.iloc[-1:].copy()
    future_predictions = []

    for _ in range(4):
        pred = model.predict(last_row[features])[0]
        future_predictions.append(pred)
        last_row["lag_12"] = last_row["lag_4"]
        last_row["lag_4"] = last_row["lag_1"]
        last_row["lag_1"] = pred

    forecast_df = pd.DataFrame({
        "Week Ahead": range(1, 5),
        "Predicted Revenue ($)": [f"${val:,.2f}" for val in future_predictions]
    })

    st.dataframe(forecast_df, use_container_width=True)

st.divider()

# --------------------------------------------------
# PDF REPORT (UNCHANGED)
# --------------------------------------------------
def generate_pdf():

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Retail Sales Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Store: {selected_store}", styles["Normal"]))
    elements.append(Paragraph(f"Category: {selected_category}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Random Forest MAE", f"${rf_mae:,.0f}"],
        ["Baseline MAE", f"${baseline_mae:,.0f}"],
        ["Improvement", f"{improvement:.1f}%"],
        ["SMAPE", f"{rf_smape:.2f}%"],
        ["Weekly Savings", f"${weekly_savings:,.0f}"],
        ["Annualized Impact", f"${annual_savings:,.0f}"],
    ]

    table = Table(data, colWidths=[250,150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 24))

    img_buffer = BytesIO()
    fig2.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    elements.append(RLImage(img_buffer, width=6*inch, height=3*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf_file = generate_pdf()

st.download_button(
    label="Download Executive Report (PDF)",
    data=pdf_file,
    file_name=f"{selected_store}_{selected_category}_forecast_report.pdf",
    mime="application/pdf"
)

st.divider()
st.caption("© 2026 Retail Sales Intelligence | Production ML System")