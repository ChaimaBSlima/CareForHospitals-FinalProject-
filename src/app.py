from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

# Settings

st.set_page_config(
    page_title="HorizonCare — Next Week Hospital Stress Forecast",
    layout="wide",
)

FORECAST_PATH = Path("data/cleaned/next_week_forecast_enhanced.csv")

# Helpers

@st.cache_data
def load_forecast(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found at: {path.as_posix()}\n"
            "Run: python src/predict_next_week.py"
        )
    df = pd.read_csv(path)

    # Basic cleanup
    if "current_week" in df.columns:
        df["current_week"] = pd.to_datetime(df["current_week"], errors="coerce")
    if "forecast_week" in df.columns:
        df["forecast_week"] = pd.to_datetime(df["forecast_week"], errors="coerce")

    return df


def kpi_card(label: str, value: str):
    st.metric(label, value)


# ----------------------------
# App
# ----------------------------
st.title("🏥 HorizonCare — Next Week Hospital Stress Forecast")
st.caption("State-level ICU/Inpatient forecasting + critical stress early warning + disease burden.")

df = load_forecast(FORECAST_PATH)

# Sidebar
st.sidebar.header("Controls")
state = st.sidebar.selectbox("Select a state", sorted(df["state"].unique()))

# Selected state row
row = df[df["state"] == state].iloc[0]

col1, col2, col3, col4 = st.columns(4)

icu_val = f"{row['icu_pct_next_week_pred']:.1f}%" if "icu_pct_next_week_pred" in row else "N/A"
inp_val = f"{row['inpatient_pct_next_week_pred']:.1f}%" if "inpatient_pct_next_week_pred" in row else "N/A"

risk_proba = row.get("critical_risk_proba", None)
risk_label = "HIGH" if row.get("critical_risk_next_week_pred", 0) == 1 else "LOW"
risk_val = f"{risk_label} ({risk_proba:.2f})" if risk_proba is not None else risk_label

disease_val = row.get("disease_burden_next_week_pred", None)
disease_val = f"{disease_val:.0f}" if disease_val is not None else "N/A"

with col1:
    kpi_card("ICU forecast (next week)", icu_val)
with col2:
    kpi_card("Inpatient forecast (next week)", inp_val)
with col3:
    kpi_card("Critical stress risk", risk_val)
with col4:
    kpi_card("Disease burden (next week)", disease_val)

st.divider()

# Weeks info
w1, w2 = st.columns(2)
with w1:
    st.write("**Current week in dataset**:", row.get("current_week", "N/A"))
with w2:
    st.write("**Forecast week**:", row.get("forecast_week", "N/A"))

# Recommendation block
st.subheader("Recommendation")
st.info(row.get("recommendation", "No recommendation text found."))

# Neighbor suggestion
st.subheader("Neighbor suggestion")
suggested = row.get("suggested_neighbor_state", None)
if suggested:
    st.write(f"Suggested neighbor state: **{suggested}**")

    if suggested in df["state"].values:
        nrow = df[df["state"] == suggested].iloc[0]
        st.write(
            f"- Neighbor ICU forecast: **{nrow['icu_pct_next_week_pred']:.1f}%**\n"
            f"- Neighbor inpatient forecast: **{nrow['inpatient_pct_next_week_pred']:.1f}%**\n"
            f"- Neighbor critical risk: **{int(nrow.get('critical_risk_next_week_pred', 0))}** "
            f"(proba: {nrow.get('critical_risk_proba', 0):.2f})\n"
            f"- Neighbor disease burden: **{nrow.get('disease_burden_next_week_pred', 0):.0f}**"
        )
else:
    st.write("No neighbor suggestion available.")

st.divider()

# Top risk states table
st.subheader("Top 15 states by critical risk probability")
if "critical_risk_proba" in df.columns:
    top = df.sort_values("critical_risk_proba", ascending=False).head(15)
    st.dataframe(
        top[[
            "state",
            "icu_pct_next_week_pred",
            "inpatient_pct_next_week_pred",
            "critical_risk_proba",
            "critical_risk_next_week_pred",
            "disease_burden_next_week_pred",
            "suggested_neighbor_state",
        ]],
        use_container_width=True
    )
else:
    st.write("critical_risk_proba column not found in forecast file.")

st.caption("To update this dashboard, re-run: python src/predict_next_week.py")
