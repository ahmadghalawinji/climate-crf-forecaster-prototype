import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

PROC   = Path("data/processed")
MODELS = Path("models")

FEATURES = [
    "avg_temp",
    "heat_stress",
    "avg_precip_mm",
    "pa_policy_score",
    "gdp_per_capita",
    "urban_pop_pct",
]

@st.cache_resource
def load_assets():
    df    = pd.read_csv(PROC / "dataset_with_crf.csv").dropna(subset=FEATURES)
    model = joblib.load(MODELS / "random_forest.pkl")
    return df, model

st.set_page_config(page_title="Climate & Youth CRF", layout="wide")
st.title("🌡 Climate Change & Youth Cardiorespiratory Fitness")
st.caption("Prototype research dashboard — SUPER Project (Horizon Europe)")

df, model = load_assets()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Scenario controls")

temp_increase = st.sidebar.slider(
    "Temperature increase (°C)", min_value=0.0, max_value=4.0, value=2.0, step=0.5
)
pa_boost = st.sidebar.slider(
    "Physical activity boost (%)", min_value=0, max_value=50, value=0, step=5
)
selected_model = st.sidebar.selectbox(
    "ML model", ["random_forest", "xgboost", "linear_regression"]
)

@st.cache_resource
def get_model(name):
    return joblib.load(MODELS / f"{name}.pkl")

active_model = get_model(selected_model)

# ── Compute scenario predictions ─────────────────────────────────────────────
def predict_scenario(df, model, delta_t, pa_delta_pct):
    X = df[FEATURES].copy()
    X["avg_temp"]        = X["avg_temp"]    + delta_t
    X["heat_stress"]     = (X["heat_stress"] + delta_t * 0.05).clip(0, 1)
    X["pa_policy_score"] = (X["pa_policy_score"] * (1 + pa_delta_pct / 100))
    return active_model.predict(X)
    
baseline_preds = predict_scenario(df, active_model, 0, 0)
scenario_preds = predict_scenario(df, active_model, temp_increase, pa_boost)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline mean VO₂max", f"{baseline_preds.mean():.1f} ml/kg/min")
col2.metric("Scenario mean VO₂max", f"{scenario_preds.mean():.1f} ml/kg/min",
            delta=f"{scenario_preds.mean() - baseline_preds.mean():.2f}")
col3.metric("Records in dataset", f"{len(df):,}")
col4.metric("Countries", df["country"].nunique())

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("VO₂max distribution — baseline vs scenario")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=baseline_preds, name="Baseline",
                               nbinsx=30, opacity=0.7, marker_color="#1d9e75"))
    fig.add_trace(go.Histogram(x=scenario_preds, name="Scenario",
                               nbinsx=30, opacity=0.7, marker_color="#d85a30"))
    fig.update_layout(barmode="overlay", height=350,
                      xaxis_title="VO₂max (ml/kg/min)", yaxis_title="Count",
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Mean VO₂max across warming scenarios")
    deltas = [0, 1, 2, 3, 4]
    means  = [predict_scenario(df, active_model, d, pa_boost).mean() for d in deltas]
    fig2 = go.Figure(go.Scatter(
        x=[f"+{d}°C" for d in deltas], y=means, mode="lines+markers",
        line=dict(color="#1d9e75", width=2.5),
        marker=dict(size=8, color="#0f6e56")
    ))
    fig2.update_layout(height=350, xaxis_title="Temperature increase",
                       yaxis_title="Mean VO₂max (ml/kg/min)",
                       yaxis_range=[min(means) - 1, max(means) + 1])
    st.plotly_chart(fig2, use_container_width=True)

# ── Mitigation comparison ─────────────────────────────────────────────────────
st.subheader("Physical activity mitigation — temperature vs PA boost")
pa_levels = list(range(0, 55, 5))
mitigation_data = []
for temp_d in [1, 2, 3]:
    for pa_d in pa_levels:
        pred = predict_scenario(df, active_model, temp_d, pa_d).mean()
        mitigation_data.append({"temp": f"+{temp_d}°C", "pa_boost": pa_d, "vo2max": pred})

mit_df = pd.DataFrame(mitigation_data)
fig3 = px.line(mit_df, x="pa_boost", y="vo2max", color="temp",
               labels={"pa_boost": "PA increase (%)", "vo2max": "Mean VO₂max",
                       "temp": "Warming"},
               color_discrete_sequence=["#ef9f27", "#d85a30", "#993c1d"])
fig3.update_layout(height=380)
st.plotly_chart(fig3, use_container_width=True)

# ── Raw data explorer ─────────────────────────────────────────────────────────
with st.expander("Explore dataset"):
    st.dataframe(df.head(100), use_container_width=True)