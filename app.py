"""
Streamlit dashboard for KPI visualization and live model inference.
Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime

@st.cache_data
def load_data(processed_path="data/processed/machines_processed_hourly.csv"):
    if not os.path.exists(processed_path):
        st.error("Processed data not found. Run main pipeline first.")
        return None
    df = pd.read_csv(processed_path, parse_dates=['timestamp'])
    return df

@st.cache_resource
def load_model(model_path="models/rf_failure_predictor.joblib"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def sidebar_controls(machines):
    st.sidebar.title("Controls")
    machine = st.sidebar.selectbox("Select machine", machines)
    window = st.sidebar.slider("Window (hours)", min_value=24, max_value=720, value=168, step=24)
    show_preds = st.sidebar.checkbox("Show predictions", value=True)
    return machine, window, show_preds

def main():
    st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
    st.title("Predictive Maintenance - KPI Dashboard")
    df = load_data()
    if df is None:
        return

    model = load_model()

    machines = sorted(df['machine_id'].unique().tolist())
    machine, window_hours, show_preds = sidebar_controls(machines)

    now = df['timestamp'].max()
    window_start = now - pd.Timedelta(hours=window_hours)
    df_sel = df[(df['machine_id'] == machine) & (df['timestamp'] >= window_start)].copy()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    avg_temp = df_sel['temperature'].mean()
    avg_vib = df_sel['vibration'].mean()
    downtime = df_sel['failure'].sum()  # crude
    health_score = max(0, 100 - (avg_temp*0.2 + avg_vib*25))  # toy score
    col1.metric("Avg Temp", f"{avg_temp:.2f}")
    col2.metric("Avg Vibration", f"{avg_vib:.3f}")
    col3.metric("Downtime Events", int(downtime))
    col4.metric("Health Score", f"{health_score:.1f}")

    # Time series
    st.subheader(f"Time-series (last {window_hours} hrs) - {machine}")
    chart_df = df_sel.set_index('timestamp')[['temperature','vibration','pressure','humidity']].reset_index()
    st.line_chart(chart_df.rename(columns={'timestamp':'index'}).set_index('index'))

    # Predictions panel
    if model is None:
        st.warning("Model not found. Train model first to see predictions.")
    elif show_preds:
        st.subheader("Predicted failure probability (last records)")
        feat_cols = ['temperature','vibration','pressure','humidity','runtime_hours','temp_roll_mean_3','vib_roll_mean_3','temp_roll_std_3','time_idx']
        recent = df_sel.tail(100).fillna(0)
        X_recent = recent[feat_cols]
        probs = model.predict_proba(X_recent)[:,1]
        recent = recent.copy()
        recent['pred_proba'] = probs
        st.write(recent[['timestamp','machine_id','temperature','vibration','pred_proba']].tail(20))

        # Quick alert
        high_risk = recent[recent['pred_proba'] > 0.5]
        if not high_risk.empty:
            st.error(f"High-risk alerts: {len(high_risk)} records with pred_proba > 0.5")
        else:
            st.success("No high-risk predictions in the selected window.")

    # Export screenshot
    if st.button("Save current view as report (png)"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(df_sel['timestamp'], df_sel['temperature'], label='temperature')
        ax.plot(df_sel['timestamp'], df_sel['vibration'], label='vibration')
        ax.legend()
        plt.xticks(rotation=20)
        outp = os.path.join("presentation", "assets", f"{machine}_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        fig.savefig(outp, bbox_inches='tight')
        st.success(f"Saved timeseries image to {outp}")

if __name__ == "__main__":
    main()
