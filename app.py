import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os

# --- Define Paths ---
BACKEND_DIR = r"D:\webstreamlit"
MODEL_PATH = os.path.join(BACKEND_DIR, "lstm_model.h5") # Update to adamax_model.h5 if you export it!
SCALER_PATH = os.path.join(BACKEND_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BACKEND_DIR, "recent_48_months.csv")
GRAPH_PATH = os.path.join(BACKEND_DIR, "optimizer_loss.png")

# --- Page Config ---
st.set_page_config(page_title="SPEI-3 Drought Predictor", page_icon="🌍", layout="wide")

# --- Header Section ---
st.title("🌍 South India Drought Prediction Dashboard")
st.markdown("**Predicting the Standardized Precipitation Evapotranspiration Index (SPEI) at a 3-month time scale using an LSTM neural network.**")

with st.expander("🔬 View Project Methodology & Architecture"):
    st.markdown("""
    * **Data Processing:** Utilized NetCDF climate datasets (1901–2024) isolated to the South India bounding box. The target variable is SPEI at a 3-month time scale (SPEI-3), smoothed via a moving average.
    * **Model Architecture:** Long Short-Term Memory (LSTM) Neural Network optimized for sequential time-series forecasting.
    * **Training pipeline:** The data was rigorously divided into a 70% training, 20% testing, and 10% validation split to prevent data leakage.
    * **Inference:** The model consumes a 48-month lookback window to forecast future drought conditions autonomously.
    """)
st.markdown("---")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    model, scaler = load_assets()
    df_recent = load_data()
except Exception as e:
    st.error(f"Error loading assets from {BACKEND_DIR}. Details: {e}")
    st.stop()

# --- Main Layout ---
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("Historical Data (Last 48 Months)")
    
    # --- UPGRADED: Interactive Plotly Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_recent['Date'], 
        y=df_recent['SPEI_SMOOTH'],
        mode='lines+markers',
        name='SPEI-3 (Smoothed)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Neutral", annotation_position="bottom right")
    fig.add_hline(y=-1.0, line_dash="dot", line_color="orange", annotation_text="Moderate Drought", annotation_position="bottom right")
    
    fig.update_layout(
        title="Recent SPEI-3 Trends in South India",
        xaxis_title="Date",
        yaxis_title="SPEI Value",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("LSTM 3-Month Forecast")
    st.write("Using recursive sequence generation to project the upcoming quarter.")

    if st.button("Run Prediction Engine", type="primary", use_container_width=True):
        with st.spinner('Calculating future steps...'):
            current_sequence = df_recent['SPEI_SMOOTH'].values.reshape(-1, 1)
            current_scaled_seq = scaler.transform(current_sequence)
            predictions_real = []
            
            for i in range(3):
                lstm_input = current_scaled_seq.reshape(1, 48, 1)
                pred_scaled = model.predict(lstm_input, verbose=0)
                pred_real = scaler.inverse_transform(pred_scaled)[0][0]
                predictions_real.append(pred_real)
                current_scaled_seq = np.append(current_scaled_seq[1:], pred_scaled).reshape(48, 1)
            
            last_date = df_recent['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='MS')
            
            forecast_df = pd.DataFrame({
                "Date": future_dates.strftime("%B %Y"),
                "Predicted SPEI-3": [f"{val:.4f}" for val in predictions_real],
                "Condition": [
                    "Severe Drought 🔴" if val <= -1.5 else
                    "Moderate Drought 🟠" if val <= -1.0 else
                    "Wet Conditions 🔵" if val >= 1.0 else
                    "Near Normal 🟢"
                    for val in predictions_real
                ]
            })
            st.table(forecast_df)

# --- Model Evaluation Section ---
st.markdown("---")
st.subheader("Model Evaluation & Algorithm Selection")
st.write("The performance of the LSTM architecture was rigorously evaluated across two optimization algorithms:")

col_eval1, col_eval2 = st.columns([1, 1.2])

with col_eval1:
    st.markdown("#### Evaluation Metrics")
    metrics_data = {
        "Metric": ["Paper R² (%)", "sklearn R² (%)", "RMSE", "MAE", "Accuracy (%)"],
        "RMSprop": ["98.35", "98.25", "0.0898", "0.0716", "97.50"],
        "Adamax": ["98.56", "98.50", "0.0831", "0.0657", "97.71"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    # Highlight the winning Adamax column in Streamlit 
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

with col_eval2:
    st.markdown("#### Validation Loss (Training Phase)")
    try:
        st.image(GRAPH_PATH, caption="RMSprop vs Adamax Validation Loss", use_column_width=True)
    except FileNotFoundError:
        st.info("Export 'optimizer_loss.png' from Jupyter to the backend folder to display the training curve.")