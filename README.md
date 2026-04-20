# 🌍 South India Drought Prediction using Deep Learning (LSTM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)

## 📌 Project Overview
This project leverages a Long Short-Term Memory (LSTM) neural network to forecast meteorological droughts in South India. By analyzing over a century of climate data, the model predicts the Standardized Precipitation Evapotranspiration Index (SPEI) at a 3-month time scale (SPEI-3) to provide early warnings for severe drought or extreme wet conditions.

The project includes a complete machine learning pipeline: from NetCDF data extraction and preprocessing to model training, evaluation, and deployment via an interactive Streamlit web dashboard.

## 📊 Dataset & Preprocessing
* **Data Source:** Historical NetCDF climate datasets spanning from 1901 to 2024.
* **Spatial Domain:** Isolated to the South India bounding box (Lat 8°N to 20°N, Lon 74°E to 80°E).
* **Target Variable:** SPEI-3 (3-month rolling average), smoothed to capture sustained meteorological trends.
* **Data Splitting:** To prevent data leakage and ensure robust validation, the time-series data was strictly partitioned into a **70% Training / 20% Testing / 10% Validation** split.

## 🧠 Model Architecture & Methodology
* **Network Type:** Sequential LSTM (Long Short-Term Memory) network, optimized for time-series forecasting.
* **Lookback Window:** The model utilizes an autoregressive lookback window of **48 months** (4 years) of historical SPEI-3 data to predict the subsequent month's value.
* **Optimization Comparison:** During the training phase, the architecture was evaluated using both `RMSprop` and `Adamax` optimizers to determine the most effective weight-updating strategy.

## 📈 Performance & Evaluation
The model demonstrated exceptional predictive accuracy, with the **Adamax optimizer** yielding the highest performance metrics on the unseen test dataset:

| Metric | Score (Adamax) |
| :--- | :--- |
| **Paper R²** | 98.56% |
| **sklearn R²** | 98.50% |
| **Accuracy** | 97.71% |
| **RMSE** | 0.0831 |
| **MAE** | 0.0657 |

## 🚀 Live Dashboard
The inference engine is deployed as an interactive web application. Users can view historical trends and run the prediction engine to generate a recursive 3-month future forecast.

**[Link to Live Streamlit Application]** https://south-india-drought-prediction-9817.streamlit.app/

## 💻 Tech Stack
* **Language:** Python
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Data Manipulation:** Pandas, NumPy, netCDF4
* **Visualization & UI:** Streamlit, Plotly, Matplotlib

## 🛠️ Local Installation & Usage
To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/South-India-Drought-Prediction.git](https://github.com/yourusername/South-India-Drought-Prediction.git)
   cd South-India-Drought-Prediction
