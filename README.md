# 📈 Stock Market Price Prediction using Google TS Mixer

This project presents a deep learning approach to forecast short-term stock prices for **Ashok Leyland** using **Google's TS Mixer**, a cutting-edge transformer-based model designed for time series forecasting. The implementation is carried out using the `darts` library, blending traditional technical analysis with modern machine learning techniques.

---

## 🔧 Project Overview

The goal of this project is to accurately forecast stock price movements over a **short-term period of 7 days (1 week)**. The model was specifically trained to detect and predict short-term trends in stock price behavior using the powerful temporal modeling capabilities of Google’s **TS Mixer**.

---

## 📊 Workflow

1. **Data Collection**  
   - Retrieved historical stock data for **Ashok Leyland** using the `yfinance` API.

2. **Feature Engineering**  
   - Enriched the dataset with a variety of **technical indicators** (e.g., Moving Averages, RSI, MACD) to improve input quality for the model.

3. **Exploratory Data Analysis (EDA)**  
   - Conducted thorough analysis to understand patterns, seasonality, volatility, and feature relationships within the data.

4. **Data Preprocessing**  
   - Formatted the dataset to match the input requirements of the `darts` time series framework.
   - Normalized values and structured sequences to support time-windowed predictions.

5. **Model Training**  
   - Trained the **Google TS Mixer** model on time-windowed data to forecast the **next 7 days** of stock prices.
   - Tuned key hyperparameters such as input/output chunk lengths, number of layers, model dimensions, and learning rate.

6. **Model Evaluation**  
   - Achieved a **MAPE of 1.67%** on the most recent 7-day forecast.
   - Cross-validation yielded an average **MAPE of 3.54%**, indicating consistent model performance across different data splits.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **darts** (Time Series Forecasting Library)
- **Google TS Mixer** (Transformer-based deep learning model)
- **yfinance** (Financial data extraction)

---

## ✅ Key Highlights

- Designed and trained a model specifically for **7-day stock price forecasting** using transformer-based architecture.
- Achieved low error rates, outperforming many traditional methods in short-term prediction tasks.
- Seamlessly integrated technical indicators to boost the predictive strength of the model.

---

## 📁 Project Structure

```bash
.
├── data/                   # Raw and processed stock data
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── models/                 # Saved models (if applicable)
├── utils/                  # Utility scripts for preprocessing and indicators
├── README.md               # Project documentation
