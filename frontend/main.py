import streamlit as st
import requests
import pandas as pd
import json

st.title("📈 Stock Price Prediction App")
ticker = st.text_input("Give the stock's Ticker Symbol").upper()
if st.button('Predict'):
    if ticker:
        with st.spinner('Training the model...'):
            train_response = requests.post(f"http://127.0.0.1:8000/get-ticker/{ticker}")
        if train_response.status_code == 200:
            st.success(f'Model trained successfully for ticker: {ticker}')

            with st.spinner('Fetching forecast...'):
                prediction_response = requests.get("http://127.0.0.1:8000/get-ticker/prediction")
            if prediction_response.status_code == 200:
                forecast_data = prediction_response.json()['forecast']
                forecast_df = pd.DataFrame(forecast_data)    

                st.subheader('Forecast for next 7 days: ')
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                st.dataframe(forecast_df.set_index('Date'))
            else:
                st.error(f"Prediction failed: {prediction_response.text}")
        else:
            st.error(f"Training failed: {train_response.text}")
    else:
        st.warning('Please enter a valid stock ticker.')