import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")
st.title('📈 Stock Price Predictor')

ticker = st.text_input(
    'Enter a valid NSE Ticker Symbol', 
    placeholder='e.g., RELIANCE.NS, TATASTEEL.NS'
)

def shift_weekends_to_weekdays_no_overlap(dates):
    shifted_dates = []
    existing_dates = set(dates)
    
    for d in dates:
        if d.weekday() == 5:  # Saturday
            new_date = d + pd.Timedelta(days=2)  # Monday
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        elif d.weekday() == 6:  # Sunday
            new_date = d + pd.Timedelta(days=2)  # Tuesday
            while new_date in existing_dates:
                new_date += pd.Timedelta(days=1)
            shifted_dates.append(new_date)
            existing_dates.add(new_date)
        else:
            shifted_dates.append(d)
    return pd.to_datetime(shifted_dates)

if st.button('Get 7-Day Forecast'):
    if not ticker:
        st.warning('Please enter a ticker symbol.')
    else:
        with st.spinner(f'Fetching forecast for {ticker}...'):
            try:
                api_url = f'https://stock-prediction-backend-st7k.onrender.com/predict/{ticker}'
                response = requests.get(api_url)

                if response.status_code==503:
                    st.warning('Model is still loading on the server. Please try again in few seconds.')

                elif response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"Forecast for {data.get('ticker')} received!")
                    
                    # Forecast
                    forecast_data = data.get('forecast')
                    forecast_df = pd.DataFrame(forecast_data)
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

                    # Shift weekend dates forward without overlap
                    forecast_df['Date'] = shift_weekends_to_weekdays_no_overlap(forecast_df['Date'])

                    # Aggregate duplicates by averaging numeric columns
                    forecast_df = forecast_df.groupby('Date').mean().reset_index()

                    # Add weekday name column
                    forecast_df['Weekday'] = forecast_df['Date'].dt.day_name()

                    # Prepare display DataFrame
                    display_df = forecast_df.set_index(forecast_df['Date'].dt.date)
                    display_df['Weekday'] = forecast_df['Weekday'].values

                    st.subheader('Predicted Prices')
                    st.dataframe(display_df[['Adj Close', 'Weekday']], use_container_width=True)

                    # Historical data processing
                    historical_data = data.get('historical')
                    historical_df = pd.DataFrame(historical_data)
                    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                    
                    historical_plot = historical_df.set_index('Date')['Adj Close'].rename('Historical Price')
                    forecast_plot = forecast_df.set_index('Date')['Adj Close'].rename('Forecasted Price')

                    chart_data = pd.concat([historical_plot, forecast_plot], axis=1)

                    st.subheader('Historical Data vs. Forecast')
                    st.line_chart(chart_data, color=["#007bff", "#ff7f0e"])

                    # Format Adj Close with currency symbol in display_df
                    if 'Adj Close' in display_df.columns:
                        display_df['Adj Close'] = display_df['Adj Close'].map('₹{:,.2f}'.format)

                else:
                    error_detail = response.json().get('detail', 'unknown error')
                    st.error(f"Error from API: {error_detail} (Status code: {response.status_code})")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the backend API. Is the server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")