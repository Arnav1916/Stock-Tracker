# Required Libraries
import streamlit as st
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import plotly.graph_objects as go
from prophet import Prophet
from dotenv import load_dotenv
import os

# Alpha Vantage API key
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Function to fetch real-time stock data using Alpha Vantage
@st.cache_data
def get_stock_data(symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
    return data

# Function to fetch historical stock data using Yahoo Finance
@st.cache_data
def get_historical_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    return data

# Function for forecasting using Prophet
def forecast_stock_prices(data):
    # Ensure the 'ds' column has no timezone information
    data.reset_index(inplace=True)
    data['ds'] = pd.to_datetime(data['Date']).dt.tz_localize(None)  # Remove timezone
    data['y'] = data['Close']
    
    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model
    model.fit(data[['ds', 'y']])
    
    # Create a dataframe for future predictions
    future = model.make_future_dataframe(periods=365)
    
    # Generate the forecast
    forecast = model.predict(future)
    
    return forecast

# Streamlit App
st.title('Stock Price Tracker and Analysis')

# Stock Symbol Input
stock_symbol = st.text_input('Enter Stock Symbol', 'AAPL')

# Alert Price Input
alert_price = st.number_input('Enter the target price for an alert', min_value=0.0, value=150.0)

# Fetch real-time stock data
st.write(f'Real-time data for {stock_symbol}')
stock_data = get_stock_data(stock_symbol)
st.write(stock_data.head())

# Plotting real-time stock data
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['4. close'], mode='lines', name='Close Price'))
st.plotly_chart(fig)

# Fetch historical data
st.write(f'Historical data for {stock_symbol}')
historical_data = get_historical_data(stock_symbol)
st.write(historical_data.tail())

# Forecasting stock prices
st.write(f'Forecasting future prices for {stock_symbol}')
forecast = forecast_stock_prices(historical_data)
st.write(forecast.tail())

# Plotting forecast
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='red', dash='dash')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='red', dash='dash')))
st.plotly_chart(fig_forecast)

# Compare the latest close price with the alert price
latest_close_price = stock_data['4. close'].iloc[-1]
if latest_close_price >= alert_price:
    st.warning(f'Alert! The stock price for {stock_symbol} has reached or exceeded ${alert_price}. Current price: ${latest_close_price:.2f}')
else:
    st.info(f'The current stock price is ${latest_close_price:.2f}. No alert triggered yet.')
