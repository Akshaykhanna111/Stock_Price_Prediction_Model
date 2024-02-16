# Create your virtual env 
# Post activating the venv execute the following commands - 
# python -m pip install --upgrade pip
# python -m pip install streamlit yfinance plotly pandas prophet numpy scikit-learn tensorflow alpha-vantage 


import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from alpha_vantage.fundamentaldata import FundamentalData
import os
import requests
import tensorflow as tf
from prophet import Prophet
from collections import Counter

np.random.seed(1234)
tf.random.set_seed(1234)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # To ensure consistency of results on Streamlit and Flask

st.set_page_config(layout = 'wide')

# Title of the web app
st.title("Stock Price Prediction Web Application")

# List of available stocks
stocks = ('AAPL', 'MSFT', 'AMZN', 'TSLA', 'BRK-A', 'JNJ', 'GOOGL')

# Dropdown to select the stock for prediction
selected_stocks = st.selectbox('Select the Stock for Prediction', stocks)


# Calculate start date based on the training period
start_date = "2012-01-01"

end_date = datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data2 = data.copy()
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)
    annual_return = data2['% Change'].mean() * 252 * 100
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    annual_return = round(annual_return, 2)
    stdev = round(stdev*100, 2)
    data2 = data2.sort_values(by='Date', ascending=False)
    return data,  data2, annual_return, stdev

st.header("Price Movements - Last 10 Years")
data, pricing_data, annual_return, stdev = load_data(selected_stocks, start_date, end_date)

# Calculate the date 10 years ago from the end_date
ten_years_ago = datetime.now() - timedelta(days=365*10)

# Filter the pricing_data DataFrame to include only data from the last 10 years
pricing_data_last_10_years = pricing_data[pricing_data['Date'] >= ten_years_ago]

# Display the filtered DataFrame using st.write()
st.write(pricing_data_last_10_years, width=1000, height=600)
st.write('Annual Return is ', annual_return,'%')
st.write('Standard Deviation is ', stdev,'%')
#st.write('Risk Adjusted Return is ', round(annual_return/(stdev*100), 2),'%')

# Function to plot the raw data
def plot_raw_data():
    st.header("Opening and Closing Price Trends - Last 10 Years")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close', line=dict(color='red')))
    fig.layout.update(xaxis_rangeslider_visible=True, width=1000, height=600)
    st.plotly_chart(fig)

# Display the raw data plot
plot_raw_data()



# Candlestick chart 

st.header("Candlestick Chart - Last 1 year")


def plot_candlestick_chart(df, candlestick_duration):
    # Resample the data based on the selected candlestick duration
    if candlestick_duration == 'Daily':
        candlestick_data = df
    elif candlestick_duration == 'Weekly':
        candlestick_data = df.resample('W', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna().reset_index()
    elif candlestick_duration == 'Monthly':
        candlestick_data = df.resample('M', on='Date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna().reset_index()

    # Calculate moving averages
    candlestick_data['MA20'] = candlestick_data['Close'].rolling(window=20).mean()
    candlestick_data['SMA20'] = candlestick_data['Close'].rolling(window=20).mean()
    candlestick_data['EMA20'] = candlestick_data['Close'].ewm(span=20, adjust=False).mean()

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=candlestick_data['Date'],
                                         open=candlestick_data['Open'],
                                         high=candlestick_data['High'],
                                         low=candlestick_data['Low'],
                                         close=candlestick_data['Close'],
                                         name='Candlestick'),
                          go.Scatter(x=candlestick_data['Date'], y=candlestick_data['MA20'], mode='lines', name='MA20', line=dict(color='blue', dash='dot', width=1)),
                          go.Scatter(x=candlestick_data['Date'], y=candlestick_data['SMA20'], mode='lines', name='SMA20', line=dict(color='red', dash='dot', width=2)),
                          go.Scatter(x=candlestick_data['Date'], y=candlestick_data['EMA20'], mode='lines', name='EMA20', line=dict(color='green'))])

    # Customize layout
    fig.update_layout(#title=f'Candlestick Chart with Moving Averages - {candlestick_duration} Candlesticks',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=True,
                      width=1000, height=600)  # Add x-axis range slider

    # Display the chart
    st.plotly_chart(fig)


# Slider widget for selecting the duration of each candlestick
candlestick_duration = st.selectbox('Select Candlestick Duration', ['Daily', 'Weekly', 'Monthly'], index = 1)

# Display the candlestick chart with moving averages
plot_candlestick_chart(data.tail(365), candlestick_duration)



# Define function to load and cache the Prophet forecast data for 4 years
@st.cache_data
def load_prophet_forecast_4_years(data):
    # Prepare data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    # Instantiate and fit Prophet model
    m = Prophet()
    m.fit(df_train)
    
    # Create future dataframe for prediction
    future = m.make_future_dataframe(periods=4*365)  # Forecast for 4 years
    
    # Make forecast
    forecast = m.predict(future)
    
    return df_train, forecast

# Define function to plot forecast
def plot_forecast(df_train, forecast):
    # Plot the forecast using Plotly
    fig = go.Figure()
    
    # Plot the actual data
    fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual'))
    
    # Plot the forecasted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    
    # Add layout
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Closing Price')
    fig.layout.update(xaxis_rangeslider_visible=True, 
                      width=1000, height=600)
    return fig

# Load and cache forecast data for 4 years
df_train_years, forecast_years = load_prophet_forecast_4_years(data)

# Yearly Prediction

# Header
st.header('Yearly Forecast using FBprophet Package')
st.write('')

# Slider for selecting years of prediction
n_years = st.slider('Select the Years of Prediction:', 1, 4, 1)


last_index = len(df_train_years) - 1
forecast_subset = forecast_years.iloc[: last_index + n_years * 365]

# Now you can pass df_train and forecast_subset to the plot_forecast function
fig_years = plot_forecast(df_train_years, forecast_subset)
st.plotly_chart(fig_years)



# Quarterly Prediction

# Header
st.header('Quarterly Forecast using FBprophet Package')
st.write('')

# Slider for selecting quarters of prediction
n_quarters = st.slider('Select the Quarters of Prediction:', 1, 4, 3)

last_index = len(df_train_years) - 1
forecast_subset = forecast_years.iloc[last_index - (n_quarters * 91 * 6): last_index + n_quarters * 91]
train_subset = df_train_years[last_index - (n_quarters * 91 * 6):last_index]

# Now you can pass df_train and forecast_subset to the plot_forecast function
fig_quarters = plot_forecast(train_subset, forecast_subset)
st.plotly_chart(fig_quarters)



# Monthly Prediction

# Header
st.header('Monthly Forecast using FBprophet Package')
st.write('')

# Slider for selecting months of prediction
n_months = st.slider('Select the Months of Prediction:', 1, 12, 8)

last_index = len(df_train_years) - 1
forecast_subset = forecast_years.iloc[last_index - (n_months * 31 * 6): last_index + n_months * 31]
train_subset = df_train_years[last_index - (n_months * 31 * 6):last_index]

# Now you can pass df_train and forecast_subset to the plot_forecast function
fig_months = plot_forecast(train_subset, forecast_subset)
st.plotly_chart(fig_months)


# Display short term future predictions for the selected period
st.header('7 Days Forecast using LSTM Model')

# Slider to choose the number of days for prediction
period = 7

# Function to load and cache the LSTM model
@st.cache_data
def load_lstm_model():
    model_path = f'../../LSTM Models/{selected_stocks}_model.tf'
    model = load_model(model_path)
    return model

# Load the trained LSTM model
model = load_lstm_model()

# Function to preprocess and normalize the data
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to create sequences for LSTM training
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 0])
    return np.array(X)

# Preprocess and normalize the data for prediction
scaled_data, scaler = preprocess_data(data[['Close']].values)
X_pred = create_sequences(scaled_data, seq_length=10)

# Make predictions for the future period and cache the results
@st.cache_data
def predict_future(period, X_pred, _model, _scaler):
    future_predictions_scaled = []
    last_sequence = X_pred[-1]  # Get the last sequence from the historical data

    for _ in range(period):
        prediction = _model.predict(np.expand_dims(last_sequence, axis=0))
        future_predictions_scaled.append(prediction)
        last_sequence = np.append(last_sequence[1:], prediction)

    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    future_predictions = _scaler.inverse_transform(future_predictions_scaled)
    return future_predictions

# Predict future prices and cache the results
cached_future_predictions = predict_future(period, X_pred, model, scaler)

# Create date range for future predictions
last_date = data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=period, freq='D')

future_df = pd.DataFrame({'Date': future_dates[:period], 
                          'Predicted Close Price': cached_future_predictions[:period].flatten()})

# Generate future dates for forecasted period
forecast_end_date = data['Date'].iloc[-1] + timedelta(days=len(future_dates))
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), end=forecast_end_date)

def plot_raw_data_with_forecast(actual_data, forecast_data, forecast_dates):
    fig = go.Figure()

    # Plot actual closing prices for last 60 days
    actual_data = actual_data.tail(90)
    fig.add_trace(go.Scatter(x=actual_data['Date'], y=actual_data['Close'], name='Actual Close', line=dict(color='blue')))

    # Plot forecasted closing prices
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_data.flatten(), name='Forecasted Close', line=dict(color='red')))

    # Update layout
    fig.update_layout(
        xaxis_title='Date',  # Add x-axis label
        yaxis_title='Closing Price',  # Add y-axis label
        xaxis=dict(
            showgrid=False,  # Show grid lines
            tickangle=-45,  # Rotate x-axis tick labels for better visibility
            tickformat="%Y-%m-%d"  # Format the x-axis tick labels
        ),
        yaxis=dict(
            showgrid=False  
        ),
        width=1000, height=600
    )

    # Plot the graph
    st.plotly_chart(fig)



# Display the raw data plot with forecast
plot_raw_data_with_forecast(data, cached_future_predictions, forecast_dates)


# Function to fetch news for selected stocks
@st.cache_data
def fetch_news(stock, count_n):
    news_data = []
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Calculate the start date 30 days ago
    start_date = datetime.now() - timedelta(days=30)

    # Format the start date in the required format (YYYYMMDDTHHMM)
    start_date_str = start_date.strftime('%Y%m%dT%H%M')

    # Make the API call using the updated URL
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&time_from={start_date_str}&limit=1000&apikey={api_key}'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for item in data['feed']:
            ticker_sentiments = item.get('ticker_sentiment', [])
            if len(ticker_sentiments) == 1 and ticker_sentiments[0]['ticker'] == stock and float(ticker_sentiments[0]['relevance_score']) > 0.1:
                # Extract relevant information
                news_item = {
                    'time_published': item['time_published'],
                    'title': item['title'],
                    'summary': item['summary'],
                    'url': item['url'],
                    'source_domain': item.get('source_domain', 'Unknown'),  # Capture source domain
                    'relevance_score': float(ticker_sentiments[0]['relevance_score']),
                    'ticker_sentiment_score': float(ticker_sentiments[0]['ticker_sentiment_score']),
                    'ticker_sentiment_label': ticker_sentiments[0]['ticker_sentiment_label']
                }
                news_data.append(news_item)
    
        # Sort news items by time_published in descending order
        news_data.sort(key=lambda x: (x['time_published'], x['relevance_score']), reverse=True)
        return news_data[:count_n]
    else:
        st.error(f"Failed to fetch news for {stock}. Error: {response.text}")

# Cache function for fundamental data tab
@st.cache_data
def load_fundamental_data(stock):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    fd = FundamentalData(api_key, output_format = 'pandas')
    balance_sheet = fd.get_balance_sheet_annual(stock)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    income_statement = fd.get_income_statement_annual(stock)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    cash_flow = fd.get_cash_flow_annual(stock)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])

    return bs, is1, cf

# Cache function for pricing data tab
# @st.cache_data
# def load_pricing_data(data):
#     data2 = data.copy()
#     data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
#     data2.dropna(inplace=True)
#     annual_return = data2['% Change'].mean() * 252 * 100
#     stdev = np.std(data2['% Change']) * np.sqrt(252)
#     annual_return = round(annual_return, 2)
#     stdev = round(stdev*100, 2)
#     return data2, annual_return, stdev

# Define the tabs
tabs = ["Stock News - Last 30 days", "Fundamental Data"]
selected_tab = st.selectbox("Stock News and Fundamental Data", tabs)

# Load content based on the selected tab
if selected_tab == "Stock News - Last 30 days":
    st.write('')
    tabs_count = ["Top 10 News", "Top 20 News", "Top 30 News"]
    selected_tab = st.selectbox("Select Count of Articles", tabs_count)
    #st.subheader(f"{selected_tab} for {selected_stocks}")
    if selected_tab == "Top 10 News":
        count_n = 10
    elif selected_tab == "Top 20 News":
        count_n = 20
    else:
        count_n = 30
    news_data = fetch_news(selected_stocks, count_n)

    # Assuming news_data is a list of dictionaries containing news information
    label_counts = Counter(item['ticker_sentiment_label'] for item in news_data)

    # Display the counts using st.metrics columns
    st.markdown('### News Sentiment')
    
    # Determine the number of columns based on the number of labels
    num_labels = len(label_counts)
    if num_labels > 0:
        columns = st.columns(num_labels)

        # Create metrics for each label in label_counts
        for i, (label, count) in enumerate(label_counts.items()):
            columns[i].metric(label, count)

        # Display each news item in an expander
        for idx, news_item in enumerate(news_data):
            with st.expander(f"News {idx + 1}", expanded=True):  # Set expanded=True
                st.write(f"**Title:** {news_item['title']}")
                published_datetime = news_item['time_published']
                published_date, published_time = published_datetime.split('T')  # Split at 'T'
                published_date_obj = datetime.strptime(published_date, '%Y%m%d').strftime('%Y-%m-%d')
                published_time_obj = datetime.strptime(published_time, '%H%M%S').strftime('%I:%M %p')
                st.write(f"**Published Date:** {published_date_obj}")
                st.write(f"**Published Time:** {published_time_obj}")
                # Display source domain
                if news_item['source_domain'] == 'www.fool.com':
                    source_text = "The Motley Fool Holdings Inc."
                else:
                    source_text = news_item['source_domain']
                st.write(f"**Source Domain:** {source_text}")
                st.write(f"**Summary:** {news_item['summary']}")
                st.write(f"**Relevance Score:** {news_item['relevance_score']}")
                st.write(f"**Sentiment Score:** {news_item['ticker_sentiment_score']}")
                st.write(f"**Sentiment Label:** {news_item['ticker_sentiment_label']}")
                st.write(f"[Read More]({news_item['url']})")
    else:
        st.write('No Articles Found in Last 30 days')

elif selected_tab == "Fundamental Data":
    st.header("Fundamental Data from Alpha Vantage API")
    balance_sheet, income_statement, cash_flow = load_fundamental_data(selected_stocks)
    st.subheader('Balance Sheet')
    st.write(balance_sheet)
    st.subheader('Income Statement')
    st.write(income_statement)
    st.subheader('Cash Flow Statement')
    st.write(cash_flow)
