# Upgrade the Pip version 
# Pip install following packages - flask yfinance pandas numpy requests tensorflow scikit-learn alpha-vantage

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import os
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # To ensure consistency of results on Streamlit and Flask


app = Flask(__name__)

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

# Function to make predictions for the future period
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

# Function to fetch news sentiment for selected stocks
def fetch_news_sentiment(stock):
    news_data = []
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    start_date = datetime.now() - timedelta(days=30)
    start_date_str = start_date.strftime('%Y%m%dT%H%M')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&time_from={start_date_str}&limit=1000&apikey={api_key}'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for item in data['feed']:
            ticker_sentiments = item.get('ticker_sentiment', [])
            if len(ticker_sentiments) == 1 and ticker_sentiments[0]['ticker'] == stock and float(ticker_sentiments[0]['relevance_score']) > 0.1:
                news_item = {
                    'ticker_sentiment_score': float(ticker_sentiments[0]['ticker_sentiment_score']),
                    'ticker_sentiment_label': ticker_sentiments[0]['ticker_sentiment_label']
                }
                news_data.append(news_item)

        # Calculate average sentiment score
        if news_data:
            sentiment_scores = [item['ticker_sentiment_score'] for item in news_data]
            avg_sentiment_score = np.mean(sentiment_scores)
            sentiment_labels = [item['ticker_sentiment_label'] for item in news_data]
            sentiment_label_counts = dict(Counter(sentiment_labels))
            return avg_sentiment_score, sentiment_label_counts
        else:
            return None, None
    else:
        return None, None

@app.route('/predict', methods=['POST'])
def predict_stock():
    # Parse request data
    data = request.json
    ticker = data.get('ticker')
    duration_months = data.get('duration_months')

    # Validate input parameters
    if not ticker or ticker not in ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'BRK-A', 'JNJ', 'GOOGL']:
        return jsonify({'error': 'Invalid or missing ticker'}), 400
    if not duration_months or not isinstance(duration_months, int) or duration_months < 1 or duration_months > 4:
        return jsonify({'error': 'Invalid duration_months (must be an integer between 1 and 4)'}), 400

    # Calculate start and end dates
    start_date = "2012-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Load data
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data2 = data.copy()
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace=True)

    # Calculate annual return and standard deviation
    annual_return = data2['% Change'].mean() * 252 * 100
    annual_return = round(annual_return, 2)
    stdev = np.std(data2['% Change']) * np.sqrt(252)
    stdev = round(stdev*100, 2)

    # Load the LSTM model
    model_path = f'../../LSTM Models/{ticker}_model.tf'
    model = load_model(model_path)

    # Preprocess data
    scaled_data, scaler = preprocess_data(data[['Close']].values)
    X_pred = create_sequences(scaled_data, seq_length=10)

    # Make predictions for the future period
    future_predictions = predict_future(duration_months * 30, X_pred, model, scaler)

    # Assuming future_dates is a list of datetime objects representing future dates
    future_dates = [datetime.today() + timedelta(days=i) for i in range(duration_months * 30)]

    # Convert future_dates list into a DataFrame
    dates_df = pd.DataFrame({'Date': future_dates})

    # Assuming future_predictions is a NumPy array containing predicted close prices
    future_predictions_df = pd.DataFrame({'Predicted_Close_Price': future_predictions.flatten()})

    # Concatenate the two DataFrames along the columns axis
    result_df = pd.concat([dates_df, future_predictions_df], axis=1)
    future_predictions_dict = {}
    for index, row in result_df.iterrows():
        future_predictions_dict[str(row['Date'])] = row['Predicted_Close_Price']

    # Fetch news sentiment
    avg_sentiment_score, sentiment_label_counts = fetch_news_sentiment(ticker) 


    # Prepare response
    response = {
        'ticker': ticker,
        'duration_months': duration_months,
        'annual_return': str(annual_return) + '%',
        'standard_deviation': str(stdev) + '%',
        'risk_adjusted_return': str(round(annual_return / (stdev * 100), 2)) + '%',
        'future_predictions': future_predictions_dict,
        'average_sentiment_score': avg_sentiment_score,
        'sentiment_label_counts': sentiment_label_counts
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)