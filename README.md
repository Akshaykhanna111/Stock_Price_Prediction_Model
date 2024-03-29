# LSTM Models for Stock Price Prediction

This Jupyter notebook (Model_Training_Code.ipynb) provides a comprehensive guide to creating and training Long Short-Term Memory (LSTM) models for stock price prediction. It outlines the process step-by-step, from importing libraries to saving the trained models and predictions. Here's an overview of the notebook:

Overview
Imports: The notebook starts by importing necessary libraries such as Pandas, NumPy, Matplotlib, scikit-learn, TensorFlow, and yfinance.

Data Fetching: A function fetch_stock_data is defined to fetch historical stock data from Yahoo Finance using the yfinance library.

Data Preprocessing: Another function preprocess_data preprocesses and normalizes the fetched data using the MinMaxScaler from scikit-learn.

Sequence Creation: The create_sequences function creates sequences for LSTM training by splitting the data into input sequences and corresponding output labels.

Model Building: The build_train_lstm_model function builds and trains an LSTM model using TensorFlow's Keras API.

Future Predictions: The make_future_predictions function makes future predictions using the trained model.

Model Evaluation: The evaluate_model function evaluates the model's performance on the test set and visualizes the results using Matplotlib.

Main Process: The main process is orchestrated by the process_stock function, which fetches data, preprocesses it, trains the model, evaluates its performance, makes future predictions, and creates a DataFrame to store the results.

Top Stocks List: A list of top stocks in NYSE is defined.

Model Training: The notebook iterates over each stock in the list, trains an LSTM model, and stores the results in a dictionary.

Output Saving: The final DataFrame with predictions is saved to a CSV file in the 'LSTM Model Predictions' folder. Additionally, each trained model is saved in the 'LSTM Models' folder.

Results Display: Finally, the notebook displays or saves the results, including mean absolute error on the test set and the DataFrame with predictions for each stock.

Usage
Create Virtual Environment: Set up a virtual environment for your project.

Install Dependencies: Use pip to install the required libraries by running the provided pip install commands.

Execute the Notebook: Run the notebook Model_Training_Code.ipynb in Jupyter or any compatible environment.

Review Results: The notebook will train LSTM models for stock price prediction for the specified stocks and provide the mean absolute error on the test set and the DataFrame with predictions for each stock.

Dependencies
pandas
numpy
matplotlib
scikit-learn
tensorflow
yfinance
Folder Structure
LSTM Model Predictions: Contains CSV files with predictions for each stock.
LSTM Models: Contains saved LSTM models for each stock.


# Stock Prediction Web Application

This web application allows users to visualize stock price movements, view fundamental data, and get sentiment analysis of news articles for selected stocks. It also provides predictions using LSTM and Prophet models.

Installation
To run the application, follow these steps:

Create a virtual environment:
python -m venv venv

Activate the virtual environment:
venv\Scripts\activate

Update pip:
python -m pip install --upgrade pip

Install the required packages:
python -m pip install streamlit yfinance plotly pandas prophet numpy scikit-learn tensorflow alpha-vantage

Usage
To run the application, execute the following command in your terminal:

streamlit run stock_prediction_app.py

The application will open in your default web browser.

Features
Stock Price Visualization
Select a stock from the dropdown menu to view its price movements over the last 10 years.
See the annual return and standard deviation for the selected stock.
Candlestick Chart
View a candlestick chart for the stock's price movements over the last year.
Choose between daily, weekly, or monthly candlesticks.
Stock Price Forecast
View yearly, quarterly, or monthly forecasts for the selected stock using the Prophet package.
Use the sliders to select the number of years, quarters, or months for the forecast.
LSTM Model Forecast
Get a 7-day forecast for the selected stock using a pre-trained LSTM model.
View the predicted closing prices along with the actual prices for the last 60 days.
Stock News and Fundamental Data
View sentiment analysis of news articles for the selected stock.
See fundamental data including balance sheet, income statement, and cash flow statement from Alpha Vantage API.

Note
This application uses cached data to speed up loading times. However, it may take some time to load data for the first time.


# Stock Prediction API using Flask

This API allows users to make stock price predictions for a given stock ticker and duration using a pre-trained LSTM model. It also provides information on the stock's annual return, standard deviation, and news sentiment.

Installation
To run the API, follow these steps:

Upgrade Pip:
python -m pip install --upgrade pip

Install the required packages:
python -m pip install flask yfinance pandas numpy requests tensorflow scikit-learn alpha-vantage

Usage
To run the API, execute the following command in your terminal:
python flask_api.py

The API will start running on http://localhost:5000/.

Endpoints
/predict - POST

Input: JSON object with ticker (stock ticker symbol) and duration_months (number of months for prediction)

Output: JSON object with the following keys:
ticker: Stock ticker symbol
duration_months: Number of months for prediction
annual_return: Annual return of the stock
standard_deviation: Standard deviation of the stock
risk_adjusted_return: Risk-adjusted return of the stock
future_predictions: Dictionary containing future date as key and predicted close price as value
average_sentiment_score: Average sentiment score of news articles
sentiment_label_counts: Dictionary containing sentiment labels and their counts

Example
Request
{
    "ticker": "AAPL",
    "duration_months": 3
}

Response
{
    "ticker": "AAPL",
    "duration_months": 3,
    "annual_return": "25.84%",
    "standard_deviation": "27.15%",
    "risk_adjusted_return": "0.95%",
    "future_predictions": {
        "2024-05-14 00:00:00": 181.2034,
        "2024-05-15 00:00:00": 180.8855,
        "2024-05-16 00:00:00": 180.5556,
        ...
    },
    "average_sentiment_score": 0.15,
    "sentiment_label_counts": {
        "positive": 5,
        "neutral": 3,
        "negative": 2
    }
}

Note
This API uses a cached LSTM model and news sentiment data to speed up processing. However, it may take some time to load data for the first time.
