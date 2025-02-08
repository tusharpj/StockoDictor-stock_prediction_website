import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.metrics import mean_squared_error

def fetch_stock_data(symbol, start_year, end_year):
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    data = yf.download(symbol, start=start, end=end)

    # Check available columns
    print(f"Available columns for {symbol}: {data.columns.tolist()}")

    # Handle missing 'Adj Close' column
    if 'Adj Close' not in data.columns:
        print("'Adj Close' column not found, using 'Close' instead.")
        data['Adj Close'] = data['Close']  # Use 'Close' column if 'Adj Close' is missing
    
    # Select relevant columns
    data = data[['High', 'Low', 'Adj Close']]  # Use relevant columns for prediction
    data.columns = ['High', 'Low', 'Close']  # Rename columns for convenience
    return data

### 2. Preprocessing Data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)  # Normalize the data
    return scaled_data, scaler

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])  # Create sequences
        Y.append(dataset[i + time_step, :])  # The target is the next day's data
    return np.array(X), np.array(Y)

### 3. Building the LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
        LSTM(50),
        Dense(3)  # Predicting high, low, and close prices
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

### 4. Train the Model
def train_model(model, X, Y, epochs=50, batch_size=32):
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

### 5. Predicting and Plotting
def predict_and_plot(data, symbol):
    scaled_data, scaler = preprocess_data(data)
    X, Y = create_dataset(scaled_data)

    # Split the data into training and testing sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Build and train the model
    model = build_model(X_train.shape)
    train_model(model, X_train, Y_train, epochs=50)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(Y_test[:, 2], predictions[:, 2])  # Using Close price for evaluation
    print(f"Mean Squared Error: {mse}")

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(Y_test):], Y_test[:, 2], color='blue', label='Actual Close Price')  # Actual values
    plt.plot(data.index[-len(Y_test):], predictions[:, 2], color='red', label='Predicted Close Price')  # Predicted values
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (in ₹)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the range of predicted prices
    predicted_high = predictions[-1][0]
    predicted_low = predictions[-1][1]
    predicted_close = predictions[-1][2]

    return f'{predicted_low:.2f}-{predicted_high:.2f}', mse

# Function to get user input for stock symbol
def get_stock_symbol():
    print("Enter the stock symbol of the company (example: TATAMOTORS.NS for Tata Motors):")
    symbol = input().strip()
    return symbol

# Main code to run the model
if __name__ == "__main__":
    # Get stock symbol from user
    symbol = get_stock_symbol()
    
    # Ensure valid symbol
    if symbol:
        start_year = 2010  # Starting year for data
        end_year = 2023    # Ending year for data

        # Fetch and process the stock data
        data = fetch_stock_data(symbol, start_year, end_year)

        # Predict and plot the results
        price_range, mse = predict_and_plot(data, symbol)
        print(f"Predicted price range for {symbol}: {price_range}")
        print(f"Mean Squared Error: {mse}")
    else:
        print("Invalid stock symbol. Please try again.")
