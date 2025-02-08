# from flask import Flask, render_template, request, jsonify, send_file
# import matplotlib.pyplot as plt
# import io
# import numpy as np

# app = Flask(__name__)

# # Dummy ML model function
# def predict_stock():
#     # Historical data ke basis pe dummy graph generate karte hain
#     x = np.linspace(1, 10, 100)
#     y = x**2  # Dummy prediction
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, y, label='Predicted Range')
#     plt.title('Stock Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Value')
#     plt.legend()
#     plt.tight_layout()

#     # Graph ko memory buffer mein save karte hain
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close()
#     return buffer

# # API for ML model output
# @app.route('/predict', methods=['GET'])
# def predict():
#     buffer = predict_stock()
#     return send_file(buffer, mimetype='image/png')

# # Route for Dashboard
# @app.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html')  # Dashboard page ko render karega

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['GET'])
# def register():
#     return render_template('register.html')

# @app.route('/login', methods=['GET'])
# def login():
#     return render_template('login.html')

# @app.route('/forgot-password', methods=['GET'])
# def forget_password():
#     return render_template('forgot-password.html')

# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('home.html')

# @app.route('/about', methods=['GET'])
# def about():
#     return render_template('about.html')

# if __name__ == '__main__':
#     app.run(debug=True)

#----------------------------------------------------------------------------------------------------------------
# ### 1. Fetch Stock Data
# def fetch_stock_data(symbol, start_year, end_year):
#     start = f"{start_year}-01-01"
#     end = f"{end_year}-12-31"
#     data = yf.download(symbol, start=start, end=end)

#     # Check if data is empty
#     if data.empty:
#         raise ValueError("No data found for the given symbol and date range.")

#     data = data[['High', 'Low', 'Adj Close']]
#     data.columns = ['High', 'Low', 'Close']
#     return data
#----------------------------------------------------------------------------------------------------------------



















#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#main code
# from flask import Flask, render_template, request, jsonify, send_file
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
# import io

# app = Flask(__name__)

# def fetch_stock_data(symbol, start_year, end_year):
#     start = f"{start_year}-01-01"
#     end = f"{end_year}-12-31"
#     data = yf.download(symbol, start=start, end=end)

#     if data.empty:
#         raise ValueError("No data found for the given symbol and date range.")

#     # Use 'Close' if 'Adj Close' is not available
#     if 'Adj Close' in data.columns:
#         data['Close'] = data['Adj Close']
#     elif 'Close' in data.columns:
#         data['Close'] = data['Close']
#     else:
#         raise ValueError("Neither 'Adj Close' nor 'Close' column found in data.")

#     data = data[['High', 'Low', 'Close']]
#     return data



# ### 2. Preprocess Data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler

# def create_dataset(dataset, time_step=60):
#     X, Y = [], []
#     for i in range(len(dataset) - time_step - 1):
#         X.append(dataset[i:(i + time_step), :])
#         Y.append(dataset[i + time_step, :])  # Predicting high, low, and close
#     return np.array(X), np.array(Y)


# ### 3. Build and Train Model
# def build_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
#         LSTM(50),
#         Dense(3)  # Output layer predicts high, low, and close prices
#     ])
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# def train_model(model, X, Y, epochs=50, batch_size=32):
#     model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)


# ### 4. Predict and Plot
# def predict_and_plot(data, symbol):
#     scaled_data, scaler = preprocess_data(data)
#     X, Y = create_dataset(scaled_data)

#     # Build and train the model
#     model = build_model(X.shape)
#     train_model(model, X, Y, epochs=10)

#     # Prepare the last 60 days of data for prediction
#     last_60_days = data[-60:]
#     last_60_days_scaled = scaler.transform(last_60_days)
#     X_test = last_60_days_scaled.reshape(1, -1, 3)

#     # Predict
#     predicted_prices = model.predict(X_test)
#     predicted_prices = scaler.inverse_transform(predicted_prices)

#     predicted_high = predicted_prices[0][0]
#     predicted_low = predicted_prices[0][1]
#     predicted_close = predicted_prices[0][2]

#     # Generate graph
#     plt.figure(figsize=(6, 4))
#     plt.plot(data.index[-60:], data['Close'][-60:], label='Actual Close Price')
#     plt.axhline(predicted_close, color='red', linestyle='--', label='Predicted Close Price')
#     plt.title(f'Stock Prediction for {symbol}')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.tight_layout()

#     # Save graph to buffer
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close()

#     # Return predicted range and graph buffer
#     predicted_range = f'{predicted_low:.2f}-{predicted_high:.2f}'
#     return predicted_range, buffer


# ### 5. Flask Routes
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get stock symbol and date range from request
#     try:
#         data = request.json
#         symbol = data.get('symbol', 'AAPL')
#         start_year = data.get('start_year', 2020)
#         end_year = data.get('end_year', 2023)

#         # Fetch stock data and make predictions
#         stock_data = fetch_stock_data(symbol, start_year, end_year)
#         predicted_range, buffer = predict_and_plot(stock_data, symbol)

#         # Send the graph and prediction range as a response
#         return jsonify({
#             'status': 'success',
#             'predicted_range': predicted_range
#         })
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 400


# @app.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html')


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/register', methods=['GET'])
# def register():
#     return render_template('register.html')


# @app.route('/login', methods=['GET'])
# def login():
#     return render_template('login.html')


# @app.route('/forgot-password', methods=['GET'])
# def forget_password():
#     return render_template('forgot-password.html')


# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('home.html')


# @app.route('/about', methods=['GET'])
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     app.run(debug=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------














#streamlit code
# ------------------------------------------------------------------------------------------------------
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score

# st.title("Indian Stock Price Predictor")

# # Search box
# search_term = st.text_input("Search for a company (NSE Ticker Symbol):", "").upper()

# if search_term:
#     try:
#         # Fetch ticker symbol dynamically
#         stock = yf.Ticker(search_term + ".NS")
#         data = stock.history(period="5y")  # Fetching 5 years of data
#         if data.empty:
#             st.error(f"No data found for {search_term}. Try another company.")
#             st.stop()
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         st.stop()
    
#     st.success(f"Data loaded successfully for {search_term}")
    
#     # Feature engineering
#     df = data[['Close', 'Volume']].copy()
#     df['SMA_10'] = df['Close'].rolling(window=10).mean()
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()
#     df.dropna(inplace=True)
    
#     # Prepare data for training
#     X = df[['Close', 'Volume', 'SMA_10', 'SMA_50']].values
#     y = df['Close'].shift(-1).dropna().values  # Predicting next day's close price
#     X = X[:-1]  # Aligning feature set with target set
    
#     # Data Preprocessing
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Splitting data
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
#     # Train model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Model Evaluation
#     y_pred = model.predict(X_test)
#     accuracy = r2_score(y_test, y_pred) * 100
    
#     # Prepare for prediction
#     last_row = df[['Close', 'Volume', 'SMA_10', 'SMA_50']].iloc[-1:].values
#     last_row_scaled = scaler.transform(last_row)
#     next_day_pred = model.predict(last_row_scaled)[0]
    
#     lower_bound = next_day_pred * 0.98
#     upper_bound = next_day_pred * 1.02
    
#     # Display results
#     st.subheader(f"Prediction for {search_term}")
#     st.write(f"Next trading day predicted closing price range: ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
#     st.write(f"Model Accuracy: {accuracy:.2f}%")
    
#     # Show historical chart
#     st.subheader("Historical Closing Prices")
#     st.line_chart(df['Close'])
# else:
#     st.info("Enter a company name to get the stock prediction.")
#---------------------------------------------------------------------------------------------------------------------



























# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend
# import matplotlib.pyplot as plt
# from flask import Flask, render_template, request, jsonify, send_file
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
# from tensorflow.keras.layers import Input
# import io

# app = Flask(__name__)

# def fetch_stock_data(symbol, start_year, end_year):
#     start = f"{start_year}-01-01"
#     end = f"{end_year}-12-31"
#     data = yf.download(symbol, start=start, end=end)

#     if data.empty:
#         raise ValueError("No data found for the given symbol and date range.")

#     # Use 'Close' if 'Adj Close' is not available
#     if 'Adj Close' in data.columns:
#         data['Close'] = data['Adj Close']
#     elif 'Close' in data.columns:
#         data['Close'] = data['Close']
#     else:
#         raise ValueError("Neither 'Adj Close' nor 'Close' column found in data.")

#     data = data[['High', 'Low', 'Close']]
#     return data



# ### 2. Preprocess Data
# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler

# def create_dataset(dataset, time_step=60):
#     X, Y = [], []
#     for i in range(len(dataset) - time_step - 1):
#         X.append(dataset[i:(i + time_step), :])
#         Y.append(dataset[i + time_step, :])  # Predicting high, low, and close
#     return np.array(X), np.array(Y)


# ### 3. Build and Train Model
# def build_model(input_shape):
#     model = Sequential([
#         Input(shape=(input_shape[1], input_shape[2])),  # Use Input layer here
#         LSTM(50, return_sequences=True),
#         LSTM(50),
#         Dense(3)
#     ])
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# def train_model(model, X, Y, epochs=50, batch_size=32):
#     model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)


# ### 4. Predict and Plot
# def predict_and_plot(data, symbol):
#     scaled_data, scaler = preprocess_data(data)
#     X, Y = create_dataset(scaled_data)

#     # Build and train the model
#     model = build_model(X.shape)
#     train_model(model, X, Y, epochs=10)

#     # Prepare the last 60 days of data for prediction
#     last_60_days = data[-60:]
#     last_60_days_scaled = scaler.transform(last_60_days)
#     X_test = last_60_days_scaled.reshape(1, -1, 3)

#     # Predict
#     predicted_prices = model.predict(X_test)
#     predicted_prices = scaler.inverse_transform(predicted_prices)

#     predicted_high = predicted_prices[0][0]
#     predicted_low = predicted_prices[0][1]
#     predicted_close = predicted_prices[0][2]

#     # Generate graph
#     plt.figure(figsize=(6, 4))
#     plt.plot(data.index[-60:], data['Close'][-60:], label='Actual Close Price')
#     plt.axhline(predicted_close, color='red', linestyle='--', label='Predicted Close Price')
#     plt.title(f'Stock Prediction for {symbol}')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.tight_layout()

    
#     # Save graph to buffer
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close()


#     # Return predicted range and graph buffer
#     predicted_range = f'{predicted_low:.2f}-{predicted_high:.2f}'
#     return predicted_range, buffer


# ### 5. Flask Routes
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get stock symbol and date range from request
#     try:
#         data = request.json
#         symbol = data.get('symbol', 'AAPL')
#         start_year = data.get('start_year', 2020)
#         end_year = data.get('end_year', 2023)

#         # Fetch stock data and make predictions
#         stock_data = fetch_stock_data(symbol, start_year, end_year)
#         predicted_range, buffer = predict_and_plot(stock_data, symbol)

#         # Send the graph and prediction range as a response
#         return jsonify({
#             'status': 'success',
#             'predicted_range': predicted_range
#         })
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 400


# @app.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html')


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/register', methods=['GET'])
# def register():
#     return render_template('register.html')


# @app.route('/login', methods=['GET'])
# def login():
#     return render_template('login.html')


# @app.route('/forgot-password', methods=['GET'])
# def forget_password():
#     return render_template('forgot-password.html')


# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('home.html')


# @app.route('/about', methods=['GET'])
# def about():
#     return render_template('about.html')


# if __name__ == '__main__':
#     app.run(debug=True)














































#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#streamlit-code-main
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from flask import Flask, request, jsonify, render_template
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# def fetch_stock_data(ticker):
#     stock = yf.Ticker(ticker + ".NS")
#     data = stock.history(period="5y")
#     if data.empty:
#         raise ValueError("No data found for the given ticker symbol.")
#     return data

# def preprocess_data(data):
#     df = data[['Close', 'Volume']].copy()
#     df['SMA_10'] = df['Close'].rolling(window=10).mean()
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()
#     df.dropna(inplace=True)
    
#     X = df[['Close', 'Volume', 'SMA_10', 'SMA_50']].values
#     y = df['Close'].shift(-1).dropna().values  # Next day's close price
#     X = X[:-1]  # Align X with y
    
#     return df, X, y

# def train_model(X, y):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     accuracy = r2_score(y_test, y_pred) * 100
    
#     return model, scaler, accuracy

# def save_model(model, scaler, filename="stock_price_model.pkl"):
#     with open(filename, 'wb') as file:
#         pickle.dump((model, scaler), file)

# def load_model(filename="stock_price_model.pkl"):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

# def predict_next_day(model, scaler, last_row):
#     last_row_scaled = scaler.transform(last_row)
#     next_day_pred = model.predict(last_row_scaled)[0]
#     lower_bound = next_day_pred * 0.98
#     upper_bound = next_day_pred * 1.02
#     return lower_bound, upper_bound

# def plot_stock_data(df, ticker):
#     plt.figure(figsize=(10, 5))
#     plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price (₹)')
#     plt.title(f'Historical Closing Prices for {ticker}')
#     plt.legend()
#     plt.grid()
#     plt.show()

# app = Flask(__name__, template_folder="templates")

# # @app.route("/")
# # def dashboard():
# #     return render_template("dashboard.html")


# @app.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html')


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/register', methods=['GET'])
# def register():
#     return render_template('register.html')


# @app.route('/login', methods=['GET'])
# def login():
#     return render_template('login.html')


# @app.route('/forgot-password', methods=['GET'])
# def forget_password():
#     return render_template('forgot-password.html')


# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('home.html')


# @app.route('/about', methods=['GET'])
# def about():
#     return render_template('about.html')


# @app.route("/predict", methods=["GET"])
# def predict():
#     ticker = request.args.get("ticker", "").upper()
#     if not ticker:
#         return jsonify({"error": "Please provide a ticker symbol."}), 400
    
#     try:
#         data = fetch_stock_data(ticker)
#         df, X, y = preprocess_data(data)
#         model, scaler, accuracy = train_model(X, y)
#         save_model(model, scaler)
        
#         last_row = X[-1:].reshape(1, -1)
#         lower, upper = predict_next_day(model, scaler, last_row)
        
#         return jsonify({
#             "ticker": ticker,
#             "accuracy": f"{accuracy:.2f}%",
#             "prediction_range": f"₹{lower:.2f} - ₹{upper:.2f}"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------












import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Fetch stock data function
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker + ".NS")
    data = stock.history(period="5y")
    if data.empty:
        raise ValueError("No data found for the given ticker symbol.")
    return data

# Data preprocessing function
def preprocess_data(data):
    df = data[['Close', 'Volume']].copy()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)
    
    X = df[['Close', 'Volume', 'SMA_10', 'SMA_50']].values
    y = df['Close'].shift(-1).dropna().values  # Next day's close price
    X = X[:-1]  # Align X with y
    
    return df, X, y

# Model training function
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred) * 100
    
    return model, scaler, accuracy

# Prediction for the next day function
def predict_next_day(model, scaler, last_row):
    last_row_scaled = scaler.transform(last_row)
    next_day_pred = model.predict(last_row_scaled)[0]
    lower_bound = next_day_pred * 0.98
    upper_bound = next_day_pred * 1.02
    return lower_bound, upper_bound

# Plot stock data function and return it as base64
def plot_stock_data(df, ticker):
    plt.figure(figsize=(7, 5))
    plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (₹)')
    plt.title(f'Historical Closing Prices for {ticker}')
    plt.legend()
    plt.grid()

    # Save plot to a BytesIO object and encode it to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

app = Flask(__name__)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET'])
def forget_password():
    return render_template('forgot-password.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol."}), 400
    
    try:
        data = fetch_stock_data(ticker)
        df, X, y = preprocess_data(data)
        model, scaler, accuracy = train_model(X, y)
        
        last_row = X[-1:].reshape(1, -1)
        lower, upper = predict_next_day(model, scaler, last_row)
        
        # Generate the stock price plot
        plot_url = plot_stock_data(df, ticker)
        
        return jsonify({
            "ticker": ticker,
            "accuracy": f"{accuracy:.2f}%",
            "prediction_range": f"₹{lower:.2f} - ₹{upper:.2f}",
            "plot_url": plot_url  # Send plot URL as base64 image
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)








































































































































# import yfinance as yf
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import io
# import base64
# from flask import Flask, request, jsonify, render_template, send_file
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

# app = Flask(__name__, template_folder="templates")

# def fetch_stock_data(ticker):
#     stock = yf.Ticker(ticker + ".NS")
#     data = stock.history(period="5y")
#     if data.empty:
#         raise ValueError("No data found for the given ticker symbol.")
#     return data

# def preprocess_data(data):
#     df = data[['Close', 'Volume']].copy()
#     df['SMA_10'] = df['Close'].rolling(window=10).mean()
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()
#     df.dropna(inplace=True)
    
#     X = df[['Close', 'Volume', 'SMA_10', 'SMA_50']].values
#     y = df['Close'].shift(-1).dropna().values
#     X = X[:-1]  # Remove the last row as it has no label
    
#     return df, X, y

# def train_model(X, y):
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Feature scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Initialize the model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train_scaled, y_train)
    
#     # Predict and calculate accuracy
#     y_pred = model.predict(X_test_scaled)
#     accuracy = r2_score(y_test, y_pred) * 100  # R^2 accuracy in percentage
    
#     return model, scaler, accuracy

# def save_model(model, scaler):
#     # Save the model and scaler to files for later use
#     with open("model.pkl", "wb") as f:
#         pickle.dump(model, f)
#     with open("scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

# def predict_next_day(model, scaler, X):
#     # Scale the input features for prediction
#     X_scaled = scaler.transform(X)
    
#     # Predict the next day's stock price
#     prediction = model.predict(X_scaled)
    
#     # Get a range for the prediction
#     predicted_price = prediction[0]
#     lower = predicted_price * 0.95  # 5% decrease
#     upper = predicted_price * 1.05  # 5% increase
    
#     return lower, upper

# @app.route("/predict", methods=["GET"])
# def predict():
#     ticker = request.args.get("ticker", "").upper()
#     if not ticker:
#         return jsonify({"error": "Please provide a ticker symbol."}), 400
    
#     try:
#         data = fetch_stock_data(ticker)
#         df, X, y = preprocess_data(data)
#         model, scaler, accuracy = train_model(X, y)
#         save_model(model, scaler)
        
#         last_row = X[-1:].reshape(1, -1)
#         lower, upper = predict_next_day(model, scaler, last_row)
        
#         return jsonify({
#             "ticker": ticker,
#             "accuracy": f"{accuracy:.2f}%",
#             "prediction_range": f"₹{lower:.2f} - ₹{upper:.2f}"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/plot")
# def plot_stock():
#     ticker = request.args.get("ticker", "").upper()
#     if not ticker:
#         return jsonify({"error": "Please provide a ticker symbol."}), 400
    
#     try:
#         data = fetch_stock_data(ticker)
#         plt.figure(figsize=(10, 5))
#         plt.plot(data.index, data["Close"], label="Closing Price", color="blue")
#         plt.xlabel("Date")
#         plt.ylabel("Stock Price (₹)")
#         plt.title(f"Historical Closing Prices for {ticker}")
#         plt.legend()
#         plt.grid()
        
#         img = io.BytesIO()
#         plt.savefig(img, format="png")
#         img.seek(0)
#         plt.close()
        
#         return send_file(img, mimetype="image/png")

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/dashboard', methods=['GET'])
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/register', methods=['GET'])
# def register():
#     return render_template('register.html')

# @app.route('/login', methods=['GET'])
# def login():
#     return render_template('login.html')

# @app.route('/forgot-password', methods=['GET'])
# def forget_password():
#     return render_template('forgot-password.html')

# @app.route('/home', methods=['GET'])
# def home():
#     return render_template('home.html')

# @app.route('/about', methods=['GET'])
# def about():
#     return render_template('about.html')

# if __name__ == "__main__":
#     app.run(debug=True)
