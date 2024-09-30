import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

def save_predictions(current_prediction):
    if os.path.exists('predax.txt'):
        # Leggi la previsione precedente e salvala in lastdax.txt
        with open('predax.txt', 'r') as f:
            previous_prediction = f.read().strip()
        with open('lastdax.txt', 'w') as f:
            f.write(previous_prediction)
    # Salva la nuova previsione in predax.txt
    with open('predax.txt', 'w') as f:
        f.write(str(current_prediction))


# Function to create lagged features and technical indicators for the 'Close' price
def create_lagged_features_and_indicators(df, n_lags):
    # Create lagged features for Close prices
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)

    # Add technical indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)  # Relative Strength Index
    macd = ta.macd(df['Close'])  # MACD and MACD Signal
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MA50'] = ta.sma(df['Close'], length=50)  # 50-day moving average
    df['MA200'] = ta.sma(df['Close'], length=200)  # 200-day moving average

    df.dropna(inplace=True)  # Remove rows with NaN values caused by shifting or technical indicators
    return df


# Function to reload data and update the model with the latest Close price and technical indicators
def update_model_with_latest_close(model, file_path, n_lags_reduced):
    # Step 1: Reload the dataset (with new Close value added)
    df = pd.read_csv(file_path)

    # Step 2: Parse the Date column
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Step 3: Create new lagged features and technical indicators for the updated dataset
    df_with_indicators = create_lagged_features_and_indicators(df.copy(), n_lags_reduced)


    # Prepare the features (lagged and technical indicators) and target (Close prices)
    features = [f'lag_{i}' for i in range(1, n_lags_reduced + 1)] + ['RSI', 'MACD', 'MACD_signal', 'MA50', 'MA200']
    X = df_with_indicators[features].values  # 19 features: 14 lagged + 5 technical indicators
    y = df_with_indicators['Close'].values

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 4: Scale the features and target values using the same scalers
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))


    # Step 5: Continue training the existing model with new data
    model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=1)

    return model, df_with_indicators, scaler_X, scaler_y



# Function to build a 5-layer ANN model with the recommended Input layer
def build_ann_model(input_shape):
    model = Sequential()

    # Define the input layer
    model.add(Input(shape=input_shape))

    # First hidden layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    
    # Second hidden layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    # Third hidden layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Fourth hidden layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    # Fifth hidden layer with 32 units and ReLU activation
    model.add(Dense(32, activation='relu'))
    
    # Output layer with 1 unit for regression output (predicting Close price)
    model.add(Dense(1))

    # Compile the model with Adam optimizer and Mean Squared Error loss
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


# Define the number of lagged days (14 lags)
n_lags_reduced = 14

# (Optional) Load a pre-trained model, or build a new one if starting fresh
try:
    model = load_model("trained_model.keras")
    print("Model loaded successfully!")
except:
    print("No pre-trained model found. Building a new model...")
    # Build a new model if no pre-trained model is found
    # Define input shape for 14 lagged features + 5 technical indicators
    input_shape = (19,)
    # Build the model
    model = build_ann_model(input_shape)


# Path to the dataset
file_path = 'dax_historical_data.csv'

# Step 1: Call update_model_with_latest_close to get df_with_indicators and scalers
model, df_with_indicators, scaler_X, scaler_y = update_model_with_latest_close(model, file_path, n_lags_reduced)
# Save the model in the newer Keras format
model.save("trained_model_updated.keras")

# Step 2: Use df_with_indicators in prediction
if len(df_with_indicators) >= n_lags_reduced:
    # Get the most recent 14 days of Close prices from the dataset
    
    # Ensure that recent_data includes both 14 lagged features and 5 technical indicators
    recent_data = df_with_indicators.iloc[-1][[f'lag_{i}' for i in range(1, n_lags_reduced + 1)] + ['RSI', 'MACD', 'MACD_signal', 'MA50', 'MA200']].values.reshape(1, -1)



    # Scale this recent data using the same scaler as before
    recent_data_scaled = scaler_X.transform(recent_data)

    # Make a single prediction using the updated model
    single_prediction_scaled = model.predict(recent_data_scaled)

    # Denormalize the prediction back to the original scale
    single_prediction = scaler_y.inverse_transform(single_prediction_scaled)

    # Output the single predicted Close value
    save_predictions(single_prediction[0][0])
else:
    print(f"Not enough data to create lagged features. Minimum required: {n_lags_reduced} rows.")

