import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.optimizers import RMSprop, Adam, SGD 
import csv


def retrieve_historical_data(api_key, api_secret, symbol, interval, limit):
    client = Client(api_key, api_secret)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    df = df.dropna()

    return df


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    closing_price_scaler = MinMaxScaler(feature_range=(0, 1))
    closing_price_scaler.fit_transform(df['close'].values.reshape(-1, 1))
    X, y = create_dataset(df_scaled)
    return X, y, scaler, closing_price_scaler


def create_dataset(data):
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)




def train_lstm_model(X_train, y_train, csv_filename):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    loss_history = []  # Lista para armazenar a perda de cada época
    
    for epoch in range(5000):
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        loss = history.history['loss'][0]
        print(f"Epoch {epoch+1}/5000 - Loss: {loss}")
        loss_history.append(loss)
    # Salvar a perda em um arquivo CSV
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])
        for epoch, loss in enumerate(loss_history):
            writer.writerow([epoch+1, loss])
    
    return model



def predict_prices(model, X_test, closing_price_scaler):
    predicted_price_scaled = model.predict(X_test)
    predicted_price = closing_price_scaler.inverse_transform(predicted_price_scaled)

    return predicted_price


def evaluate_predictions(actual_direction, predicted_direction):
    precision = precision_score(actual_direction, predicted_direction, average='weighted', zero_division=1)
    recall = recall_score(actual_direction, predicted_direction, average='weighted', zero_division=1)
    f1 = f1_score(actual_direction, predicted_direction, average='weighted', zero_division=1)

    return precision, recall, f1


def get_current_value(api_key, api_secret, symbol):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, limit=1380)
    last_minute_kline = klines[-2]
    df_temp = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                            'taker_buy_quote_asset_volume', 'ignore'])
    df_temp['close'] = df_temp['close'].astype(float)
    df_temp.set_index('timestamp', inplace=True)
    return df_temp
