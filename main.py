import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import *
import h5py


api_key = 'YAyh2qolCAKFryDqOQTbZEiMS3tlGZyRoTy6HvyWZwrm49p4PWYxdyBglDF2MHce'
api_secret = 'JswOacoTqkHFFbrB6560ALbHrtIq4mxbzXMEHCNM1i4nkaxokyE6jINYnX94BcrS'
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
limit = 10000

# Retrieve historical price data from Binance
df = retrieve_historical_data(api_key, api_secret, symbol, interval, limit)

# Preprocess the data
X, y, scaler, closing_price_scaler = preprocess_data(df)

# Split the data into training and testing sets
X_train, y_train = X[:int(0.8*len(df))], y[:int(0.8*len(df))]
X_test, y_test = X[int(0.8*len(df)):], y[int(0.8*len(df)):]

# Train or load the LSTM model
train_model = True

if train_model:
    model = train_lstm_model(X_train, y_train,'lstm_5000_32_17082023.csv')
    model.save('models\lstm_5000_32_17082023.h5')
else:
    model_path ='models\lstm_5000_32_17082023.h5'
    # Carregar o arquivo .h5 com o h5py
    with h5py.File(model_path, 'r') as f:
        # Obter o modelo do arquivo
        model = load_model(f)

# Make predictions
predicted_price = predict_prices(model, X_test, closing_price_scaler)

# Display test charts
plt.plot(df.index[-140:], df['close'].values[-140:], color='blue', label='Actual Price')
plt.plot(df.index[-140:], predicted_price, color='red', label='Predicted Price')
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Reshape predictions and calculate directional signals
predicted_price_reshaped = predicted_price.reshape(-1)
predicted_direction = np.sign(np.diff(predicted_price_reshaped))
actual_direction = np.sign(np.diff(y_test))

# Evaluate predictions
precision, recall, f1 = evaluate_predictions(actual_direction, predicted_direction)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Get current value
current_value_df = get_current_value(api_key, api_secret, symbol)
print(current_value_df)
