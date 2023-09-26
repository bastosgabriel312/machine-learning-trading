import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.optimizers import RMSprop, Adam, SGD
import csv
from datetime import datetime
import time
import pickle

# Função para calcular o ERRO MÉDIO ABSOLUTO
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true).astype(float), np.array(predictions).astype(float)
    return np.mean(np.abs(y_true - predictions))

# Função para calcular as métricas e imprimir
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-scores)
    mae_score = mae(y, y_pred)
    return r2, mae_score, rmse_scores.mean()

# Recupera os dados históricos da binance e retorna um dataframe
def retrieve_historical_data_binance(client, symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    return df


def get_klines_df(client, symbol, interval, total_periods=2000):
    klines_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    periods_received = 0
    last_timestamp = None

    while periods_received < total_periods:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=total_periods - periods_received,
            endTime=last_timestamp
        )

        if not klines:
            break

        for kline in klines:
            klines_df.loc[len(klines_df)] = kline

        periods_received += len(klines)

        last_timestamp = klines[0][0]

    # Convert timestamp column to datetime
    klines_df['timestamp_to_date'] = pd.to_datetime(klines_df['timestamp'], unit='ms')

    return klines_df.sort_values(by='timestamp')


# Calcula manualmente a métrica de mvrv_ratio
def retrieve_mvrv_ratio(df_binance):
    df_binance['mvrv_ratio'] = df_binance['close'].astype(float) / ((df_binance['close'].astype(float) + df_binance['high'].astype(float)+ df_binance['low'].astype(float) + df_binance['open'].astype(float)) / 4.0)
    return df_binance

# Retorna o datafrime da Binance com o mvrv calculado
def get_concatenate_klines_mvrv(client, symbol, interval, limit):
    df_binance_klines = get_klines_df(client, symbol, interval, limit)

    #from_date,to_date = timestamp_to_datestring(df_binance_klines['timestamp'].iloc[-0]), timestamp_to_datestring(df_binance_klines['timestamp'].iloc[-1])
    df_binance_klines['timestamp'] = df_binance_klines['timestamp'] // 1000
    df_binance_klines = retrieve_mvrv_ratio(df_binance_klines)
    return df_binance_klines

# Função para calcular os quartis em uma coluna de DataFrame
def calculate_quartiles(df, column_name):
    """
    Calcula os quartis (25%, 50%, 75%) em uma coluna de um DataFrame.

    :param df: DataFrame
    :param column_name: Nome da coluna
    :return: Um dicionário com os valores dos quartis
    """
    if column_name not in df.columns:
        raise ValueError(f"A coluna '{column_name}' não existe no DataFrame.")
    df[column_name] = df[column_name].astype('float64')
    values = df[column_name].values
    first_quartile = np.percentile(values, 25)
    second_quartile = np.percentile(values, 50)
    third_quartile = np.percentile(values, 75)

    return {
        'first_quartile': first_quartile,
        'second_quartile': second_quartile,
        'third_quartile': third_quartile
    }


# Função para adicionar colunas indicando quartis a um DataFrame
def add_quartile_columns(df, column_name):
    """
    Adiciona colunas ao DataFrame indicando se cada valor está no primeiro, segundo ou terceiro quartil.

    :param df: DataFrame
    :param column_name: Nome da coluna
    :return: O DataFrame com as novas colunas
    """
    quartiles = calculate_quartiles(df, column_name)
    first_quartile = quartiles['first_quartile']
    second_quartile = quartiles['second_quartile']
    third_quartile = quartiles['third_quartile']
    df[f'{column_name}_in_first_quartile'] = (df[column_name] <= first_quartile).astype(int)
    df[f'{column_name}_in_second_quartile'] = ((first_quartile < df[column_name]) & (df[column_name] <= second_quartile)).astype(int)
    df[f'{column_name}_in_third_quartile'] = ((second_quartile < df[column_name]) & (df[column_name] <= third_quartile)).astype(int)
    df.drop(column_name, axis=1)
    return df

# Função que transforma timestamp em data
def timestamp_to_datestring(timestamp):
    ts = timestamp/ 1000
    data = datetime.fromtimestamp(ts)
    return data.strftime('%Y-%m-%d')

# Função utilizada para normalizar uma coluna
def normalize_data(df, column_name):
  # Inicialize o MinMaxScaler
  scaler = MinMaxScaler()
  # Ajuste o scaler aos dados
  scaler.fit(df[[column_name]])

  # Aplique a transformação aos dados
  return scaler.transform(df[[column_name]])

# Função principal para pré processamento dos dados, utiliza as técnicas de quartile e normalização dos dados históricos
def preprocess_manual(historical_data):
  # adiciona a coluna que no dataset será o alvo
  historical_data = calcular_target_high(historical_data.copy())
  # Cria uma lista vazia para representar o df preprocessado
  df_preprocessed = historical_data.copy()

  # Separa os parametros que serão normalizados e divididos em quartis
  list_columns_quartil = ['volume','quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume','open']
  list_columns_normalize = ['open','high','low','close','mvrv_ratio']

  list_columns_normalize = ['volume','quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume','open','high','low','close','mvrv_ratio']

  for column in list_columns_normalize:
    df_preprocessed[f'normalized_{column}'] = normalize_data(historical_data, column)

  # Retira as colunas divididas em quartis
  list_drop = ['ignore','timestamp','timestamp_to_date', 'close_time'] +list_columns_normalize +list_columns_quartil
  df_preprocessed = df_preprocessed.drop(list_drop,axis=1)

  return df_preprocessed

def calcular_target_high(historical_data, num_periodos=4):
    max_highs = []

    for i in range(len(historical_data)):
        # calcula o valor máximo da coluna 'high' nos próximos períodos.
        max_high = historical_data['high'].iloc[i:i + num_periodos].max()
        max_highs.append(max_high)

    historical_data['target-high'] = max_highs
    historical_data_alvo = historical_data.copy()
    historical_data_alvo['target-high'] = historical_data_alvo['target-high'].astype('float64')
    return historical_data_alvo

