from utils import *
import warnings
import os
import pandas as pd
import numpy as np
import pickle
from binance.client import Client

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

STOP_LOSS = {'BTCUSDT':0.90,'ETCUSDT':0.90,'BNBUSDT':0.90}
STOP_GAIN = {'BTCUSDT':0.99,'ETCUSDT':0.99,'BNBUSDT':0.99}



# CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)
symbol = 'BNBUSDT'  # MOEDA
TEMPO_DO_MODELO = '6m'
DIAS_DE_BT = '20D'
interval = Client.KLINE_INTERVAL_15MINUTE  # INTERVALO


# Carregar o modelo salvo
with open(f'models/rl_{symbol}_{TEMPO_DO_MODELO}.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Recupere o modelo e o scaler
modelo = dados_carregados['modelo']
scalers = dados_carregados['scalers']

# Carregar dados históricos de preços (substitua por seus próprios dados)
dados_historicos = pd.read_csv(f'backtesting/{symbol}_01_10_{DIAS_DE_BT}.csv')

# Parâmetros da estratégia
saldo_inicial = 1000  # Saldo inicial da conta
saldo = saldo_inicial
ativo = False  # Para rastrear o ativo em carteira
preco_compra = 0  # Preço de compra do ativo
count_periodos = 0

# Lista para armazenar os resultados de cada período
resultados = []
high_real = 0
# Realizar backtesting
for indice, row in dados_historicos.iterrows():
    #representação do maior high entre 4 periodos
    high_real = row.high if row.high > high_real else high_real
    if count_periodos == 4:
        count_periodos = 0
        high_real = 0 
    if count_periodos < 1:
        entrada_modelo = row.drop(['Unnamed: 0', 'timestamp', 'timestamp_to_date', 'close_time', 'ignore'])
        # Normalizar cada coluna individualmente usando os scalers correspondentes
        entrada_modelo_preprocessada = {}
        for col in entrada_modelo.index:
            if col in scalers:
                scaler = scalers[col]
                valor = np.array(row[col]).reshape(1, -1)  # Transforma o valor em uma matriz 2D
                entrada_modelo_preprocessada[f'normalized_{col}'] = scaler.transform(valor)[0][0]
            else:
                entrada_modelo_preprocessada[col] = row[col]
        # Transforma o dicionário em um DataFrame 2D
        entrada_modelo_preprocessada_df = pd.DataFrame([entrada_modelo_preprocessada])

        if not ativo:   
            previsao = modelo.predict(entrada_modelo_preprocessada_df.values)

        # Verificar se o target-high foi atingido e realizar a compra ou venda
        if (not ativo) and (STOP_GAIN[symbol] * previsao) >= row.close:
            print(f">>>>>>>>> COMPRA")
            print(f"previsão: {previsao}")
            # Compra no início do período
            preco_compra = row.close
            lucro = 0
            ativo = True
            resultados.append({
                'Tipo de Operação': f'Compra ({count_periodos})',
                'Preço de Compra': row.close,
                'High do periodo': row.high,
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
            })
        elif ativo and row.high >= (STOP_GAIN[symbol] * previsao):
            ativo = False
            lucro = (STOP_GAIN[symbol] * previsao) - preco_compra
            resultados.append({
                'Tipo de Operação': f'Stop Gain ({count_periodos})',
                'Preço de Compra': preco_compra,
                'High do periodo': row.high,
                'Lucro': float(lucro),
                'Valor Previsto': float(previsao),
            })
        elif ativo and row.low < (STOP_LOSS[symbol] * preco_compra):
            ativo = False
            lucro = (STOP_LOSS[symbol] * preco_compra) - preco_compra
            resultados.append({
                'Tipo de Operação': f'Stop Loss ({count_periodos})',
                'Preço de Compra': preco_compra,
                'High do periodo': row.high,
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
            })
    count_periodos += 1
# Exportar o DataFrame para um arquivo CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f'results/resultados_backtest_{symbol}_{TEMPO_DO_MODELO}_{DIAS_DE_BT}.csv', index=False)
