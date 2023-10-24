from utils import *
import warnings
import os
import pandas as pd
import numpy as np
import pickle
from binance.client import Client

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# PRECO DE COMPRA * 0.3 -- todo: utilizar boas praticas de stop_loss
STOP_LOSS = {'BTCUSDT':0.5,'ETCUSDT':0.5,'BNBUSDT':0.5}
# PRECO DE COMPRA * 1.3 -- todo: utilizar boas praticas de stop_gain
STOP_GAIN = {'BTCUSDT':1.20,'ETCUSDT':1.20,'BNBUSDT':1.20}



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
ativo = None  # Para rastrear o ativo em carteira
preco_compra = 0  # Preço de compra do ativo
count_periodos = 0

# Lista para armazenar os resultados de cada período
resultados = []

# Realizar backtesting
for indice, row in dados_historicos.iterrows():
    if count_periodos == 5:
        count_periodos = 0
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
        previsao = modelo.predict(entrada_modelo_preprocessada_df.values)

    # Verificar se o target-high foi atingido e realizar a compra ou venda
    if ativo is None:
        # Compra no início do período
        ativo = saldo / row.close
        preco_compra = row.close
        saldo = 0
        resultados.append({
            'Tipo de Operação': f'Compra ({count_periodos})',
            'Preço de Compra': row.close,
            'Saldo': saldo,
            'Ativo em Carteira': ativo
        })
        
    elif row.close >= previsao and preco_compra < row.close:
            # Vende quando atinge o high e o preço é maior do que o preço de compra
            saldo = ativo * row.close
            ativo = None
            lucro = saldo - saldo_inicial
            resultados.append({
                'Tipo de Operação': f'Venda ({count_periodos})',
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
                'Valor Real': row.close,
                'Diferença': float(previsao - row.close),
                'Saldo': saldo
            })
    elif (row.close >= (STOP_GAIN[symbol] * preco_compra)):
            # Vende quando atinge o stop gain
            saldo = ativo * row.close
            ativo = None
            lucro = saldo - saldo_inicial
            resultados.append({
                'Tipo de Operação': f'Venda por STOP GAIN ({count_periodos})',
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
                'Valor Real': row.close,
                'Diferença': float(previsao - row.close),
                'Saldo': saldo
            })
    # vende se atingir o stop_loss
    elif (row.close <= (STOP_LOSS[symbol] * preco_compra)):
        # Vende por prejuizo
            saldo = ativo * row.close
            ativo = None
            lucro = saldo - saldo_inicial
            resultados.append({
                'Tipo de Operação': f'Venda por STOP GAIN ({count_periodos})',
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
                'Valor Real': row.close,
                'Diferença': float(previsao - row.close),
                'Saldo': saldo
            })
    elif ativo is not None:
        resultados.append({
            'Tipo de Operação': f'(no final do período {count_periodos})',
            'Lucro': 'x',
            'Valor Previsto': float(previsao),
            'Valor Real': row.close,
            'Diferença': float(previsao - row.close),
            'Saldo': saldo
        })
    count_periodos += 1
if ativo is not None:
    # venda final
    saldo = ativo * row.close
    ativo = None
    lucro = saldo - saldo_inicial
    resultados.append({
                'Tipo de Operação': f'(venda final {count_periodos})',
                'Lucro': lucro,
                'Valor Previsto': float(previsao),
                'Valor Real': row.close,
                'Diferença': float(previsao - row.close),
                'Saldo': saldo
            })
# Exportar o DataFrame para um arquivo CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f'results/resultados_backtest_{symbol}_{TEMPO_DO_MODELO}_{DIAS_DE_BT}.csv', index=False)
