from utils import *
import warnings
import os
import pandas as pd
import numpy as np
import pickle
from binance.client import Client

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)
symbol = 'BTCUSDT'  # MOEDA
interval = Client.KLINE_INTERVAL_15MINUTE  # INTERVALO

# Carregar o modelo salvo
with open(f'models/rl_{symbol}.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Recupere o modelo e o scaler
modelo = dados_carregados['modelo']
scalers = dados_carregados['scalers']

# Carregar dados históricos de preços (substitua por seus próprios dados)
dados_historicos = pd.read_csv(f'backtesting/{symbol}_28_09.csv')

# Parâmetros da estratégia
saldo_inicial = 100  # Saldo inicial da conta
saldo = saldo_inicial
ativo = None  # Para rastrear o ativo em carteira
preco_compra = 0  # Preço de compra do ativo
count_high_previsto = 0
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
    if row.close is not None and row.close >= previsao:
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
        elif row.high > preco_compra:
            # Vende quando atinge o high e o preço é maior do que o preço de compra
            saldo = ativo * row.close
            ativo = None
            lucro = saldo - saldo_inicial
            resultados.append({
                'Tipo de Operação': f'Venda ({count_periodos})',
                'Lucro': lucro,
                'Valor Previsto': previsao,
                'Valor Real': row.close,
                'Diferença': previsao - row.close,
                'Saldo': saldo
            })
            count_high_previsto += 1
    elif ativo is not None:
        # Se o high não foi atingido e temos ativo em carteira, considere vender no final do período
        saldo = ativo * row.close
        ativo = None
        lucro = saldo - saldo_inicial
        resultados.append({
            'Tipo de Operação': f'Venda (no final do período {count_periodos})',
            'Lucro': lucro,
            'Valor Previsto': previsao,
            'Valor Real': row.close,
            'Diferença': previsao - row.close,
            'Saldo': saldo
        })
    count_periodos += 1

# Exportar o DataFrame para um arquivo CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(f'results/resultados_backtest_{symbol}.csv', index=False)

# Calcular quantas vezes atingiu o high previsto
print(f'Atingiu o high previsto em {count_high_previsto} vezes de {len(dados_historicos) // 4}')
print(f'Usamos o primeiro período para prever o high do 4º período. Assim que chega no 4º período, o script compara se foi atingido. Se sim, ele incrementa o contador.')
