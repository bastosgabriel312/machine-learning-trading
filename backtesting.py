from utils import * 

###### CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)
symbol = 'BTCUSDT'                                 ## MOEDA
interval = Client.KLINE_INTERVAL_15MINUTE          ## INTERVALO

# # Carregar o modelo salvo
with open(f'models/rl_{symbol}.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Recupere o modelo e o scaler
modelo = dados_carregados['modelo']
scalers = dados_carregados['scalers']

# Carregar dados históricos de preços (substitua por seus próprios dados)
dados_historicos = pd.read_csv(f'backtesting/{symbol}.csv')


# Parâmetros da estratégia
saldo_inicial = 10000  # Saldo inicial da conta
saldo = saldo_inicial
ativo = None  # Para rastrear o ativo em carteira
count_high_previsto = 0

# Realizar backtesting
for indice, row in dados_historicos.iterrows():
    entrada_modelo = row.drop(['Unnamed: 0','timestamp','timestamp_to_date','close_time','ignore'])

    # Normalizar cada coluna individualmente usando os scalers correspondentes
    entrada_modelo_preprocessada = {}
    for col in entrada_modelo.index:
        if col in scalers:
            scaler = scalers[col]
            valor = np.array(row[col]).reshape(1, -1) # Transforma o valor em uma matriz 2D
            entrada_modelo_preprocessada[f'normalized_{col}'] = scaler.transform(valor)[0][0]
        else:
            entrada_modelo_preprocessada[col] = row[col]
            
    # Transforma o dicionário em um DataFrame 2D
    entrada_modelo_preprocessada_df = pd.DataFrame([entrada_modelo_preprocessada])
    # Crie um DataFrame com os nomes das colunas
    colunas = list(entrada_modelo_preprocessada.keys())
    entrada_modelo_preprocessada_df = pd.DataFrame([entrada_modelo_preprocessada], columns=colunas)

    previsao = modelo.predict(entrada_modelo_preprocessada_df)
    
    # Verificar se o target-high foi atingido e realizar a venda
    if row.high is not None and previsao >= row.high:

        print('>>> calcula se vale a pena o lucro com a taxa da binance')
        count_high_previsto += 1

 
#calcular quantas vezes atingiu o high previsto
print(f'atingiu o high previsto em {count_high_previsto} vezes de {len(dados_historicos)}')
