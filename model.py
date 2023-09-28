from utils import * 

###### CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)

symbol = 'BTCUSDT'                                 ## MOEDA
interval = Client.KLINE_INTERVAL_15MINUTE          ## INTERVALO
limit = 17280                                       ## LIMITE

###### RETORNA OS DADOS HISTÓRICOS COM MÉTRICA DE MVRV
df = get_concatenate_klines_mvrv(client,symbol,interval,limit)

###### PREPROCESSA OS DADOS
dataset,scalers = preprocess_manual(df)

COLUNAS_DATASET_COM_ALV0 = dataset.columns
COLUNAS_DATASET_SEM_ALVO = dataset.copy().drop('target-high', axis=1).columns

# Separando as variáveis independentes (X) e a variável alvo (y)
dataset = dataset.copy()
X = dataset.drop(['target-high'], axis=1)
y = dataset['target-high']

#REGRESSÃO LINEAR MULTIPLA (ALVO TARGET-HIGH)
# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Criando um objeto de regressão linear
lr = LinearRegression()

# Treinando o modelo com os dados de treinamento
lr.fit(X_train, y_train)

# Fazendo previsões com os dados de teste
y_pred = lr.predict(X_test)

print(X_test.columns)
# Avaliando o desempenho do modelo
mse_lr = mean_squared_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)

print(f'mean_squared_error: {mse_lr}')
print(f'r2 score: {r2_lr}')

# Salvar um modelo treinado
with open(f'models/rl_{symbol}.pkl', 'wb') as arquivo:
    pickle.dump({'modelo': lr, 'scalers': scalers}, arquivo)

