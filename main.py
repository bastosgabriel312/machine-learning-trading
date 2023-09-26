from utils import * 

###### CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)
symbol = 'BTCUSDT'                                 ## MOEDA
interval = Client.KLINE_INTERVAL_15MINUTE          ## INTERVALO
limit = 30000                                       ## LIMITE

###### RETORNA OS DADOS HISTÓRICOS COM MÉTRICA DE MVRV
df = get_concatenate_klines_mvrv(client,symbol,interval,limit)
df.to_csv(f'backtesting/{symbol}.csv')

# # Carregar o modelo salvo
with open(f'models/rl_{symbol}.pkl', 'rb') as arquivo:
    modelo_carregado = pickle.load(arquivo)

