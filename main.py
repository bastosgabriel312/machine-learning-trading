from utils import * 

#### NO MOMENTO SENDO UTILIZADO PARA SALVAR DADOS DE BACKTESTING

###### CONFIGURAÇÕES DA BINANCE
api_key_binance = os.getenv('API_KEY')
api_secret_binance = os.getenv('API_SECRET')
client = Client(api_key_binance, api_secret_binance)

symbol = 'ETCUSDT'                                 ## MOEDA
interval = Client.KLINE_INTERVAL_15MINUTE          ## INTERVALO
limit = 96 # 4 X 24  = 24H                          ## LIMITE

###### RETORNA OS DADOS HISTÓRICOS COM MÉTRICA DE MVRV
df = get_concatenate_klines_mvrv(client,symbol,interval,limit)
df.to_csv(f'backtesting/{symbol}_28_09.csv')
