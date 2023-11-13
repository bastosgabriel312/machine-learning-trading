# machine-learning-trading

Este projeto consiste na avaliação e implementação de modelos de machine learning para previsão de preços no mercado financeiro de criptomoedas. O modelo é treinado utilizando dados históricos de preços provenientes da binance e realiza a previsão de valores futuros com base nesses dados.

## Pré-requisitos

- Python 3.9.8
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- Binance API (chave de API e segredo)
- pickle

## Instalação

1. Clone este repositório:
git clone https://github.com/bastosgabriel312/machine-learning-trading.git


2. Instale as dependências:
pip install requirements.txt


## Utilização

1. Preencha sua chave de API e segredo no arquivo `model.py`:
api_key = 'sua-chave-de-api'
api_secret = 'seu-segredo'


2. Execute o arquivo `model.py` para treinar e salvar o modelo:
python model.py


3. Para carregar o modelo e realizar predições execute o `main.py`.

## Estrutura do projeto

- `main.py`: arquivo que realiza a extração de informações para backtesting (posteriormente será responsável pelo programa principal).
- `backtesting.py`: arquivo responsável por realizar o teste retroativo (backtesting) e salvar os resultados em \results
- `model.py`: Arquivo que contém o código para treinar e salvar o modelo.
- `utils.py`: Módulo com funções utilitárias para pré-processamento dos dados.
- `notebook-etapa-avaliacao`: Arquivo responsável pela criação e avaliação dos modelos
- `models/`: Pasta que armazena os modelos treinados.
- `data/`: Pasta para armazenar os dados históricos.
- `results/`: Pasta para armazenar os resultados das previsões.

## Demonstração e Capturas de Tela
   Exemplo de resultados extraidos do backtesting.py (utilizando dados de 20 dias do token BNB):<br>
| Tipo de Operação | Preço de Compra | High do Período | Lucro                | Valor Previsto      |
|-------------------|-----------------|-----------------|----------------------|---------------------|
| Compra (0)        | 210.5           | 213.1           | 0.0                  | 213.95435914475374 |
| Stop Gain (0)     | 210.5           | 214.4           | 1.3148155533062038   | 213.95435914475374 |
| Compra (0)        | 213.4           | 219.0           | 0.0                  | 219.06261516643522 |
| Stop Gain (0)     | 213.4           | 218.4           | 3.4719890147708554   | 219.06261516643522 |

  
  
