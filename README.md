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
- h5py

## Instalação

1. Clone este repositório:
git clone https://github.com/bastosgabriel312/machine-learning-trading.git


2. Instale as dependências:
pip install tensorflow keras scikit-learn pandas numpy matplotlib python-binance h5py pytz


## Utilização

1. Preencha sua chave de API e segredo no arquivo `model.py`:
api_key = 'sua-chave-de-api'
api_secret = 'seu-segredo'


2. Execute o arquivo `model.py` para treinar e salvar o modelo:
python model.py


3. Para carregar o modelo e realizar predições execute o `main.py`.

## Estrutura do projeto

- `main.py`: Arquivo principal que contém o código para carregar o modelo e prever os valores.
- `model.py`: Arquivo que contém o código para treinar e salvar o modelo.
- `utils.py`: Módulo com funções utilitárias para pré-processamento dos dados.
- `models/`: Pasta que armazena os modelos treinados.
- `data/`: Pasta opcional para armazenar os dados históricos.
- `results/`: Pasta opcional para armazenar os resultados das previsões.

## Resultados

- .
