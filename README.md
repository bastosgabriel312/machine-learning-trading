# deep-learning-trading

Este projeto consiste em uma implementação de uma rede neural LSTM (Long Short-Term Memory) para previsão de preços no mercado financeiro de criptomoedas. O modelo é treinado utilizando dados históricos de preços e realiza a previsão de preços futuros com base nesses dados.

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
git clone https://github.com/bastosgabriel312/deep-learning-trading.git


2. Instale as dependências:
pip install tensorflow keras scikit-learn pandas numpy matplotlib python-binance h5py


## Utilização

1. Preencha sua chave de API e segredo no arquivo `main.py`:
api_key = 'sua-chave-de-api'
api_secret = 'seu-segredo'


2. Execute o arquivo `main.py` para treinar o modelo e realizar as previsões:
python main.py


3. Os resultados serão exibidos no console e um gráfico com as previsões será gerado.

## Estrutura do projeto

- `main.py`: Arquivo principal que contém o código para treinamento e previsão utilizando a rede LSTM.
- `utils.py`: Módulo com funções utilitárias para pré-processamento dos dados, criação do modelo LSTM e avaliação das previsões.
- `models/`: Pasta que armazena os modelos treinados em formato HDF5.
- `data/`: Pasta opcional para armazenar os dados históricos.
- `results/`: Pasta opcional para armazenar os resultados das previsões.

## Resultados

- Os resultados das previsões são exibidos no console e um gráfico com as previsões é gerado.
