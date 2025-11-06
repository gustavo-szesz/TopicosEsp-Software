# Data Explorer & ML Playground

Protótipo de aplicação web em Python (Streamlit) para upload de CSV, análise de dados, visualizações interativas e treinamento de modelos de machine learning.

Arquivos principais:

- `app.py` - aplicação Streamlit com upload, EDA, visualizações e interface de ML.
- `model.py` - utilitários de treinamento, seleção de modelos e avaliação.
- `utils.py` - funções auxiliares de EDA e plot.
- `requirements.txt` - dependências.

Como rodar (local):

1. Crie um virtualenv e instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Rode a aplicação (Streamlit prototype):

```bash
streamlit run app.py
```

Ou rode a versão Flask criada a partir deste protótipo:

```bash
# ative o virtualenv
source .venv/bin/activate
# execute o app Flask
python app_flask.py
```

Funcionalidades implementadas no protótipo:

- Upload de CSV e preview dos dados
- Resumo básico (tipos, missing, únicos)
- Gráficos rápidos (histograma, barras, boxplot, scatter, pizza)
- Mapa simples quando colunas de lat/lon são detectadas
- Treinamento de modelos básicos (LinearRegression, RandomForest, KNN, LogisticRegression)
- Ajuste de hiperparâmetros simples via UI
- Métricas exibidas após treino

Próximos passos sugeridos:

- Melhor tratamento de features categóricas (encoding automático/one-hot)
- Salvamento/versão de modelos treinados e download
- Testes unitários e cobertura
- Melhor UX para predição com entradas tipadas
