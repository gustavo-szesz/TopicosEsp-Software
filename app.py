import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from model import get_model, train_model, evaluate_model
from utils import summarize_dataframe, plot_basic, plot_scatter_matrix

st.set_page_config(page_title="Data + ML Explorer", layout="wide")

st.title("Data Explorer & ML Playground")

st.markdown("Faça upload de um arquivo CSV, explore os dados, crie visualizações e treine modelos de ML.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"]) 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.header("Visão geral")
    st.sidebar.write(f"Linhas: {df.shape[0]}")
    st.sidebar.write(f"Colunas: {df.shape[1]}")

    # Show dataframe and summary
    st.subheader("Preview dos dados")
    st.dataframe(df.head())

    st.subheader("Resumo estatístico")
    st.write(summarize_dataframe(df))

    st.subheader("Visualizações")
    with st.expander("Criar gráfico rápido"):
        cols = df.columns.tolist()
        x = st.selectbox("Eixo X / Categórica", cols, index=0)
        y = st.selectbox("Eixo Y (opcional) / Numérica", [None] + cols, index=0)
        chart_type = st.selectbox("Tipo de gráfico", ["Histograma", "Barra", "Boxplot", "Scatter", "Pizza", "Mapa (lat/lon)"])

        if chart_type == "Histograma":
            fig = px.histogram(df, x=x)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Barra":
            fig = px.bar(df, x=x, y=y if y else None)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Boxplot":
            if y:
                fig = px.box(df, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Escolha uma coluna Y numérica para boxplot.")
        elif chart_type == "Scatter":
            if y:
                fig = px.scatter(df, x=x, y=y, color=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Escolha uma coluna Y numérica para scatter.")
        elif chart_type == "Pizza":
            fig = px.pie(df, names=x)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Mapa (lat/lon)":
            # Try to infer lat/lon
            lat_cols = [c for c in df.columns if 'lat' in c.lower() or 'latitude' in c.lower()]
            lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'longitude' in c.lower()]
            if lat_cols and lon_cols:
                lat = lat_cols[0]
                lon = lon_cols[0]
                st.map(df[[lat, lon]].dropna())
            else:
                st.info("Não foi possível encontrar colunas de latitude/longitude automaticamente. Renomeie suas colunas para incluir 'lat'/'lon' ou 'latitude'/'longitude'.")

    st.subheader("Análise Avançada")
    with st.expander("Matriz de correlação e pairplot"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            st.write("Pairplot (primeiras 6 colunas numéricas)")
            cols_for_pair = numeric_cols[:6]
            fig2 = px.scatter_matrix(df[cols_for_pair])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Poucas colunas numéricas para correlação/pairplot.")

    st.subheader("Machine Learning")
    with st.expander("Configurar e treinar modelo"):
        st.write("Escolha a variável alvo (target) e o tipo de tarefa")
        cols = df.columns.tolist()
        target = st.selectbox("Target", [None] + cols, index=0)
        task = st.selectbox("Tipo de tarefa", ["auto", "regression", "classification"]) 

        if target is not None:
            # Prepare data
            X = df.drop(columns=[target]).select_dtypes(include=[np.number, object, 'category'])
            y = df[target]

            st.write(f"Features: {X.shape[1]}, Observações: {X.shape[0]}")

            # Model selection
            model_choice = st.selectbox("Modelo", ["LinearRegression", "RandomForest", "KNN", "LogisticRegression"])

            # Model hyperparams
            params = {}
            if model_choice == "RandomForest":
                params['n_estimators'] = st.number_input('n_estimators', value=100, step=10)
                params['max_depth'] = st.number_input('max_depth (0=None)', value=0, step=1)
            elif model_choice == "KNN":
                params['n_neighbors'] = st.number_input('n_neighbors', value=5, step=1)
            elif model_choice == "LinearRegression":
                pass
            elif model_choice == "LogisticRegression":
                params['max_iter'] = st.number_input('max_iter', value=100, step=10)

            test_size = st.slider('Proporção de teste', 0.05, 0.5, 0.2)

            if st.button('Treinar modelo'):
                model = get_model(model_choice, params)
                report = train_model(model, X, y, task=task, test_size=test_size)
                st.subheader('Relatório de Treinamento')
                st.write(report['metrics'])
                if 'confusion_matrix' in report:
                    st.write('Confusion matrix')
                    st.write(report['confusion_matrix'])

                    fig = px.imshow(report['confusion_matrix'], text_auto=True)
                    st.plotly_chart(fig)

                st.success('Treinamento concluído. Use o painel de predição para testar.')

    with st.expander("Predição interativa"):
        st.write('Forneça valores para as features e gere uma predição com o modelo treinado na sessão (se existir).')
        if 'last_model' in st.session_state:
            model_info = st.session_state['last_model']
            model_obj = model_info['model']
            feature_names = model_info['features']
            st.write(f"Modelo disponível: {model_info['name']}")
            input_vals = {}
            for f in feature_names:
                input_vals[f] = st.text_input(f, '')
            if st.button('Prever (pontual)'):
                # simple casting: try numeric
                x_in = []
                for f in feature_names:
                    v = input_vals[f]
                    try:
                        x_in.append(float(v))
                    except:
                        x_in.append(v)
                import numpy as _np
                x_arr = _np.array([x_in], dtype=object)
                pred = model_obj.predict(x_arr)
                st.write('Predição:', pred)
        else:
            st.info('Nenhum modelo treinado nesta sessão.')

else:
    st.info('Faça upload de um arquivo CSV para começar.')
