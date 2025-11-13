import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import get_model, train_model
from utils import clean_price_column, preprocess_dates
from sklearn.model_selection import train_test_split
from datetime import datetime
import calendar

st.set_page_config(page_title="Airbnb Price Analyzer", layout="wide")

st.title("Airbnb Price Analyzer")
st.markdown("Análise de preços, disponibilidade e tendências temporais")

uploaded_file = st.file_uploader("Carregue o dataset do Airbnb (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Aplicar limpeza automática
    if 'price' in df.columns:
        df['price'] = clean_price_column(df['price'])
    if 'service fee' in df.columns:
        df['service fee'] = clean_price_column(df['service fee'])
    
    # Processar datas
    df = preprocess_dates(df)
    
    st.sidebar.header("Filtros")
    
    # Filtros
    if 'neighbourhood group' in df.columns:
        bairros = ['Todos'] + df['neighbourhood group'].dropna().unique().tolist()
        bairro_selecionado = st.sidebar.selectbox("Bairro", bairros)
        if bairro_selecionado != 'Todos':
            df = df[df['neighbourhood group'] == bairro_selecionado]
    
    if 'room type' in df.columns:
        tipos_quarto = ['Todos'] + df['room type'].dropna().unique().tolist()
        tipo_selecionado = st.sidebar.selectbox("Tipo de Quarto", tipos_quarto)
        if tipo_selecionado != 'Todos':
            df = df[df['room type'] == tipo_selecionado]
    
    # Análise de Preços
    st.header("Análise de Preços")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric("Preço Médio", f"${avg_price:.2f}")
    with col2:
        if 'availability 365' in df.columns:
            avg_availability = df['availability 365'].mean()
            st.metric("Disponibilidade Média (dias/ano)", f"{avg_availability:.0f}")
    with col3:
        if 'review rate number' in df.columns:
            avg_rating = df['review rate number'].mean()
            st.metric("Avaliação Média", f"{avg_rating:.1f}/5")
    
    # Visualização 1: Preços por Região
    st.subheader("Preços por Região e Tipo de Quarto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'neighbourhood group' in df.columns and 'price' in df.columns:
            # Limitar outliers para melhor visualização
            df_viz = df[df['price'] <= df['price'].quantile(0.95)]
            fig = px.box(df_viz, x='neighbourhood group', y='price', 
                        title="Distribuição de Preços por Bairro",
                        labels={'neighbourhood group': 'Bairro', 'price': 'Preço ($)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'room type' in df.columns and 'price' in df.columns:
            price_by_room = df.groupby('room type')['price'].mean().reset_index()
            fig = px.bar(price_by_room, x='room type', y='price',
                        title="Preço Médio por Tipo de Quarto",
                        labels={'room type': 'Tipo de Quarto', 'price': 'Preço Médio ($)'},
                        color='price')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Visualização 2: Análise Temporal
    st.subheader("Análise Temporal")
    
    if 'last_review_year' in df.columns and 'last_review_month' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Agrupar por ano e mês de forma segura
            monthly_data = df.groupby(['last_review_year', 'last_review_month']).agg({
                'price': 'mean',
                'availability 365': 'mean',
                'id': 'count'
            }).reset_index()
            
            # Criar string de data de forma segura
            monthly_data = monthly_data.dropna(subset=['last_review_year', 'last_review_month'])
            monthly_data['last_review_year'] = monthly_data['last_review_year'].astype(int)
            monthly_data['last_review_month'] = monthly_data['last_review_month'].astype(int)
            
            # Criar coluna de período para ordenação
            monthly_data['period'] = monthly_data['last_review_year'] * 100 + monthly_data['last_review_month']
            monthly_data = monthly_data.sort_values('period')
            
            # Criar label para o eixo X
            monthly_data['period_label'] = monthly_data['last_review_month'].astype(str) + '/' + monthly_data['last_review_year'].astype(str)
            
            fig = px.line(monthly_data, x='period_label', y='price', 
                         title="Evolução do Preço Médio ao Longo do Tempo",
                         labels={'period_label': 'Período (Mês/Ano)', 'price': 'Preço Médio ($)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Disponibilidade por período
            fig = px.line(monthly_data, x='period_label', y='availability 365',
                         title="Evolução da Disponibilidade Média",
                         labels={'period_label': 'Período (Mês/Ano)', 'availability 365': 'Disponibilidade Média (dias)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Mapa de Calor de Preços
    st.subheader("Mapa de Preços por Localização")
    
    if all(col in df.columns for col in ['lat', 'long', 'price']):
        df_map = df.dropna(subset=['lat', 'long', 'price'])
        # Limitar para melhor performance
        df_map = df_map.head(1000)
        
        if not df_map.empty:
            fig = px.scatter_mapbox(df_map, 
                                  lat='lat', 
                                  lon='long', 
                                  color='price',
                                  size='price',
                                  hover_data=['neighbourhood', 'room type'],
                                  color_continuous_scale='viridis',
                                  zoom=10,
                                  title="Mapa de Preços por Localização")
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
    
    # Análise de Correlação
    st.subheader("Correlações com Preço")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1 and 'price' in numeric_cols:
        corr_matrix = df[numeric_cols].corr()
        
        # Focar nas correlações com preço
        price_correlations = corr_matrix['price'].sort_values(ascending=False)
        
        st.write("**Correlações com Preço:**")
        corr_df = pd.DataFrame({
            'Variável': price_correlations.index,
            'Correlação': price_correlations.values
        })
        st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', subset=['Correlação']))
        
        # Heatmap das principais correlações
        top_correlations = price_correlations[1:9]  # Excluir price consigo mesmo
        if len(top_correlations) > 1:
            cols_for_heatmap = ['price'] + top_correlations.index.tolist()
            fig = px.imshow(corr_matrix.loc[cols_for_heatmap, cols_for_heatmap],
                           text_auto=True, 
                           aspect="auto",
                           title="Matriz de Correlação - Principais Variáveis",
                           color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    # Machine Learning para Predição de Preços
    st.header("Predição de Preços")
    
    with st.expander("Configurar Modelo de Predição"):
        st.write("Treine um modelo para prever preços baseado nas características do imóvel")
        
        # Seleção de features
        feature_options = [
            'minimum nights', 'number of reviews', 'reviews per month',
            'review rate number', 'calculated host listings count', 'availability 365',
            'Construction year', 'service fee'
        ]
        
        available_features = [col for col in feature_options if col in df.columns]
        selected_features = st.multiselect("Selecione as features para o modelo",
                                          available_features,
                                          default=available_features[:4])
        
        if selected_features and 'price' in df.columns:
            # Preparar dados
            ml_df = df[selected_features + ['price']].dropna()
            X = ml_df[selected_features]
            y = ml_df['price']
            
            st.write(f"Dados disponíveis para treino: {len(X)} observações")
            
            if len(X) > 10:
                model_choice = st.selectbox("Modelo", 
                                           ["RandomForest", "LinearRegression", "KNN"])
                
                test_size = st.slider("Proporção para teste", 0.1, 0.4, 0.2)
                
                if st.button("Treinar Modelo de Predição"):
                    with st.spinner("Treinando modelo..."):
                        model = get_model(model_choice, {}, task='regression')
                        report = train_model(model, X, y, task='regression', test_size=test_size)
                        
                        if report is not None:
                            st.session_state['price_model'] = {
                                'model': report['model'],
                                'features': report['features'],
                                'metrics': report['metrics'],
                                'feature_importance': report.get('feature_importance')
                            }
                            
                            st.success("Modelo treinado com sucesso!")
                            
                            # Mostrar métricas
                            st.write("** Métricas do Modelo:**")
                            metrics = report['metrics']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² Score", f"{metrics['r2']:.3f}")
                            with col2:
                                st.metric("RMSE", f"${metrics['rmse']:.2f}")
                            with col3:
                                st.metric("MAE", f"${metrics['mae']:.2f}")
                            
                            # Mostrar importância das features se disponível
                            if report.get('feature_importance'):
                                st.write("** Importância das Features:**")
                                importance_df = pd.DataFrame({
                                    'Feature': list(report['feature_importance'].keys()),
                                    'Importância': list(report['feature_importance'].values())
                                })
                                st.dataframe(importance_df)
    
    # Predição Interativa
    with st.expander(" Fazer Predição de Preço"):
        if 'price_model' in st.session_state:
            model_info = st.session_state['price_model']
            st.write("**Insira os valores para predição de preço:**")
            
            input_vals = {}
            col1, col2 = st.columns(2)
            
            # Calcular estatísticas para guiar o usuário
            stats = {}
            for feature in model_info['features']:
                if feature in df.columns:
                    stats[feature] = {
                        'min': df[feature].min(),
                        'max': df[feature].max(),
                        'mean': df[feature].mean()
                    }
            
            for i, feature in enumerate(model_info['features']):
                with col1 if i % 2 == 0 else col2:
                    if feature in stats:
                        default_val = stats[feature]['mean']
                        min_val = float(stats[feature]['min'])
                        max_val = float(stats[feature]['max'])
                        
                        input_vals[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=float(default_val),
                            step=1.0 if feature in ['minimum nights', 'number of reviews', 'Construction year'] else 0.1,
                            help=f"Mín: {min_val:.1f}, Máx: {max_val:.1f}, Média: {default_val:.1f}"
                        )
            
            if st.button("Calcular Preço Previsto"):
                try:
                    x_in = [input_vals[feature] for feature in model_info['features']]
                    x_arr = np.array([x_in]).reshape(1, -1)
                    pred = model_info['model'].predict(x_arr)
                    predicted_price = pred[0]
                    
                    st.success(f"**Preço Previsto: ${predicted_price:.2f}**")
                    
                    # Mostrar comparação com a média
                    avg_price = df['price'].mean()
                    diff = predicted_price - avg_price
                    diff_pct = (diff / avg_price) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Preço Médio do Mercado", f"${avg_price:.2f}")
                    with col2:
                        st.metric("Diferença", f"${diff:.2f}", f"{diff_pct:+.1f}%")
                        
                except Exception as e:
                    st.error(f" Erro na predição: {e}")
        else:
            st.info(" Treine um modelo primeiro na seção acima")
    
    # Insights Automáticos
    st.header(" Insights do Mercado")
    
    if 'price' in df.columns:
        insights = []
        
        # Insight 1: Melhor época para preços
        if all(col in df.columns for col in ['last_review_month', 'price']):
            monthly_avg = df.groupby('last_review_month')['price'].mean()
            if not monthly_avg.empty:
                best_month_idx = monthly_avg.idxmin()
                worst_month_idx = monthly_avg.idxmax()
                best_month_name = calendar.month_name[int(best_month_idx)] if not pd.isna(best_month_idx) else "N/A"
                worst_month_name = calendar.month_name[int(worst_month_idx)] if not pd.isna(worst_month_idx) else "N/A"
                
                insights.append(f" **Melhor mês para preços**: {best_month_name} (${monthly_avg[best_month_idx]:.2f})")
                insights.append(f" **Pior mês para preços**: {worst_month_name} (${monthly_avg[worst_month_idx]:.2f})")
        
        # Insight 2: Tipo de quarto mais caro
        if 'room type' in df.columns:
            room_prices = df.groupby('room type')['price'].mean()
            if not room_prices.empty:
                most_expensive = room_prices.idxmax()
                least_expensive = room_prices.idxmin()
                insights.append(f" **Tipo mais caro**: {most_expensive} (${room_prices[most_expensive]:.2f})")
                insights.append(f" **Tipo mais barato**: {least_expensive} (${room_prices[least_expensive]:.2f})")
        
        # Insight 3: Bairro mais caro
        if 'neighbourhood group' in df.columns:
            neighborhood_prices = df.groupby('neighbourhood group')['price'].mean()
            if not neighborhood_prices.empty:
                most_expensive_hood = neighborhood_prices.idxmax()
                insights.append(f" **Bairro mais caro**: {most_expensive_hood} (${neighborhood_prices[most_expensive_hood]:.2f})")
        
        # Insight 4: Relação preço-disponibilidade
        if 'availability 365' in df.columns:
            correlation = df['price'].corr(df['availability 365'])
            if not pd.isna(correlation):
                if correlation < -0.2:
                    insights.append("**Alta correlação negativa**: Preços menores quando disponibilidade é maior")
                elif correlation > 0.2:
                    insights.append(" **Alta correlação positiva**: Preços maiores quando disponibilidade é maior")
                else:
                    insights.append(" **Baixa correlação**: Preço e disponibilidade não têm relação forte")
        
        # Mostrar insights
        for insight in insights:
            st.write(f"• {insight}")

else:
    st.info("👆 Faça upload do dataset do Airbnb para começar a análise")