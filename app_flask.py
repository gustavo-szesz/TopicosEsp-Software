import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import joblib
from model import get_model, train_model
from utils import clean_price_column, preprocess_dates, normalize_neighbourhood_names
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-123')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE_DIR, 'uploads')
MODELS = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle filters
        neighbourhood_filter = request.form.get('neighbourhood_filter', 'Todos')
        room_type_filter = request.form.get('room_type_filter', 'Todos')
        
        session['neighbourhood_filter'] = neighbourhood_filter
        session['room_type_filter'] = room_type_filter
        
        return redirect(url_for('index'))
    
    df = None
    plots = {}
    metrics = {}
    insights = []
    ml_options = {}

    ml_metrics = session.get('price_model', {}).get('metrics', {})
    if ml_metrics:
        metrics['model_rmse'] = f"RMSE: {ml_metrics.get('rmse', 'N/A')}"
        metrics['model_mae'] = f"MAE: {ml_metrics.get('mae', 'N/A')}"
        metrics['model_r2_score'] = f"R² Score: {ml_metrics.get('r2', 'N/A')}" # Corrigido: usando a chave 'r2' do model.py
        metrics['model_mean_actual_price'] = f"Preço Real Médio: ${ml_metrics.get('mean_actual_price', 'N/A')}"
        metrics['model_mean_predicted_price'] = f"Preço Previsto Médio: ${ml_metrics.get('mean_predicted_price', 'N/A')}"
    
    if 'df_path' in session and os.path.exists(session['df_path']):
        df = pd.read_csv(session['df_path'])
        
        # Apply cleaning
        if 'price' in df.columns:
            df['price'] = clean_price_column(df['price'])
        if 'service fee' in df.columns:
            df['service fee'] = clean_price_column(df['service fee'])
        
        df = preprocess_dates(df)
        
        # Normalize neighbourhood names
        if 'neighbourhood group' in df.columns:
            df = normalize_neighbourhood_names(df)
        
        # Apply filters
        neighbourhood_filter = session.get('neighbourhood_filter', 'Todos')
        room_type_filter = session.get('room_type_filter', 'Todos')
        
        if neighbourhood_filter != 'Todos' and 'neighbourhood group' in df.columns:
            df = df[df['neighbourhood group'] == neighbourhood_filter]
        if room_type_filter != 'Todos' and 'room type' in df.columns:
            df = df[df['room type'] == room_type_filter]
        
        # DEBUG: Verificar dados
        debug_data(df)
        
        # Get ML options from current data
        ml_options = get_ml_options(df)
        
        # Calculate metrics
        if 'price' in df.columns:
            price_clean = df['price'].dropna()
            if len(price_clean) > 0:
                metrics['avg_price'] = f"${price_clean.mean():.2f}"
                metrics['median_price'] = f"${price_clean.median():.2f}"
                metrics['min_price'] = f"${price_clean.min():.2f}"
                metrics['max_price'] = f"${price_clean.max():.2f}"
        
        if 'availability 365' in df.columns:
            availability_clean = df['availability 365'].dropna()
            if len(availability_clean) > 0:
                metrics['avg_availability'] = f"{availability_clean.mean():.0f}"
        
        if 'review rate number' in df.columns:
            rating_clean = df['review rate number'].dropna()
            if len(rating_clean) > 0:
                metrics['avg_rating'] = f"{rating_clean.mean():.1f}/5"
        
        # Create plots
        plots = create_plots(df)
        
        # Generate insights
        insights = generate_insights(df)
    
    # Get filter options
    filter_options = get_filter_options(session)
    
    return render_template('index.html', 
                         plots=plots, 
                         metrics=metrics, 
                         insights=insights,
                         filter_options=filter_options,
                         ml_options=ml_options,
                         has_data=df is not None)

def debug_data(df):
    """Função para debug dos dados"""
    print("\n=== DEBUG DOS DADOS ===")
    print(f"Total de registros: {len(df)}")
    
    # Verificar dados de preço
    if 'price' in df.columns:
        price_data = df['price'].dropna()
        print(f"\nDados de preço: {len(price_data)} registros")
        if len(price_data) > 0:
            print(f"Preço: ${price_data.min():.2f} a ${price_data.max():.2f} (média: ${price_data.mean():.2f})")
    
    # Verificar neighbourhood groups
    if 'neighbourhood group' in df.columns:
        neighbourhoods = df['neighbourhood group'].dropna().unique()
        print(f"\nBairros disponíveis: {list(neighbourhoods)}")
        if len(neighbourhoods) > 0:
            for hood in neighbourhoods:
                hood_data = df[df['neighbourhood group'] == hood]['price'].dropna()
                if len(hood_data) > 0:
                    print(f"  {hood}: ${hood_data.mean():.2f} (n={len(hood_data)})")
    
    # Verificar room types
    if 'room type' in df.columns:
        room_types = df['room type'].dropna().unique()
        print(f"Tipos de quarto disponíveis: {list(room_types)}")
        if len(room_types) > 0:
            for room in room_types:
                room_data = df[df['room type'] == room]['price'].dropna()
                if len(room_data) > 0:
                    print(f"  {room}: ${room_data.mean():.2f} (n={len(room_data)})")
    
    print("=== FIM DEBUG ===\n")

def get_ml_options(df):
    """Get available options for ML features from the dataset"""
    options = {}
    
    if 'neighbourhood group' in df.columns:
        options['neighbourhoods'] = sorted(df['neighbourhood group'].dropna().unique().tolist())
    
    if 'room type' in df.columns:
        options['room_types'] = sorted(df['room type'].dropna().unique().tolist())
    
    # Get numeric ranges for other features
    numeric_features = ['minimum nights', 'availability 365', 'number of reviews', 
                       'review rate number', 'reviews per month', 'calculated host listings count',
                       'Construction year']
    
    for feature in numeric_features:
        if feature in df.columns:
            feature_data = df[feature].dropna()
            if len(feature_data) > 0:
                options[feature] = {
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'mean': float(feature_data.mean())
                }
    
    return options

@app.route('/upload', methods=['POST'])
def upload():
    if 'csv_file' not in request.files:
        flash('Nenhum arquivo selecionado', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['csv_file']
    if file.filename == '':
        flash('Nenhum arquivo selecionado', 'danger')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        filename = f"{uuid.uuid4().hex}.csv"
        filepath = os.path.join(UPLOADS, filename)
        file.save(filepath)
        
        session['df_path'] = filepath
        session.pop('neighbourhood_filter', None)
        session.pop('room_type_filter', None)
        session.pop('price_model', None)
        
        flash('Dataset carregado com sucesso!', 'success')
    else:
        flash('Por favor, envie um arquivo CSV', 'danger')
    
    return redirect(url_for('index'))

@app.route('/train_model', methods=['POST'])
def train_model_route():
    if 'df_path' not in session:
        flash('Nenhum dataset carregado', 'danger')
        return redirect(url_for('index'))
    
    df = pd.read_csv(session['df_path'])
    
    # Apply cleaning
    if 'price' in df.columns:
        df['price'] = clean_price_column(df['price'])
    
    features = request.form.getlist('features')
    model_choice = request.form.get('model_choice', 'RandomForest')
    test_size = float(request.form.get('test_size', 0.2))
    
    if not features or 'price' not in df.columns:
        flash('Selecione features válidas', 'danger')
        return redirect(url_for('index'))
    
    # Prepare data
    ml_df = df[features + ['price']].dropna()
    
    if len(ml_df) < 10:
        flash('Dados insuficientes para treinamento após limpeza', 'danger')
        return redirect(url_for('index'))
    
    # Prepare features with proper encoding
    X_processed = pd.DataFrame()
    label_encoders = {}
    
    for feature in features:
        if feature in ['neighbourhood group', 'room type']:
            # Label encoding for categorical variables
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(ml_df[feature].astype(str))
            label_encoders[feature] = le
            print(f"Encoded {feature}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            # Keep numeric variables as is
            X_processed[feature] = ml_df[feature]
    
    y = ml_df['price']
    
    print(f"Features após processamento: {X_processed.columns.tolist()}")
    print(f"Shape X: {X_processed.shape}, Shape y: {y.shape}")
    
    # Train model
    try:
        model = get_model(model_choice, {}, task='regression')
        report = train_model(model, X_processed, y, task='regression', test_size=test_size)
        
        if report is not None:
            # Save model
            model_filename = f"price_model_{uuid.uuid4().hex}.joblib"
            model_path = os.path.join(MODELS, model_filename)
            joblib.dump({
                'model': report['model'],
                'label_encoders': label_encoders
            }, model_path)
            
            session['price_model'] = {
                'model_path': model_path,
                'features': features,
                'metrics': report['metrics'],
                'feature_importance': report.get('feature_importance', {}),
                'label_encoders': {k: {
                    'classes': v.classes_.tolist(),
                    'name': k
                } for k, v in label_encoders.items()}
            }
            
            flash('Modelo treinado com sucesso!', 'success')
        else:
            flash('Erro no treinamento do modelo - relatório vazio', 'danger')
            
    except Exception as e:
        flash(f'Erro no treinamento: {str(e)}', 'danger')
        print(f"Training error: {e}")
    
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'price_model' not in session:
        flash('Nenhum modelo treinado', 'danger')
        return redirect(url_for('index'))
    
    model_info = session['price_model']
    
    try:
        # Get input values
        input_vals = {}
        prediction_features = []
        
        for feature in model_info['features']:
            value = request.form.get(feature, '')
            input_vals[feature] = value
            
            if feature in model_info.get('label_encoders', {}):
                # For categorical features, encode the value
                le_info = model_info['label_encoders'][feature]
                if value in le_info['classes']:
                    encoded_value = le_info['classes'].index(value)
                else:
                    # If value not in training data, use the first class
                    encoded_value = 0
                prediction_features.append(encoded_value)
                print(f"Encoded {feature} '{value}' -> {encoded_value}")
            else:
                # For numeric features
                try:
                    num_value = float(value)
                    prediction_features.append(num_value)
                except ValueError:
                    prediction_features.append(0.0)
        
        # Load model and predict
        saved_data = joblib.load(model_info['model_path'])
        model = saved_data['model']
        
        x_arr = np.array([prediction_features]).reshape(1, -1)
        prediction = model.predict(x_arr)[0]
        
        # Store prediction in session for display
        session['last_prediction'] = {
            'price': float(prediction),
            'inputs': input_vals
        }
        
        flash(f'Preço previsto: ${prediction:.2f}', 'success')
        
    except Exception as e:
        flash(f'Erro na predição: {str(e)}', 'danger')
        print(f"Prediction error: {e}")
    
    return redirect(url_for('index'))

def create_plots(df):
    plots = {}
    
    print("\n=== DEBUG DOS GRÁFICOS ===")
    
    # 1. CORRIGIDO: Gráfico de Preço Médio por Bairro (usando neighbourhood group)
    if 'neighbourhood group' in df.columns and 'price' in df.columns:
        df_clean = df.dropna(subset=['neighbourhood group', 'price'])
        df_clean = df_clean[df_clean['neighbourhood group'] != 'nan']
        df_clean = df_clean[df_clean['neighbourhood group'] != 'Nan']
        
        if len(df_clean) > 0:
            # Calcular preço médio por bairro
            avg_prices = df_clean.groupby('neighbourhood group')['price'].mean().sort_values(ascending=False)
            
            print(f"Preços por bairro encontrados: {avg_prices.to_dict()}")
            
            # Criar DataFrame para o gráfico
            plot_data = pd.DataFrame({
                'Bairro': avg_prices.index,
                'Preço Médio': avg_prices.values
            })
            
            fig = px.bar(
                plot_data,
                x='Bairro',
                y='Preço Médio',
                title="Preço Médio por Bairro",
                labels={'Preço Médio': 'Preço Médio ($)', 'Bairro': ''},
                color='Preço Médio',
                color_continuous_scale='viridis'
            )
            
            # Formatando os valores no hover e definindo texto explícito (evita placeholders quebrados)
            formatted_text = plot_data['Preço Médio'].apply(lambda v: f"${v:.2f}")
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Preço Médio: %{text}<extra></extra>',
                text=formatted_text,
                textposition='outside'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False,
                yaxis_title="Preço Médio ($)",
                xaxis_title=""
            )
            plots['price_by_neighbourhood'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            print(f"Gráfico 1 criado: Preço por Bairro ({len(avg_prices)} bairros)")
    
    # 2. CORRIGIDO: Gráfico Preço Médio por Tipo de Quarto
    if 'room type' in df.columns and 'price' in df.columns:
        df_clean = df.dropna(subset=['room type', 'price'])
        
        if len(df_clean) > 0:
            # Calcular preço médio por tipo de quarto
            room_prices = df_clean.groupby('room type')['price'].mean().sort_values(ascending=False)
            
            print(f"Preços por tipo de quarto encontrados: {room_prices.to_dict()}")
            
            # Criar DataFrame para o gráfico
            plot_data = pd.DataFrame({
                'Tipo de Quarto': room_prices.index,
                'Preço Médio': room_prices.values
            })
            
            fig = px.bar(
                plot_data,
                x='Tipo de Quarto',
                y='Preço Médio',
                title="Preço Médio por Tipo de Quarto",
                labels={'Preço Médio': 'Preço Médio ($)', 'Tipo de Quarto': ''},
                color='Preço Médio',
                color_continuous_scale='blues'
            )
            
            # Formatando os valores no hover e nas barras (usar texto explícito para evitar problemas de formatação)
            formatted_text = plot_data['Preço Médio'].apply(lambda v: f"${v:.2f}")
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Preço Médio: %{text}<extra></extra>',
                text=formatted_text,
                textposition='outside'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False,
                yaxis_title="Preço Médio ($)",
                xaxis_title=""
            )
            
            plots['price_by_room_type'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            print(f"Gráfico 2 criado: Preço por Tipo de Quarto ({len(room_prices)} tipos)")
    
    # 3. CORRIGIDO: Gráfico Disponibilidade vs Preço
    if all(col in df.columns for col in ['availability 365', 'price']):
        df_clean = df.dropna(subset=['availability 365', 'price'])
        
        # Filtrar para dados realistas - NÃO limitar preços altos
        df_clean = df_clean[
            (df_clean['price'] > 0) & 
            (df_clean['availability 365'] >= 0) & 
            (df_clean['availability 365'] <= 365)
        ]
        
        if len(df_clean) > 10:
            # Amostrar para melhor performance (máximo 1000 pontos)
            sample_size = min(1000, len(df_clean))
            sample_df = df_clean.sample(sample_size, random_state=42)
            
            # Calcular correlação para o título
            correlation = sample_df['price'].corr(sample_df['availability 365'])
            corr_text = f" (Correlação: {correlation:.2f})"
            
            fig = px.scatter(
                sample_df,
                x='availability 365',
                y='price',
                title=f"Relação: Disponibilidade vs Preço{corr_text}",
                labels={
                    'availability 365': 'Dias Disponíveis no Ano',
                    'price': 'Preço ($)'
                },
                opacity=0.6,
                color='price',
                color_continuous_scale='reds'
            )
            
            # Melhorar o hover
            fig.update_traces(
                hovertemplate=(
                    '<b>Disponibilidade:</b> %{x} dias<br>' +
                    '<b>Preço:</b> $%{y:.2f}<br>' +
                    '<extra></extra>'
                )
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Dias Disponíveis no Ano",
                yaxis_title="Preço ($)",
                showlegend=False
            )
            
            plots['availability_vs_price'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            print(f"Gráfico 3 criado: Disponibilidade vs Preço ({len(sample_df)} pontos)")
        else:
            print("Gráfico 3: Dados insuficientes para Disponibilidade vs Preço")
    
    # 4. CORRIGIDO: Gráfico de Top 10 Bairros Mais Caros (usando neighbourhood específico)
    if 'neighbourhood' in df.columns and 'price' in df.columns:
        df_clean = df.dropna(subset=['neighbourhood', 'price'])
        
        if len(df_clean) > 0:
            # Calcular preço médio por bairro específico
            neighbourhood_prices = df_clean.groupby('neighbourhood')['price'].mean()
            
            # Top 10 mais caros
            top_expensive = neighbourhood_prices.nlargest(10).sort_values(ascending=True)  # Ordenar para gráfico
            
            if len(top_expensive) > 0:
                plot_data = pd.DataFrame({
                    'Bairro': top_expensive.index,
                    'Preço Médio': top_expensive.values
                })
                
                fig = px.bar(
                    plot_data,
                    y='Bairro',
                    x='Preço Médio',
                    title="Top 10 Bairros Mais Caros",
                    labels={'Preço Médio': 'Preço Médio ($)', 'Bairro': ''},
                    color='Preço Médio',
                    color_continuous_scale='reds',
                    orientation='h'
                )
                
                # Usar texto pré-formatado para os rótulos das barras horizontais
                formatted_text = plot_data['Preço Médio'].apply(lambda v: f"${v:.2f}")
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Preço Médio: %{text}<extra></extra>',
                    text=formatted_text,
                    textposition='outside'
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    xaxis_title="Preço Médio ($)",
                    yaxis_title=""
                )
                
                plots['top_expensive_neighbourhoods'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                print(f"Gráfico 4 criado: Top 10 Bairros Mais Caros ({len(top_expensive)} bairros)")
    
    # 5. NOVO: Gráfico de Top 10 Bairros Mais Baratos
    if 'neighbourhood' in df.columns and 'price' in df.columns:
        df_clean = df.dropna(subset=['neighbourhood', 'price'])
        
        if len(df_clean) > 0:
            # Calcular preço médio por bairro específico
            neighbourhood_prices = df_clean.groupby('neighbourhood')['price'].mean()
            
            # Top 10 mais baratos (excluindo preços muito baixos irreais)
            realistic_prices = neighbourhood_prices[neighbourhood_prices > 50]  # Filtra preços irreais
            top_cheap = realistic_prices.nsmallest(10).sort_values(ascending=False)  # Ordenar para gráfico
            
            if len(top_cheap) > 0:
                plot_data = pd.DataFrame({
                    'Bairro': top_cheap.index,
                    'Preço Médio': top_cheap.values
                })
                
                fig = px.bar(
                    plot_data,
                    y='Bairro',
                    x='Preço Médio',
                    title="Top 10 Bairros Mais Baratos",
                    labels={'Preço Médio': 'Preço Médio ($)', 'Bairro': ''},
                    color='Preço Médio',
                    color_continuous_scale='greens',
                    orientation='h'
                )
                
                # Usar texto pré-formatado para os rótulos das barras horizontais
                formatted_text = plot_data['Preço Médio'].apply(lambda v: f"${v:.2f}")
                fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Preço Médio: %{text}<extra></extra>',
                    text=formatted_text,
                    textposition='outside'
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    xaxis_title="Preço Médio ($)",
                    yaxis_title=""
                )
                
                plots['top_cheap_neighbourhoods'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                print(f"Gráfico 5 criado: Top 10 Bairros Mais Baratos ({len(top_cheap)} bairros)")
    
    print(f"=== GRÁFICOS CRIADOS: {len(plots)} ===\n")
    return plots

def generate_insights(df):
    insights = []
    
    if 'price' not in df.columns or len(df) == 0:
        insights.append("Dados insuficientes para gerar insights")
        return insights
    
    # Basic price insights
    price_clean = df['price'].dropna()
    if len(price_clean) > 0:
        avg_price = price_clean.mean()
        median_price = price_clean.median()
        insights.append(f"Preço médio: ${avg_price:.2f}")
        insights.append(f"Preço mediano: ${median_price:.2f}")
    
    # Room type insight
    if 'room type' in df.columns:
        room_data = df.dropna(subset=['room type', 'price'])
        if len(room_data) > 0:
            room_prices = room_data.groupby('room type')['price'].mean()
            if len(room_prices) > 1:
                most_expensive = room_prices.idxmax()
                least_expensive = room_prices.idxmin()
                insights.append(f"Tipo mais caro: {most_expensive} (${room_prices[most_expensive]:.2f})")
                insights.append(f"Tipo mais barato: {least_expensive} (${room_prices[least_expensive]:.2f})")
    
    # Neighbourhood insight
    if 'neighbourhood group' in df.columns:
        hood_data = df.dropna(subset=['neighbourhood group', 'price'])
        if len(hood_data) > 0:
            hood_prices = hood_data.groupby('neighbourhood group')['price'].mean()
            if len(hood_prices) > 1:
                most_expensive_hood = hood_prices.idxmax()
                least_expensive_hood = hood_prices.idxmin()
                insights.append(f"Bairro mais caro: {most_expensive_hood} (${hood_prices[most_expensive_hood]:.2f})")
                insights.append(f"Bairro mais barato: {least_expensive_hood} (${hood_prices[least_expensive_hood]:.2f})")
    
    return insights

def get_filter_options(session):
    options = {'neighbourhoods': ['Todos'], 'room_types': ['Todos']}
    
    if 'df_path' in session and os.path.exists(session['df_path']):
        df = pd.read_csv(session['df_path'])
        
        # Normalize neighbourhood names for filter options too
        if 'neighbourhood group' in df.columns:
            df = normalize_neighbourhood_names(df)
            neighbourhoods = df['neighbourhood group'].dropna().unique().tolist()
            # Remove duplicates and sort
            neighbourhoods = sorted(list(set(neighbourhoods)))
            options['neighbourhoods'] += neighbourhoods
        
        if 'room type' in df.columns:
            room_types = df['room type'].dropna().unique().tolist()
            room_types = sorted(list(set(room_types)))
            options['room_types'] += room_types
    
    return options

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_DEBUG', '1') not in ('0', 'false', 'False')
    app.run(debug=debug, host=host, port=port)