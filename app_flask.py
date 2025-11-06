import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import plotly.express as px
import plotly.io as pio
from model import get_model, train_model
from utils import summarize_dataframe
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE_DIR, 'uploads')
MODELS = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Simple in-process storage for current dataframe/report/model
STATE = {
    'df_path': None,
    'report': None,
    'model_file': None
}


@app.route('/', methods=['GET'])
def index():
    df = None
    summary_html = None
    preview_html = None
    plot_div = None
    numeric_cols = []
    model_info = None
    if STATE['df_path'] and os.path.exists(STATE['df_path']):
        df = pd.read_csv(STATE['df_path'])
        preview_html = df.head().to_html(classes='table table-sm')
        summary_df = summarize_dataframe(df)
        summary_html = summary_df.to_html(classes='table table-sm', index=False)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # small example plot
        if numeric_cols:
            fig = px.histogram(df, x=numeric_cols[0])
            plot_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    if STATE['report']:
        model_info = STATE['report']
    return render_template('index.html', preview_html=preview_html, summary_html=summary_html, plot_div=plot_div, numeric_cols=numeric_cols, model_info=model_info)


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('csv_file')
    if not f:
        flash('Nenhum arquivo enviado', 'warning')
        return redirect(url_for('index'))
    filename = f"{uuid.uuid4().hex}.csv"
    path = os.path.join(UPLOADS, filename)
    f.save(path)
    STATE['df_path'] = path
    flash('Arquivo carregado com sucesso', 'success')
    return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def train():
    if not STATE['df_path']:
        flash('Nenhum dataset carregado', 'warning')
        return redirect(url_for('index'))
    df = pd.read_csv(STATE['df_path'])
    target = request.form.get('target')
    model_choice = request.form.get('model_choice')
    test_size = float(request.form.get('test_size', 0.2))
    # collect params
    params = {}
    if model_choice == 'RandomForest':
        params['n_estimators'] = int(request.form.get('n_estimators', 100))
        params['max_depth'] = int(request.form.get('max_depth', 0))
    if model_choice == 'KNN':
        params['n_neighbors'] = int(request.form.get('n_neighbors', 5))
    if model_choice == 'LogisticRegression':
        params['max_iter'] = int(request.form.get('max_iter', 100))

    if not target or target not in df.columns:
        flash('Target inválido', 'danger')
        return redirect(url_for('index'))

    X = df.drop(columns=[target])
    y = df[target]

    model = get_model(model_choice, params=params, task='auto')
    report = train_model(model, X, y, test_size=test_size)
    # Save model
    model_file = os.path.join(MODELS, 'last_model.joblib')
    joblib.dump(report['model'], model_file)
    STATE['report'] = report
    STATE['model_file'] = model_file
    flash('Treinamento concluído', 'success')
    return redirect(url_for('index'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not STATE['report']:
        flash('Nenhum modelo treinado nesta sessão', 'warning')
        return redirect(url_for('index'))
    report = STATE['report']
    features = report.get('features', [])
    prediction = None
    if request.method == 'POST':
        vals = []
        for f in features:
            v = request.form.get(f, '')
            try:
                vals.append(float(v))
            except Exception:
                vals.append(v)
        import numpy as _np
        X_in = _np.array([vals], dtype=object)
        model = joblib.load(STATE['model_file'])
        pred = model.predict(X_in)
        prediction = pred.tolist()
    return render_template('predict.html', features=features, prediction=prediction)


if __name__ == '__main__':
    # Allow configuring host/port via environment variables to avoid collisions
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_DEBUG', '1') not in ('0', 'false', 'False')
    app.run(debug=debug, host=host, port=port)
