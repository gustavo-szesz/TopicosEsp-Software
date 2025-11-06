from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np


def get_model(name: str, params: dict = None, task: str = 'auto'):
    """Return an untrained sklearn model instance based on name and params."""
    params = params or {}
    if name == 'LinearRegression':
        return LinearRegression(**{k: v for k, v in params.items() if v is not None})
    if name == 'RandomForest':
        # choose regressor or classifier by task hint
        if task == 'classification':
            return RandomForestClassifier(n_estimators=int(params.get('n_estimators', 100)),
                                          max_depth=None if int(params.get('max_depth', 0)) == 0 else int(params.get('max_depth')))
        return RandomForestRegressor(n_estimators=int(params.get('n_estimators', 100)),
                                     max_depth=None if int(params.get('max_depth', 0)) == 0 else int(params.get('max_depth')))
    if name == 'KNN':
        if task == 'classification':
            return KNeighborsClassifier(n_neighbors=int(params.get('n_neighbors', 5)))
        return KNeighborsRegressor(n_neighbors=int(params.get('n_neighbors', 5)))
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=int(params.get('max_iter', 100)))

    raise ValueError(f"Modelo desconhecido: {name}")


def train_model(model, X: pd.DataFrame, y: pd.Series, task: str = 'auto', test_size: float = 0.2):
    """Train the model and return a report dict with metrics and artifacts.

    This function does minimal preprocessing: drops rows with NA in X or y and attempts
    a simple label encoding for non-numeric y in classification.
    """
    report = {}
    # align X and y
    df = pd.concat([X, y], axis=1)
    df = df.dropna()
    y = df[y.name]
    X = df.drop(columns=[y.name])

    # infer task if needed
    if task == 'auto':
        if y.dtype.kind in 'biufc':
            task = 'regression'
        else:
            task = 'classification'

    # simple encoding for categorical features
    X_proc = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=42)

    # if classification and y non-numeric, encode
    y_train_enc = y_train
    y_test_enc = y_test
    if task == 'classification' and y.dtype.kind not in 'biufc':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        report['label_encoder'] = le

    # fit
    model.fit(X_train, y_train_enc)

    # predictions
    y_pred = model.predict(X_test)

    # metrics
    metrics = {}
    if task == 'regression':
        metrics['r2'] = float(r2_score(y_test, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    else:
        # classification
        metrics['accuracy'] = float(accuracy_score(y_test_enc, y_pred))
        try:
            metrics['precision'] = float(precision_score(y_test_enc, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test_enc, y_pred, average='weighted', zero_division=0))
        except Exception:
            pass
        report['confusion_matrix'] = confusion_matrix(y_test_enc, y_pred).tolist()

    report['metrics'] = metrics
    report['model'] = model
    report['features'] = X_proc.columns.tolist()
    report['task'] = task

    # store last model in a way caller can persist (e.g., session_state)
    return report


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, task: str = 'auto'):
    """Evaluate a trained model on X,y. Minimal wrapper returning common metrics."""
    X_proc = pd.get_dummies(X, drop_first=True)
    y_pred = model.predict(X_proc)
    out = {}
    if task == 'regression':
        out['r2'] = r2_score(y, y_pred)
        out['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
    else:
        out['accuracy'] = accuracy_score(y, y_pred)
        out['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
    return out
