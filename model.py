from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def get_model(name: str, params: dict = None, task: str = 'auto'):
    params = params or {}
    if name == 'LinearRegression':
        return LinearRegression(**{k: v for k, v in params.items() if v is not None})
    if name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=int(params.get('n_estimators', 100)),
            max_depth=None if int(params.get('max_depth', 0)) == 0 else int(params.get('max_depth')),
            random_state=42,
            n_jobs=-1
        )
    if name == 'KNN':
        return KNeighborsRegressor(n_neighbors=int(params.get('n_neighbors', 5)))
    
    raise ValueError(f"Modelo desconhecido: {name}")

def train_model(model, X: pd.DataFrame, y: pd.Series, task: str = 'auto', test_size: float = 0.2):
    try:
        report = {}
        
        if X.empty or y.empty:
            return None
            
        # Combinar e limpar dados
        df_combined = pd.concat([X, y], axis=1)
        df_clean = df_combined.dropna()
        
        if df_clean.empty:
            return None
            
        y_clean = df_clean[y.name]
        X_clean = df_clean.drop(columns=[y.name])
        
        if X_clean.empty or len(X_clean) < 10:
            return None

        # Usar apenas colunas numéricas
        X_numeric = X_clean.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return None

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_clean, test_size=test_size, random_state=42
        )

        # Treinar modelo
        model.fit(X_train, y_train)

        # Fazer predições
        y_pred = model.predict(X_test)

        # Calcular métricas
        metrics = {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mean_actual_price': float(y_test.mean()),
            'mean_predicted_price': float(y_pred.mean())
        }

        report['metrics'] = metrics
        report['model'] = model
        report['features'] = X_numeric.columns.tolist()
        report['feature_importance'] = get_feature_importance(model, X_numeric.columns)

        return report
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        return None

def get_feature_importance(model, feature_names):
    """Get feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            importance_dict = dict(zip(feature_names, model.coef_))
            return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    except:
        return None
    return None