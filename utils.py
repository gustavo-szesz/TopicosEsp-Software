import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

def clean_price_column(price_series):
    """Convert price columns from string '$xxx ' to float"""
    def clean_price(value):
        if pd.isna(value):
            return np.nan
        try:
            # Normalize to string
            s = str(value).strip()

            # Remove common currency symbols and words
            for sym in ['$', '€', '£', 'USD', 'BRL']:
                s = s.replace(sym, '')

            # Remove spaces
            s = s.replace(' ', '')

            # Handle thousands and decimal separators:
            # - If both '.' and ',' present, assume ',' is thousands separator (e.g. '1,234.56') -> remove commas
            # - If only ',' present, assume comma is decimal separator (e.g. '3,50') -> replace with '.'
            if ',' in s and '.' in s:
                s = s.replace(',', '')
            elif ',' in s and '.' not in s:
                s = s.replace(',', '.')

            # Remove any remaining non-numeric characters except dot and minus
            import re
            s = re.sub(r"[^0-9\.\-]", '', s)

            if s == '':
                return np.nan

            return float(s)
        except (ValueError, TypeError):
            return np.nan
    
    return price_series.apply(clean_price)

def preprocess_dates(df):
    """Process date columns for temporal analysis"""
    df_processed = df.copy()
    
    # Process last review date - maneira mais segura
    if 'last review' in df_processed.columns:
        # Converter para datetime de forma segura
        # Let pandas infer format and coerce invalid values
        df_processed['last_review_date'] = pd.to_datetime(
            df_processed['last review'], 
            errors='coerce'
        )
        
        # Extrair ano e mês de forma segura
        df_processed['last_review_year'] = df_processed['last_review_date'].dt.year
        df_processed['last_review_month'] = df_processed['last_review_date'].dt.month
        
        # Remover anos inválidos
        df_processed = df_processed[
            (df_processed['last_review_year'] >= 2000) & 
            (df_processed['last_review_year'] <= 2024)
        ]
    
    return df_processed

def normalize_neighbourhood_names(df):
    """Normalize neighbourhood names to handle case variations and typos"""
    df_processed = df.copy()
    
    if 'neighbourhood group' in df_processed.columns:
        # Converter para string e tratar NaN
        df_processed['neighbourhood group'] = df_processed['neighbourhood group'].fillna('Unknown')
        df_processed['neighbourhood group'] = df_processed['neighbourhood group'].astype(str)
        
        # Mapeamento mais abrangente
        neighbourhood_mapping = {
            'brookln': 'Brooklyn',
            'manhatan': 'Manhattan', 
            'manhattan': 'Manhattan',
            'brooklyn': 'Brooklyn',
            'queens': 'Queens',
            'bronx': 'Bronx',
            'staten island': 'Staten Island',
            'staten': 'Staten Island',
            'unknown': 'Unknown',
            'nan': 'Unknown',
            'none': 'Unknown',
            'null': 'Unknown',
            '': 'Unknown'
        }
        
        # Aplicar mapeamento - primeiro lowercase para consistência
        df_processed['neighbourhood group'] = df_processed['neighbourhood group'].str.lower().str.strip()
        df_processed['neighbourhood group'] = df_processed['neighbourhood group'].map(
            lambda x: neighbourhood_mapping.get(x, x.title())
        )
        
        # Remover 'Unknown' do dataset principal para análise
        df_processed = df_processed[df_processed['neighbourhood group'] != 'Unknown']
    
    return df_processed

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for c in df.columns:
        col = df[c]
        sample_vals = list(col.dropna().unique()[:3])
        sample_strs = [str(v) for v in sample_vals]

        summary.append({
            'column': c,
            'dtype': str(col.dtype),
            'n_missing': int(col.isna().sum()),
            'n_unique': int(col.nunique(dropna=True)),
            'sample_unique': '; '.join(sample_strs)
        })
    return pd.DataFrame(summary)