import pandas as pd
import numpy as np
import plotly.express as px


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary with dtypes, missing, unique counts and basic stats."""
    summary = []
    for c in df.columns:
        col = df[c]
        # sample_unique may contain mixed types or lists; convert to a single string so
        # the resulting DataFrame is arrow/pyarrow-serializable for Streamlit display.
        sample_vals = list(col.dropna().unique()[:5])
        try:
            # join if elements are list-like
            sample_strs = [', '.join(map(str, v)) if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)) else str(v) for v in sample_vals]
        except Exception:
            sample_strs = [str(v) for v in sample_vals]

        summary.append({
            'column': c,
            'dtype': str(col.dtype),
            'n_missing': int(col.isna().sum()),
            'n_unique': int(col.nunique(dropna=True)),
            'sample_unique': '; '.join(sample_strs)
        })
    return pd.DataFrame(summary)


def plot_basic(df: pd.DataFrame, x: str, y: str = None, kind: str = 'hist'):
    if kind == 'hist':
        return px.histogram(df, x=x)
    if kind == 'bar':
        return px.bar(df, x=x, y=y)
    if kind == 'box':
        return px.box(df, x=x, y=y)
    if kind == 'scatter':
        return px.scatter(df, x=x, y=y)
    if kind == 'pie':
        return px.pie(df, names=x)


def plot_scatter_matrix(df: pd.DataFrame, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()[:6]
    return px.scatter_matrix(df[columns])
