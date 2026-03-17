import pandas as pd
import numpy as np

from src.data.loaders import load_price_signals
from src.backtest.engine import build_monthly_returns
from src.paths import PROCESSED_DIR

FEATURE_COLS = [
    "liquidity",
    "mom_12_1",
    "mom_6_1",
    "rev_1m",
    "vol_12m",
    "beta_12m",
]

TARGET_COL = "fwd_ret_1m"

def winsorize_signals(series,lower=.01,upper=.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo,hi)

def zscore_signals(series):
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std

def normalize_signals(df):
    df = df.copy()

    for col in FEATURE_COLS:
        df[col] = df.groupby('date')[col].transform(winsorize_signals)  
        df[col + '_z'] = df.groupby('date')[col].transform(zscore_signals)  
    return df

def build_ml_dataset(save_output=True):
    signals = load_price_signals().copy()
    monthly_returns = build_monthly_returns().copy()

    signals["date"] = pd.to_datetime(signals["date"])
    monthly_returns["date"] = pd.to_datetime(monthly_returns["date"])

    df = signals.merge(
        monthly_returns[['date','ticker','fwd_ret_1m']],
        on=['date','ticker'],
        how='left'
    )

    cols = ["date", "ticker"] + FEATURE_COLS + [TARGET_COL]
    df = df[cols].copy()

    df = df.sort_values(['date','ticker']).copy()

    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df.dropna(subset=FEATURE_COLS).copy()

    df = normalize_signals(df)
    df[TARGET_COL] = df.groupby('date')[TARGET_COL].transform(winsorize_signals)

    df["vol_12m_z"] = -df["vol_12m_z"]
    df["beta_12m_z"] = -df["beta_12m_z"]

    normalized_signals = [signal + '_z' for signal in FEATURE_COLS]
    cols = ['date', 'ticker'] + normalized_signals + [TARGET_COL]
    df = df[cols].copy()

    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df.dropna(subset=normalized_signals).copy()

    
    df = df.sort_values(['date','ticker']).copy()

    

    if save_output:
        path = PROCESSED_DIR / "ml_dataset.parquet"
        df.to_parquet(path, index=False)

    return df