import pandas as pd
import numpy as np

from src.data.loaders import load_price_signals, load_ridge_signal, load_xgb_signal
from src.paths import PROCESSED_DIR
from src.config import ACTIVE_PORTFOLIO_SIZE

SIGNAL_DIRECTIONS = {
    "liquidity": 1,
    "mom_12_1": 1,
    "mom_6_1": 1,
    "rev_1m": 1,
    "vol_12m": -1,
    "beta_12m": -1,
    "ml_signal": 1,
}

def winsorize_signals(series,lower=.01,upper=.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo,hi)

def zscore_signals(series):
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std

def merge_ml_signal(
    signals: pd.DataFrame,
    model_name: str = "xgb",
) -> pd.DataFrame:
    signals = signals.copy()

    if model_name == "xgb":
        ml = load_xgb_signal().copy()
    elif model_name == "ridge":
        ml = load_ridge_signal().copy()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    ml["date"] = pd.to_datetime(ml["date"])

    keep_cols = ["date", "ticker", "ml_signal"]
    if "model_name" in ml.columns:
        ml = ml.loc[ml["model_name"] == model_name, keep_cols].copy()
    else:
        ml = ml[keep_cols].copy()

    ml = ml.drop_duplicates(subset=["date", "ticker"])

    signals = signals.merge(
        ml,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    return signals



def normalize_signals(df):
    df = df.copy()

    for col,direction in SIGNAL_DIRECTIONS.items():
        df[col] = df.groupby('date')[col].transform(winsorize_signals)  
        df[col + '_z'] = df.groupby('date')[col].transform(zscore_signals)  
        df[col + '_z'] = direction * df[col + '_z']
    return df


def compute_alpha_score(df):
    df = df.copy()

    z_cols = [f"{col}_z" for col in SIGNAL_DIRECTIONS.keys()]
    df["alpha"] = df[z_cols].mean(axis=1)


    return df

def alpha_rank_and_select(df):
    df = df.copy()

    df['rank'] = df.groupby('date')['alpha'].rank(
        ascending=False,
        method='first'
    )
    df['selected'] = df['rank'] <= ACTIVE_PORTFOLIO_SIZE

    return df

def build_alpha_model(model_name: str = "xgb"):
    signals = load_price_signals().copy()

    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.sort_values(["date", "ticker"]).copy()

    signals = merge_ml_signal(signals, model_name=model_name)
    signals = normalize_signals(signals)
    signals = compute_alpha_score(signals)
    signals = alpha_rank_and_select(signals)

    path = PROCESSED_DIR / f'alpha_model_{model_name}.parquet'
    signals.to_parquet(path, index=False)

    return signals


