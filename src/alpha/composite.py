import pandas as pd
import numpy as np

from src.data.loaders import load_final_signals, load_ridge_signal, load_xgb_signal, load_alpha_signals
from src.paths import PROCESSED_DIR
from src.config import ACTIVE_CANDIDATE_SIZE, ACTIVE_PORTFOLIO_SIZE

"""SIGNAL_DIRECTIONS = {
    "liquidity": 1,
    "mom_12_1": 1,
    "mom_6_1": 1,
    "rev_1m": 1,
    "vol_12m": -1,
    "beta_12m": -1,
    "ml_signal": 1,
}"""

FEATURES = [
    # price
    "mom_12_1",
    "mom_9_1",
    "mom_6_1",
    "ra_res_mom_12_1",
    "ra_res_mom_9_1",
    "ra_res_mom_6_1",
    "neg_vol_12m",
    "fip_quality",

    # quality/profitability
    "net_margin",
    "operating_margin",
    "roe",
    "roa",
    "accrual_quality",
    "operating_cf_to_assets",
    "fcf_to_assets",

    # leverage/safety
    "neg_net_debt_to_assets",
    "neg_net_debt_to_equity",

    # value
    "earnings_yield",
    "fcf_yield",
    "ebitda_ev",

    # capital return
    "shareholder_yield",

    # earnings/revisions
    "earnings_surprise",
    "eps_revision_7d_pct",
    "eps_revision_30d_pct",
    "eps_revision_60d_pct",
    "eps_revision_90d_pct",
    "eps_revision_7d_yield",
    "eps_revision_30d_yield",
    "eps_revision_60d_yield",
    "eps_revision_90d_yield",
    "revision_breadth_7d",
    "revision_breadth_30d",
    "revision_ratio_7d",
    "revision_ratio_30d",
]

def winsorize_signals(series,lower=.01,upper=.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo,hi)

def zscore_signals(series):
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std

"""def merge_ml_signal(
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

    return signals"""



def normalize_signals(df):
    df = df.copy()

    for col in FEATURES:
        df[col] = df.groupby('date')[col].transform(winsorize_signals)  
        df[col + '_z'] = df.groupby('date')[col].transform(zscore_signals)  
    
    return df


def compute_alpha_score(df):
    df = df.copy()
    
    alpha_signals = load_alpha_signals()

    missing = set(alpha_signals) - set(df.columns)
    if missing:
        raise ValueError(f"Missing alpha signal columns: {missing}")

    df["alpha"] = df[alpha_signals].mean(axis=1, skipna=True)
    df["alpha_n_signals"] = df[alpha_signals].notna().sum(axis=1)

    df.loc[df["alpha_n_signals"] < max(1, len(alpha_signals) // 2), "alpha"] = np.nan
    """required = ["ra_res_mom_12_1_z", "ra_res_mom_9_1_z", "ra_res_mom_6_1_z", "fip_quality_z"]

    df["alpha"] = np.where(
        df[required].notna().all(axis=1),
        (.6 / 3) * df["ra_res_mom_12_1_z"] +
        (.6 / 3) * df["ra_res_mom_9_1_z"] +
        (.6 / 3) * df["ra_res_mom_6_1_z"] +
        (0.4) * df["fip_quality_z"],
        np.nan
    )"""

    return df


def alpha_rank_and_select(df):
    df = df.copy()

    df["rank_long"] = df.groupby("date")["alpha"].rank(ascending=False, method="first")
    df["rank_short"] = df.groupby("date")["alpha"].rank(ascending=True, method="first")

    df["long_candidate"] = (df["rank_long"] <= ACTIVE_CANDIDATE_SIZE) & df["alpha"].notna() #& df["fip_quality"].notna()
    df["short_candidate"] = (df["rank_short"] <= ACTIVE_CANDIDATE_SIZE) & df["alpha"].notna() #& df["fip_quality"].notna()

    return df

def apply_fip_filter(df):
    df = df.copy()

    long_fip_rank = (
        df.loc[df["long_candidate"]]
        .groupby("date")["fip_quality"]
        .rank(ascending=False, method="first")
    )

    short_fip_rank = (
        df.loc[df["short_candidate"]]
        .groupby("date")["fip_quality"]
        .rank(ascending=False, method="first")
    )

    df["fip_rank_long"] = np.nan
    df["fip_rank_short"] = np.nan

    df.loc[df["long_candidate"], "fip_rank_long"] = long_fip_rank
    df.loc[df["short_candidate"], "fip_rank_short"] = short_fip_rank

    df["long_selected"] = df["long_candidate"] & (df["fip_rank_long"] <= ACTIVE_PORTFOLIO_SIZE)
    df["short_selected"] = df["short_candidate"] & (df["fip_rank_short"] <= ACTIVE_PORTFOLIO_SIZE)

    return df

def apply_alpha_selection(df):
    df = df.copy()

    df["long_selected"] = (
        df["long_candidate"] &
        (df["rank_long"] <= ACTIVE_PORTFOLIO_SIZE)
    )

    df["short_selected"] = (
        df["short_candidate"] &
        (df["rank_short"] <= ACTIVE_PORTFOLIO_SIZE)
    )

    return df

def build_alpha_model():
    signals = load_final_signals().copy()

    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.sort_values(["date", "ticker"]).copy()

   # signals = merge_ml_signal(signals, model_name=model_name)

    signals = normalize_signals(signals)
    signals = compute_alpha_score(signals)
    signals = alpha_rank_and_select(signals)
    #signals = apply_fip_filter(signals)
    signals = apply_alpha_selection(signals)

    path = PROCESSED_DIR / f'alpha_model.parquet'
    signals.to_parquet(path, index=False)

    return signals


