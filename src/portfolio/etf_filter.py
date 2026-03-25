import pandas as pd
import numpy as np

from src.data.loaders import load_allocation_returns
from src.paths import PROCESSED_DIR


DEFENSIVE_ASSET = "SHY"


def build_etf_trend_filter():
    df = load_allocation_returns().copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    asset_cols = [c for c in df.columns if c not in ["date", "active_sleeve"]]

    long_df = df.melt(
        id_vars="date",
        value_vars=asset_cols,
        var_name="ticker",
        value_name="ret_1m"
    ).sort_values(["ticker", "date"]).copy()

    long_df["equity"] = long_df.groupby("ticker")["ret_1m"].transform(
        lambda x: (1 + x).cumprod()
    )

    long_df["mom_12m"] = long_df.groupby("ticker")["equity"].transform(
        lambda x: x.pct_change(11).shift(1)
    )

    long_df["ma_10"] = long_df.groupby("ticker")["equity"].transform(
        lambda x: x.rolling(10).mean()
    )

    long_df["above_ma"] = long_df["equity"] > long_df["ma_10"]

    long_df["trend_ok"] = (
        (long_df["mom_12m"] > 0) &
        (long_df["above_ma"])
    )

    return long_df

def build_etf_eligibility(save_output=True):
    etf_df = build_etf_trend_filter().copy()

    eligible = etf_df[["date", "ticker", "trend_ok"]].copy()

    if save_output:
        eligible.to_parquet(
            PROCESSED_DIR / "etf_eligibility.parquet",
            index=False
        )

    return eligible