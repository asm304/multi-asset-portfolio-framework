import pandas as pd

from src.data.loaders import load_etf_prices, load_backtest_results
from src.paths import PROCESSED_DIR

def build_monthly_etf_returns():
    etf_prices = load_etf_prices().copy()

    etf_prices["date"] = pd.to_datetime(etf_prices["date"])
    etf_prices = etf_prices.sort_values(["ticker", "date"]).copy()

    etf_prices['month'] = etf_prices["date"].dt.to_period('M')

    monthly = (
        etf_prices.groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly = monthly.sort_values(['ticker', 'date']).copy()

    monthly['ret_1m'] = monthly.groupby('ticker')['adj_close'].pct_change()

    monthly_returns = monthly[['date','ticker','ret_1m']].copy()

    path = PROCESSED_DIR / 'monthly_etf_returns.parquet'
    monthly_returns.to_parquet(path, index=False)

    return monthly_returns

def build_allocation_return_table():
    etf_returns = build_monthly_etf_returns().copy()
    backtest_df = load_backtest_results().copy()

    etf_returns["date"] = pd.to_datetime(etf_returns["date"])
    backtest_df["date"] = pd.to_datetime(backtest_df["date"])

    etf_wide = etf_returns.pivot(
        index="date",
        columns="ticker",
        values="ret_1m"
    ).reset_index()

    etf_wide.columns.name = None

    active = backtest_df[["date", "net_ret"]].copy()
    active = active.rename(columns={"net_ret": "active_sleeve"})

    allocation_df = active.merge(
        etf_wide,
        on="date",
        how="left"
    ).sort_values("date").copy()

    allocation_df = allocation_df.drop(columns=["VTI"], errors="ignore")

    path = PROCESSED_DIR / "allocation_returns.parquet"
    allocation_df.to_parquet(path, index=False)

    return allocation_df

