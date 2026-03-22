import pandas as pd
import numpy as np

from src.data.loaders import load_stock_prices,load_eligible_universe,load_etf_prices
from src.paths import PROCESSED_DIR

def build_price_signals():
    stock_prices = load_stock_prices().copy()
    eligible = load_eligible_universe().copy()
    etf_prices = load_etf_prices().copy()

    stock_prices["date"] = pd.to_datetime(stock_prices["date"])
    eligible["date"] = pd.to_datetime(eligible["date"])
    etf_prices["date"] = pd.to_datetime(etf_prices["date"])

    stock_prices = stock_prices.sort_values(["ticker", "date"]).copy()
    etf_prices = etf_prices.sort_values(["ticker", "date"]).copy()

    stock_prices["ret_1d"] = (
        stock_prices.groupby("ticker")["close"].pct_change()
    )
    stock_prices["month"] = stock_prices["date"].dt.to_period("M")
    monthly_stock = (
        stock_prices.groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly_stock = monthly_stock.sort_values(["ticker", "date"]).copy()
    monthly_stock["ret_1m"] = (
        monthly_stock.groupby("ticker")["close"].pct_change()
    )

    monthly_stock["mom_12_1"] = (
        monthly_stock.groupby("ticker")["close"]
        .pct_change(11)
        .groupby(monthly_stock['ticker'])
        .shift(1)
    )

    monthly_stock['mom_9_1'] = (
        monthly_stock.groupby('ticker')['close']
        .pct_change(8)
        .groupby(monthly_stock['ticker'])
        .shift(1)
    )

    monthly_stock['mom_6_1'] = (
        monthly_stock.groupby('ticker')['close']
        .pct_change(5)
        .groupby(monthly_stock['ticker'])
        .shift(1)
    )

    vti = etf_prices[etf_prices["ticker"] == "VTI"].copy()
    vti = vti.sort_values("date").copy()

    vti["month"] = vti["date"].dt.to_period("M")
    vti_monthly = (
        vti.groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    vti_monthly = vti_monthly.sort_values("date").copy()

    vti_monthly["mkt_ret_1m"] = vti_monthly['adj_close'].pct_change()

    vti_monthly["mkt_mom_12_1"] = vti_monthly['adj_close'].pct_change(11).shift(1)
    vti_monthly["mkt_mom_9_1"] = vti_monthly['adj_close'].pct_change(8).shift(1)
    vti_monthly["mkt_mom_6_1"] = vti_monthly['adj_close'].pct_change(5).shift(1)

    vti["mkt_ret_1d"] = vti['adj_close'].pct_change()

    beta_df = stock_prices.merge(
        vti[["date", "mkt_ret_1d"]],
        on="date",
        how="left"
    )

    def rolling_beta(group):
        cov = group["ret_1d"].rolling(756, min_periods=252).cov(group["mkt_ret_1d"])
        var = group["mkt_ret_1d"].rolling(756, min_periods=252).var()
        return cov / var

    beta_df["beta_36m"] = (
        beta_df.groupby("ticker")
        .apply(rolling_beta, include_groups=False)
        .reset_index(level=0, drop=True)
    )

    beta_df["beta_36m"] = beta_df["beta_36m"].clip(-5, 5)
    beta_df["month"] = beta_df["date"].dt.to_period("M")

    beta_monthly = (
        beta_df.groupby(["ticker", "month"], as_index=False)
        .tail(1)[["date", "ticker", "beta_36m"]]
        .copy()
    )

    daily_sign = np.sign(stock_prices["ret_1d"])

    stock_prices["pos_days_231"] = (
        (daily_sign > 0)
        .groupby(stock_prices["ticker"])
        .rolling(231)
        .sum()
        .reset_index(level=0, drop=True)
    )

    stock_prices["neg_days_231"] = (
        (daily_sign < 0)
        .groupby(stock_prices["ticker"])
        .rolling(231)
        .sum()
        .reset_index(level=0, drop=True)
    )

    stock_prices["pos_days_231"] = stock_prices.groupby("ticker")["pos_days_231"].shift(21)
    stock_prices["neg_days_231"] = stock_prices.groupby("ticker")["neg_days_231"].shift(21)

    fip_monthly = (
        stock_prices.groupby(["ticker", "month"], as_index=False)
        .tail(1)[["date", "ticker", "pos_days_231", "neg_days_231"]]
        .copy()
    )

    monthly_stock = monthly_stock.merge(
        beta_monthly,
        on=["date", "ticker"],
        how="left"
    )

    monthly_stock = monthly_stock.merge(
        vti_monthly[["date", "mkt_mom_12_1", "mkt_mom_9_1", "mkt_mom_6_1", "mkt_ret_1m"]],
        on="date",
        how="left"
    )

    monthly_stock = monthly_stock.merge(
        fip_monthly,
        on=["date", "ticker"],
        how="left"
    )

    monthly_stock["res_mom_12_1"] = (
        monthly_stock["mom_12_1"] - monthly_stock["beta_36m"] * monthly_stock["mkt_mom_12_1"]
    )

    monthly_stock["res_mom_9_1"] = (
        monthly_stock["mom_9_1"] - monthly_stock["beta_36m"] * monthly_stock["mkt_mom_9_1"]
    )

    monthly_stock["res_mom_6_1"] = (
        monthly_stock["mom_6_1"] - monthly_stock["beta_36m"] * monthly_stock["mkt_mom_6_1"]
    )

    denom = (monthly_stock["pos_days_231"] + monthly_stock["neg_days_231"]).replace(0, np.nan)

    monthly_stock["fip_quality"] = (
        np.sign(monthly_stock["mom_12_1"]) *
        (monthly_stock["pos_days_231"] - monthly_stock["neg_days_231"]) / denom
    )


    """
    monthly_stock["rev_1m"] = -monthly_stock["ret_1m"]

    stock_prices["vol_12m"] = (
        stock_prices.groupby("ticker")["ret_1d"]
        .transform(lambda x: x.rolling(252, min_periods=252).std())
    )
    vol_monthly = (
        stock_prices.groupby(["ticker", "month"], as_index=False)
        .tail(1)[["date", "ticker", "vol_12m"]]
        .copy()
    )

    vti = etf_prices[etf_prices["ticker"] == "VTI"].copy()
    vti = vti.sort_values("date").copy()
    vti["mkt_ret_1d"] = vti["adj_close"].pct_change()

    beta_df = stock_prices.merge(
        vti[["date", "mkt_ret_1d"]],
        on="date",
        how="left"
    )
    def rolling_beta(group):
        cov = group["ret_1d"].rolling(252, min_periods=252).cov(group["mkt_ret_1d"])
        var = group["mkt_ret_1d"].rolling(252, min_periods=252).var()
        return cov / var

    beta_df["beta_12m"] = (
        beta_df.groupby("ticker")
        .apply(rolling_beta, include_groups=False)
        .reset_index(level=0, drop=True)
    )
    beta_df["beta_12m"] = beta_df["beta_12m"].clip(-5, 5)

    beta_df["month"] = beta_df["date"].dt.to_period("M")
    beta_monthly = (
        beta_df.groupby(["ticker", "month"], as_index=False)
        .tail(1)[["date", "ticker", "beta_12m"]]
        .copy()
    )
    """
    signals = eligible[["date", "ticker"]].copy()

    signals = signals.merge(
        monthly_stock[
            [
                "date", "ticker",
                "res_mom_12_1", "res_mom_9_1", "res_mom_6_1",
                "mom_12_1", "mom_9_1", "mom_6_1",
                "beta_36m", "mkt_ret_1m", "fip_quality"
            ]
        ],
        on=["date", "ticker"],
        how="left"
    )

    """
    signals = signals.merge(
        vol_monthly,
        on=["date", "ticker"],
        how="left"
    )

    signals = signals.merge(
        beta_monthly,
        on=["date", "ticker"],
        how="left"
    )

    signals = signals.rename(columns={"adv_20": "liquidity"})
    """
    
    path = PROCESSED_DIR / "price_signals.parquet"
    signals.to_parquet(path, index=False)

    return signals