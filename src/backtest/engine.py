import pandas as pd
import numpy as np

from src.data.loaders import load_stock_prices
from src.paths import PROCESSED_DIR

def build_monthly_returns():
    stock_prices = load_stock_prices().copy()

    stock_prices["date"] = pd.to_datetime(stock_prices["date"])
    stock_prices = stock_prices.sort_values(["ticker", "date"]).copy()

    stock_prices["month"] = stock_prices["date"].dt.to_period("M")

    monthly = (
        stock_prices.groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly = monthly.sort_values(["ticker", "date"]).copy()

    monthly['ret_1m'] = monthly.groupby('ticker')['close'].pct_change()

    monthly["fwd_ret_1m"] = (
        monthly.groupby("ticker")["ret_1m"].shift(-1)
    )

    return monthly[["date", "ticker", "fwd_ret_1m"]].copy()

def backtest_top100_equal_weight(alpha_df,transaction_cost_bps=10):
    alpha_df = alpha_df.copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns()

    portfolio = alpha_df.merge(
        monthly_returns,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one"
    )

    portfolio = portfolio[portfolio["selected"]].copy()
    portfolio = portfolio.dropna(subset=["fwd_ret_1m"]).copy()

    portfolio["weight"] = (
        1.0 / portfolio.groupby("date")["ticker"].transform("count")
    )

    portfolio["weighted_ret"] = portfolio["weight"] * portfolio["fwd_ret_1m"]

    monthly_portfolio = (
        portfolio.groupby("date", as_index=False)
        .agg(
            gross_ret=("weighted_ret", "sum"),
            n_holdings=("ticker", "count")
        )
        .sort_values("date")
        .copy()
    )

    holdings_by_date = (
        portfolio.groupby("date")["ticker"]
        .apply(set)
        .sort_index()
    )

    turnover = []
    prev_holdings = None

    for dt, current_holdings in holdings_by_date.items():
        if prev_holdings is None:
            turnover.append((dt, 1.0))
        else:
            overlap = len(prev_holdings & current_holdings)
            n_prev = len(prev_holdings) if len(prev_holdings) > 0 else 1
            approx_turnover = 1 - (overlap / n_prev)
            turnover.append((dt, approx_turnover))
        prev_holdings = current_holdings
    
    turnover_df = pd.DataFrame(turnover, columns=["date", "turnover"])

    monthly_portfolio = monthly_portfolio.merge(
        turnover_df,
        on="date",
        how="left",
        validate="one_to_one"
    )

    tc = transaction_cost_bps / 10000.0

    monthly_portfolio["cost"] = monthly_portfolio["turnover"] * tc
    monthly_portfolio["net_ret"] = monthly_portfolio["gross_ret"] - monthly_portfolio["cost"]

    monthly_portfolio["equity_curve"] = (1 + monthly_portfolio["net_ret"]).cumprod()

    return monthly_portfolio, portfolio

def compute_performance_metrics(backtest_df):
    backtest_df = backtest_df.copy()

    rets = backtest_df["net_ret"].dropna()

    if len(rets) == 0:
        return pd.Series(
            {
                "n_months": 0,
                "annual_return": np.nan,
                "annual_volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "avg_turnover": np.nan,
                "avg_n_holdings": np.nan,
            }
        )

    equity = (1 + rets).cumprod()
    ann_return = equity.iloc[-1] ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)

    sharpe = np.nan
    if pd.notna(ann_vol) and ann_vol != 0:
        sharpe = ann_return / ann_vol


   
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    metrics = {
        "n_months": len(rets),
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": backtest_df["turnover"].mean(),
        "avg_n_holdings": backtest_df["n_holdings"].mean(),
    }

    return pd.Series(metrics)

def run_backtest(alpha_df, transaction_cost_bps=10, save_output=True, run_name='default'):
    backtest_df, holdings_df = backtest_top100_equal_weight(
        alpha_df=alpha_df,
        transaction_cost_bps=transaction_cost_bps
    )

    metrics = compute_performance_metrics(backtest_df)

    if save_output:
        backtest_df.to_parquet(
            PROCESSED_DIR / f"backtest_results_{run_name}.parquet",
            index=False,
        )
        holdings_df.to_parquet(
            PROCESSED_DIR / f"backtest_holdings_{run_name}.parquet",
            index=False,
        )
        metrics.to_frame(name="value").reset_index(names="metric").to_parquet(
            PROCESSED_DIR / f"backtest_metrics_{run_name}.parquet",
            index=False,
        )
    return backtest_df, holdings_df, metrics

    

