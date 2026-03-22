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

def build_rebalanced_holdings(alpha_df, rebalance_freq="M"):
    df = alpha_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).copy()

    all_dates = sorted(df["date"].drop_duplicates())
    df["rebalance_period"] = df["date"].dt.to_period(rebalance_freq)

    rebalance_dates = (
        df.groupby("rebalance_period")["date"]
        .min()
        .sort_values()
        .tolist()
    )
    rebalance_dates = set(rebalance_dates)

    current_longs = set()
    current_shorts = set()

    holdings_rows = []

    for dt in all_dates:
        if dt in rebalance_dates:
            snap = df[df["date"] == dt].copy()

            current_longs = set(snap.loc[snap["long_selected"], "ticker"])
            current_shorts = set(snap.loc[snap["short_selected"], "ticker"])

        for ticker in current_longs:
            holdings_rows.append(
                {"date": dt, "ticker": ticker, "side": "long"}
            )

        for ticker in current_shorts:
            holdings_rows.append(
                {"date": dt, "ticker": ticker, "side": "short"}
            )
    holdings = pd.DataFrame(holdings_rows)

    if holdings.empty:
        return pd.DataFrame(columns=["date", "ticker", "side"])

    holdings = holdings.drop_duplicates(subset=["date", "ticker", "side"]).copy()
    holdings = holdings.sort_values(["date", "side", "ticker"]).copy()

    return holdings




def backtest_long_short_equal_weight(alpha_df,transaction_cost_bps=20, rebalance_freq='M'):
    alpha_df = alpha_df.copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns()
    holdings = build_rebalanced_holdings(alpha_df, rebalance_freq=rebalance_freq)

    portfolio = holdings.merge(
        monthly_returns,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one"
    )

    portfolio = portfolio.dropna(subset=["fwd_ret_1m"]).copy()

    counts = (
        portfolio.groupby(["date", "side"])["ticker"]
        .count()
        .unstack(fill_value=0)
        .rename(columns={"long": "n_longs", "short": "n_shorts"})
        .reset_index()
    )

    portfolio = portfolio.merge(
        counts,
        on="date",
        how="left",
        validate="many_to_one",
    )

    portfolio["weight"] = 0.0

    long_mask = portfolio["side"] == "long"
    short_mask = portfolio["side"] == "short"

    portfolio.loc[long_mask, "weight"] = (
        0.5 / portfolio.loc[long_mask, "n_longs"]
    )

    portfolio.loc[short_mask, "weight"] = (
        -0.5 / portfolio.loc[short_mask, "n_shorts"]
    )

    portfolio["weighted_ret"] = portfolio["weight"] * portfolio["fwd_ret_1m"]

    monthly_portfolio = (
        portfolio.groupby("date", as_index=False)
        .agg(
            gross_ret=("weighted_ret", "sum"),
            n_longs=("n_longs", "max"),
            n_shorts=("n_shorts", "max")
        )
        .sort_values("date")
        .copy()
    )

    holdings_by_date = (
        portfolio.groupby("date")
        .apply(lambda x: set(zip(x["ticker"], x["side"])))
        .sort_index()
    )

    turnover = []
    prev_holdings = None

    for dt, current_holdings in holdings_by_date.items():
        if prev_holdings is None:
            approx_turnover = 1.0
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
        validate="one_to_one",
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
                "avg_n_longs": np.nan,
                "avg_n_shorts": np.nan
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
        "avg_n_longs": backtest_df["n_longs"].mean(),
        "avg_n_shorts": backtest_df["n_shorts"].mean()
    }

    return pd.Series(metrics)

def run_backtest(alpha_df, transaction_cost_bps=20, rebalance_freq='M', save_output=True, run_name='default'):
    
    backtest_df, holdings_df = backtest_long_short_equal_weight(
        alpha_df=alpha_df,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq=rebalance_freq,
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

    

