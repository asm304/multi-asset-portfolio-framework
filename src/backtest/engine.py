import pandas as pd
import numpy as np

from src.data.loaders import load_stock_prices
from src.paths import PROCESSED_DIR
from src.config import ACTIVE_PORTFOLIO_SIZE

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

def select_with_buffer(snap, prev_holdings, portfolio_size=ACTIVE_PORTFOLIO_SIZE, buffer_size=100):
    snap = snap.sort_values("alpha", ascending=False).copy()
    ranked_names = snap["ticker"].tolist()

    rank_map = {ticker: rank + 1 for rank, ticker in enumerate(ranked_names)}

    keep = [
        ticker for ticker in prev_holdings
        if ticker in rank_map and rank_map[ticker] <= buffer_size
    ]

    selected = list(keep)

    for ticker in ranked_names:
        if ticker not in selected:
            selected.append(ticker)
        if len(selected) >= portfolio_size:
            break

    return selected[:portfolio_size]


def backtest_long_only_weighted(
    alpha_df,
    transaction_cost_bps=10,
    rebalance_freq="Q",
    use_buffer=True,
    buffer_size=100
):
    df = alpha_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).copy()

    monthly_returns = build_monthly_returns().copy()
    monthly_returns["date"] = pd.to_datetime(monthly_returns["date"])

    df = df.merge(
        monthly_returns,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    df = df.dropna(subset=["alpha", "fwd_ret_1m"]).copy()

    df["rebalance_period"] = df["date"].dt.to_period(rebalance_freq)

    all_dates = sorted(df["date"].drop_duplicates())
    rebalance_dates = (
        df.groupby("rebalance_period")["date"]
        .min()
        .sort_values()
        .tolist()
    )

    rebalance_dates = set(rebalance_dates)

    prev_weights = {}
    tc = transaction_cost_bps / 10000.0

    monthly_rows = []
    holdings_rows = []

    for dt in all_dates:
        snap = df[df["date"] == dt].copy()

        if dt in rebalance_dates:
            if use_buffer:
                selected_names = select_with_buffer(
                    snap=snap,
                    prev_holdings=set(prev_weights.keys()),
                    portfolio_size=ACTIVE_PORTFOLIO_SIZE,
                    buffer_size=buffer_size,
                )
                selected = (
                    snap[snap["ticker"].isin(selected_names)]
                    .sort_values("alpha", ascending=False)
                    .copy()
                )
            else:
                selected = (
                    snap.sort_values("alpha", ascending=False)
                    .head(ACTIVE_PORTFOLIO_SIZE)
                    .copy()
                )

            selected = selected.dropna(subset=["vol_12m"]).copy()
            selected["vol_12m"] = selected["vol_12m"].clip(lower=1e-6)
            
            n_holdings = len(selected)

            if n_holdings > 0:
                inv_vol = 1.0 / selected.set_index("ticker")["vol_12m"]
                weights = inv_vol / inv_vol.sum()
                new_weights = weights.to_dict()

            else:
                new_weights = {}

            all_names = set(prev_weights.keys()) | set(new_weights.keys())

            turnover = sum(
                abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                for t in all_names
            )

            cost = turnover * tc
            prev_weights = new_weights.copy()

        else:
            turnover = 0.0
            cost = 0.0

        snap["weight"] = snap["ticker"].map(prev_weights).fillna(0.0)
        snap["weighted_ret"] = snap["weight"] * snap["fwd_ret_1m"]

        gross_ret = snap["weighted_ret"].sum()
        net_ret = gross_ret - cost
        n_holdings = len(prev_weights)

        monthly_rows.append(
            {
                "date": dt,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "cost": cost,
                "net_ret": net_ret,
                "n_holdings": n_holdings,
            }
        )

        current_holdings = snap[snap["weight"] > 0].copy()
        if not current_holdings.empty:
            holdings_rows.append(
                current_holdings[["date", "ticker", "alpha", "vol_12m", "fwd_ret_1m", "weight"]]
            )

    backtest_df = pd.DataFrame(monthly_rows).sort_values("date").copy()
    backtest_df["equity_curve"] = (1 + backtest_df["net_ret"]).cumprod()

    if holdings_rows:
        holdings_df = (
            pd.concat(holdings_rows, ignore_index=True)
            .sort_values(["date", "ticker"])
            .copy()
        )
    else:
        holdings_df = pd.DataFrame(
            columns=["date", "ticker", "alpha", "vol_12m", "fwd_ret_1m", "weight"]
        )

    return backtest_df, holdings_df

"""def build_rebalanced_holdings(alpha_df, rebalance_freq="Q"):
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




def backtest_long_short_equal_weight(alpha_df,transaction_cost_bps=20, rebalance_freq='Q'):
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
        1.0 / portfolio.loc[long_mask, "n_longs"]
    )

    portfolio.loc[short_mask, "weight"] = (
        -0.0 / portfolio.loc[short_mask, "n_shorts"]
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
"""

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

def run_backtest(alpha_df, transaction_cost_bps=10, rebalance_freq='Q', save_output=True, run_name='default', use_buffer=True, buffer_size=100):
    
    backtest_df, holdings_df = backtest_long_only_weighted(
        alpha_df=alpha_df,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_freq=rebalance_freq,
        use_buffer=use_buffer,
        buffer_size=buffer_size,
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

    

