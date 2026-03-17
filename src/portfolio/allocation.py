import pandas as pd
import numpy as np

from src.data.loaders import load_allocation_returns
from src.paths import PROCESSED_DIR

DEFAULT_WEIGHTS = {
    "active_sleeve": 0.35,
    "EFA": 0.15,
    "EEM": 0.10,
    "TLT": 0.15,
    "GLD": 0.10,
    "DBC": 0.10,
    "VNQ": 0.05,
}

def compute_portfolio_metrics(portfolio_df):
    df = portfolio_df.copy()

    monthly_ret = df["portfolio_ret"]

    n_months = len(df)
    annual_return = (1 + monthly_ret).prod() ** (12 / n_months) - 1
    annual_volatility = monthly_ret.std() * np.sqrt(12)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan

    running_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] / running_max - 1
    max_drawdown = drawdown.min()

    metrics = pd.Series({
        "n_months": n_months,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    })

    return metrics

def run_allocation_backtest(weights=None, save_output=True):
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    allocation_df = load_allocation_returns().copy()
    allocation_df["date"] = pd.to_datetime(allocation_df["date"])
    allocation_df = allocation_df.sort_values("date").copy()

    asset_cols = list(weights.keys())

    missing_cols = [col for col in asset_cols if col not in allocation_df.columns]
    if missing_cols:
        raise ValueError(f"Missing allocation columns: {missing_cols}")

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Weights must sum to 1. Got {weight_sum:.6f}")

    df = allocation_df[["date"] + asset_cols].copy()
    df = df.dropna(subset=asset_cols).copy()

    weight_vector = np.array([weights[col] for col in asset_cols])

    df["portfolio_ret"] = np.dot(df[asset_cols].values,weight_vector)
    df["equity_curve"] = (1 + df["portfolio_ret"]).cumprod()

    for col in asset_cols:
        df[f"contrib_{col}"] = df[col] * weights[col]

    metrics = compute_portfolio_metrics(df)

    if save_output:
        df.to_parquet(PROCESSED_DIR / "allocation_backtest.parquet", index=False)
        metrics.to_frame().to_parquet(PROCESSED_DIR / 'allocation_metrics.parquet', index=False)

    return df, metrics