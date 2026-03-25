import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.data.loaders import load_allocation_returns
from src.paths import PROCESSED_DIR

LOOKBACK_MONTHS = 24
TILT_STRENGTH = 0.30
MAX_WEIGHT = 0.40


def portfolio_vol(weights, cov):
    return np.sqrt(weights @ cov @ weights)


def risk_contributions(weights, cov):
    port_vol = portfolio_vol(weights, cov)
    if port_vol == 0:
        return np.zeros_like(weights)

    marginal_contrib = cov @ weights / port_vol
    total_contrib = weights * marginal_contrib
    return total_contrib


def compute_risk_parity_weights(cov):
    n = cov.shape[0]
    init = np.ones(n) / n
    target_rc = np.ones(n) / n

    def objective(weights):
        rc = risk_contributions(weights, cov)
        rc_pct = rc / rc.sum() if rc.sum() > 0 else np.zeros_like(rc)
        return np.sum((rc_pct - target_rc) ** 2)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(0.0, MAX_WEIGHT) for _ in range(n)]

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        return init

    weights = result.x
    weights = np.clip(weights, 0, MAX_WEIGHT)
    weights = weights / weights.sum()
    return weights


def compute_return_tilt(returns):
    mean_ret = returns.mean()
    positive = mean_ret - mean_ret.min()

    if positive.sum() == 0:
        return np.ones(len(mean_ret)) / len(mean_ret)

    tilt = positive / positive.sum()
    return tilt.values


def apply_constraints(weights):
    weights = np.clip(weights, 0, MAX_WEIGHT)
    weights = weights / weights.sum()
    return weights


def run_risk_parity_allocation(save_output=True):
    df = load_allocation_returns().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    df = df.drop(columns=["VTI"], errors="ignore")

    asset_cols = [c for c in df.columns if c != "date"]
    df = df.dropna(subset=asset_cols).copy()

    weights_list = []
    returns_list = []

    for i in range(LOOKBACK_MONTHS, len(df) - 1):
        window = df.iloc[i - LOOKBACK_MONTHS:i]
        next_row = df.iloc[i + 1]

        returns = window[asset_cols]
        cov = returns.cov().values

        rp_weights = compute_risk_parity_weights(cov)
        tilt_weights = compute_return_tilt(returns)

        final_weights = (
            (1 - TILT_STRENGTH) * rp_weights
            + TILT_STRENGTH * tilt_weights
        )
        final_weights = apply_constraints(final_weights)

        portfolio_ret = np.dot(next_row[asset_cols].values, final_weights)

        weights_list.append([next_row["date"]] + list(final_weights))
        returns_list.append([next_row["date"], portfolio_ret])

    weights_df = pd.DataFrame(weights_list, columns=["date"] + asset_cols)
    returns_df = pd.DataFrame(returns_list, columns=["date", "portfolio_ret"])
    returns_df["equity_curve"] = (1 + returns_df["portfolio_ret"]).cumprod()

    if save_output:
        weights_df.to_parquet(
            PROCESSED_DIR / "allocation_weights.parquet",
            index=False
        )
        returns_df.to_parquet(
            PROCESSED_DIR / "allocation_backtest.parquet",
            index=False
        )

    return returns_df, weights_df