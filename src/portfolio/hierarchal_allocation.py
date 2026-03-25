import pandas as pd
import numpy as np

from src.data.loaders import load_allocation_returns
from src.portfolio.regime import build_active_weights
from src.portfolio.etf_filter import build_etf_eligibility
from src.paths import PROCESSED_DIR


DEFENSIVE_ASSET = "SHY"
LOOKBACK_VOL = 12
MAX_ETF_WEIGHT = 0.30


def compute_inverse_vol_weights(vols, max_weight=MAX_ETF_WEIGHT):
    inv_vol = 1.0 / vols.replace(0, np.nan)
    inv_vol = inv_vol.dropna()

    if len(inv_vol) == 0:
        return pd.Series(dtype=float)

    weights = inv_vol / inv_vol.sum()
    weights = weights.clip(upper=max_weight)

    if weights.sum() == 0:
        return pd.Series(dtype=float)

    weights = weights / weights.sum()
    return weights


def build_hierarchical_allocation(save_output=True):
    df = load_allocation_returns().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    active_weights = build_active_weights(save_output=False).copy()
    active_weights["date"] = pd.to_datetime(active_weights["date"])

    etf_elig = build_etf_eligibility(save_output=False).copy()
    etf_elig["date"] = pd.to_datetime(etf_elig["date"])

    asset_cols = [c for c in df.columns if c not in ["date", "active_sleeve"]]
    risk_assets = [c for c in asset_cols if c != DEFENSIVE_ASSET]

    vol_df = df[["date"] + risk_assets].copy()
    for col in risk_assets:
        vol_df[col] = vol_df[col].rolling(LOOKBACK_VOL).std() * np.sqrt(12)

    vol_long = vol_df.melt(
        id_vars="date",
        value_vars=risk_assets,
        var_name="ticker",
        value_name="vol_12m"
    )

    elig = etf_elig.merge(vol_long, on=["date", "ticker"], how="left")

    weights_rows = []

    for dt in df["date"]:
        active_w = active_weights.loc[
            active_weights["date"] == dt, "active_weight"
        ]
        active_w = active_w.iloc[0] if len(active_w) > 0 else 0.40

        etf_budget = 1.0 - active_w

        snap = elig[(elig["date"] == dt) & (elig["ticker"].isin(risk_assets))].copy()
        snap = snap[snap["trend_ok"]].copy()
        snap = snap.dropna(subset=["vol_12m"])

        etf_weights = {}

        if not snap.empty:
            vols = snap.set_index("ticker")["vol_12m"]
            w = compute_inverse_vol_weights(vols)

            if len(w) > 0:
                etf_weights = (w * etf_budget).to_dict()

        allocated_to_risk_etfs = sum(etf_weights.values())
        shy_weight = etf_budget - allocated_to_risk_etfs

        final_weights = {"date": dt, "active_sleeve": active_w, DEFENSIVE_ASSET: shy_weight}

        for asset in risk_assets:
            final_weights[asset] = etf_weights.get(asset, 0.0)
        
        weights_rows.append(final_weights)

    weights_df = pd.DataFrame(weights_rows).sort_values("date").copy()
    weight_cols = [c for c in weights_df.columns if c != "date"]

    shifted_weights = weights_df.copy()
    shifted_weights[weight_cols] = shifted_weights[weight_cols].shift(1)

    returns_df = df[["date", "active_sleeve"] + asset_cols].copy()
    returns_df = returns_df.merge(
        shifted_weights,
        on="date",
        how="left",
        suffixes=("_ret", "_w")
    )

    returns_df = returns_df.dropna(subset=["active_sleeve_w"]).copy()

    returns_df["portfolio_ret"] = 0.0

    returns_df["portfolio_ret"] += (
        returns_df["active_sleeve_w"].fillna(0.0) * returns_df["active_sleeve_ret"]
    )
    

    for asset in asset_cols:
        returns_df["portfolio_ret"] += (
            returns_df[f"{asset}_w"].fillna(0.0) * returns_df[f"{asset}_ret"]
        )

    returns_df["equity_curve"] = (1 + returns_df["portfolio_ret"]).cumprod()
    returns_df = returns_df[["date", "portfolio_ret", "equity_curve"]].copy()

    if save_output:
        weights_df.to_parquet(
            PROCESSED_DIR / "allocation_weights_hierarchical.parquet",
            index=False
        )
        returns_df.to_parquet(
            PROCESSED_DIR / "allocation_backtest_hierarchical.parquet",
            index=False
        )

    return returns_df, weights_df


def compute_allocation_metrics(backtest_df):
    rets = backtest_df["portfolio_ret"].dropna()

    equity = (1 + rets).cumprod()
    ann_return = equity.iloc[-1] ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return pd.Series({
        "n_months": len(rets),
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    })