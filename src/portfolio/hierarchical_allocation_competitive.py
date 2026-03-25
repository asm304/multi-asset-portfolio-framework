import pandas as pd
import numpy as np

from src.data.loaders import load_allocation_returns, load_risk_free
from src.portfolio.regime import build_active_weights
from src.portfolio.etf_filter import build_etf_eligibility
from src.paths import PROCESSED_DIR


DEFENSIVE_ASSET = "SHY"
LOOKBACK_VOL = 12
MAX_ETF_WEIGHT = 0.30
TRANSACTION_COST_BPS = 5


def compute_inverse_vol_weights(vols):
    vols = vols.replace(0, np.nan).dropna()
    if len(vols) == 0:
        return pd.Series(dtype=float)

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights


def get_active_band(target_weight):
    if pd.isna(target_weight):
        return 0.30, 0.70

    if np.isclose(target_weight, 0.60):
        return 0.40, 0.75
    elif np.isclose(target_weight, 0.40):
        return 0.25, 0.60
    else:  # risk-off regime from 0.20 target
        return 0.20, 0.45


def allocate_with_cap(base_weights, total_budget, max_weight):
    if len(base_weights) == 0 or total_budget <= 0:
        return pd.Series(dtype=float)

    base_weights = base_weights.dropna()
    if len(base_weights) == 0:
        return pd.Series(dtype=float)

    base_weights = base_weights / base_weights.sum()

    final = pd.Series(0.0, index=base_weights.index)
    remaining_assets = list(base_weights.index)
    remaining_budget = total_budget

    while len(remaining_assets) > 0 and remaining_budget > 1e-12:
        sub = base_weights.loc[remaining_assets]
        sub = sub / sub.sum()

        proposed = sub * remaining_budget
        over_cap = proposed[proposed > max_weight + 1e-12]

        if len(over_cap) == 0:
            final.loc[remaining_assets] += proposed
            remaining_budget = 0.0
            break

        capped_names = over_cap.index.tolist()
        for name in capped_names:
            final.loc[name] = max_weight

        remaining_budget = total_budget - final.sum()
        remaining_assets = [a for a in remaining_assets if a not in capped_names]

    return final[final > 0]


def build_hierarchical_allocation_competitive(save_output=True):
    df = load_allocation_returns().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    active_targets = build_active_weights(save_output=False).copy()
    active_targets["date"] = pd.to_datetime(active_targets["date"])

    etf_elig = build_etf_eligibility(save_output=False).copy()
    etf_elig["date"] = pd.to_datetime(etf_elig["date"])

    investable_cols = [c for c in df.columns if c != "date"]
    risk_etf_cols = [c for c in investable_cols if c not in ["active_sleeve", DEFENSIVE_ASSET]]

    vol_assets = ["active_sleeve"] + risk_etf_cols
    vol_df = df[["date"] + vol_assets].copy()
    for col in vol_assets:
        vol_df[col] = vol_df[col].rolling(LOOKBACK_VOL).std() * np.sqrt(12)

    etf_vol_long = vol_df.melt(
        id_vars="date",
        value_vars=risk_etf_cols,
        var_name="ticker",
        value_name="vol_12m"
    )

    etf_info = etf_elig.merge(etf_vol_long, on=["date", "ticker"], how="left")

    active_vol = vol_df[["date", "active_sleeve"]].rename(columns={"active_sleeve": "active_vol_12m"})

    weights_rows = []

    for dt in df["date"]:
        active_target = active_targets.loc[
            active_targets["date"] == dt, "active_weight"
        ]
        active_target = active_target.iloc[0] if len(active_target) > 0 else 0.40
        active_min, active_max = get_active_band(active_target)

        snap = etf_info[(etf_info["date"] == dt) & (etf_info["ticker"].isin(risk_etf_cols))].copy()
        snap = snap[snap["trend_ok"]].copy()
        snap = snap.dropna(subset=["vol_12m"])

        active_vol_row = active_vol.loc[active_vol["date"] == dt, "active_vol_12m"]
        active_vol_value = active_vol_row.iloc[0] if len(active_vol_row) > 0 else np.nan

        comp_vols = {}

        if pd.notna(active_vol_value) and active_vol_value > 0:
            comp_vols["active_sleeve"] = active_vol_value

        for _, r in snap.iterrows():
            comp_vols[r["ticker"]] = r["vol_12m"]

        comp_vols = pd.Series(comp_vols, dtype=float)
        base_weights = compute_inverse_vol_weights(comp_vols)

        final_weights = {"date": dt}

        for col in investable_cols:
            final_weights[col] = 0.0

        if len(base_weights) == 0:
            final_weights[DEFENSIVE_ASSET] = 1.0
            weights_rows.append(final_weights)
            continue

        base_active = base_weights.get("active_sleeve", 0.0)
        active_w = float(np.clip(base_active, active_min, active_max))

        remaining_budget = 1.0 - active_w

        eligible_etfs = [a for a in base_weights.index if a != "active_sleeve"]
        if len(eligible_etfs) > 0 and remaining_budget > 0:
            etf_base = base_weights.loc[eligible_etfs]
            etf_base = etf_base / etf_base.sum()

            etf_alloc = allocate_with_cap(
                base_weights=etf_base,
                total_budget=remaining_budget,
                max_weight=MAX_ETF_WEIGHT
            )

            for asset, w in etf_alloc.items():
                final_weights[asset] = float(w)

            allocated_to_etfs = float(etf_alloc.sum())
            final_weights[DEFENSIVE_ASSET] = remaining_budget - allocated_to_etfs
        else:
            final_weights[DEFENSIVE_ASSET] = remaining_budget

        final_weights["active_sleeve"] = active_w

        total_w = sum(final_weights[c] for c in investable_cols)
        if total_w > 0:
            for c in investable_cols:
                final_weights[c] = final_weights[c] / total_w

        weights_rows.append(final_weights)

    weights_df = pd.DataFrame(weights_rows).sort_values("date").copy()

    weight_cols = [c for c in weights_df.columns if c != "date"]
    shifted_weights = weights_df.copy()
    shifted_weights[weight_cols] = shifted_weights[weight_cols].shift(1)

    returns_df = df.copy()
    returns_df = returns_df.merge(
        shifted_weights,
        on="date",
        how="left",
        suffixes=("_ret", "_w")
    )

    returns_df = returns_df.dropna(subset=["active_sleeve_w"]).copy()

    returns_df["gross_ret"] = 0.0
    for asset in investable_cols:
        returns_df["gross_ret"] += (
            returns_df[f"{asset}_w"].fillna(0.0) * returns_df[f"{asset}_ret"]
        )
    
    weight_cols_w = [f"{asset}_w" for asset in investable_cols]

    turnover_vals = []
    prev_weights = None

    for _, row in returns_df.iterrows():
        current_weights = row[weight_cols_w].fillna(0.0).values.astype(float)

        if prev_weights is None:
            turnover = 0.0
        else:
            turnover = np.abs(current_weights - prev_weights).sum()

        turnover_vals.append(turnover)
        prev_weights = current_weights.copy()

    returns_df["turnover"] = turnover_vals

    tc = TRANSACTION_COST_BPS / 10000.0
    returns_df["cost"] = returns_df["turnover"] * tc
    returns_df["portfolio_ret"] = returns_df["gross_ret"] - returns_df["cost"]

    returns_df["equity_curve"] = (1 + returns_df["portfolio_ret"]).cumprod()
    returns_df = returns_df[
        ["date", "gross_ret", "turnover", "cost", "portfolio_ret", "equity_curve"]
    ].copy()

    if save_output:
        weights_df.to_parquet(
            PROCESSED_DIR / "allocation_weights_hierarchical_competitive.parquet",
            index=False
        )
        returns_df.to_parquet(
            PROCESSED_DIR / "allocation_backtest_hierarchical_competitive.parquet",
            index=False
        )

    return returns_df, weights_df


def compute_allocation_metrics(backtest_df):
    df = backtest_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rf = load_risk_free().copy()
    rf["date"] = pd.to_datetime(rf["date"])

    df = df.merge(
        rf[["date", "rf_monthly"]],
        on="date",
        how="left"
    )

    df["rf_monthly"] = df["rf_monthly"].fillna(0.0)

    rets = df["portfolio_ret"].dropna()
    excess_rets = (df["portfolio_ret"] - df["rf_monthly"]).dropna()

    equity = (1 + rets).cumprod()
    ann_return = equity.iloc[-1] ** (12 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(12)

    excess_equity = (1 + excess_rets).cumprod()
    ann_excess_return = excess_equity.iloc[-1] ** (12 / len(excess_rets)) - 1
    sharpe = ann_excess_return / ann_vol if ann_vol != 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    avg_turnover = backtest_df['turnover'].mean()
    avg_cost = backtest_df['cost'].mean()

    ann_turnover = avg_turnover * 12
    ann_cost = avg_cost * 12

    return pd.Series({
        "n_months": len(rets),
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_monthly_turnover": avg_turnover,
        "avg_monthly_cost": avg_cost,
        "annual_turnover": ann_turnover,
        "annual_cost": ann_cost
    })