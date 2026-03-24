import pandas as pd
import numpy as np

from src.data.loaders import load_alpha_model
from src.backtest.engine import build_monthly_returns


def ic_evaluation(min_names=20):
    alpha_df = load_alpha_model().copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns().copy()
    monthly_returns["date"] = pd.to_datetime(monthly_returns["date"])

    evaluation = alpha_df.merge(
        monthly_returns,
        on=['date', 'ticker'],
        how='left',
        validate='one_to_one'
    )
    
    evaluation = evaluation.dropna(subset=["alpha", "fwd_ret_1m"]).copy()

    monthly_counts = (
        evaluation.groupby("date")["ticker"]
        .count()
        .rename("n_names")
    )

    valid_dates = monthly_counts[monthly_counts >= min_names].index
    evaluation = evaluation[evaluation["date"].isin(valid_dates)].copy()

    def spearman_ic(group):
        if group["alpha"].nunique() < 2 or group["fwd_ret_1m"].nunique() < 2:
            return np.nan
        return group["alpha"].corr(group["fwd_ret_1m"], method="spearman")
    
    monthly_ics = (
        evaluation.groupby("date")
        .apply(spearman_ic)
        .rename("ic")
        .dropna()
    )

    if len(monthly_ics) == 0:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "ic_tstat": np.nan,
            "hit_rate": np.nan,
            "n_months": 0,
            "avg_names": np.nan,
            "monthly_ics": monthly_ics,
            "monthly_counts": monthly_counts,
        }

    mean_ic = monthly_ics.mean()
    ic_std = monthly_ics.std(ddof=1)
    ic_tstat = mean_ic / (ic_std / np.sqrt(len(monthly_ics))) if ic_std > 0 else np.nan
    hit_rate = (monthly_ics > 0).mean()

    avg_names = monthly_counts.loc[monthly_ics.index].mean()

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_tstat": ic_tstat,
        "hit_rate": hit_rate,
        "n_months": len(monthly_ics),
        "avg_names": avg_names,
        "monthly_ics": monthly_ics,
        "monthly_counts": monthly_counts,
    }


def quantile_spread_evaluation(n_buckets=5, min_names=20):
    alpha_df = load_alpha_model().copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns().copy()
    monthly_returns["date"] = pd.to_datetime(monthly_returns["date"])

    evaluation = alpha_df.merge(
        monthly_returns,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one"
    )

    evaluation = evaluation.dropna(subset=["alpha", "fwd_ret_1m"]).copy()

    def assign_bucket(group):
        if len(group) < min_names:
            group["bucket"] = np.nan
            return group
        group["bucket"] = pd.qcut(
            group["alpha"],
            q=n_buckets,
            labels=False,
            duplicates="drop"
        )
        return group

    evaluation = (
        evaluation.groupby("date", group_keys=False)
        .apply(assign_bucket)
        .dropna(subset=["bucket"])
    )

    bucket_returns = (
        evaluation.groupby(["date", "bucket"])["fwd_ret_1m"]
        .mean()
        .unstack()
    )

    top_col = bucket_returns.columns.max()
    bottom_col = bucket_returns.columns.min()

    bucket_returns["spread"] = bucket_returns[top_col] - bucket_returns[bottom_col]

    return bucket_returns

