import pandas as pd
import numpy as np

from src.data.loaders import load_alpha_model
from src.backtest.engine import build_monthly_returns
from src.paths import PROCESSED_DIR



FEATURES = [c for c in load_alpha_model().columns if c.endswith("_z")]

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
    

    results = {}

    for col in FEATURES:

        signal_eval = evaluation.dropna(subset=[col, "fwd_ret_1m"]).copy()

        signal_counts = (
            signal_eval.groupby("date")["ticker"]
            .count()
            .rename("n_names")
        )

        valid_signal_dates = signal_counts[signal_counts >= min_names].index
        signal_eval = signal_eval[signal_eval["date"].isin(valid_signal_dates)].copy()
        
        def spearman_ic(group):
            if group[col].nunique() < 2 or group["fwd_ret_1m"].nunique() < 2:
                return np.nan
            return group[col].corr(group["fwd_ret_1m"], method="spearman")
    
        monthly_ics = (
            signal_eval.groupby("date")
            .apply(spearman_ic)
            .rename("ic")
            .dropna()
        )

        if len(monthly_ics) == 0:
            results[col] = {
                "mean_ic": np.nan,
                "ic_std": np.nan,
                "ic_tstat": np.nan,
                "hit_rate": np.nan,
                "n_months": 0,
                "avg_names": np.nan,
                "monthly_ics": monthly_ics,
            }
            continue

        mean_ic = monthly_ics.mean()
        ic_std = monthly_ics.std(ddof=1)
        ic_tstat = mean_ic / (ic_std / np.sqrt(len(monthly_ics))) if ic_std > 0 else np.nan
        hit_rate = (monthly_ics > 0).mean()
        avg_names = signal_counts.loc[monthly_ics.index].mean()

        results[col] = {
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_tstat": ic_tstat,
            "hit_rate": hit_rate,
            "n_months": len(monthly_ics),
            "avg_names": avg_names,
            "monthly_ics": monthly_ics,
        }

    return results

def signal_corr_matrix(features=None, min_names=20):
    if features is None:
        features = FEATURES
    alpha_df = load_alpha_model().copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    corr_mats = []

    for dt, group in alpha_df.groupby('date'):

        if len(group) < min_names:
            continue
        
        corr = group[features].corr(method='spearman', min_periods=min_names)
        corr_mats.append(corr)

    if len(corr_mats) == 0:
        return pd.DataFrame(index=features, columns=features, dtype=float)

    avg_corr = (
        pd.concat(corr_mats, keys=range(len(corr_mats)))
        .groupby(level=1)
        .mean()
    )
    
    return avg_corr.loc[features, features]


def evaluate_signal_ic(evaluation, signal_col, min_names=20):
    signal_eval = evaluation.dropna(subset=[signal_col, "fwd_ret_1m"]).copy()

    signal_counts = signal_eval.groupby("date")["ticker"].count()
    valid_dates = signal_counts[signal_counts >= min_names].index
    signal_eval = signal_eval[signal_eval["date"].isin(valid_dates)].copy()

    def spearman_ic(group):
        if group[signal_col].nunique() < 2 or group["fwd_ret_1m"].nunique() < 2:
            return np.nan
        return group[signal_col].corr(group["fwd_ret_1m"], method="spearman")

    monthly_ics = (
        signal_eval.groupby("date")
        .apply(spearman_ic)
        .dropna()
    )

    if len(monthly_ics) == 0:
        return {
            "mean_ic": np.nan,
            "ic_tstat": np.nan,
            "hit_rate": np.nan,
            "n_months": 0,
            "avg_names": np.nan,
            "monthly_ics": monthly_ics,
        }

    mean_ic = monthly_ics.mean()
    ic_std = monthly_ics.std(ddof=1)
    ic_tstat = mean_ic / (ic_std / np.sqrt(len(monthly_ics))) if ic_std > 0 else np.nan

    return {
        "mean_ic": mean_ic,
        "ic_tstat": ic_tstat,
        "hit_rate": (monthly_ics > 0).mean(),
        "n_months": len(monthly_ics),
        "avg_names": signal_counts.loc[monthly_ics.index].mean(),
        "monthly_ics": monthly_ics,
    }


def incremental_ic_selection_by_tstat(
    candidate_signals=None,
    min_names=50,
    max_signals=15,
    min_tstat_improvement=0.05,
):
    if candidate_signals is None:
        candidate_signals = FEATURES.copy()

    alpha_df = load_alpha_model().copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns().copy()
    monthly_returns["date"] = pd.to_datetime(monthly_returns["date"])

    evaluation = alpha_df.merge(
        monthly_returns[["date", "ticker", "fwd_ret_1m"]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    selected = []
    remaining = candidate_signals.copy()
    history = []

    current_tstat = None
    current_ic = None

    while remaining and len(selected) < max_signals:
        trials = []

        for sig in remaining:
            test_signals = selected + [sig]

            evaluation["candidate_alpha"] = evaluation[test_signals].mean(
                axis=1,
                skipna=True
            )

            stats = evaluate_signal_ic(
                evaluation,
                "candidate_alpha",
                min_names=min_names,
            )

            tstat_improvement = (
                stats["ic_tstat"]
                if current_tstat is None
                else stats["ic_tstat"] - current_tstat
            )

            ic_improvement = (
                stats["mean_ic"]
                if current_ic is None
                else stats["mean_ic"] - current_ic
            )

            trials.append({
                "candidate": sig,
                "mean_ic": stats["mean_ic"],
                "ic_tstat": stats["ic_tstat"],
                "hit_rate": stats["hit_rate"],
                "n_months": stats["n_months"],
                "avg_names": stats["avg_names"],
                "ic_improvement": ic_improvement,
                "tstat_improvement": tstat_improvement,
                "selected_set": test_signals,
            })

        trials_df = pd.DataFrame(trials).dropna(subset=["ic_tstat"])

        if trials_df.empty:
            break

        trials_df = trials_df.sort_values(
            ["ic_tstat", "mean_ic"],
            ascending=False
        )

        best = trials_df.iloc[0]

        if current_tstat is not None and best["tstat_improvement"] < min_tstat_improvement:
            break

        selected.append(best["candidate"])
        remaining.remove(best["candidate"])

        current_tstat = best["ic_tstat"]
        current_ic = best["mean_ic"]

        history.append(best.to_dict())

    pd.Series(selected, name="signal").to_frame().to_parquet(
        PROCESSED_DIR / "alpha_signals.parquet",
        index=False
    )


    return selected, pd.DataFrame(history)


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

