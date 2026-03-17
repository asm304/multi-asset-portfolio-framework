import pandas as pd
import numpy as np


def compute_monthly_rank_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []

    for date, group in df.groupby("date"):
        group = group.dropna(subset=["ml_signal", "fwd_ret_1m"]).copy()

        if len(group) < 10:
            continue

        ic = group["ml_signal"].corr(group["fwd_ret_1m"], method="spearman")

        rows.append({
            "date": date,
            "rank_ic": ic,
            "n_stocks": len(group)
        })

    return pd.DataFrame(rows).sort_values("date")


def summarize_rank_ic(ic_df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "mean_rank_ic": ic_df["rank_ic"].mean(),
        "std_rank_ic": ic_df["rank_ic"].std(),
        "ir_rank_ic": ic_df["rank_ic"].mean() / ic_df["rank_ic"].std() if ic_df["rank_ic"].std() != 0 else np.nan,
        "positive_ic_pct": (ic_df["rank_ic"] > 0).mean(),
        "n_months": len(ic_df)
    })


def compute_prediction_deciles(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    out = []

    for date, group in df.groupby("date"):
        group = group.dropna(subset=["ml_signal", "fwd_ret_1m"]).copy()

        if len(group) < 10:
            continue

        try:
            group["pred_decile"] = pd.qcut(
                group["ml_signal"],
                q=10,
                labels=False,
                duplicates="drop"
            )
        except ValueError:
            continue

        decile_ret = (
            group.groupby("pred_decile")["fwd_ret_1m"]
            .mean()
            .reset_index()
        )
        decile_ret["date"] = date
        out.append(decile_ret)

    return pd.concat(out, ignore_index=True)


def summarize_deciles(decile_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        decile_df.groupby("pred_decile")["fwd_ret_1m"]
        .mean()
        .reset_index()
        .sort_values("pred_decile")
    )

    return summary