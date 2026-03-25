import pandas as pd
import numpy as np

from src.data.loaders import load_allocation_returns
from src.paths import PROCESSED_DIR


def build_active_regime():
    df = load_allocation_returns().copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").copy()

    active = df[["date", "active_sleeve"]].copy()

    active["equity"] = (1 + active["active_sleeve"]).cumprod()

    active["mom_12m"] = active["equity"].pct_change(11).shift(1)

    active["ma_10"] = active["equity"].rolling(10).mean()
    active["above_ma"] = active["equity"] > active["ma_10"]

    running_max = active["equity"].cummax()
    active["drawdown"] = active["equity"] / running_max - 1

    return active

def compute_active_weight(active):

    active = active.copy()

    conditions = [
        (active["mom_12m"] > 0) &
        (active["above_ma"]) &
        (active["drawdown"] > -0.15),

        (active["mom_12m"] > -0.05) &
        (active["drawdown"] > -0.25),
    ]

    weights = [
        0.60,  
        0.40,  
    ]

    active["active_weight"] = np.select(
        conditions,
        weights,
        default=0.20  
    )

    return active

def build_active_weights(save_output=True):

    active = build_active_regime()
    active = compute_active_weight(active)

    weights = active[["date", "active_weight"]].copy()

    if save_output:
        weights.to_parquet(
            PROCESSED_DIR / "active_weights.parquet",
            index=False
        )

    return weights