import requests
import pandas as pd
import numpy as np

from src.paths import RAW_DIR


def fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["observations"]

    df = pd.DataFrame(data)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


def download_risk_free(api_key: str, series_id: str = "TB3MS"):

    rf = fetch_fred_series(series_id=series_id, api_key=api_key).copy()

    rf["date"] = rf["date"] + pd.offsets.MonthEnd(0)
    rf = rf.rename(columns={"value": "rf_annual_pct"})

    rf["rf_monthly"] = (
        (1 + rf["rf_annual_pct"] / 100.0) ** (1 / 12)
        - 1
    )
    rf = rf.dropna().copy()

    path = RAW_DIR / "risk_free.parquet"
    rf.to_parquet(path, index=False)

    return rf
