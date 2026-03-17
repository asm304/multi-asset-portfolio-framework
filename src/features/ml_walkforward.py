import pandas as pd
import numpy as np

from src.data.loaders import load_ml_dataset
from src.features.ml_models import make_ridge_model, make_xgb_model
from src.paths import PROCESSED_DIR


FEATURE_COLS = [
    "liquidity_z",
    "mom_12_1_z",
    "mom_6_1_z",
    "rev_1m_z",
    "vol_12m_z",
    "beta_12m_z",
]

TARGET_COL = "fwd_ret_1m"


def _get_model(model_name: str):
    if model_name == "ridge":
        return make_ridge_model()
    elif model_name == "xgb":
        return make_xgb_model()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_walkforward_ml(
    model_name="ridge",
    train_months=60,
    save_output=True
):
    df = load_ml_dataset().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).copy()

    unique_dates = sorted(df["date"].unique())
    predictions = []

    for i in range(train_months, len(unique_dates)):
        pred_date = unique_dates[i]
        train_start = unique_dates[i - train_months]
        train_end = unique_dates[i - 1]

        train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
        test_mask = df["date"] == pred_date

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]

        X_test = test_df[FEATURE_COLS]

        model = _get_model(model_name)
        model.fit(X_train, y_train)

        test_df["ml_signal"] = model.predict(X_test)
        test_df["model_name"] = model_name

        predictions.append(
            test_df[["date", "ticker", "ml_signal", TARGET_COL, "model_name"]].copy()
        )

        if (i - train_months) % 12 == 0:
            print(f"[{model_name}] predicted {pred_date.date()} using {train_start.date()} to {train_end.date()}")

    if not predictions:
        return pd.DataFrame()

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df = pred_df.sort_values(["date", "ticker"]).copy()

    if save_output:
        path = PROCESSED_DIR / f"ml_predictions_{model_name}.parquet"
        pred_df.to_parquet(path, index=False)

    return pred_df