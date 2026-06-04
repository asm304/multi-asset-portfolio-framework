import pandas as pd
from src.data.loaders import load_stock_prices, load_fundamental_signals
from src.paths import PROCESSED_DIR


def build_stock_universe(
    min_price = 5,
    min_adv = 10_000_000,
    min_history_days = 252,
    min_market_cap=1_000_000_000
):
    df = load_stock_prices().copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker','date'])

    fund = load_fundamental_signals()[["date", "ticker", "market_cap"]]

    df['dollar_volume'] = df['close'] * df['volume']

    df['adv_20'] = (
        df.groupby('ticker')['dollar_volume'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    )
    
    df['history_days'] = df.groupby('ticker').cumcount() + 1

    df = df.merge(
        fund,
        on=["date", "ticker"],
        how="left"
    )

    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .copy()
    )
    monthly["eligible"] = (
        (monthly["close"] > min_price) &
        (monthly["adv_20"] > min_adv) &
        (monthly["history_days"] >= min_history_days) &
        (monthly["market_cap"] >= min_market_cap)
    )
    universe = monthly[[
        "date", "ticker", "close", "volume",
        "dollar_volume", "adv_20", "history_days", "market_cap", "eligible"
    ]].copy()

    path = PROCESSED_DIR / "stock_universe.parquet"
    universe.to_parquet(path, index=False)

    eligible_universe = universe[universe["eligible"]].copy()

    path = PROCESSED_DIR / "eligible_universe.parquet"
    eligible_universe.to_parquet(path, index=False)

    return eligible_universe

