import pandas as pd
import yfinance as yf
from src.paths import RAW_DIR, PROCESSED_DIR
import re




def download_etf_prices(tickers,start,end):
    data = yf.download(tickers,start=start,end=end)['Close']
    data = data.stack().reset_index()
    data.columns = ["date", "ticker", "adj_close"]

    path = RAW_DIR / "etf_prices.parquet"
    data.to_parquet(path, index=False)


    return data

def get_stock_tickers():
    

    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        sep="|"
    )

    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
        sep="|"
    )  

    nasdaq = nasdaq[
        (nasdaq["ETF"] == "N") &
        (nasdaq["Test Issue"] == "N")
    ]   

    other = other[
        (other["ETF"] == "N") &
        (other["Test Issue"] == "N")
    ]


    nasdaq_symbols = nasdaq["Symbol"]
    other_symbols = other["ACT Symbol"]

    tickers = pd.concat([nasdaq_symbols, other_symbols]).dropna().unique().tolist()

    tickers = [t.strip() for t in tickers]

    tickers = [
        t for t in tickers
        if re.fullmatch(r"[A-Z\-\.]+", t)
    ]

    tickers = list(set(tickers))

    tickers = [t.replace(".", "-") for t in tickers]

    tickers.sort()

    return tickers
    
def download_stock_prices(tickers,start,end):
    data = []

    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]

        print(f"Downloading batch {i//100 + 1} of {(len(tickers)-1)//100 + 1}")

        batch_data = yf.download(
            tickers=batch,
            start=start,
            end=end,
            threads=False,
            progress=False

        )
        batch_data = batch_data.stack(level=1).reset_index()
        #print(batch_data.head())
        #print(batch_data.columns)

        batch_data = batch_data.rename(columns={
            'Date': 'date',
            'Ticker': 'ticker',
            'Adj Close': 'adj_close',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Open': 'open',
            'Volume': 'volume'
        })
        data.append(batch_data)

    final_data = pd.concat(data, ignore_index=True)
    final_data = final_data.drop(columns=["adj_close"])
    final_data.columns.name = None

    path = RAW_DIR / "stock_prices.parquet"
    final_data.to_parquet(path, index=False)

    return final_data

    
def load_etf_prices():
    path = RAW_DIR / "etf_prices.parquet"
    return pd.read_parquet(path)

def load_stock_prices():
    path = RAW_DIR / "stock_prices.parquet"
    return pd.read_parquet(path)

def load_stock_universe():
    path = PROCESSED_DIR / "stock_universe.parquet"
    return pd.read_parquet(path)
    
def load_eligible_universe():
    path = PROCESSED_DIR / 'eligible_universe.parquet'
    return pd.read_parquet(path)

def load_price_signals():
    path = PROCESSED_DIR / "price_signals.parquet"
    return pd.read_parquet(path)

def load_alpha_model():
    path = PROCESSED_DIR / 'alpha_model.parquet'
    return pd.read_parquet(path)

def load_xgb_alpha_model():
    path = PROCESSED_DIR / 'alpha_model_xgb.parquet'
    return pd.read_parquet(path)

def load_ridge_alpha_model():
    path = PROCESSED_DIR / 'alpha_model_ridge.parquet'
    return pd.read_parquet(path)

def load_backtest_results():
    path = PROCESSED_DIR / 'backtest_results.parquet'
    return pd.read_parquet(path)

def load_allocation_returns():
    path = PROCESSED_DIR / 'allocation_returns.parquet'
    return pd.read_parquet(path)

def load_ml_dataset():
    path = PROCESSED_DIR / "ml_dataset.parquet"
    return pd.read_parquet(path)

def load_ridge_signal():
    path = PROCESSED_DIR / 'ml_predictions_ridge.parquet'
    return pd.read_parquet(path)

def load_xgb_signal():
    path = PROCESSED_DIR / 'ml_predictions_xgb.parquet'
    return pd.read_parquet(path)

def load_fundamentals():
    pass

def load_metadata():
    pass