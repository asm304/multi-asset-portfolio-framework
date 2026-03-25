import pandas as pd
import yfinance as yf
from src.paths import RAW_DIR, PROCESSED_DIR
import re
import time




def download_etf_prices(tickers,start,end):
    data = yf.download(tickers,start=start,end=end, auto_adjust=True, progress=False, threads=True)['Close']
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

def download_stock_sectors_from_universe():
    eligible = load_eligible_universe().copy()
    tickers = sorted(eligible["ticker"].dropna().unique().tolist())

    sector_rows = []

    print(f"Fetching sector info for {len(tickers)} tickers...")

    for i, symbol in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"Processed {i} tickers")

        try:
            info = yf.Ticker(symbol).info
            sector = info.get("sector")
            industry = info.get("industry")

            fetch_status = "ok"
            if sector is None and industry is None:
                fetch_status = "missing_metadata"


            sector_rows.append({
                "ticker": symbol,
                "sector": sector,
                "industry": industry,
                "fetch_status": fetch_status,
                "error_msg": None,
            })

        except Exception as e:

            if "Too Many Requests" in str(e):
                print(f"Rate limited at {symbol}, sleeping...")
                time.sleep(5)

                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    sector = info.get("sector")
                    industry = info.get("industry")

                    sector_rows.append({
                        "ticker": symbol,
                        "sector": sector,
                        "industry": industry,
                        "fetch_status": "retry_ok",
                        "error_msg": None,
                    })

                except Exception as e2:
                    sector_rows.append({
                        "ticker": symbol,
                        "sector": None,
                        "industry": None,
                        "fetch_status": "error",
                        "error_msg": str(e2),
                    })
            else:
                sector_rows.append({
                    "ticker": symbol,
                    "sector": None,
                    "industry": None,
                    "fetch_status": "error",
                    "error_msg": str(e),
                })
        time.sleep(0.1)

    sector_df = pd.DataFrame(sector_rows).drop_duplicates(subset=["ticker"])

    path = RAW_DIR / "stock_sectors.parquet"
    sector_df.to_parquet(path, index=False)

    return sector_df


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

def load_stock_sectors():
    path = RAW_DIR / "stock_sectors.parquet"
    return pd.read_parquet(path)

def load_allocation_weights():
    path = PROCESSED_DIR / "allocation_weights.parquet"
    return pd.read_parquet(path)

def load_allocation_backtest():
    path = PROCESSED_DIR / 'allocation_backtest.parquet'
    return pd.read_parquet(path)

def load_active_weights():
    path = PROCESSED_DIR / 'active_weights.parquet'
    return pd.read_parquet(path)

def load_eligile_etfs():
    path = PROCESSED_DIR / 'etf_eligibility.parquet'
    return pd.read_parquet(path)

def load_portfolio_weights():
    path = PROCESSED_DIR / 'allocation_weights_hierarchical.parquet'
    return pd.read_parquet(path)

def load_hierarchal_portfolio_backtest():
    path = PROCESSED_DIR / 'allocation_backtest_hierarchical.parquet'
    return pd.read_parquet(path)

def load_risk_free():
    path = RAW_DIR / "risk_free.parquet"
    return pd.read_parquet(path)

def load_competitive_hierarchal_portfolio_weights():
    path = PROCESSED_DIR / 'allocation_weights_hierarchical_competitive.parquet'
    return pd.read_parquet(path)

def load_competitive_hierarchal_portfolio_backtest():
    path = PROCESSED_DIR / "allocation_backtest_hierarchical_competitive.parquet"
    return pd.read_parquet(path)  

def load_fundamentals():
    pass

def load_metadata():
    pass