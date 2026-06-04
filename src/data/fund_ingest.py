import requests
import time
import json
import gzip
from pathlib import Path
from dotenv import load_dotenv
import os
from src.data.loaders import get_stock_tickers
from src.paths import RAW_DIR

load_dotenv()

API_TOKEN = os.getenv("EODHD_API_KEY")

if API_TOKEN is None:
    raise ValueError("EODHD_API_KEY not found. Check .env file.")

BASE_URL = "https://eodhd.com/api/fundamentals/{}?api_token={}&fmt=json"

SAVE_DIR = RAW_DIR / "fundamentals_json"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FAILED_LOG = SAVE_DIR / "failed_tickers.txt"


def fetch_fundamentals(ticker):
    ticker_api = f"{ticker}.US"
    url = BASE_URL.format(ticker_api, API_TOKEN)

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if not data or "General" not in data:
                return None
            
            return data
        
        elif response.status_code == 429:
            print(f"Rate limited on {ticker}, sleeping...")
            time.sleep(2)
            return None
        
        else:
            print(f"Error {response.status_code} for {ticker}")
            return None

    except Exception as e:
        print(f"Exception for {ticker}: {e}")
        return None


def save_json_gzip(data, ticker):
    filepath = SAVE_DIR / f"{ticker}.US.json.gz"

    with gzip.open(filepath, "wt") as f:
        json.dump(data, f)


def ingest_all(tickers, delay=0.2, max_retries=3):
    failed = []

    failed_set = set()
    if FAILED_LOG.exists():
        with open(FAILED_LOG) as f:
            failed_set = set(line.strip() for line in f)

    for i, ticker in enumerate(tickers):
        filepath = SAVE_DIR / f"{ticker}.US.json.gz"

        if filepath.exists():
            continue

        print(f"{i+1}/{len(tickers)} - {ticker}")

        success = False

        for attempt in range(max_retries):
            data = fetch_fundamentals(ticker)

            if data:
                save_json_gzip(data, ticker)
                success = True
                break
            else:
                print(f"Retry {attempt+1} for {ticker}")
                time.sleep(2)

        if not success:
            failed.append(ticker)
            with open(FAILED_LOG, "a") as f:
                f.write(ticker + "\n")

        if i % 500 == 0 and i > 0:
            print(f"Checkpoint reached: {i} tickers processed")

        time.sleep(delay)

    print(f"\nDone. Failed this run: {len(failed)}")

if __name__ == "__main__":
    tickers = get_stock_tickers()
    
    ingest_all(tickers)

