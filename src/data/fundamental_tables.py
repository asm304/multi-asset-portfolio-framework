import json
import gzip
from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR, PROCESSED_DIR


JSON_DIR = RAW_DIR / "fundamentals_json"

def load_fund_json(filepath: Path) -> dict:
    with gzip.open(filepath, "rt") as f:
        return json.load(f)

def get_ticker(data, filepath):
    ticker = data.get("General", {}).get("Code")
    
    if ticker is None:
        ticker = filepath.name.replace(".US.json.gz", "")
    
    return ticker
    
def section_to_df(data: dict, path: list[str], ticker: str) -> pd.DataFrame:
    section = data
    for key in path:
        section = section.get(key, {})

    if not isinstance(section, dict) or len(section) == 0:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(section, orient="index")
    df = df.reset_index(names="period_end")
    df["ticker"] = ticker

    return df

def parse_financial_statement(statement_name: str, frequency: str = "quarterly") -> pd.DataFrame:
    rows = []

    for filepath in JSON_DIR.glob("*.json.gz"):
        data = load_fund_json(filepath)
        ticker = get_ticker(data, filepath)

        df = section_to_df(
            data=data,
            path=["Financials", statement_name, frequency],
            ticker=ticker,
        )

        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    for col in ["date", "filing_date", "period_end"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    
    out = clean_numeric_columns(out)

    return out

def parse_earnings_history() -> pd.DataFrame:
    rows = []

    for filepath in JSON_DIR.glob("*.json.gz"):
        data = load_fund_json(filepath)
        ticker = get_ticker(data, filepath)

        df = section_to_df(
            data=data,
            path=["Earnings", "History"],
            ticker=ticker,
        )

        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    for col in ["date", "reportDate", "period_end"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    out = clean_numeric_columns(out)

    return out

def parse_earnings_trend() -> pd.DataFrame:
    rows = []

    for filepath in JSON_DIR.glob("*.json.gz"):
        data = load_fund_json(filepath)
        ticker = get_ticker(data, filepath)

        df = section_to_df(
            data=data,
            path=["Earnings", "Trend"],
            ticker=ticker,
        )

        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    for col in ["date", "period_end"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    out = clean_numeric_columns(out)

    return out


def parse_company_metadata() -> pd.DataFrame:
    rows = []

    for filepath in JSON_DIR.glob("*.json.gz"):
        
        data = load_fund_json(filepath)
        ticker = get_ticker(data, filepath)

        general = data.get("General", {})
        highlights = data.get("Highlights", {})
        valuation = data.get("Valuation", {})
        shares = data.get("SharesStats", {})

        row = {
            "ticker": ticker,
            **{f"general_{k}": v for k, v in general.items()},
            **{f"highlights_{k}": v for k, v in highlights.items()},
            **{f"valuation_{k}": v for k, v in valuation.items()},
            **{f"shares_{k}": v for k, v in shares.items()},
        }

        rows.append(row)

    return pd.DataFrame(rows)

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    skip_cols = {"ticker", "period_end", "date", "filing_date", "reportDate", "currency_symbol", "currency", "beforeAfterMarket", "type", "fiscalQuarter"}

    for col in df.columns:
        if col not in skip_cols:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def build_fundamental_tables():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    income_q = parse_financial_statement("Income_Statement", "quarterly")
    balance_q = parse_financial_statement("Balance_Sheet", "quarterly")
    cashflow_q = parse_financial_statement("Cash_Flow", "quarterly")

    earnings_history = parse_earnings_history()
    earnings_trend = parse_earnings_trend()
    metadata = parse_company_metadata()

    income_q.to_parquet(PROCESSED_DIR / "fund_income_quarterly.parquet", index=False)
    balance_q.to_parquet(PROCESSED_DIR / "fund_balance_quarterly.parquet", index=False)
    cashflow_q.to_parquet(PROCESSED_DIR / "fund_cashflow_quarterly.parquet", index=False)
    earnings_history.to_parquet(PROCESSED_DIR / "earnings_history.parquet", index=False)
    earnings_trend.to_parquet(PROCESSED_DIR / "earnings_trend_quarterly.parquet", index=False)
    metadata.to_parquet(PROCESSED_DIR / "fund_metadata.parquet", index=False)

    print("Saved:")
    print("income_q:", income_q.shape)
    print("balance_q:", balance_q.shape)
    print("cashflow_q:", cashflow_q.shape)
    print("earnings_history:", earnings_history.shape)
    print("earnings_trend:", earnings_trend.shape)
    print("metadata:", metadata.shape)

    return {
        "income_q": income_q,
        "balance_q": balance_q,
        "cashflow_q": cashflow_q,
        "earnings_history": earnings_history,
        "earnings_trend": earnings_trend,
        "metadata": metadata,
    }


if __name__ == "__main__":
    build_fundamental_tables()