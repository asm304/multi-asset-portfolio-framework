import pandas as pd

from src.data.loaders import (
    load_stock_prices,
    load_fund_income_quarterly,
    load_fund_balance_quarterly,
    load_fund_cashflow_quarterly,
    load_earnings_history,
    load_earnings_trend_quarterly,
)
from src.paths import PROCESSED_DIR


def load_fundamental_tables():
    return {
        "income_q": load_fund_income_quarterly(),
        "balance_q": load_fund_balance_quarterly(),
        "cashflow_q": load_fund_cashflow_quarterly(),
        "earnings_history": load_earnings_history(),
        "earnings_trend": load_earnings_trend_quarterly(),
    }


def prepare_effective_dates(tables):
    income_q = tables["income_q"].copy()
    balance_q = tables["balance_q"].copy()
    cashflow_q = tables["cashflow_q"].copy()
    earnings_history = tables["earnings_history"].copy()
    earnings_trend = tables["earnings_trend"].copy()

    income_q["effective_date"] = income_q["filing_date"]
    balance_q["effective_date"] = balance_q["filing_date"]
    cashflow_q["effective_date"] = cashflow_q["filing_date"]

    earnings_history["effective_date"] = earnings_history["reportDate"]
    earnings_trend["effective_date"] = earnings_trend["date"]

    return {
        "income_q": income_q,
        "balance_q": balance_q,
        "cashflow_q": cashflow_q,
        "earnings_history": earnings_history,
        "earnings_trend": earnings_trend,
    }


def build_price_panel():
    prices = load_stock_prices()[["date", "ticker", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.dropna(subset=["date", "ticker"])
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prices


def merge_asof_panel(prices, df, date_col="effective_date"):
    df=df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "ticker"])
    df = df.sort_values(["effective_date", "ticker"]).reset_index(drop=True)

    if "date" in df.columns:
        df = df.rename(columns={"date": "period_date"})

    merged = pd.merge_asof(
        prices,
        df,
        left_on="date",
        right_on=date_col,
        by="ticker",
        direction="backward"
    )

    return merged


def build_income_signals(prices, tables):

    income_pt = merge_asof_panel(prices, tables["income_q"])

    income_pt['net_margin'] = income_pt["netIncome"] / income_pt["totalRevenue"]
    income_pt["operating_margin"] = income_pt["operatingIncome"] / income_pt["totalRevenue"]

    income_signals = income_pt[["date", "ticker", "net_margin", "operating_margin", "netIncome", "ebitda"]]

    return income_signals

def build_balance_signals(prices, tables):
    balance_pt = merge_asof_panel(prices, tables["balance_q"])

    balance_pt["neg_net_debt_to_assets"] = -balance_pt["netDebt"] / balance_pt["totalAssets"]
    balance_pt["neg_net_debt_to_equity"] = -balance_pt["netDebt"] / balance_pt["totalStockholderEquity"]
    balance_pt["shares_outstanding"] = balance_pt["commonStockSharesOutstanding"]

    balance_signals = balance_pt[["date", "ticker", "neg_net_debt_to_assets", "neg_net_debt_to_equity", "totalAssets", "shares_outstanding", "netDebt", "totalStockholderEquity"]]

    return balance_signals

def build_cashflow_signals(prices, tables):
    cashflow_pt = merge_asof_panel(prices, tables["cashflow_q"])

    cashflow_pt["operating_cf"] = cashflow_pt["totalCashFromOperatingActivities"]
    cashflow_pt["fcf"] = cashflow_pt["freeCashFlow"]
    cashflow_pt["net_buybacks"] = -cashflow_pt["salePurchaseOfStock"].fillna(0)
    cashflow_pt["dividends"] = -cashflow_pt["dividendsPaid"].fillna(0)

    cashflow_signals = cashflow_pt[["date", "ticker", "operating_cf", "fcf", "net_buybacks", "dividends"]]

    return cashflow_signals

def build_earnings_history_signals(prices, tables):
    earnings_history_pt = merge_asof_panel(prices, tables["earnings_history"])

    earnings_history_pt["earnings_surprise"] = earnings_history_pt["surprisePercent"]

    earnings_history_signals = earnings_history_pt[["date", "ticker", "earnings_surprise"]]

    return earnings_history_signals

def build_earnings_trend_signals(prices, tables):
    earnings_trend_pt = merge_asof_panel(prices, tables["earnings_trend"])

    earnings_trend_pt["eps_revision_7d"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend7daysAgo"]) 
    earnings_trend_pt["eps_revision_30d"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend30daysAgo"]) 
    earnings_trend_pt["eps_revision_60d"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend60daysAgo"]) 
    earnings_trend_pt["eps_revision_90d"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend90daysAgo"]) 

    earnings_trend_pt["eps_revision_7d_pct"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend7daysAgo"]) / abs(earnings_trend_pt["epsTrend7daysAgo"].replace(0, pd.NA))
    earnings_trend_pt["eps_revision_30d_pct"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend30daysAgo"]) / abs(earnings_trend_pt["epsTrend30daysAgo"].replace(0, pd.NA))
    earnings_trend_pt["eps_revision_60d_pct"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend60daysAgo"]) / abs(earnings_trend_pt["epsTrend60daysAgo"].replace(0, pd.NA))
    earnings_trend_pt["eps_revision_90d_pct"] = (earnings_trend_pt["epsTrendCurrent"] - earnings_trend_pt["epsTrend90daysAgo"]) / abs(earnings_trend_pt["epsTrend90daysAgo"].replace(0, pd.NA))

    earnings_trend_pt["revision_breadth_7d"] = (earnings_trend_pt["epsRevisionsUpLast7days"].fillna(0) - earnings_trend_pt["epsRevisionsDownLast7days"].fillna(0)) / earnings_trend_pt["earningsEstimateNumberOfAnalysts"].replace(0, pd.NA)
    earnings_trend_pt["revision_breadth_30d"] = (earnings_trend_pt["epsRevisionsUpLast30days"].fillna(0) - earnings_trend_pt["epsRevisionsDownLast30days"].fillna(0)) / earnings_trend_pt["earningsEstimateNumberOfAnalysts"].replace(0, pd.NA)
    earnings_trend_pt["revision_ratio_7d"] = (earnings_trend_pt["epsRevisionsUpLast7days"].fillna(0) - earnings_trend_pt["epsRevisionsDownLast7days"].fillna(0)) / (earnings_trend_pt["epsRevisionsUpLast7days"].fillna(0) + earnings_trend_pt["epsRevisionsDownLast7days"].fillna(0)).replace(0, pd.NA)
    earnings_trend_pt["revision_ratio_30d"] = (earnings_trend_pt["epsRevisionsUpLast30days"].fillna(0) - earnings_trend_pt["epsRevisionsDownLast30days"].fillna(0)) / (earnings_trend_pt["epsRevisionsUpLast30days"].fillna(0) + earnings_trend_pt["epsRevisionsDownLast30days"].fillna(0)).replace(0, pd.NA)

    earnings_trend_signals = earnings_trend_pt[["date", "ticker", "eps_revision_7d", "eps_revision_30d", "eps_revision_60d", "eps_revision_90d", "eps_revision_7d_pct", "eps_revision_30d_pct", "eps_revision_60d_pct", "eps_revision_90d_pct", "revision_breadth_7d", "revision_breadth_30d", "revision_ratio_7d", "revision_ratio_30d"]]

    return earnings_trend_signals

def build_fundamental_signals():
    
    tables = load_fundamental_tables()
    tables = prepare_effective_dates(tables)

    prices = build_price_panel()

    income_signals = build_income_signals(prices, tables)
    balance_signals = build_balance_signals(prices, tables)
    cashflow_signals = build_cashflow_signals(prices, tables)
    earnings_history_signals = build_earnings_history_signals(prices, tables)
    earnings_trend_signals = build_earnings_trend_signals(prices, tables)

    fund_signals = (
        prices[["date", "ticker", "close"]]
        .merge(income_signals, on=["date", "ticker"], how="left")
        .merge(balance_signals, on=["date", "ticker"], how="left")
        .merge(cashflow_signals, on=["date", "ticker"], how="left")
        .merge(earnings_history_signals, on=["date", "ticker"], how="left")
        .merge(earnings_trend_signals, on=["date", "ticker"], how="left")
    )

    fund_signals["market_cap"] = fund_signals["close"] * fund_signals["shares_outstanding"]
    fund_signals["enterprise_value"] = fund_signals["market_cap"] + fund_signals["netDebt"]

    fund_signals["accrual_quality"] = -(
        (fund_signals["netIncome"] - fund_signals["operating_cf"]) / fund_signals["totalAssets"]
    )

    fund_signals["eps_revision_7d_yield"] = fund_signals["eps_revision_7d"] / fund_signals["close"].replace(0, pd.NA)
    fund_signals["eps_revision_30d_yield"] = fund_signals["eps_revision_30d"] / fund_signals["close"].replace(0, pd.NA)
    fund_signals["eps_revision_60d_yield"] = fund_signals["eps_revision_60d"] / fund_signals["close"].replace(0, pd.NA)
    fund_signals["eps_revision_90d_yield"] = fund_signals["eps_revision_90d"] / fund_signals["close"].replace(0, pd.NA)

    fund_signals["earnings_yield"] = fund_signals["netIncome"] / fund_signals["market_cap"]
    fund_signals["fcf_yield"] = fund_signals["fcf"] / fund_signals["market_cap"]
    fund_signals["operating_cf_to_assets"] = fund_signals["operating_cf"] / fund_signals["totalAssets"]
    fund_signals["fcf_to_assets"] = fund_signals["fcf"] / fund_signals["totalAssets"]

    fund_signals["roe"] = fund_signals["netIncome"] / fund_signals["totalStockholderEquity"]
    fund_signals["roa"] = fund_signals["netIncome"] / fund_signals["totalAssets"]

    fund_signals["ebitda_ev"] = fund_signals["ebitda"] / fund_signals["enterprise_value"]
    fund_signals.loc[fund_signals["enterprise_value"] <= 0, "ebitda_ev"] = pd.NA

    fund_signals["shareholder_yield"] = (
        fund_signals["net_buybacks"] + fund_signals["dividends"]
    ) / fund_signals["market_cap"]

    fund_signals.to_parquet(PROCESSED_DIR / "debug_fundamental_signals.parquet", index=False)

    FINAL_SIGNAL_COLS = [
    "date",
    "ticker",

    # profitability / margins
    "net_margin",
    "operating_margin",
    "roe",
    "roa",

    # balance sheet / safety
    "neg_net_debt_to_assets",
    "neg_net_debt_to_equity",

    # cash flow / quality
    "accrual_quality",
    "operating_cf_to_assets",
    "fcf_to_assets",

    # value
    "earnings_yield",
    "fcf_yield",
    "ebitda_ev",

    # capital return
    "shareholder_yield",

    # earnings
    "earnings_surprise",

    # revisions
    "eps_revision_7d_pct",
    "eps_revision_30d_pct",
    "eps_revision_60d_pct",
    "eps_revision_90d_pct",
    "eps_revision_7d_yield",
    "eps_revision_30d_yield",
    "eps_revision_60d_yield",
    "eps_revision_90d_yield",
    "revision_breadth_7d",
    "revision_breadth_30d",
    "revision_ratio_7d",
    "revision_ratio_30d",

    #eligibility
    "market_cap",

    ]

    fund_signals_final = fund_signals[FINAL_SIGNAL_COLS].copy()

    fund_signals_final["date"] = pd.to_datetime(fund_signals_final["date"])
    fund_signals_final["month"] = fund_signals_final["date"].dt.to_period("M")

    monthly_fundamental_signals = (
        fund_signals_final
        .sort_values(["ticker", "date"])
        .groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .drop(columns=["month"])
        .copy()
    )

    monthly_fundamental_signals.to_parquet(
        PROCESSED_DIR / "fundamental_signals.parquet",
        index=False
    )

    return monthly_fundamental_signals

