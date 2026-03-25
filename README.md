# Systematic Multi-Asset Portfolio Construction & Backtesting Framework

## Overview

This project implements a systematic portfolio construction and backtesting framework that combines a cross-sectional equity **active sleeve** with a **multi-asset ETF allocation layer**.

The goal is to simulate how quantitative portfolio teams research, construct, and evaluate portfolios under more realistic conditions, including:

- signal normalization
- portfolio construction constraints
- regime-aware allocation
- turnover-based transaction costs
- risk-adjusted performance evaluation

The framework is designed as a research pipeline rather than a single backtest. It separates alpha generation, active portfolio construction, and top-level asset allocation into modular components.

---

## Project Architecture

The framework is organized into two portfolio layers:

### 1. Active Equity Sleeve
A systematic equity strategy that ranks stocks cross-sectionally using price-based signals and constructs a concentrated active portfolio.

### 2. Multi-Asset Allocation Layer
A top-level allocation engine that combines the active sleeve with diversified ETF exposures across major asset classes such as:

- developed equities
- emerging equities
- Treasuries
- investment-grade credit
- inflation-protected bonds
- gold
- commodities
- REITs
- short-duration defensive assets

---

## Core Components

### Active Sleeve
The active sleeve currently focuses on momentum-style equity selection with risk controls.

Implemented features include:

- multi-horizon momentum signals
- residual / risk-adjusted momentum
- FIP-style momentum quality filtering
- beta-bucket normalization
- cross-sectional winsorization and z-scoring
- stock ranking and portfolio selection
- inverse-volatility weighting
- sector-cap constraints
- turnover-aware rebalancing
- transaction cost modeling

### Allocation Layer
The allocation framework builds a diversified portfolio on top of the active sleeve.

Implemented features include:

- ETF trend filtering
- active sleeve regime detection
- defensive fallback allocation to short-duration assets
- hierarchical allocation logic
- competitive allocation with active-sleeve guardrails
- inverse-volatility-based ETF weighting
- allocation-level turnover and transaction costs
- risk-free-rate-adjusted Sharpe ratio using 3-month Treasury data from FRED

---

## Current Allocation Approaches

The project includes multiple allocation frameworks to compare trade-offs between return, volatility, and drawdown.

### 1. Fixed Hierarchical Allocation
Uses regime-based fixed active-sleeve weights and allocates the remainder across ETFs.

### 2. Flat Risk-Parity Allocation
A prior research version that allowed all sleeves to compete equally under a risk-parity framework.

### 3. Competitive Hierarchical Allocation
The current preferred version.  
In this design:

- the active sleeve competes with eligible ETFs
- active allocation is constrained within regime-based guardrails
- ETF exposures are filtered using trend signals
- turnover and transaction costs are included
- Sharpe is computed using excess return over 3-month Treasury bills

---

## Research Workflow

1. Download and preprocess equity and ETF data  
2. Build eligible stock universe  
3. Engineer equity signals  
4. Normalize signals cross-sectionally  
5. Construct active sleeve and backtest equity strategy  
6. Build ETF return table and allocation inputs  
7. Apply regime logic and ETF trend filters  
8. Construct multi-asset portfolio weights  
9. Backtest allocation layer with turnover costs  
10. Evaluate return, volatility, drawdown, turnover, and Sharpe  

---

## Realism Features

This project includes several implementation choices intended to make results more realistic and more useful in an interview or research setting:

- walk-forward style signal usage
- forward return alignment
- transaction costs in the active sleeve
- transaction costs in the allocation layer
- turnover tracking
- defensive fallback allocation
- risk-free-rate-adjusted Sharpe ratio
- constrained portfolio weights
- regime-aware allocation bands

---

## Current Results

### Active Sleeve
The active equity sleeve currently produces approximately:

- **~11% annual return**
- **~0.67 Sharpe ratio**
- materially higher volatility and drawdown than the diversified portfolio

### Competitive Multi-Asset Portfolio
The current best diversified allocation version produces approximately:

- **~7.7% annual return**
- **~7.5% annualized volatility**
- **~0.88 Sharpe ratio** using 3-month Treasury bills as the risk-free rate
- **~14% max drawdown**
- allocation-level turnover and transaction costs included

These results reflect the trade-off between preserving active-sleeve alpha and reducing total portfolio risk through diversification.

---

## Why This Project Matters

This project is meant to demonstrate more than just factor backtesting. It is designed to reflect how real portfolio teams think about:

- alpha generation
- signal diversification
- portfolio construction
- turnover control
- transaction costs
- asset allocation
- drawdown management
- risk-adjusted returns

Rather than focusing only on prediction, the framework emphasizes the full process of turning signals into an investable portfolio.

---

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- matplotlib
- yfinance
- parquet-based data pipeline

---

## Repository Structure

```text
src/
├── alpha/
│   └── composite.py
├── backtest/
│   └── engine.py
├── data/
│   ├── loaders.py
│   ├── risk_free.py
│   └── ...
├── portfolio/
│   ├── regime.py
│   ├── etf_filter.py
│   ├── hierarchical_allocation.py
│   ├── hierarchical_allocation_competitive.py
│   └── ...
├── config.py
└── paths.py