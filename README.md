# Systematic Multi-Asset Portfolio Construction & Backtesting Framework

## Overview
This project implements a systematic portfolio construction and backtesting framework combining cross-sectional equity selection with multi-asset allocation using ETFs.

The framework simulates how quantitative investment teams design, build, and evaluate portfolios under realistic assumptions.

## Key Features
- Multi-signal alpha model (momentum, reversal, volatility, liquidity, machine learning)
- Cross-sectional normalization and ranking
- Top-N portfolio construction (equal-weight)
- Monthly rebalancing with forward return alignment
- Transaction cost modeling based on turnover
- Portfolio performance analytics (Sharpe, volatility, drawdown)

## Workflow
1. Load and preprocess price + signal data  
2. Normalize signals cross-sectionally  
3. Construct composite alpha score  
4. Rank securities and select portfolio  
5. Backtest with transaction costs  
6. Evaluate performance metrics  

## Results
- ~14% annual return  
- ~0.62 Sharpe ratio  
- Realistic turnover and drawdowns  

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib
