import pandas as pd
import numpy as np

from src.data.loaders import load_stock_prices, load_alpha_model
from src.backtest.engine import build_monthly_returns


def ic_evaluation():
    alpha_df = load_alpha_model().copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])

    monthly_returns = build_monthly_returns()

    evaluation = alpha_df.merge(
        monthly_returns,
        on=['date', 'ticker'],
        how='left',
        validate='one_to_one'
    )

    evaluation['mkt_ret_1m'] = evaluation.groupby('ticker')['mkt_ret_1m'].shift(-1)
    evaluation = evaluation.dropna(subset=["fwd_ret_1m"]).copy()

    evaluation['excess_return'] = evaluation['fwd_ret_1m'] - evaluation['beta_36m'] * evaluation['mkt_ret_1m']
    evaluation['ranked_excess_ret'] = evaluation.groupby('date')['excess_return'].rank(ascending=False, method='first')

    monthly_ics = evaluation.groupby('date').apply(
        lambda x: x[['rank_long', 'ranked_excess_ret']].corr(method='spearman').iloc[0,1]
    )

    mean_ic = monthly_ics.mean()
    ic_std = monthly_ics.std()

    t_stat = mean_ic / (ic_std / np.sqrt(len(monthly_ics)))

    return mean_ic, t_stat, monthly_ics

