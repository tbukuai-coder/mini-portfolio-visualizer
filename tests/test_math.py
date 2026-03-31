import numpy as np
import pandas as pd


def _calculate_portfolio(prices, allocations, initial_investment, rebalance_freq):
    """Logic copied from app.py (kept local so tests don't import streamlit app)."""
    weights = np.array(allocations) / 100
    returns = prices.pct_change().dropna()

    if rebalance_freq == "None":
        portfolio_returns = (returns * weights).sum(axis=1)
        return initial_investment * (1 + portfolio_returns).cumprod()

    freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
    rebal_dates = returns.resample(freq_map[rebalance_freq]).last().index

    portfolio_value = pd.Series(index=returns.index, dtype=float)
    current_value = initial_investment
    current_weights = weights.copy()

    for i, date in enumerate(returns.index):
        daily_return = (returns.loc[date] * current_weights).sum()
        current_value *= (1 + daily_return)
        portfolio_value.loc[date] = current_value

        if i < len(returns.index) - 1:
            individual_returns = 1 + returns.loc[date].values
            current_weights = current_weights * individual_returns
            current_weights = current_weights / current_weights.sum()

        if date in rebal_dates:
            current_weights = weights.copy()

    return portfolio_value


def test_calculate_portfolio_constant_prices_is_flat():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"A": 100.0, "B": 200.0}, index=idx)

    pv = _calculate_portfolio(prices, allocations=[50, 50], initial_investment=10000, rebalance_freq="None")

    assert len(pv) == len(idx) - 1  # pct_change drops the first row
    assert float(pv.iloc[0]) == 10000.0
    assert float(pv.iloc[-1]) == 10000.0


def test_calculate_portfolio_monotonic_when_returns_positive():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    # Both assets steadily go up 1% per day
    prices = pd.DataFrame(
        {
            "A": 100.0 * (1.01 ** np.arange(len(idx))),
            "B": 50.0 * (1.01 ** np.arange(len(idx))),
        },
        index=idx,
    )

    pv = _calculate_portfolio(prices, allocations=[60, 40], initial_investment=10000, rebalance_freq="None")

    # Strictly increasing series
    assert (pv.diff().dropna() > 0).all()
