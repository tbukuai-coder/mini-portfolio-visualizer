"""mini-portfolio-visualizer tests.

NOTE:
This file used to be an executable validation script (prints + network calls).
Pytest collects files named test_*.py, but it only runs functions/classes
prefixed with `test_`.

We keep this as a *manual* smoke script and prevent pytest from collecting it.
Run it directly if needed:

    python test_app.py

"""

# Tell pytest to ignore this module during collection.
__test__ = False

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data(tickers, start, end):
    """Fetch adjusted close prices for tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if len(tickers) == 1:
        return data[["Close"]].rename(columns={"Close": tickers[0]})
    return data["Close"]


def calculate_portfolio(prices, allocations, initial_investment, rebalance_freq):
    """Calculate portfolio value over time with optional rebalancing."""
    weights = np.array(allocations) / 100

    # Daily returns
    returns = prices.pct_change().dropna()

    if rebalance_freq == "None":
        # Buy and hold
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
    else:
        # Rebalancing
        freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
        rebal_dates = returns.resample(freq_map[rebalance_freq]).last().index

        portfolio_value = pd.Series(index=returns.index, dtype=float)
        current_value = initial_investment
        current_weights = weights.copy()

        for i, date in enumerate(returns.index):
            daily_return = (returns.loc[date] * current_weights).sum()
            current_value *= (1 + daily_return)
            portfolio_value.loc[date] = current_value

            # Update weights based on drift
            if i < len(returns.index) - 1:
                individual_returns = 1 + returns.loc[date].values
                current_weights = current_weights * individual_returns
                current_weights = current_weights / current_weights.sum()

            # Rebalance if needed
            if date in rebal_dates:
                current_weights = weights.copy()

    return portfolio_value


def calculate_metrics(portfolio_value, risk_free_rate):
    """Calculate portfolio performance metrics."""
    returns = portfolio_value.pct_change().dropna()

    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100

    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1) * 100

    volatility = returns.std() * np.sqrt(252) * 100

    # Max drawdown
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Sharpe ratio
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "final_value": portfolio_value.iloc[-1],
        "years": years,
    }


if __name__ == "__main__":
    # Manual smoke run with Marc Faber Portfolio
    TICKERS = ["VNQ", "SPY", "EFA", "VWO", "BND", "GLD"]
    ALLOCATIONS = [25, 13, 8, 4, 25, 25]
    INITIAL = 10000
    START = "2008-01-01"
    END = "2024-01-01"
    RISK_FREE = 0.02

    print("=" * 65)
    print("TESTING APP.PY FUNCTIONS - Marc Faber Portfolio")
    print("=" * 65)
    print(f"\nTickers: {TICKERS}")
    print(f"Allocations: {ALLOCATIONS}%")
    print(f"Initial: ${INITIAL:,}")
    print(f"Period: {START} to {END}")
    print("Rebalancing: Annually")

    print("\nFetching daily data...")
    prices = fetch_data(TICKERS, START, END)
    prices = prices.dropna()
    print(
        f"Data range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"Trading days: {len(prices)}")

    print("\nCalculating portfolio...")
    portfolio_value = calculate_portfolio(prices, ALLOCATIONS, INITIAL, "Annually")

    print("Calculating metrics...")
    metrics = calculate_metrics(portfolio_value, RISK_FREE)

    print("\n" + "=" * 65)
    print("APP.PY RESULTS (Daily Data)")
    print("=" * 65)
    print(f"  Final Value:    ${metrics['final_value']:,.2f}")
    print(f"  Total Return:   {metrics['total_return']:.2f}%")
    print(f"  CAGR:           {metrics['cagr']:.2f}%")
    print(f"  Volatility:     {metrics['volatility']:.2f}%")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.2f}")
    print(f"  Years:          {metrics['years']:.2f}")

    print("\n" + "=" * 65)
    print("COMPARISON: Validation Script vs App.py")
    print("=" * 65)
    print(
        """
Validation Script (Monthly Data):
  Final Value:    $21,017.57
  CAGR:           4.78%
  Volatility:     13.59%
  Max Drawdown:   -40.09%
  Sharpe:         0.27

Note: Small differences expected because:
  - Validation uses monthly data (like Portfolio Visualizer)
  - App uses daily data (more granular)
  - Rebalancing dates differ slightly
"""
    )
