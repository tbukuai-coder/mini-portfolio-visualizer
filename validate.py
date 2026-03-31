"""
Validate mini-portfolio-visualizer calculations against expected values.
Run a test case that can be compared with portfoliovisualizer.com
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Test parameters (use these same values on portfoliovisualizer.com)
TICKERS = ["VTI", "VXUS", "BND"]
ALLOCATIONS = [60, 25, 15]  # percentages
INITIAL_INVESTMENT = 10000
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"  # Use a fixed end date for comparison
REBALANCE = "Annually"
RISK_FREE_RATE = 0.02

print("=" * 60)
print("MINI PORTFOLIO VISUALIZER - VALIDATION TEST")
print("=" * 60)
print(f"\nTest Parameters:")
print(f"  Tickers: {TICKERS}")
print(f"  Allocations: {ALLOCATIONS}%")
print(f"  Initial Investment: ${INITIAL_INVESTMENT:,}")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Rebalancing: {REBALANCE}")
print(f"  Risk-Free Rate: {RISK_FREE_RATE*100}%")
print("\n" + "-" * 60)

# Fetch data
print("\nFetching data from Yahoo Finance...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
prices = data['Close'].dropna()

print(f"Data range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
print(f"Trading days: {len(prices)}")

# Calculate portfolio with annual rebalancing
weights = np.array(ALLOCATIONS) / 100
returns = prices.pct_change().dropna()

# Annual rebalancing
rebal_dates = returns.resample('Y').last().index

portfolio_value = pd.Series(index=returns.index, dtype=float)
current_value = INITIAL_INVESTMENT
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

# Calculate metrics
final_value = portfolio_value.iloc[-1]
total_return = (final_value / INITIAL_INVESTMENT - 1) * 100
years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
cagr = ((final_value / INITIAL_INVESTMENT) ** (1/years) - 1) * 100

daily_returns = portfolio_value.pct_change().dropna()
volatility = daily_returns.std() * np.sqrt(252) * 100

rolling_max = portfolio_value.cummax()
drawdown = (portfolio_value - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

excess_returns = daily_returns.mean() * 252 - RISK_FREE_RATE
sharpe = excess_returns / (daily_returns.std() * np.sqrt(252))

print("\n" + "=" * 60)
print("OUR RESULTS:")
print("=" * 60)
print(f"  Final Value:    ${final_value:,.2f}")
print(f"  Total Return:   {total_return:.2f}%")
print(f"  CAGR:           {cagr:.2f}%")
print(f"  Volatility:     {volatility:.2f}%")
print(f"  Max Drawdown:   {max_drawdown:.2f}%")
print(f"  Sharpe Ratio:   {sharpe:.2f}")

print("\n" + "=" * 60)
print("TO VERIFY ON PORTFOLIO VISUALIZER:")
print("=" * 60)
print("""
1. Go to: https://www.portfoliovisualizer.com/backtest-portfolio
2. Enter:
   - Asset 1: VTI, 60%
   - Asset 2: VXUS, 25%
   - Asset 3: BND, 15%
   - Initial Amount: $10,000
   - Start Year: 2019
   - End Year: 2023 (they use full years)
   - Rebalancing: Annually
3. Compare the results!

Note: Small differences are expected due to:
   - Dividend handling (we use adjusted close)
   - Exact rebalancing dates
   - Data source differences
""")

# Also show individual asset returns for verification
print("\n" + "-" * 60)
print("Individual Asset Performance (for cross-check):")
print("-" * 60)
for ticker in TICKERS:
    start_price = prices[ticker].iloc[0]
    end_price = prices[ticker].iloc[-1]
    asset_return = (end_price / start_price - 1) * 100
    print(f"  {ticker}: ${start_price:.2f} → ${end_price:.2f} ({asset_return:+.2f}%)")
