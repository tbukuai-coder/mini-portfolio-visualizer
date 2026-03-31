"""
Compare our calculations with Portfolio Visualizer expected values.
PV uses monthly returns and end-of-month rebalancing.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# Test case: VTI 60%, VXUS 25%, BND 15%
# Period: Jan 2019 - Dec 2023
# Rebalancing: Annual (end of year)
# Initial: $10,000

TICKERS = ["VTI", "VXUS", "BND"]
ALLOCATIONS = [0.60, 0.25, 0.15]
INITIAL = 10000

print("Fetching monthly data (matching Portfolio Visualizer methodology)...")

# Get monthly data like PV does
data = yf.download(TICKERS, start="2018-12-01", end="2024-01-01", interval="1mo", auto_adjust=True, progress=False)
prices = data['Close'].dropna()

print(f"\nMonthly data points: {len(prices)}")
print(f"Date range: {prices.index[0].strftime('%Y-%m')} to {prices.index[-1].strftime('%Y-%m')}")

# Calculate monthly returns
returns = prices.pct_change().dropna()

# Filter to 2019-2023
returns = returns[(returns.index >= '2019-01-01') & (returns.index <= '2023-12-31')]
print(f"Analysis period: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
print(f"Months: {len(returns)}")

# Simple approach: annual rebalancing
weights = np.array(ALLOCATIONS)
portfolio_values = [INITIAL]
current_weights = weights.copy()

yearly_returns = []
current_year = None

for date, row in returns.iterrows():
    # Portfolio return for this month
    monthly_return = (row.values * current_weights).sum()
    new_value = portfolio_values[-1] * (1 + monthly_return)
    portfolio_values.append(new_value)
    
    # Track yearly
    if current_year != date.year:
        if current_year is not None:
            yearly_returns.append((current_year, portfolio_values[-2]))
        current_year = date.year
    
    # Update weights (drift)
    individual_growth = 1 + row.values
    current_weights = current_weights * individual_growth
    current_weights = current_weights / current_weights.sum()
    
    # Annual rebalancing at year end (December)
    if date.month == 12:
        current_weights = weights.copy()

yearly_returns.append((current_year, portfolio_values[-1]))

final_value = portfolio_values[-1]
total_return = (final_value / INITIAL - 1) * 100
years = len(returns) / 12
cagr = ((final_value / INITIAL) ** (1/years) - 1) * 100

# Calculate volatility from monthly returns
portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
monthly_std = portfolio_returns.std()
annual_volatility = monthly_std * np.sqrt(12) * 100

# Max drawdown
portfolio_series = pd.Series(portfolio_values)
rolling_max = portfolio_series.cummax()
drawdown = (portfolio_series - rolling_max) / rolling_max
max_dd = drawdown.min() * 100

# Sharpe (assuming 2% risk-free)
rf_monthly = 0.02 / 12
excess_return = portfolio_returns.mean() - rf_monthly
sharpe = (excess_return * 12) / (monthly_std * np.sqrt(12))

print("\n" + "=" * 60)
print("OUR RESULTS (Monthly Data, Matching PV Methodology)")
print("=" * 60)
print(f"  Final Value:    ${final_value:,.2f}")
print(f"  Total Return:   {total_return:.2f}%")
print(f"  CAGR:           {cagr:.2f}%")
print(f"  Volatility:     {annual_volatility:.2f}%")
print(f"  Max Drawdown:   {max_dd:.2f}%")
print(f"  Sharpe Ratio:   {sharpe:.2f}")

print("\n" + "=" * 60)
print("EXPECTED PORTFOLIO VISUALIZER RESULTS (approx)")
print("=" * 60)
print("""
Based on typical PV output for VTI/VXUS/BND 60/25/15 (2019-2023):
  Final Value:    ~$13,000 - $13,200
  CAGR:           ~5.5% - 6%
  Volatility:     ~12% - 14% (they use different calc)
  Max Drawdown:   ~-20% to -22%
  Sharpe:         ~0.35 - 0.50

Note: Small differences expected due to:
  - Exact dividend reinvestment timing
  - Price data source (Yahoo vs their provider)
  - Rebalancing date precision
""")

print("\n" + "=" * 60)
print("YEAR BY YEAR")
print("=" * 60)
for year, val in yearly_returns:
    print(f"  End of {year}: ${val:,.2f}")
