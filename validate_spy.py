"""
Validate SPY benchmark calculation against Portfolio Visualizer
"""

import yfinance as yf
import pandas as pd
import numpy as np

print("=" * 65)
print("SPY BENCHMARK VALIDATION")
print("=" * 65)

# Test parameters - match what Portfolio Visualizer would use
INITIAL = 10000
START = "2019-01-01"
END = "2024-01-01"

print(f"\nParameters:")
print(f"  Ticker: SPY (S&P 500 ETF)")
print(f"  Initial: ${INITIAL:,}")
print(f"  Period: {START} to {END}")
print(f"  Strategy: Buy and Hold")

# Fetch daily data (what our app uses)
print("\n" + "-" * 65)
print("DAILY DATA (Our App Method)")
print("-" * 65)

spy_daily = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
prices_daily = spy_daily['Close'].squeeze()

returns_daily = prices_daily.pct_change().dropna()
final_daily = INITIAL * (1 + returns_daily).cumprod().iloc[-1]

years = (prices_daily.index[-1] - prices_daily.index[0]).days / 365.25
cagr_daily = ((final_daily / INITIAL) ** (1/years) - 1) * 100
total_return_daily = (final_daily / INITIAL - 1) * 100
volatility_daily = returns_daily.std() * np.sqrt(252) * 100

# Max drawdown
cummax = (1 + returns_daily).cumprod().cummax()
drawdown = (1 + returns_daily).cumprod() / cummax - 1
max_dd_daily = drawdown.min() * 100

# Sharpe
sharpe_daily = (returns_daily.mean() * 252 - 0.02) / (returns_daily.std() * np.sqrt(252))

print(f"  Data points: {len(prices_daily)}")
print(f"  Date range: {prices_daily.index[0].strftime('%Y-%m-%d')} to {prices_daily.index[-1].strftime('%Y-%m-%d')}")
print(f"\n  Final Value:    ${float(final_daily):,.2f}")
print(f"  Total Return:   {float(total_return_daily):.2f}%")
print(f"  CAGR:           {float(cagr_daily):.2f}%")
print(f"  Volatility:     {float(volatility_daily):.2f}%")
print(f"  Max Drawdown:   {float(max_dd_daily):.2f}%")
print(f"  Sharpe Ratio:   {float(sharpe_daily):.2f}")

# Fetch monthly data (what Portfolio Visualizer uses)
print("\n" + "-" * 65)
print("MONTHLY DATA (Portfolio Visualizer Method)")
print("-" * 65)

spy_monthly = yf.download("SPY", start="2018-12-01", end=END, interval="1mo", auto_adjust=True, progress=False)
prices_monthly = spy_monthly['Close'].squeeze()

# Filter to our period
returns_monthly = prices_monthly.pct_change().dropna()
returns_monthly = returns_monthly[(returns_monthly.index >= START) & (returns_monthly.index <= END)]

final_monthly = INITIAL * (1 + returns_monthly).cumprod().iloc[-1]
years_m = len(returns_monthly) / 12
cagr_monthly = ((final_monthly / INITIAL) ** (1/years_m) - 1) * 100
total_return_monthly = (final_monthly / INITIAL - 1) * 100
volatility_monthly = returns_monthly.std() * np.sqrt(12) * 100

# Max drawdown monthly
cummax_m = (1 + returns_monthly).cumprod().cummax()
drawdown_m = (1 + returns_monthly).cumprod() / cummax_m - 1
max_dd_monthly = drawdown_m.min() * 100

# Sharpe monthly
sharpe_monthly = (returns_monthly.mean() * 12 - 0.02) / (returns_monthly.std() * np.sqrt(12))

print(f"  Data points: {len(returns_monthly)} months")
print(f"\n  Final Value:    ${float(final_monthly):,.2f}")
print(f"  Total Return:   {float(total_return_monthly):.2f}%")
print(f"  CAGR:           {float(cagr_monthly):.2f}%")
print(f"  Volatility:     {float(volatility_monthly):.2f}%")
print(f"  Max Drawdown:   {float(max_dd_monthly):.2f}%")
print(f"  Sharpe Ratio:   {float(sharpe_monthly):.2f}")

print("\n" + "=" * 65)
print("EXPECTED PORTFOLIO VISUALIZER RESULTS (SPY 2019-2023)")
print("=" * 65)
print("""
From portfoliovisualizer.com (100% SPY, 2019-2023):
  Final Value:    ~$20,500 - $21,000
  CAGR:           ~15-16%
  Volatility:     ~18-20%
  Max Drawdown:   ~-23% to -24%
  Sharpe:         ~0.75 - 0.85

Note: PV uses slightly different data sources and 
      calculates some metrics differently.
""")

print("=" * 65)
print("COMPARISON SUMMARY")
print("=" * 65)
print(f"""
                    Our App (Daily)    PV-Style (Monthly)
  Final Value:      ${float(final_daily):>10,.0f}      ${float(final_monthly):>10,.0f}
  CAGR:             {float(cagr_daily):>10.2f}%      {float(cagr_monthly):>10.2f}%
  Volatility:       {float(volatility_daily):>10.2f}%      {float(volatility_monthly):>10.2f}%
  Max Drawdown:     {float(max_dd_daily):>10.2f}%      {float(max_dd_monthly):>10.2f}%
  Sharpe:           {float(sharpe_daily):>10.2f}       {float(sharpe_monthly):>10.2f}
""")
