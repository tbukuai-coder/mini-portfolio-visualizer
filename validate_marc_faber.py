"""
Validate against Marc Faber Portfolio from Portfolio Visualizer
URL: https://www.portfoliovisualizer.com/backtest-portfolio?s=y&benchmark=-1&benchmarkSymbol=SPY&portfolioNames=true&portfolioName1=Marc+Faber+Portfolio&symbol1=VNQ&allocation1_1=25.00&symbol2=SPY&allocation2_1=13.00&symbol3=EFA&allocation3_1=8.00&symbol4=VWO&allocation4_1=4.00&symbol5=BND&allocation5_1=25.00&symbol6=GLD&allocation6_1=25.00

Marc Faber Portfolio:
- VNQ (REITs): 25%
- SPY (US Stocks): 13%
- EFA (International Developed): 8%
- VWO (Emerging Markets): 4%
- BND (US Bonds): 25%
- GLD (Gold): 25%
"""

import yfinance as yf
import pandas as pd
import numpy as np

# Marc Faber Portfolio
TICKERS = ["VNQ", "SPY", "EFA", "VWO", "BND", "GLD"]
ALLOCATIONS = [25, 13, 8, 4, 25, 25]  # percentages
INITIAL = 10000

# Use a good date range (all these ETFs existed by 2007)
START_DATE = "2008-01-01"
END_DATE = "2024-01-01"

print("=" * 65)
print("MARC FABER PORTFOLIO VALIDATION")
print("=" * 65)
print(f"\nPortfolio:")
for t, a in zip(TICKERS, ALLOCATIONS):
    print(f"  {t}: {a}%")
print(f"\nInitial Investment: ${INITIAL:,}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Rebalancing: Annual")

print("\n" + "-" * 65)
print("Fetching data...")

# Fetch monthly data (PV uses monthly)
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, interval="1mo", auto_adjust=True, progress=False)
prices = data['Close'].dropna()

# Also fetch SPY as benchmark
spy_data = yf.download("SPY", start=START_DATE, end=END_DATE, interval="1mo", auto_adjust=True, progress=False)
spy_prices = spy_data['Close'].dropna()

print(f"Data range: {prices.index[0].strftime('%Y-%m')} to {prices.index[-1].strftime('%Y-%m')}")
print(f"Months: {len(prices)}")

# Calculate returns
returns = prices.pct_change().dropna()
spy_returns = spy_prices.pct_change().dropna()

# Align dates
common_dates = returns.index.intersection(spy_returns.index)
returns = returns.loc[common_dates]
spy_returns = spy_returns.loc[common_dates]

weights = np.array(ALLOCATIONS) / 100

# Calculate portfolio with annual rebalancing
portfolio_values = [INITIAL]
spy_values = [INITIAL]
current_weights = weights.copy()

for date, row in returns.iterrows():
    # Portfolio return
    monthly_return = (row.values * current_weights).sum()
    new_value = portfolio_values[-1] * (1 + monthly_return)
    portfolio_values.append(new_value)
    
    # SPY benchmark
    spy_ret = spy_returns.loc[date].values[0] if hasattr(spy_returns.loc[date], 'values') else spy_returns.loc[date]
    spy_values.append(spy_values[-1] * (1 + spy_ret))
    
    # Update weights (drift)
    individual_growth = 1 + row.values
    current_weights = current_weights * individual_growth
    current_weights = current_weights / current_weights.sum()
    
    # Annual rebalancing at year end
    if date.month == 12:
        current_weights = weights.copy()

# Calculate metrics
def calc_metrics(values, name):
    series = pd.Series(values)
    returns = series.pct_change().dropna()
    
    final = values[-1]
    total_return = (final / INITIAL - 1) * 100
    years = (len(values) - 1) / 12
    cagr = ((final / INITIAL) ** (1/years) - 1) * 100
    
    monthly_std = returns.std()
    volatility = monthly_std * np.sqrt(12) * 100
    
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    rf_monthly = 0.02 / 12
    excess = returns.mean() - rf_monthly
    sharpe = (excess * 12) / (monthly_std * np.sqrt(12)) if monthly_std > 0 else 0
    
    return {
        "name": name,
        "final": final,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "max_dd": max_dd,
        "sharpe": sharpe
    }

portfolio_metrics = calc_metrics(portfolio_values, "Marc Faber Portfolio")
spy_metrics = calc_metrics(spy_values, "SPY Benchmark")

print("\n" + "=" * 65)
print("OUR RESULTS")
print("=" * 65)

def print_metrics(m):
    print(f"\n{m['name']}:")
    print(f"  Final Value:    ${m['final']:,.2f}")
    print(f"  Total Return:   {m['total_return']:.2f}%")
    print(f"  CAGR:           {m['cagr']:.2f}%")
    print(f"  Volatility:     {m['volatility']:.2f}%")
    print(f"  Max Drawdown:   {m['max_dd']:.2f}%")
    print(f"  Sharpe Ratio:   {m['sharpe']:.2f}")

print_metrics(portfolio_metrics)
print_metrics(spy_metrics)

print("\n" + "=" * 65)
print("COMPARISON")
print("=" * 65)
print(f"\nMarc Faber vs SPY:")
print(f"  Return diff:    {portfolio_metrics['cagr'] - spy_metrics['cagr']:+.2f}% CAGR")
print(f"  Volatility:     {portfolio_metrics['volatility']:.1f}% vs {spy_metrics['volatility']:.1f}% (SPY)")
print(f"  Max Drawdown:   {portfolio_metrics['max_dd']:.1f}% vs {spy_metrics['max_dd']:.1f}% (SPY)")
print(f"  Sharpe:         {portfolio_metrics['sharpe']:.2f} vs {spy_metrics['sharpe']:.2f} (SPY)")

# Year by year
print("\n" + "=" * 65)
print("YEAR BY YEAR VALUES")
print("=" * 65)
print(f"\n{'Year':<6} {'Portfolio':>12} {'SPY':>12}")
print("-" * 32)

# Get year-end values
year_values = {}
for i, date in enumerate(returns.index):
    if date.month == 12:
        year_values[date.year] = (portfolio_values[i+1], spy_values[i+1])

for year in sorted(year_values.keys()):
    pv, sv = year_values[year]
    print(f"{year:<6} ${pv:>10,.0f} ${sv:>10,.0f}")

print("\n" + "=" * 65)
print("TO VERIFY ON PORTFOLIO VISUALIZER:")
print("=" * 65)
print("""
Visit the URL and compare:
https://www.portfoliovisualizer.com/backtest-portfolio?s=y&benchmark=-1&benchmarkSymbol=SPY&portfolioNames=true&portfolioName1=Marc+Faber+Portfolio&symbol1=VNQ&allocation1_1=25.00&symbol2=SPY&allocation2_1=13.00&symbol3=EFA&allocation3_1=8.00&symbol4=VWO&allocation4_1=4.00&symbol5=BND&allocation5_1=25.00&symbol6=GLD&allocation6_1=25.00

Expected differences of 1-3% are normal due to:
- Data source variations
- Dividend reinvestment timing
- Exact rebalancing dates
""")
