# Mini Portfolio Visualizer 📊

A simple Streamlit app for backtesting portfolio allocations with historical data.

## Features

- **Custom Allocations**: Enter any tickers with your desired allocation percentages
- **Historical Backtesting**: Uses Yahoo Finance (via `yfinance`)
- **Rebalancing Options**: None, monthly, quarterly, or annually
- **Benchmark Comparison**: Compare against a benchmark symbol (default: SPY)
- **Key Metrics**:
  - Total Return, **CAGR**
  - **Annual Return (mean×252)** (annualized arithmetic return)
  - Volatility, Max Drawdown
  - Sharpe Ratio, Sortino Ratio
  - **Alpha (annualized)** and **Beta** vs benchmark
- **Interactive Charts**: Portfolio vs benchmark value over time, allocation pie chart, individual asset performance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Enter tickers (comma-separated, e.g., `VTI, VXUS, BND`)
2. Set allocation percentages for each (must sum to 100%)
3. Choose initial investment amount
4. Select date range for backtesting
5. Pick rebalancing frequency
6. (Optional) set a **Benchmark Symbol** (e.g., `SPY`, `QQQ`, `VTI`)
7. Click "Run Backtest"

### Notes / gotchas

- **Effective start/end dates** depend on data availability:
  - The app drops dates with missing prices across tickers (to avoid NaNs), so the backtest starts on the first day where all tickers have data.
- **Benchmark alignment**:
  - Benchmark series is aligned to the portfolio date index.
  - Both portfolio and benchmark value series are anchored to the same `initial_investment` on the first common date.

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall percentage gain/loss over the period. |
| **CAGR** | Compound Annual Growth Rate based on start and end portfolio value. |
| **Annual Return (mean×252)** | Annualized arithmetic mean return: `mean(daily_return) * 252`. Often used for Sharpe/Sortino. |
| **Volatility** | Annualized standard deviation of daily returns: `std(daily_return) * sqrt(252)`. |
| **Max Drawdown** | Worst peak-to-trough decline in the portfolio value series. |
| **Sharpe Ratio** | `(annual_return - risk_free_rate) / volatility`. |
| **Sortino Ratio** | Like Sharpe but uses downside deviation (only negative daily returns). |
| **Alpha (ann.)** | Annualized CAPM alpha vs benchmark (uses daily rf = `rf/252`). |
| **Beta** | `cov(rp, rb) / var(rb)` vs benchmark. |

## Tech Stack

- Python 3.8+
- Streamlit
- yfinance
- pandas / numpy
- Plotly

## License

MIT
