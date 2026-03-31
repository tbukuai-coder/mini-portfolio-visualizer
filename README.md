# Mini Portfolio Visualizer 📊

A simple Streamlit app for backtesting portfolio allocations with historical data.

## Features

- **Custom Allocations**: Enter any tickers with your desired allocation percentages
- **Historical Backtesting**: Uses Yahoo Finance data
- **Key Metrics**: Total return, CAGR, volatility, max drawdown, Sharpe ratio
- **Interactive Charts**: Portfolio value over time, allocation pie chart, individual asset performance
- **Rebalancing Options**: None, monthly, quarterly, or annually

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
6. Click "Run Backtest"

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall percentage gain/loss |
| **CAGR** | Compound Annual Growth Rate |
| **Volatility** | Annualized standard deviation of returns |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |

## Tech Stack

- Python 3.8+
- Streamlit
- yfinance
- pandas / numpy
- Plotly

## License

MIT
