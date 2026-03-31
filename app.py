import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Mini Portfolio Visualizer", page_icon="📊", layout="wide")

st.title("📊 Mini Portfolio Visualizer")
st.markdown("Backtest your portfolio allocations with historical data")

# Sidebar inputs
st.sidebar.header("Portfolio Settings")

# Tickers input
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="VNQ, SPY, EFA, VWO, BND, GLD",
    help="Enter stock/ETF tickers separated by commas"
)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Allocations input
st.sidebar.markdown("### Allocations (%)")
allocations = []
total_alloc = 0

# Default allocations for Marc Faber Portfolio
default_allocations = {
    "VNQ": 25.0,
    "SPY": 13.0,
    "EFA": 8.0,
    "VWO": 4.0,
    "BND": 25.0,
    "GLD": 25.0
}

for ticker in tickers:
    default_val = default_allocations.get(ticker, 100.0 / len(tickers) if tickers else 0.0)
    alloc = st.sidebar.number_input(
        f"{ticker}",
        min_value=0.0,
        max_value=100.0,
        value=default_val,
        step=1.0,
        key=f"alloc_{ticker}"
    )
    allocations.append(alloc)
    total_alloc += alloc

# Show allocation total
if tickers:
    if abs(total_alloc - 100) > 0.01:
        st.sidebar.warning(f"⚠️ Allocations sum to {total_alloc:.1f}% (should be 100%)")
    else:
        st.sidebar.success(f"✅ Allocations: {total_alloc:.1f}%")

# Initial investment
initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=100,
    max_value=10000000,
    value=10000,
    step=1000
)

# Date range
col1, col2 = st.sidebar.columns(2)
default_start = datetime.now() - timedelta(days=5*365)
start_date = col1.date_input("Start Date", value=default_start)
end_date = col2.date_input("End Date", value=datetime.now())

# Rebalancing
rebalance_freq = st.sidebar.selectbox(
    "Rebalancing",
    ["None", "Monthly", "Quarterly", "Annually"],
    index=3
)

# Risk-free rate for Sharpe
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
) / 100

# Benchmark
st.sidebar.markdown("### Benchmark")
benchmark_symbol = st.sidebar.text_input(
    "Benchmark Symbol",
    value="SPY",
    help="Compare your portfolio against a benchmark (e.g., SPY, QQQ, VTI)"
)


def fetch_data(tickers, start, end):
    """Fetch adjusted close prices for tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if len(tickers) == 1:
        return data[['Close']].rename(columns={'Close': tickers[0]})
    return data['Close']


def fetch_benchmark(start, end, symbol="SPY"):
    """Fetch benchmark data."""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    # Handle both single ticker (Series) and multi-index DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close'][symbol]
    return data['Close'].squeeze()


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
    cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1/years) - 1) * 100
    
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
        "years": years
    }


# Run analysis
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    if not tickers:
        st.error("Please enter at least one ticker")
    elif abs(total_alloc - 100) > 0.01:
        st.error("Allocations must sum to 100%")
    else:
        with st.spinner("Fetching data and calculating..."):
            try:
                # Fetch data
                prices = fetch_data(tickers, start_date, end_date)
                
                if prices.empty:
                    st.error("No data found for the given tickers and date range")
                else:
                    # Drop rows with any NaN (handles different inception dates)
                    prices = prices.dropna()
                    
                    if len(prices) < 2:
                        st.error("Insufficient data for the selected date range")
                    else:
                        # Calculate portfolio
                        portfolio_value = calculate_portfolio(
                            prices, allocations, initial_investment, rebalance_freq
                        )
                        
                        # Calculate metrics
                        metrics = calculate_metrics(portfolio_value, risk_free_rate)
                        
                        # Fetch and calculate benchmark
                        benchmark_value = None
                        benchmark_metrics = None
                        if benchmark_symbol.strip():
                            try:
                                benchmark_prices = fetch_benchmark(start_date, end_date, benchmark_symbol.strip().upper())
                                # Ensure it's a Series
                                if isinstance(benchmark_prices, pd.DataFrame):
                                    benchmark_prices = benchmark_prices.iloc[:, 0]
                                # Align with portfolio dates
                                common_idx = portfolio_value.index.intersection(benchmark_prices.index)
                                if len(common_idx) > 0:
                                    benchmark_prices = benchmark_prices.loc[common_idx]
                                    # Calculate benchmark as buy-and-hold
                                    benchmark_returns = benchmark_prices.pct_change().dropna()
                                    benchmark_value = initial_investment * (1 + benchmark_returns).cumprod()
                                    benchmark_metrics = calculate_metrics(benchmark_value, risk_free_rate)
                            except Exception as e:
                                st.warning(f"Could not fetch benchmark {benchmark_symbol}: {e}")
                        
                        # Display metrics
                        st.header("📈 Results")
                        
                        # Portfolio metrics
                        st.subheader("Your Portfolio")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "Final Value",
                            f"${metrics['final_value']:,.2f}",
                            f"+${metrics['final_value'] - initial_investment:,.2f}"
                        )
                        col2.metric(
                            "Total Return",
                            f"{metrics['total_return']:.2f}%"
                        )
                        col3.metric(
                            "CAGR",
                            f"{metrics['cagr']:.2f}%"
                        )
                        col4.metric(
                            "Sharpe Ratio",
                            f"{metrics['sharpe']:.2f}"
                        )
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "Volatility",
                            f"{metrics['volatility']:.2f}%"
                        )
                        col2.metric(
                            "Max Drawdown",
                            f"{metrics['max_drawdown']:.2f}%"
                        )
                        col3.metric(
                            "Time Period",
                            f"{metrics['years']:.1f} years"
                        )
                        col4.metric(
                            "Initial Investment",
                            f"${initial_investment:,}"
                        )
                        
                        # Benchmark metrics
                        if benchmark_metrics:
                            st.subheader(f"📊 Benchmark: {benchmark_symbol.upper()}")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            # Show with delta comparison to portfolio
                            col1.metric(
                                "Final Value",
                                f"${benchmark_metrics['final_value']:,.2f}",
                                f"{metrics['final_value'] - benchmark_metrics['final_value']:+,.0f} vs benchmark",
                                delta_color="normal"
                            )
                            col2.metric(
                                "Total Return",
                                f"{benchmark_metrics['total_return']:.2f}%",
                                f"{metrics['total_return'] - benchmark_metrics['total_return']:+.2f}%",
                                delta_color="normal"
                            )
                            col3.metric(
                                "CAGR",
                                f"{benchmark_metrics['cagr']:.2f}%",
                                f"{metrics['cagr'] - benchmark_metrics['cagr']:+.2f}%",
                                delta_color="normal"
                            )
                            col4.metric(
                                "Sharpe Ratio",
                                f"{benchmark_metrics['sharpe']:.2f}",
                                f"{metrics['sharpe'] - benchmark_metrics['sharpe']:+.2f}",
                                delta_color="normal"
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric(
                                "Volatility",
                                f"{benchmark_metrics['volatility']:.2f}%",
                                f"{metrics['volatility'] - benchmark_metrics['volatility']:+.2f}%",
                                delta_color="inverse"
                            )
                            col2.metric(
                                "Max Drawdown",
                                f"{benchmark_metrics['max_drawdown']:.2f}%",
                                f"{metrics['max_drawdown'] - benchmark_metrics['max_drawdown']:+.2f}%",
                                delta_color="inverse"
                            )
                        
                        # Portfolio value chart
                        st.header("📊 Portfolio Value Over Time")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=portfolio_value.index,
                            y=portfolio_value.values,
                            mode='lines',
                            name='Portfolio',
                            line=dict(color='#00d4aa', width=2)
                        ))
                        
                        # Add benchmark line
                        if benchmark_value is not None:
                            fig.add_trace(go.Scatter(
                                x=benchmark_value.index,
                                y=benchmark_value.values,
                                mode='lines',
                                name=f'{benchmark_symbol.upper()} Benchmark',
                                line=dict(color='#ff6b6b', width=2, dash='dot')
                            ))
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Value ($)",
                            hovermode='x unified',
                            template='plotly_dark',
                            height=400,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Allocation pie chart
                        st.header("🥧 Portfolio Allocation")
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=tickers,
                            values=allocations,
                            hole=0.4
                        )])
                        fig_pie.update_layout(
                            template='plotly_dark',
                            height=300
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Individual asset performance
                        st.header("📉 Individual Asset Performance")
                        
                        normalized = prices / prices.iloc[0] * 100
                        
                        fig_assets = go.Figure()
                        for ticker in tickers:
                            fig_assets.add_trace(go.Scatter(
                                x=normalized.index,
                                y=normalized[ticker],
                                mode='lines',
                                name=ticker
                            ))
                        
                        fig_assets.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Normalized Value (Start = 100)",
                            hovermode='x unified',
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig_assets, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Enter Tickers**: Comma-separated stock/ETF symbols (e.g., VTI, VXUS, BND)
    2. **Set Allocations**: Adjust percentages for each ticker (must sum to 100%)
    3. **Initial Investment**: How much you're starting with
    4. **Date Range**: Select backtest period
    5. **Rebalancing**: How often to rebalance to target allocations
    6. **Click Run Backtest**: See your results!
    
    **Metrics Explained:**
    - **CAGR**: Compound Annual Growth Rate - your average yearly return
    - **Volatility**: How much the portfolio swings (lower = smoother ride)
    - **Max Drawdown**: Worst peak-to-trough decline
    - **Sharpe Ratio**: Risk-adjusted return (higher = better return per unit of risk)
    """)
