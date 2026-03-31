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
# Slider gives a nicer UX than freeform input.
risk_free_rate_pct = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Used in Sharpe ratio: (annual_return - risk_free_rate) / volatility",
)
risk_free_rate = risk_free_rate_pct / 100

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
        # Anchor the series at initial_investment on the first available price date
        norm_prices = prices / prices.iloc[0]
        portfolio_value = initial_investment * (norm_prices * weights).sum(axis=1)
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


def calculate_metrics(portfolio_value, risk_free_rate, benchmark_value=None):
    """Calculate portfolio performance metrics.

    Notes on definitions (aligned with common portfolio analytics libraries):

    - Total Return (%): (final / initial - 1) * 100
    - CAGR (%): ((final / initial) ** (1 / years) - 1) * 100
    - Annualized Mean Return (%): mean(daily_return) * 252 * 100
      (This is the same definition used in many libraries for Sharpe.)
    - Volatility (%): std(daily_return) * sqrt(252) * 100

    Where daily_return is computed from the portfolio value series.
    """

    returns = portfolio_value.pct_change().dropna()

    # Alpha/Beta vs benchmark (optional)
    alpha_ann = None
    beta = None
    if benchmark_value is not None:
        bench_ret = benchmark_value.pct_change().dropna()
        common = returns.index.intersection(bench_ret.index)
        if len(common) >= 3:
            rp = returns.loc[common]
            rb = bench_ret.loc[common]

            # Convert annual risk-free rate to daily (simple approximation)
            rf_daily = risk_free_rate / 252

            var_b = float(rb.var())
            if var_b > 0:
                beta = float(rp.cov(rb) / var_b)

                # CAPM-style annualized alpha
                alpha_daily = float((rp - rf_daily).mean() - beta * (rb - rf_daily).mean())
                alpha_ann = alpha_daily * 252 * 100  # percent

    initial_value = float(portfolio_value.iloc[0])
    final_value = float(portfolio_value.iloc[-1])

    total_return = (final_value / initial_value - 1) * 100

    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100

    annual_return_mean = returns.mean() * 252  # decimal
    annual_return_mean_pct = float(annual_return_mean) * 100

    volatility = float(returns.std() * np.sqrt(252) * 100)

    # Max drawdown
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min() * 100)

    # Sharpe ratio (uses annualized mean return)
    sharpe = (
        float((annual_return_mean - risk_free_rate) / (returns.std() * np.sqrt(252)))
        if returns.std() > 0
        else 0.0
    )

    # Sortino ratio: use downside deviation (only negative returns)
    downside = returns[returns < 0]
    downside_deviation = downside.std() * np.sqrt(252)
    sortino = (
        float((annual_return_mean - risk_free_rate) / downside_deviation)
        if downside_deviation > 0
        else 0.0
    )

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_return_mean": annual_return_mean_pct,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "initial_value": initial_value,
        "final_value": final_value,
        "years": float(years),
        "alpha_ann": alpha_ann,
        "beta": beta,
        # Human-readable formulas for transparency / UI display
        "formulas": {
            "total_return": "(final / initial - 1) * 100",
            "cagr": "((final / initial) ** (1 / years) - 1) * 100",
            "annual_return_mean": "mean(daily_return) * 252 * 100",
            "volatility": "std(daily_return) * sqrt(252) * 100",
            "max_drawdown": "min((value - rolling_max) / rolling_max) * 100",
            "sharpe": "(mean(daily_return) * 252 - risk_free_rate) / (std(daily_return) * sqrt(252))",
            "sortino": "(mean(daily_return) * 252 - risk_free_rate) / (std(daily_return[daily_return<0]) * sqrt(252))",
            "beta": "cov(rp, rb) / var(rb)",
            "alpha_ann": "(((rp - rf_d).mean() - beta * (rb - rf_d).mean()) * 252) * 100",
        },
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
                                    # Calculate benchmark as buy-and-hold, starting exactly at initial_investment
                                    # (Using prices/first_price keeps the series anchored on the first common date.)
                                    benchmark_value = initial_investment * (benchmark_prices / benchmark_prices.iloc[0])
                                    benchmark_metrics = calculate_metrics(benchmark_value, risk_free_rate)
                            except Exception as e:
                                st.warning(f"Could not fetch benchmark {benchmark_symbol}: {e}")

                        # Calculate metrics (pass benchmark series so alpha/beta can be computed)
                        metrics = calculate_metrics(portfolio_value, risk_free_rate, benchmark_value)
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
                                    # Calculate benchmark as buy-and-hold, starting exactly at initial_investment
                                    # (Using prices/first_price keeps the series anchored on the first common date.)
                                    benchmark_value = initial_investment * (benchmark_prices / benchmark_prices.iloc[0])
                                    benchmark_metrics = calculate_metrics(benchmark_value, risk_free_rate)
                            except Exception as e:
                                st.warning(f"Could not fetch benchmark {benchmark_symbol}: {e}")
                        
                        # Display metrics
                        st.header("📈 Results")

                        # Side-by-side rows: each metric is one row with 2 columns
                        st.subheader("📌 Portfolio vs Benchmark")

                        def _fmt_delta(delta_str: str) -> str:
                            # User requested: no brackets. Keep leading sign.
                            return delta_str

                        has_bench = benchmark_metrics is not None

                        def row(label: str, p_value: str, p_delta: str | None, b_value: str | None, help_text: str | None = None):
                            lcol, rcol = st.columns(2)
                            with lcol:
                                st.metric(label, p_value, _fmt_delta(p_delta) if (has_bench and p_delta) else None, help=help_text)
                            with rcol:
                                if b_value is not None:
                                    st.metric(label, b_value)
                                else:
                                    st.metric(label, "—")

                        # Compute deltas (portfolio - benchmark)
                        fv_delta = metrics["final_value"] - benchmark_metrics["final_value"] if has_bench else None
                        tr_delta = metrics["total_return"] - benchmark_metrics["total_return"] if has_bench else None
                        cagr_delta = metrics["cagr"] - benchmark_metrics["cagr"] if has_bench else None
                        annret_delta = metrics["annual_return_mean"] - benchmark_metrics["annual_return_mean"] if has_bench else None
                        vol_delta = metrics["volatility"] - benchmark_metrics["volatility"] if has_bench else None
                        mdd_delta = metrics["max_drawdown"] - benchmark_metrics["max_drawdown"] if has_bench else None
                        sharpe_delta = metrics["sharpe"] - benchmark_metrics["sharpe"] if has_bench else None
                        sortino_delta = metrics["sortino"] - benchmark_metrics.get("sortino", 0.0) if has_bench else None

                        row(
                            "Final Value",
                            f"${metrics['final_value']:,.2f}",
                            f"{fv_delta:+,.0f}" if has_bench else f"+${metrics['final_value'] - initial_investment:,.2f}",
                            f"${benchmark_metrics['final_value']:,.2f}" if has_bench else None,
                        )
                        row(
                            "Total Return",
                            f"{metrics['total_return']:.2f}%",
                            f"{tr_delta:+.2f}%" if has_bench else None,
                            f"{benchmark_metrics['total_return']:.2f}%" if has_bench else None,
                        )
                        row(
                            "CAGR",
                            f"{metrics['cagr']:.2f}%",
                            f"{cagr_delta:+.2f}%" if has_bench else None,
                            f"{benchmark_metrics['cagr']:.2f}%" if has_bench else None,
                        )
                        row(
                            "Annual Return (mean×252)",
                            f"{metrics['annual_return_mean']:.2f}%",
                            f"{annret_delta:+.2f}%" if has_bench else None,
                            f"{benchmark_metrics['annual_return_mean']:.2f}%" if has_bench else None,
                            help_text="Annualized arithmetic return = mean(daily_return) * 252",
                        )
                        row(
                            "Volatility",
                            f"{metrics['volatility']:.2f}%",
                            f"{vol_delta:+.2f}%" if has_bench else None,
                            f"{benchmark_metrics['volatility']:.2f}%" if has_bench else None,
                        )
                        row(
                            "Max Drawdown",
                            f"{metrics['max_drawdown']:.2f}%",
                            f"{mdd_delta:+.2f}%" if has_bench else None,
                            f"{benchmark_metrics['max_drawdown']:.2f}%" if has_bench else None,
                        )
                        row(
                            "Sharpe Ratio",
                            f"{metrics['sharpe']:.2f}",
                            f"{sharpe_delta:+.2f}" if has_bench else None,
                            f"{benchmark_metrics['sharpe']:.2f}" if has_bench else None,
                        )
                        row(
                            "Sortino Ratio",
                            f"{metrics['sortino']:.2f}",
                            f"{sortino_delta:+.2f}" if has_bench else None,
                            f"{benchmark_metrics.get('sortino', 0.0):.2f}" if has_bench else None,
                            help_text="Sharpe-like ratio using downside deviation (only negative returns)",
                        )

                        # Alpha/Beta only make sense when benchmark exists
                        if has_bench:
                            alpha_str = "—" if metrics.get("alpha_ann") is None else f"{metrics['alpha_ann']:.2f}%"
                            beta_str = "—" if metrics.get("beta") is None else f"{metrics['beta']:.2f}"

                            row(
                                "Alpha (ann.)",
                                alpha_str,
                                None,
                                "0.00%",
                                help_text="Annualized CAPM alpha vs benchmark (uses rf/252)",
                            )
                            row(
                                "Beta",
                                beta_str,
                                None,
                                "1.00",
                                help_text="Beta vs benchmark = cov(rp, rb) / var(rb)",
                            )

                        # Time period row (no delta)
                        row(
                            "Time Period",
                            f"{metrics['years']:.1f} years",
                            None,
                            f"{benchmark_metrics['years']:.1f} years" if has_bench else None,
                        )

                        # Show formulas (transparency / reference)
                        with st.expander("🧮 Metric formulas"):
                            st.code(
                                "\n".join(
                                    [
                                        f"Total Return (%):        {metrics['formulas']['total_return']}",
                                        f"CAGR (%):               {metrics['formulas']['cagr']}",
                                        f"Annual Return (%):      {metrics['formulas']['annual_return_mean']}",
                                        f"Volatility (%):         {metrics['formulas']['volatility']}",
                                        f"Max Drawdown (%):       {metrics['formulas']['max_drawdown']}",
                                        f"Sharpe Ratio:           {metrics['formulas']['sharpe']}",
                                        f"Sortino Ratio:          {metrics['formulas']['sortino']}",
                                        f"Beta:                  {metrics['formulas']['beta']}",
                                        f"Alpha (ann.):          {metrics['formulas']['alpha_ann']}",
                                    ]
                                ),
                                language="text",
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
