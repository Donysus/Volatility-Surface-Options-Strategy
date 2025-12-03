# Dynamic Volatility Surface & Options Strategy Simulator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Executive Summary

A full-fledged quantitative finance project demonstrating expertise in options pricing, volatility modeling, and systematic trading strategy development. The toolkit builds, visualizes, and trades on a dynamic implied volatility surface derived from real market data, integrates Greeks-based hedging, and runs full-scale backtests with realistic P&L attribution. It highlights practical applications of derivatives theory in risk management, volatility arbitrage, and market-neutral trading using robust computational finance methods.


## Business Problem

Volatility is the key driver of derivatives pricing and portfolio risk. However, traders and quants often face uncertainty due to discrepancies between implied volatility (market expectation) and realized volatility (actual outcome). This project addresses that inefficiency by:
	1.	Building a volatility surface that models implied vol across strike and maturity.
	2.	Designing strategies (like delta-hedged volatility arbitrage and gamma scalping) that exploit mispricings between IV and RV.
	3.	Backtesting these approaches under realistic trading conditions with transaction costs, slippage, and Greeks-based exposure tracking.

    
## Methodology

	•	Data Acquisition: Option chain and spot data fetched via yfinance and optionally Polygon.io. Includes caching and realized volatility estimations using Close-to-Close, Parkinson, and Garman-Klass methods.
	•	Pricing Engine: Vectorized Black–Scholes model with fast implied volatility solvers (Brent’s, Newton-Raphson), validated with put-call parity.
	•	Volatility Surface Modeling: Construction using cubic spline, RBF, and SVI interpolation with arbitrage-free smoothing.
	•	Greeks & Hedging: Full analytical Greeks calculation enabling dynamic delta-neutral hedging and exposure control.
	•	Trading & Backtesting: Strategy simulation including portfolio-level P&L attribution, transaction costs, and slippage.
	•	Visualization: Interactive 3D volatility surface plots, smiles, term structures, and P&L attributions via Plotly and Matplotlib.

    
## Technical Skills

Python, NumPy, Pandas, SciPy, Numba, Plotly, Matplotlib, QuantLib (optional), Options Theory, Risk Management, Statistical Analysis, Data Visualization, Portfolio Simulation, Backtesting Systems Design


## Results & Business Insights

	Built an end-to-end volatility surface engine supporting real-time data and robust interpolation.
	•	Simulated delta-hedged volatility arbitrage and gamma scalping strategies with full Greeks attribution.
	•	Delivered realistic P&L decomposition showing how volatility mispricing can generate excess returns under controlled risk exposure.
	•	Quantified volatility risk premium by analyzing IV–RV divergence over time.
	•	Produced interactive risk dashboards for performance, drawdown, and exposure visualization.


## Next Steps

	•	Integrate machine learning-based IV surface forecasting (e.g., LSTM or Random Forest regressors).
	•	Extend to multi-asset volatility modeling and cross-sectional arbitrage.
	•	Connect with paper-trading APIs (e.g., Alpaca, IBKR) for real-world execution testing.
	•	Incorporate stochastic volatility models (Heston, SABR) for enhanced realism.

    
## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Volatility-Surface-Options-Strategy.git
cd Volatility-Surface-Options-Strategy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (optional for basic usage)
```

### Basic Usage

```python
from src.data import OptionDataFetcher
from src.pricing import calculate_implied_volatility_vectorized
from src.surface import VolatilitySurface
from src.visualization import VolatilityVisualizer

#Fetch Data NOTE: If you need Indian Data, in requirements.txt comment out the last library, it enables the NSE stock exchange to be used
fetcher = OptionDataFetcher()
spy_chain = fetcher.get_option_chain("SPY")
spot = fetcher.get_spot_price("SPY")

#IV Calcs:
spy_chain['iv'] = calculate_implied_volatility_vectorized(spy_chain)


surface = VolatilitySurface(method='cubic')
surface.fit(
    strikes=spy_chain['strike'].values,
    maturities=spy_chain['T'].values,
    implied_vols=spy_chain['iv'].values,
    spot=spot
)

#Graphs
viz = VolatilityVisualizer(style='plotly')
K_grid, T_grid, IV_grid = surface.get_surface_grid()
fig = viz.plot_volatility_surface_3d(K_grid, T_grid, IV_grid, spot)
fig.show()
```

### Run Complete Example

See the Jupyter notebook for a comprehensive walkthrough:

```bash
jupyter notebook notebooks/volatility_surface_demo.ipynb
```

## Project Structure

```
Volatility-Surface-Options-Strategy/
├── src/
│   ├── data/               # Data fetching and processing
│   │   └── data_fetcher.py
│   ├── pricing/            # Black-Scholes and IV solver
│   │   └── black_scholes.py
│   ├── greeks/             # Greeks calculation
│   │   └── greeks_calculator.py
│   ├── surface/            # Volatility surface construction
│   │   └── volatility_surface.py
│   ├── strategies/         # Trading strategies
│   │   └── volatility_strategies.py
│   ├── backtest/           # Backtesting engine
│   │   ├── backtester.py
│   │   └── metrics.py
│   ├── visualization/      # Plotting tools
│   │   └── plots.py
│   └── utils/              # Helper functions
│       └── helpers.py
├── notebooks/              # Jupyter notebooks
│   └── volatility_surface_demo.ipynb
├── tests/                  # Unit tests
├── data/                   # Data storage (gitignored)
│   └── cache/
├── results/                # Backtest results (gitignored)
├── config.yaml             # Configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md
```

## Configuration

Edit `config.yaml` to customize:
- Data providers and caching
- IV solver parameters
- Strategy settings (thresholds, costs)
- Backtesting parameters
- Visualization preferences

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Key Concepts Demonstrated

1. **Options Pricing Theory**
   - Black-Scholes-Merton model
   - Risk-neutral valuation
   - Put-call parity

2. **Implied Volatility**
   - Market's expectation of future volatility
   - Volatility smile and skew
   - Term structure of volatility

3. **Greeks & Risk Management**
   - First-order (Delta) and second-order (Gamma) risks
   - Vega exposure and volatility trading
   - Theta decay and time value

4. **Volatility Trading**
   - Realized vs implied volatility
   - Volatility risk premium
   - Delta-neutral strategies

5. **Quantitative Analysis**
   - Time series analysis
   - Statistical arbitrage
   - Performance attribution

## Learning Resources

- **Books**:
  - "Option Volatility and Pricing" - Sheldon Natenberg
  - "The Volatility Surface" - Jim Gatheral
  - "Dynamic Hedging" - Nassim Taleb

- **Papers**:
  - "The Volatility Smile and Its Implied Tree" - Derman & Kani
  - "Stochastic Volatility Inspired (SVI) Model" - Gatheral

## Disclaimer

This project is for **educational and research purposes only**.

**If you find this project useful, please consider giving it a star!**
