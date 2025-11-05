# 🧠 Dynamic Volatility Surface & Options Strategy Simulator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for **volatility modeling**, **options pricing**, and **strategy backtesting**. This project demonstrates advanced quantitative finance concepts including implied volatility surfaces, Greeks calculation, and delta-hedged volatility trading strategies.

## 🎯 Project Overview

This is a professional-grade resume project showcasing expertise in:
- **Derivatives Pricing**: Black-Scholes model with vectorized IV solver
- **Volatility Modeling**: Surface construction using SVI/cubic interpolation
- **Quantitative Trading**: Volatility arbitrage and gamma scalping strategies
- **Risk Management**: Comprehensive Greeks calculation and delta hedging
- **Performance Analytics**: Advanced metrics including Sharpe, Sortino, and P&L attribution

## ✨ Key Features

### 1. Data Acquisition
- 📊 Real-time option chain data from **yfinance** (SPY, AAPL, etc.)
- 🔄 Support for **Polygon.io** for historical data
- 💾 Intelligent caching system for efficient backtesting
- 📈 Historical and realized volatility calculations (Close-to-Close, Parkinson, Garman-Klass)

### 2. Implied Volatility Engine
- ⚡ **Vectorized Black-Scholes pricing** with numba acceleration
- 🎯 **Root-finding IV solver** using scipy (Brent's method, Newton-Raphson)
- ✅ Put-call parity validation
- 🔧 Handles edge cases (ATM, ITM, OTM, near expiry)

### 3. Volatility Surface
- 📐 **Multiple interpolation methods**: Cubic spline, RBF, SVI model
- 🗺️ **Surface smoothing** to eliminate arbitrage violations
- 📊 **2D Volatility Smiles** by expiry
- 📈 **Term Structure** visualization (ATM vol across maturities)
- 🌐 **3D Surface plots** (Strike × Maturity × IV)

### 4. Greeks Calculator
- 🔢 **Analytical Greeks**: Delta, Gamma, Vega, Theta, Rho
- ⚖️ **Delta-neutral hedging** calculations
- 📊 **P&L attribution** (Theta P&L, Gamma P&L, Vega P&L)
- 🎯 Position-level and portfolio-level aggregation

### 5. Trading Strategies

#### Volatility Arbitrage
- **Logic**: Long underpriced options (IV < RV), short overpriced (IV > RV)
- **Entry**: |IV - RV| > threshold (default 3%)
- **Exit**: Spread converges or position expires
- **Hedging**: Delta-neutral via dynamic rebalancing

#### Gamma Scalping
- **Logic**: Long gamma, profit from realized volatility
- **Mechanism**: Buy ATM straddles, hedge delta dynamically
- **P&L Source**: Gamma gains from spot moves > theta decay
- **Rebalancing**: Triggered by delta threshold

### 6. Backtesting Engine
- 📊 Full **portfolio simulation** with position tracking
- 💰 **Transaction costs** and **slippage** modeling
- 📈 **Mark-to-market** P&L with Greeks evolution
- 🎯 **Trade logging** and audit trail

### 7. Performance Metrics
- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Drawdown Analysis**: Maximum drawdown, recovery periods
- **Distribution Metrics**: VaR, CVaR, Omega ratio, tail ratio
- **Greeks P&L Attribution**: Theta vs Gamma vs Vega contributions
- **Volatility Analysis**: IV-RV correlation and forecast accuracy

### 8. Visualization Suite
- 📊 **Interactive Plotly** charts (3D surfaces, time series)
- 📈 **Matplotlib** for publication-quality static plots
- 🎨 Volatility smile evolution over time
- 💹 P&L curves with attribution breakdown
- 📉 Drawdown visualization
- 🎯 Greeks evolution heatmaps

## 🚀 Quick Start

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

# 1. Fetch option data
fetcher = OptionDataFetcher()
spy_chain = fetcher.get_option_chain("SPY")
spot = fetcher.get_spot_price("SPY")

# 2. Calculate implied volatilities
spy_chain['iv'] = calculate_implied_volatility_vectorized(spy_chain)

# 3. Build volatility surface
surface = VolatilitySurface(method='cubic')
surface.fit(
    strikes=spy_chain['strike'].values,
    maturities=spy_chain['T'].values,
    implied_vols=spy_chain['iv'].values,
    spot=spot
)

# 4. Visualize
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

## 📁 Project Structure

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

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data providers and caching
- IV solver parameters
- Strategy settings (thresholds, costs)
- Backtesting parameters
- Visualization preferences

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📚 Key Concepts Demonstrated

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

## 🎓 Learning Resources

- **Books**:
  - "Option Volatility and Pricing" - Sheldon Natenberg
  - "The Volatility Surface" - Jim Gatheral
  - "Dynamic Hedging" - Nassim Taleb

- **Papers**:
  - "The Volatility Smile and Its Implied Tree" - Derman & Kani
  - "Stochastic Volatility Inspired (SVI) Model" - Gatheral

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended as financial advice or for use in live trading without thorough testing and risk management. Options trading involves significant risk and is not suitable for all investors.

---

**⭐ If you find this project useful, please consider giving it a star!**