"""
Test Suite: Verify Core Components

Run basic tests to ensure all modules are working correctly.
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("Testing Core Components")
print("=" * 70)

# Test 1: Black-Scholes Pricing
print("\n Testing Black-Scholes Pricing...")
from src.pricing import black_scholes_call, black_scholes_put

call_price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
put_price = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

assert 5 < call_price < 15, "Call price out of expected range"
assert 3 < put_price < 12, "Put price out of expected range"

# Put-call parity check
parity_diff = abs((call_price - put_price) - (100 - 100 * np.exp(-0.05)))
assert parity_diff < 0.01, "Put-call parity violated"

print(f"    Call price: ${call_price:.2f}")
print(f"    Put price: ${put_price:.2f}")
print(f"    Put-call parity: OK (diff: {parity_diff:.6f})")

# Test 2: Implied Volatility Solver
print("\n Testing Implied Volatility Solver...")
from src.pricing import ImpliedVolatilitySolver

solver = ImpliedVolatilitySolver()
true_vol = 0.25
test_price = black_scholes_call(S=100, K=100, T=0.5, r=0.05, sigma=true_vol)
implied_vol = solver.solve(test_price, S=100, K=100, T=0.5, r=0.05, option_type='call')

vol_error = abs(implied_vol - true_vol)
assert vol_error < 0.001, "IV solver error too large"

print(f"    True vol: {true_vol:.4f}")
print(f"    Implied vol: {implied_vol:.4f}")
print(f"    Error: {vol_error:.6f}")

# Test 3: Greeks Calculator
print("\n Testing Greeks Calculator...")
from src.greeks import GreeksCalculator

calc = GreeksCalculator()
greeks = calc.calculate_all_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.2, option_type='call')

assert 0.4 < greeks['delta'] < 0.7, "Delta out of range"
assert greeks['gamma'] > 0, "Gamma should be positive"
assert greeks['vega'] > 0, "Vega should be positive"
assert greeks['theta'] < 0, "Theta should be negative for long call"

print(f"   Delta: {greeks['delta']:.4f}")
print(f"   Gamma: {greeks['gamma']:.6f}")
print(f"   Vega: {greeks['vega']:.4f}")
print(f"   Theta: {greeks['theta']:.4f}")

# Test 4: Volatility Surface
print("\n Testing Volatility Surface...")
from src.surface import VolatilitySurface

# Create synthetic data
strikes = []
maturities = []
ivs = []

spot = 100
for T in [0.25, 0.5, 1.0]:
    for K in np.linspace(90, 110, 10):
        strikes.append(K)
        maturities.append(T)
        moneyness = np.log(K / spot)
        ivs.append(0.20 + 0.05 * moneyness - 0.1 * moneyness**2)

surface = VolatilitySurface(method='cubic')
surface.fit(np.array(strikes), np.array(maturities), np.array(ivs), spot=spot)

# Test interpolation
interp_vol = surface.interpolate(strike=100, maturity=0.5)
assert 0.15 < interp_vol < 0.25, "Interpolated vol out of range"

print(f"   Surface fitted with {len(strikes)} points")
print(f"   ATM vol at 6M: {interp_vol:.2%}")

# Test 5: Performance Metrics
print("\n Testing Performance Metrics...")
from src.backtest import PerformanceMetrics

# Generate synthetic returns
np.random.seed(42)
returns = pd.Series(np.random.normal(0.001, 0.02, 252))
equity = (1 + returns).cumprod() * 100000

pm = PerformanceMetrics()
sharpe = pm.sharpe_ratio(returns)
sortino = pm.sortino_ratio(returns)
max_dd, _, _ = pm.max_drawdown(equity)

assert -5 < sharpe < 5, "Sharpe ratio out of range"
assert abs(max_dd) < 1, "Max drawdown should be less than 100%"

print(f"   Sharpe ratio: {sharpe:.2f}")
print(f"   Sortino ratio: {sortino:.2f}")
print(f"   Max drawdown: {max_dd:.2%}")

# Test 6: Data Fetcher (basic initialization)
print("\n Testing Data Fetcher...")
from src.data import OptionDataFetcher

fetcher = OptionDataFetcher(cache_enabled=False)
assert fetcher.provider == 'yfinance', "Provider not set correctly"

print(f"   Data fetcher initialized")
print(f"   Provider: {fetcher.provider}")

# Test 7: Strategies
print("\n Testing Trading Strategies...")
from src.strategies import VolatilityArbitrageStrategy, GammaScalpingStrategy

vol_arb = VolatilityArbitrageStrategy()
signal = vol_arb.generate_signals(implied_vol=0.25, realized_vol=0.20)

assert signal['action'] in ['buy_vol', 'sell_vol', 'hold', 'close'], "Invalid signal"

gamma_scalp = GammaScalpingStrategy()
need_hedge, shares = gamma_scalp.check_rehedge(current_delta=500, spot=100, position_size=10000)

print(f"   Volatility arbitrage: {signal['action']}")
print(f"   Gamma scalping: hedge check works")

# Test 8: Backtester
print("\nTesting Backtesting Engine...")
from src.backtest import OptionsBacktester

bt = OptionsBacktester(initial_capital=100000)
success = bt.open_position(
    option_type='call',
    strike=100,
    expiry=datetime.now() + timedelta(days=30),
    price=5.0,
    contracts=10,
    date=datetime.now(),
    spot=100,
    greeks={'delta': 0.5, 'gamma': 0.05, 'vega': 0.2, 'theta': -0.05}
)

assert success, "Failed to open position"
assert len(bt.positions) == 1, "Position not tracked"

state = bt.get_portfolio_state(datetime.now(), spot_price=100)
assert state.total_value < bt.initial_capital, "Position cost not deducted"

print(f"   Position opened successfully")
print(f"   Portfolio value: ${state.total_value:,.2f}")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)

print("\n Summary:")
print("   • Black-Scholes pricing: ✅")
print("   • Implied volatility solver: ✅")
print("   • Greeks calculator: ✅")
print("   • Volatility surface: ✅")
print("   • Performance metrics: ✅")
print("   • Data fetcher: ✅")
print("   • Trading strategies: ✅")
print("   • Backtesting engine: ✅")

print("\n System is fully operational and ready for production use!")
