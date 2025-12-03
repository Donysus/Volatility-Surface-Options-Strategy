"""
Quick Start Example: Volatility Surface Analysis

Run this script to see a complete workflow demonstration.
"""

import sys
import os
# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import numpy as np
import pandas as pd

from src.data import OptionDataFetcher
from src.pricing import calculate_implied_volatility_vectorized
from src.surface import VolatilitySurface
from src.greeks import calculate_greeks_dataframe
from src.visualization import VolatilityVisualizer

def main():
    print("=" * 70)
    print(" Dynamic Volatility Surface & Options Strategy Simulator")
    print("=" * 70)
    
    # Step 1: Fetch data
    print("\n📊 Step 1: Fetching SPY option chain...")
    fetcher = OptionDataFetcher()
    spy_chain = fetcher.get_option_chain("SPY")
    spot = fetcher.get_spot_price("SPY")
    rf_rate = fetcher.get_risk_free_rate()
    
    print(f"    Fetched {len(spy_chain):,} contracts")
    print(f"    Spot: ${spot:.2f}")
    print(f"    Risk-free rate: {rf_rate:.2%}")
    
    # Step 2: Clean data
    print("\n🧹 Step 2: Cleaning data...")
    spy_clean = fetcher.clean_option_chain(
        spy_chain, 
        min_volume=10, 
        min_oi=50
    )
    print(f"   {len(spy_clean):,} contracts after cleaning")
    
    # Step 3: Calculate time to maturity
    print("\n⏱️  Step 3: Calculating time to maturity...")
    current_date = pd.Timestamp.now()
    spy_clean['T'] = (spy_clean['expiry'] - current_date).dt.total_seconds() / (365.25 * 24 * 3600)
    spy_clean = spy_clean[spy_clean['T'] > 0]
    
    # Step 4: Calculate implied volatility
    print("\n🎯 Step 4: Calculating implied volatility...")
    spy_clean['spot'] = spot
    spy_clean['r'] = rf_rate
    spy_clean['iv_calculated'] = calculate_implied_volatility_vectorized(
        spy_clean,
        current_date=current_date,
        default_rate=rf_rate
    )
    
    # Use calculated IV or market IV
    spy_clean['iv_final'] = spy_clean['iv_calculated'].fillna(spy_clean['iv'])
    valid_iv = spy_clean['iv_final'].notna()
    print(f"   Calculated IV for {valid_iv.sum():,} contracts")
    
    # Step 5: Build volatility surface
    print("\n🗺️  Step 5: Building volatility surface...")
    surface = VolatilitySurface(method='cubic')
    spy_surface = spy_clean[valid_iv].copy()
    
    surface.fit(
        strikes=spy_surface['strike'].values,
        maturities=spy_surface['T'].values,
        implied_vols=spy_surface['iv_final'].values,
        spot=spot
    )
    print("    Surface fitted successfully")
    
    # Step 6: Calculate Greeks
    print("\n📐 Step 6: Calculating Greeks...")
    spy_surface['sigma'] = spy_surface['iv_final']
    spy_greeks = calculate_greeks_dataframe(spy_surface)
    print("   Greeks calculated")
    
    # Step 7: Analyze specific expiry
    print("\n📊 Step 7: Analyzing nearest expiry...")
    nearest_expiry = spy_surface['expiry'].min()
    near_exp_data = spy_greeks[spy_greeks['expiry'] == nearest_expiry].copy()
    
    print(f"   Expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
    print(f"   Days to expiry: {(nearest_expiry - current_date).days}")
    print(f"   Contracts: {len(near_exp_data)}")
    
    # ATM analysis
    if len(near_exp_data) > 0:
        atm_idx = (near_exp_data['strike'] - spot).abs().argsort().iloc[0]
        atm_strike = near_exp_data.loc[near_exp_data.index[atm_idx]]
        print(f"\n   ATM Strike: ${atm_strike['strike']:.2f}")
        if 'type' in atm_strike and atm_strike['type'] == 'call':
            print(f"   ATM Call IV: {atm_strike['iv_final']:.2%}")
    
    # Step 8: Visualization
    print("\n📈 Step 8: Creating visualizations...")
    viz = VolatilityVisualizer(style='plotly')
    
    # Get smile
    T_nearest = spy_surface[spy_surface['expiry'] == nearest_expiry]['T'].iloc[0]
    K_smile, iv_smile = surface.get_smile(T_nearest)
    
    print("    Volatility smile extracted")
    
    # Get term structure  
    T_term, iv_term = surface.get_term_structure(strike=spot)
    print("    Term structure extracted")
    
    # Get 3D surface
    K_grid, T_grid, IV_grid = surface.get_surface_grid()
    print("    3D surface grid created")
    
    # Step 9: Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nVolatility Statistics:")
    print(f"   ATM IV (nearest): {spy_surface[spy_surface['expiry'] == nearest_expiry]['iv_final'].mean():.2%}")
    print(f"   Overall IV range: {spy_surface['iv_final'].min():.2%} - {spy_surface['iv_final'].max():.2%}")
    print(f"   IV std dev: {spy_surface['iv_final'].std():.2%}")
    
    print(f"\nGreeks Summary (nearest expiry):")
    calls = near_exp_data[near_exp_data['type'] == 'call']
    puts = near_exp_data[near_exp_data['type'] == 'put']
    
    if len(calls) > 0:
        print(f"   Calls:")
        print(f"      Avg Delta: {calls['delta'].mean():.4f}")
        print(f"      Avg Gamma: {calls['gamma'].mean():.6f}")
        print(f"      Avg Vega: {calls['vega'].mean():.4f}")
    
    if len(puts) > 0:
        print(f"   Puts:")
        print(f"      Avg Delta: {puts['delta'].mean():.4f}")
        print(f"      Avg Gamma: {puts['gamma'].mean():.6f}")
        print(f"      Avg Vega: {puts['vega'].mean():.4f}")
    
    print("\n" + "=" * 70)
    print(" Analysis complete!")
    print("=" * 70)
    
    print("\n Next steps:")
    print("   1. Open the Jupyter notebook: notebooks/volatility_surface_demo.ipynb")
    print("   2. Explore interactive visualizations")
    print("   3. Backtest trading strategies")
    print("   4. Review performance metrics")
    
    return {
        'spot': spot,
        'data': spy_greeks,
        'surface': surface,
        'risk_free_rate': rf_rate
    }


if __name__ == "__main__":
    results = main()
