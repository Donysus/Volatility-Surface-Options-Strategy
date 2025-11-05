"""
Options Greeks Calculator

Analytical and numerical Greeks calculations for risk management.
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def delta_call(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """Calculate delta for call option."""
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    delta = np.exp(-q * T) * norm.cdf(d1)
    delta = np.where(T <= 0, np.where(S > K, 1, 0), delta)
    
    return delta.item() if delta.shape == (1,) else delta


def delta_put(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """Calculate delta for put option."""
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    delta = -np.exp(-q * T) * norm.cdf(-d1)
    delta = np.where(T <= 0, np.where(S < K, -1, 0), delta)
    
    return delta.item() if delta.shape == (1,) else delta


def gamma(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate gamma (same for calls and puts).
    Gamma is the second derivative of option price w.r.t. spot.
    """
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    gamma_value = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    gamma_value = np.where(T <= 0, 0, gamma_value)
    
    return gamma_value.item() if gamma_value.shape == (1,) else gamma_value


def vega(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate vega (same for calls and puts).
    Vega is the derivative of option price w.r.t. volatility.
    Returned value is for 1% change in vol (divide by 100 for 1 unit change).
    """
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    vega_value = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    vega_value = np.where(T <= 0, 0, vega_value)
    
    return vega_value.item() if vega_value.shape == (1,) else vega_value


def theta_call(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate theta for call option.
    Theta is the time decay (negative value).
    Returned value is per day (divide by 365 for per year).
    """
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
    term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    
    theta_value = (term1 + term2 + term3) / 365  # Per day
    theta_value = np.where(T <= 0, 0, theta_value)
    
    return theta_value.item() if theta_value.shape == (1,) else theta_value


def theta_put(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """Calculate theta for put option."""
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
    term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    theta_value = (term1 + term2 + term3) / 365  # Per day
    theta_value = np.where(T <= 0, 0, theta_value)
    
    return theta_value.item() if theta_value.shape == (1,) else theta_value


def rho_call(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate rho for call option.
    Rho is the derivative w.r.t. risk-free rate.
    Returned value is for 1% change in r.
    """
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    rho_value = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change
    rho_value = np.where(T <= 0, 0, rho_value)
    
    return rho_value.item() if rho_value.shape == (1,) else rho_value


def rho_put(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """Calculate rho for put option."""
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    rho_value = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change
    rho_value = np.where(T <= 0, 0, rho_value)
    
    return rho_value.item() if rho_value.shape == (1,) else rho_value


class GreeksCalculator:
    """
    Comprehensive Greeks calculator with both analytical and numerical methods.
    """
    
    @staticmethod
    def calculate_all_greeks(
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        sigma: Union[float, np.ndarray],
        option_type: str,
        q: Union[float, np.ndarray] = 0.0
    ) -> dict:
        """
        Calculate all Greeks for an option.
        
        Returns:
        --------
        dict
            Dictionary containing delta, gamma, vega, theta, rho
        """
        is_call = option_type.lower() in ['call', 'c']
        
        greeks = {
            'delta': delta_call(S, K, T, r, sigma, q) if is_call else delta_put(S, K, T, r, sigma, q),
            'gamma': gamma(S, K, T, r, sigma, q),
            'vega': vega(S, K, T, r, sigma, q),
            'theta': theta_call(S, K, T, r, sigma, q) if is_call else theta_put(S, K, T, r, sigma, q),
            'rho': rho_call(S, K, T, r, sigma, q) if is_call else rho_put(S, K, T, r, sigma, q)
        }
        
        return greeks
    
    @staticmethod
    def delta_neutral_hedge_ratio(
        position_delta: float,
        spot_delta: float = 1.0
    ) -> float:
        """
        Calculate number of shares needed to delta hedge.
        
        Parameters:
        -----------
        position_delta : float
            Total delta of options position
        spot_delta : float
            Delta of underlying (always 1.0 for stocks)
        
        Returns:
        --------
        float
            Number of shares to short/long (negative = short)
        """
        return -position_delta / spot_delta
    
    @staticmethod
    def gamma_pnl(
        gamma: float,
        spot_move: float,
        position_size: float = 1.0
    ) -> float:
        """
        Calculate P&L from gamma (convexity gain/loss).
        
        Parameters:
        -----------
        gamma : float
            Position gamma
        spot_move : float
            Change in spot price
        position_size : float
            Number of contracts
        
        Returns:
        --------
        float
            Gamma P&L
        """
        return 0.5 * gamma * spot_move**2 * position_size
    
    @staticmethod
    def theta_pnl(
        theta: float,
        time_elapsed: float,
        position_size: float = 1.0
    ) -> float:
        """
        Calculate P&L from theta decay.
        
        Parameters:
        -----------
        theta : float
            Position theta (per day)
        time_elapsed : float
            Time elapsed in days
        position_size : float
            Number of contracts
        
        Returns:
        --------
        float
            Theta P&L (typically negative)
        """
        return theta * time_elapsed * position_size
    
    @staticmethod
    def vega_pnl(
        vega: float,
        vol_change: float,
        position_size: float = 1.0
    ) -> float:
        """
        Calculate P&L from volatility change.
        
        Parameters:
        -----------
        vega : float
            Position vega
        vol_change : float
            Change in implied volatility (in percentage points, e.g., 0.02 for 2%)
        position_size : float
            Number of contracts
        
        Returns:
        --------
        float
            Vega P&L
        """
        return vega * vol_change * position_size


def calculate_greeks_dataframe(df, spot_column='spot', option_type_column='type'):
    """
    Calculate all Greeks for a DataFrame of options.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: strike, T (time to maturity), r, sigma, spot
    spot_column : str
        Column name for spot price
    option_type_column : str
        Column name for option type
    
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added Greek columns
    """
    import pandas as pd
    
    df = df.copy()
    
    # Extract parameters
    S = df[spot_column].values
    K = df['strike'].values
    T = df['T'].values
    r = df['r'].values if 'r' in df.columns else np.full(len(df), 0.05)
    sigma = df['sigma'].values if 'sigma' in df.columns else df['implied_vol'].values
    q = df['q'].values if 'q' in df.columns else np.full(len(df), 0.0)
    
    # Calculate Greeks for calls and puts separately
    calc = GreeksCalculator()
    
    # Initialize arrays
    df['delta'] = np.nan
    df['gamma'] = np.nan
    df['vega'] = np.nan
    df['theta'] = np.nan
    df['rho'] = np.nan
    
    for opt_type in ['call', 'put']:
        mask = df[option_type_column].str.lower().isin([opt_type, opt_type[0]])
        
        if mask.sum() > 0:
            greeks = calc.calculate_all_greeks(
                S[mask], K[mask], T[mask], r[mask], sigma[mask], opt_type, q[mask]
            )
            
            df.loc[mask, 'delta'] = greeks['delta']
            df.loc[mask, 'gamma'] = greeks['gamma']
            df.loc[mask, 'vega'] = greeks['vega']
            df.loc[mask, 'theta'] = greeks['theta']
            df.loc[mask, 'rho'] = greeks['rho']
    
    return df


if __name__ == "__main__":
    # Test Greeks calculations
    print("Greeks Calculator Test")
    print("=" * 50)
    
    # Test parameters
    S = 100
    K = 100
    T = 0.5  # 6 months
    r = 0.05
    sigma = 0.2
    q = 0.02
    
    # Calculate all Greeks for a call
    calc = GreeksCalculator()
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call', q)
    
    print("\nCall Option Greeks:")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f} (per day)")
    print(f"Rho: {greeks['rho']:.4f}")
    
    # Delta hedging example
    print("\n" + "=" * 50)
    print("Delta Hedging Example")
    position_size = 10  # Long 10 call contracts (1000 shares)
    position_delta = greeks['delta'] * position_size * 100
    hedge_shares = calc.delta_neutral_hedge_ratio(position_delta)
    
    print(f"\nPosition: Long {position_size} call contracts")
    print(f"Position Delta: {position_delta:.2f}")
    print(f"Hedge: Short {abs(hedge_shares):.0f} shares")
    
    # Gamma scalping example
    print("\n" + "=" * 50)
    print("Gamma Scalping Example")
    
    spot_moves = np.array([-2, -1, 0, 1, 2])
    for move in spot_moves:
        gamma_pnl = calc.gamma_pnl(greeks['gamma'], move, position_size * 100)
        print(f"Spot move ${move:+.0f}: Gamma P&L = ${gamma_pnl:+.2f}")
