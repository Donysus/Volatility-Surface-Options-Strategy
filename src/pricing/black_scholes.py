"""
Black-Scholes Pricing & Implied Volatility Engine

Vectorized implementation of Black-Scholes model with efficient IV solver.
"""

import warnings
from typing import Optional, Union

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton
from numba import jit, vectorize


# ==================== Black-Scholes Pricing ====================

@jit(nopython=True)
def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d1 in Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


@jit(nopython=True)
def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d2 in Black-Scholes formula."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def black_scholes_call(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate Black-Scholes call option price.
    
    Parameters:
    -----------
    S : float or array
        Spot price
    K : float or array
        Strike price
    T : float or array
        Time to maturity (in years)
    r : float or array
        Risk-free rate
    sigma : float or array
        Volatility
    q : float or array
        Dividend yield
    
    Returns:
    --------
    float or array
        Call option price
    """
    # Ensure arrays
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    # Avoid division by zero for T=0
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    # Handle T=0 case (intrinsic value)
    mask = T <= 0
    call_price = np.where(
        mask,
        np.maximum(S - K, 0),
        S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    )
    
    return call_price.item() if call_price.shape == (1,) else call_price


def black_scholes_put(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate Black-Scholes put option price.
    
    Parameters: Same as black_scholes_call
    
    Returns:
    --------
    float or array
        Put option price
    """
    # Ensure arrays
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    # Avoid division by zero for T=0
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    # Handle T=0 case (intrinsic value)
    mask = T <= 0
    put_price = np.where(
        mask,
        np.maximum(K - S, 0),
        K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    )
    
    return put_price.item() if put_price.shape == (1,) else put_price


def black_scholes_price(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str,
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate Black-Scholes option price for call or put.
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    
    Other parameters: Same as black_scholes_call
    """
    if option_type.lower() in ['call', 'c']:
        return black_scholes_call(S, K, T, r, sigma, q)
    elif option_type.lower() in ['put', 'p']:
        return black_scholes_put(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


# ==================== Vega Calculation ====================

def vega(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    q: Union[float, np.ndarray] = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate vega (derivative of option price w.r.t. volatility).
    Vega is the same for calls and puts.
    
    Returns:
    --------
    float or array
        Vega (per 1% change in volatility, so divide by 100 for per unit)
    """
    S, K, T, r, sigma, q = [np.atleast_1d(x) for x in [S, K, T, r, sigma, q]]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    vega_value = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    # Handle edge cases
    vega_value = np.where(T <= 0, 0, vega_value)
    
    return vega_value.item() if vega_value.shape == (1,) else vega_value


# ==================== Implied Volatility Solver ====================

class ImpliedVolatilitySolver:
    """
    Efficient implied volatility solver with multiple methods.
    """
    
    def __init__(
        self,
        method: str = "brentq",
        max_iter: int = 100,
        tolerance: float = 1e-6,
        bounds: tuple = (0.001, 5.0)
    ):
        """
        Initialize IV solver.
        
        Parameters:
        -----------
        method : str
            Solver method: 'brentq', 'newton', 'bisect'
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        bounds : tuple
            (min_vol, max_vol) bounds for solver
        """
        self.method = method
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bounds = bounds
    
    def solve(
        self,
        price: Union[float, np.ndarray],
        S: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: Union[float, np.ndarray],
        option_type: str,
        q: Union[float, np.ndarray] = 0.0,
        initial_guess: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Solve for implied volatility.
        
        Parameters:
        -----------
        price : float or array
            Observed option price
        S, K, T, r, q : float or array
            Black-Scholes parameters
        option_type : str
            'call' or 'put'
        initial_guess : float, optional
            Initial guess for Newton method
        
        Returns:
        --------
        float or array
            Implied volatility
        """
        # Convert to arrays for vectorization
        is_scalar = np.isscalar(price)
        price, S, K, T, r, q = [np.atleast_1d(x) for x in [price, S, K, T, r, q]]
        
        iv_results = np.full_like(price, np.nan, dtype=float)
        
        for i in range(len(price)):
            try:
                # Skip if price is invalid
                if price[i] <= 0 or T[i] <= 0:
                    continue
                
                # Check for arbitrage violations
                if option_type.lower() in ['call', 'c']:
                    intrinsic = max(S[i] - K[i], 0)
                else:
                    intrinsic = max(K[i] - S[i], 0)
                
                if price[i] < intrinsic:
                    continue  # Arbitrage - skip
                
                # Solve for IV
                if self.method == "brentq":
                    iv_results[i] = self._solve_brentq(
                        price[i], S[i], K[i], T[i], r[i], option_type, q[i]
                    )
                elif self.method == "newton":
                    iv_results[i] = self._solve_newton(
                        price[i], S[i], K[i], T[i], r[i], option_type, q[i], initial_guess
                    )
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                    
            except Exception as e:
                # Silently continue on errors
                pass
        
        return iv_results.item() if is_scalar else iv_results
    
    def _solve_brentq(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        q: float
    ) -> float:
        """Solve using Brent's method."""
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type, q) - price
        
        return brentq(
            objective,
            self.bounds[0],
            self.bounds[1],
            maxiter=self.max_iter,
            xtol=self.tolerance
        )
    
    def _solve_newton(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        q: float,
        initial_guess: Optional[float] = None
    ) -> float:
        """Solve using Newton-Raphson method."""
        if initial_guess is None:
            # Use approximation as initial guess
            initial_guess = self._initial_guess_approximation(price, S, K, T, r, option_type)
        
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type, q) - price
        
        def fprime(sigma):
            return vega(S, K, T, r, sigma, q)
        
        return newton(
            objective,
            initial_guess,
            fprime=fprime,
            maxiter=self.max_iter,
            tol=self.tolerance
        )
    
    def _initial_guess_approximation(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str
    ) -> float:
        """
        Brenner-Subrahmanyam approximation for ATM options.
        Good initial guess for Newton's method.
        """
        if abs(S - K) / S < 0.1:  # Near ATM
            return price / S * np.sqrt(2 * np.pi / T)
        else:
            return 0.3  # Default guess for non-ATM


# ==================== Vectorized IV Calculation ====================

def calculate_implied_volatility_vectorized(
    df,
    spot_column: str = 'spot',
    strike_column: str = 'strike',
    price_column: str = 'mid',
    expiry_column: str = 'expiry',
    type_column: str = 'type',
    rate_column: Optional[str] = None,
    div_column: Optional[str] = None,
    default_rate: float = 0.05,
    default_div: float = 0.0,
    current_date = None,
    method: str = "brentq"
):
    """
    Vectorized IV calculation for entire DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Option chain data
    spot_column : str
        Column name for spot price
    strike_column : str
        Column name for strike
    price_column : str
        Column name for option price
    expiry_column : str
        Column name for expiry date
    type_column : str
        Column name for option type
    rate_column : str, optional
        Column name for risk-free rate
    div_column : str, optional
        Column name for dividend yield
    default_rate : float
        Default risk-free rate if not in df
    default_div : float
        Default dividend yield if not in df
    current_date : datetime, optional
        Current date for T calculation
    method : str
        IV solver method
    
    Returns:
    --------
    pd.Series
        Implied volatilities
    """
    import pandas as pd
    from datetime import datetime
    
    if current_date is None:
        current_date = datetime.now()
    
    df = df.copy()
    
    # Calculate time to maturity
    if pd.api.types.is_datetime64_any_dtype(df[expiry_column]):
        df['T'] = (df[expiry_column] - current_date).dt.total_seconds() / (365.25 * 24 * 3600)
    else:
        df['T'] = (pd.to_datetime(df[expiry_column]) - current_date).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Get parameters
    S = df[spot_column].values if spot_column in df.columns else df['spot'].values
    K = df[strike_column].values
    T = df['T'].values
    prices = df[price_column].values
    option_types = df[type_column].values
    
    r = df[rate_column].values if rate_column and rate_column in df.columns else np.full(len(df), default_rate)
    q = df[div_column].values if div_column and div_column in df.columns else np.full(len(df), default_div)
    
    # Initialize solver
    solver = ImpliedVolatilitySolver(method=method)
    
    # Solve for each option type separately (for efficiency)
    iv_results = np.full(len(df), np.nan)
    
    # Convert option types to lowercase strings
    option_types_str = pd.Series(option_types).str.lower()
    
    for opt_type in ['call', 'put']:
        mask = (option_types_str == opt_type) | (option_types_str == opt_type[0])
        mask = mask.values  # Convert to numpy array
        
        if mask.sum() > 0:
            iv_results[mask] = solver.solve(
                prices[mask],
                S[mask],
                K[mask],
                T[mask],
                r[mask],
                opt_type,
                q[mask]
            )
    
    return pd.Series(iv_results, index=df.index, name='implied_vol')


if __name__ == "__main__":
    # Example usage
    print("Black-Scholes Pricing Engine Test")
    print("=" * 50)
    
    # Parameters
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.02
    
    # Calculate prices
    call_price = black_scholes_call(S, K, T, r, sigma, q)
    put_price = black_scholes_put(S, K, T, r, sigma, q)
    
    print(f"\nCall Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")
    
    # Test put-call parity
    parity_lhs = call_price - put_price
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check:")
    print(f"C - P = {parity_lhs:.4f}")
    print(f"S*e^(-qT) - K*e^(-rT) = {parity_rhs:.4f}")
    print(f"Difference: {abs(parity_lhs - parity_rhs):.6f}")
    
    # Test IV solver
    print("\n" + "=" * 50)
    print("Implied Volatility Solver Test")
    
    solver = ImpliedVolatilitySolver(method="brentq")
    
    # Solve for IV
    implied_vol_call = solver.solve(call_price, S, K, T, r, 'call', q)
    implied_vol_put = solver.solve(put_price, S, K, T, r, 'put', q)
    
    print(f"\nOriginal σ: {sigma:.4f}")
    print(f"Implied σ (call): {implied_vol_call:.4f}")
    print(f"Implied σ (put): {implied_vol_put:.4f}")
    print(f"Error: {abs(implied_vol_call - sigma):.6f}")
    
    # Vectorized test
    print("\n" + "=" * 50)
    print("Vectorized Calculation Test")
    
    strikes = np.array([90, 95, 100, 105, 110])
    call_prices = black_scholes_call(S, strikes, T, r, sigma, q)
    
    print(f"\nStrikes: {strikes}")
    print(f"Call Prices: {call_prices}")
    
    implied_vols = solver.solve(call_prices, S, strikes, T, r, 'call', q)
    print(f"Implied Vols: {implied_vols}")
