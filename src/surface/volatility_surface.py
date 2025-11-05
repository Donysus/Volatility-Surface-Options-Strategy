"""
Volatility Surface Construction

Build and interpolate implied volatility surfaces using various methods.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize
from typing import Optional, Tuple, Union
import warnings


class VolatilitySurface:
    """
    Volatility surface with interpolation and smoothing.
    """
    
    def __init__(self, method: str = "cubic"):
        """
        Initialize volatility surface.
        
        Parameters:
        -----------
        method : str
            Interpolation method: 'cubic', 'linear', 'rbf', 'svi'
        """
        self.method = method
        self.surface_data = None
        self.strikes = None
        self.maturities = None
        self.spot = None
        
    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        spot: float
    ):
        """
        Fit the volatility surface to market data.
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Times to maturity
        implied_vols : np.ndarray
            Implied volatilities
        spot : float
            Current spot price
        """
        self.strikes = strikes
        self.maturities = maturities
        self.implied_vols = implied_vols
        self.spot = spot
        
        # Remove NaN values
        mask = ~np.isnan(implied_vols)
        self.strikes_clean = strikes[mask]
        self.maturities_clean = maturities[mask]
        self.implied_vols_clean = implied_vols[mask]
        
    def interpolate(
        self,
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Interpolate volatility at given strike and maturity.
        
        Parameters:
        -----------
        strike : float or array
            Strike price(s)
        maturity : float or array
            Time(s) to maturity
        
        Returns:
        --------
        float or array
            Interpolated implied volatility
        """
        if self.method in ['cubic', 'linear']:
            return self._griddata_interpolate(strike, maturity)
        elif self.method == 'rbf':
            return self._rbf_interpolate(strike, maturity)
        elif self.method == 'svi':
            return self._svi_interpolate(strike, maturity)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _griddata_interpolate(
        self,
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Interpolate using scipy's griddata."""
        points = np.column_stack([self.strikes_clean, self.maturities_clean])
        values = self.implied_vols_clean
        
        strike_arr = np.atleast_1d(strike)
        maturity_arr = np.atleast_1d(maturity)
        xi = np.column_stack([strike_arr, maturity_arr])
        
        result = griddata(points, values, xi, method=self.method)
        
        return result.item() if np.isscalar(strike) and np.isscalar(maturity) else result
    
    def _rbf_interpolate(
        self,
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Interpolate using Radial Basis Functions."""
        points = np.column_stack([self.strikes_clean, self.maturities_clean])
        values = self.implied_vols_clean
        
        if not hasattr(self, '_rbf_interpolator'):
            self._rbf_interpolator = RBFInterpolator(points, values, kernel='thin_plate_spline')
        
        strike_arr = np.atleast_1d(strike)
        maturity_arr = np.atleast_1d(maturity)
        xi = np.column_stack([strike_arr, maturity_arr])
        
        result = self._rbf_interpolator(xi)
        
        return result.item() if np.isscalar(strike) and np.isscalar(maturity) else result
    
    def _svi_interpolate(
        self,
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Interpolate using SVI (Stochastic Volatility Inspired) model."""
        # Fit SVI for each maturity
        unique_maturities = np.unique(self.maturities_clean)
        svi_params = {}
        
        for T in unique_maturities:
            mask = self.maturities_clean == T
            K_slice = self.strikes_clean[mask]
            iv_slice = self.implied_vols_clean[mask]
            
            # Fit SVI parameters for this maturity
            svi_params[T] = self._fit_svi(K_slice, iv_slice, self.spot, T)
        
        # Interpolate using fitted SVI
        strike_arr = np.atleast_1d(strike)
        maturity_arr = np.atleast_1d(maturity)
        result = np.zeros(len(strike_arr))
        
        for i, (K, T) in enumerate(zip(strike_arr, maturity_arr)):
            # Find closest maturity params
            closest_T = min(svi_params.keys(), key=lambda x: abs(x - T))
            params = svi_params[closest_T]
            result[i] = self._svi_vol(K, self.spot, T, params)
        
        return result.item() if np.isscalar(strike) and np.isscalar(maturity) else result
    
    @staticmethod
    def _fit_svi(strikes, ivs, spot, maturity):
        """Fit SVI parameters: total variance w = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
        # Convert to log-moneyness
        k = np.log(strikes / spot)
        w = ivs**2 * maturity  # Total variance
        
        # Initial guess
        a0 = np.mean(w)
        b0 = 0.1
        rho0 = 0.0
        m0 = 0.0
        sigma0 = 0.1
        
        def svi_func(params):
            a, b, rho, m, sigma = params
            w_svi = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            return np.sum((w - w_svi)**2)
        
        # Constraints: -1 < rho < 1, b > 0, sigma > 0
        bounds = [(None, None), (0.001, None), (-0.999, 0.999), (None, None), (0.001, None)]
        
        try:
            result = minimize(svi_func, [a0, b0, rho0, m0, sigma0], bounds=bounds, method='L-BFGS-B')
            return result.x
        except:
            return [a0, b0, rho0, m0, sigma0]
    
    @staticmethod
    def _svi_vol(strike, spot, maturity, params):
        """Calculate SVI implied volatility."""
        a, b, rho, m, sigma = params
        k = np.log(strike / spot)
        w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        return np.sqrt(max(w / maturity, 0.001))
    
    def get_smile(self, maturity: float, strikes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility smile for a specific maturity.
        
        Parameters:
        -----------
        maturity : float
            Time to maturity
        strikes : np.ndarray, optional
            Strike prices. If None, generates strikes around spot.
        
        Returns:
        --------
        strikes, vols : tuple of arrays
            Strike prices and corresponding implied volatilities
        """
        if strikes is None:
            strikes = np.linspace(self.spot * 0.7, self.spot * 1.3, 50)
        
        maturities = np.full(len(strikes), maturity)
        vols = self.interpolate(strikes, maturities)
        
        return strikes, vols
    
    def get_term_structure(self, strike: Optional[float] = None, maturities: Optional[np.ndarray] = None):
        """
        Get volatility term structure for a specific strike (or ATM).
        
        Parameters:
        -----------
        strike : float, optional
            Strike price. If None, uses ATM.
        maturities : np.ndarray, optional
            Maturities to plot. If None, uses available maturities.
        
        Returns:
        --------
        maturities, vols : tuple of arrays
            Maturities and corresponding implied volatilities
        """
        if strike is None:
            strike = self.spot
        
        if maturities is None:
            maturities = np.unique(self.maturities_clean)
        
        strikes = np.full(len(maturities), strike)
        vols = self.interpolate(strikes, maturities)
        
        return maturities, vols
    
    def get_surface_grid(
        self,
        strike_range: Tuple[float, float] = None,
        maturity_range: Tuple[float, float] = None,
        n_strikes: int = 30,
        n_maturities: int = 20
    ):
        """
        Generate a grid for 3D surface plotting.
        
        Returns:
        --------
        K_grid, T_grid, IV_grid : tuple of 2D arrays
            Strike grid, maturity grid, and implied vol grid
        """
        if strike_range is None:
            strike_range = (self.spot * 0.7, self.spot * 1.3)
        if maturity_range is None:
            maturity_range = (self.maturities_clean.min(), self.maturities_clean.max())
        
        K_range = np.linspace(strike_range[0], strike_range[1], n_strikes)
        T_range = np.linspace(maturity_range[0], maturity_range[1], n_maturities)
        
        K_grid, T_grid = np.meshgrid(K_range, T_range)
        
        IV_grid = np.zeros_like(K_grid)
        for i in range(n_maturities):
            for j in range(n_strikes):
                IV_grid[i, j] = self.interpolate(K_grid[i, j], T_grid[i, j])
        
        return K_grid, T_grid, IV_grid


def calculate_moneyness(strikes, spot, method='log'):
    """
    Calculate moneyness.
    
    Parameters:
    -----------
    strikes : array
        Strike prices
    spot : float
        Spot price
    method : str
        'log' for log-moneyness, 'simple' for K/S, 'delta' for delta-based
    
    Returns:
    --------
    array
        Moneyness values
    """
    if method == 'log':
        return np.log(strikes / spot)
    elif method == 'simple':
        return strikes / spot
    elif method == 'delta':
        # Approximate delta-based moneyness
        return (strikes - spot) / spot
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test volatility surface
    print("Volatility Surface Test")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    spot = 100
    
    strikes = []
    maturities = []
    ivs = []
    
    for T in [0.25, 0.5, 1.0]:
        for K in np.linspace(80, 120, 15):
            strikes.append(K)
            maturities.append(T)
            # Generate synthetic smile: ATM vol 20%, skew
            moneyness = np.log(K / spot)
            base_vol = 0.20
            smile = base_vol + 0.05 * moneyness - 0.1 * moneyness**2
            ivs.append(smile + np.random.normal(0, 0.005))
    
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    ivs = np.array(ivs)
    
    # Fit surface
    surface = VolatilitySurface(method='cubic')
    surface.fit(strikes, maturities, ivs, spot)
    
    # Test interpolation
    K_test = 100
    T_test = 0.5
    iv_interp = surface.interpolate(K_test, T_test)
    print(f"\nInterpolated IV at K={K_test}, T={T_test}: {iv_interp:.4f}")
    
    # Get smile
    K_smile, iv_smile = surface.get_smile(0.5)
    print(f"\nSmile at T=0.5: {len(K_smile)} points")
    print(f"ATM vol: {surface.interpolate(spot, 0.5):.4f}")
    
    # Get term structure
    T_term, iv_term = surface.get_term_structure(strike=100)
    print(f"\nTerm structure at K=100: {len(T_term)} points")
    print(f"Maturities: {T_term}")
    print(f"IVs: {iv_term}")
