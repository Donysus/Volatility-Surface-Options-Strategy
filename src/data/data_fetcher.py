"""
Data Fetching Module

Fetches option chain data from multiple sources (yfinance, polygon.io)
with caching support for efficient backtesting.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    warnings.warn("polygon-api-client not installed. Polygon.io features disabled.")


class OptionDataFetcher:
    """
    Fetches and caches option chain data from multiple sources.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        cache_enabled: bool = True,
        provider: str = "yfinance"
    ):
        """
        Initialize the data fetcher.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached data
        cache_enabled : bool
            Whether to use caching
        provider : str
            Data provider: 'yfinance' or 'polygon'
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.provider = provider
        
        # Load environment variables
        load_dotenv()
        
        # Initialize polygon client if needed
        if provider == "polygon" and POLYGON_AVAILABLE:
            api_key = os.getenv("POLYGON_API_KEY")
            if api_key:
                self.polygon_client = RESTClient(api_key)
            else:
                warnings.warn("POLYGON_API_KEY not found. Falling back to yfinance.")
                self.provider = "yfinance"
    
    def get_option_chain(
        self,
        ticker: str,
        date: Optional[datetime] = None,
        use_cache: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Fetch option chain for a given ticker and date.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'SPY', 'AAPL')
        date : datetime, optional
            Date for historical data. If None, fetches current data.
        use_cache : bool, optional
            Override cache setting for this call
        
        Returns:
        --------
        pd.DataFrame
            Option chain with columns: strike, expiry, type, bid, ask, last, volume, oi, iv
        """
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(ticker, date)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch data based on provider
        if self.provider == "yfinance":
            data = self._fetch_yfinance(ticker, date)
        elif self.provider == "polygon":
            data = self._fetch_polygon(ticker, date)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache the data
        if use_cache and data is not None:
            self._save_to_cache(cache_key, data)
        
        return data
    
    def _fetch_yfinance(
        self,
        ticker: str,
        date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch option chain using yfinance."""
        if date is not None:
            warnings.warn("yfinance doesn't support historical option chains. Using current data.")
        
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            all_options = []
            
            for expiry in expirations:
                opt_chain = stock.option_chain(expiry)
                
                # Process calls
                calls = opt_chain.calls.copy()
                calls['type'] = 'call'
                calls['expiry'] = pd.to_datetime(expiry)
                
                # Process puts
                puts = opt_chain.puts.copy()
                puts['type'] = 'put'
                puts['expiry'] = pd.to_datetime(expiry)
                
                all_options.append(calls)
                all_options.append(puts)
            
            if not all_options:
                return pd.DataFrame()
            
            df = pd.concat(all_options, ignore_index=True)
            
            # Standardize column names
            df = df.rename(columns={
                'impliedVolatility': 'iv',
                'openInterest': 'oi',
                'lastPrice': 'last'
            })
            
            # Select relevant columns
            columns = ['strike', 'expiry', 'type', 'bid', 'ask', 'last', 'volume', 'oi', 'iv']
            df = df[[col for col in columns if col in df.columns]]
            
            # Calculate mid price
            df['mid'] = (df['bid'] + df['ask']) / 2
            df.loc[df['mid'] == 0, 'mid'] = df.loc[df['mid'] == 0, 'last']
            
            # Add fetch timestamp
            df['fetch_time'] = datetime.now()
            
            return df
            
        except Exception as e:
            warnings.warn(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()
    
    def _fetch_polygon(
        self,
        ticker: str,
        date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch option chain using Polygon.io."""
        if not POLYGON_AVAILABLE:
            raise ImportError("polygon-api-client not installed")
        
        # Implementation for polygon.io
        # This is a placeholder - polygon requires more complex logic
        warnings.warn("Polygon.io implementation pending. Using yfinance.")
        return self._fetch_yfinance(ticker, date)
    
    def get_spot_price(
        self,
        ticker: str,
        date: Optional[datetime] = None
    ) -> float:
        """
        Get spot price for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        date : datetime, optional
            Date for historical price
        
        Returns:
        --------
        float
            Spot price
        """
        stock = yf.Ticker(ticker)
        
        if date is None:
            # Get current price
            info = stock.info
            return info.get('currentPrice', info.get('regularMarketPrice', np.nan))
        else:
            # Get historical price
            hist = stock.history(start=date, end=date + timedelta(days=1))
            if not hist.empty:
                return hist['Close'].iloc[0]
            return np.nan
    
    def get_dividend_yield(
        self,
        ticker: str
    ) -> float:
        """Get dividend yield for a ticker."""
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('dividendYield', 0.0) or 0.0
    
    def get_risk_free_rate(
        self,
        maturity: str = "^TNX"  # 10-year treasury
    ) -> float:
        """
        Get current risk-free rate from treasury yields.
        
        Parameters:
        -----------
        maturity : str
            Treasury ticker (^IRX for 13-week, ^FVX for 5-year, ^TNX for 10-year)
        
        Returns:
        --------
        float
            Risk-free rate (annualized)
        """
        try:
            treasury = yf.Ticker(maturity)
            hist = treasury.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100.0  # Convert from percentage
        except:
            pass
        
        # Fallback to environment variable or default
        return float(os.getenv("RISK_FREE_RATE", 0.05))
    
    def _get_cache_key(
        self,
        ticker: str,
        date: Optional[datetime]
    ) -> str:
        """Generate cache key for a ticker and date."""
        if date is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        else:
            date_str = date.strftime("%Y-%m-%d")
        return f"{ticker}_{date_str}_{self.provider}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            # Check if cache is recent (within 1 hour for current data)
            if datetime.now().timestamp() - cache_file.stat().st_mtime < 3600:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    warnings.warn(f"Error loading cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            warnings.warn(f"Error saving cache: {e}")
    
    def clean_option_chain(
        self,
        df: pd.DataFrame,
        min_volume: int = 0,
        min_oi: int = 0,
        max_bid_ask_spread: float = 0.5
    ) -> pd.DataFrame:
        """
        Clean and filter option chain data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw option chain data
        min_volume : int
            Minimum volume filter
        min_oi : int
            Minimum open interest filter
        max_bid_ask_spread : float
            Maximum bid-ask spread (as fraction of mid price)
        
        Returns:
        --------
        pd.DataFrame
            Cleaned option chain
        """
        df = df.copy()
        
        # Remove NaN values
        df = df.dropna(subset=['strike', 'expiry', 'mid'])
        
        # Filter by volume and OI
        if 'volume' in df.columns:
            df = df[df['volume'] >= min_volume]
        if 'oi' in df.columns:
            df = df[df['oi'] >= min_oi]
        
        # Filter by bid-ask spread
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
            df = df[df['spread_pct'] <= max_bid_ask_spread]
            df = df.drop('spread_pct', axis=1)
        
        # Remove zero prices
        df = df[df['mid'] > 0]
        
        # Sort by expiry and strike
        df = df.sort_values(['expiry', 'strike'])
        
        return df.reset_index(drop=True)


class HistoricalVolatilityCalculator:
    """
    Calculate historical (realized) volatility for backtesting.
    """
    
    @staticmethod
    def get_price_history(
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data."""
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        return hist
    
    @staticmethod
    def calculate_realized_volatility(
        prices: pd.Series,
        window: int = 20,
        annualization_factor: int = 252
    ) -> pd.Series:
        """
        Calculate rolling realized volatility.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Rolling window size
        annualization_factor : int
            Factor to annualize volatility (252 for daily data)
        
        Returns:
        --------
        pd.Series
            Realized volatility series
        """
        returns = np.log(prices / prices.shift(1))
        realized_vol = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
        return realized_vol
    
    @staticmethod
    def calculate_parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
        annualization_factor: int = 252
    ) -> pd.Series:
        """
        Calculate Parkinson volatility (uses high-low range).
        More efficient estimator than close-to-close.
        """
        hl_ratio = np.log(high / low)
        parkinson_vol = hl_ratio.rolling(window=window).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2)))
        ) * np.sqrt(annualization_factor)
        return parkinson_vol
    
    @staticmethod
    def calculate_garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        annualization_factor: int = 252
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator.
        Uses open, high, low, close for better estimation.
        """
        hl = np.log(high / low)
        co = np.log(close / open_)
        
        gk_vol = (0.5 * hl**2 - (2*np.log(2) - 1) * co**2).rolling(window=window).mean()
        gk_vol = np.sqrt(gk_vol * annualization_factor)
        
        return gk_vol


if __name__ == "__main__":
    # Example usage
    fetcher = OptionDataFetcher()
    
    # Fetch SPY option chain
    print("Fetching SPY option chain...")
    spy_chain = fetcher.get_option_chain("SPY")
    print(f"Fetched {len(spy_chain)} option contracts")
    print(spy_chain.head())
    
    # Clean the data
    spy_clean = fetcher.clean_option_chain(spy_chain, min_volume=10, min_oi=100)
    print(f"\nAfter cleaning: {len(spy_clean)} contracts")
    
    # Get spot price and risk-free rate
    spot = fetcher.get_spot_price("SPY")
    rf = fetcher.get_risk_free_rate()
    print(f"\nSPY Spot: ${spot:.2f}")
    print(f"Risk-free rate: {rf*100:.2f}%")
    
    # Calculate realized volatility
    print("\nCalculating realized volatility...")
    vol_calc = HistoricalVolatilityCalculator()
    hist = vol_calc.get_price_history("SPY", datetime.now() - timedelta(days=365), datetime.now())
    realized_vol = vol_calc.calculate_realized_volatility(hist['Close'])
    print(f"Current 20-day realized vol: {realized_vol.iloc[-1]*100:.2f}%")
