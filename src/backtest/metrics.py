"""
Performance Metrics and Analytics

Comprehensive risk metrics including Sharpe, Sortino, max drawdown, and Greeks P&L attribution.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats


class PerformanceMetrics:
    """Calculate performance and risk metrics for strategy backtests."""
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        risk_free_rate : float
            Annual risk-free rate
        annualization_factor : int
            252 for daily, 12 for monthly
        
        Returns:
        --------
        float
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (focuses on downside volatility).
        
        Returns:
        --------
        float
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf
        
        return np.sqrt(annualization_factor) * excess_returns.mean() / downside_std
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.
        
        Returns:
        --------
        max_dd : float
            Maximum drawdown (as decimal)
        peak_date : Timestamp
            Date of peak before max drawdown
        trough_date : Timestamp
            Date of trough (max drawdown)
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = equity_curve[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Returns:
        --------
        float
            Calmar ratio
        """
        annual_return = returns.mean() * annualization_factor
        max_dd, _, _ = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else -np.inf
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def var(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
        
        Returns:
        --------
        float
            VaR (positive number representing loss)
        """
        return -np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def cvar(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Returns:
        --------
        float
            CVaR (average loss beyond VaR)
        """
        var = PerformanceMetrics.var(returns, confidence)
        return -returns[returns <= -var].mean()
    
    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio (probability-weighted gains vs losses).
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
        threshold : float
            Threshold return (typically 0)
        
        Returns:
        --------
        float
            Omega ratio
        """
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()
        
        if losses == 0:
            return np.inf if gains > 0 else 0
        
        return gains / losses
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate percentage of winning periods."""
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def tail_ratio(returns: pd.Series, percentile: float = 95) -> float:
        """
        Calculate tail ratio (right tail / left tail).
        Measures skewness of returns distribution.
        """
        right_tail = np.percentile(returns, percentile)
        left_tail = np.percentile(returns, 100 - percentile)
        
        if left_tail >= 0:
            return np.inf
        
        return abs(right_tail / left_tail)


class GreeksPnLAttribution:
    """Attribute P&L to different Greeks components."""
    
    @staticmethod
    def decompose_pnl(
        portfolio_history: pd.DataFrame,
        spot_prices: pd.Series
    ) -> pd.DataFrame:
        """
        Decompose P&L into theta, gamma, vega, and other components.
        
        Parameters:
        -----------
        portfolio_history : pd.DataFrame
            Portfolio history with Greeks
        spot_prices : pd.Series
            Spot price time series
        
        Returns:
        --------
        pd.DataFrame
            P&L attribution
        """
        pnl_attr = pd.DataFrame(index=portfolio_history.index)
        
        # Calculate spot moves
        spot_moves = spot_prices.diff()
        
        # Theta P&L (time decay)
        pnl_attr['theta_pnl'] = portfolio_history['theta']
        
        # Delta P&L (directional)
        pnl_attr['delta_pnl'] = portfolio_history['delta'] * spot_moves
        
        # Gamma P&L (convexity)
        pnl_attr['gamma_pnl'] = 0.5 * portfolio_history['gamma'] * (spot_moves ** 2)
        
        # Vega P&L (volatility)
        if 'implied_vol' in portfolio_history.columns:
            vol_changes = portfolio_history['implied_vol'].diff()
            pnl_attr['vega_pnl'] = portfolio_history['vega'] * vol_changes
        else:
            pnl_attr['vega_pnl'] = 0
        
        # Total P&L
        if 'total_value' in portfolio_history.columns:
            pnl_attr['total_pnl'] = portfolio_history['total_value'].diff()
        
        # Unexplained P&L
        explained = pnl_attr[['theta_pnl', 'delta_pnl', 'gamma_pnl', 'vega_pnl']].sum(axis=1)
        pnl_attr['unexplained_pnl'] = pnl_attr.get('total_pnl', 0) - explained
        
        return pnl_attr
    
    @staticmethod
    def calculate_greeks_contribution(pnl_attribution: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate percentage contribution of each Greek to total P&L.
        
        Returns:
        --------
        dict
            Percentage contributions
        """
        total_pnl = pnl_attribution['total_pnl'].sum()
        
        if total_pnl == 0:
            return {
                'theta_contribution': 0,
                'delta_contribution': 0,
                'gamma_contribution': 0,
                'vega_contribution': 0,
                'unexplained_contribution': 0
            }
        
        return {
            'theta_contribution': pnl_attribution['theta_pnl'].sum() / total_pnl,
            'delta_contribution': pnl_attribution['delta_pnl'].sum() / total_pnl,
            'gamma_contribution': pnl_attribution['gamma_pnl'].sum() / total_pnl,
            'vega_contribution': pnl_attribution['vega_pnl'].sum() / total_pnl,
            'unexplained_contribution': pnl_attribution['unexplained_pnl'].sum() / total_pnl
        }


class VolatilityAnalysis:
    """Analysis tools for implied vs realized volatility."""
    
    @staticmethod
    def calculate_correlation(
        implied_vol: pd.Series,
        realized_vol: pd.Series
    ) -> float:
        """Calculate correlation between implied and realized volatility."""
        return implied_vol.corr(realized_vol)
    
    @staticmethod
    def calculate_vol_risk_premium(
        implied_vol: pd.Series,
        realized_vol: pd.Series
    ) -> pd.Series:
        """Calculate volatility risk premium (IV - RV)."""
        return implied_vol - realized_vol
    
    @staticmethod
    def vol_forecast_accuracy(
        implied_vol: pd.Series,
        realized_vol: pd.Series
    ) -> Dict[str, float]:
        """
        Analyze how well implied vol forecasts realized vol.
        
        Returns:
        --------
        dict
            RMSE, MAE, and correlation
        """
        mse = ((implied_vol - realized_vol) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = (implied_vol - realized_vol).abs().mean()
        corr = implied_vol.corr(realized_vol)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'mean_forecast_error': (implied_vol - realized_vol).mean()
        }


def generate_performance_report(
    returns: pd.Series,
    equity_curve: pd.Series,
    portfolio_history: pd.DataFrame,
    spot_prices: pd.Series,
    risk_free_rate: float = 0.05
) -> Dict:
    """
    Generate comprehensive performance report.
    
    Returns:
    --------
    dict
        Complete performance metrics
    """
    pm = PerformanceMetrics()
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe = pm.sharpe_ratio(returns, risk_free_rate)
    sortino = pm.sortino_ratio(returns, risk_free_rate)
    max_dd, peak_date, trough_date = pm.max_drawdown(equity_curve)
    calmar = pm.calmar_ratio(returns, equity_curve)
    
    # Distribution metrics
    var_95 = pm.var(returns, 0.95)
    cvar_95 = pm.cvar(returns, 0.95)
    omega = pm.omega_ratio(returns)
    
    # Trading metrics
    win_rate = pm.win_rate(returns)
    profit_factor = pm.profit_factor(returns)
    tail_ratio = pm.tail_ratio(returns)
    
    # Greeks P&L attribution
    greeks_attr = GreeksPnLAttribution()
    pnl_decomp = greeks_attr.decompose_pnl(portfolio_history, spot_prices)
    greeks_contrib = greeks_attr.calculate_greeks_contribution(pnl_decomp)
    
    report = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'max_dd_peak_date': peak_date,
        'max_dd_trough_date': trough_date,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'omega_ratio': omega,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'tail_ratio': tail_ratio,
        'greeks_contribution': greeks_contrib
    }
    
    return report


if __name__ == "__main__":
    # Test performance metrics
    print("Performance Metrics Test")
    print("=" * 50)
    
    # Generate synthetic returns
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    equity_curve = (1 + returns).cumprod() * 100000
    
    pm = PerformanceMetrics()
    
    # Calculate metrics
    sharpe = pm.sharpe_ratio(returns)
    sortino = pm.sortino_ratio(returns)
    max_dd, peak, trough = pm.max_drawdown(equity_curve)
    
    print(f"\nSharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Win Rate: {pm.win_rate(returns):.2%}")
    print(f"Profit Factor: {pm.profit_factor(returns):.2f}")
    print(f"VaR (95%): {pm.var(returns, 0.95):.2%}")
    print(f"CVaR (95%): {pm.cvar(returns, 0.95):.2%}")
