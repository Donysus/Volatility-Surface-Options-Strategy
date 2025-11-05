"""
Visualization Suite for Volatility Surfaces and Strategy Analysis

Comprehensive plotting tools using Plotly and Matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
from datetime import datetime


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class VolatilityVisualizer:
    """Visualization tools for volatility surfaces and analysis."""
    
    def __init__(self, style: str = "plotly"):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        style : str
            'plotly' or 'matplotlib'
        """
        self.style = style
        
    def plot_volatility_smile(
        self,
        strikes: np.ndarray,
        implied_vols: np.ndarray,
        spot: float,
        maturity: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D volatility smile.
        
        Parameters:
        -----------
        strikes : array
            Strike prices
        implied_vols : array
            Implied volatilities
        spot : float
            Current spot price
        maturity : float
            Time to maturity
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if title is None:
            title = f"Volatility Smile (T={maturity:.2f}y)"
        
        if self.style == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=strikes,
                y=implied_vols * 100,
                mode='lines+markers',
                name='Implied Vol',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add ATM line
            fig.add_vline(x=spot, line_dash="dash", line_color="red", 
                         annotation_text="ATM", annotation_position="top")
            
            fig.update_layout(
                title=title,
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(strikes, implied_vols * 100, 'o-', linewidth=2, markersize=6)
            ax.axvline(spot, color='r', linestyle='--', label='ATM')
            ax.set_xlabel('Strike')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_term_structure(
        self,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        realized_vol: Optional[float] = None,
        title: str = "Volatility Term Structure",
        save_path: Optional[str] = None
    ):
        """
        Plot volatility term structure.
        
        Parameters:
        -----------
        maturities : array
            Times to maturity
        implied_vols : array
            Implied volatilities
        realized_vol : float, optional
            Current realized volatility for comparison
        """
        if self.style == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=maturities,
                y=implied_vols * 100,
                mode='lines+markers',
                name='Implied Vol',
                line=dict(color='blue', width=2)
            ))
            
            if realized_vol is not None:
                fig.add_hline(y=realized_vol * 100, line_dash="dash", 
                             line_color="green", annotation_text="Realized Vol")
            
            fig.update_layout(
                title=title,
                xaxis_title="Time to Maturity (years)",
                yaxis_title="Volatility (%)",
                template='plotly_white',
                height=500
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(maturities, implied_vols * 100, 'o-', linewidth=2, label='Implied Vol')
            if realized_vol is not None:
                ax.axhline(realized_vol * 100, color='g', linestyle='--', label='Realized Vol')
            ax.set_xlabel('Time to Maturity (years)')
            ax.set_ylabel('Volatility (%)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_volatility_surface_3d(
        self,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        IV_grid: np.ndarray,
        spot: float,
        title: str = "Volatility Surface",
        save_path: Optional[str] = None
    ):
        """
        Plot 3D volatility surface.
        
        Parameters:
        -----------
        K_grid : 2D array
            Strike price grid
        T_grid : 2D array
            Maturity grid
        IV_grid : 2D array
            Implied volatility grid
        spot : float
            Current spot price
        """
        if self.style == "plotly":
            fig = go.Figure(data=[go.Surface(
                x=K_grid,
                y=T_grid,
                z=IV_grid * 100,
                colorscale='Viridis',
                colorbar=dict(title="IV (%)")
            )])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Maturity (years)",
                    zaxis_title="Implied Vol (%)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                template='plotly_white',
                height=600
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(K_grid, T_grid, IV_grid * 100, 
                                   cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Strike')
            ax.set_ylabel('Maturity (years)')
            ax.set_zlabel('Implied Vol (%)')
            ax.set_title(title)
            
            fig.colorbar(surf, ax=ax, shrink=0.5)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_iv_vs_realized(
        self,
        dates: pd.DatetimeIndex,
        implied_vol: pd.Series,
        realized_vol: pd.Series,
        title: str = "Implied vs Realized Volatility",
        save_path: Optional[str] = None
    ):
        """Plot time series comparison of implied and realized volatility."""
        if self.style == "plotly":
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=implied_vol * 100,
                mode='lines', name='Implied Vol',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=realized_vol * 100,
                mode='lines', name='Realized Vol',
                line=dict(color='green', width=2)
            ))
            
            # Add difference
            fig.add_trace(go.Scatter(
                x=dates, y=(implied_vol - realized_vol) * 100,
                mode='lines', name='IV - RV',
                line=dict(color='red', width=1, dash='dot'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                yaxis2=dict(title="Spread (%)", overlaying='y', side='right'),
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            ax1.plot(dates, implied_vol * 100, label='Implied Vol', linewidth=2)
            ax1.plot(dates, realized_vol * 100, label='Realized Vol', linewidth=2)
            ax1.set_ylabel('Volatility (%)')
            ax1.set_title(title)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(dates, (implied_vol - realized_vol) * 100, 
                    color='red', linewidth=2)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('IV - RV (%)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig


class StrategyVisualizer:
    """Visualization tools for strategy analysis."""
    
    @staticmethod
    def plot_pnl_curve(
        dates: pd.DatetimeIndex,
        pnl: pd.Series,
        title: str = "Cumulative P&L",
        benchmark: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """Plot cumulative P&L over time."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=pnl,
            mode='lines', name='Strategy P&L',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=dates, y=benchmark,
                mode='lines', name='Benchmark',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_pnl_attribution(
        dates: pd.DatetimeIndex,
        theta_pnl: pd.Series,
        gamma_pnl: pd.Series,
        vega_pnl: pd.Series,
        total_pnl: pd.Series,
        title: str = "P&L Attribution",
        save_path: Optional[str] = None
    ):
        """Plot P&L broken down by Greeks."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Component P&L", "Cumulative P&L"),
            vertical_spacing=0.12
        )
        
        # Stacked bar chart
        fig.add_trace(go.Bar(x=dates, y=theta_pnl, name='Theta P&L', 
                            marker_color='red'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=gamma_pnl, name='Gamma P&L', 
                            marker_color='green'), row=1, col=1)
        fig.add_trace(go.Bar(x=dates, y=vega_pnl, name='Vega P&L', 
                            marker_color='blue'), row=1, col=1)
        
        # Cumulative
        fig.add_trace(go.Scatter(x=dates, y=theta_pnl.cumsum(), name='Cumulative Theta', 
                                line=dict(color='red', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=gamma_pnl.cumsum(), name='Cumulative Gamma', 
                                line=dict(color='green', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=vega_pnl.cumsum(), name='Cumulative Vega', 
                                line=dict(color='blue', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=total_pnl.cumsum(), name='Total', 
                                line=dict(color='black', width=3, dash='dash')), row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            template='plotly_white',
            height=700,
            barmode='relative',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Daily P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_drawdown(
        dates: pd.DatetimeIndex,
        equity_curve: pd.Series,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ):
        """Plot equity curve with drawdown."""
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Equity Curve", "Drawdown"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=dates, y=equity_curve,
            mode='lines', name='Equity',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=dates, y=running_max,
            mode='lines', name='Peak',
            line=dict(color='green', width=1, dash='dash')
        ), row=1, col=1)
        
        # Drawdown
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown,
            mode='lines', name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ), row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_greeks_evolution(
        dates: pd.DatetimeIndex,
        delta: pd.Series,
        gamma: pd.Series,
        vega: pd.Series,
        theta: pd.Series,
        title: str = "Greeks Evolution",
        save_path: Optional[str] = None
    ):
        """Plot evolution of Greeks over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Delta", "Gamma", "Vega", "Theta"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        fig.add_trace(go.Scatter(x=dates, y=delta, name='Delta', 
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=gamma, name='Gamma', 
                                line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=dates, y=vega, name='Vega', 
                                line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=theta, name='Theta', 
                                line=dict(color='red')), row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


if __name__ == "__main__":
    # Test visualizations
    print("Visualization Suite Test")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Volatility smile
    strikes = np.linspace(90, 110, 20)
    spot = 100
    ivs = 0.20 + 0.05 * ((strikes - spot) / spot) - 0.1 * ((strikes - spot) / spot)**2
    
    viz = VolatilityVisualizer(style='plotly')
    fig = viz.plot_volatility_smile(strikes, ivs, spot, 0.5)
    print("Created volatility smile plot")
    
    # Term structure
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    term_ivs = np.array([0.18, 0.20, 0.22, 0.21])
    fig = viz.plot_term_structure(maturities, term_ivs, realized_vol=0.19)
    print("Created term structure plot")
    
    # 3D surface
    K_range = np.linspace(90, 110, 15)
    T_range = np.linspace(0.25, 2.0, 10)
    K_grid, T_grid = np.meshgrid(K_range, T_range)
    IV_grid = 0.20 + 0.05 * ((K_grid - spot) / spot) - 0.1 * ((K_grid - spot) / spot)**2
    
    fig = viz.plot_volatility_surface_3d(K_grid, T_grid, IV_grid, spot)
    print("Created 3D volatility surface")
    
    print("\nAll visualizations created successfully!")
