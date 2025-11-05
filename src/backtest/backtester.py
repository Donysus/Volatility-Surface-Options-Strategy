"""
Backtesting Engine for Options Strategies

Portfolio simulator with P&L attribution, Greeks tracking, and transaction costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class Position:
    """Represents an option position."""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    entry_price: float
    contracts: int  # Positive = long, negative = short
    entry_date: datetime
    entry_spot: float
    
    # Greeks at entry
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_vega: float = 0.0
    entry_theta: float = 0.0
    
    # Current values
    current_price: float = 0.0
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_vega: float = 0.0
    current_theta: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_current_values(self, price: float, delta: float, gamma: float, 
                             vega: float, theta: float):
        """Update current option values and Greeks."""
        self.current_price = price
        self.current_delta = delta
        self.current_gamma = gamma
        self.current_vega = vega
        self.current_theta = theta
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.contracts * 100
    
    def close(self, exit_price: float, exit_date: datetime) -> float:
        """Close position and return realized P&L."""
        self.realized_pnl = (exit_price - self.entry_price) * self.contracts * 100
        return self.realized_pnl


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions_value: float
    hedge_value: float
    total_value: float
    
    # Aggregate Greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    
    # P&L components
    options_pnl: float = 0.0
    hedge_pnl: float = 0.0
    theta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    transaction_costs: float = 0.0


class OptionsBacktester:
    """
    Backtesting engine for options strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 0.5
    ):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        transaction_cost_bps : float
            Transaction costs in basis points
        slippage_bps : float
            Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.hedge_position = 0.0  # Shares of underlying
        self.portfolio_history: List[PortfolioState] = []
        self.trade_log: List[Dict] = []
        
    def open_position(
        self,
        option_type: str,
        strike: float,
        expiry: datetime,
        price: float,
        contracts: int,
        date: datetime,
        spot: float,
        greeks: Dict
    ) -> bool:
        """
        Open a new option position.
        
        Returns:
        --------
        bool
            True if position opened successfully
        """
        # Calculate cost
        cost = abs(price * contracts * 100)
        transaction_cost = cost * self.transaction_cost_bps / 10000
        total_cost = cost + transaction_cost
        
        # Check if we have enough cash
        if contracts > 0 and total_cost > self.cash:
            return False
        
        # Create position
        position = Position(
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            entry_price=price,
            contracts=contracts,
            entry_date=date,
            entry_spot=spot,
            entry_delta=greeks.get('delta', 0),
            entry_gamma=greeks.get('gamma', 0),
            entry_vega=greeks.get('vega', 0),
            entry_theta=greeks.get('theta', 0)
        )
        
        # Update cash
        if contracts > 0:
            self.cash -= total_cost
        else:
            self.cash += (cost - transaction_cost)
        
        # Add to positions
        self.positions.append(position)
        
        # Log trade
        self.trade_log.append({
            'date': date,
            'action': 'open',
            'type': option_type,
            'strike': strike,
            'expiry': expiry,
            'price': price,
            'contracts': contracts,
            'cost': total_cost
        })
        
        return True
    
    def close_position(
        self,
        position: Position,
        price: float,
        date: datetime
    ):
        """Close an existing position."""
        # Calculate proceeds
        proceeds = abs(price * position.contracts * 100)
        transaction_cost = proceeds * self.transaction_cost_bps / 10000
        
        # Update cash
        if position.contracts > 0:
            self.cash += (proceeds - transaction_cost)
        else:
            self.cash -= (proceeds + transaction_cost)
        
        # Record P&L
        realized_pnl = position.close(price, date)
        
        # Move to closed positions
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        # Log trade
        self.trade_log.append({
            'date': date,
            'action': 'close',
            'type': position.option_type,
            'strike': position.strike,
            'expiry': position.expiry,
            'price': price,
            'contracts': position.contracts,
            'pnl': realized_pnl
        })
    
    def update_hedge(
        self,
        target_hedge: float,
        spot_price: float,
        date: datetime
    ):
        """Update delta hedge position."""
        shares_to_trade = target_hedge - self.hedge_position
        
        if abs(shares_to_trade) < 0.01:
            return
        
        # Calculate costs
        notional = abs(shares_to_trade * spot_price)
        transaction_cost = notional * self.transaction_cost_bps / 10000
        slippage = notional * self.slippage_bps / 10000
        
        # Update cash
        self.cash -= (shares_to_trade * spot_price + transaction_cost + slippage)
        
        # Update hedge position
        self.hedge_position = target_hedge
        
        # Log trade
        self.trade_log.append({
            'date': date,
            'action': 'hedge',
            'shares': shares_to_trade,
            'price': spot_price,
            'cost': transaction_cost + slippage
        })
    
    def update_positions(
        self,
        option_prices: pd.DataFrame,
        greeks: pd.DataFrame,
        date: datetime
    ):
        """Update all position values and Greeks."""
        for position in self.positions:
            # Find matching option
            mask = (
                (option_prices['type'] == position.option_type) &
                (option_prices['strike'] == position.strike) &
                (option_prices['expiry'] == position.expiry)
            )
            
            if mask.sum() > 0:
                option_data = option_prices[mask].iloc[0]
                greeks_data = greeks[mask].iloc[0]
                
                position.update_current_values(
                    price=option_data['mid'],
                    delta=greeks_data['delta'],
                    gamma=greeks_data['gamma'],
                    vega=greeks_data['vega'],
                    theta=greeks_data['theta']
                )
    
    def get_portfolio_state(
        self,
        date: datetime,
        spot_price: float
    ) -> PortfolioState:
        """Get current portfolio state."""
        # Calculate positions value
        positions_value = sum(p.current_price * p.contracts * 100 for p in self.positions)
        
        # Calculate hedge value
        hedge_value = self.hedge_position * spot_price
        
        # Aggregate Greeks
        total_delta = sum(p.current_delta * p.contracts * 100 for p in self.positions)
        total_gamma = sum(p.current_gamma * p.contracts * 100 for p in self.positions)
        total_vega = sum(p.current_vega * p.contracts * 100 for p in self.positions)
        total_theta = sum(p.current_theta * p.contracts * 100 for p in self.positions)
        
        # Calculate total value
        total_value = self.cash + positions_value + hedge_value
        
        # Calculate P&L components (since last snapshot)
        options_pnl = sum(p.unrealized_pnl for p in self.positions)
        
        state = PortfolioState(
            timestamp=date,
            cash=self.cash,
            positions_value=positions_value,
            hedge_value=hedge_value,
            total_value=total_value,
            delta=total_delta,
            gamma=total_gamma,
            vega=total_vega,
            theta=total_theta,
            options_pnl=options_pnl
        )
        
        self.portfolio_history.append(state)
        return state
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame
        history_df = pd.DataFrame([
            {
                'date': s.timestamp,
                'total_value': s.total_value,
                'cash': s.cash,
                'delta': s.delta,
                'gamma': s.gamma,
                'vega': s.vega,
                'theta': s.theta
            }
            for s in self.portfolio_history
        ])
        
        # Calculate returns
        history_df['returns'] = history_df['total_value'].pct_change()
        
        # Summary statistics
        total_return = (history_df['total_value'].iloc[-1] / self.initial_capital - 1)
        total_pnl = history_df['total_value'].iloc[-1] - self.initial_capital
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': history_df['total_value'].iloc[-1],
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(self.trade_log),
            'num_closed_positions': len(self.closed_positions),
            'num_open_positions': len(self.positions)
        }
        
        return summary
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        return pd.DataFrame(self.trade_log)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame([
            {
                'date': s.timestamp,
                'total_value': s.total_value,
                'cash': s.cash,
                'positions_value': s.positions_value,
                'hedge_value': s.hedge_value,
                'delta': s.delta,
                'gamma': s.gamma,
                'vega': s.vega,
                'theta': s.theta,
                'options_pnl': s.options_pnl
            }
            for s in self.portfolio_history
        ])


if __name__ == "__main__":
    # Test backtester
    print("Backtesting Engine Test")
    print("=" * 50)
    
    # Initialize backtester
    bt = OptionsBacktester(initial_capital=100000)
    
    # Open a position
    print(f"\nInitial capital: ${bt.cash:,.2f}")
    
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
    
    print(f"Position opened: {success}")
    print(f"Cash after opening: ${bt.cash:,.2f}")
    print(f"Number of positions: {len(bt.positions)}")
    
    # Get portfolio state
    state = bt.get_portfolio_state(datetime.now(), spot_price=100)
    print(f"\nPortfolio Value: ${state.total_value:,.2f}")
    print(f"Delta: {state.delta:.2f}")
    print(f"Gamma: {state.gamma:.4f}")
    print(f"Vega: {state.vega:.2f}")
    print(f"Theta: {state.theta:.2f}")
    
    # Get summary
    summary = bt.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"Number of trades: {summary['num_trades']}")
