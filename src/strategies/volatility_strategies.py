"""
Options Trading Strategies

Implementation of volatility arbitrage and gamma scalping strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class VolatilityArbitrageStrategy:
    """
    Volatility arbitrage strategy: Long underpriced options, short overpriced.
    Based on discrepancy between implied and realized volatility.
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.03,  # Enter when |IV - RV| > 3%
        exit_threshold: float = 0.01,   # Exit when |IV - RV| < 1%
        max_positions: int = 10,
        delta_hedge: bool = True
    ):
        """
        Initialize volatility arbitrage strategy.
        
        Parameters:
        -----------
        entry_threshold : float
            Minimum IV - RV spread to enter trade
        exit_threshold : float
            Maximum spread to exit trade
        max_positions : int
            Maximum number of concurrent positions
        delta_hedge : bool
            Whether to delta hedge positions
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_positions = max_positions
        self.delta_hedge = delta_hedge
        self.positions = []
        
    def generate_signals(
        self,
        implied_vol: float,
        realized_vol: float,
        current_positions: int = 0
    ) -> Dict[str, any]:
        """
        Generate trading signals based on IV vs RV.
        
        Parameters:
        -----------
        implied_vol : float
            Current implied volatility
        realized_vol : float
            Current realized volatility
        current_positions : int
            Number of current open positions
        
        Returns:
        --------
        dict
            Signal dictionary with action, reason, spread
        """
        spread = implied_vol - realized_vol
        
        signal = {
            'action': 'hold',
            'reason': '',
            'spread': spread,
            'implied_vol': implied_vol,
            'realized_vol': realized_vol
        }
        
        # Check for entry signals
        if current_positions < self.max_positions:
            if spread > self.entry_threshold:
                # IV > RV: Options overpriced, sell volatility
                signal['action'] = 'sell_vol'
                signal['reason'] = f'IV ({implied_vol:.2%}) > RV ({realized_vol:.2%}), sell options'
                
            elif spread < -self.entry_threshold:
                # IV < RV: Options underpriced, buy volatility
                signal['action'] = 'buy_vol'
                signal['reason'] = f'IV ({implied_vol:.2%}) < RV ({realized_vol:.2%}), buy options'
        
        # Check for exit signals
        if current_positions > 0:
            if abs(spread) < self.exit_threshold:
                signal['action'] = 'close'
                signal['reason'] = f'Spread converged to {spread:.2%}'
        
        return signal
    
    def construct_position(
        self,
        action: str,
        option_data: pd.DataFrame,
        spot: float,
        capital: float = 10000
    ) -> Dict:
        """
        Construct option position based on signal.
        
        Parameters:
        -----------
        action : str
            'buy_vol' or 'sell_vol'
        option_data : pd.DataFrame
            Available options with strikes, prices, greeks
        spot : float
            Current spot price
        capital : float
            Capital allocated to position
        
        Returns:
        --------
        dict
            Position details
        """
        # Select ATM straddle for simplicity
        atm_strike = option_data.iloc[(option_data['strike'] - spot).abs().argsort()[0]]['strike']
        
        call = option_data[(option_data['strike'] == atm_strike) & 
                          (option_data['type'] == 'call')].iloc[0]
        put = option_data[(option_data['strike'] == atm_strike) & 
                         (option_data['type'] == 'put')].iloc[0]
        
        straddle_price = call['mid'] + put['mid']
        
        # Calculate position size
        contracts = int(capital / (straddle_price * 100))
        
        position = {
            'action': action,
            'strike': atm_strike,
            'call_price': call['mid'],
            'put_price': put['mid'],
            'total_price': straddle_price,
            'contracts': contracts,
            'delta': (call['delta'] + put['delta']) * contracts,
            'gamma': (call['gamma'] + put['gamma']) * contracts,
            'vega': (call['vega'] + put['vega']) * contracts,
            'theta': (call['theta'] + put['theta']) * contracts,
            'cost': straddle_price * contracts * 100
        }
        
        # Adjust sign for sell positions
        if action == 'sell_vol':
            for key in ['delta', 'gamma', 'vega', 'theta', 'cost']:
                if key != 'gamma':  # Gamma sign doesn't flip
                    position[key] *= -1
        
        return position


class GammaScalpingStrategy:
    """
    Gamma scalping strategy: Long gamma, hedge delta dynamically.
    Profit from realized volatility being higher than implied.
    """
    
    def __init__(
        self,
        rebalance_threshold: float = 0.05,  # Rehedge when delta > 5%
        gamma_target: float = 0.10,          # Target gamma exposure
        transaction_cost_bps: float = 1.0    # Transaction costs in bps
    ):
        """
        Initialize gamma scalping strategy.
        
        Parameters:
        -----------
        rebalance_threshold : float
            Delta threshold to trigger rehedge
        gamma_target : float
            Target gamma exposure (% of notional)
        transaction_cost_bps : float
            Transaction costs in basis points
        """
        self.rebalance_threshold = rebalance_threshold
        self.gamma_target = gamma_target
        self.transaction_cost_bps = transaction_cost_bps
        self.hedge_position = 0  # Current hedge (shares)
        
    def check_rehedge(
        self,
        current_delta: float,
        spot: float,
        position_size: float
    ) -> Tuple[bool, float]:
        """
        Check if rehedging is needed.
        
        Parameters:
        -----------
        current_delta : float
            Current position delta
        spot : float
            Current spot price
        position_size : float
            Position notional
        
        Returns:
        --------
        bool
            Whether to rehedge
        float
            Number of shares to trade
        """
        delta_pct = abs(current_delta) / position_size if position_size > 0 else 0
        
        if delta_pct > self.rebalance_threshold:
            # Calculate hedge needed
            hedge_shares = -current_delta  # Delta-neutral hedge
            shares_to_trade = hedge_shares - self.hedge_position
            
            return True, shares_to_trade
        
        return False, 0
    
    def calculate_hedge_pnl(
        self,
        shares_traded: float,
        entry_price: float,
        exit_price: float
    ) -> Tuple[float, float]:
        """
        Calculate P&L from hedging activity.
        
        Returns:
        --------
        gross_pnl : float
            Gross P&L from hedge
        transaction_cost : float
            Transaction costs
        """
        gross_pnl = shares_traded * (exit_price - entry_price)
        transaction_cost = abs(shares_traded * entry_price) * self.transaction_cost_bps / 10000
        
        return gross_pnl, transaction_cost
    
    def select_option(
        self,
        option_data: pd.DataFrame,
        spot: float,
        target_maturity: float = 0.25
    ) -> pd.Series:
        """
        Select option for gamma scalping (typically ATM call/put or straddle).
        
        Parameters:
        -----------
        option_data : pd.DataFrame
            Available options
        spot : float
            Current spot price
        target_maturity : float
            Target time to maturity
        
        Returns:
        --------
        pd.Series
            Selected option
        """
        # Filter by maturity
        option_data = option_data[
            (option_data['T'] >= target_maturity - 0.1) & 
            (option_data['T'] <= target_maturity + 0.1)
        ].copy()
        
        # Find ATM option
        option_data['atm_distance'] = abs(option_data['strike'] - spot)
        
        # Prefer straddles (buy both call and put) for max gamma
        atm_strike = option_data.iloc[option_data['atm_distance'].argmin()]['strike']
        
        # Return both call and put at ATM
        atm_options = option_data[option_data['strike'] == atm_strike]
        
        return atm_options


class DeltaHedger:
    """
    Delta hedging utility for maintaining delta-neutral positions.
    """
    
    def __init__(
        self,
        rebalance_frequency: str = 'daily',
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 0.5
    ):
        """
        Initialize delta hedger.
        
        Parameters:
        -----------
        rebalance_frequency : str
            'continuous', 'daily', 'hourly'
        transaction_cost_bps : float
            Transaction costs in basis points
        slippage_bps : float
            Slippage in basis points
        """
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.hedge_history = []
        
    def calculate_hedge(
        self,
        position_delta: float,
        spot_delta: float = 1.0
    ) -> float:
        """
        Calculate required hedge position.
        
        Parameters:
        -----------
        position_delta : float
            Current portfolio delta
        spot_delta : float
            Delta of underlying (1.0 for stocks)
        
        Returns:
        --------
        float
            Number of shares to hold (negative = short)
        """
        return -position_delta / spot_delta
    
    def execute_hedge(
        self,
        target_position: float,
        current_position: float,
        spot_price: float,
        timestamp: datetime
    ) -> Dict:
        """
        Execute hedge trade.
        
        Parameters:
        -----------
        target_position : float
            Target hedge position
        current_position : float
            Current hedge position
        spot_price : float
            Current spot price
        timestamp : datetime
            Trade timestamp
        
        Returns:
        --------
        dict
            Trade details including costs
        """
        shares_to_trade = target_position - current_position
        
        if abs(shares_to_trade) < 0.01:  # Negligible trade
            return None
        
        # Calculate costs
        notional = abs(shares_to_trade * spot_price)
        transaction_cost = notional * self.transaction_cost_bps / 10000
        slippage = notional * self.slippage_bps / 10000
        total_cost = transaction_cost + slippage
        
        trade = {
            'timestamp': timestamp,
            'shares': shares_to_trade,
            'price': spot_price,
            'notional': shares_to_trade * spot_price,
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'total_cost': total_cost,
            'new_position': target_position
        }
        
        self.hedge_history.append(trade)
        
        return trade
    
    def get_total_hedge_cost(self) -> float:
        """Get cumulative hedging costs."""
        return sum(trade['total_cost'] for trade in self.hedge_history)
    
    def get_hedge_pnl(self) -> pd.DataFrame:
        """Get detailed hedge P&L breakdown."""
        return pd.DataFrame(self.hedge_history)


if __name__ == "__main__":
    # Test strategies
    print("Strategy Framework Test")
    print("=" * 50)
    
    # Test volatility arbitrage
    vol_arb = VolatilityArbitrageStrategy(entry_threshold=0.03)
    
    # Scenario 1: IV > RV
    signal = vol_arb.generate_signals(implied_vol=0.25, realized_vol=0.20)
    print(f"\nScenario 1 - IV > RV:")
    print(f"Action: {signal['action']}")
    print(f"Reason: {signal['reason']}")
    
    # Scenario 2: IV < RV
    signal = vol_arb.generate_signals(implied_vol=0.18, realized_vol=0.22)
    print(f"\nScenario 2 - IV < RV:")
    print(f"Action: {signal['action']}")
    print(f"Reason: {signal['reason']}")
    
    # Test gamma scalping
    gamma_scalp = GammaScalpingStrategy(rebalance_threshold=0.05)
    
    # Check rehedge
    need_hedge, shares = gamma_scalp.check_rehedge(
        current_delta=500, spot=100, position_size=10000
    )
    print(f"\nGamma Scalping - Rehedge Check:")
    print(f"Need hedge: {need_hedge}")
    print(f"Shares to trade: {shares:.0f}")
    
    # Test delta hedger
    hedger = DeltaHedger()
    hedge_position = hedger.calculate_hedge(position_delta=50)
    print(f"\nDelta Hedging:")
    print(f"Position delta: 50")
    print(f"Required hedge: {hedge_position:.0f} shares")
    
    # Execute hedge
    trade = hedger.execute_hedge(
        target_position=hedge_position,
        current_position=0,
        spot_price=100,
        timestamp=datetime.now()
    )
    if trade:
        print(f"Trade executed: {trade['shares']:.0f} shares @ ${trade['price']:.2f}")
        print(f"Total cost: ${trade['total_cost']:.2f}")
