"""
Dynamic Volatility Surface & Options Strategy Simulator

A comprehensive toolkit for volatility modeling, options pricing, and strategy backtesting.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data
from . import pricing
from . import surface
from . import greeks
from . import strategies
from . import backtest
from . import visualization
from . import utils

__all__ = [
    "data",
    "pricing",
    "surface",
    "greeks",
    "strategies",
    "backtest",
    "visualization",
    "utils",
]
