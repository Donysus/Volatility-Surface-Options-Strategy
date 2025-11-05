"""Utility functions for the volatility surface project."""

import os
import yaml
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(directory: str) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency."""
    return f"${value:,.{decimals}f}"


def calculate_time_to_maturity(expiry, current_date=None):
    """Calculate time to maturity in years."""
    if current_date is None:
        current_date = datetime.now()
    
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, '%Y-%m-%d')
    
    return (expiry - current_date).days / 365.25
