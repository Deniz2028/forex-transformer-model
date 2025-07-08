# src/backtest/k_pips_utils.py
"""Enhanced K-pips calculation utilities for consistent threshold computation - FIXED VERSION."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        df: OHLC DataFrame
        period: Period for ATR calculation
        
    Returns:
        ATR values as Series
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr

def compute_k_pips(df: pd.DataFrame, horizon: int, pair_name: str = None, 
                   k_pips_mult: float = 1.0, mode: str = 'realistic') -> float:
    """Compute REALISTIC k-pips threshold - FIXED to generate trades!
    
    REALISTIC FORMULA: k_pips = max(0.5 * ATR, 2.0 * spread) * k_pips_mult
    
    Args:
        df: OHLC DataFrame with spread column
        horizon: Prediction horizon
        pair_name: Name of currency pair for logging
        k_pips_mult: Multiplier for k-pips adjustment (0.5 = more trades)
        mode: Calculation mode ('realistic', 'conservative', 'aggressive')
        
    Returns:
        Computed k-pips threshold
    """
    # Enhanced ATR period calculation
    atr_period = max(4, horizon // 2)
    
    # Calculate ATR and average spread
    atr = calculate_atr(df, period=atr_period)
    atr_mean = atr.mean()
    avg_spread = df['spread'].mean()
    
    # FIXED k-pips formula - Much more realistic for trading
    if mode == 'realistic':
        base_k_pips = max(0.5 * atr_mean, 2.0 * avg_spread)  # REALISTIC
    elif mode == 'conservative':
        base_k_pips = max(0.3 * atr_mean, 1.5 * avg_spread)  # Conservative
    elif mode == 'aggressive':
        base_k_pips = max(0.8 * atr_mean, 3.0 * avg_spread)  # Very aggressive
    else:
        # Default to realistic
        base_k_pips = max(0.5 * atr_mean, 2.0 * avg_spread)
    
    # Apply multiplier
    k_pips = base_k_pips * k_pips_mult
    
    # Auto-optimization for trade frequency
    if k_pips_mult == 'auto':
        k_pips = auto_optimize_k_pips(df, horizon, base_k_pips, pair_name)
    
    # REMOVED: Special pair adjustments (they were making k-pips too small)
    
    logger.info(f"K-pips for {pair_name}: {k_pips:.6f} "
                f"(ATR: {atr_mean:.6f}, Spread: {avg_spread:.6f}, "
                f"ATR_period: {atr_period}, mult: {k_pips_mult}, mode: {mode})")
    
    return k_pips

def auto_optimize_k_pips(df: pd.DataFrame, horizon: int, base_k_pips: float, 
                        pair_name: str) -> float:
    """Auto-optimize k-pips for reasonable trade frequency (15-35%)."""
    
    from ..utils.labels import create_labels
    
    test_multipliers = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    best_k_pips = base_k_pips
    best_score = 0
    
    for mult in test_multipliers:
        test_k_pips = base_k_pips * mult
        
        # Test labels
        try:
            labels = create_labels(df, test_k_pips, horizon, 'three_class')
            
            # Calculate trade frequency (non-flat labels)
            trade_freq = np.mean(labels != 1)  # Not flat
            
            # Score: prefer 20-30% trade frequency
            if 0.15 <= trade_freq <= 0.35:
                score = 1.0 - abs(trade_freq - 0.25)  # Prefer 25%
                if score > best_score:
                    best_score = score
                    best_k_pips = test_k_pips
                    print(f"   🎯 Auto k-pips: {test_k_pips:.6f} (mult={mult:.1f}, freq={trade_freq:.1%})")
        except Exception as e:
            continue
    
    return best_k_pips

def validate_k_pips(k_pips: float, pair_name: str, avg_spread: float) -> Dict[str, Any]:
    """Enhanced validation with REALISTIC requirements."""
    validation = {
        'is_valid': True,
        'warnings': [],
        'recommendations': []
    }
    
    # Check if k-pips is reasonable for trading
    if k_pips < avg_spread * 1.5:
        validation['warnings'].append(f"K-pips {k_pips:.6f} might be too small vs spread {avg_spread:.6f}")
        validation['recommendations'].append("Consider increasing k_pips_mult to 1.2+")
    
    if k_pips > avg_spread * 10:
        validation['warnings'].append(f"K-pips {k_pips:.6f} very large vs spread {avg_spread:.6f}")
        validation['recommendations'].append("Consider decreasing k_pips_mult to 0.5-0.8")
    
    # Pair-specific realistic validations
    if pair_name == 'XAU_USD' and k_pips < 0.5:
        validation['warnings'].append("Gold k-pips might be too small for volatility")
        validation['recommendations'].append("Consider k_pips_mult >= 1.0 for XAU_USD")
    
    if pair_name in ['EUR_USD', 'GBP_USD'] and k_pips > 0.001:
        validation['warnings'].append("Major pair k-pips might be too large")
        validation['recommendations'].append("Consider k_pips_mult <= 1.0 for major pairs")
    
    return validation

def create_labels_test(df: pd.DataFrame, k_pips: float, horizon: int) -> np.ndarray:
    """Quick test for k-pips optimization."""
    future_price = df['close'].shift(-horizon)
    price_change = (future_price - df['close']) / df['close']
    
    # Three-class labels: 0=long, 1=flat, 2=short
    labels = np.ones(len(df))  # Default flat
    labels[price_change > k_pips] = 0   # Long
    labels[price_change < -k_pips] = 2  # Short
    
    return labels

__all__ = ['compute_k_pips', 'calculate_atr', 'validate_k_pips', 'auto_optimize_k_pips']
