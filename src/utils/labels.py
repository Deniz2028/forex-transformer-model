"""Enhanced label generation utilities with dynamic k-pips calculation."""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union

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
                   mode: str = 'enhanced') -> float:
    """Compute dynamic k-pips threshold with enhanced formula.
    
    Enhanced formula: k_pips = max(0.2 * ATR, 0.8 * spread)
    ATR period = horizon // 2 (minimum 4)
    
    Args:
        df: OHLC DataFrame with spread column
        horizon: Prediction horizon
        pair_name: Name of currency pair for logging
        mode: Calculation mode ('enhanced', 'conservative', 'aggressive')
        
    Returns:
        Computed k-pips threshold
    """
    # Enhanced ATR period calculation: horizon // 2 with minimum of 4
    atr_period = max(4, horizon // 2)
    
    # Calculate ATR and average spread
    atr = calculate_atr(df, period=atr_period)
    atr_mean = atr.mean()
    avg_spread = df['spread'].mean()
    
    # Enhanced k-pips formula
    if mode == 'enhanced':
        k_pips = 0.0008
        #k_pips = max(0.2 * atr_mean, 0.8 * avg_spread)
    elif mode == 'conservative':
        k_pips = max(0.15 * atr_mean, 0.6 * avg_spread)
    elif mode == 'aggressive':
        k_pips = max(0.3 * atr_mean, 1.0 * avg_spread)
    else:
        # Default to enhanced
        k_pips = max(0.2 * atr_mean, 0.8 * avg_spread)
    
    # Special adjustments for specific pairs (removed rescue mode reduction)
    # USD_JPY and XAU_USD no longer get automatic reduction
    
    logger.info(f"K-pips for {pair_name}: {k_pips:.6f} "
                f"(ATR: {atr_mean:.6f}, Spread: {avg_spread:.6f}, "
                f"ATR_period: {atr_period}, Mode: {mode})")
    
    return k_pips

def create_labels(df: pd.DataFrame, k_pips: float, horizon: int, 
                  target_mode: str = 'binary') -> pd.Series:
    """Create labels based on future price movement.
    
    Args:
        df: OHLC DataFrame
        k_pips: Threshold for label creation
        horizon: Future horizon to look ahead
        target_mode: 'binary' or 'three_class'
        
    Returns:
        Label series
    """
    future_price = df['close'].shift(-horizon)
    price_change = (future_price - df['close']) / df['close']
    
    if target_mode == 'three_class':
        # Three class: long (0), flat (1), short (2)
        labels = pd.Series(1, index=df.index)  # Default flat
        labels[price_change > k_pips] = 0  # Long
        labels[price_change < -k_pips] = 2  # Short
    else:
        # Binary: up (1) or not (0)
        labels = (price_change > k_pips).astype(int)
    
    # Check for single class problem and warn
    unique_labels = labels.dropna().unique()
    if len(unique_labels) == 1:
        logger.warning(f"WARN: Single class problem detected for {target_mode} mode! "
                      f"Only class {unique_labels[0]} present with k_pips={k_pips:.6f}. "
                      f"Consider adjusting k_pips threshold or switching target mode.")
        print(f"   ⚠️ WARN: Single class problem! Only class {unique_labels[0]} present")
    elif len(unique_labels) == 2 and target_mode == 'three_class':
        logger.warning(f"WARN: Only 2 classes found in three_class mode: {unique_labels}. "
                      f"Missing flat class with k_pips={k_pips:.6f}")
        print(f"   ⚠️ WARN: Missing flat class in three_class mode: {unique_labels}")
    
    return labels

def analyze_label_distribution(labels: pd.Series, pair_name: str = None, 
                              target_mode: str = 'binary') -> dict:
    """Analyze label distribution and return statistics.
    
    Args:
        labels: Label series
        pair_name: Name of currency pair
        target_mode: Target mode for interpretation
        
    Returns:
        Dictionary with label statistics
    """
    clean_labels = labels.dropna()
    value_counts = clean_labels.value_counts().sort_index()
    
    stats = {
        'total_samples': len(clean_labels),
        'class_counts': value_counts.to_dict(),
        'class_proportions': (value_counts / len(clean_labels)).to_dict(),
        'unique_classes': len(value_counts),
        'is_single_class': len(value_counts) == 1
    }
    
    if target_mode == 'binary':
        if len(value_counts) >= 2:
            minority_ratio = value_counts.min() / len(clean_labels)
            stats['minority_ratio'] = minority_ratio
            stats['is_imbalanced'] = minority_ratio < 0.3
        else:
            stats['minority_ratio'] = 0.0
            stats['is_imbalanced'] = True
    else:
        # Three class mode
        if len(value_counts) >= 2:
            stats['minority_ratio'] = value_counts.min() / len(clean_labels)
        else:
            stats['minority_ratio'] = 0.0
    
    # Enhanced logging for three-class mode
    if pair_name:
        if target_mode == 'three_class':
            class_names = {0: 'Long', 1: 'Flat', 2: 'Short'}
            class_str = ', '.join([f"{class_names.get(k, k)}: {v}" for k, v in stats['class_counts'].items()])
            logger.info(f"Three-class distribution for {pair_name}: {class_str}")
        else:
            logger.info(f"Binary distribution for {pair_name}: {stats['class_counts']}")
            
        if stats['is_single_class']:
            logger.warning(f"Single class problem for {pair_name} in {target_mode} mode!")
    
    return stats

def encode_labels_three_class(price_changes: pd.Series, k_pips: float) -> pd.Series:
    """Encode price changes to three-class labels.
    
    Args:
        price_changes: Series of price change ratios
        k_pips: Threshold for classification
        
    Returns:
        Encoded labels: 0=long, 1=flat, 2=short
    """
    labels = pd.Series(1, index=price_changes.index)  # Default flat (1)
    labels[price_changes > k_pips] = 0   # Long (0)
    labels[price_changes < -k_pips] = 2  # Short (2)
    
    return labels

def validate_labels(labels: pd.Series, target_mode: str = 'binary') -> bool:
    """Validate label consistency and warn about issues.
    
    Args:
        labels: Label series
        target_mode: Expected target mode
        
    Returns:
        True if labels are valid, False otherwise
    """
    clean_labels = labels.dropna()
    unique_values = set(clean_labels.unique())
    
    if target_mode == 'binary':
        expected_values = {0, 1}
        if not unique_values.issubset(expected_values):
            logger.error(f"Invalid binary labels: found {unique_values}, expected subset of {expected_values}")
            return False
    elif target_mode == 'three_class':
        expected_values = {0, 1, 2}
        if not unique_values.issubset(expected_values):
            logger.error(f"Invalid three-class labels: found {unique_values}, expected subset of {expected_values}")
            return False
    
    # Check for single class
    if len(unique_values) == 1:
        logger.warning(f"Single class detected in {target_mode} mode: {unique_values}")
    
    return True

def get_optimal_k_pips_range(df: pd.DataFrame, horizon: int, 
                           target_mode: str = 'binary') -> Tuple[float, float, float]:
    """Get optimal k-pips range for experimentation.
    
    Args:
        df: OHLC DataFrame
        horizon: Prediction horizon
        target_mode: Target mode
        
    Returns:
        Tuple of (conservative, balanced, aggressive) k-pips values
    """
    atr_period = max(4, horizon // 2)
    atr = calculate_atr(df, period=atr_period)
    atr_mean = atr.mean()
    avg_spread = df['spread'].mean()
    
    # Different ranges for different target modes
    if target_mode == 'three_class':
        conservative = max(0.15 * atr_mean, 0.6 * avg_spread)
        balanced = max(0.25 * atr_mean, 0.8 * avg_spread)
        aggressive = max(0.35 * atr_mean, 1.0 * avg_spread)
    else:
        conservative = max(0.1 * atr_mean, 0.5 * avg_spread)
        balanced = max(0.2 * atr_mean, 0.8 * avg_spread)
        aggressive = max(0.3 * atr_mean, 1.2 * avg_spread)
    
    return conservative, balanced, aggressive

def auto_adjust_k_pips(df: pd.DataFrame, horizon: int, initial_k_pips: float,
                      target_mode: str = 'binary', max_iterations: int = 3) -> float:
    """Automatically adjust k-pips to avoid single class problems.
    
    Args:
        df: OHLC DataFrame
        horizon: Prediction horizon
        initial_k_pips: Starting k-pips value
        target_mode: Target mode
        max_iterations: Maximum adjustment iterations
        
    Returns:
        Adjusted k-pips value
    """
    current_k_pips = initial_k_pips
    
    for iteration in range(max_iterations):
        labels = create_labels(df, current_k_pips, horizon, target_mode)
        unique_classes = len(labels.dropna().unique())
        
        if target_mode == 'binary' and unique_classes >= 2:
            break
        elif target_mode == 'three_class' and unique_classes >= 2:  # Accept 2+ classes for three_class
            break
        
        # Reduce k-pips by 20% each iteration
        current_k_pips *= 0.8
        logger.info(f"Auto-adjusting k-pips to {current_k_pips:.6f} (iteration {iteration + 1})")
    
    if current_k_pips != initial_k_pips:
        logger.info(f"K-pips auto-adjusted from {initial_k_pips:.6f} to {current_k_pips:.6f}")
    
    return current_k_pips

__all__ = [
    'compute_k_pips',
    'create_labels', 
    'analyze_label_distribution',
    'encode_labels_three_class',
    'validate_labels',
    'get_optimal_k_pips_range',
    'auto_adjust_k_pips'
]
