# src/backtest/trade_logic.py
"""Trade logic engine with threshold analysis and signal generation."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TradeSignalEngine:
    """Advanced trade signal engine with threshold optimization."""
    
    def __init__(self, signal_threshold: float = 0.5, 
                 k_pips_mult: float = 1.0,
                 target_mode: str = 'binary',
                 pair_name: str = None):
        self.signal_threshold = signal_threshold
        self.k_pips_mult = k_pips_mult
        self.target_mode = target_mode
        self.pair_name = pair_name
        
        # Prediction tracking for threshold analysis
        self.prediction_history = []
        self.threshold_analysis = {}
        
    def analyze_prediction_distribution(self, predictions: List[Dict], 
                                      window_id: int, 
                                      save_plots: bool = True) -> Dict[str, Any]:
        """Analyze prediction distribution and recommend thresholds.
        
        Args:
            predictions: List of model predictions
            window_id: Training window identifier
            save_plots: Whether to save histogram plots
            
        Returns:
            Analysis results with threshold recommendations
        """
        if not predictions:
            return {'status': 'no_predictions'}
        
        # Extract prediction probabilities
        if self.target_mode == 'binary':
            probs = [pred.get('confidence', 0.5) for pred in predictions]
            signals = [pred.get('signal', 0) for pred in predictions]
        else:  # three_class
            probs = []
            signals = []
            for pred in predictions:
                raw_output = pred.get('raw_output', [[0.33, 0.34, 0.33]])
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    # Get max probability and corresponding class
                    class_probs = raw_output[0] if isinstance(raw_output[0], list) else raw_output
                    max_prob = max(class_probs)
                    max_class = class_probs.index(max_prob)
                    probs.append(max_prob)
                    signals.append(max_class)
        
        probs = np.array(probs)
        signals = np.array(signals)
        
        # Calculate distribution statistics
        analysis = {
            'window_id': window_id,
            'total_predictions': len(probs),
            'mean_confidence': np.mean(probs),
            'std_confidence': np.std(probs),
            'min_confidence': np.min(probs),
            'max_confidence': np.max(probs),
            'median_confidence': np.median(probs),
            'q25_confidence': np.percentile(probs, 25),
            'q75_confidence': np.percentile(probs, 75)
        }
        
        # Signal distribution
        unique_signals, signal_counts = np.unique(signals, return_counts=True)
        signal_dist = dict(zip(unique_signals, signal_counts))
        analysis['signal_distribution'] = signal_dist
        
        # Threshold recommendations
        if self.target_mode == 'binary':
            analysis.update(self._analyze_binary_thresholds(probs, signals))
        else:
            analysis.update(self._analyze_multiclass_thresholds(probs, signals))
        
        # Save histogram for first two windows
        if save_plots and window_id <= 2:
            self._save_prediction_histogram(probs, signals, window_id, analysis)
        
        # Store for overall analysis
        self.threshold_analysis[window_id] = analysis
        
        return analysis
    
    def _analyze_binary_thresholds(self, probs: np.ndarray, 
                                  signals: np.ndarray) -> Dict[str, Any]:
        """Analyze optimal thresholds for binary classification."""
        
        # Test different thresholds
        test_thresholds = np.arange(0.3, 0.8, 0.05)
        threshold_analysis = []
        
        for thresh in test_thresholds:
            long_signals = (probs > thresh).sum()
            short_signals = (probs < (1 - thresh)).sum()
            flat_signals = len(probs) - long_signals - short_signals
            
            trade_ratio = (long_signals + short_signals) / len(probs)
            
            threshold_analysis.append({
                'threshold': thresh,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'flat_signals': flat_signals,
                'trade_ratio': trade_ratio,
                'total_trades': long_signals + short_signals
            })
        
        # Find optimal threshold (target 10-30% trade ratio)
        optimal_thresh = 0.5
        target_trade_ratio = 0.2  # 20% of predictions become trades
        
        best_diff = float('inf')
        for analysis in threshold_analysis:
            diff = abs(analysis['trade_ratio'] - target_trade_ratio)
            if diff < best_diff:
                best_diff = diff
                optimal_thresh = analysis['threshold']
        
        return {
            'threshold_analysis': threshold_analysis,
            'recommended_threshold': optimal_thresh,
            'current_trade_ratio': next(
                (a['trade_ratio'] for a in threshold_analysis 
                 if abs(a['threshold'] - self.signal_threshold) < 0.01), 
                0.0
            )
        }
    
    def _analyze_multiclass_thresholds(self, probs: np.ndarray, 
                                     signals: np.ndarray) -> Dict[str, Any]:
        """Analyze optimal thresholds for three-class classification."""
        
        # For three-class, analyze confidence distribution per class
        class_probs = {0: [], 1: [], 2: []}  # long, flat, short
        
        for prob, signal in zip(probs, signals):
            class_probs[signal].append(prob)
        
        class_analysis = {}
        for class_id, class_prob_list in class_probs.items():
            if class_prob_list:
                class_analysis[class_id] = {
                    'count': len(class_prob_list),
                    'mean_confidence': np.mean(class_prob_list),
                    'std_confidence': np.std(class_prob_list),
                    'min_confidence': np.min(class_prob_list),
                    'max_confidence': np.max(class_prob_list)
                }
            else:
                class_analysis[class_id] = {
                    'count': 0,
                    'mean_confidence': 0,
                    'std_confidence': 0,
                    'min_confidence': 0,
                    'max_confidence': 0
                }
        
        # Recommend threshold based on mean confidence
        mean_confidences = [class_analysis[i]['mean_confidence'] for i in [0, 2]]  # long, short only
        recommended_threshold = max(0.4, np.mean([c for c in mean_confidences if c > 0]))
        
        return {
            'class_analysis': class_analysis,
            'recommended_threshold': min(0.7, recommended_threshold)
        }
    
    def _save_prediction_histogram(self, probs: np.ndarray, signals: np.ndarray,
                                  window_id: int, analysis: Dict[str, Any]) -> None:
        """Save prediction distribution histogram."""
        
        try:
            # Create output directory
            plot_dir = Path("backtest_results/threshold_analysis")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Probability distribution
            ax1.hist(probs, bins=30, alpha=0.7, edgecolor='black')
            ax1.axvline(self.signal_threshold, color='red', linestyle='--', 
                       label=f'Current Threshold: {self.signal_threshold:.2f}')
            ax1.axvline(analysis.get('recommended_threshold', 0.5), color='green', 
                       linestyle='--', label=f'Recommended: {analysis.get("recommended_threshold", 0.5):.2f}')
            ax1.set_xlabel('Prediction Confidence/Probability')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{self.pair_name} - Window {window_id} - Probability Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Signal distribution
            signal_dist = analysis['signal_distribution']
            signal_labels = []
            signal_values = []
            
            if self.target_mode == 'binary':
                signal_labels = ['Short/Flat (0)', 'Long (1)']
                signal_values = [signal_dist.get(0, 0), signal_dist.get(1, 0)]
            else:
                signal_labels = ['Long (0)', 'Flat (1)', 'Short (2)']
                signal_values = [signal_dist.get(0, 0), signal_dist.get(1, 0), signal_dist.get(2, 0)]
            
            bars = ax2.bar(signal_labels, signal_values, alpha=0.7)
            ax2.set_ylabel('Count')
            ax2.set_title(f'{self.pair_name} - Window {window_id} - Signal Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, signal_values):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(signal_values),
                            f'{value}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = plot_dir / f"{self.pair_name}_window_{window_id}_threshold_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   📊 Threshold analysis plot saved: {plot_path}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to save threshold analysis plot: {e}")
    
    def generate_trade_signals(self, predictions: List[Dict], 
                             k_pips: float,
                             risk_filters: Dict[str, Any] = None) -> List[Dict]:
        """Generate trade signals with edge detection and risk filtering.
        
        Args:
            predictions: Model predictions
            k_pips: K-pips threshold for edge detection
            risk_filters: Additional risk filters (session, news, etc.)
            
        Returns:
            List of trade signals
        """
        trade_signals = []
        filtered_count = 0
        
        for pred in predictions:
            signal_dict = {
                'timestamp': pred['timestamp'],
                'raw_prediction': pred,
                'signal_type': 'flat',
                'confidence': 0.0,
                'edge_pips': 0.0,
                'risk_filters_passed': True,
                'filter_reasons': []
            }
            
            # Extract probabilities based on target mode
            if self.target_mode == 'binary':
                prob = pred.get('confidence', 0.5)
                signal = pred.get('signal', 0)
                
                if signal == 1 and prob > self.signal_threshold:
                    # Check edge requirement
                    edge_pips = (prob - 0.5) * 2 * k_pips  # Convert confidence to expected pips
                    if edge_pips > k_pips:
                        signal_dict.update({
                            'signal_type': 'long',
                            'confidence': prob,
                            'edge_pips': edge_pips
                        })
                
            else:  # three_class
                raw_output = pred.get('raw_output', [[0.33, 0.34, 0.33]])
                if isinstance(raw_output, list) and len(raw_output) > 0:
                    class_probs = raw_output[0] if isinstance(raw_output[0], list) else raw_output
                    
                    if len(class_probs) >= 3:
                        prob_long = class_probs[0]
                        prob_flat = class_probs[1] 
                        prob_short = class_probs[2]
                        
                        max_prob = max(class_probs)
                        max_class = class_probs.index(max_prob)
                        
                        if max_prob > self.signal_threshold:
                            edge_pips = (max_prob - 0.5) * 2 * k_pips
                            
                            if max_class == 0 and edge_pips > k_pips:  # Long
                                signal_dict.update({
                                    'signal_type': 'long',
                                    'confidence': prob_long,
                                    'edge_pips': edge_pips
                                })
                            elif max_class == 2 and edge_pips > k_pips:  # Short
                                signal_dict.update({
                                    'signal_type': 'short',
                                    'confidence': prob_short,
                                    'edge_pips': edge_pips
                                })
                            # Class 1 (flat) doesn't generate trades
            
            # Apply risk filters
            if signal_dict['signal_type'] != 'flat' and risk_filters:
                signal_dict = self._apply_risk_filters(signal_dict, risk_filters)
                if not signal_dict['risk_filters_passed']:
                    filtered_count += 1
            
            # Only add non-flat signals that pass filters
            if signal_dict['signal_type'] != 'flat' and signal_dict['risk_filters_passed']:
                trade_signals.append(signal_dict)
        
        # Log filtering statistics
        total_predictions = len(predictions)
        potential_trades = len([s for s in trade_signals if s['signal_type'] != 'flat']) + filtered_count
        
        logger.info(f"   🎯 Signal generation: {len(trade_signals)} trades from {total_predictions} predictions")
        logger.info(f"   📊 Trade ratio: {len(trade_signals)/total_predictions*100:.1f}%")
        
        if filtered_count > 0:
            logger.info(f"   🛡️ Risk filters: {filtered_count} signals filtered out")
        
        if filtered_count > potential_trades * 0.8:
            logger.warning(f"   ⚠️ WARNING: Risk filters eliminated {filtered_count}/{potential_trades} signals!")
        
        return trade_signals
    
    def _apply_risk_filters(self, signal_dict: Dict[str, Any], 
                           risk_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk filters to trade signals."""
        
        filters_passed = True
        filter_reasons = []
        
        timestamp = signal_dict['timestamp']
        
        # Session filter
        if 'session_filter' in risk_filters:
            session_ok = self._check_session_filter(timestamp, risk_filters['session_filter'])
            if not session_ok:
                filters_passed = False
                filter_reasons.append('outside_trading_session')
        
        # Volatility filter
        if 'min_volatility' in risk_filters:
            vol_ok = self._check_volatility_filter(timestamp, risk_filters['min_volatility'])
            if not vol_ok:
                filters_passed = False
                filter_reasons.append('low_volatility')
        
        # News filter
        if 'avoid_news' in risk_filters:
            news_ok = self._check_news_filter(timestamp, risk_filters['avoid_news'])
            if not news_ok:
                filters_passed = False
                filter_reasons.append('news_time')
        
        signal_dict['risk_filters_passed'] = filters_passed
        signal_dict['filter_reasons'] = filter_reasons
        
        return signal_dict
    
    def _check_session_filter(self, timestamp: pd.Timestamp, session_config: Dict) -> bool:
        """Check if timestamp is within allowed trading sessions."""
        hour = timestamp.hour
        
        # Default: allow major sessions
        allowed_sessions = session_config.get('allowed', ['london', 'ny', 'overlap'])
        
        london_session = (8 <= hour <= 16)
        ny_session = (13 <= hour <= 21)
        tokyo_session = (23 <= hour <= 7) or (0 <= hour <= 7)
        overlap_session = (13 <= hour <= 16)
        
        if 'london' in allowed_sessions and london_session:
            return True
        if 'ny' in allowed_sessions and ny_session:
            return True
        if 'tokyo' in allowed_sessions and tokyo_session:
            return True
        if 'overlap' in allowed_sessions and overlap_session:
            return True
            
        return False
    
    def _check_volatility_filter(self, timestamp: pd.Timestamp, min_vol: float) -> bool:
        """Check minimum volatility requirement."""
        # Simplified - in practice would need recent volatility data
        return True  # Placeholder
    
    def _check_news_filter(self, timestamp: pd.Timestamp, news_config: Dict) -> bool:
        """Check if timestamp conflicts with news events."""
        # Simplified - in practice would need news calendar
        return True  # Placeholder
    
    def get_threshold_recommendations(self) -> Dict[str, Any]:
        """Get overall threshold recommendations from all windows."""
        
        if not self.threshold_analysis:
            return {'status': 'no_analysis'}
        
        # Aggregate recommendations from all windows
        all_recommendations = []
        all_trade_ratios = []
        
        for window_id, analysis in self.threshold_analysis.items():
            rec_thresh = analysis.get('recommended_threshold', self.signal_threshold)
            trade_ratio = analysis.get('current_trade_ratio', 0.0)
            
            all_recommendations.append(rec_thresh)
            all_trade_ratios.append(trade_ratio)
        
        # Calculate final recommendation
        if all_recommendations:
            final_recommendation = np.mean(all_recommendations)
            avg_trade_ratio = np.mean(all_trade_ratios)
            
            return {
                'status': 'success',
                'current_threshold': self.signal_threshold,
                'recommended_threshold': final_recommendation,
                'current_trade_ratio': avg_trade_ratio,
                'recommendation_std': np.std(all_recommendations),
                'windows_analyzed': len(self.threshold_analysis),
                'should_adjust': abs(final_recommendation - self.signal_threshold) > 0.05
            }
        
        return {'status': 'insufficient_data'}

__all__ = ['TradeSignalEngine']
