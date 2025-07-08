# src/backtest/threshold_optimizer.py
"""ROC-based threshold optimization for optimal signal generation."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

class ThresholdOptimizer:
    """ROC-based threshold optimization for binary and three-class models."""
    
    def __init__(self, pair_name: str, target_mode: str = 'binary'):
        self.pair_name = pair_name
        self.target_mode = target_mode
        
    def optimize_threshold_roc(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              method: str = 'youden') -> Dict[str, float]:
        """Optimize threshold using ROC analysis.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            method: Optimization method ('youden', 'f1', 'precision_recall')
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if self.target_mode == 'binary':
            return self._optimize_binary_threshold(y_true, y_prob, method)
        else:
            return self._optimize_multiclass_threshold(y_true, y_prob, method)
    
    def _optimize_binary_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  method: str) -> Dict[str, float]:
        """Optimize threshold for binary classification."""
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        optimal_threshold = 0.5
        best_score = 0.0
        method_name = method
        
        if method == 'youden':
            # Youden's J statistic (maximize TPR - FPR)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[best_idx]
            best_score = j_scores[best_idx]
            method_name = "Youden's J"
            
        elif method == 'f1':
            # Maximize F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
            best_score = f1_scores[best_idx]
            method_name = "F1 Score"
            
        elif method == 'precision_recall':
            # Balance precision and recall (minimize |precision - recall|)
            pr_diff = np.abs(precision - recall)
            best_idx = np.argmin(pr_diff)
            optimal_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
            best_score = 1 - pr_diff[best_idx]  # Higher is better
            method_name = "Precision-Recall Balance"
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        
        # Trade frequency analysis
        trade_frequency = np.mean(y_pred_optimal)
        
        # Final validation - ensure reasonable trade frequency (5-30%)
        if trade_frequency < 0.05:
            print(f"   ⚠️ Trade frequency too low ({trade_frequency:.1%}), adjusting threshold")
            optimal_threshold = np.percentile(y_prob, 95)  # Top 5% predictions
            trade_frequency = 0.05
        elif trade_frequency > 0.30:
            print(f"   ⚠️ Trade frequency too high ({trade_frequency:.1%}), adjusting threshold")
            optimal_threshold = np.percentile(y_prob, 70)  # Top 30% predictions
            trade_frequency = 0.30
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'method': method_name,
            'score': float(best_score),
            'trade_frequency': float(trade_frequency),
            'original_threshold': 0.5,
            'improvement': float(optimal_threshold - 0.5)
        }
    
    def _optimize_multiclass_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                     method: str) -> Dict[str, float]:
        """Optimize threshold for three-class classification."""
        
        # For three-class, we optimize the confidence threshold
        # y_prob should be the max probability across classes
        if len(y_prob.shape) > 1:
            max_probs = np.max(y_prob, axis=1)
            predicted_classes = np.argmax(y_prob, axis=1)
        else:
            max_probs = y_prob
            predicted_classes = y_true  # Fallback
        
        # Test different confidence thresholds
        test_thresholds = np.linspace(0.4, 0.8, 50)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in test_thresholds:
            # Apply threshold - only make predictions above threshold
            confident_predictions = max_probs >= threshold
            
            if np.sum(confident_predictions) == 0:
                continue
                
            # Calculate accuracy on confident predictions
            confident_y_true = y_true[confident_predictions]
            confident_y_pred = predicted_classes[confident_predictions]
            
            if len(confident_y_true) > 0:
                accuracy = np.mean(confident_y_true == confident_y_pred)
                trade_frequency = np.mean(confident_predictions)
                
                # Combined score: accuracy weighted by reasonable trade frequency
                if 0.05 <= trade_frequency <= 0.30:  # Reasonable range
                    score = accuracy * (1.0 - abs(trade_frequency - 0.15) / 0.15)  # Prefer ~15%
                else:
                    score = accuracy * 0.5  # Penalize extreme frequencies
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        # Calculate final metrics
        confident_mask = max_probs >= best_threshold
        final_trade_frequency = np.mean(confident_mask)
        
        return {
            'optimal_threshold': float(best_threshold),
            'method': f"Confidence-based ({method})",
            'score': float(best_score),
            'trade_frequency': float(final_trade_frequency),
            'original_threshold': 0.5,
            'improvement': float(best_threshold - 0.5)
        }
    
    def save_threshold_analysis(self, y_true: np.ndarray, y_prob: np.ndarray, 
                               optimization_result: Dict[str, float],
                               window_id: int) -> None:
        """Save threshold analysis plots."""
        
        try:
            # Create output directory
            plot_dir = Path("backtest_results/threshold_optimization")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            if self.target_mode == 'binary':
                self._plot_binary_analysis(y_true, y_prob, optimization_result, window_id, plot_dir)
            else:
                self._plot_multiclass_analysis(y_true, y_prob, optimization_result, window_id, plot_dir)
                
        except Exception as e:
            print(f"   WARNING: Failed to save threshold analysis: {e}")
    
    def _plot_binary_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                             result: Dict[str, float], window_id: int, plot_dir: Path) -> None:
        """Plot binary threshold analysis."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
        ax1.axvline(x=result['optimal_threshold'], color='green', linestyle='--', 
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{self.pair_name} - ROC Curve (Window {window_id})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        ax2.plot(recall, precision, 'g-', linewidth=2, label='PR Curve')
        ax2.axhline(y=result['optimal_threshold'], color='green', linestyle='--',
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{self.pair_name} - Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Threshold vs Metrics
        test_thresholds = np.linspace(0.1, 0.9, 50)
        trade_freqs = []
        accuracies = []
        
        for thresh in test_thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            trade_freq = np.mean(y_pred)
            accuracy = np.mean(y_true == y_pred)
            trade_freqs.append(trade_freq)
            accuracies.append(accuracy)
        
        ax3.plot(test_thresholds, trade_freqs, 'b-', label='Trade Frequency')
        ax3.plot(test_thresholds, accuracies, 'r-', label='Accuracy')
        ax3.axvline(x=result['optimal_threshold'], color='green', linestyle='--',
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Rate')
        ax3.set_title(f'{self.pair_name} - Threshold Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Probability Distribution
        ax4.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Class 0', color='red')
        ax4.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Class 1', color='blue')
        ax4.axvline(x=result['optimal_threshold'], color='green', linestyle='--',
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Default: 0.5')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'{self.pair_name} - Probability Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = plot_dir / f"{self.pair_name}_threshold_analysis_window_{window_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Threshold analysis saved: {plot_path}")
    
    def _plot_multiclass_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 result: Dict[str, float], window_id: int, plot_dir: Path) -> None:
        """Plot multiclass threshold analysis - simplified version."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get max probabilities
        if len(y_prob.shape) > 1:
            max_probs = np.max(y_prob, axis=1)
        else:
            max_probs = y_prob
        
        # 1. Confidence distribution by class
        for class_id in np.unique(y_true):
            class_probs = max_probs[y_true == class_id]
            ax1.hist(class_probs, bins=20, alpha=0.7, label=f'Class {class_id}')
        
        ax1.axvline(x=result['optimal_threshold'], color='green', linestyle='--',
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax1.set_xlabel('Max Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{self.pair_name} - Confidence Distribution (Window {window_id})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trade frequency vs threshold
        test_thresholds = np.linspace(0.4, 0.8, 30)
        trade_freqs = []
        
        for thresh in test_thresholds:
            trade_freq = np.mean(max_probs >= thresh)
            trade_freqs.append(trade_freq)
        
        ax2.plot(test_thresholds, trade_freqs, 'b-', linewidth=2, label='Trade Frequency')
        ax2.axvline(x=result['optimal_threshold'], color='green', linestyle='--',
                   label=f'Optimal: {result["optimal_threshold"]:.3f}')
        ax2.axhline(y=0.15, color='orange', linestyle=':', alpha=0.7, label='Target: 15%')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Trade Frequency')
        ax2.set_title(f'{self.pair_name} - Trade Frequency vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = plot_dir / f"{self.pair_name}_confidence_analysis_window_{window_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Confidence analysis saved: {plot_path}")

__all__ = ['ThresholdOptimizer']