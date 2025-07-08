# ========================
# src/ensemble/ensemble_monitor.py
"""Ensemble performance monitoring and analysis."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EnsembleMonitor:
    """Monitor ensemble performance and model contributions"""
    
    def __init__(self, ensemble_name: str = "ensemble", save_dir: str = "monitoring"):
        self.ensemble_name = ensemble_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        self.prediction_history = []
        self.model_contributions = []
        
        logger.info(f"📊 EnsembleMonitor initialized: {ensemble_name}")
        
    def log_prediction(self, predictions: Dict[str, torch.Tensor], 
                      true_labels: torch.Tensor, ensemble_pred: torch.Tensor,
                      market_metadata: Dict[str, Any] = None):
        """Log prediction results"""
        timestamp = datetime.now().isoformat()
        
        # Calculate individual accuracies
        individual_accs = {}
        for name, pred in predictions.items():
            if pred.dim() > 1:
                pred_classes = torch.argmax(pred, dim=1)
            else:
                pred_classes = (pred > 0.5).long()
                
            if true_labels.dim() > 1:
                true_classes = torch.argmax(true_labels, dim=1)
            else:
                true_classes = true_labels.long()
                
            acc = (pred_classes == true_classes).float().mean().item()
            individual_accs[name] = acc
        
        # Calculate ensemble accuracy
        if ensemble_pred.dim() > 1:
            ensemble_classes = torch.argmax(ensemble_pred, dim=1)
        else:
            ensemble_classes = (ensemble_pred > 0.5).long()
            
        if true_labels.dim() > 1:
            true_classes = torch.argmax(true_labels, dim=1)
        else:
            true_classes = true_labels.long()
            
        ensemble_acc = (ensemble_classes == true_classes).float().mean().item()
        
        # Calculate diversity metrics
        diversity_score = self._calculate_diversity(predictions)
        confidence_score = self._calculate_confidence(ensemble_pred)
        
        # Log metrics
        metric_entry = {
            'timestamp': timestamp,
            'ensemble_accuracy': ensemble_acc,
            'individual_accuracies': individual_accs,
            'diversity_score': diversity_score,
            'confidence_score': confidence_score,
            'batch_size': len(true_labels),
            'market_metadata': market_metadata or {}
        }
        
        self.metrics_history.append(metric_entry)
        
        # Log detailed predictions
        pred_entry = {
            'timestamp': timestamp,
            'predictions': {name: pred.cpu().numpy().tolist() 
                          for name, pred in predictions.items()},
            'ensemble_prediction': ensemble_pred.cpu().numpy().tolist(),
            'true_labels': true_labels.cpu().numpy().tolist()
        }
        
        self.prediction_history.append(pred_entry)
        
        logger.debug(f"📈 Logged prediction: ensemble_acc={ensemble_acc:.4f}, diversity={diversity_score:.4f}")
    
    def _calculate_diversity(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Calculate prediction diversity using disagreement rate"""
        pred_list = list(predictions.values())
        if len(pred_list) < 2:
            return 0.0
        
        # Convert to hard predictions
        hard_preds = []
        for pred in pred_list:
            if pred.dim() > 1:
                hard_pred = torch.argmax(pred, dim=1)
            else:
                hard_pred = (pred > 0.5).long()
            hard_preds.append(hard_pred)
        
        # Calculate pairwise disagreement
        total_pairs = 0
        total_disagreement = 0
        
        for i in range(len(hard_preds)):
            for j in range(i + 1, len(hard_preds)):
                disagreement = (hard_preds[i] != hard_preds[j]).float().mean().item()
                total_disagreement += disagreement
                total_pairs += 1
        
        return total_disagreement / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_confidence(self, prediction: torch.Tensor) -> float:
        """Calculate prediction confidence"""
        if prediction.dim() > 1:
            # For multi-class, use max probability
            max_probs = torch.max(prediction, dim=1)[0]
            return max_probs.mean().item()
        else:
            # For binary, use distance from 0.5
            confidence = torch.abs(prediction - 0.5) * 2
            return confidence.mean().item()
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No metrics recorded"}
        
        # Extract metrics
        ensemble_accs = [m['ensemble_accuracy'] for m in self.metrics_history]
        diversity_scores = [m['diversity_score'] for m in self.metrics_history]
        confidence_scores = [m['confidence_score'] for m in self.metrics_history]
        
        # Individual model performance
        all_individual_accs = {}
        for metric in self.metrics_history:
            for model_name, acc in metric['individual_accuracies'].items():
                if model_name not in all_individual_accs:
                    all_individual_accs[model_name] = []
                all_individual_accs[model_name].append(acc)
        
        model_stats = {}
        for model_name, accs in all_individual_accs.items():
            model_stats[model_name] = {
                'mean_accuracy': np.mean(accs),
                'std_accuracy': np.std(accs),
                'min_accuracy': np.min(accs),
                'max_accuracy': np.max(accs),
                'recent_trend': np.mean(accs[-10:]) - np.mean(accs[:10]) if len(accs) >= 20 else 0.0
            }
        
        # Ensemble vs best individual
        best_individual_accs = []
        for metric in self.metrics_history:
            best_acc = max(metric['individual_accuracies'].values())
            best_individual_accs.append(best_acc)
        
        improvement_over_best = np.mean(ensemble_accs) - np.mean(best_individual_accs)
        
        report = {
            'summary': {
                'total_predictions': len(self.metrics_history),
                'ensemble_performance': {
                    'mean_accuracy': np.mean(ensemble_accs),
                    'std_accuracy': np.std(ensemble_accs),
                    'best_accuracy': np.max(ensemble_accs),
                    'worst_accuracy': np.min(ensemble_accs)
                },
                'diversity_metrics': {
                    'mean_diversity': np.mean(diversity_scores),
                    'std_diversity': np.std(diversity_scores)
                },
                'confidence_metrics': {
                    'mean_confidence': np.mean(confidence_scores),
                    'std_confidence': np.std(confidence_scores)
                },
                'improvement_over_best_individual': improvement_over_best
            },
            'individual_models': model_stats,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance-based improvement recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return ["No data available for recommendations"]
        
        # Analyze recent performance
        recent_metrics = self.metrics_history[-50:] if len(self.metrics_history) >= 50 else self.metrics_history
        
        # Check individual model performance
        for metric in recent_metrics:
            for model_name, acc in metric['individual_accuracies'].items():
                if acc < 0.55:
                    recommendations.append(
                        f"🔴 {model_name} performance düşük (<55%). Retraining gerekli."
                    )
                    break
        
        # Check diversity
        recent_diversity = [m['diversity_score'] for m in recent_metrics]
        if np.mean(recent_diversity) < 0.2:
            recommendations.append(
                "⚠️ Model diversity düşük. Farklı mimariler veya hiperparametreler deneyin."
            )
        
        # Check confidence vs accuracy relationship
        recent_confidence = [m['confidence_score'] for m in recent_metrics]
        recent_accuracy = [m['ensemble_accuracy'] for m in recent_metrics]
        
        if np.corrcoef(recent_confidence, recent_accuracy)[0, 1] < 0.3:
            recommendations.append(
                "🎯 Confidence ve accuracy arasında zayıf korelasyon. Calibration gerekli."
            )
        
        # Check ensemble improvement
        recent_ensemble_accs = [m['ensemble_accuracy'] for m in recent_metrics]
        recent_best_individual = []
        for m in recent_metrics:
            best_individual = max(m['individual_accuracies'].values())
            recent_best_individual.append(best_individual)
        
        improvement = np.mean(recent_ensemble_accs) - np.mean(recent_best_individual)
        if improvement < 0.02:
            recommendations.append(
                "📈 Ensemble improvement düşük (<2%). Meta-learning veya advanced voting strategies kullanın."
            )
        
        if not recommendations:
            recommendations.append("✅ Ensemble performance iyi görünüyor!")
        
        return recommendations
    
    def save_report(self, filepath: str = None):
        """Save performance report to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"ensemble_report_{timestamp}.json"
        
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Performance report saved: {filepath}")
        return filepath
    
    def create_visualization_dashboard(self, save_path: str = None) -> str:
        """Create comprehensive visualization dashboard"""
        if not self.metrics_history:
            logger.warning("No metrics to visualize")
            return None
        
        # Setup figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Ensemble Performance Dashboard: {self.ensemble_name}', fontsize=16)
        
        # Extract data
        timestamps = [m['timestamp'] for m in self.metrics_history]
        ensemble_accs = [m['ensemble_accuracy'] for m in self.metrics_history]
        diversity_scores = [m['diversity_score'] for m in self.metrics_history]
        confidence_scores = [m['confidence_score'] for m in self.metrics_history]
        
        # 1. Ensemble Accuracy Over Time
        axes[0, 0].plot(ensemble_accs, marker='o', alpha=0.7)
        axes[0, 0].set_title('Ensemble Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Individual vs Ensemble Accuracy
        all_individual_accs = {}
        for metric in self.metrics_history:
            for model_name, acc in metric['individual_accuracies'].items():
                if model_name not in all_individual_accs:
                    all_individual_accs[model_name] = []
                all_individual_accs[model_name].append(acc)
        
        # Plot individual models
        for model_name, accs in all_individual_accs.items():
            axes[0, 1].plot(accs, alpha=0.6, label=model_name)
        axes[0, 1].plot(ensemble_accs, 'k-', linewidth=2, label='Ensemble')
        axes[0, 1].set_title('Individual vs Ensemble Performance')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Performance Distribution
        model_means = [np.mean(accs) for accs in all_individual_accs.values()]
        model_names = list(all_individual_accs.keys())
        
        bars = axes[0, 2].bar(range(len(model_names)), model_means, alpha=0.7)
        axes[0, 2].axhline(np.mean(ensemble_accs), color='red', linestyle='--', label='Ensemble Mean')
        axes[0, 2].set_title('Mean Model Performance')
        axes[0, 2].set_ylabel('Mean Accuracy')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Diversity Score Over Time
        axes[1, 0].plot(diversity_scores, color='green', marker='s', alpha=0.7)
        axes[1, 0].set_title('Model Diversity Over Time')
        axes[1, 0].set_ylabel('Diversity Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Confidence vs Accuracy Scatter
        axes[1, 1].scatter(confidence_scores, ensemble_accs, alpha=0.6)
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Ensemble Accuracy')
        axes[1, 1].set_title('Confidence vs Accuracy')
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(confidence_scores, ensemble_accs)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Ensemble Improvement Distribution
        ensemble_improvements = []
        for metric in self.metrics_history:
            best_individual = max(metric['individual_accuracies'].values())
            improvement = metric['ensemble_accuracy'] - best_individual
            ensemble_improvements.append(improvement)
        
        axes[1, 2].hist(ensemble_improvements, bins=20, alpha=0.7, color='purple')
        axes[1, 2].axvline(np.mean(ensemble_improvements), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(ensemble_improvements):.4f}')
        axes[1, 2].set_title('Ensemble Improvement Distribution')
        axes[1, 2].set_xlabel('Improvement over Best Individual')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save dashboard
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"ensemble_dashboard_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Dashboard saved: {save_path}")
        return str(save_path)