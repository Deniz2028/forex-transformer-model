# ========================
# src/utils/training_monitor.py - YENİ DOSYA
# ========================

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Advanced training monitoring and visualization"""
    
    def __init__(self, save_dir: str = "training_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.training_data = {}
        self.ensemble_metrics = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"📊 TrainingMonitor initialized, saving to {save_dir}")
    
    def log_training_progress(self, model_name: str, epoch: int, metrics: Dict[str, float]):
        """Log training progress for a model"""
        
        if model_name not in self.training_data:
            self.training_data[model_name] = {
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'learning_rate': [],
                'timestamps': []
            }
        
        data = self.training_data[model_name]
        data['epochs'].append(epoch)
        data['train_loss'].append(metrics.get('train_loss', 0))
        data['val_loss'].append(metrics.get('val_loss', 0))
        data['train_acc'].append(metrics.get('train_acc', 0))
        data['val_acc'].append(metrics.get('val_acc', 0))
        data['learning_rate'].append(metrics.get('learning_rate', 0))
        data['timestamps'].append(datetime.now().isoformat())
    
    def plot_training_curves(self, save_plots: bool = True) -> None:
        """Generate comprehensive training curves"""
        
        if not self.training_data:
            logger.warning("No training data available for plotting")
            return
        
        n_models = len(self.training_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble Training Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss Comparison
        ax1 = axes[0, 0]
        for model_name, data in self.training_data.items():
            if data['train_loss']:
                ax1.plot(data['epochs'], data['train_loss'], 
                        label=model_name, linewidth=2, alpha=0.8)
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss Comparison
        ax2 = axes[0, 1]
        for model_name, data in self.training_data.items():
            if data['val_loss']:
                ax2.plot(data['epochs'], data['val_loss'], 
                        label=model_name, linewidth=2, alpha=0.8)
        ax2.set_title('Validation Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Accuracy Comparison
        ax3 = axes[0, 2]
        for model_name, data in self.training_data.items():
            if data['train_acc']:
                train_acc_pct = [acc * 100 for acc in data['train_acc']]
                ax3.plot(data['epochs'], train_acc_pct, 
                        label=model_name, linewidth=2, alpha=0.8)
        ax3.set_title('Training Accuracy', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation Accuracy Comparison
        ax4 = axes[1, 0]
        for model_name, data in self.training_data.items():
            if data['val_acc']:
                val_acc_pct = [acc * 100 for acc in data['val_acc']]
                ax4.plot(data['epochs'], val_acc_pct, 
                        label=model_name, linewidth=2, alpha=0.8)
        ax4.set_title('Validation Accuracy', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Learning Rate Schedule
        ax5 = axes[1, 1]
        for model_name, data in self.training_data.items():
            if data['learning_rate']:
                ax5.plot(data['epochs'], data['learning_rate'], 
                        label=model_name, linewidth=2, alpha=0.8)
        ax5.set_title('Learning Rate Schedule', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance Summary
        ax6 = axes[1, 2]
        final_val_accs = []
        model_names = []
        
        for model_name, data in self.training_data.items():
            if data['val_acc']:
                final_val_accs.append(data['val_acc'][-1] * 100)
                model_names.append(model_name.replace('model_', 'M'))
        
        if final_val_accs:
            bars = ax6.bar(model_names, final_val_accs, alpha=0.7)
            ax6.set_title('Final Validation Accuracy', fontweight='bold')
            ax6.set_ylabel('Accuracy (%)')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, final_val_accs):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.save_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training curves saved to {plot_path}")
        
        plt.show()
    
    def plot_ensemble_performance(self, ensemble_results: Dict[str, Any], 
                                save_plots: bool = True) -> None:
        """Plot ensemble vs individual model performance"""
        
        if not ensemble_results:
            logger.warning("No ensemble results available for plotting")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Ensemble Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        ax1 = axes[0]
        individual_accs = ensemble_results.get('individual_accuracies', {})
        ensemble_acc = ensemble_results.get('ensemble_accuracy', 0)
        
        if individual_accs:
            model_names = list(individual_accs.keys())
            model_accs = [acc * 100 for acc in individual_accs.values()]
            
            bars = ax1.bar(model_names, model_accs, alpha=0.7, label='Individual Models')
            ax1.axhline(y=ensemble_acc * 100, color='red', linestyle='--', 
                       linewidth=3, label=f'Ensemble ({ensemble_acc*100:.1f}%)')
            
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, acc in zip(bars, model_accs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Improvement Analysis
        ax2 = axes[1]
        if individual_accs:
            best_individual = max(individual_accs.values())
            improvement = ensemble_acc - best_individual
            improvement_pct = (improvement / best_individual) * 100
            
            categories = ['Best Individual', 'Ensemble']
            values = [best_individual * 100, ensemble_acc * 100]
            colors = ['lightblue', 'orange']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_title('Ensemble Improvement')
            ax2.set_ylabel('Accuracy (%)')
            
            # Add improvement annotation
            ax2.annotate(f'+{improvement*100:.2f}%\n({improvement_pct:.1f}% improvement)',
                        xy=(1, ensemble_acc * 100), xytext=(1.2, ensemble_acc * 100 + 2),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontweight='bold', color='green')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Model Contribution Weights
        ax3 = axes[2]
        model_weights = ensemble_results.get('model_weights', {})
        
        if model_weights:
            weights_data = [(k.replace('model_', 'M'), v) for k, v in model_weights.items()]
            model_names, weights = zip(*weights_data)
            
            wedges, texts, autotexts = ax3.pie(weights, labels=model_names, autopct='%1.1f%%',
                                              startangle=90, colors=sns.color_palette("husl", len(weights)))
            ax3.set_title('Model Contribution Weights')
            
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.save_dir / f"ensemble_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Ensemble performance plots saved to {plot_path}")
        
        plt.show()
    
    def generate_training_report(self, ensemble_results: Dict[str, Any] = None) -> str:
        """Generate comprehensive training report"""
        
        report = []
        report.append("=" * 60)
        report.append("🎯 ENSEMBLE TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Training Summary
        report.append("📊 TRAINING SUMMARY")
        report.append("-" * 30)
        
        if self.training_data:
            total_models = len(self.training_data)
            report.append(f"Total models trained: {total_models}")
            
            # Calculate training statistics
            total_epochs = sum(len(data['epochs']) for data in self.training_data.values())
            avg_epochs = total_epochs / total_models if total_models > 0 else 0
            
            report.append(f"Total epochs trained: {total_epochs}")
            report.append(f"Average epochs per model: {avg_epochs:.1f}")
            
            # Final accuracies
            final_accs = []
            for model_name, data in self.training_data.items():
                if data['val_acc']:
                    final_acc = data['val_acc'][-1]
                    final_accs.append(final_acc)
                    report.append(f"  {model_name}: {final_acc*100:.2f}%")
            
            if final_accs:
                report.append(f"Average final accuracy: {np.mean(final_accs)*100:.2f}%")
                report.append(f"Best individual accuracy: {max(final_accs)*100:.2f}%")
                report.append(f"Accuracy std deviation: {np.std(final_accs)*100:.2f}%")
        
        report.append("")
        
        # Ensemble Performance
        if ensemble_results:
            report.append("🎭 ENSEMBLE PERFORMANCE")
            report.append("-" * 30)
            
            ensemble_acc = ensemble_results.get('ensemble_accuracy', 0)
            individual_accs = ensemble_results.get('individual_accuracies', {})
            
            report.append(f"Ensemble accuracy: {ensemble_acc*100:.2f}%")
            
            if individual_accs:
                best_individual = max(individual_accs.values())
                improvement = ensemble_acc - best_individual
                improvement_pct = (improvement / best_individual) * 100
                
                report.append(f"Best individual: {best_individual*100:.2f}%")
                report.append(f"Improvement: +{improvement*100:.2f}% ({improvement_pct:.1f}%)")
                
                # Model contributions
                model_weights = ensemble_results.get('model_weights', {})
                if model_weights:
                    report.append("\nModel contributions:")
                    for model, weight in model_weights.items():
                        acc = individual_accs.get(model, 0)
                        report.append(f"  {model}: {weight*100:.1f}% (acc: {acc*100:.1f}%)")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.save_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📋 Training report saved to {report_path}")
        
        return report_text
    
    def save_training_data(self):
        """Save all training data to JSON"""
        
        data_to_save = {
            'training_data': self.training_data,
            'ensemble_metrics': self.ensemble_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = self.save_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        logger.info(f"💾 Training data saved to {json_path}")


# ========================
# src/utils/performance_analyzer.py - YENİ DOSYA
# ========================

class PerformanceAnalyzer:
    """Advanced performance analysis for ensemble models"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_model_diversity(self, models: List[torch.nn.Module], 
                               X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """Analyze diversity between models"""
        
        device = next(models[0].parameters()).device
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        predictions = []
        
        # Get predictions from all models
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(X_test)
                if pred.dim() > 1:
                    pred_classes = torch.argmax(pred, dim=1)
                else:
                    pred_classes = (pred > 0.5).long()
                predictions.append(pred_classes.cpu().numpy())
        
        predictions = np.array(predictions)  # [n_models, n_samples]
        
        # Calculate diversity metrics
        n_models = len(models)
        disagreement_rates = []
        
        # Pairwise disagreement
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreement_rates.append(disagreement)
        
        avg_disagreement = np.mean(disagreement_rates)
        
        # Prediction entropy (how much models disagree on each sample)
        sample_entropies = []
        for sample_idx in range(predictions.shape[1]):
            sample_preds = predictions[:, sample_idx]
            unique, counts = np.unique(sample_preds, return_counts=True)
            probs = counts / len(sample_preds)
            entropy = -np.sum(probs * np.log2(probs + 1e-8))
            sample_entropies.append(entropy)
        
        avg_entropy = np.mean(sample_entropies)
        
        # Model variance
        model_accuracies = []
        for i, model in enumerate(models):
            acc = np.mean(predictions[i] == y_test.cpu().numpy())
            model_accuracies.append(acc)
        
        accuracy_variance = np.var(model_accuracies)
        
        diversity_metrics = {
            'avg_disagreement_rate': avg_disagreement,
            'avg_prediction_entropy': avg_entropy,
            'accuracy_variance': accuracy_variance,
            'model_count': n_models,
            'individual_accuracies': model_accuracies
        }
        
        logger.info(f"📊 Diversity Analysis:")
        logger.info(f"   Average disagreement: {avg_disagreement:.3f}")
        logger.info(f"   Average entropy: {avg_entropy:.3f}")
        logger.info(f"   Accuracy variance: {accuracy_variance:.4f}")
        
        return diversity_metrics
    
    def identify_difficult_samples(self, models: List[torch.nn.Module], 
                                  X_test: torch.Tensor, y_test: torch.Tensor,
                                  threshold: float = 0.5) -> Dict[str, Any]:
        """Identify samples that are difficult for the ensemble"""
        
        device = next(models[0].parameters()).device
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        # Get ensemble predictions
        all_predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(X_test)
                if pred.dim() > 1:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)
                all_predictions.append(pred)
        
        # Calculate ensemble prediction (mean)
        ensemble_pred = torch.mean(torch.stack(all_predictions), dim=0)
        
        # Calculate prediction confidence (variance across models)
        pred_variance = torch.var(torch.stack(all_predictions), dim=0)
        
        if ensemble_pred.dim() > 1:
            # Multi-class: use max probability as confidence
            confidence = torch.max(ensemble_pred, dim=1)[0]
            ensemble_classes = torch.argmax(ensemble_pred, dim=1)
        else:
            # Binary: use distance from 0.5
            confidence = torch.abs(ensemble_pred - 0.5) * 2
            ensemble_classes = (ensemble_pred > 0.5).long()
        
        # Identify difficult samples (low confidence)
        difficult_mask = confidence < threshold
        difficult_indices = torch.where(difficult_mask)[0]
        
        # Calculate ensemble accuracy
        correct_predictions = (ensemble_classes == y_test.long()).float()
        ensemble_accuracy = correct_predictions.mean().item()
        
        analysis = {
            'total_samples': len(X_test),
            'difficult_samples': len(difficult_indices),
            'difficult_percentage': len(difficult_indices) / len(X_test) * 100,
            'ensemble_accuracy': ensemble_accuracy,
            'avg_confidence': confidence.mean().item(),
            'difficult_indices': difficult_indices.cpu().tolist(),
            'difficult_confidences': confidence[difficult_indices].cpu().tolist()
        }
        
        logger.info(f"🔍 Difficult Sample Analysis:")
        logger.info(f"   Total samples: {analysis['total_samples']}")
        logger.info(f"   Difficult samples: {analysis['difficult_samples']} ({analysis['difficult_percentage']:.1f}%)")
        logger.info(f"   Ensemble accuracy: {analysis['ensemble_accuracy']*100:.2f}%")
        
        return analysis

# KULLANIM ÖRNEĞİ:
"""
# Training monitoring
monitor = TrainingMonitor(save_dir="ensemble_logs")

# During training
for epoch in range(epochs):
    # ... training code ...
    monitor.log_training_progress('model_0', epoch, {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rate': current_lr
    })

# After training
monitor.plot_training_curves()
monitor.plot_ensemble_performance(ensemble_results)
report = monitor.generate_training_report(ensemble_results)
print(report)

# Performance analysis
analyzer = PerformanceAnalyzer()
diversity_metrics = analyzer.analyze_model_diversity(models, X_test, y_test)
difficult_samples = analyzer.identify_difficult_samples(models, X_test, y_test)
"""