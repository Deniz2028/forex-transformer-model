# src/utils/training_timer.py
"""
Training time tracking utility for temporal training.
Zaman takibi, performans metrikleri ve kaynak kullanımı.
"""

import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Training metrics container."""
    pair_name: str
    model_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Training details
    epochs_completed: int = 0
    total_epochs: int = 0
    best_val_acc: float = 0.0
    final_train_acc: float = 0.0
    overfitting_gap: float = 0.0
    
    # Performance metrics
    training_duration: Optional[timedelta] = None
    avg_epoch_time: Optional[timedelta] = None
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    gpu_utilized: bool = False
    
    # Data metrics
    train_samples: int = 0
    val_samples: int = 0
    sequence_length: int = 0
    n_features: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pair_name': self.pair_name,
            'model_type': self.model_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'epochs_completed': self.epochs_completed,
            'total_epochs': self.total_epochs,
            'best_val_acc': self.best_val_acc,
            'final_train_acc': self.final_train_acc,
            'overfitting_gap': self.overfitting_gap,
            'training_duration_seconds': self.training_duration.total_seconds() if self.training_duration else None,
            'avg_epoch_time_seconds': self.avg_epoch_time.total_seconds() if self.avg_epoch_time else None,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cpu_percent': self.avg_cpu_percent,
            'gpu_utilized': self.gpu_utilized,
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }

class TrainingTimer:
    """
    Training time tracking ve performance monitoring sınıfı.
    """
    
    def __init__(self, log_directory: str = "logs/temporal"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.current_metrics: Optional[TrainingMetrics] = None
        self.all_metrics: Dict[str, TrainingMetrics] = {}
        
        # System monitoring
        self.process = psutil.Process()
        self.monitoring_enabled = True
        self.memory_samples = []
        self.cpu_samples = []
        
        logger.info(f"⏱️ TrainingTimer initialized: {self.log_directory}")
    
    def start_training(self, pair_name: str, model_type: str = "enhanced_transformer",
                      total_epochs: int = 50, train_samples: int = 0, 
                      val_samples: int = 0, sequence_length: int = 64,
                      n_features: int = 0) -> None:
        """
        Start training time tracking.
        
        Args:
            pair_name: Currency pair name
            model_type: Model architecture type
            total_epochs: Total epochs planned
            train_samples: Number of training samples
            val_samples: Number of validation samples
            sequence_length: Input sequence length
            n_features: Number of input features
        """
        self.current_metrics = TrainingMetrics(
            pair_name=pair_name,
            model_type=model_type,
            start_time=datetime.now(),
            total_epochs=total_epochs,
            train_samples=train_samples,
            val_samples=val_samples,
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        # Reset monitoring data
        self.memory_samples = []
        self.cpu_samples = []
        
        # Check GPU availability
        try:
            import torch
            self.current_metrics.gpu_utilized = torch.cuda.is_available()
        except ImportError:
            self.current_metrics.gpu_utilized = False
        
        logger.info(f"⏱️ Started training timer for {pair_name} ({model_type})")
        logger.info(f"   📊 Data: {train_samples} train, {val_samples} val samples")
        logger.info(f"   🎯 Target: {total_epochs} epochs")
        
    def update_epoch(self, epoch: int, train_acc: float, val_acc: float) -> None:
        """
        Update metrics after each epoch.
        
        Args:
            epoch: Current epoch number
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        if not self.current_metrics:
            logger.warning("⚠️ Training timer not started")
            return
        
        self.current_metrics.epochs_completed = epoch
        self.current_metrics.final_train_acc = train_acc
        
        # Update best validation accuracy
        if val_acc > self.current_metrics.best_val_acc:
            self.current_metrics.best_val_acc = val_acc
        
        # Update overfitting gap
        self.current_metrics.overfitting_gap = train_acc - self.current_metrics.best_val_acc
        
        # System monitoring
        if self.monitoring_enabled:
            try:
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                
                if memory_mb > self.current_metrics.peak_memory_mb:
                    self.current_metrics.peak_memory_mb = memory_mb
                
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # Log every 10 epochs
                if epoch % 10 == 0:
                    elapsed = datetime.now() - self.current_metrics.start_time
                    avg_epoch_time = elapsed / epoch if epoch > 0 else timedelta(0)
                    
                    logger.info(f"   ⏱️ Epoch {epoch}: {elapsed} elapsed, "
                              f"{avg_epoch_time.total_seconds():.1f}s/epoch, "
                              f"Memory: {memory_mb:.1f}MB")
                
            except Exception as e:
                logger.warning(f"⚠️ System monitoring error: {e}")
    
    def finish_training(self, success: bool = True) -> TrainingMetrics:
        """
        Finish training and calculate final metrics.
        
        Args:
            success: Whether training completed successfully
            
        Returns:
            Final training metrics
        """
        if not self.current_metrics:
            logger.warning("⚠️ Training timer not started")
            return None
        
        # Set end time and calculate duration
        self.current_metrics.end_time = datetime.now()
        self.current_metrics.training_duration = (
            self.current_metrics.end_time - self.current_metrics.start_time
        )
        
        # Calculate average epoch time
        if self.current_metrics.epochs_completed > 0:
            self.current_metrics.avg_epoch_time = (
                self.current_metrics.training_duration / self.current_metrics.epochs_completed
            )
        
        # Calculate average CPU usage
        if self.cpu_samples:
            self.current_metrics.avg_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # Store metrics
        self.all_metrics[self.current_metrics.pair_name] = self.current_metrics
        
        # Log final results
        self._log_final_results(success)
        
        # Save to file
        self._save_metrics()
        
        final_metrics = self.current_metrics
        self.current_metrics = None
        
        return final_metrics
    
    def _log_final_results(self, success: bool) -> None:
        """Log final training results."""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        
        logger.info(f"⏱️ TRAINING COMPLETED: {metrics.pair_name}")
        logger.info(f"   ✅ Success: {success}")
        logger.info(f"   ⏰ Duration: {metrics.training_duration}")
        logger.info(f"   📊 Epochs: {metrics.epochs_completed}/{metrics.total_epochs}")
        logger.info(f"   🎯 Best Val Acc: {metrics.best_val_acc:.2f}%")
        logger.info(f"   📈 Final Train Acc: {metrics.final_train_acc:.2f}%")
        logger.info(f"   📉 Overfitting Gap: {metrics.overfitting_gap:.2f}pp")
        
        if metrics.avg_epoch_time:
            logger.info(f"   ⚡ Avg Epoch Time: {metrics.avg_epoch_time.total_seconds():.1f}s")
        
        logger.info(f"   💾 Peak Memory: {metrics.peak_memory_mb:.1f}MB")
        logger.info(f"   🖥️ Avg CPU: {metrics.avg_cpu_percent:.1f}%")
        logger.info(f"   🎮 GPU Used: {metrics.gpu_utilized}")
        
        # Performance warnings
        if metrics.training_duration and metrics.training_duration > timedelta(hours=2):
            logger.warning(f"   ⚠️ Long training time: {metrics.training_duration}")
        
        if metrics.peak_memory_mb > 8000:  # 8GB
            logger.warning(f"   ⚠️ High memory usage: {metrics.peak_memory_mb:.1f}MB")
        
        if metrics.overfitting_gap > 15:
            logger.warning(f"   ⚠️ High overfitting gap: {metrics.overfitting_gap:.2f}pp")
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Individual training log
            if self.current_metrics:
                individual_file = self.log_directory / f"training_{self.current_metrics.pair_name}_{timestamp}.json"
                with open(individual_file, 'w') as f:
                    json.dump(self.current_metrics.to_dict(), f, indent=2)
                
                logger.info(f"💾 Individual metrics saved: {individual_file}")
            
            # All training logs
            if self.all_metrics:
                all_file = self.log_directory / f"all_training_metrics_{timestamp}.json"
                all_data = {
                    'timestamp': timestamp,
                    'total_trainings': len(self.all_metrics),
                    'metrics': {pair: metrics.to_dict() for pair, metrics in self.all_metrics.items()}
                }
                
                with open(all_file, 'w') as f:
                    json.dump(all_data, f, indent=2)
                
                logger.info(f"💾 All metrics saved: {all_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all completed trainings.
        
        Returns:
            Summary statistics dictionary
        """
        if not self.all_metrics:
            return {}
        
        metrics_list = list(self.all_metrics.values())
        
        # Calculate aggregated stats
        total_trainings = len(metrics_list)
        successful_trainings = len([m for m in metrics_list if m.end_time is not None])
        
        durations = [m.training_duration.total_seconds() for m in metrics_list if m.training_duration]
        val_accs = [m.best_val_acc for m in metrics_list]
        overfit_gaps = [m.overfitting_gap for m in metrics_list]
        memory_peaks = [m.peak_memory_mb for m in metrics_list]
        
        summary = {
            'total_trainings': total_trainings,
            'successful_trainings': successful_trainings,
            'success_rate': successful_trainings / total_trainings if total_trainings > 0 else 0,
            
            'duration_stats': {
                'avg_seconds': sum(durations) / len(durations) if durations else 0,
                'min_seconds': min(durations) if durations else 0,
                'max_seconds': max(durations) if durations else 0,
                'total_hours': sum(durations) / 3600 if durations else 0
            },
            
            'accuracy_stats': {
                'avg_val_acc': sum(val_accs) / len(val_accs) if val_accs else 0,
                'best_val_acc': max(val_accs) if val_accs else 0,
                'worst_val_acc': min(val_accs) if val_accs else 0
            },
            
            'overfitting_stats': {
                'avg_gap': sum(overfit_gaps) / len(overfit_gaps) if overfit_gaps else 0,
                'max_gap': max(overfit_gaps) if overfit_gaps else 0,
                'problematic_pairs': [m.pair_name for m in metrics_list if m.overfitting_gap > 15]
            },
            
            'memory_stats': {
                'avg_peak_mb': sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                'max_peak_mb': max(memory_peaks) if memory_peaks else 0
            }
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print training summary to console."""
        summary = self.get_summary_stats()
        
        if not summary:
            logger.info("📊 No training metrics available")
            return
        
        print(f"\n📊 TRAINING SUMMARY")
        print(f"{'='*50}")
        print(f"🎯 Total Trainings: {summary['total_trainings']}")
        print(f"✅ Successful: {summary['successful_trainings']} ({summary['success_rate']*100:.1f}%)")
        
        duration_stats = summary['duration_stats']
        print(f"\n⏰ TIMING:")
        print(f"   Total Time: {duration_stats['total_hours']:.1f} hours")
        print(f"   Avg per Training: {duration_stats['avg_seconds']/60:.1f} minutes")
        print(f"   Range: {duration_stats['min_seconds']/60:.1f} - {duration_stats['max_seconds']/60:.1f} minutes")
        
        acc_stats = summary['accuracy_stats']
        print(f"\n🎯 ACCURACY:")
        print(f"   Average Val Acc: {acc_stats['avg_val_acc']:.2f}%")
        print(f"   Best Val Acc: {acc_stats['best_val_acc']:.2f}%")
        print(f"   Worst Val Acc: {acc_stats['worst_val_acc']:.2f}%")
        
        overfit_stats = summary['overfitting_stats']
        print(f"\n📉 OVERFITTING:")
        print(f"   Average Gap: {overfit_stats['avg_gap']:.2f}pp")
        print(f"   Max Gap: {overfit_stats['max_gap']:.2f}pp")
        if overfit_stats['problematic_pairs']:
            print(f"   Problematic Pairs: {', '.join(overfit_stats['problematic_pairs'])}")
        
        memory_stats = summary['memory_stats']
        print(f"\n💾 MEMORY:")
        print(f"   Avg Peak: {memory_stats['avg_peak_mb']:.1f}MB")
        print(f"   Max Peak: {memory_stats['max_peak_mb']:.1f}MB")

# Context manager for easy usage
class TimedTraining:
    """Context manager for training time tracking."""
    
    def __init__(self, timer: TrainingTimer, pair_name: str, **kwargs):
        self.timer = timer
        self.pair_name = pair_name
        self.kwargs = kwargs
        self.success = False
    
    def __enter__(self):
        self.timer.start_training(self.pair_name, **self.kwargs)
        return self.timer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        self.timer.finish_training(self.success)

__all__ = [
    'TrainingMetrics',
    'TrainingTimer', 
    'TimedTraining'
]