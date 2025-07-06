# src/ensemble/dynamic_selector.py
"""Dynamic model selection based on recent performance."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DynamicModelSelector:
    """Dynamic model selection based on recent performance"""
    
    def __init__(self, models: List, window_size: int = 100):
        self.models = models
        self.window_size = window_size
        self.performance_history = {i: deque(maxlen=window_size) 
                                  for i in range(len(models))}
        self.current_weights = np.ones(len(models)) / len(models)
        self.update_frequency = 10  # Update weights every N predictions
        self.update_counter = 0
        
        logger.info(f"🎯 DynamicModelSelector initialized with {len(models)} models")
        logger.info(f"   Window size: {window_size}, Update frequency: {self.update_frequency}")
        
    def update_performance(self, model_predictions: Dict[int, torch.Tensor], 
                         true_labels: torch.Tensor):
        """Update performance history"""
        for model_idx, pred in model_predictions.items():
            if model_idx < len(self.models):
                # Calculate accuracy for this batch
                if pred.dim() > 1:
                    pred_classes = torch.argmax(pred, dim=1)
                else:
                    pred_classes = (pred > 0.5).long()
                
                if true_labels.dim() > 1:
                    true_classes = torch.argmax(true_labels, dim=1)
                else:
                    true_classes = true_labels.long()
                
                accuracy = (pred_classes == true_classes).float().mean().item()
                self.performance_history[model_idx].append(accuracy)
        
        self.update_counter += 1
        
        # Update weights periodically
        if self.update_counter >= self.update_frequency:
            self._update_weights()
            self.update_counter = 0
            
    def _update_weights(self):
        """Update model weights based on recent performance"""
        weights = []
        for i in range(len(self.models)):
            if len(self.performance_history[i]) > 0:
                # Use recent performance with exponential decay
                recent_scores = list(self.performance_history[i])
                decay_weights = np.exp(-0.1 * np.arange(len(recent_scores)))
                weighted_perf = np.average(recent_scores, weights=decay_weights)
                weights.append(max(weighted_perf, 0.1))  # Minimum weight
            else:
                weights.append(1.0)
                
        # Softmax normalization with temperature
        temperature = 2.0  # Controls sharpness of weights
        weights = np.array(weights)
        weights = np.exp(weights / temperature) / np.sum(np.exp(weights / temperature))
        
        # Smooth transition (momentum)
        momentum = 0.7
        self.current_weights = momentum * self.current_weights + (1 - momentum) * weights
        
        logger.debug(f"🔄 Weights updated: {self.current_weights}")
        
    def get_ensemble_prediction(self, model_predictions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Get weighted ensemble prediction"""
        weighted_preds = []
        
        for model_idx, pred in model_predictions.items():
            if model_idx < len(self.current_weights):
                weight = self.current_weights[model_idx]
                weighted_preds.append(pred * weight)
            
        if weighted_preds:
            return torch.sum(torch.stack(weighted_preds), dim=0)
        else:
            # Fallback to simple average
            return torch.mean(torch.stack(list(model_predictions.values())), dim=0)
    
    def get_top_models(self, k: int = 3) -> List[int]:
        """Get indices of top-k performing models"""
        model_scores = []
        for i in range(len(self.models)):
            if len(self.performance_history[i]) > 0:
                score = np.mean(list(self.performance_history[i]))
            else:
                score = 0.5  # Default score
            model_scores.append((i, score))
        
        # Sort by score and return top-k indices
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in model_scores[:k]]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for i in range(len(self.models)):
            history = list(self.performance_history[i])
            if history:
                stats[f'model_{i}'] = {
                    'mean_accuracy': np.mean(history),
                    'std_accuracy': np.std(history),
                    'recent_accuracy': history[-10:] if len(history) >= 10 else history,
                    'weight': self.current_weights[i],
                    'sample_count': len(history)
                }
            else:
                stats[f'model_{i}'] = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'recent_accuracy': [],
                    'weight': self.current_weights[i],
                    'sample_count': 0
                }
        
        return stats