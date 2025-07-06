# ========================
# src/ensemble/base_ensemble.py
"""Base ensemble classes and voting strategies."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """Base ensemble class for different voting strategies"""
    
    def __init__(self, models: List[nn.Module], device: torch.device):
        self.models = models
        self.device = device
        self.model_weights = None
        
        logger.info(f"🎭 BaseEnsemble initialized with {len(models)} models")
        
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Abstract prediction method"""
        pass
    
    def _get_individual_predictions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from all models"""
        predictions = {}
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(X), dim=1)
                predictions[f'model_{i}'] = pred
                
        return predictions
    
    def get_model_count(self) -> int:
        """Get number of models in ensemble"""
        return len(self.models)
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about each model"""
        info = []
        for i, model in enumerate(self.models):
            model_info = {
                'index': i,
                'type': type(model).__name__,
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': next(model.parameters()).device
            }
            info.append(model_info)
        return info


class VotingStrategies:
    """Ensemble voting strategies"""
    
    @staticmethod
    def simple_average(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple average ensemble"""
        pred_stack = torch.stack(list(predictions.values()))
        result = torch.mean(pred_stack, dim=0)
        logger.debug(f"Simple average ensemble: {len(predictions)} models")
        return result
    
    @staticmethod
    def weighted_average(predictions: Dict[str, torch.Tensor], 
                        weights: Dict[str, float]) -> torch.Tensor:
        """Weighted average ensemble"""
        weighted_preds = []
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_preds.append(pred * weight)
            total_weight += weight
            
        result = torch.sum(torch.stack(weighted_preds), dim=0) / total_weight
        logger.debug(f"Weighted average ensemble: weights={weights}")
        return result
    
    @staticmethod
    def confidence_weighted(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Confidence-weighted ensemble using entropy"""
        confidences = {}
        
        for name, pred in predictions.items():
            # Entropy-based confidence (lower entropy = higher confidence)
            entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)
            confidence = 1.0 / (1.0 + entropy)  # Inverse entropy
            confidences[name] = confidence
            
        # Weighted average using confidences
        return VotingStrategies.performance_weighted(predictions, confidences)
    
    @staticmethod
    def performance_weighted(predictions: Dict[str, torch.Tensor], 
                           performances: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performance-weighted ensemble"""
        weighted_preds = []
        total_weights = None
        
        for name, pred in predictions.items():
            if name in performances:
                weight = performances[name]
                if total_weights is None:
                    total_weights = weight.clone()
                else:
                    total_weights += weight
                    
                # Broadcast weight to match prediction dimensions
                weight_expanded = weight.unsqueeze(1).expand_as(pred)
                weighted_preds.append(pred * weight_expanded)
        
        if total_weights is None or len(weighted_preds) == 0:
            # Fallback to simple average
            return VotingStrategies.simple_average(predictions)
        
        ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        total_weights_expanded = total_weights.unsqueeze(1).expand_as(ensemble_pred)
        
        return ensemble_pred / total_weights_expanded
    
    @staticmethod
    def majority_voting(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hard majority voting (for classification)"""
        # Convert to hard predictions
        hard_preds = []
        for pred in predictions.values():
            hard_pred = torch.argmax(pred, dim=1)
            hard_preds.append(hard_pred)
        
        hard_preds = torch.stack(hard_preds)  # [n_models, batch_size]
        
        # Count votes for each class
        batch_size = hard_preds.shape[1]
        n_classes = predictions[list(predictions.keys())[0]].shape[1]
        
        vote_counts = torch.zeros(batch_size, n_classes, device=hard_preds.device)
        
        for i in range(batch_size):
            for j in range(n_classes):
                vote_counts[i, j] = (hard_preds[:, i] == j).sum()
        
        # Convert back to probabilities (one-hot for winner)
        winners = torch.argmax(vote_counts, dim=1)
        result = torch.zeros_like(vote_counts)
        result[torch.arange(batch_size), winners] = 1.0
        
        return result
