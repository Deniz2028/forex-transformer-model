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
        
        # Move all models to device
        for model in self.models:
            model.to(device)
        
        logger.info(f"🎭 BaseEnsemble initialized with {len(models)} models on {device}")
        
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Abstract prediction method"""
        pass
    
    def eval(self):
        """Set all models to evaluation mode"""
        for model in self.models:
            model.eval()
        logger.debug("🔍 All ensemble models set to eval mode")
    
    def train(self):
        """Set all models to training mode"""
        for model in self.models:
            model.train()
        logger.debug("🏋️ All ensemble models set to train mode")
    
   # src/ensemble/base_ensemble.py - _get_individual_predictions metodunu güncelleyin:

    def _get_individual_predictions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from all models"""
        predictions = {}
        
        # Ensure input is on correct device
        X = X.to(self.device)
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                try:
                    pred = model(X)
                    
                    # Handle different output formats
                    if pred.dim() > 1 and pred.shape[1] > 1:
                        # Multi-class output - apply softmax
                        pred = torch.softmax(pred, dim=1)
                    else:
                        # Binary output - apply sigmoid if not already applied
                        if pred.max() > 1.0 or pred.min() < 0.0:
                            # Logits - apply sigmoid
                            pred = torch.sigmoid(pred.squeeze())
                        else:
                            # Already probabilities - just squeeze
                            pred = pred.squeeze()
                        
                        if pred.dim() == 0:
                            pred = pred.unsqueeze(0)
                    
                    predictions[f'model_{i}'] = pred
                    
                except Exception as e:
                    logger.error(f"Error in model {i} prediction: {e}")
                    logger.error(f"Model type: {type(model).__name__}")
                    logger.error(f"Input shape: {X.shape}")
                    # Skip this model if prediction fails
                    continue
                    
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

    def get_individual_predictions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Public interface for getting individual predictions"""
        return self._get_individual_predictions(X)


class VotingStrategies:
    """Ensemble voting strategies"""
    
    @staticmethod
    def simple_average(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple average ensemble"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list, dim=0)
        result = torch.mean(pred_stack, dim=0)
        
        logger.debug(f"Simple average ensemble: {len(predictions)} models")
        return result
    
    @staticmethod
    def weighted_average(predictions: Dict[str, torch.Tensor], 
                        weights: Dict[str, float]) -> torch.Tensor:
        """Weighted average ensemble"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        weighted_preds = []
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_preds.append(pred * weight)
            total_weight += weight
            
        if total_weight == 0:
            return VotingStrategies.simple_average(predictions)
            
        result = torch.sum(torch.stack(weighted_preds, dim=0), dim=0) / total_weight
        logger.debug(f"Weighted average ensemble: weights={weights}")
        return result
    
    @staticmethod
    def confidence_weighted(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Confidence-weighted ensemble using entropy"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        confidences = {}
        
        for name, pred in predictions.items():
            if pred.dim() > 1:
                # Multi-class: use entropy
                entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)
                confidence = 1.0 / (1.0 + entropy)
            else:
                # Binary: use distance from 0.5
                confidence = torch.abs(pred - 0.5) * 2
            
            confidences[name] = confidence
        
        # Use performance_weighted with confidence scores
        return VotingStrategies.performance_weighted(predictions, confidences)
    
    @staticmethod
    def performance_weighted(predictions: Dict[str, torch.Tensor], 
                           performances: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performance-weighted ensemble"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        weighted_preds = []
        total_weights = 0
        
        for name, pred in predictions.items():
            if name in performances:
                weight = performances[name]
                
                # Convert weight to tensor if needed
                if isinstance(weight, (float, int)):
                    weight = torch.tensor(weight, device=pred.device, dtype=pred.dtype)
                
                # Handle different prediction dimensions
                if pred.dim() > 1:
                    # Multi-class predictions
                    if weight.dim() == 0:
                        weight = weight.unsqueeze(0).unsqueeze(1)
                    elif weight.dim() == 1:
                        weight = weight.unsqueeze(1)
                    weight = weight.expand_as(pred)
                else:
                    # Binary predictions
                    if weight.dim() == 0:
                        weight = weight.unsqueeze(0)
                    weight = weight.expand_as(pred)
                
                weighted_preds.append(pred * weight)
                
                # Sum up total weights properly
                if pred.dim() > 1:
                    total_weights += weight[:, 0]  # Use first column for total
                else:
                    total_weights += weight
        
        if len(weighted_preds) == 0:
            # Fallback to simple average
            return VotingStrategies.simple_average(predictions)
        
        ensemble_pred = torch.sum(torch.stack(weighted_preds, dim=0), dim=0)
        
        # Normalize by total weights
        if isinstance(total_weights, (int, float)):
            total_weights = torch.tensor(total_weights, device=ensemble_pred.device)
        
        if ensemble_pred.dim() > 1:
            total_weights = total_weights.unsqueeze(1).expand_as(ensemble_pred)
        else:
            total_weights = total_weights.expand_as(ensemble_pred)
        
        return ensemble_pred / (total_weights + 1e-8)
    
    @staticmethod
    def majority_voting(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hard majority voting (for classification)"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Convert to hard predictions
        hard_preds = []
        sample_pred = list(predictions.values())[0]
        
        for pred in predictions.values():
            if pred.dim() > 1:
                # Multi-class
                hard_pred = torch.argmax(pred, dim=1)
            else:
                # Binary
                hard_pred = (pred > 0.5).long()
            hard_preds.append(hard_pred)
        
        hard_preds = torch.stack(hard_preds, dim=0)  # [n_models, batch_size]
        
        # Determine number of classes
        if sample_pred.dim() > 1:
            n_classes = sample_pred.shape[1]
        else:
            n_classes = 2  # Binary case
        
        batch_size = hard_preds.shape[1]
        device = hard_preds.device
        
        # Count votes for each class
        vote_counts = torch.zeros(batch_size, n_classes, device=device)
        
        for i in range(batch_size):
            for j in range(n_classes):
                vote_counts[i, j] = (hard_preds[:, i] == j).sum()
        
        # Convert back to probabilities
        if n_classes > 2:
            # Multi-class: return one-hot
            winners = torch.argmax(vote_counts, dim=1)
            result = torch.zeros_like(vote_counts, dtype=torch.float32)
            result[torch.arange(batch_size), winners] = 1.0
        else:
            # Binary: return probability
            total_votes = vote_counts.sum(dim=1, keepdim=True)
            prob_class_1 = vote_counts[:, 1] / (total_votes.squeeze() + 1e-8)
            result = prob_class_1
        
        return result
    
    @staticmethod
    def uncertainty_weighted(predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uncertainty-weighted ensemble with uncertainty estimation"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Stack all predictions
        pred_stack = torch.stack(list(predictions.values()), dim=0)  # [n_models, batch, ...]
        
        # Mean and variance across models
        ensemble_mean = torch.mean(pred_stack, dim=0)
        ensemble_var = torch.var(pred_stack, dim=0)
        
        # Uncertainty as total variance
        if ensemble_var.dim() > 1:
            uncertainty = torch.sum(ensemble_var, dim=1)
        else:
            uncertainty = ensemble_var
        
        return ensemble_mean, uncertainty
