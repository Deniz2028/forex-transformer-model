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
    def confidence_weighted(predictions: List[torch.Tensor]) -> torch.Tensor:
        """Confidence-weighted ensemble prediction with SHAPE and STRING FIX"""
        
        if not predictions:
            raise ValueError("No predictions provided")
        
        # 🔧 FIX: Filter out non-tensor objects (strings, None, etc.)
        valid_predictions = []
        for i, pred in enumerate(predictions):
            if isinstance(pred, torch.Tensor):
                valid_predictions.append(pred)
            else:
                logger.warning(f"⚠️ Skipping invalid prediction {i}: {type(pred)} - {pred}")
        
        if not valid_predictions:
            raise ValueError("No valid tensor predictions found")
        
        predictions = valid_predictions  # Use only valid predictions
        
        # 🔧 FIX: Normalize shapes first
        normalized_predictions = []
        
        for pred in predictions:
            if pred.dim() == 1:
                # Single output → convert to 2-class probabilities
                probs = torch.sigmoid(pred)
                pred_2d = torch.stack([1 - probs, probs], dim=1)
                normalized_predictions.append(pred_2d)
            elif pred.dim() == 2:
                # Already correct format, apply softmax
                pred_2d = torch.softmax(pred, dim=1)
                normalized_predictions.append(pred_2d)
            else:
                raise ValueError(f"Unexpected prediction shape: {pred.shape}")
        
        # Calculate confidence as max probability
        confidences = []
        for pred in normalized_predictions:
            conf = torch.max(pred, dim=1)[0]  # Max probability per sample
            confidences.append(conf)
        
        # Use performance weighted with calculated confidences
        return VotingStrategies.performance_weighted(normalized_predictions, confidences)
    
    @staticmethod
    def performance_weighted(predictions: List[torch.Tensor], 
                           confidences: List[torch.Tensor]) -> torch.Tensor:
        """Performance-weighted ensemble prediction with SHAPE FIX"""
        
        if not predictions or len(predictions) != len(confidences):
            raise ValueError("Predictions and confidences must have same length")
        
        # 🔧 FIX: Normalize all predictions to same shape
        normalized_predictions = []
        target_shape = None
        
        for i, pred in enumerate(predictions):
            if pred.dim() == 1:
                # Single output → convert to probabilities
                # Assume sigmoid output for binary classification
                probs = torch.sigmoid(pred)
                # Create 2-class format: [1-prob, prob]
                pred_2d = torch.stack([1 - probs, probs], dim=1)
                normalized_predictions.append(pred_2d)
                if target_shape is None:
                    target_shape = pred_2d.shape
            elif pred.dim() == 2:
                # Already in correct format
                normalized_predictions.append(pred)
                if target_shape is None:
                    target_shape = pred.shape
            else:
                raise ValueError(f"Unexpected prediction shape: {pred.shape}")
        
        # Ensure all predictions have same shape
        final_predictions = []
        for pred in normalized_predictions:
            if pred.shape != target_shape:
                # Handle shape mismatch
                if pred.dim() == 2 and target_shape[1] == 2:
                    # Apply softmax to ensure probabilities
                    pred = torch.softmax(pred, dim=1)
                final_predictions.append(pred)
            else:
                final_predictions.append(pred)
        
        # Apply performance weights
        weighted_preds = []
        total_weight = 0.0
        
        for pred, conf in zip(final_predictions, confidences):
            weight = conf.mean().item() if conf.numel() > 1 else conf.item()
            weighted_pred = pred * weight
            weighted_preds.append(weighted_pred)
            total_weight += weight
        
        # 🔧 FIX: Now all tensors have same shape for stacking
        try:
            ensemble_pred = torch.sum(torch.stack(weighted_preds, dim=0), dim=0)
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
            
            return ensemble_pred
            
        except RuntimeError as e:
            # Debug info
            shapes = [pred.shape for pred in weighted_preds]
            raise RuntimeError(f"Shape mismatch in stacking. Shapes: {shapes}. Original error: {e}")
    
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
