# ========================
# src/ensemble/ensemble_classifier.py
"""Advanced ensemble classifier with multiple voting strategies."""

from .base_ensemble import BaseEnsemble, VotingStrategies
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EnsembleClassifier(BaseEnsemble):
    """Advanced ensemble classifier with multiple voting strategies"""
    
    def __init__(self, models: List[nn.Module], device: torch.device,
                 strategy: str = 'confidence_weighted'):
        super().__init__(models, device)
        self.strategy = strategy
        self.performance_weights = {}
        
        # Validate strategy
        valid_strategies = [
            'simple_average', 'weighted_average', 'confidence_weighted', 
            'performance_weighted', 'majority_voting'
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")
        
        logger.info(f"🎯 EnsembleClassifier initialized with '{strategy}' strategy")
        
    def set_performance_weights(self, weights: Dict[str, float]):
        """Set performance-based weights for models"""
        self.performance_weights = weights
        logger.info(f"📊 Performance weights set: {weights}")
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with selected strategy"""
        predictions = self._get_individual_predictions(X)
        
        if self.strategy == 'simple_average':
            return VotingStrategies.simple_average(predictions)
        elif self.strategy == 'weighted_average':
            if not self.performance_weights:
                logger.warning("No performance weights set, falling back to simple average")
                return VotingStrategies.simple_average(predictions)
            return VotingStrategies.weighted_average(predictions, self.performance_weights)
        elif self.strategy == 'confidence_weighted':
            return VotingStrategies.confidence_weighted(predictions)
        elif self.strategy == 'majority_voting':
            return VotingStrategies.majority_voting(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction with uncertainty estimation"""
        predictions = self._get_individual_predictions(X)
        
        # Stack all predictions
        pred_stack = torch.stack(list(predictions.values()))  # [n_models, batch, classes]
        
        # Mean and variance across models
        ensemble_mean = torch.mean(pred_stack, dim=0)
        ensemble_var = torch.var(pred_stack, dim=0)
        
        # Uncertainty as total variance
        uncertainty = torch.sum(ensemble_var, dim=1)
        
        return ensemble_mean, uncertainty
    
    def get_individual_predictions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual model predictions (public interface)"""
        return self._get_individual_predictions(X)
    
    def evaluate_diversity(self, X: torch.Tensor) -> Dict[str, float]:
        """Evaluate ensemble diversity metrics"""
        predictions = self._get_individual_predictions(X)
        pred_stack = torch.stack(list(predictions.values()))
        
        # Disagreement rate (how often models disagree)
        hard_preds = torch.argmax(pred_stack, dim=2)  # [n_models, batch]
        disagreement = []
        
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                disagree_rate = (hard_preds[i] != hard_preds[j]).float().mean().item()
                disagreement.append(disagree_rate)
        
        avg_disagreement = np.mean(disagreement) if disagreement else 0.0
        
        # Prediction entropy (diversity in soft predictions)
        ensemble_pred = torch.mean(pred_stack, dim=0)
        entropy = -torch.sum(ensemble_pred * torch.log(ensemble_pred + 1e-8), dim=1)
        avg_entropy = entropy.mean().item()
        
        return {
            'disagreement_rate': avg_disagreement,
            'prediction_entropy': avg_entropy,
            'model_count': len(self.models)
        }
