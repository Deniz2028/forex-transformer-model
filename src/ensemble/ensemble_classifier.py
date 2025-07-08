# ========================
# src/ensemble/ensemble_classifier.py
"""Advanced ensemble classifier with multiple voting strategies."""

from .base_ensemble import BaseEnsemble, VotingStrategies
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EnsembleClassifier(BaseEnsemble):
    """Advanced ensemble classifier with multiple voting strategies"""
    
    def __init__(self, models: List[nn.Module], device: torch.device,
                 strategy: str = 'confidence_weighted'):
        super().__init__(models, device)
        self.strategy = strategy
        self.performance_weights = {}
        self.performance_weights_tensor = None
        
        # Validate strategy
        valid_strategies = [
            'simple_average', 'weighted_average', 'confidence_weighted', 
            'performance_weighted', 'majority_voting', 'uncertainty_weighted'
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")
        
        logger.info(f"🎯 EnsembleClassifier initialized with '{strategy}' strategy")
    
    def set_performance_weights(self, weights: Dict[str, float]):
        """Set performance-based weights for models"""
        self.performance_weights = weights
        
        # Convert to tensor for GPU compatibility
        if isinstance(weights, dict):
            weight_values = list(weights.values())
        elif isinstance(weights, (list, tuple)):
            weight_values = list(weights)
        else:
            weight_values = weights
            
        self.performance_weights_tensor = torch.tensor(
            weight_values, 
            device=self.device, 
            dtype=torch.float32
        )
        
        logger.info(f"📊 Performance weights set: {self.performance_weights_tensor}")
    
    def get_individual_predictions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Public interface for getting individual predictions - RETURNS TENSORS"""
        return self._get_individual_predictions(X)
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with selected strategy"""
        predictions = self._get_individual_predictions(X)
        
        # 🔧 FIX: Convert dict values to list for VotingStrategies
        prediction_tensors = list(predictions.values())
        
        if self.strategy == 'simple_average':
            return VotingStrategies.simple_average(prediction_tensors)
        elif self.strategy == 'weighted_average':
            if not self.performance_weights:
                logger.warning("No performance weights set, falling back to simple average")
                return VotingStrategies.simple_average(prediction_tensors)
            return VotingStrategies.weighted_average(prediction_tensors, self.performance_weights)
        elif self.strategy == 'confidence_weighted':
            return VotingStrategies.confidence_weighted(prediction_tensors)
        elif self.strategy == 'performance_weighted':
            return self._performance_weighted_prediction(predictions)
        elif self.strategy == 'majority_voting':
            return VotingStrategies.majority_voting(prediction_tensors)
        elif self.strategy == 'uncertainty_weighted':
            ensemble_pred, uncertainty = VotingStrategies.uncertainty_weighted(prediction_tensors)
            return ensemble_pred
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    
    def predict_with_uncertainty(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction with uncertainty estimation"""
        predictions = self._get_individual_predictions(X)
        
        # Stack all predictions
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list, dim=0)  # [n_models, batch, ...]
        
        # Mean and variance across models
        ensemble_mean = torch.mean(pred_stack, dim=0)
        ensemble_var = torch.var(pred_stack, dim=0)
        
        # Uncertainty as total variance
        if ensemble_var.dim() > 1:
            uncertainty = torch.sum(ensemble_var, dim=1)
        else:
            uncertainty = ensemble_var
        
        return ensemble_mean, uncertainty
    
    def evaluate_diversity(self, X: torch.Tensor) -> Dict[str, float]:
        """Evaluate ensemble diversity metrics"""
        predictions = self._get_individual_predictions(X)
        
        if not predictions:
            return {'diversity_error': 'No predictions available'}
        
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list, dim=0)  # [n_models, batch, ...]
        
        # Disagreement rate (how often models disagree)
        if pred_stack.dim() == 3:  # Multi-class case
            hard_preds = torch.argmax(pred_stack, dim=2)  # [n_models, batch]
        else:  # Binary case
            hard_preds = (pred_stack > 0.5).long()  # [n_models, batch]
        
        disagreement = []
        n_models = len(self.models)
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagree_rate = (hard_preds[i] != hard_preds[j]).float().mean().item()
                disagreement.append(disagree_rate)
        
        avg_disagreement = np.mean(disagreement) if disagreement else 0.0
        
        # Prediction entropy (diversity in soft predictions)
        ensemble_pred = torch.mean(pred_stack, dim=0)
        
        if ensemble_pred.dim() > 1:
            # Multi-class entropy
            entropy = -torch.sum(ensemble_pred * torch.log(ensemble_pred + 1e-8), dim=1)
        else:
            # Binary entropy
            p = ensemble_pred
            entropy = -(p * torch.log(p + 1e-8) + (1-p) * torch.log(1-p + 1e-8))
        
        avg_entropy = entropy.mean().item()
        
        # Model variance (how much models disagree on average)
        model_variance = torch.var(pred_stack, dim=0)
        if model_variance.dim() > 1:
            avg_variance = model_variance.mean().item()
        else:
            avg_variance = model_variance.mean().item()
        
        return {
            'disagreement_rate': avg_disagreement,
            'prediction_entropy': avg_entropy,
            'model_variance': avg_variance,
            'model_count': len(self.models)
        }
    
    def _performance_weighted_prediction(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performance-weighted ensemble prediction"""
        if not hasattr(self, 'performance_weights_tensor') or self.performance_weights_tensor is None:
            logger.warning("No performance weights tensor set, using simple average")
            return VotingStrategies.simple_average(predictions)
        
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list, dim=0)  # [n_models, batch, ...]
        
        # Expand weights to match prediction dimensions
        weights = self.performance_weights_tensor  # [n_models]
        
        if pred_stack.dim() == 3:
            # Multi-class case: [n_models, batch, classes]
            weights = weights.unsqueeze(1).unsqueeze(2)  # [n_models, 1, 1]
            weights = weights.expand_as(pred_stack)
        else:
            # Binary case: [n_models, batch]
            weights = weights.unsqueeze(1)  # [n_models, 1]
            weights = weights.expand_as(pred_stack)
        
        # Weighted sum
        weighted_pred = torch.sum(pred_stack * weights, dim=0)
        
        # Normalize by sum of weights
        weight_sum = self.performance_weights_tensor.sum()
        if pred_stack.dim() == 3:
            weight_sum = weight_sum.unsqueeze(0).unsqueeze(1)
        else:
            weight_sum = weight_sum.unsqueeze(0)
        
        return weighted_pred / weight_sum
    
    def get_model_contributions(self, X: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about each model's contribution"""
        predictions = self._get_individual_predictions(X)
        
        contributions = {}
        
        for i, (name, pred) in enumerate(predictions.items()):
            model = self.models[i]
            
            # Basic prediction stats
            if pred.dim() > 1:
                # Multi-class
                confidence = torch.max(pred, dim=1)[0].mean().item()
                entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1).mean().item()
            else:
                # Binary
                confidence = torch.abs(pred - 0.5).mean().item() * 2
                p = pred
                entropy = -(p * torch.log(p + 1e-8) + (1-p) * torch.log(1-p + 1e-8)).mean().item()
            
            # Performance weight
            perf_weight = self.performance_weights.get(name, 1.0)
            
            contributions[name] = {
                'model_type': type(model).__name__,
                'prediction_shape': list(pred.shape),
                'avg_confidence': confidence,
                'avg_entropy': entropy,
                'performance_weight': perf_weight,
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
        
        return contributions
    
    def save_ensemble(self, save_dir: str):
        """Save ensemble models and configuration"""
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f"model_{i}.pth")
            torch.save(model.state_dict(), model_path)
        
        # Save ensemble configuration
        config = {
            'strategy': self.strategy,
            'model_count': len(self.models),
            'model_types': [type(model).__name__ for model in self.models],
            'performance_weights': self.performance_weights,
            'device': str(self.device)
        }
        
        config_path = os.path.join(save_dir, "ensemble_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Ensemble saved to {save_dir}")
    
    def load_ensemble(self, load_dir: str, model_factory_fn):
        """Load ensemble models and configuration"""
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(load_dir, "ensemble_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Restore ensemble settings
        self.strategy = config['strategy']
        self.performance_weights = config.get('performance_weights', {})
        
        # Load individual models
        self.models = []
        for i in range(config['model_count']):
            model_path = os.path.join(load_dir, f"model_{i}.pth")
            
            # Create model using factory function
            model = model_factory_fn(config['model_types'][i])
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            
            self.models.append(model)
        
        # Set performance weights if available
        if self.performance_weights:
            self.set_performance_weights(self.performance_weights)
        
        logger.info(f"📂 Ensemble loaded from {load_dir}")
    
    def __len__(self):
        """Return number of models in ensemble"""
        return len(self.models)
    
    def __getitem__(self, idx):
        """Get model by index"""
        return self.models[idx]
    
    def __repr__(self):
        return f"EnsembleClassifier(models={len(self.models)}, strategy='{self.strategy}', device='{self.device}')"
