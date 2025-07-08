# ========================
# src/ensemble/meta_learner.py
"""Meta-learning for ensemble combination."""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MetaLearner(nn.Module):
    """Meta-learner for combining ensemble predictions"""
    
    def __init__(self, n_models: int, n_classes: int, hidden_dim: int = 64, 
                 use_market_features: bool = True):
        super().__init__()
        
        self.n_models = n_models
        self.n_classes = n_classes
        self.use_market_features = use_market_features
        
        # Input: predictions from all models + optional market metadata
        input_dim = n_models * n_classes
        if use_market_features:
            input_dim += 10  # Market features (volatility, volume, etc.)
        
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Feature importance attention
        self.attention = nn.MultiheadAttention(
            embed_dim=n_classes,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        logger.info(f"🧠 MetaLearner initialized: {n_models} models → {n_classes} classes")
        
    def forward(self, model_predictions: torch.Tensor, 
                market_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            model_predictions: [batch, n_models, n_classes]
            market_features: [batch, n_market_features] (optional)
        """
        batch_size = model_predictions.shape[0]
        
        # Apply attention to model predictions
        attended_preds, attention_weights = self.attention(
            model_predictions, model_predictions, model_predictions
        )
        
        # Flatten attended predictions
        flat_preds = attended_preds.view(batch_size, -1)
        
        # Combine with market features if available
        if self.use_market_features and market_features is not None:
            combined = torch.cat([flat_preds, market_features], dim=1)
        else:
            combined = flat_preds
        
        return self.meta_net(combined)


class AdvancedEnsemble(nn.Module):
    """Advanced ensemble with meta-learning"""
    
    def __init__(self, base_models: List[nn.Module], n_classes: int, 
                 use_meta_learning: bool = True):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.use_meta_learning = use_meta_learning
        
        if use_meta_learning:
            self.meta_learner = MetaLearner(len(base_models), n_classes)
            logger.info(f"🎭 AdvancedEnsemble with MetaLearner: {len(base_models)} models")
        else:
            self.meta_learner = None
            logger.info(f"🎭 AdvancedEnsemble without MetaLearner: {len(base_models)} models")
        
    def forward(self, X: torch.Tensor, market_features: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through advanced ensemble"""
        # Get predictions from all base models
        model_preds = []
        for model in self.base_models:
            pred = torch.softmax(model(X), dim=1)
            model_preds.append(pred)
            
        model_preds = torch.stack(model_preds, dim=1)  # [batch, n_models, n_classes]
        
        if self.use_meta_learning and self.meta_learner is not None:
            # Meta-learner final decision
            return self.meta_learner(model_preds, market_features)
        else:
            # Simple average fallback
            return torch.mean(model_preds, dim=1)
    
    def get_model_contributions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual model contributions"""
        contributions = {}
        for i, model in enumerate(self.base_models):
            with torch.no_grad():
                pred = torch.softmax(model(X), dim=1)
                contributions[f'model_{i}'] = pred
        return contributions