"""Loss functions for the LSTM model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance.
    
    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.
    
    Args:
        alpha: Weighting factor for rare class (default: 1)
        gamma: Focusing parameter (default: 2)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Focal Loss.
        
        Args:
            inputs: Predicted probabilities or raw logits
            targets: Ground truth labels
            
        Returns:
            Computed focal loss
        """
        # Convert raw logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DynamicFocalLoss(nn.Module):
    """Dynamic Focal Loss with market volatility adaptation
    
    Args:
        alpha: Balancing parameter (default: 0.25)
        gamma: Base focusing parameter (default: 2.0)
        volatility_adjustment: Enable volatility-based gamma adjustment (default: True)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, volatility_adjustment=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.volatility_adjustment = volatility_adjustment
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                volatility: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with optional volatility adjustment
        
        Args:
            inputs: Raw model logits
            targets: Ground truth labels
            volatility: Market volatility tensor (optional)
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Dynamic gamma adjustment based on market volatility
        if self.volatility_adjustment and volatility is not None:
            # Higher volatility = lower gamma (focus less on hard examples)
            # Clamp volatility between 0-0.5 to prevent negative gamma
            dynamic_gamma = self.gamma * (1.0 - volatility.clamp(0, 0.5))
        else:
            dynamic_gamma = self.gamma
        
        focal_loss = self.alpha * (1 - pt) ** dynamic_gamma * ce_loss
        
        return focal_loss.mean()

def get_loss_function(loss_type: str, target_mode: str, class_weights: torch.Tensor = None, 
                      pos_weight: torch.Tensor = None, device: torch.device = None):
    """Factory function to get appropriate loss function.
    
    Args:
        loss_type: Type of loss ('focal', 'dynamic_focal', 'bce', 'crossentropy')
        target_mode: Target mode ('binary', 'three_class')
        class_weights: Weights for each class
        pos_weight: Weight for positive class in binary classification
        device: Device to put tensors on
        
    Returns:
        Appropriate loss function
    """
    if target_mode == 'binary':
        if loss_type == 'focal':
            return FocalLoss(alpha=1, gamma=2)
        elif loss_type == 'dynamic_focal':
            return DynamicFocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'bce_weighted' and pos_weight is not None:
            if device:
                pos_weight = pos_weight.to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCELoss()
    else:  # three_class
        if class_weights is not None:
            if device:
                class_weights = class_weights.to(device)
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()

__all__ = ['FocalLoss', 'DynamicFocalLoss', 'get_loss_function']
