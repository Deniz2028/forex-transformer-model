"""
Enhanced Loss Functions for Financial Time Series Classification

Bu modül, finansal zaman serisi sınıflandırması için optimize edilmiş kayıp fonksiyonları içerir.
Class imbalance, market volatility ve PDF optimizasyonları ile uyumlu tasarlandı.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Dict, Any


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
        label_smoothing: Label smoothing factor (default: 0.0)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predicted probabilities or raw logits [batch_size, num_classes or 1]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Computed focal loss
        """
        # Handle binary classification
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Convert raw logits to probabilities if needed
        if inputs.requires_grad:  # Likely raw logits
            probs = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:  # Already probabilities
            probs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
            ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Calculate focal loss components
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DynamicFocalLoss(nn.Module):
    """
    Dynamic Focal Loss with market volatility adaptation.
    
    PDF önerisi: Market volatility'e göre gamma parametresini dinamik olarak ayarla.
    
    Args:
        alpha: Balancing parameter (default: 0.25)
        gamma: Base focusing parameter (default: 2.0)
        volatility_adjustment: Enable volatility-based gamma adjustment
        min_gamma: Minimum gamma value (default: 0.5)
        max_gamma: Maximum gamma value (default: 3.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 volatility_adjustment: bool = True, min_gamma: float = 0.5, 
                 max_gamma: float = 3.0):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.volatility_adjustment = volatility_adjustment
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional volatility adjustment.
        
        Args:
            inputs: Raw model logits [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size]
            volatility: Market volatility tensor [batch_size] (optional)
            
        Returns:
            Dynamic focal loss
        """
        # Handle dimension mismatch
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        # Calculate BCE loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Dynamic gamma adjustment based on market volatility
        if self.volatility_adjustment and volatility is not None:
            # Ensure volatility has same shape as inputs
            if volatility.dim() != inputs.dim():
                volatility = volatility.expand_as(inputs)
            
            # Higher volatility = higher gamma (focus more on hard examples during volatile periods)
            # Normalize volatility to [0, 1] range and apply to gamma
            volatility_norm = torch.clamp(volatility, 0, 1)
            dynamic_gamma = self.gamma + volatility_norm * (self.max_gamma - self.gamma)
            dynamic_gamma = torch.clamp(dynamic_gamma, self.min_gamma, self.max_gamma)
        else:
            dynamic_gamma = self.gamma
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** dynamic_gamma * ce_loss
        
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with optional positive weight.
    
    Args:
        pos_weight: Weight for positive class
        reduction: Reduction method
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of weighted BCE loss."""
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                inputs, targets, reduction=self.reduction
            )


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for three-class classification.
    
    Args:
        alpha: Class weights [num_classes]
        gamma: Focusing parameter
        reduction: Reduction method
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of multi-class focal loss."""
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Factory Functions
def get_loss_function(loss_type: str = 'bce', target_mode: str = 'binary', 
                      use_focal_loss: bool = False, class_weights: Optional[torch.Tensor] = None,
                      pos_weight: Optional[Union[float, torch.Tensor]] = None, 
                      device: Optional[torch.device] = None, 
                      **kwargs) -> nn.Module:
    """
    Factory function to get appropriate loss function.
    
    Args:
        loss_type: Type of loss ('focal', 'dynamic_focal', 'bce', 'weighted_bce', 'crossentropy')
        target_mode: Target mode ('binary', 'three_class')
        use_focal_loss: Whether to use focal loss (fallback parameter)
        class_weights: Weights for each class [num_classes]
        pos_weight: Weight for positive class in binary classification
        device: Device to put tensors on
        **kwargs: Additional arguments for loss functions
        
    Returns:
        Appropriate loss function
    """
    # Handle legacy parameter
    if use_focal_loss and loss_type == 'bce':
        loss_type = 'focal'
    
    # Move weights to device if specified
    if device:
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if pos_weight is not None and isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)
    
    if target_mode == 'binary':
        if loss_type == 'focal':
            alpha = kwargs.get('focal_alpha', 1.0)
            gamma = kwargs.get('focal_gamma', 2.0)
            label_smoothing = kwargs.get('label_smoothing', 0.0)
            return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
            
        elif loss_type == 'dynamic_focal':
            alpha = kwargs.get('focal_alpha', 0.25)
            gamma = kwargs.get('focal_gamma', 2.0)
            return DynamicFocalLoss(alpha=alpha, gamma=gamma)
            
        elif loss_type == 'weighted_bce':
            return WeightedBCELoss(pos_weight=pos_weight)
            
        elif loss_type == 'bce_with_logits':
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor(pos_weight) if not isinstance(pos_weight, torch.Tensor) else pos_weight
                if device:
                    pos_weight_tensor = pos_weight_tensor.to(device)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                return nn.BCEWithLogitsLoss()
                
        else:  # Default: 'bce'
            return nn.BCELoss()
            
    else:  # three_class
        if loss_type == 'focal':
            alpha = class_weights
            gamma = kwargs.get('focal_gamma', 2.0)
            return MultiClassFocalLoss(alpha=alpha, gamma=gamma)
            
        elif loss_type == 'weighted_crossentropy':
            return nn.CrossEntropyLoss(weight=class_weights)
            
        else:  # Default: 'crossentropy'
            return nn.CrossEntropyLoss()


def calculate_class_weights(y: torch.Tensor, num_classes: int = 2, 
                           method: str = 'balanced') -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        num_classes: Number of classes
        method: Weighting method ('balanced', 'inverse_freq')
        
    Returns:
        Class weights tensor
    """
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.array(y)
    
    # Count class frequencies
    class_counts = np.bincount(y_np.astype(int), minlength=num_classes)
    
    if method == 'balanced':
        # Sklearn-style balanced weights
        n_samples = len(y_np)
        weights = n_samples / (num_classes * class_counts)
    elif method == 'inverse_freq':
        # Simple inverse frequency
        weights = 1.0 / (class_counts + 1e-8)
    else:
        # Uniform weights
        weights = np.ones(num_classes)
    
    return torch.FloatTensor(weights)


def calculate_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """
    Calculate positive class weight for binary classification.
    
    Args:
        y: Binary target labels
        
    Returns:
        Positive weight tensor
    """
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.array(y)
    
    # Count positive and negative samples
    pos_count = np.sum(y_np == 1)
    neg_count = np.sum(y_np == 0)
    
    if pos_count == 0:
        return torch.tensor(1.0)
    
    pos_weight = neg_count / pos_count
    return torch.tensor(pos_weight, dtype=torch.float32)


# Convenience functions for quick loss setup
def get_forex_loss(target_mode: str = 'binary', use_focal: bool = True, 
                   class_imbalance_ratio: float = 1.0, device: Optional[torch.device] = None) -> nn.Module:
    """
    Get loss function optimized for forex trading.
    
    Args:
        target_mode: 'binary' or 'three_class'
        use_focal: Whether to use focal loss
        class_imbalance_ratio: Ratio of negative to positive samples
        device: PyTorch device
        
    Returns:
        Optimized loss function for forex
    """
    if target_mode == 'binary':
        if use_focal:
            # Forex-optimized focal loss parameters
            alpha = 0.25 if class_imbalance_ratio < 2.0 else 0.75
            gamma = 2.0
            return FocalLoss(alpha=alpha, gamma=gamma)
        else:
            # Use weighted BCE for class imbalance
            pos_weight = torch.tensor(class_imbalance_ratio) if class_imbalance_ratio != 1.0 else None
            if device and pos_weight is not None:
                pos_weight = pos_weight.to(device)
            return WeightedBCELoss(pos_weight=class_imbalance_ratio if class_imbalance_ratio != 1.0 else None)
    else:
        # Three-class classification
        if use_focal:
            return MultiClassFocalLoss(gamma=2.0)
        else:
            return nn.CrossEntropyLoss()


# Export all classes and functions
__all__ = [
    'FocalLoss', 
    'DynamicFocalLoss', 
    'WeightedBCELoss',
    'MultiClassFocalLoss',
    'get_loss_function',
    'calculate_class_weights',
    'calculate_pos_weight',
    'get_forex_loss'
]
