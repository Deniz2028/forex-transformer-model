# src/losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class VolatilityAdaptiveFocalLoss(nn.Module):
    def __init__(self, base_alpha=0.25, base_gamma=2.0):
        super().__init__()
        self.base_alpha = base_alpha
        self.base_gamma = base_gamma
        
    def forward(self, inputs, targets, volatility_factor=1.0):
        alpha = self.base_alpha * (0.5 + 0.5 * volatility_factor)
        gamma = self.base_gamma * (0.8 + 0.4 * volatility_factor)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt)**gamma * ce_loss
        
        return focal_loss.mean()

class ForexFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, inputs, targets, market_regime='normal'):
        if market_regime == 'high_volatility':
            gamma = self.gamma * 1.25
        elif market_regime == 'low_volatility':
            gamma = self.gamma * 0.8
        else:
            gamma = self.gamma
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.class_weights is not None:
            alpha_t = self.class_weights[targets]
        else:
            alpha_t = self.alpha
        
        focal_loss = alpha_t * (1 - pt)**gamma * ce_loss
        return focal_loss.mean()
