# src/models/__init__.py
"""Model definitions and utilities."""

from .lstm import PairSpecificLSTM, create_model as create_lstm_model
from .losses import FocalLoss, get_loss_function

# Enhanced Transformer için koşullu import
try:
    from .enhanced_transformer import (
        EnhancedTransformer,
        create_enhanced_transformer,
        PositionalEncoding,
        MultiHeadAttention,
        TransformerBlock
    )
    ENHANCED_TRANSFORMER_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSFORMER_AVAILABLE = False
    EnhancedTransformer = None
    create_enhanced_transformer = None
    PositionalEncoding = None
    MultiHeadAttention = None
    TransformerBlock = None

# Factory import
from .factory import create_model, get_model_info, suggest_training_params

# Base exports
__all__ = [
    'PairSpecificLSTM', 
    'create_lstm_model',
    'FocalLoss', 
    'get_loss_function',
    'create_model',
    'get_model_info',
    'suggest_training_params'
]

# Enhanced Transformer exports (conditional)
if ENHANCED_TRANSFORMER_AVAILABLE:
    __all__.extend([
        'EnhancedTransformer',
        'create_enhanced_transformer',
        'PositionalEncoding',
        'MultiHeadAttention',
        'TransformerBlock'
    ])
