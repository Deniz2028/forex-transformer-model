"""
Enhanced Transformer Model with advanced features for time series classification.

Bu modül, forex tahmini için özel olarak tasarlanmış gelişmiş transformer mimarisi içerir:
- Multi-head attention with custom positional encoding
- Residual connections and layer normalization
- Advanced feature extraction layers
- Dropout and regularization techniques
- Binary and three-class classification support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds position information to input embeddings using sine and cosine functions
    of different frequencies. This helps the model understand sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Custom multi-head attention implementation with improved features.
    
    Features:
    - Scaled dot-product attention
    - Multiple attention heads
    - Dropout regularization
    - Optional attention weights output
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            (output, attention_weights) tuple
        """
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Attention computation
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights.mean(dim=1)  # Average over heads


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    
    Components:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections
    - Layer normalization
    - Dropout regularization
    """
    
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        
        # Feed-forward with residual connection
        residual = attn_output
        ff_output = self.ff_network(attn_output)
        output = self.layer_norm(ff_output + residual)
        
        return output


class EnhancedTransformer(nn.Module):
    """
    Enhanced Transformer model for forex time series classification.
    
    Features:
    - Input embedding and positional encoding
    - Multiple transformer blocks
    - Global feature extraction
    - Classification head with dropout
    - Binary and three-class support
    - Advanced regularization
    """
    
    def __init__(self, 
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 12,
                 num_layers: int = 6,
                 ff_dim: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 128,
                 target_mode: str = 'binary'):
        super(EnhancedTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.target_mode = target_mode
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Global feature extraction
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concat avg and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        if target_mode == 'binary':
            self.classifier = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
                # Sigmoid removed - will use BCEWithLogitsLoss instead
            )
        else:  # three_class
            self.classifier = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 3),
                nn.Softmax(dim=-1)
            )
        
        # Weight initialization
        self._init_weights()
        
        logger.info(f"✅ EnhancedTransformer initialized:")
        logger.info(f"   Input size: {input_size}")
        logger.info(f"   Model dim: {d_model}")
        logger.info(f"   Heads: {nhead}")
        logger.info(f"   Layers: {num_layers}")
        logger.info(f"   FF dim: {ff_dim}")
        logger.info(f"   Target mode: {target_mode}")
        logger.info(f"   Total parameters: {self.get_parameter_count():,}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the enhanced transformer.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Output predictions
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Positional encoding (convert to seq_first format)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Global feature extraction
        # x: (batch_size, seq_len, d_model)
        x_transposed = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        avg_pooled = self.global_pooling(x_transposed).squeeze(-1)  # (batch_size, d_model)
        max_pooled = self.global_max_pooling(x_transposed).squeeze(-1)  # (batch_size, d_model)
        
        # Concatenate pooled features
        global_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch_size, d_model * 2)
        
        # Feature fusion
        fused_features = self.feature_fusion(global_features)  # (batch_size, d_model // 2)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'EnhancedTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'max_seq_len': self.max_seq_len,
            'target_mode': self.target_mode,
            'architecture': f'Enhanced Transformer (d_model={self.d_model}, heads={self.nhead}, layers={self.num_layers})'
        }
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weights from each layer
        """
        attention_weights = []
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Collect attention weights from each layer
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block.attention(x, x, x)
            attention_weights.append(attn_weights.detach())
        
        return attention_weights


def create_enhanced_transformer(config: Dict[str, Any], input_size: int, device: torch.device) -> EnhancedTransformer:
    """
    Factory function to create Enhanced Transformer from configuration.
    
    Args:
        config: Model configuration dictionary
        input_size: Number of input features
        device: Device to create model on
        
    Returns:
        Initialized Enhanced Transformer model
    """
    model_config = config.get('model', {})
    transformer_config = config.get('transformer', {})
    
    # Extract transformer-specific parameters with defaults
    d_model = transformer_config.get('d_model', 256)
    nhead = transformer_config.get('nhead', 12)
    num_layers = transformer_config.get('num_layers', 6)
    ff_dim = transformer_config.get('ff_dim', 512)
    dropout = model_config.get('dropout_rate', 0.1)
    max_seq_len = config.get('data', {}).get('sequence_length', 128)
    target_mode = model_config.get('target_mode', 'binary')
    
    logger.info(f"🏭 Creating Enhanced Transformer with:")
    logger.info(f"   d_model: {d_model}")
    logger.info(f"   nhead: {nhead}")
    logger.info(f"   num_layers: {num_layers}")
    logger.info(f"   ff_dim: {ff_dim}")
    logger.info(f"   dropout: {dropout}")
    logger.info(f"   max_seq_len: {max_seq_len}")
    logger.info(f"   target_mode: {target_mode}")
    
    model = EnhancedTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        max_seq_len=max_seq_len,
        target_mode=target_mode
    )
    
    return model.to(device)


__all__ = [
    'EnhancedTransformer',
    'create_enhanced_transformer',
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock'
]
