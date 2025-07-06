# src/models/hybrid_lstm_transformer.py
"""
Hibrit LSTM-Transformer Model Implementation
Mevcut proje yapısına uygun şekilde tasarlandı.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class HybridLSTMTransformer(nn.Module):
    """
    Hibrit LSTM-Transformer model.
    
    LSTM: Sequential pattern learning için
    Transformer: Attention-based feature extraction için
    Fusion: İki branch'i birleştiren intelligent layer
    """
    
    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 96,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
        fusion_strategy: str = 'concat'  # 'concat', 'attention', 'gated'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.d_model = d_model
        self.fusion_strategy = fusion_strategy
        
        # LSTM Branch - Sequential Pattern Learning
        self.lstm = nn.LSTM(
            input_dim, 
            lstm_hidden, 
            num_layers=2,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Transformer Branch - Attention-based Learning
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = PositionalEmbedding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Fusion Layer
        if fusion_strategy == 'concat':
            fusion_input_dim = lstm_hidden + d_model
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_input_dim // 2, fusion_input_dim // 4)
            )
            classifier_input = fusion_input_dim // 4
            
        elif fusion_strategy == 'attention':
            self.fusion_attention = CrossAttentionFusion(lstm_hidden, d_model, dropout)
            classifier_input = d_model
            
        elif fusion_strategy == 'gated':
            self.fusion = GatedFusion(lstm_hidden, d_model, dropout)
            classifier_input = d_model
            
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ HybridLSTMTransformer initialized:")
        logger.info(f"   LSTM hidden: {lstm_hidden}, Transformer d_model: {d_model}")
        logger.info(f"   Fusion strategy: {fusion_strategy}")
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Class predictions [batch_size, num_classes]
        """
        batch_size, seq_len, features = x.shape
        
        # LSTM Branch - Sequential Processing
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state as LSTM representation
        lstm_features = hidden[-1]  # [batch_size, lstm_hidden]
        
        # Transformer Branch - Attention-based Processing
        x_projected = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x_embedded = self.pos_embedding(x_projected)
        
        transformer_out = self.transformer(x_embedded)  # [batch_size, seq_len, d_model]
        # Global average pooling for sequence representation
        transformer_features = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # Feature Fusion
        if self.fusion_strategy == 'concat':
            # Simple concatenation fusion
            combined = torch.cat([lstm_features, transformer_features], dim=1)
            fused = self.fusion(combined)
            
        elif self.fusion_strategy == 'attention':
            # Cross-attention fusion
            fused = self.fusion_attention(lstm_features, transformer_features)
            
        elif self.fusion_strategy == 'gated':
            # Gated fusion
            fused = self.fusion(lstm_features, transformer_features)
        
        # Apply dropout and classify
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.embedding[:seq_len, :].unsqueeze(0)


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion of LSTM and Transformer features"""
    
    def __init__(self, lstm_dim: int, transformer_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Project LSTM features to transformer dimension
        self.lstm_projection = nn.Linear(lstm_dim, transformer_dim)
        
        # Cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=transformer_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(transformer_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lstm_features: torch.Tensor, 
                transformer_features: torch.Tensor) -> torch.Tensor:
        # Project LSTM to transformer space
        lstm_proj = self.lstm_projection(lstm_features).unsqueeze(1)  # Add seq dim
        transformer_query = transformer_features.unsqueeze(1)
        
        # Cross-attention: transformer attends to LSTM
        attended, _ = self.attention(
            query=transformer_query,
            key=lstm_proj,
            value=lstm_proj
        )
        
        # Residual connection and normalization
        fused = self.norm(transformer_query + self.dropout(attended))
        
        return fused.squeeze(1)  # Remove seq dimension


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining LSTM and Transformer features"""
    
    def __init__(self, lstm_dim: int, transformer_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Project LSTM to transformer dimension
        self.lstm_projection = nn.Linear(lstm_dim, transformer_dim)
        
        # Gating mechanism
        self.gate_lstm = nn.Linear(transformer_dim, transformer_dim)
        self.gate_transformer = nn.Linear(transformer_dim, transformer_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lstm_features: torch.Tensor, 
                transformer_features: torch.Tensor) -> torch.Tensor:
        # Project LSTM to transformer space
        lstm_proj = self.lstm_projection(lstm_features)
        
        # Compute gates
        gate_l = torch.sigmoid(self.gate_lstm(lstm_proj))
        gate_t = torch.sigmoid(self.gate_transformer(transformer_features))
        
        # Gated combination
        fused = gate_l * lstm_proj + gate_t * transformer_features
        
        return self.dropout(fused)


def create_hybrid_model(config: Dict[str, Any], n_features: int, 
                       device: torch.device) -> HybridLSTMTransformer:
    """
    Factory function for creating hybrid LSTM-Transformer model
    
    Args:
        config: Model configuration dictionary
        n_features: Number of input features
        device: Target device
        
    Returns:
        Initialized HybridLSTMTransformer model
    """
    # Extract hybrid-specific config
    hybrid_config = config.get('hybrid', {})
    model_config = config.get('model', {})
    
    # Default parameters
    lstm_hidden = hybrid_config.get('lstm_hidden', 96)
    d_model = hybrid_config.get('d_model', 512)
    nhead = hybrid_config.get('nhead', 8)
    num_layers = hybrid_config.get('num_layers', 4)
    dropout = model_config.get('dropout_rate', 0.1)
    fusion_strategy = hybrid_config.get('fusion_strategy', 'concat')
    
    # Determine number of classes from target mode
    target_mode = config.get('target_mode', 'three_class')
    num_classes = 3 if target_mode == 'three_class' else 2
    
    model = HybridLSTMTransformer(
        input_dim=n_features,
        lstm_hidden=lstm_hidden,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        fusion_strategy=fusion_strategy
    ).to(device)
    
    logger.info(f"✅ Hybrid model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def validate_hybrid_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate hybrid model configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    import copy
    config = copy.deepcopy(config)
    
    hybrid_config = config.get('hybrid', {})
    
    # Validate parameters
    lstm_hidden = hybrid_config.get('lstm_hidden', 96)
    d_model = hybrid_config.get('d_model', 512)
    nhead = hybrid_config.get('nhead', 8)
    num_layers = hybrid_config.get('num_layers', 4)
    fusion_strategy = hybrid_config.get('fusion_strategy', 'concat')
    
    # Validation checks
    if not isinstance(lstm_hidden, int) or lstm_hidden < 32 or lstm_hidden > 512:
        raise ValueError(f"lstm_hidden must be between 32 and 512, got {lstm_hidden}")
    
    if not isinstance(d_model, int) or d_model < 128 or d_model > 1024:
        raise ValueError(f"d_model must be between 128 and 1024, got {d_model}")
    
    if d_model % nhead != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
    
    if fusion_strategy not in ['concat', 'attention', 'gated']:
        raise ValueError(f"fusion_strategy must be one of ['concat', 'attention', 'gated'], got {fusion_strategy}")
    
    logger.info(f"✅ Hybrid config validated: LSTM({lstm_hidden}), Transformer({d_model}), fusion({fusion_strategy})")
    
    return config
