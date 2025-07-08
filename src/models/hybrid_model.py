# ========================
# src/models/hybrid_model.py
"""Hybrid LSTM-Transformer Model Implementation - Loss Function Fixed"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)


class ConcatenateFusion(nn.Module):
    """Simple concatenation fusion"""
    
    def __init__(self, lstm_dim: int, transformer_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(lstm_dim + transformer_dim, (lstm_dim + transformer_dim) // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear((lstm_dim + transformer_dim) // 2, (lstm_dim + transformer_dim) // 4)
        )
    
    def forward(self, lstm_features: torch.Tensor, transformer_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        return self.fusion(combined)


class AttentionFusion(nn.Module):
    """Cross-attention based fusion"""
    
    def __init__(self, lstm_dim: int, transformer_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Project LSTM to transformer dimension
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
        
    def forward(self, lstm_features: torch.Tensor, transformer_features: torch.Tensor) -> torch.Tensor:
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
    """Gated fusion mechanism"""
    
    def __init__(self, lstm_dim: int, transformer_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Project LSTM to transformer dimension
        self.lstm_projection = nn.Linear(lstm_dim, transformer_dim)
        
        # Gating mechanism
        self.gate_lstm = nn.Linear(transformer_dim, transformer_dim)
        self.gate_transformer = nn.Linear(transformer_dim, transformer_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lstm_features: torch.Tensor, transformer_features: torch.Tensor) -> torch.Tensor:
        # Project LSTM to transformer space
        lstm_proj = self.lstm_projection(lstm_features)
        
        # Compute gates
        gate_l = torch.sigmoid(self.gate_lstm(lstm_proj))
        gate_t = torch.sigmoid(self.gate_transformer(transformer_features))
        
        # Gated combination
        fused = gate_l * lstm_proj + gate_t * transformer_features
        
        return self.dropout(fused)


class HybridLSTMTransformer(nn.Module):
    """Hybrid LSTM-Transformer model combining both architectures"""
    
    def __init__(self, input_dim: int, lstm_hidden: int = 96, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1,
                 num_classes: int = 2, fusion_strategy: str = 'concat'):
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        
        # TARGET_MODE ATTRIBUTE - DÜZELTME: Binary classification için
        if num_classes == 2:
            self.target_mode = 'binary'
            self.final_output_size = 1  # Binary için tek output
        elif num_classes == 3:
            self.target_mode = 'three_class'
            self.final_output_size = 3  # Three-class için 3 output
        else:
            self.target_mode = 'binary'
            self.final_output_size = 1
        
        logger.info(f"🎯 Hybrid model target mode: {self.target_mode}, output size: {self.final_output_size}")
        
        # Validate attention heads
        if d_model % nhead != 0:
            logger.warning(f"d_model ({d_model}) is not divisible by nhead ({nhead})")
            nhead = min(nhead, d_model)
            while d_model % nhead != 0 and nhead > 1:
                nhead -= 1
            logger.warning(f"Adjusted nhead to {nhead}")
            self.nhead = nhead
        
        # LSTM component
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Input projection for transformer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer component
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature fusion
        lstm_output_dim = lstm_hidden  # Not bidirectional
        if fusion_strategy == 'concat':
            self.fusion = ConcatenateFusion(lstm_output_dim, d_model)
            final_dim = (lstm_output_dim + d_model) // 4
        elif fusion_strategy == 'attention':
            self.fusion = AttentionFusion(lstm_output_dim, d_model, dropout)
            final_dim = d_model
        elif fusion_strategy == 'gated':
            self.fusion = GatedFusion(lstm_output_dim, d_model, dropout)
            final_dim = d_model
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Classification head - DÜZELTME: Target mode'a göre output size
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, self.final_output_size)  # Binary: 1, Multi: 3
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ HybridLSTMTransformer initialized:")
        logger.info(f"   LSTM hidden: {lstm_hidden}, Transformer d_model: {d_model}")
        logger.info(f"   Fusion strategy: {fusion_strategy}")
        logger.info(f"   Target mode: {self.target_mode}, Output size: {self.final_output_size}")

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
            x: Input tensor [batch_size, seq_len, features] or [batch_size, features]
            
        Returns:
            Class predictions:
            - Binary mode: [batch_size, 1] with sigmoid probabilities
            - Three-class mode: [batch_size, 3] with logits
        """
        # Handle input dimensions
        if x.dim() == 2:
            # [batch_size, features] -> [batch_size, 1, features]
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            # [features] -> [1, 1, features]
            x = x.unsqueeze(0).unsqueeze(0)
        
        batch_size, seq_len, features = x.shape
        
        # LSTM Branch - Sequential Processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state as LSTM representation
        if len(hidden.shape) == 3:  # [num_layers, batch, hidden]
            lstm_features = hidden[-1]  # Son layer'ın hidden state'i [batch, hidden]
        else:
            lstm_features = hidden  # Zaten doğru shape'de
        
        # Transformer Branch - Attention-based Processing
        x_projected = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x_embedded = self.pos_encoder(x_projected)  # Positional encoding ekle
        
        transformer_out = self.transformer(x_embedded)  # [batch_size, seq_len, d_model]
        
        # Global average pooling for sequence representation
        transformer_features = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # Feature Fusion
        if self.fusion_strategy == 'concat':
            # Concatenate LSTM and Transformer features
            fused_features = self.fusion(lstm_features, transformer_features)
            
        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            fused_features = self.fusion(lstm_features, transformer_features)
            
        elif self.fusion_strategy == 'gated':
            # Gated fusion
            fused_features = self.fusion(lstm_features, transformer_features)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Classification
        output = self.classifier(fused_features)
        
        # DÜZELTME: Output processing based on target mode
        if self.target_mode == 'binary':
            # Binary classification: sigmoid to get probabilities [0, 1]
            pass  # logits olarak bırak
        else:
            # Three-class: return logits (softmax will be applied in loss function)
            pass  # Keep as logits [batch_size, 3]
        
        return output


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
    
    # DÜZELTME: Target mode'a göre doğru class sayısını belirle
    target_mode = config.get('target_mode', model_config.get('target_mode', 'binary'))
    
    if target_mode == 'binary':
        num_classes = 2  # Binary classification için 2 class (ama output 1 olacak)
    elif target_mode == 'three_class':
        num_classes = 3  # Three-class classification
    else:
        logger.warning(f"Unknown target mode: {target_mode}, defaulting to binary")
        target_mode = 'binary'
        num_classes = 2
    
    logger.info(f"🏗️ Creating hybrid model: target_mode={target_mode}, num_classes={num_classes}")
    
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
    
    # SET TARGET_MODE explicitly if not set during initialization
    model.target_mode = target_mode
    
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
        # Auto-correct nhead
        new_nhead = min(nhead, d_model)
        while d_model % new_nhead != 0 and new_nhead > 1:
            new_nhead -= 1
        hybrid_config['nhead'] = new_nhead
        logger.warning(f"nhead adjusted from {nhead} to {new_nhead} for d_model={d_model}")
    
    if fusion_strategy not in ['concat', 'attention', 'gated']:
        raise ValueError(f"fusion_strategy must be one of ['concat', 'attention', 'gated'], got {fusion_strategy}")
    
    logger.info(f"✅ Hybrid config validated: LSTM({lstm_hidden}), Transformer({d_model}), fusion({fusion_strategy})")
    
    return config


# Test function
def test_hybrid_model():
    """Test hybrid model with different configurations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    configs = [
        {'target_mode': 'binary', 'hybrid': {'fusion_strategy': 'concat'}},
        {'target_mode': 'three_class', 'hybrid': {'fusion_strategy': 'attention'}},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Testing config {i+1}: {config}")
        
        try:
            model = create_hybrid_model(config, n_features=23, device=device)
            
            # Test forward pass
            batch_size = 8
            if config['target_mode'] == 'binary':
                test_input = torch.randn(batch_size, 23).to(device)
            else:
                test_input = torch.randn(batch_size, 64, 23).to(device)
            
            output = model(test_input)
            
            expected_shape = (batch_size, 1) if config['target_mode'] == 'binary' else (batch_size, 3)
            assert output.shape == expected_shape, f"Wrong output shape: {output.shape} vs {expected_shape}"
            
            print(f"✅ Test passed: input {test_input.shape} -> output {output.shape}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_hybrid_model()
