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
import copy
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Define type aliases for clarity
TransformerConfig = Dict[str, Any]
LSTMConfig = Dict[str, Any]
ModelInstance = nn.Module
HYBRID_MODEL_AVAILABLE = False  # Flag for hybrid model availability

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


def validate_transformer_config(config: TransformerConfig) -> TransformerConfig:
    """
    Transformer konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    d_model = model_config.get('d_model', 256)
    nhead = model_config.get('nhead', 8)
    num_layers = model_config.get('num_layers', 4)
    dropout = model_config.get('dropout_rate', 0.1)
    
    logger.info(f"🔍 Transformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # d_model validasyonu
    if not isinstance(d_model, int) or d_model < 64 or d_model > 1024:
        raise ValueError(f"d_model geçersiz: {d_model}. 64 ile 1024 arasında integer olmalı")
    
    # nhead validasyonu
    if not isinstance(nhead, int) or nhead < 1 or nhead > 16:
        raise ValueError(f"nhead geçersiz: {nhead}. 1 ile 16 arasında integer olmalı")
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            model_config['nhead'] = nhead
            logger.warning(f"   🔧 nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı")
    
    # num_layers validasyonu
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 8:
        raise ValueError(f"num_layers geçersiz: {num_layers}. 1 ile 8 arasında integer olmalı")
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.5:
        raise ValueError(f"dropout geçersiz: {dropout}. 0.0 ile 0.5 arasında float olmalı")
    
    logger.info(f"   ✅ Transformer konfigürasyonu doğrulandı")
    return config


def validate_enhanced_transformer_config(config: TransformerConfig) -> TransformerConfig:
    """
    Enhanced Transformer konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    d_model = model_config.get('d_model', 512)
    nhead = model_config.get('nhead', 8)
    num_layers = model_config.get('num_layers', 6)
    dropout = model_config.get('dropout_rate', 0.1)
    
    logger.info(f"🔍 Enhanced Transformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # Enhanced transformer için daha yüksek minimum değerler
    if not isinstance(d_model, int) or d_model < 128 or d_model > 1024:
        raise ValueError(f"Enhanced d_model geçersiz: {d_model}. 128 ile 1024 arasında integer olmalı")
    
    if not isinstance(nhead, int) or nhead < 4 or nhead > 16:
        raise ValueError(f"Enhanced nhead geçersiz: {nhead}. 4 ile 16 arasında integer olmalı")
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        valid_heads = [h for h in [4, 8, 16] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            model_config['nhead'] = nhead
            logger.warning(f"   🔧 Enhanced nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(f"Enhanced d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı")
    
    if not isinstance(num_layers, int) or num_layers < 2 or num_layers > 12:
        raise ValueError(f"Enhanced num_layers geçersiz: {num_layers}. 2 ile 12 arasında integer olmalı")
    
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.3:
        raise ValueError(f"Enhanced dropout geçersiz: {dropout}. 0.0 ile 0.3 arasında float olmalı")
    
    logger.info(f"   ✅ Enhanced Transformer konfigürasyonu doğrulandı")
    return config


def validate_lstm_config(config: LSTMConfig) -> LSTMConfig:
    """
    LSTM konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout_rate', 0.45)
    
    logger.info(f"🔍 LSTM konfigürasyonu doğrulanıyor...")
    logger.info(f"   hidden_size: {hidden_size}, layers: {num_layers}, dropout: {dropout}")
    
    # hidden_size validasyonu
    if not isinstance(hidden_size, int) or hidden_size < 16 or hidden_size > 512:
        raise ValueError(f"hidden_size geçersiz: {hidden_size}. 16 ile 512 arasında integer olmalı")
    
    # num_layers validasyonu  
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 4:
        raise ValueError(f"num_layers geçersiz: {num_layers}. 1 ile 4 arasında integer olmalı")
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.8:
        raise ValueError(f"dropout geçersiz: {dropout}. 0.0 ile 0.8 arasında float olmalı")
    
    # Performans uyarıları
    if hidden_size > 256:
        logger.warning(f"   ⚠️ Büyük hidden_size ({hidden_size}) overfitting riskini artırabilir")
    
    if num_layers > 3:
        logger.warning(f"   ⚠️ Çok katman ({num_layers}) gradient vanishing problemine yol açabilir")
    
    logger.info(f"   ✅ LSTM konfigürasyonu doğrulandı")
    return config


def get_model_info(model: ModelInstance) -> Dict[str, Any]:
    """
    Farklı mimarilerde birleşik model bilgisi al.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Model bilgi sözlüğü
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model tipine göre detaylar
        model_type = type(model).__name__
        
        info = {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'device': str(next(model.parameters()).device),
        }
        
        # Model-specific bilgiler
        if hasattr(model, 'hidden_size'):
            info['hidden_size'] = model.hidden_size
        if hasattr(model, 'num_layers'):
            info['num_layers'] = model.num_layers
        if hasattr(model, 'd_model'):
            info['d_model'] = model.d_model
        if hasattr(model, 'nhead'):
            info['nhead'] = model.nhead
        if hasattr(model, 'lstm_hidden'):
            info['lstm_hidden'] = model.lstm_hidden
        
        return info
        
    except Exception as e:
        logger.error(f"   ❌ Model bilgisi alınamadı: {e}")
        return {
            'model_type': type(model).__name__,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'error': str(e)
        }


def get_model_complexity_score(model: ModelInstance) -> float:
    """
    Model karmaşıklık skorunu hesapla.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Karmaşıklık skoru (million parametreler cinsinden)
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        complexity = total_params / 1e6  # Million parametreler
        
        return complexity
        
    except Exception as e:
        logger.error(f"   ❌ Karmaşıklık skoru hesaplanamadı: {e}")
        return 1.0


def test_hybrid_model_creation():
    """Test hibrit model oluşturma"""
    if not HYBRID_MODEL_AVAILABLE:
        logger.warning("⚠️ Hibrit model mevcut değil, test atlanıyor")
        return False
        
    try:
        # Test konfigürasyonu
        test_config = {
            'model': {
                'target_mode': 'three_class',
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'ff_dim': 256,
                'dropout_rate': 0.1
            },
            'data': {
                'sequence_length': 50
            }
        }
        device = torch.device('cpu')
        n_features = 20
        
        logger.info(f"🧪 Hibrit model testi başlıyor...")
        logger.info(f"   Config: {test_config}")
        
        model = create_enhanced_transformer(test_config, n_features, device)
        
        # Test input
        batch_size, seq_len = 8, 50
        test_input = torch.randn(batch_size, seq_len, n_features)
        
        logger.info(f"   Test input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            
        expected_classes = 3 if test_config['model'].get('target_mode') == 'three_class' else 1
        assert output.shape == (batch_size, expected_classes), f"Wrong output shape: {output.shape}"
        
        logger.info(f"✅ Hibrit model testi başarılı!")
        logger.info(f"   Input: {test_input.shape} → Output: {output.shape}")
        logger.info(f"   Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hibrit model testi başarısız: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


__all__ = [
    'EnhancedTransformer',
    'create_enhanced_transformer',
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock',
    'validate_transformer_config',
    'validate_enhanced_transformer_config',
    'validate_lstm_config',
    'get_model_info',
    'get_model_complexity_score',
    'test_hybrid_model_creation'
]
