# src/models/pdf_hybrid.py
"""
PDF Hybrid LSTM-Transformer Model - Exact Implementation from Page 16

PDF'deki exact architecture implementasyonu:
- Bidirectional LSTM
- Transformer Encoder  
- MLP classifier
- OneCycleLR scheduler support
- 70-75% accuracy target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PDFHybridForexModel(nn.Module):
    """
    PDF Hybrid LSTM-Transformer Model
    
    Exact implementation from PDF page 16:
    - Bidirectional LSTM (input_dim -> d_model//2 * 2)
    - Transformer Encoder (d_model, nhead, num_layers)  
    - MLP classifier (d_model -> 256 -> 64 -> output)
    
    Target: 70-75% validation accuracy
    """
    
    def __init__(
        self,
        input_dim: int = 82,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.15,
        target_mode: str = 'binary'
    ):
        super(PDFHybridForexModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.target_mode = target_mode
        
        # PDF Architecture: Bidirectional LSTM
        # input_dim -> d_model//2, bidirectional=True -> d_model output
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout in LSTM layer
        )
        
        # PDF Architecture: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,  # PDF specification
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # PDF Architecture: MLP Classifier
        if target_mode == 'three_class':
            output_size = 3
        else:  # binary
            output_size = 2  # For CrossEntropy compatibility
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ PDF Hybrid Model initialized:")
        logger.info(f"   📐 Architecture: LSTM({input_dim}->{d_model}) + Transformer({d_model}x{num_layers}) + MLP")
        logger.info(f"   🎯 Target mode: {target_mode} ({output_size} outputs)")
        logger.info(f"   📊 Parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """Initialize weights using PDF recommendations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PDF Hybrid forward pass.
        
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            [batch_size, output_size] classification output
        """
        batch_size, seq_len, input_dim = x.shape
        
        # PDF Step 1: Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch_size, seq_len, d_model]
        
        # PDF Step 2: Transformer Encoder
        # Input already in batch_first format
        transformer_out = self.transformer(lstm_out)  # [batch_size, seq_len, d_model]
        
        # PDF Step 3: Global pooling
        # Take mean across sequence dimension
        pooled = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # PDF Step 4: MLP Classification
        logits = self.mlp(pooled)  # [batch_size, output_size]
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = self.count_parameters()
        
        # Calculate component parameters
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        
        return {
            'model_type': 'PDFHybridForexModel',
            'total_parameters': total_params,
            'lstm_parameters': lstm_params,
            'transformer_parameters': transformer_params,
            'mlp_parameters': mlp_params,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'input_dim': self.input_dim,
            'target_mode': self.target_mode,
            'device': str(next(self.parameters()).device)
        }


def create_pdf_hybrid_model(config: Dict[str, Any], n_features: int, device: torch.device) -> PDFHybridForexModel:
    """
    Factory function to create PDF Hybrid model.
    
    Args:
        config: Configuration dictionary
        n_features: Number of input features
        device: Target device
        
    Returns:
        Initialized PDF Hybrid model
    """
    model_config = config.get('model', {})
    transformer_config = config.get('transformer', {})
    
    # PDF optimal parameters (from page 12)
    d_model = transformer_config.get('d_model', 512)
    nhead = transformer_config.get('nhead', 8)
    num_layers = transformer_config.get('num_layers', 4)
    dropout = model_config.get('dropout_rate', 0.15)
    target_mode = model_config.get('target_mode', 'binary')
    
    logger.info(f"🏭 Creating PDF Hybrid model...")
    logger.info(f"   📐 d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    logger.info(f"   🎯 Target mode: {target_mode}")
    logger.info(f"   📊 Input features: {n_features}")
    
    # Validate parameters
    if d_model % nhead != 0:
        logger.warning(f"d_model ({d_model}) not divisible by nhead ({nhead})")
        # Auto-fix
        valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0]
        if valid_heads:
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            logger.warning(f"Auto-fixed nhead to: {nhead}")
    
    model = PDFHybridForexModel(
        input_dim=n_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        target_mode=target_mode
    )
    
    model = model.to(device)
    logger.info(f"✅ PDF Hybrid model created successfully")
    
    return model


def get_pdf_training_params(model: PDFHybridForexModel) -> Dict[str, Any]:
    """
    Get PDF-optimized training parameters.
    
    Returns training parameters optimized for PDF Hybrid model
    based on page 18-19 recommendations.
    """
    complexity = model.count_parameters() / 1e6  # Million parameters
    
    # PDF recommendations based on model size
    if complexity < 5.0:  # Small model
        base_lr = 1e-3
        batch_size = 24
        warmup_steps = 1000
    elif complexity < 15.0:  # Medium model  
        base_lr = 5e-4
        batch_size = 16
        warmup_steps = 2000
    else:  # Large model
        base_lr = 2e-4
        batch_size = 8
        warmup_steps = 3000
    
    return {
        'learning_rate': base_lr,
        'batch_size': batch_size,
        'scheduler': 'OneCycleLR',
        'max_lr': base_lr,
        'pct_start': 0.3,          # PDF recommendation
        'div_factor': 25.0,        # PDF recommendation
        'final_div_factor': 10000.0,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,      # PDF recommendation
        'optimizer': 'AdamW',
        'warmup_steps': warmup_steps,
        'epochs': 50,              # PDF recommendation
        'patience': 10,
        'model_complexity': complexity
    }


# Test function
def test_pdf_hybrid_model():
    """Test PDF Hybrid model creation and forward pass."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test configuration
        test_config = {
            'model': {
                'target_mode': 'binary',
                'dropout_rate': 0.15
            },
            'transformer': {
                'd_model': 512,
                'nhead': 8,
                'num_layers': 4
            }
        }
        
        # Create model
        model = create_pdf_hybrid_model(test_config, n_features=82, device=device)
        
        # Test forward pass
        batch_size, seq_len = 16, 64
        test_input = torch.randn(batch_size, seq_len, 82, device=device)
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_output_size = 2 if test_config['model']['target_mode'] == 'binary' else 3
        assert output.shape == (batch_size, expected_output_size), f"Wrong output shape: {output.shape}"
        
        # Get model info
        info = model.get_model_info()
        logger.info(f"✅ PDF Hybrid model test passed!")
        logger.info(f"   Input: {test_input.shape} → Output: {output.shape}")
        logger.info(f"   Parameters: {info['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF Hybrid model test failed: {e}")
        return False


__all__ = [
    'PDFHybridForexModel',
    'create_pdf_hybrid_model', 
    'get_pdf_training_params',
    'test_pdf_hybrid_model'
]