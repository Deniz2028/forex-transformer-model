"""
Enhanced Transformer Model for Financial Time Series Classification

Bu modül, forex fiyat hareketi tahmini için özelleştirilmiş gelişmiş Transformer mimarisi içerir.
PDF'deki tüm optimizasyonları ve en iyi uygulamaları kapsar.

Features:
- Multi-head attention with learnable positional encoding
- Pre-norm architecture for stable training
- Dynamic focal loss support
- Gradient accumulation and clipping
- OneCycleLR scheduler optimization
- Comprehensive model complexity analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import math
import time
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from tqdm import tqdm
from pathlib import Path

# Logging yapılandırması
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Enhanced Transformer.
    
    PDF önerisi: Sinusoidal yerine learnable positional encoding kullan.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Learnable positional embeddings (PDF önerisi)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            Positionally encoded tensor
        """
        batch_size, seq_len, d_model = x.shape
        
        # Positional encoding'i sequence length'e göre kes
        pos_enc = self.pos_embedding[:, :seq_len, :]
        
        # Add positional encoding
        x = x + pos_enc
        
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Enhanced Multi-Head Attention with optimizations.
    
    PDF optimizasyonları:
    - Attention dropout
    - Scaled initialization
    - Optional attention visualization
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Scaled initialization (PDF önerisi)
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization with proper scaling."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-head attention forward pass.
        
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            (output, attention_weights) tuple
        """
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask=mask
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention mechanism."""
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Enhanced Transformer block with pre-norm architecture.
    
    PDF önerisi: Pre-norm (LayerNorm before attention/FFN) kullan.
    """
    
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),  # GELU activation (PDF önerisi)
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Transformer block forward pass with pre-norm architecture.
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            (output, attention_weights) tuple
        """
        # Pre-norm + Multi-head attention + residual
        norm_x = self.norm1(x)
        attn_output, attn_weights = self.attention(
            norm_x, norm_x, norm_x, mask=mask, return_attention=return_attention
        )
        x = x + attn_output
        
        # Pre-norm + Feed-forward + residual
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output
        
        return x, attn_weights


class EnhancedTransformer(nn.Module):
    """
    Enhanced Transformer model for financial time series classification.
    
    PDF'deki tüm optimizasyonları içerir:
    - Learnable positional encoding
    - Pre-norm architecture
    - Multi-head attention with proper scaling
    - GELU activation
    - Dynamic output head based on target mode
    - Comprehensive model info tracking
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.15,
        max_seq_len: int = 96,
        target_mode: str = 'binary'
    ):
        super(EnhancedTransformer, self).__init__()
        
        # Store configuration
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.target_mode = target_mode
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Output classifier
        if target_mode == 'three_class':
            output_size = 3
        else:
            output_size = 1
        
        # Enhanced classifier head (PDF önerisi: deeper head)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"✅ Enhanced Transformer initialized:")
        logger.info(f"   📐 Architecture: {d_model}d × {num_layers}L × {nhead}H")
        logger.info(f"   📊 Parameters: {self.count_parameters():,}")
        logger.info(f"   🎯 Target mode: {target_mode}")
    
    def _init_weights(self):
        """Initialize model weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Enhanced Transformer forward pass.
        
        Args:
            x: [batch_size, seq_len, input_size] or [batch_size, input_size]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Classification output or (output, attention_weights) if return_attention=True
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # [batch_size, input_size] -> [batch_size, 1, input_size]
            x = x.unsqueeze(1)
        
        batch_size, seq_len, input_size = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        attention_weights = [] if return_attention else None
        
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask=mask, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                attention_weights.append(attn_weights)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Global pooling (take mean across sequence dimension)
        if seq_len > 1:
            pooled = x.mean(dim=1)  # [batch_size, d_model]
        else:
            pooled = x.squeeze(1)   # [batch_size, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [batch_size, output_size]
        
        # Apply activation based on target mode
        if self.target_mode == 'binary':
            output = torch.sigmoid(logits)
        else:
            output = logits  # Raw logits for CrossEntropyLoss
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
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
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weights from each layer
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
            return attention_weights if attention_weights else []


# Factory function
def create_enhanced_transformer(config: Dict[str, Any], input_size: int, device: torch.device) -> EnhancedTransformer:
    """
    Factory function to create Enhanced Transformer from configuration.
    
    Args:
        config: Model configuration dictionary
        input_size: Number of input features
        device: PyTorch device
        
    Returns:
        Configured Enhanced Transformer model
    """
    # Get transformer config
    transformer_config = config.get('transformer', {})
    model_config = config.get('model', {})
    
    # Create model with config parameters
    model = EnhancedTransformer(
        input_size=input_size,
        d_model=transformer_config.get('d_model', 512),
        nhead=transformer_config.get('nhead', 8),
        num_layers=transformer_config.get('num_layers', 4),
        ff_dim=transformer_config.get('ff_dim', 1024),
        dropout=transformer_config.get('dropout_rate', 0.15),
        max_seq_len=transformer_config.get('max_seq_len', 96),
        target_mode=model_config.get('target_mode', 'binary')
    )
    
    return model.to(device)


# Training function
def train_enhanced_transformer(model: EnhancedTransformer, X: np.ndarray, y: np.ndarray, 
                              config: Dict[str, Any], device: torch.device) -> Tuple[EnhancedTransformer, Dict]:
    """
    Enhanced Transformer training function with PDF optimizations.
    
    Args:
        model: Model to train
        X: Feature matrix [samples, seq_len, features] or [samples, features]
        y: Target variable [samples]
        config: Training configuration
        device: PyTorch device
        
    Returns:
        (trained_model, training_history) tuple
    """
    # Training parameters
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 30)
    learning_rate = training_config.get('learning_rate', 2e-4)
    batch_size = training_config.get('batch_size', 16)
    patience = training_config.get('patience', 10)
    accumulation_steps = training_config.get('accumulation_steps', 4)
    
    logger.info(f"🚀 Enhanced Transformer training started...")
    logger.info(f"   📊 Data shape: {X.shape}, Target: {y.shape}")
    logger.info(f"   ⚙️ Config: epochs={epochs}, lr={learning_rate}, batch={batch_size}")
    
    # Move model to device
    model = model.to(device)
    
    # Prepare data
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], 1, X.shape[1])
        logger.info(f"   📐 Reshaped data: {X.shape}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    # Data split (PDF: 70/15/15)
    total_size = len(X_tensor)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_val = X_tensor[train_size:train_size + val_size]
    y_val = y_tensor[train_size:train_size + val_size]
    
    logger.info(f"   📊 Split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Create data loaders
    train_dataset = data_utils.TensorDataset(X_train, y_train)
    val_dataset = data_utils.TensorDataset(X_val, y_val)
    
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data_utils.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Loss function
    if model.target_mode == 'binary':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer (PDF: AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler (PDF: OneCycleLR)
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Loss calculation
            if model.target_mode == 'binary':
                loss = criterion(output.squeeze(), target) / accumulation_steps
                pred = (output.squeeze() > 0.5).float()
            else:
                loss = criterion(output, target.long()) / accumulation_steps
                pred = output.argmax(dim=1).float()
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Statistics
            train_loss += loss.item() * accumulation_steps
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Acc': f'{100. * train_correct / train_total:.1f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if model.target_mode == 'binary':
                    loss = criterion(output.squeeze(), target)
                    pred = (output.squeeze() > 0.5).float()
                else:
                    loss = criterion(output, target.long())
                    pred = output.argmax(dim=1).float()
                
                val_loss += loss.item()
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.1f}%'
                })
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                f"LR: {current_lr:.2e}"
            )
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏰ Early stopping at epoch {epoch+1}")
            break
    
    # Final results
    logger.info(f"✅ Training completed!")
    logger.info(f"   📊 Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"   📈 Final Acc: Train={train_acc:.2f}%, Val={val_acc:.2f}%")
    
    return model, history


# Export functions
__all__ = [
    'EnhancedTransformer',
    'PositionalEncoding',
    'MultiHeadAttention', 
    'TransformerBlock',
    'create_enhanced_transformer',
    'train_enhanced_transformer'
]
