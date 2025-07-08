"""LSTM model implementation for forex prediction with batch normalization fix."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PairSpecificLSTM(nn.Module):
    """Pair-specific LSTM model for forex prediction.
    
    This model is designed to handle specific currency pairs with
    customizable architecture and regularization techniques.
    
    Args:
        input_size: Number of input features
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        output_size: Number of output classes/values
        dropout: Dropout rate for regularization
        use_layer_norm: Whether to use layer normalization
        target_mode: Type of prediction ('binary' or 'three_class')
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 2, dropout: float = 0.45, use_layer_norm: bool = True, 
                 target_mode: str = 'binary'):
        super(PairSpecificLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.target_mode = target_mode
        
        # First LSTM layer with dropout
        self.lstm1 = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        
        if use_layer_norm:
            self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Dense layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        
        # Output layer depends on target mode
        if target_mode == 'three_class':
            output_size = 3
        elif target_mode == 'binary':
            output_size = 2
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
            
        self.fc2 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        
        # 🔧 FIX: GroupNorm yerine BatchNorm1d - batch_size=1 için güvenli
        # GroupNorm batch size'dan bağımsızdır
        self.group_norm = nn.GroupNorm(
            num_groups=min(8, hidden_size // 4),  # 8 grup veya feature boyutuna göre
            num_channels=hidden_size // 4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size) 
               or (batch_size, input_size) - will add sequence dimension if missing
            
        Returns:
            Output predictions
        """
        # DÜZELTME: Input tensor boyutunu kontrol et ve gerekirse sequence dimension ekle
        if x.dim() == 2:
            # [batch_size, features] -> [batch_size, 1, features]
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            # [features] -> [1, 1, features] 
            x = x.unsqueeze(0).unsqueeze(0)
        
        # Şimdi x guaranteed [batch_size, seq_len, features] formatında
        
        # First LSTM layer
        out1, _ = self.lstm1(x)
        if self.use_layer_norm:
            out1 = self.layer_norm1(out1)
        out1 = self.dropout(out1)
        
        # Second LSTM layer
        out2, _ = self.lstm2(out1)
        if self.use_layer_norm:
            out2 = self.layer_norm2(out2)
        
        # Take last timestep - şimdi güvenli çünkü sequence dimension garantili
        if out2.dim() == 3:
            last_output = out2[:, -1, :]  # [batch_size, hidden_size//2]
        else:
            # Fallback: eğer hala 2D ise, olduğu gibi kullan
            last_output = out2
        
        # Dense layers
        out = self.fc1(last_output)
        
        # 🔧 FIX: GroupNorm batch_size=1 için güvenli
        out = self.group_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        # Apply appropriate activation
        if self.target_mode == 'binary':
            if self.training:
                # Eğitimde: 2 boyutlu softmax (logit)
                return out  # [batch_size, 2] - CrossEntropyLoss için raw logits
            else:
                # Tahminde: yalnızca pozitif sınıfın olasılığı [batch_size, 1]
                probs = torch.softmax(out, dim=1)
                return probs[:, 1:2]
        else:  # three_class
            return torch.softmax(out, dim=1)
    
    def get_model_info(self) -> dict:
        """Get model information including parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'target_mode': self.target_mode,
            'use_layer_norm': self.use_layer_norm
        }

def create_model(config: dict, input_size: int, device: torch.device) -> PairSpecificLSTM:
    """Factory function to create LSTM model from configuration.
    
    Args:
        config: Model configuration dictionary
        input_size: Number of input features
        device: PyTorch device for model
        
    Returns:
        LSTM model instance
    """
    model_config = config.get('model', {})
    
    # Extract configuration parameters
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout_rate', 0.45)
    use_layer_norm = model_config.get('use_layer_norm', True)
    target_mode = model_config.get('target_mode', 'binary')
    
    # Create model
    model = PairSpecificLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
        target_mode=target_mode
    )
    
    return model.to(device)


# Backward compatibility functions
def create_lstm_model(config: dict, input_size: int, device: torch.device) -> PairSpecificLSTM:
    """Backward compatibility wrapper for create_model."""
    return create_model(config, input_size, device)


def get_model_info(model: PairSpecificLSTM) -> dict:
    """Get comprehensive model information."""
    if hasattr(model, 'get_model_info'):
        return model.get_model_info()
    else:
        # Fallback for models without get_model_info method
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': 'PairSpecificLSTM'
        }


def validate_lstm_config(config: dict) -> dict:
    """Validate LSTM configuration parameters."""
    import copy
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    # Set reasonable defaults if not provided
    if 'hidden_size' not in model_config:
        model_config['hidden_size'] = 64
    if 'num_layers' not in model_config:
        model_config['num_layers'] = 2
    if 'dropout_rate' not in model_config:
        model_config['dropout_rate'] = 0.45
    if 'use_layer_norm' not in model_config:
        model_config['use_layer_norm'] = True
    if 'target_mode' not in model_config:
        model_config['target_mode'] = 'binary'
    
    # Validate ranges
    hidden_size = model_config['hidden_size']
    if not isinstance(hidden_size, int) or hidden_size < 16 or hidden_size > 512:
        raise ValueError(f"hidden_size must be between 16 and 512, got {hidden_size}")
    
    num_layers = model_config['num_layers']
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 4:
        raise ValueError(f"num_layers must be between 1 and 4, got {num_layers}")
    
    dropout = model_config['dropout_rate']
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.8:
        raise ValueError(f"dropout_rate must be between 0.0 and 0.8, got {dropout}")
    
    return config


# Export list for the module
__all__ = [
    'PairSpecificLSTM',
    'create_model', 
    'create_lstm_model',
    'get_model_info',
    'validate_lstm_config'
]