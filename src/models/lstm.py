"""LSTM model implementation for forex prediction."""

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
                 output_size: int = 1, dropout: float = 0.45, use_layer_norm: bool = True, 
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
            
        self.fc2 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size // 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output predictions
        """
        # First LSTM layer
        out1, _ = self.lstm1(x)
        if self.use_layer_norm:
            out1 = self.layer_norm1(out1)
        out1 = self.dropout(out1)
        
        # Second LSTM layer
        out2, _ = self.lstm2(out1)
        if self.use_layer_norm:
            out2 = self.layer_norm2(out2)
        
        # Take last timestep
        last_output = out2[:, -1, :]
        
        # Dense layers
        out = self.fc1(last_output)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        # Apply appropriate activation
        if self.target_mode == 'binary':
            return torch.sigmoid(out)
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
        device: Device to create model on
        
    Returns:
        Initialized LSTM model
    """
    model_config = config.get('model', {})
    
    model = PairSpecificLSTM(
        input_size=input_size,
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout_rate', 0.45),
        use_layer_norm=model_config.get('use_layer_norm', True),
        target_mode=model_config.get('target_mode', 'binary')
    )
    
    return model.to(device)

__all__ = ['PairSpecificLSTM', 'create_model']
