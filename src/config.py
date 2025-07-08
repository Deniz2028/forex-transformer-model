# src/config.py dosyasını oluşturun veya düzeltin

"""Configuration management module."""

from pathlib import Path, PurePath
import yaml
import os
from typing import Dict, Any, Union

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / 'configs' / 'default.yaml'

def load(path: Union[PurePath, str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses default config.
        
    Returns:
        Dictionary containing configuration.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    
    path = Path(path)
    
    if not path.exists():
        print(f"⚠️ Config file not found: {path}")
        return get_default_config()  # Return default instead of empty dict
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or get_default_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return get_default_config()

def save(obj: Dict[str, Any], path: Union[PurePath, str]) -> bool:
    """Save configuration to YAML file.
    
    Args:
        obj: Dictionary to save.
        path: Path to save file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(obj, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        'system': {
            'name': 'Enhanced LSTM/Transformer System',
            'version': '2.0.0'
        },
        'data': {
            'granularity': 'M15',
            'lookback_candles': 50000,
            'sequence_length': 64
        },
        'model': {
            'target_mode': 'binary',
            'use_focal_loss': True,
            'dropout_rate': 0.45,
            'hidden_size': 64,
            'num_layers': 2
        },
        'training': {
            'epochs': 30,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 5,
            'use_smote': False
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'ff_dim': 256,
            'max_seq_len': 64
        },
        'loss': {
            'type': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'use_dynamic_focal': False,
            'use_focal_loss': True
        },
        'scheduler': {
            'type': 'OneCycleLR',
            'use_onecycle': True,
            'max_lr': 0.01,
            'pct_start': 0.25,
            'div_factor': 10.0,
            'final_div_factor': 1000.0
        },
        'optimization': {
            'optimizer_type': 'adam',
            'weight_decay': 1e-4,
            'gradient_accumulation_steps': 1,
            'use_mixed_precision': False,
            'gradient_clip_norm': 1.0
        },
        'forex': {
            'market_regime_adaptive': False,
            'volatility_window': 20,
            'class_balancing_strategy': 'auto'
        },
        'api': {
            'api_key': '8d8619f4119fec7e59d73c61b76b480d-d0947fd967a22401c1e48bc1516ad0eb',
            'account_id': '101-004-35700665-002',
            'environment': 'practice'
        }
    }

def get_model_config(model_type: str, target_mode: str = 'binary') -> Dict[str, Any]:
    """Model tipine göre konfigürasyon döndür"""
    
    base_config = get_default_config().copy()
    base_config['model']['target_mode'] = target_mode
    
    if model_type == 'enhanced_transformer':
        base_config['model'].update({
            'dropout_rate': 0.1,
        })
        base_config['training'].update({
            'learning_rate': 5e-4,
            'batch_size': 16,
            'scheduler': 'OneCycleLR'
        })
        base_config['transformer'].update({
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'ff_dim': 512
        })
    elif model_type == 'lstm':
        base_config['model'].update({
            'dropout_rate': 0.45,
            'hidden_size': 512,
            'num_layers': 4
        })
        base_config['training'].update({
            'learning_rate': 1e-3,
            'batch_size': 32
        })
    
    return base_config

# Backward compatibility
DEFAULT_HYBRID_CONFIG = {
    'lstm_hidden': 96,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 4,
    'fusion_strategy': 'concat'
}

print("🔧 Config module initialized with enhanced parameters support!")
