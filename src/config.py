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
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return {}

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
            'name': 'Modular LSTM System',
            'version': '1.0.0'
        },
        'data': {
            'granularity': 'M15',
            'lookback_candles': 10000,
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
            'epochs': 80,
            'batch_size': 64,
            'learning_rate': 0.001,
            'patience': 5
        },
        'api': {
            'api_key': '8d8619f4119fec7e59d73c61b76b480d-d0947fd967a22401c1e48bc1516ad0eb',
            'account_id': '101-004-35700665-002',
            'environment': 'practice'
        }
    }

__all__ = ['load', 'save', 'get_default_config', 'DEFAULT_CONFIG_PATH']
