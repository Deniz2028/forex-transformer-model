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

# Hibrit model için default konfigürasyon
DEFAULT_HYBRID_CONFIG = {
    'lstm_hidden': 96,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 4,
    'fusion_strategy': 'concat'  # 'concat', 'attention', 'gated'
}

# Model-specific konfigürasyonlar güncelleme
def get_model_config(model_type: str, target_mode: str = 'three_class') -> Dict[str, Any]:
    """Model tipine göre konfigürasyon döndür"""
    
    base_config = {
        'target_mode': target_mode,
        'model': {
            'dropout_rate': 0.1,
        },
        'training': {
            'learning_rate': 1e-3,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'scheduler': 'ReduceLROnPlateau',
            'optimizer': 'AdamW',
            'patience': 7,
            'gradient_clip': 0.5
        }
    }
    
    if model_type == 'lstm' or model_type == 'pairspecificlstm':
        base_config.update({
            'model_type': 'lstm',
            'model': {
                **base_config['model'],
                'hidden_size': 64,
                'num_layers': 2,
                'dropout_rate': 0.45
            }
        })
        
    elif model_type == 'transformer':
        base_config.update({
            'model_type': 'transformer',
            'model': {
                **base_config['model'],
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'dropout_rate': 0.1
            },
            'training': {
                **base_config['training'],
                'learning_rate': 5e-4,
                'scheduler': 'OneCycleLR',
                'batch_size': 32
            }
        })
        
    elif model_type == 'enhanced_transformer':
        base_config.update({
            'model_type': 'enhanced_transformer',
            'model': {
                **base_config['model'],
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'dropout_rate': 0.1
            },
            'training': {
                **base_config['training'],
                'learning_rate': 3e-4,
                'scheduler': 'OneCycleLR',
                'batch_size': 32
            }
        })
        
    elif model_type == 'hybrid_lstm_transformer' or model_type == 'hybrid':
        base_config.update({
            'model_type': 'hybrid_lstm_transformer',
            'hybrid': DEFAULT_HYBRID_CONFIG,
            'model': {
                **base_config['model'],
                'dropout_rate': 0.1
            },
            'training': {
                **base_config['training'],
                'learning_rate': 5e-4,
                'scheduler': 'OneCycleLR',
                'batch_size': 64,
                'weight_decay': 1e-4,
                'gradient_clip': 1.0,
                'warmup_steps': 500
            }
        })
        
    return base_config

# CLI argümanları için hibrit model seçenekleri
def add_hybrid_model_args(parser):
    """CLI'ye hibrit model argümanları ekle"""
    parser.add_argument('--fusion_strategy', 
                       choices=['concat', 'attention', 'gated'],
                       default='concat',
                       help='Hibrit model fusion strategy')
    
    parser.add_argument('--lstm_hidden', type=int, default=96,
                       help='LSTM hidden size for hybrid model')
    
    parser.add_argument('--transformer_d_model', type=int, default=512,
                       help='Transformer d_model for hybrid model')
    
    parser.add_argument('--transformer_nhead', type=int, default=8,
                       help='Number of attention heads for hybrid model')
    
    parser.add_argument('--transformer_layers', type=int, default=4,
                       help='Number of transformer layers for hybrid model')

# Test komutları
TEST_CONFIGS = {
    'hybrid_quick_test': {
        'model_type': 'hybrid_lstm_transformer',
        'target_mode': 'three_class',
        'hybrid': {
            'lstm_hidden': 64,
            'd_model': 256,
            'nhead': 4,
            'num_layers': 2,
            'fusion_strategy': 'concat'
        },
        'training': {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'weight_decay': 1e-4
        }
    },
    
    'hybrid_production': {
        'model_type': 'hybrid_lstm_transformer',
        'target_mode': 'three_class',
        'hybrid': {
            'lstm_hidden': 128,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 4,
            'fusion_strategy': 'attention'
        },
        'training': {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'scheduler': 'OneCycleLR',
            'gradient_clip': 1.0
        }
    }
}

# CLI'den kullanım örnekleri
EXAMPLE_COMMANDS = {
    'hybrid_train': [
        "# Basit hibrit model eğitimi",
        "python -m src.cli train --model_type hybrid_lstm_transformer \\",
        "  --granularity M5 --epochs 30 --cache",
        "",
        "# Gelişmiş hibrit model (attention fusion)",
        "python -m src.cli train --model_type hybrid \\", 
        "  --fusion_strategy attention --lstm_hidden 128 \\",
        "  --transformer_d_model 512 --epochs 50",
        "",
        "# Hibrit model ile Optuna optimization",
        "python -m src.cli tune --model_type hybrid_lstm_transformer \\",
        "  --optuna_runs 50 --epochs 30"
    ]
}

__all__ = [
    'load', 
    'save', 
    'get_default_config', 
    'DEFAULT_CONFIG_PATH',
    'get_model_config', 
    'add_hybrid_model_args', 
    'DEFAULT_HYBRID_CONFIG',
    'TEST_CONFIGS', 
    'EXAMPLE_COMMANDS'
]
