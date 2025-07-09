# src/config/config_manager.py - ConfigManager implementasyonu

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Enhanced configuration manager for forex prediction system"""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize config manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._default_config = None
    
    def load_config(self, config_path: str) -> 'EnhancedConfig':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.load_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            logger.info(f"✅ Config loaded from {config_path}")
            return EnhancedConfig(config_dict)
            
        except Exception as e:
            logger.error(f"❌ Failed to load config from {config_path}: {str(e)}")
            return self.load_default_config()
    
    def load_default_config(self) -> 'EnhancedConfig':
        """Load default configuration"""
        if self._default_config is None:
            self._default_config = self._create_default_config()
        
        return EnhancedConfig(self._default_config)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration dictionary"""
        return {
            'data': {
                'sequence_length': 64,
                'target_horizon': 64,
                'timeframe': 'M15',
                'max_candles': 50000,
                'features': {
                    'basic': ['open', 'high', 'low', 'close', 'volume'],
                    'technical': ['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'ema_12', 'ema_26'],
                    'multi_timeframe': True
                },
                'normalization': {
                    'method': 'robust',
                    'feature_range': (-1, 1)
                }
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 5,
                'min_delta': 1e-4,
                'gradient_clip': 1.0,
                'scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'factor': 0.5,
                    'patience': 3
                }
            },
            'models': {
                'lstm': {
                    'hidden_size': 96,
                    'num_layers': 2,
                    'dropout': 0.15,
                    'bidirectional': True,
                    'batch_first': True
                },
                'enhanced_transformer': {
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 4,
                    'dropout': 0.15,
                    'dim_feedforward': 2048,
                    'activation': 'relu'
                },
                'hybrid_lstm_transformer': {
                    'lstm_hidden': 96,
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 4,
                    'dropout': 0.15,
                    'fusion_dropout': 0.1
                }
            },
            'ensemble': {
                'voting_strategy': 'confidence_weighted',
                'diversity_factor': 0.2,
                'cv_folds': 5,
                'min_models': 2,
                'max_models': 5
            },
            'optimization': {
                'loss': {
                    'type': 'CrossEntropyLoss',
                    'focal_alpha': 0.25,
                    'focal_gamma': 2.0
                },
                'optimizer': {
                    'type': 'Adam',
                    'lr': 0.001,
                    'weight_decay': 1e-4,
                    'betas': [0.9, 0.999]
                },
                'scheduler': {
                    'type': 'OneCycleLR',
                    'max_lr': 0.01,
                    'pct_start': 0.25,
                    'div_factor': 10.0,
                    'final_div_factor': 1000.0
                }
            },
            'validation': {
                'split_ratio': 0.8,
                'method': 'time_series',
                'cv_folds': 5,
                'purge_gap': 10
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_dir': 'logs'
            }
        }
    
    def save_config(self, config: 'EnhancedConfig', save_path: str):
        """Save configuration to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Config saved to {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save config to {save_path}: {str(e)}")
            raise


class EnhancedConfig:
    """Enhanced configuration class with utility methods"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize enhanced config"""
        self.config = config_dict
    
    def get_model_config(self, model_type: str, target_mode: str = 'binary') -> Dict[str, Any]:
        """Get model configuration for specific model type"""
        
        # Base model configs
        model_configs = self.config.get('models', {})
        
        if model_type not in model_configs:
            logger.warning(f"Model type {model_type} not found in config, using defaults")
            model_configs[model_type] = self._get_default_model_config(model_type)
        
        # Get model-specific config
        model_config = model_configs[model_type].copy()
        
        # Add training parameters
        training_config = self.config.get('training', {})
        model_config.update({
            'epochs': training_config.get('epochs', 10),
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.001),
            'weight_decay': training_config.get('weight_decay', 1e-4)
        })
        
        # Add target mode configuration
        if target_mode == 'binary':
            model_config['num_classes'] = 2
        elif target_mode == 'multiclass':
            model_config['num_classes'] = 3
        else:
            model_config['num_classes'] = 2  # Default to binary
        
        # Add data configuration
        data_config = self.config.get('data', {})
        model_config.update({
            'sequence_length': data_config.get('sequence_length', 64),
            'target_horizon': data_config.get('target_horizon', 64)
        })
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        training_config = self.config.get('training', {})
        
        # Add default values if missing
        defaults = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 5,
            'min_delta': 1e-4,
            'gradient_clip': 1.0
        }
        
        for key, value in defaults.items():
            if key not in training_config:
                training_config[key] = value
        
        return training_config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_ensemble_config(self) -> Dict[str, Any]:
        """Get ensemble configuration"""
        return self.config.get('ensemble', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.config.get('optimization', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return self.config.get('validation', {})
    
    def _get_default_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type"""
        
        default_configs = {
            'lstm': {
                'hidden_size': 96,
                'num_layers': 2,
                'dropout': 0.15,
                'bidirectional': True,
                'batch_first': True
            },
            'enhanced_transformer': {
                'd_model': 512,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.15,
                'dim_feedforward': 2048,
                'activation': 'relu'
            },
            'hybrid_lstm_transformer': {
                'lstm_hidden': 96,
                'd_model': 512,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.15,
                'fusion_dropout': 0.1
            },
            'transformer': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 3,
                'dropout': 0.1,
                'dim_feedforward': 1024
            }
        }
        
        return default_configs.get(model_type, {
            'hidden_size': 64,
            'dropout': 0.1
        })
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator"""
        return key in self.config


# Utility functions for config management
def create_default_config_file(config_path: str = "configs/default.yaml"):
    """Create default configuration file if it doesn't exist"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        config_manager = ConfigManager()
        default_config = config_manager.load_default_config()
        config_manager.save_config(default_config, config_path)
        logger.info(f"✅ Default config file created at {config_path}")
    else:
        logger.info(f"Config file already exists at {config_path}")


def validate_config(config: EnhancedConfig) -> bool:
    """Validate configuration for common issues"""
    
    issues = []
    
    # Check required sections
    required_sections = ['data', 'training', 'models']
    for section in required_sections:
        if section not in config.config:
            issues.append(f"Missing required section: {section}")
    
    # Check data configuration
    data_config = config.get_data_config()
    if 'sequence_length' not in data_config:
        issues.append("Missing sequence_length in data config")
    
    # Check training configuration
    training_config = config.get_training_config()
    if training_config.get('epochs', 0) <= 0:
        issues.append("Invalid epochs value in training config")
    
    if training_config.get('batch_size', 0) <= 0:
        issues.append("Invalid batch_size value in training config")
    
    # Check model configurations
    models_config = config.config.get('models', {})
    if not models_config:
        issues.append("No model configurations found")
    
    # Log issues
    if issues:
        logger.error("❌ Config validation failed:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    else:
        logger.info("✅ Config validation passed")
        return True


# Example usage and testing
if __name__ == "__main__":
    # Test config manager
    config_manager = ConfigManager()
    
    # Create default config
    config = config_manager.load_default_config()
    
    # Test model config retrieval
    lstm_config = config.get_model_config('lstm', 'binary')
    print("LSTM Config:", lstm_config)
    
    # Test config validation
    is_valid = validate_config(config)
    print("Config valid:", is_valid)
    
    # Create default config file
    create_default_config_file()