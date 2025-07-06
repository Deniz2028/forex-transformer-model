# ========================  
# src/ensemble/ensemble_trainer.py
"""Train and validate ensemble models."""

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import logging
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Train and validate ensemble models"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None):
        self.device = device
        self.config = config or {}
        self.trained_models = []
        self.model_histories = []
        
        logger.info(f"🏋️ EnsembleTrainer initialized on {device}")
        
    def train_diverse_models(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           model_configs: List[Dict], epochs: int = 50) -> List[nn.Module]:
        """Train diverse models for ensemble"""
        from ..models.factory import create_model
        from ..training.unified_trainer import UnifiedTrainer
        
        models = []
        
        for i, config in enumerate(model_configs):
            logger.info(f"🎭 Training ensemble model {i+1}/{len(model_configs)}")
            logger.info(f"   Model type: {config.get('model_type', 'unknown')}")
            
            # Different initialization seeds for diversity
            torch.manual_seed(42 + i * 100)
            np.random.seed(42 + i * 100)
            
            try:
                # Create model
                model = create_model(
                    config['model_type'], 
                    config, 
                    X_train.shape[-1], 
                    self.device
                )
                
                # Train model
                trainer = UnifiedTrainer(
                    device=self.device,
                    model_type=config['model_type']
                )
                
                # Convert to numpy for trainer
                X_np = X_train.cpu().numpy()
                y_np = y_train.cpu().numpy()
                
                trained_model, history = trainer.train_pair_model(
                    pair_name=f"ensemble_model_{i}",
                    X=X_np,
                    y=y_np,
                    config=config,
                    epochs=epochs
                )
                
                models.append(trained_model)
                self.model_histories.append(history)
                
                logger.info(f"   ✅ Model {i+1} training completed")
                
            except Exception as e:
                logger.error(f"   ❌ Model {i+1} training failed: {e}")
                continue
        
        self.trained_models = models
        logger.info(f"🎉 Ensemble training completed: {len(models)}/{len(model_configs)} models successful")
        
        return models
    
    def generate_diverse_configs(self, base_config: Dict, n_models: int = 5, 
                               diversity_factor: float = 0.2) -> List[Dict]:
        """Generate diverse model configurations"""
        configs = []
        
        # Model types to use (if available)
        available_types = ['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer']
        
        for i in range(n_models):
            config = base_config.copy()
            
            # Vary model types
            model_type = available_types[i % len(available_types)]
            config['model_type'] = model_type
            
            # Add diversity to hyperparameters
            if model_type == 'lstm':
                config['model'] = config.get('model', {}).copy()
                base_hidden = config['model'].get('hidden_size', 64)
                config['model']['hidden_size'] = int(base_hidden * (1 + diversity_factor * (i - n_models//2) / n_models))
                config['model']['dropout_rate'] = 0.3 + 0.3 * (i / n_models)
                
            elif model_type == 'enhanced_transformer':
                config['model'] = config.get('model', {}).copy()
                base_d_model = config['model'].get('d_model', 256)
                config['model']['d_model'] = int(base_d_model * (1 + diversity_factor * (i - n_models//2) / n_models))
                config['model']['nhead'] = [4, 8, 8, 4, 8][i % 5]
                
            elif model_type == 'hybrid_lstm_transformer':
                config['hybrid'] = config.get('hybrid', {}).copy()
                base_lstm = config['hybrid'].get('lstm_hidden', 96)
                base_d_model = config['hybrid'].get('d_model', 512)
                
                config['hybrid']['lstm_hidden'] = int(base_lstm * (1 + diversity_factor * (i - n_models//2) / n_models))
                config['hybrid']['d_model'] = int(base_d_model * (1 + diversity_factor * (i - n_models//2) / n_models))
                config['hybrid']['fusion_strategy'] = ['concat', 'attention', 'gated'][i % 3]
            
            # Vary training parameters
            base_lr = config.get('training', {}).get('learning_rate', 1e-3)
            config.setdefault('training', {})['learning_rate'] = base_lr * (0.5 + i / n_models)
            
            configs.append(config)
            
        logger.info(f"📋 Generated {len(configs)} diverse configurations")
        return configs
    
    def calculate_model_weights(self, models: List[nn.Module], 
                              X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
        """Calculate performance-based weights"""
        weights = {}
        
        for i, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                
                # Convert to class predictions
                if pred.dim() > 1:
                    pred_classes = torch.argmax(pred, dim=1)
                else:
                    pred_classes = (pred > 0.5).long()
                
                # Calculate accuracy
                if y_val.dim() > 1:
                    y_classes = torch.argmax(y_val, dim=1)
                else:
                    y_classes = y_val.long()
                
                accuracy = (pred_classes == y_classes).float().mean().item()
                weights[f'model_{i}'] = accuracy
                
                logger.debug(f"Model {i} validation accuracy: {accuracy:.4f}")
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            n_models = len(models)
            weights = {f'model_{i}': 1.0/n_models for i in range(n_models)}
        
        logger.info(f"📊 Model weights calculated: {weights}")
        return weights
    
    def save_ensemble(self, models: List[nn.Module], save_dir: str, 
                     ensemble_config: Dict = None):
        """Save ensemble models and configuration"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for i, model in enumerate(models):
            model_path = save_path / f"model_{i}.pth"
            torch.save(model.state_dict(), model_path)
            
        # Save ensemble configuration
        config = {
            'model_count': len(models),
            'model_types': [type(model).__name__ for model in models],
            'ensemble_config': ensemble_config or {},
            'training_histories': self.model_histories
        }
        
        config_path = save_path / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"💾 Ensemble saved to {save_path}")
    
    def load_ensemble(self, load_dir: str, model_configs: List[Dict]) -> List[nn.Module]:
        """Load ensemble models"""
        from ..models.factory import create_model
        
        load_path = Path(load_dir)
        
        # Load configuration
        config_path = load_path / "ensemble_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                ensemble_config = json.load(f)
            logger.info(f"📂 Loading ensemble with {ensemble_config['model_count']} models")
        
        models = []
        for i, config in enumerate(model_configs):
            model_path = load_path / f"model_{i}.pth"
            if model_path.exists():
                # Create model architecture
                model = create_model(
                    config['model_type'],
                    config,
                    config.get('n_features', 20),  # Default feature count
                    self.device
                )
                
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                models.append(model)
                
                logger.info(f"✅ Model {i} loaded successfully")
            else:
                logger.warning(f"⚠️ Model {i} file not found: {model_path}")
        
        logger.info(f"🎭 Loaded {len(models)} ensemble models")
        return models