# ========================
# src/ensemble/ensemble_trainer.py - COMPLETE FIXED VERSION
# ========================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from pathlib import Path

from src.models.factory import create_model
from src.training.unified_trainer import UnifiedTrainer
from src.ensemble.base_ensemble import VotingStrategies

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Enhanced ensemble trainer with proper target handling and CUDA fixes"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.models = []
        self.model_weights = []
        self.training_results = {}
        self.model_histories = []
        self.performance_metrics = {}
        
        logger.info(f"🏋️ EnsembleTrainer initialized on {device}")
    
    def generate_diverse_configs(self, base_configs: List[Dict], ensemble_size: int, 
                                diversity_factor: float = 0.2) -> List[Dict]:
        """Generate diverse model configurations for ensemble"""
        
        configs = []
        np.random.seed(42)  # Reproducible diversity
        
        for i in range(ensemble_size):
            model_type = base_configs[i % len(base_configs)]['model_type']
            config = base_configs[i % len(base_configs)].copy()
            
            # Add diversity through parameter variations
            if model_type == 'lstm':
                config['hidden_size'] = np.random.choice([256, 384, 512])
                config['num_layers'] = np.random.choice([3, 4, 5])
                config['dropout'] = np.random.uniform(0.3, 0.5)
                config['batch_size'] = np.random.choice([16, 24, 32])
                
            elif model_type == 'enhanced_transformer':
                config['d_model'] = np.random.choice([384, 512, 640])
                config['nhead'] = np.random.choice([6, 8, 12])
                config['num_layers'] = np.random.choice([3, 4, 6])
                config['dropout'] = np.random.uniform(0.1, 0.3)
                config['batch_size'] = np.random.choice([12, 16, 20])
            
            # Training diversity
            config['learning_rate'] = np.random.uniform(5e-4, 2e-3)
            config['weight_decay'] = np.random.uniform(1e-5, 1e-4)
            
            configs.append(config)
        
        logger.info(f"📋 Generated {len(configs)} diverse configurations")
        return configs
    
    def train_diverse_models(self, configs: List[Dict], X_train, y_train, 
                           X_val=None, y_val=None, epochs=30) -> List[Dict]:
        """Train diverse ensemble models with COMPLETE CUDA fix"""
        
        results = []
        successful_models = 0
        
        for i, config in enumerate(configs):
            logger.info(f"🎭 Training ensemble model {i+1}/{len(configs)}")
            logger.info(f"   Model type: {config['model_type']}")
            
            try:
                # 🔧 CRITICAL TARGET VALIDATION BEFORE EVERYTHING
                logger.info(f"🔧 Pre-training target validation for model {i+1}:")
                logger.info(f"   y_train: shape={y_train.shape}, dtype={y_train.dtype}")
                logger.info(f"   y_train range: {y_train.min()}-{y_train.max()}")
                logger.info(f"   y_train unique: {torch.unique(y_train)}")
                
                # FORCE targets to be in correct range [0, 1] for binary classification
                y_train_fixed = torch.clamp(y_train, 0, 1).long()
                y_val_fixed = torch.clamp(y_val, 0, 1).long() if y_val is not None else None
                
                # Verify target fixing
                logger.info(f"✅ Targets fixed for model {i+1}:")
                logger.info(f"   Fixed y_train unique: {torch.unique(y_train_fixed)}")
                logger.info(f"   Fixed y_train dtype: {y_train_fixed.dtype}")
                if y_val_fixed is not None:
                    logger.info(f"   Fixed y_val unique: {torch.unique(y_val_fixed)}")
                
                # Set reproducible seed for this model
                torch.manual_seed(42 + i * 100)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + i * 100)
                
                # Create model with proper configuration
                model = create_model(
                    config['model_type'], 
                    config, 
                    n_features=X_train.shape[-1], 
                    device=self.device
                )
                
                # Extract training parameters
                training_params = {
                    'epochs': epochs,
                    'batch_size': config.get('batch_size', 32),
                    'learning_rate': config.get('learning_rate', 1e-3),
                    'weight_decay': config.get('weight_decay', 1e-4),
                    'patience': config.get('patience', 10)
                }
                
                logger.info(f"   🎛️ Model {i} params: batch_size={training_params['batch_size']}, lr={training_params['learning_rate']:.2e}")
                
                # Initialize trainer with proper model type detection
                trainer = UnifiedTrainer(self.device, config['model_type'])
                
                # 🔧 CRITICAL: Pass FIXED targets to trainer
                result = trainer.train_model(
                    f"ensemble_model_{i}", 
                    model, 
                    X_train, 
                    y_train_fixed,  # ← FIXED targets, guaranteed [0,1] range
                    X_val, 
                    y_val_fixed,    # ← FIXED targets, guaranteed [0,1] range
                    **training_params
                )
                
                # Validate training result
                if result and not result.get('placeholder', False):
                    result['model'] = model
                    result['config'] = config
                    result['model_index'] = i
                    results.append(result)
                    successful_models += 1
                    
                    # Store results for ensemble analysis
                    self.training_results[f'model_{i}'] = {
                        'train_acc': result.get('final_train_acc', 0.0),
                        'val_acc': result.get('final_val_acc', 0.0),
                        'train_loss': result.get('final_train_loss', float('inf')),
                        'val_loss': result.get('final_val_loss', float('inf')),
                        'training_time': result.get('training_time', 0.0),
                        'epochs_trained': result.get('epochs_trained', 0),
                        'model_type': config['model_type']
                    }
                    
                    # Store training history
                    if 'history' in result:
                        self.model_histories.append(result['history'])
                    
                    logger.info(f"   ✅ Model {i+1} trained successfully")
                    logger.info(f"      Train Acc: {result.get('final_train_acc', 0):.3f}, Val Acc: {result.get('final_val_acc', 0):.3f}")
                else:
                    logger.warning(f"   ⚠️ Model {i+1} training returned placeholder/empty result")
                    
            except Exception as e:
                logger.error(f"   ❌ Model {i+1} training failed: {e}")
                logger.info(f"   🔄 Continuing with remaining models...")
                
                # Clear CUDA cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        logger.info(f"🎭 Ensemble training completed: {successful_models}/{len(configs)} models successful")
        
        if successful_models == 0:
            raise RuntimeError("No models trained successfully in ensemble")
        
        return results
    
    def calculate_model_weights(self, models_results: List[Dict], 
                               strategy: str = 'performance') -> Dict[str, float]:
        """Calculate weights for ensemble models based on performance"""
        
        if strategy == 'performance':
            weights = {}
            for result in models_results:
                model_name = result['model_name']
                
                # Prioritize validation accuracy, fallback to training accuracy
                if 'final_val_acc' in result and result['final_val_acc'] is not None:
                    weight = result['final_val_acc']
                elif 'final_train_acc' in result and result['final_train_acc'] is not None:
                    weight = result['final_train_acc']
                else:
                    weight = 0.5  # Default weight
                    
                weights[model_name] = max(weight, 0.1)  # Minimum weight of 0.1
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Fallback to equal weights
                n_models = len(models_results)
                weights = {r['model_name']: 1.0/n_models for r in models_results}
            
        elif strategy == 'equal':
            weights = {r['model_name']: 1.0 / len(models_results) for r in models_results}
            
        else:
            raise ValueError(f"Unknown weighting strategy: {strategy}")
        
        logger.info(f"🎯 Model weights ({strategy}): {[(k, f'{v:.3f}') for k, v in weights.items()]}")
        return weights
    
    def predict_ensemble(self, models_results: List[Dict], X_test, 
                        strategy: str = 'confidence_weighted') -> torch.Tensor:
        """Make ensemble predictions with proper output handling"""
        
        predictions = []
        confidences = []
        
        for i, result in enumerate(models_results):
            model = result['model']
            model.eval()
            
            with torch.no_grad():
                X_test_device = X_test.to(self.device)
                pred = model(X_test_device)
                
                # 🔧 FIXED: Proper output processing for different model types
                if pred.dim() == 1:
                    # Single output (BCE) - convert to probabilities
                    pred_probs = torch.sigmoid(pred)
                    pred_2d = torch.stack([1 - pred_probs, pred_probs], dim=1)
                    predictions.append(pred_2d)
                    confidences.append(torch.abs(pred_probs - 0.5) * 2)  # Distance from 0.5
                    
                elif pred.dim() == 2:
                    if pred.shape[1] == 2:
                        # Binary classification with 2 outputs (CrossEntropy)
                        pred_probs = torch.softmax(pred, dim=1)
                        predictions.append(pred_probs)
                        confidences.append(torch.max(pred_probs, dim=1)[0])
                    elif pred.shape[1] == 1:
                        # Single output reshaped - treat as BCE
                        pred_probs = torch.sigmoid(pred.squeeze(1))
                        pred_2d = torch.stack([1 - pred_probs, pred_probs], dim=1)
                        predictions.append(pred_2d)
                        confidences.append(torch.abs(pred_probs - 0.5) * 2)
                    else:
                        # Multi-class (3+ classes)
                        pred_probs = torch.softmax(pred, dim=1)
                        predictions.append(pred_probs)
                        confidences.append(torch.max(pred_probs, dim=1)[0])
                else:
                    raise ValueError(f"Unexpected prediction shape: {pred.shape}")
        
        # Combine predictions using specified strategy
        if strategy == 'confidence_weighted':
            final_pred = VotingStrategies.confidence_weighted(predictions)
        elif strategy == 'performance_weighted':
            model_weights = self.calculate_model_weights(models_results, 'performance')
            weight_values = [model_weights.get(r['model_name'], 1.0/len(models_results)) 
                           for r in models_results]
            weight_tensors = [torch.tensor(w, device=self.device) for w in weight_values]
            final_pred = VotingStrategies.performance_weighted(predictions, weight_tensors)
        elif strategy == 'simple_average':
            final_pred = VotingStrategies.simple_average(predictions)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
        
        return final_pred
    
    def evaluate_ensemble(self, models_results: List[Dict], X_val, y_val, 
                         strategy: str = 'confidence_weighted') -> Dict[str, float]:
        """Evaluate ensemble performance on validation data"""
        
        # Ensemble prediction
        ensemble_pred = self.predict_ensemble(models_results, X_val, strategy)
        
        # Convert to class predictions
        if ensemble_pred.dim() == 2:
            ensemble_classes = torch.argmax(ensemble_pred, dim=1)
        else:
            ensemble_classes = (ensemble_pred > 0.5).long()
        
        # Calculate ensemble accuracy
        y_val_device = y_val.to(self.device)
        ensemble_accuracy = (ensemble_classes == y_val_device).float().mean().item()
        
        # Individual model accuracies
        individual_accuracies = {}
        for result in models_results:
            model = result['model']
            model.eval()
            
            with torch.no_grad():
                pred = model(X_val.to(self.device))
                
                if pred.dim() == 2 and pred.shape[1] == 2:
                    pred_classes = torch.argmax(pred, dim=1)
                elif pred.dim() == 2 and pred.shape[1] == 1:
                    pred_classes = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
                elif pred.dim() == 1:
                    pred_classes = (torch.sigmoid(pred) > 0.5).long()
                else:
                    pred_classes = torch.argmax(pred, dim=1)
                
                accuracy = (pred_classes == y_val_device).float().mean().item()
                individual_accuracies[result['model_name']] = accuracy
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies,
            'best_individual': max(individual_accuracies.values()) if individual_accuracies else 0.0,
            'improvement': ensemble_accuracy - max(individual_accuracies.values()) if individual_accuracies else 0.0
        }
    
    def save_ensemble(self, models_results: List[Dict], save_path: str, 
                     ensemble_config: Dict = None):
        """Save ensemble models and comprehensive metadata"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        saved_models = 0
        for i, result in enumerate(models_results):
            try:
                model_path = save_dir / f"model_{i}.pth"
                torch.save({
                    'model_state_dict': result['model'].state_dict(),
                    'config': result['config'],
                    'training_results': {k: v for k, v in result.items() if k != 'model'}
                }, model_path)
                saved_models += 1
                logger.info(f"   💾 Model {i} saved: {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save model {i}: {e}")
        
        # Enhanced ensemble metadata
        metadata = {
            'num_models': len(models_results),
            'saved_models': saved_models,
            'model_types': [r['config']['model_type'] for r in models_results],
            'training_results': self.training_results,
            'performance_metrics': self.performance_metrics,
            'device': str(self.device),
            'ensemble_config': ensemble_config or {},
            'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_training_time': sum(r.get('training_time', 0) for r in models_results),
            'average_accuracy': np.mean([r.get('final_val_acc', 0) for r in models_results if r.get('final_val_acc')])
        }
        
        # Save metadata
        try:
            metadata_path = save_dir / 'ensemble_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"✅ Ensemble metadata saved: {metadata_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble metadata: {e}")
        
        # Save training plots if available
        if self.model_histories:
            try:
                self._save_training_plots(save_dir)
            except Exception as e:
                logger.warning(f"⚠️ Could not save training plots: {e}")
        
        logger.info(f"💾 Ensemble saved to {save_path}")
        logger.info(f"   📊 Successfully saved {saved_models}/{len(models_results)} models")
    
    def _save_training_plots(self, save_dir: Path):
        """Save training history visualization plots"""
        
        try:
            import matplotlib.pyplot as plt
            
            plots_dir = save_dir / "training_plots"
            plots_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(15, 5))
            
            # Training loss plot
            plt.subplot(1, 3, 1)
            for i, history in enumerate(self.model_histories):
                if 'train_loss' in history and history['train_loss']:
                    plt.plot(history['train_loss'], label=f'Model {i+1}', alpha=0.7)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Training accuracy plot
            plt.subplot(1, 3, 2)
            for i, history in enumerate(self.model_histories):
                if 'train_acc' in history and history['train_acc']:
                    plt.plot(history['train_acc'], label=f'Model {i+1}', alpha=0.7)
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Validation accuracy plot
            plt.subplot(1, 3, 3)
            for i, history in enumerate(self.model_histories):
                if 'val_acc' in history and history['val_acc']:
                    plt.plot(history['val_acc'], label=f'Model {i+1}', alpha=0.7)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   📊 Training plots saved to {plots_dir}")
            
        except ImportError:
            logger.warning("   ⚠️ matplotlib not available, skipping plots")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not save training plots: {e}")
    
    def load_ensemble(self, load_dir: str, model_configs: List[Dict]) -> Tuple[List[torch.nn.Module], Dict]:
        """Load saved ensemble models and metadata"""
        
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {load_dir}")
        
        # Load metadata
        metadata_path = load_path / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📋 Loaded ensemble metadata: {metadata.get('num_models', 0)} models")
        else:
            metadata = {}
            logger.warning("⚠️ No metadata found, using default configuration")
        
        # Load individual models
        models = []
        for i, config in enumerate(model_configs):
            model_path = load_path / f"model_{i}.pth"
            
            if model_path.exists():
                try:
                    # Create model with configuration
                    model = create_model(
                        config['model_type'], 
                        config, 
                        n_features=config.get('n_features', 82), 
                        device=self.device
                    )
                    
                    # Load state dict
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    models.append(model)
                    logger.info(f"   ✅ Model {i} loaded successfully")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to load model {i}: {e}")
            else:
                logger.warning(f"   ⚠️ Model file not found: {model_path}")
        
        logger.info(f"📦 Loaded {len(models)}/{len(model_configs)} models successfully")
        
        return models, metadata