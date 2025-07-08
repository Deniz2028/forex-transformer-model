# ========================
# src/ensemble/ensemble_trainer.py - Düzeltilmiş Gerçek Eğitim Entegrasyonu
# ========================

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Enhanced Ensemble Trainer with real training integration"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None):
        self.device = device
        self.config = config or {}
        self.trained_models = []
        self.model_histories = []
        self.training_results = {}
        
        logger.info(f"🏋️ EnsembleTrainer initialized on {device}")
    
    def train_diverse_models(self, X_train: torch.Tensor, y_train: torch.Tensor,
                           model_configs: List[Dict], epochs: int = 50,
                           X_val: torch.Tensor = None, y_val: torch.Tensor = None) -> List[nn.Module]:
        """Train diverse models for ensemble with REAL training"""
        
        try:
            from ..models.factory import create_model
            from ..training.unified_trainer import UnifiedTrainer
        except ImportError as e:
            logger.error(f"❌ Import hatası: {e}")
            logger.info("📦 Alternative import deneniyor...")
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                sys.path.append(str(project_root))
                
                from src.models.factory import create_model
                from src.training.unified_trainer import UnifiedTrainer
            except ImportError as e2:
                logger.error(f"❌ Alternative import de başarısız: {e2}")
                return []
        
        models = []
        self.model_histories = []
        self.training_results = {}
        
        logger.info(f"📋 Generated {len(model_configs)} diverse configurations")
        
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
                
                # Create trainer for this model type
                trainer = UnifiedTrainer(
                    device=self.device,
                    model_type=config['model_type']
                )
                
                # GERÇEK EĞİTİM - placeholder değil!
                training_params = self._get_training_params(config, i)
                
                training_results = trainer.train_model(
                    model_name=f'ensemble_model_{i}',
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    **training_params
                )
                
                # Store results
                self.training_results[f'model_{i}'] = training_results
                self.model_histories.append(training_results.get('history', {}))
                
                models.append(model)
                logger.info(f"   ✅ Model {i+1} training completed")
                
                # Log training summary
                if 'final_val_acc' in training_results:
                    val_acc = training_results['final_val_acc']
                    train_acc = training_results.get('final_train_acc', 0)
                    train_time = training_results.get('training_time', 0)
                    logger.info(f"   📊 Final Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%, Time: {train_time:.1f}s")
                
            except Exception as e:
                logger.error(f"   ❌ Model {i+1} training failed: {e}")
                logger.info(f"   🔄 Continuing with remaining models...")
                continue
        
        logger.info(f"🎉 Ensemble training completed: {len(models)}/{len(model_configs)} models successful")
        
        # Generate training report
        self._generate_training_report(len(model_configs), len(models))
        
        return models
    
    def _get_training_params(self, config: Dict, model_index: int) -> Dict[str, Any]:
        """
        Get training parameters based on model type and diversity with BATCH SIZE FIX.
        
        🔧 FIX: Minimum batch size 2 to prevent BatchNorm error
        """
        model_type = config.get("model_type", "lstm")

        # Ortak varsayılanlar
        params = {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "patience": 5,
        }

        if model_type == "lstm":
            # 🔧 FIX: Batch size minimum 2 olmalı (BatchNorm için)
            batch_size = max(2, 32 + model_index * 8)  # En az 2, tercihen 32, 40, 48...
            params.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": 1e-3 * (0.8 ** model_index),
                    "patience": 5 + model_index,
                }
            )

        elif model_type == "enhanced_transformer":
            # Transformer için de minimum batch size
            batch_size = max(2, 32)  # En az 2
            params.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": 3e-4 * (0.9 ** model_index),
                    "weight_decay": 0.01 + 0.005 * model_index,
                    "patience": 8,
                }
            )

        elif model_type == "hybrid_lstm_transformer":
            # Hybrid için de minimum batch size
            batch_size = max(2, 24 + model_index * 4)  # En az 2, tercihen 24, 28, 32...
            params.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": 5e-4 * (0.85 ** model_index),
                    "patience": 10,
                }
            )

        # Hafif çeşitlilik: ikinci modelde ekstra regularization
        if model_index == 1:
            params["weight_decay"] *= 1.2

        logger.info(f"   🎛️ Model {model_index} params: batch_size={params['batch_size']}, lr={params['learning_rate']:.2e}")
        
        return params


    def _generate_training_report(self, total_models: int, successful_models: int):
        """Generate comprehensive training report"""
        
        logger.info(f"\n📊 ENSEMBLE TRAINING REPORT")
        logger.info(f"{'='*50}")
        logger.info(f"Total models attempted: {total_models}")
        logger.info(f"Successfully trained: {successful_models}")
        logger.info(f"Success rate: {successful_models/total_models*100:.1f}%")
        
        if self.training_results:
            # Calculate statistics
            training_times = [r.get('training_time', 0) for r in self.training_results.values()]
            final_val_accs = [r.get('final_val_acc', 0) for r in self.training_results.values() if 'final_val_acc' in r]
            final_train_accs = [r.get('final_train_acc', 0) for r in self.training_results.values() if 'final_train_acc' in r]
            epochs_trained = [r.get('epochs_trained', 0) for r in self.training_results.values()]
            
            if training_times:
                logger.info(f"Average training time: {np.mean(training_times):.1f}s")
                logger.info(f"Total training time: {sum(training_times):.1f}s")
            
            if final_val_accs:
                logger.info(f"Average validation accuracy: {np.mean(final_val_accs)*100:.2f}%")
                logger.info(f"Best individual accuracy: {max(final_val_accs)*100:.2f}%")
                logger.info(f"Accuracy std deviation: {np.std(final_val_accs)*100:.2f}%")
            
            if final_train_accs:
                logger.info(f"Average train accuracy: {np.mean(final_train_accs)*100:.2f}%")
            
            if epochs_trained:
                logger.info(f"Average epochs trained: {np.mean(epochs_trained):.1f}")
        
        logger.info(f"{'='*50}")
    
    def calculate_model_weights(self, models: List[nn.Module], X_val: torch.Tensor, 
                              y_val: torch.Tensor) -> Dict[str, float]:
        """Calculate performance-based model weights with enhanced metrics"""
        
        logger.info(f"📊 Calculating confidence-weighted model weights...")
        logger.info(f"   📍 Tensors moved to device: {self.device}")
        logger.info(f"   📊 X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Move tensors to device
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        weights = {}
        performance_metrics = {}
        
        for i, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                try:
                    pred = model(X_val)
                    
                    # Handle different output formats
                    if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                        if pred.dim() > 1:
                            pred = pred.squeeze()
                        pred_classes = (pred > 0.5).long()
                        y_classes = y_val.long()
                    else:
                        # Multi-class
                        if pred.dim() > 1:
                            pred_classes = torch.argmax(pred, dim=1)
                        else:
                            pred_classes = (pred > 0.5).long()
                        y_classes = y_val.long()
                    
                    # Calculate accuracy
                    accuracy = (pred_classes == y_classes).float().mean().item()
                    weights[f'model_{i}'] = accuracy
                    
                    # Enhanced performance metrics
                    performance_metrics[f'model_{i}'] = {
                        'accuracy': accuracy,
                        'correct_predictions': (pred_classes == y_classes).sum().item(),
                        'total_predictions': len(y_classes),
                        'model_type': type(model).__name__
                    }
                    
                    logger.info(f"   🎯 Model {i+1}: Accuracy={accuracy:.4f}, Weight={accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error evaluating model {i+1}: {e}")
                    weights[f'model_{i}'] = 0.0
                    performance_metrics[f'model_{i}'] = {
                        'accuracy': 0.0,
                        'error': str(e)
                    }
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            n_models = len(models)
            normalized_weights = {f'model_{i}': 1.0/n_models for i in range(n_models)}
            logger.warning("⚠️ All models failed evaluation, using equal weights")
        
        logger.info(f"   ✅ Final normalized weights: {list(normalized_weights.values())}")
        
        # Store performance metrics for analysis
        self.performance_metrics = performance_metrics
        
        return normalized_weights
    
    def save_ensemble(self, models: List[nn.Module], save_dir: str, 
                     ensemble_config: Dict = None):
        """Save ensemble models and comprehensive training data"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        saved_models = 0
        for i, model in enumerate(models):
            try:
                model_path = save_path / f"model_{i}.pth"
                torch.save(model.state_dict(), model_path)
                saved_models += 1
            except Exception as e:
                logger.error(f"❌ Failed to save model {i}: {e}")
        
        # Enhanced ensemble configuration
        config = {
            'model_count': len(models),
            'saved_models': saved_models,
            'model_types': [type(model).__name__ for model in models],
            'training_results': self.training_results,
            'performance_metrics': getattr(self, 'performance_metrics', {}),
            'device': str(self.device),
            'ensemble_config': ensemble_config or {},
            'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save configuration
        try:
            config_path = save_path / "ensemble_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"✅ Ensemble configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble config: {e}")
        
        # Save training plots if histories available
        if self.model_histories:
            try:
                self._save_training_plots(save_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not save training plots: {e}")
        
        logger.info(f"💾 Enhanced ensemble saved to {save_dir}")
        logger.info(f"   📊 Saved {saved_models}/{len(models)} models successfully")
    
    def _save_training_plots(self, save_path: Path):
        """Save training history plots"""
        
        plots_dir = save_path / "training_plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # Training loss plot
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            for i, history in enumerate(self.model_histories):
                if 'train_loss' in history and history['train_loss']:
                    plt.plot(history['train_loss'], label=f'Model {i+1}', alpha=0.7)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Validation accuracy plot
            plt.subplot(1, 2, 2)
            for i, history in enumerate(self.model_histories):
                if 'val_acc' in history and history['val_acc']:
                    plt.plot(history['val_acc'], label=f'Model {i+1}', alpha=0.7)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   📊 Training plots saved to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Could not save training plots: {e}")

    def load_ensemble(self, load_dir: str, model_configs: List[Dict]) -> Tuple[List[nn.Module], Dict]:
        """Load saved ensemble models"""
        
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {load_dir}")
        
        # Load configuration
        config_path = load_path / "ensemble_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                ensemble_config = json.load(f)
        else:
            ensemble_config = {}
        
        # Load individual models
        models = []
        for i, config in enumerate(model_configs):
            model_path = load_path / f"model_{i}.pth"
            
            if model_path.exists():
                try:
                    from ..models.factory import create_model
                    
                    # Create model architecture
                    model = create_model(
                        config['model_type'],
                        config,
                        input_size=config.get('input_size', 23),
                        device=self.device
                    )
                    
                    # Load state dict
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    models.append(model)
                    
                    logger.info(f"✅ Loaded ensemble model {i+1}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load model {i}: {e}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
        
        logger.info(f"🎉 Loaded {len(models)} ensemble models from {load_dir}")
        
        return models, ensemble_config

    def generate_diverse_configs(self, base_config: Dict, ensemble_models: List[str] = None, 
                               ensemble_size: int = 3, diversity_factor: float = 0.2, 
                               n_models: int = None) -> List[Dict]:
        """Generate diverse model configurations for ensemble"""
        
        # Support both old and new parameter names for compatibility
        if n_models is not None:
            ensemble_size = n_models
        
        if ensemble_models is None:
            ensemble_models = ['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer']
        
        configs = []
        
        # Ensure we have the right number of models
        model_types = ensemble_models * ((ensemble_size // len(ensemble_models)) + 1)
        model_types = model_types[:ensemble_size]
        
        for i, model_type in enumerate(model_types):
            config = base_config.copy()
            config['model_type'] = model_type
            
            # Add diversity based on model type
            if model_type == 'lstm':
                config.update({
                    'hidden_size': 64 + (i * 16),  # 64, 80, 96...
                    'num_layers': 2 + (i % 2),     # 2 or 3
                    'dropout': 0.3 + (i * 0.1),   # 0.3, 0.4, 0.5...
                    'bidirectional': i % 2 == 0    # Alternate
                })
            
            elif model_type == 'enhanced_transformer':
                config.update({
                    'd_model': 128,  # PDF optimal: 512
                    'nhead': 8,      # PDF optimal: 8
                    'num_layers': 2, # PDF optimal: 4
                    'dropout': 0.15 + (i * 0.025),  # 0.15, 0.175, 0.2
                    'ff_dim': 512   # 2048 yerine 1024 (memory için)
                })
            
            elif model_type == 'hybrid_lstm_transformer':
                config.update({
                    'lstm_hidden': 96 + (i * 32),   # 96, 128, 160
                    'd_model': 512,
                    'nhead': 8,
                    'num_layers': 2 + (i % 2),     # 2 or 3
                    'fusion_strategy': ['concat', 'add', 'attention'][i % 3]
                })
            
            configs.append(config)
        
        logger.info(f"📋 Generated {len(configs)} diverse configurations")
        for i, config in enumerate(configs):
            logger.info(f"   Model {i+1}: {config['model_type']} with diversity params")
        
        return configs

# KULLANIM ÖRNEĞİ:
"""
# Temel kullanım:
trainer = EnsembleTrainer(device=torch.device('cuda'))

# Model konfigürasyonları
model_configs = [
    {'model_type': 'lstm', 'hidden_dim': 64, 'num_layers': 2},
    {'model_type': 'enhanced_transformer', 'hidden_dim': 128, 'num_heads': 8},
    {'model_type': 'hybrid_lstm_transformer', 'hidden_dim': 96}
]

# Ensemble eğitimi
models = trainer.train_diverse_models(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model_configs=model_configs,
    epochs=30
)

# Model ağırlıklarını hesapla
weights = trainer.calculate_model_weights(models, X_val, y_val)

# Ensemble'ı kaydet
trainer.save_ensemble(models, 'saved_ensembles/my_ensemble')
"""