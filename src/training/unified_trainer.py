# ========================
# src/training/unified_trainer.py - COMPLETE FILE
# ========================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """Unified trainer for all model types with real training implementation"""
    
    def __init__(self, device: torch.device, model_type: str = 'lstm'):
        self.device = device
        self.model_type = model_type.upper()
        
        logger.info(f"🔧 UnifiedTrainer başlatıldı:")
        logger.info(f"   Model type: {self.model_type}")
        logger.info(f"   Target mode: binary")
        logger.info(f"   Device: {device}")
        logger.info(f"   Reproducibility seed: 42")
    
    def train_model(self, model_name: str, model, X_train, y_train, 
                   X_val=None, y_val=None, epochs=30, **kwargs) -> Dict[str, Any]:
        """Universal model training method"""
        
        logger.info(f"🏋️ {model_name} için {self.model_type} modeli eğitiliyor...")
        
        if self.model_type == 'LSTM':
            return self._train_lstm_real(model_name, model, X_train, y_train, 
                                        X_val, y_val, epochs, **kwargs)
        elif self.model_type in ['ENHANCED_TRANSFORMER', 'TRANSFORMER']:
            return self._train_transformer_real(model_name, model, X_train, y_train,
                                               X_val, y_val, epochs, **kwargs)
        elif self.model_type == 'HYBRID_LSTM_TRANSFORMER':
            return self._train_hybrid_real(model_name, model, X_train, y_train,
                                          X_val, y_val, epochs, **kwargs)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using placeholder")
            return self._placeholder_training(model_name)
    
    def _train_lstm_real(self, model_name: str, model, X_train, y_train, 
                        X_val, y_val, epochs=30, **kwargs) -> Dict[str, Any]:
        """Gerçek LSTM eğitimi - placeholder yerine"""
        
        # Model parametrelerini logla
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Model parametreleri: {total_params:,}")
        
        # Hyperparameters
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        weight_decay = kwargs.get('weight_decay', 0.01)
        patience = kwargs.get('patience', 5)
        
        logger.info(f"   🎯 Eğitim parametreleri: LR={learning_rate}, Batch={batch_size}, Epochs={epochs}")
        
        # DataLoaders oluştur
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
        else:
            val_loader = None
        
        # Optimizer ve loss function
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss function - target mode'a göre
        if hasattr(model, 'target_mode') and model.target_mode == 'binary':
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=False
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Handle different output formats
                if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()
                    batch_y = batch_y.float()
                    loss = criterion(outputs, batch_y)
                    
                    # Accuracy calculation for binary
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == batch_y).sum().item()
                else:
                    # Multi-class
                    if batch_y.dim() == 1:
                        batch_y = batch_y.long()
                    loss = criterion(outputs, batch_y)
                    
                    # Accuracy calculation for multi-class
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_total += batch_y.size(0)
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = model(batch_X)
                        
                        # Handle different output formats for validation
                        if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                            if outputs.dim() > 1:
                                outputs = outputs.squeeze()
                            batch_y = batch_y.float()
                            loss = criterion(outputs, batch_y)
                            
                            predicted = (outputs > 0.5).float()
                            val_correct += (predicted == batch_y).sum().item()
                        else:
                            if batch_y.dim() == 1:
                                batch_y = batch_y.long()
                            loss = criterion(outputs, batch_y)
                            
                            _, predicted = torch.max(outputs.data, 1)
                            val_correct += (predicted == batch_y).sum().item()
                        
                        val_loss += loss.item()
                        val_total += batch_y.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Logging every 5 epochs or on early stopping
                if epoch % 5 == 0 or patience_counter >= patience:
                    logger.info(f"   Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {avg_val_loss:.4f}, "
                               f"Train Acc: {train_accuracy*100:.2f}%, "
                               f"Val Acc: {val_accuracy*100:.2f}%")
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"   Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation data
                if epoch % 10 == 0:
                    logger.info(f"   Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Train Acc: {train_accuracy*100:.2f}%")
        
        # Restore best model if validation was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"   ✅ LSTM modeli eğitildi - Süre: {training_time:.2f}s")
        
        return {
            'history': history,
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0
        }
    
    def _train_transformer_real(self, model_name: str, model, X_train, y_train, 
                               X_val, y_val, epochs=30, **kwargs) -> Dict[str, Any]:
        """Gelişmiş Enhanced Transformer eğitimi"""
        
        logger.info(f"🚀 Enhanced Transformer eğitimi başlıyor...")
        
        # Model parametrelerini logla
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Model parametreleri: {total_params:,}")
        
        # Enhanced hyperparameters
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.0003)
        weight_decay = kwargs.get('weight_decay', 0.01)
        patience = kwargs.get('patience', 8)  # Daha geç early stopping
        max_epochs = min(epochs, 50)  # Max 50 epoch
        
        logger.info(f"   🎯 Eğitim parametreleri: LR={learning_rate}, Batch={batch_size}, Max Epochs={max_epochs}")
        
        # Data augmentation için helper function
        def augment_data(X, y, noise_factor=0.01):
            """Basit data augmentation"""
            if torch.rand(1) < 0.3:  # %30 şansla augment
                noise = torch.randn_like(X) * noise_factor
                X_aug = X + noise
                return X_aug, y
            return X, y
        
        # DataLoaders with augmentation
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
        
        # Advanced optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function with label smoothing
        if hasattr(model, 'target_mode') and model.target_mode == 'binary':
            criterion = nn.BCELoss()
            use_label_smoothing = False
        else:
            # Label smoothing for multi-class
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            use_label_smoothing = True
        
        # Advanced learning rate scheduler
        total_steps = len(train_loader) * max_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 3,  # Peak LR
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup + ramp up
            anneal_strategy='cos',
            div_factor=10,  # max_lr / div_factor = initial_lr
            final_div_factor=100  # max_lr / final_div_factor = final_lr
        )
        
        # Training history with more metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'grad_norms': [],
            'loss_components': []
        }
        
        # Training loop variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        logger.info(f"   🔥 Starting enhanced training with {len(train_loader)} batches per epoch")
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            grad_norms = []
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Data augmentation
                if epoch > 5:  # Start augmentation after 5 epochs
                    batch_X, batch_y = augment_data(batch_X, batch_y, noise_factor=0.01)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Loss calculation
                if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()
                    batch_y = batch_y.float()
                    loss = criterion(outputs, batch_y)
                    
                    # Accuracy for binary
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == batch_y).sum().item()
                else:
                    # Multi-class
                    if batch_y.dim() == 1:
                        batch_y = batch_y.long()
                    loss = criterion(outputs, batch_y)
                    
                    # Accuracy for multi-class
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping and norm tracking
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                
                optimizer.step()
                scheduler.step()  # Step-wise scheduling
                
                train_loss += loss.item()
                train_total += batch_y.size(0)
                
                # Log progress every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.debug(f"   Batch {batch_idx}/{len(train_loader)}: "
                                f"Loss={loss.item():.4f}, LR={current_lr:.6f}")
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            avg_grad_norm = np.mean(grad_norms)
            current_lr = scheduler.get_last_lr()[0]
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['learning_rates'].append(current_lr)
            history['grad_norms'].append(avg_grad_norm)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    
                    # Validation loss and accuracy
                    if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze()
                        batch_y = batch_y.float()
                        loss = criterion(outputs, batch_y)
                        
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == batch_y).sum().item()
                    else:
                        if batch_y.dim() == 1:
                            batch_y = batch_y.long()
                        loss = criterion(outputs, batch_y)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()
                    
                    val_loss += loss.item()
                    val_total += batch_y.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Enhanced logging for hybrid model
            if epoch % 3 == 0 or patience_counter >= patience:
                memory_info = f", GPU Mem: {history['memory_usage'][-1]:.1f}GB" if history['memory_usage'] else ""
                logger.info(f"   Epoch {epoch+1}/{max_epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, "
                           f"Train Acc: {train_accuracy*100:.2f}%, "
                           f"Val Acc: {val_accuracy*100:.2f}%, "
                           f"LR: {current_lr:.6f}{memory_info}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"   Early stopping at epoch {epoch+1} (patience={patience})")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"   ✅ Hybrid model eğitimi tamamlandı")
        logger.info(f"   📊 Süre: {training_time:.2f}s, En iyi Val Loss: {best_val_loss:.4f}")
        
        # Final memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'history': history,
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'max_memory_usage': max(history['memory_usage']) if history['memory_usage'] else 0,
            'mixed_precision': use_amp,
            'model_size': f"{total_params/1e6:.1f}M parameters"
        }
    
    def _placeholder_training(self, model_name: str) -> Dict[str, Any]:
        """Placeholder training for backward compatibility"""
        import time
        time.sleep(1)  # Simulate training time
        logger.info(f"   ✅ {model_name} modeli eğitildi (placeholder)")
        return {
            'history': {'train_loss': [], 'val_loss': []},
            'training_time': 1.0,
            'epochs_trained': 1,
            'placeholder': True
        }
    
    def get_training_summary(self, training_results: Dict[str, Any]) -> str:
        """Generate a training summary report"""
        
        if training_results.get('placeholder', False):
            return f"📋 Placeholder training completed in {training_results['training_time']:.1f}s"
        
        summary = []
        summary.append(f"📋 Training Summary:")
        summary.append(f"   Model Type: {self.model_type}")
        summary.append(f"   Training Time: {training_results['training_time']:.2f}s")
        summary.append(f"   Epochs Trained: {training_results['epochs_trained']}")
        
        if 'final_train_acc' in training_results:
            summary.append(f"   Final Train Acc: {training_results['final_train_acc']*100:.2f}%")
        
        if 'final_val_acc' in training_results:
            summary.append(f"   Final Val Acc: {training_results['final_val_acc']*100:.2f}%")
        
        if 'best_val_loss' in training_results:
            summary.append(f"   Best Val Loss: {training_results['best_val_loss']:.4f}")
        
        # Enhanced features for advanced models
        if 'enhanced_features' in training_results:
            features = training_results['enhanced_features']
            summary.append(f"   Enhanced Features:")
            for feature, enabled in features.items():
                if enabled:
                    summary.append(f"     ✅ {feature.replace('_', ' ').title()}")
        
        # Memory usage for hybrid models
        if 'max_memory_usage' in training_results and training_results['max_memory_usage'] > 0:
            summary.append(f"   Max GPU Memory: {training_results['max_memory_usage']:.1f}GB")
        
        # Mixed precision info
        if 'mixed_precision' in training_results and training_results['mixed_precision']:
            summary.append(f"   ⚡ Mixed Precision: Enabled")
        
        return "\n".join(summary)


# ========================
# Compatibility functions for existing code
# ========================

def create_trainer(device: torch.device, model_type: str = 'lstm') -> UnifiedTrainer:
    """Factory function to create a trainer instance"""
    return UnifiedTrainer(device=device, model_type=model_type)


def train_model_unified(model, X_train, y_train, X_val=None, y_val=None, 
                       device=None, model_type='lstm', epochs=30, **kwargs):
    """Unified training function for backward compatibility"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = UnifiedTrainer(device=device, model_type=model_type)
    
    results = trainer.train_model(
        model_name=f"{model_type}_model",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        **kwargs
    )
    
    return results


# ========================
# Training utilities
# ========================

def setup_training_environment(seed: int = 42):
    """Setup reproducible training environment"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"🎯 Training environment setup with seed: {seed}")


def get_device_info():
    """Get detailed device information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    info = {
        'device': device,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0,
        'memory_available': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3) if torch.cuda.is_available() else 0
    }
    
    logger.info(f"🖥️ Device: {info['device_name']}")
    if torch.cuda.is_available():
        logger.info(f"   Memory: {info['memory_available']:.1f}GB / {info['memory_total']:.1f}GB available")
    
    return info


# ========================
# Export all public functions
# ========================

__all__ = [
    'UnifiedTrainer',
    'create_trainer', 
    'train_model_unified',
    'setup_training_environment',
    'get_device_info'
]


# ========================
# Usage Example
# ========================

"""
# Basic usage:
trainer = UnifiedTrainer(device=torch.device('cuda'), model_type='lstm')
results = trainer.train_model(
    model_name='my_lstm',
    model=lstm_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=30,
    batch_size=32,
    learning_rate=0.001
)

# Print training summary
print(trainer.get_training_summary(results))

# Enhanced Transformer training:
transformer_trainer = UnifiedTrainer(device=torch.device('cuda'), model_type='enhanced_transformer')
transformer_results = transformer_trainer.train_model(
    model_name='enhanced_transformer',
    model=transformer_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=50,
    batch_size=32,
    learning_rate=0.0003,
    patience=8
)

# Hybrid model training:
hybrid_trainer = UnifiedTrainer(device=torch.device('cuda'), model_type='hybrid_lstm_transformer')
hybrid_results = hybrid_trainer.train_model(
    model_name='hybrid_model',
    model=hybrid_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=40,
    batch_size=24,
    learning_rate=0.0005,
    patience=10
)
"""y)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()
                    
                    val_loss += loss.item()
                    val_total += batch_y.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Early stopping with improved logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Enhanced logging every 5 epochs
            if epoch % 5 == 0 or patience_counter >= patience:
                logger.info(f"   Epoch {epoch+1}/{max_epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, "
                           f"Train Acc: {train_accuracy*100:.2f}%, "
                           f"Val Acc: {val_accuracy*100:.2f}%, "
                           f"LR: {current_lr:.6f}, "
                           f"Grad Norm: {avg_grad_norm:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"   Early stopping at epoch {epoch+1} (patience={patience})")
                break
            
            # Learning rate plateau detection
            if len(history['val_loss']) > 10:
                recent_losses = history['val_loss'][-10:]
                if max(recent_losses) - min(recent_losses) < 0.001:
                    logger.info(f"   Loss plateau detected at epoch {epoch+1}")
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"   ✅ Enhanced Transformer eğitimi tamamlandı")
        logger.info(f"   📊 Süre: {training_time:.2f}s, En iyi Val Loss: {best_val_loss:.4f}")
        
        return {
            'history': history,
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_epoch': len(history['val_loss']) - patience_counter,
            'avg_grad_norm': np.mean(history['grad_norms']),
            'enhanced_features': {
                'data_augmentation': True,
                'label_smoothing': use_label_smoothing,
                'onecycle_lr': True,
                'gradient_clipping': True
            }
        }
    
    def _train_hybrid_real(self, model_name: str, model, X_train, y_train, 
                          X_val, y_val, epochs=30, **kwargs) -> Dict[str, Any]:
        """Hybrid LSTM-Transformer gerçek eğitimi"""
        
        logger.info(f"🚀 Enhanced Transformer eğitimi başlıyor...")
        
        # Model parametrelerini logla
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Model parametreleri: {total_params:,}")
        
        # Hybrid model için özel hyperparameters
        batch_size = kwargs.get('batch_size', 32)  # Küçük batch size (13M parametre)
        learning_rate = kwargs.get('learning_rate', 0.0005)  # Düşük LR
        weight_decay = kwargs.get('weight_decay', 0.01)
        patience = kwargs.get('patience', 10)  # Daha fazla patience
        max_epochs = min(epochs, 40)  # Max 40 epoch
        
        logger.info(f"   🎯 Eğitim parametreleri: LR={learning_rate}, Batch={batch_size}")
        logger.info(f"   🎯 Target mode: {getattr(model, 'target_mode', 'unknown')}")
        
        # Memory-efficient DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        
        # Optimizer for large model
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Loss function - DÜZELTME: Target mode'a göre doğru loss
        if hasattr(model, 'target_mode') and model.target_mode == 'binary':
            criterion = nn.BCELoss()
            logger.info(f"   🎯 Binary classification: Using BCELoss")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info(f"   🎯 Multi-class classification: Using CrossEntropyLoss")
        
        # Learning rate scheduler for large models
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=5, 
            factor=0.7, 
            verbose=True,
            min_lr=1e-7
        )
        
        # Mixed precision training for 13M parameter model
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        use_amp = scaler is not None
        
        if use_amp:
            logger.info(f"   ⚡ Mixed precision training enabled")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'memory_usage': []
        }
        
        # Training loop variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        for epoch in range(max_epochs):
            # Memory monitoring
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                history['memory_usage'].append(memory_allocated)
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        
                        # DÜZELTME: Output handling için target mode kontrolü
                        if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                            # Binary mode: outputs should be [batch, 1] with sigmoid
                            if outputs.dim() > 1 and outputs.size(1) > 1:
                                outputs = outputs[:, 0]  # Take first column if multi-output
                            outputs = outputs.squeeze()
                            batch_y = batch_y.float()
                            loss = criterion(outputs, batch_y)
                            
                            # Binary accuracy
                            predicted = (outputs > 0.5).float()
                            train_correct += (predicted == batch_y).sum().item()
                        else:
                            # Multi-class mode
                            if batch_y.dim() == 1:
                                batch_y = batch_y.long()
                            loss = criterion(outputs, batch_y)
                            
                            # Multi-class accuracy
                            _, predicted = torch.max(outputs.data, 1)
                            train_correct += (predicted == batch_y).sum().item()
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular precision
                    outputs = model(batch_X)
                    
                    # DÜZELTME: Same output handling logic
                    if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                        if outputs.dim() > 1 and outputs.size(1) > 1:
                            outputs = outputs[:, 0]  # Take first column if multi-output
                        outputs = outputs.squeeze()
                        batch_y = batch_y.float()
                        loss = criterion(outputs, batch_y)
                        
                        # Binary accuracy
                        predicted = (outputs > 0.5).float()
                        train_correct += (predicted == batch_y).sum().item()
                    else:
                        # Multi-class mode
                        if batch_y.dim() == 1:
                            batch_y = batch_y.long()
                        loss = criterion(outputs, batch_y)
                        
                        # Multi-class accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        train_correct += (predicted == batch_y).sum().item()
                    
                    # Regular backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                train_total += batch_y.size(0)
                
                # Memory cleanup for large model
                if batch_idx % 50 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['learning_rates'].append(current_lr)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    # DÜZELTME: Validation loss calculation
                    if hasattr(model, 'target_mode') and model.target_mode == 'binary':
                        if outputs.dim() > 1 and outputs.size(1) > 1:
                            outputs = outputs[:, 0]
                        outputs = outputs.squeeze()
                        batch_y = batch_y.float()
                        loss = criterion(outputs, batch_y)
                        
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == batch_y).sum().item()
                    else:
                        if batch_y.dim() == 1:
                            batch_y = batch_y.long()
                        loss = criterion(outputs, batch_
