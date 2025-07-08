# ========================
# src/training/unified_trainer.py - COMPLETE AND FIXED FILE
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

        # 🔧 CRITICAL TARGET FIX - EN BAŞTA
        print(f"🔧 UnifiedTrainer: Re-checking targets...")
        print(f"   y_train: shape={y_train.shape}, dtype={y_train.dtype}, min={y_train.min()}, max={y_train.max()}")
        print(f"   y_train unique: {torch.unique(y_train)}")
        
        # FORCE targets to be correct type and range
        y_train = torch.clamp(y_train, 0, 1).long()
        if y_val is not None:
            y_val = torch.clamp(y_val, 0, 1).long()
            print(f"   y_val: shape={y_val.shape}, dtype={y_val.dtype}, min={y_val.min()}, max={y_val.max()}")
        
        print(f"🔧 UnifiedTrainer: Targets fixed!")
        
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
    
    def _train_lstm_real(self, model_name: str, model, X_train, y_train, X_val, y_val,
                        epochs, batch_size, learning_rate, weight_decay, patience):
        """Real LSTM training with FIXED logging and validation accuracy"""
        
        logger.info(f"🏋️ {model_name} için LSTM modeli eğitiliyor...")
        logger.info(f"   📊 Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        # 🔧 CRITICAL FIX: Move tensors to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        if y_val is not None:
            y_val = y_val.to(self.device)
        
        logger.info(f"   📍 Tensors moved to device: {self.device}")
        
        start_time = time.time()
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        
        # 🔧 FIXED: Proper loss function selection based on model output
        # Check model output size to determine loss function
        with torch.no_grad():
            sample_output = model(X_train[:1])  # Get sample output
            output_size = sample_output.shape[-1]
        
        if output_size == 1:
            # Single output for binary classification
            criterion = nn.BCEWithLogitsLoss()
            use_sigmoid = True
            logger.info(f"   🔥 Using BCEWithLogitsLoss (single output)")
        else:
            # Multiple outputs (2 for binary, 3 for three_class)
            criterion = nn.CrossEntropyLoss()
            use_sigmoid = False
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs)")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training history - TRAIN ACC EKLENDI
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        logger.info(f"   🚀 Starting training: {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0  # ✅ TRAIN ACCURACY İÇİN EKLENEN
            train_total = 0    # ✅ TRAIN ACCURACY İÇİN EKLENEN
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(batch_X)
                
                # 🔧 FIXED: Proper target formatting based on loss function
                if use_sigmoid:
                    if pred.dim() > 1:
                        pred = pred.squeeze(-1)
                    batch_y = batch_y.float()
                else:
                    batch_y = batch_y.long()
                
                loss = criterion(pred, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
                
                # ✅ TRAIN ACCURACY CALCULATION - EKLENEN KISIM
                with torch.no_grad():
                    if use_sigmoid:
                        pred_classes = (torch.sigmoid(pred) > 0.5).long()
                        train_correct += (pred_classes == batch_y.long()).sum().item()
                    else:
                        pred_classes = torch.argmax(pred, dim=1)
                        train_correct += (pred_classes == batch_y.long()).sum().item()
                    train_total += batch_y.size(0)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = train_correct / train_total  # ✅ TRAIN ACCURACY HESAPLAMA
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)  # ✅ HISTORY'E EKLEME
            
            # 🔧 FIX: Proper validation phase with accuracy calculation
            val_loss = 0.0
            val_acc = 0.0
            
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    
                    # 🔧 FIXED: Proper validation accuracy calculation
                    if use_sigmoid:
                        if val_pred.dim() > 1:
                            val_pred = val_pred.squeeze(-1)
                        val_loss = criterion(val_pred, y_val.float()).item()
                        val_pred_classes = (torch.sigmoid(val_pred) > 0.5).long()
                        val_acc = (val_pred_classes == y_val.long()).float().mean().item()
                    else:
                        val_loss = criterion(val_pred, y_val.long()).item()
                        val_pred_classes = torch.argmax(val_pred, dim=1)
                        val_acc = (val_pred_classes == y_val.long()).float().mean().item()
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Learning rate scheduler step
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
                
                # ✅ ENHANCED LOGGING - TRAIN ACC EKLENDI
                if (epoch + 1) % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}, LR: {current_lr:.6f}")
            else:
                # No validation data - sadece train acc göster
                if (epoch + 1) % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, LR: {current_lr:.6f}")
        
        training_time = time.time() - start_time
        
        # Final results - TRAIN ACC EKLENDI
        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'final_train_acc': train_acc,  # ✅ EKLENEN
            'placeholder': False,
        }
        
        if X_val is not None:
            results['final_val_loss'] = val_loss
            results['final_val_acc'] = val_acc
        
        logger.info(f"   ✅ LSTM training completed in {training_time:.1f}s")
        if X_val is not None:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
        else:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}")
        
        return results
    
    def _train_transformer_real(self, model_name: str, model, X_train, y_train, X_val, y_val,
                            epochs, batch_size, learning_rate, weight_decay, patience):
        """Real Enhanced Transformer training with FIXED device handling"""
        
        logger.info(f"🏋️ {model_name} için ENHANCED_TRANSFORMER modeli eğitiliyor...")
        logger.info(f"   📊 Enhanced Transformer parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        # 🔧 CRITICAL FIX: Move tensors to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
        if y_val is not None:
            y_val = y_val.to(self.device)

        logger.info(f"   📍 Tensors moved to device: {self.device}")
        
        start_time = time.time()
        
        # 🔧 FIXED: Check model output to determine loss function
        with torch.no_grad():
            sample_output = model(X_train[:1])
            output_size = sample_output.shape[-1]
        
        if output_size == 1:
            criterion = nn.BCEWithLogitsLoss()
            use_sigmoid = True
            logger.info(f"   🔥 Using BCEWithLogitsLoss (single output)")
        else:
            criterion = nn.CrossEntropyLoss()
            use_sigmoid = False
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs)")
        
        # Optimizer and scheduler for transformer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # ✅ FIXED: Training history with train_acc
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        logger.info(f"   🚀 Starting Enhanced Transformer training: {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0  # ✅ TRAIN ACCURACY İÇİN EKLENEN
            train_total = 0    # ✅ TRAIN ACCURACY İÇİN EKLENEN
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(batch_X)
                
                # 🔧 FIXED: Proper target formatting
                if use_sigmoid:
                    if pred.dim() > 1:
                        pred = pred.squeeze(-1)
                    batch_y = batch_y.float()
                else:
                    batch_y = batch_y.long()
                
                loss = criterion(pred, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for transformer
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
                
                # ✅ TRAIN ACCURACY CALCULATION - TRANSFORMER İÇİN
                with torch.no_grad():
                    if use_sigmoid:
                        pred_classes = (torch.sigmoid(pred) > 0.5).long()
                        train_correct += (pred_classes == batch_y.long()).sum().item()
                    else:
                        pred_classes = torch.argmax(pred, dim=1)
                        train_correct += (pred_classes == batch_y.long()).sum().item()
                    train_total += batch_y.size(0)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = train_correct / train_total  # ✅ TRAIN ACCURACY HESAPLAMA
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)  # ✅ HISTORY'E EKLEME
            
            # Validation phase (same fix as LSTM)
            val_loss = 0.0
            val_acc = 0.0
            
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    
                    # 🔧 FIXED: Proper validation
                    if use_sigmoid:
                        if val_pred.dim() > 1:
                            val_pred = val_pred.squeeze(-1)
                        val_loss = criterion(val_pred, y_val.float()).item()
                        val_pred_classes = (torch.sigmoid(val_pred) > 0.5).long()
                        val_acc = (val_pred_classes == y_val.long()).float().mean().item()
                    else:
                        val_loss = criterion(val_pred, y_val.long()).item()
                        val_pred_classes = torch.argmax(val_pred, dim=1)
                        val_acc = (val_pred_classes == y_val.long()).float().mean().item()
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
                
                # ✅ ENHANCED LOGGING - TRAIN ACC EKLENDI
                if (epoch + 1) % 5 == 0:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}")
        
        training_time = time.time() - start_time
        
        # ✅ FIXED: Final results with train_acc
        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'final_train_acc': train_acc,  # ✅ EKLENEN
            'placeholder': False
        }
        
        if X_val is not None:
            results['final_val_loss'] = val_loss
            results['final_val_acc'] = val_acc
        
        logger.info(f"   ✅ Enhanced Transformer training completed in {training_time:.1f}s")
        if X_val is not None:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
        else:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}")
        
        return results
    def _train_hybrid_real(self, model_name: str, model, X_train, y_train,
                          X_val, y_val, epochs=30, **kwargs) -> Dict[str, Any]:
        """Gerçek Hybrid LSTM-Transformer eğitimi"""
        
        start_time = time.time()
        
        # Model parametrelerini logla
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Hybrid LSTM-Transformer parametreleri: {total_params:,}")
        
        # Hybrid-specific hyperparameters
        batch_size = kwargs.get('batch_size', 24)  # Smaller batch for large model
        learning_rate = kwargs.get('learning_rate', 0.0005)
        weight_decay = kwargs.get('weight_decay', 0.01)
        patience = kwargs.get('patience', 10)
        
        # Move tensors to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Advanced learning rate scheduler
        total_steps = len(X_train) // batch_size * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos'
        )
        
        # Binary classification loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        max_memory_used = 0.0
        
        logger.info(f"   🚀 Starting Hybrid LSTM-Transformer training: {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                pred = model(batch_X)
                
                # Handle output format
                if pred.dim() > 1:
                    pred = pred.squeeze()
                
                loss = criterion(pred, batch_y.float())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Step per batch for OneCycleLR
                
                epoch_train_loss += loss.item()
                
                # Track memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                    max_memory_used = max(max_memory_used, memory_used)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    
                    # Handle output format
                    if val_pred.dim() > 1:
                        val_pred = val_pred.squeeze()
                    
                    val_loss = criterion(val_pred, y_val.float()).item()
                    val_pred_classes = (torch.sigmoid(val_pred) > 0.5).long()
                    val_acc = (val_pred_classes == y_val.long()).float().mean().item()
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
                
                # Log progress
                if (epoch + 1) % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        training_time = time.time() - start_time
        
        # Final results
        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'max_memory_usage': max_memory_used,
            'placeholder': False,
            'enhanced_features': {
                'onecycle_lr': True,
                'gradient_clipping': True,
                'early_stopping': True,
                'memory_tracking': True
            }
        }
        
        if X_val is not None:
            results['final_val_loss'] = val_loss
            results['final_val_acc'] = val_acc
        
        logger.info(f"   ✅ Hybrid LSTM-Transformer training completed in {training_time:.1f}s")
        if max_memory_used > 0:
            logger.info(f"   💾 Max GPU memory used: {max_memory_used:.1f}GB")
        
        return results
    
    def _placeholder_training(self, model_name: str) -> Dict[str, Any]:
        """Placeholder training for unknown model types"""
        
        logger.warning(f"   ⚠️ Using placeholder training for {model_name}")
        
        return {
            'model_name': model_name,
            'model_type': self.model_type,
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
        
        if 'final_train_loss' in training_results:
            summary.append(f"   Final Train Loss: {training_results['final_train_loss']:.4f}")
        
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
    model_name='my_transformer',
    model=transformer_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=50,
    batch_size=32,
    learning_rate=0.0003
)

# Hybrid LSTM-Transformer training:
hybrid_trainer = UnifiedTrainer(device=torch.device('cuda'), model_type='hybrid_lstm_transformer')
hybrid_results = hybrid_trainer.train_model(
    model_name='my_hybrid',
    model=hybrid_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=40,
    batch_size=24,
    learning_rate=0.0005
)
"""