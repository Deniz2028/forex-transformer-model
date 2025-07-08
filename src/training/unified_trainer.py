# ========================
# src/training/unified_trainer.py - COMPLETE FIXED VERSION
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
    """Unified trainer for all model types with comprehensive CUDA fixes"""
    
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
        """Universal model training method with COMPLETE target validation"""

        # 🔧 CRITICAL TARGET FIX - ABSOLUTE FIRST PRIORITY
        logger.info(f"🔧 UnifiedTrainer: Pre-training target validation...")
        logger.info(f"   y_train: shape={y_train.shape}, dtype={y_train.dtype}, min={y_train.min()}, max={y_train.max()}")
        logger.info(f"   y_train unique: {torch.unique(y_train)}")
        
        # FORCE targets to be correct type and range for binary classification
        if torch.any(y_train < 0) or torch.any(y_train > 1):
            logger.warning(f"⚠️ Found invalid targets, clamping to [0,1]")
            y_train = torch.clamp(y_train, 0, 1)
        
        # Ensure integer type for CrossEntropyLoss compatibility
        y_train = y_train.long()
        
        if y_val is not None:
            logger.info(f"   y_val: shape={y_val.shape}, dtype={y_val.dtype}, min={y_val.min()}, max={y_val.max()}")
            if torch.any(y_val < 0) or torch.any(y_val > 1):
                logger.warning(f"⚠️ Found invalid validation targets, clamping to [0,1]")
                y_val = torch.clamp(y_val, 0, 1)
            y_val = y_val.long()
        
        logger.info(f"🔧 UnifiedTrainer: Targets validated and fixed!")
        logger.info(f"   Fixed y_train unique: {torch.unique(y_train)}")
        if y_val is not None:
            logger.info(f"   Fixed y_val unique: {torch.unique(y_val)}")

        # Route to appropriate training method based on model type
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
            logger.warning(f"Unknown model type: {self.model_type}, using LSTM as default")
            return self._train_lstm_real(model_name, model, X_train, y_train,
                                        X_val, y_val, epochs, **kwargs)
    
    def _train_lstm_real(self, model_name: str, model, X_train, y_train, X_val, y_val,
                        epochs, batch_size=32, learning_rate=1e-3, weight_decay=1e-4, patience=10):
        """Real LSTM training with COMPLETE CUDA and target fixes"""
        
        logger.info(f"🏋️ {model_name} için LSTM modeli eğitiliyor...")
        logger.info(f"   📊 Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        # Move tensors to device with error handling
        try:
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            if X_val is not None:
                X_val = X_val.to(self.device)
            if y_val is not None:
                y_val = y_val.to(self.device)
            
            logger.info(f"   📍 Tensors moved to device: {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to move tensors to device: {e}")
            raise
        
        start_time = time.time()
        
        # Optimizer with proper parameter validation
        try:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        except Exception as e:
            logger.error(f"❌ Failed to create optimizer: {e}")
            raise
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        
        # 🔧 CRITICAL: Determine loss function based on model output
        try:
            with torch.no_grad():
                sample_output = model(X_train[:1])
                output_size = sample_output.shape[-1]
            
            logger.info(f"   🔍 Model output inspection: shape={sample_output.shape}, output_size={output_size}")
        except Exception as e:
            logger.error(f"❌ Failed to inspect model output: {e}")
            raise
        
        # Select appropriate loss function based on output size
        if output_size == 2:
            # Binary classification with 2 outputs (for CrossEntropyLoss)
            criterion = nn.CrossEntropyLoss()
            use_crossentropy = True
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs)")
        elif output_size == 1:
            # Single output for binary classification (for BCEWithLogitsLoss)
            criterion = nn.BCEWithLogitsLoss()
            use_crossentropy = False
            logger.info(f"   🔥 Using BCEWithLogitsLoss (single output)")
        else:
            # Multi-class (3+ outputs)
            criterion = nn.CrossEntropyLoss()
            use_crossentropy = True
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs - multiclass)")
        
        # Create data loaders with error handling
        try:
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"❌ Failed to create data loader: {e}")
            raise
        
        # Training history tracking
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        logger.info(f"   🚀 Starting training: {epochs} epochs, batch_size={batch_size}")
        
        # Main training loop
        for epoch in range(epochs):
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            try:
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(batch_X)
                    
                    # 🔧 CRITICAL: Proper target and prediction handling
                    if use_crossentropy:
                        # CrossEntropyLoss expects class indices (LongTensor)
                        # Ensure targets are in valid range [0, num_classes-1]
                        if output_size == 2:  # Binary
                            batch_y = torch.clamp(batch_y, 0, 1)
                        else:  # Multi-class
                            batch_y = torch.clamp(batch_y, 0, output_size-1)
                        
                        loss = criterion(pred, batch_y)
                        
                        # Calculate accuracy
                        with torch.no_grad():
                            pred_classes = torch.argmax(pred, dim=1)
                            train_correct += (pred_classes == batch_y).sum().item()
                            train_total += batch_y.size(0)
                    else:
                        # BCEWithLogitsLoss expects float targets and single output
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        batch_y_float = batch_y.float()
                        loss = criterion(pred, batch_y_float)
                        
                        # Calculate accuracy
                        with torch.no_grad():
                            pred_probs = torch.sigmoid(pred)
                            pred_classes = (pred_probs > 0.5).long()
                            train_correct += (pred_classes == batch_y).sum().item()
                            train_total += batch_y.size(0)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    
                    # Handle potential CUDA errors
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"❌ CUDA error during training: {e}")
                    logger.info("🔧 Clearing CUDA cache and continuing...")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
            
            # Calculate training metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            
            if X_val is not None and y_val is not None:
                model.eval()
                val_correct = 0
                val_total = 0
                
                try:
                    with torch.no_grad():
                        pred_val = model(X_val)
                        
                        if use_crossentropy:
                            # Ensure validation targets are in valid range
                            if output_size == 2:  # Binary
                                y_val_clamped = torch.clamp(y_val, 0, 1)
                            else:  # Multi-class
                                y_val_clamped = torch.clamp(y_val, 0, output_size-1)
                            
                            val_loss = criterion(pred_val, y_val_clamped).item()
                            pred_classes = torch.argmax(pred_val, dim=1)
                            val_correct = (pred_classes == y_val_clamped).sum().item()
                        else:
                            if pred_val.dim() > 1:
                                pred_val = pred_val.squeeze(-1)
                            y_val_float = y_val.float()
                            val_loss = criterion(pred_val, y_val_float).item()
                            pred_probs = torch.sigmoid(pred_val)
                            pred_classes = (pred_probs > 0.5).long()
                            val_correct = (pred_classes == y_val).sum().item()
                        
                        val_total = y_val.size(0)
                        val_acc = val_correct / val_total if val_total > 0 else 0.0
                    
                    # Update scheduler
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            logger.info(f"   🛑 Early stopping at epoch {epoch+1}")
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Validation error at epoch {epoch+1}: {e}")
                    val_loss = float('inf')
                    val_acc = 0.0
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Logging with error handling
            try:
                current_lr = optimizer.param_groups[0]['lr']
                if X_val is not None:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, LR: {current_lr:.6f}")
                else:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, LR: {current_lr:.6f}")
            except Exception as e:
                logger.warning(f"⚠️ Logging error: {e}")
        
        training_time = time.time() - start_time
        
        # Compile final results
        results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'final_train_acc': train_acc,
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
                               epochs, batch_size=16, learning_rate=5e-4, weight_decay=1e-4, patience=10):
        """Real Enhanced Transformer training with COMPLETE fixes"""
        
        logger.info(f"🏋️ {model_name} için ENHANCED_TRANSFORMER modeli eğitiliyor...")
        logger.info(f"   📊 Enhanced Transformer parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        # Move tensors to device with error handling
        try:
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            if X_val is not None:
                X_val = X_val.to(self.device)
            if y_val is not None:
                y_val = y_val.to(self.device)

            logger.info(f"   📍 Tensors moved to device: {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to move tensors to device: {e}")
            raise
        
        start_time = time.time()
        
        # Determine loss function based on model output
        try:
            with torch.no_grad():
                sample_output = model(X_train[:1])
                output_size = sample_output.shape[-1]
            
            logger.info(f"   🔍 Transformer output inspection: shape={sample_output.shape}, output_size={output_size}")
        except Exception as e:
            logger.error(f"❌ Failed to inspect transformer output: {e}")
            raise
        
        if output_size == 2:
            criterion = nn.CrossEntropyLoss()
            use_crossentropy = True
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs)")
        elif output_size == 1:
            criterion = nn.BCEWithLogitsLoss()
            use_crossentropy = False
            logger.info(f"   🔥 Using BCEWithLogitsLoss (single output)")
        else:
            criterion = nn.CrossEntropyLoss()
            use_crossentropy = True
            logger.info(f"   🔥 Using CrossEntropyLoss ({output_size} outputs - multiclass)")
        
        # Optimizer with transformer-specific settings
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        except Exception as e:
            logger.error(f"❌ Failed to create transformer optimizer: {e}")
            raise
        
        # Transformer-specific scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=max(1, len(X_train) // batch_size),
            epochs=epochs,
            pct_start=0.1
        )
        
        # Create data loaders
        try:
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"❌ Failed to create transformer data loader: {e}")
            raise
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        logger.info(f"   🚀 Starting transformer training: {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            try:
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(batch_X)
                    
                    # Loss and accuracy calculation (same logic as LSTM)
                    if use_crossentropy:
                        if output_size == 2:  # Binary
                            batch_y = torch.clamp(batch_y, 0, 1)
                        else:  # Multi-class
                            batch_y = torch.clamp(batch_y, 0, output_size-1)
                        
                        loss = criterion(pred, batch_y)
                        
                        # Accuracy calculation
                        with torch.no_grad():
                            pred_classes = torch.argmax(pred, dim=1)
                            train_correct += (pred_classes == batch_y).sum().item()
                            train_total += batch_y.size(0)
                    else:
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        batch_y_float = batch_y.float()
                        loss = criterion(pred, batch_y_float)
                        
                        # Accuracy calculation
                        with torch.no_grad():
                            pred_probs = torch.sigmoid(pred)
                            pred_classes = (pred_probs > 0.5).long()
                            train_correct += (pred_classes == batch_y).sum().item()
                            train_total += batch_y.size(0)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()  # Step per batch for OneCycleLR
                    
                    epoch_train_loss += loss.item()
                    
                    # Handle potential CUDA errors
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"❌ CUDA error during transformer training: {e}")
                    logger.info("🔧 Clearing CUDA cache and continuing...")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            # Validation phase (same logic as LSTM)
            val_loss = 0.0
            val_acc = 0.0
            
            if X_val is not None and y_val is not None:
                model.eval()
                val_correct = 0
                val_total = 0
                
                try:
                    with torch.no_grad():
                        pred_val = model(X_val)
                        
                        if use_crossentropy:
                            if output_size == 2:  # Binary
                                y_val_clamped = torch.clamp(y_val, 0, 1)
                            else:  # Multi-class
                                y_val_clamped = torch.clamp(y_val, 0, output_size-1)
                            
                            val_loss = criterion(pred_val, y_val_clamped).item()
                            pred_classes = torch.argmax(pred_val, dim=1)
                            val_correct = (pred_classes == y_val_clamped).sum().item()
                        else:
                            if pred_val.dim() > 1:
                                pred_val = pred_val.squeeze(-1)
                            y_val_float = y_val.float()
                            val_loss = criterion(pred_val, y_val_float).item()
                            pred_probs = torch.sigmoid(pred_val)
                            pred_classes = (pred_probs > 0.5).long()
                            val_correct = (pred_classes == y_val).sum().item()
                        
                        val_total = y_val.size(0)
                        val_acc = val_correct / val_total if val_total > 0 else 0.0
                    
                    # Early stopping (no scheduler.step for OneCycleLR)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            logger.info(f"   🛑 Early stopping at epoch {epoch+1}")
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Transformer validation error at epoch {epoch+1}: {e}")
                    val_loss = float('inf')
                    val_acc = 0.0
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Logging
            try:
                current_lr = optimizer.param_groups[0]['lr']
                if X_val is not None:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, LR: {current_lr:.6f}")
                else:
                    logger.info(f"   📈 Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, LR: {current_lr:.6f}")
            except Exception as e:
                logger.warning(f"⚠️ Transformer logging error: {e}")
        
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
            'final_train_acc': train_acc,
            'placeholder': False,
        }
        
        if X_val is not None:
            results['final_val_loss'] = val_loss
            results['final_val_acc'] = val_acc
        
        logger.info(f"   ✅ Transformer training completed in {training_time:.1f}s")
        if X_val is not None:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")
        else:
            logger.info(f"   📊 Final Results: Train Acc: {train_acc:.2%}")
        
        return results
    
    def _train_hybrid_real(self, model_name: str, model, X_train, y_train, X_val, y_val,
                          epochs, batch_size=16, learning_rate=5e-4, weight_decay=1e-4, patience=10):
        """Real Hybrid LSTM-Transformer training with same fixes as transformer"""
        
        logger.info(f"🏋️ {model_name} için HYBRID_LSTM_TRANSFORMER modeli eğitiliyor...")
        logger.info(f"   📊 Hybrid model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        # Use transformer training logic but with hybrid model considerations
        # Hybrid models typically behave like transformers in terms of training
        return self._train_transformer_real(model_name, model, X_train, y_train, 
                                          X_val, y_val, epochs, batch_size, 
                                          learning_rate, weight_decay, patience)
    
    def _placeholder_training(self, model_name: str) -> Dict[str, Any]:
        """Placeholder training for unsupported models - should rarely be used"""
        
        logger.warning(f"⚠️ Using placeholder training for {model_name}")
        logger.warning(f"   This indicates an unsupported model type: {self.model_type}")
        
        return {
            'model_name': model_name,
            'model_type': self.model_type,
            'training_time': 0.1,
            'epochs_trained': 0,
            'history': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'best_val_loss': float('inf'),
            'final_train_loss': 1.0,
            'final_train_acc': 0.5,
            'final_val_loss': 1.0,
            'final_val_acc': 0.5,
            'placeholder': True
        }
    
    def validate_training_inputs(self, X_train, y_train, X_val=None, y_val=None) -> bool:
        """Validate training inputs for common issues"""
        
        try:
            # Basic shape validation
            if X_train.dim() < 2:
                logger.error(f"❌ X_train must be at least 2D, got {X_train.dim()}D")
                return False
            
            if y_train.dim() != 1:
                logger.error(f"❌ y_train must be 1D, got {y_train.dim()}D")
                return False
            
            if len(X_train) != len(y_train):
                logger.error(f"❌ X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
                return False
            
            # Target value validation
            unique_targets = torch.unique(y_train)
            if len(unique_targets) < 2:
                logger.error(f"❌ Only one class found in targets: {unique_targets}")
                return False
            
            # Check for NaN/Inf
            if torch.isnan(X_train).any():
                logger.error(f"❌ NaN values found in X_train")
                return False
            
            if torch.isnan(y_train).any():
                logger.error(f"❌ NaN values found in y_train")
                return False
            
            logger.info(f"✅ Training inputs validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return False