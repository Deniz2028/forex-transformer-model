"""Enhanced training module with LR schedulers and overfitting detection."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from ..models.lstm import PairSpecificLSTM
from ..models.losses import get_loss_function

class MultiPairTrainer:
    """Enhanced multi-pair LSTM trainer with advanced features.
    
    Handles training of pair-specific LSTM models with features like:
    - Focal loss for class imbalance
    - Enhanced LR schedulers (ReduceLROnPlateau)
    - Early stopping with min_delta
    - Overfitting detection and dropout adjustment
    - Three-class mode support
    
    Args:
        device: PyTorch device for training
        use_focal_loss: Whether to use focal loss
        target_mode: Type of prediction ('binary' or 'three_class')
    """
    
    def __init__(self, device: torch.device, use_focal_loss: bool = True, 
                 target_mode: str = 'binary'):
        self.device = device
        self.use_focal_loss = use_focal_loss
        self.target_mode = target_mode
        self.pair_models = {}
        self.pair_histories = {}
        
    def train_pair_model(self, pair_name: str, X: np.ndarray, y: np.ndarray, 
                        epochs: int = 120, batch_size: int = 128, dropout: float = 0.45, 
                        dropout_upper: float = 0.70, learning_rate: float = 0.0008,
                        hidden_size: int = 96, num_layers: int = 2) -> Tuple[PairSpecificLSTM, Dict]:
        """Train a pair-specific LSTM model with enhanced features.
        
        Args:
            pair_name: Name of the currency pair
            X: Input sequences
            y: Target labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout: Initial dropout rate
            dropout_upper: Upper limit for dropout sweep
            learning_rate: Learning rate
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        print(f"\n🚀 {pair_name} Enhanced Model Training ({self.target_mode})...")
        
        # Enhanced regularization for specific pairs
        if pair_name in ['EUR_USD', 'AUD_USD'] and dropout == 0.45:
            dropout = 0.55
            print(f"   🛡️ Enhanced regularization for {pair_name}: dropout={dropout}")
        
        # Create model
        input_size = X.shape[2]
        model = PairSpecificLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
            target_mode=self.target_mode
        ).to(self.device)
        
        model_info = model.get_model_info()
        print(f"   🤖 {pair_name} Model: {model_info['total_parameters']:,} params")
        
        # Test Pipeline: 70/15/15 chronological split
        total_size = len(X)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size
        
        # Sequential split - respects time order (CRITICAL!)
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        print(f"   📊 Test Pipeline Split:")
        print(f"      Train: {len(X_train):,} ({len(X_train)/total_size*100:.1f}%)")
        print(f"      Val:   {len(X_val):,} ({len(X_val)/total_size*100:.1f}%)")
        print(f"      Test:  {len(X_test):,} ({len(X_test)/total_size*100:.1f}%)")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        
        if self.target_mode == 'binary':
            y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            y_test = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        else:
            y_train = torch.LongTensor(y_train).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
            y_test = torch.LongTensor(y_test).to(self.device)
        
        # Setup enhanced loss function
        criterion = self._get_enhanced_criterion(y_train, pair_name)
        
        # Enhanced optimizer and schedulers
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # 1. ReduceLROnPlateau (patience 2)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5, verbose=True
        )
        
        # Training state with enhanced early stopping
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping patience
        min_delta = 0.002  # Minimum improvement threshold
        
        history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'learning_rates': [], 'overfitting_warnings': []
        }
        
        # Overfitting detection and dropout adjustment
        overfitting_adjustments = 0
        max_overfitting_adjustments = 2
        overfitting_threshold = 0.15
        consecutive_overfitting = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, X_train, y_train, optimizer, criterion, batch_size
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                model, X_val, y_val, criterion, batch_size
            )
            
            # LR Scheduler step
            lr_scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Overfitting detection and dropout adjustment
            overfitting_gap = (train_acc - val_acc) / 100.0
            
            if overfitting_gap > overfitting_threshold:
                consecutive_overfitting += 1
                if consecutive_overfitting >= 3 and overfitting_adjustments < max_overfitting_adjustments:
                    print(f"   🚨 Overfitting detected! Gap: {overfitting_gap:.3f}")
                    print(f"   🔧 Increasing dropout by 0.05 and restarting model...")
                    
                    # Increase dropout and restart
                    new_dropout = min(dropout + 0.05, dropout_upper)
                    if new_dropout != dropout:
                        dropout = new_dropout
                        overfitting_adjustments += 1
                        
                        # Create new model with higher dropout
                        model = PairSpecificLSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            use_layer_norm=True,
                            target_mode=self.target_mode
                        ).to(self.device)
                        
                        # Reset optimizer and scheduler
                        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
                        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', patience=2, factor=0.5
                        )
                        
                        # Reset training state
                        best_val_acc = 0
                        best_val_loss = float('inf')
                        patience_counter = 0
                        consecutive_overfitting = 0
                        
                        history['overfitting_warnings'].append({
                            'epoch': epoch,
                            'gap': overfitting_gap,
                            'new_dropout': dropout,
                            'action': 'model_restart'
                        })
                        
                        print(f"   🔄 Model restarted with dropout={dropout}")
                        continue
            else:
                consecutive_overfitting = 0
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            # Enhanced early stopping with min_delta
            val_improvement = val_acc - best_val_acc
            
            if val_improvement > min_delta:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{pair_name}_best_model.pth')
            else:
                patience_counter += 1
            
            # Progress logging
            if (epoch + 1) % 20 == 0 or epoch < 5:
                print(f"   Epoch [{epoch+1:3d}/{epochs}] "
                      f"Loss: {train_loss:.4f}→{val_loss:.4f} | "
                      f"Acc: {train_acc:.1f}%→{val_acc:.1f}% | "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"   ⏰ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
            
            # Learning rate minimum threshold
            if current_lr < 1e-6:
                print(f"   🔚 Stopping: Learning rate too low ({current_lr:.2e})")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'{pair_name}_best_model.pth'))
        
        # Store results
        self.pair_models[pair_name] = model
        self.pair_histories[pair_name] = history
        
        # Final statistics
        final_train_acc = history['train_acc'][-1]
        final_overfitting_gap = final_train_acc - best_val_acc
        
        print(f"   ✅ {pair_name} Training completed!")
        print(f"   📊 Best Val Acc: {best_val_acc:.2f}% | Final Train: {final_train_acc:.2f}%")
        print(f"   📈 Overfitting Gap: {final_overfitting_gap:.2f}pp | Adjustments: {overfitting_adjustments}")
        
        return model, history
    
    def _get_enhanced_criterion(self, y_train: torch.Tensor, pair_name: str) -> nn.Module:
        """Get enhanced loss criterion with fixed class weight handling."""
        if self.use_focal_loss and self.target_mode == 'binary':
            from ..models.losses import FocalLoss
            criterion = FocalLoss(alpha=1, gamma=2)
            print(f"   🎯 Using Focal Loss for {pair_name}")
            
        elif self.target_mode == 'binary':
            # Binary classification with class weights
            y_flat = y_train.cpu().numpy().astype(int).flatten()
            class_counts = np.bincount(y_flat)
            
            if len(class_counts) > 1 and class_counts[1] > 0:
                pos_weight = torch.FloatTensor([class_counts[0] / class_counts[1]]).to(self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"   ⚖️ Binary BCE with pos_weight: {pos_weight.item():.3f}")
            else:
                criterion = nn.BCELoss()
                print(f"   ⚠️ Single class detected - using regular BCE")
                
        else:
            # Three-class mode with fixed class weight handling
            y_flat = y_train.cpu().numpy().astype(int).flatten()
            unique_classes = np.unique(y_flat)
            n_classes = 3  # Expected number of classes
            
            # Check if we have all expected classes
            if len(unique_classes) < n_classes:
                print(f"   ⚠️ Missing classes detected: {unique_classes} (expected: 0,1,2)")
                # Use unweighted CrossEntropy - let model handle imbalance
                criterion = nn.CrossEntropyLoss()
                print(f"   ⚖️ Using unweighted CrossEntropy (missing classes)")
            else:
                # Calculate proper class weights
                class_counts = np.bincount(y_flat, minlength=n_classes)
                total_samples = len(y_flat)
                
                # Compute balanced class weights
                class_weights = torch.FloatTensor([
                    total_samples / (n_classes * count) if count > 0 else 1.0 
                    for count in class_counts
                ]).to(self.device)
                
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                
                # Log class distribution
                class_names = {0: 'Long', 1: 'Flat', 2: 'Short'}
                class_info = ', '.join([f"{class_names.get(i, i)}: {count}" 
                                      for i, count in enumerate(class_counts)])
                print(f"   ⚖️ Weighted CrossEntropy: {class_info}")
                print(f"   📊 Class weights: {class_weights.cpu().numpy()}")
        
        return criterion
    
    def _train_epoch(self, model: PairSpecificLSTM, X_train: torch.Tensor, 
                    y_train: torch.Tensor, optimizer: optim.Optimizer, 
                    criterion: nn.Module, batch_size: int) -> Tuple[float, float]:
        """Execute one training epoch with enhanced features."""
        model.train()
        train_loss = 0
        train_correct = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate loss and predictions based on target mode
            if self.target_mode == 'binary':
                if self.use_focal_loss:
                    loss = criterion(outputs, batch_y)
                    predicted = (outputs > 0.5).float()
                else:
                    loss = criterion(outputs, batch_y)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                loss = criterion(outputs, batch_y)
                predicted = torch.argmax(outputs, dim=1).unsqueeze(1).float()
                batch_y = batch_y.unsqueeze(1).float()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (predicted == batch_y).sum().item()
            num_batches += 1
        
        train_loss /= num_batches
        train_acc = 100 * train_correct / len(X_train)
        
        return train_loss, train_acc
    
    def _validate_epoch(self, model: PairSpecificLSTM, X_val: torch.Tensor, 
                       y_val: torch.Tensor, criterion: nn.Module, 
                       batch_size: int) -> Tuple[float, float]:
        """Execute one validation epoch with enhanced metrics."""
        model.eval()
        val_loss = 0
        val_correct = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                outputs = model(batch_X)
                
                # Calculate loss and predictions based on target mode
                if self.target_mode == 'binary':
                    if self.use_focal_loss:
                        loss = criterion(outputs, batch_y)
                        predicted = (outputs > 0.5).float()
                    else:
                        loss = criterion(outputs, batch_y)
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    loss = criterion(outputs, batch_y)
                    predicted = torch.argmax(outputs, dim=1).unsqueeze(1).float()
                    batch_y = batch_y.unsqueeze(1).float()
                
                val_loss += loss.item()
                val_correct += (predicted == batch_y).sum().item()
                val_batches += 1
        
        val_loss /= val_batches
        val_acc = 100 * val_correct / len(X_val)
        
        return val_loss, val_acc
    
    def get_model(self, pair_name: str) -> PairSpecificLSTM:
        """Get trained model for a specific pair."""
        return self.pair_models.get(pair_name)
    
    def get_history(self, pair_name: str) -> Dict:
        """Get training history for a specific pair."""
        return self.pair_histories.get(pair_name)
    
    def get_all_models(self) -> Dict[str, PairSpecificLSTM]:
        """Get all trained models."""
        return self.pair_models.copy()
    
    def get_all_histories(self) -> Dict[str, Dict]:
        """Get all training histories."""
        return self.pair_histories.copy()
    
    def save_training_summary(self, filepath: str) -> None:
        """Save comprehensive training summary."""
        import json
        from datetime import datetime
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'target_mode': self.target_mode,
            'use_focal_loss': self.use_focal_loss,
            'trained_pairs': list(self.pair_models.keys()),
            'pair_summaries': {}
        }
        
        for pair_name, history in self.pair_histories.items():
            if history:
                summary['pair_summaries'][pair_name] = {
                    'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0,
                    'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
                    'epochs_trained': len(history['train_acc']),
                    'overfitting_adjustments': len(history.get('overfitting_warnings', [])),
                    'final_lr': history['learning_rates'][-1] if history.get('learning_rates') else 0
                }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Training summary saved to: {filepath}")

def create_trainer(config: dict, device: torch.device) -> MultiPairTrainer:
    """Factory function to create enhanced trainer from configuration.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Initialized trainer instance
    """
    model_config = config.get('model', {})
    
    return MultiPairTrainer(
        device=device,
        use_focal_loss=model_config.get('use_focal_loss', True),
        target_mode=model_config.get('target_mode', 'binary')
    )

__all__ = ['MultiPairTrainer', 'create_trainer']
