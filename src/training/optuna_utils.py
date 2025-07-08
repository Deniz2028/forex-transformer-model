"""Enhanced Optuna optimization utilities with expanded search space."""

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import yaml
import os
from typing import Dict, Any
from ..models.lstm import PairSpecificLSTM

class OptunaOptimizer:
    """Enhanced Optuna-based hyperparameter optimizer with expanded search space.
    
    Performs efficient hyperparameter search using Optuna's TPE sampler
    with early stopping capabilities and expanded parameter ranges.
    
    Args:
        pair_name: Name of the currency pair
        device: PyTorch device for training
    """
    
    def __init__(self, pair_name: str, device: torch.device):
        self.pair_name = pair_name
        self.device = device
        
    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, 
                 target_mode: str = 'binary') -> float:
        """Enhanced Optuna objective function with expanded search space.
        
        Args:
            trial: Optuna trial object
            X: Input features
            y: Target labels
            target_mode: Type of prediction task
            
        Returns:
            Validation accuracy to maximize
        """
        # Expanded search space
        horizon = trial.suggest_categorical('horizon', [32, 48, 64, 96])
        seq_len = trial.suggest_categorical('seq_len', [64, 96, 128])
        dropout = trial.suggest_float('dropout', 0.40, 0.70)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 96, 128])
        k_pips_mult = trial.suggest_categorical('k_pips_mult', [0.8, 1.0, 1.2, 1.5])
        
        # Additional hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.0005, 0.002, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        
        try:
            # Create model with expanded parameters
            model = PairSpecificLSTM(
                input_size=X.shape[2],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                target_mode=target_mode
            ).to(self.device)
            
            # Enhanced training split with validation
            total_size = len(X)
            train_size = int(total_size * 0.7)
            val_size = int(total_size * 0.15)
            test_size = total_size - train_size - val_size
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            
            # Setup loss and optimizer based on target mode
            if target_mode == 'binary':
                y_train = y_train.unsqueeze(1)
                y_val = y_val.unsqueeze(1)
                criterion = nn.BCELoss()
            else:  # three_class
                y_train = y_train.long()
                y_val = y_val.long()
                
                # Calculate class weights for imbalanced data
                unique_classes, counts = np.unique(y_train.cpu().numpy(), return_counts=True)
                total_samples = len(y_train)
                class_weights = torch.FloatTensor([total_samples / (len(unique_classes) * count) for count in counts]).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            
            # Enhanced training loop with early stopping
            best_val_acc = 0
            patience_counter = 0
            max_patience = 5
            
            epochs = 30  # Quick training for optimization
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    if target_mode == 'binary':
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        train_correct += (predicted == batch_y).sum().item()
                    else:
                        predicted = torch.argmax(outputs, dim=1)
                        train_correct += (predicted == batch_y).sum().item()
                
                train_accuracy = 100 * train_correct / len(X_train)
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_loss = 0
                
                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        batch_X = X_val[i:i+batch_size]
                        batch_y = y_val[i:i+batch_size]
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        if target_mode == 'binary':
                            predicted = (torch.sigmoid(outputs) > 0.5).float()
                            val_correct += (predicted == batch_y).sum().item()
                        else:
                            predicted = torch.argmax(outputs, dim=1)
                            val_correct += (predicted == batch_y).sum().item()
                
                val_accuracy = 100 * val_correct / len(X_val)
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Calculate overfitting gap
                train_val_gap = abs(train_accuracy - val_accuracy)
                
                # Enhanced early stopping criteria
                if (val_accuracy >= 0.62 and train_val_gap <= 0.15) or patience_counter >= max_patience:
                    if val_accuracy >= 0.62 and train_val_gap <= 0.15:
                        # Report intermediate values for Optuna pruning
                        trial.report(val_accuracy, epoch)
                        print(f"   🎯 Early success: val_acc={val_accuracy:.3f}, gap={train_val_gap:.3f}")
                    break
                
                # Optuna pruning
                trial.report(val_accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_acc
            
        except Exception as e:
            print(f"   ❌ Optuna trial failed: {e}")
            return 0.0
    
    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 40, 
                target_mode: str = 'binary') -> Dict[str, Any]:
        """Run enhanced Optuna optimization with expanded search space.
        
        Args:
            X: Input features
            y: Target labels
            n_trials: Number of trials to run
            target_mode: Type of prediction task
            
        Returns:
            Dictionary with best parameters and optimization history
        """
        print(f"🔍 {self.pair_name} Enhanced Optuna optimization ({n_trials} trials, {target_mode} mode)...")
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize with timeout protection
        try:
            study.optimize(
                lambda trial: self.objective(trial, X, y, target_mode), 
                n_trials=n_trials,
                timeout=1800  # 30 minutes timeout
            )
        except KeyboardInterrupt:
            print(f"   ⏹️ Optimization interrupted by user")
        
        best_params = study.best_params
        optimization_history = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'optimization_direction': 'maximize'
        }
        
        print(f"   ✅ Best params: {best_params}")
        print(f"   🎯 Best score: {study.best_value:.4f}")
        print(f"   📊 Completed trials: {len(study.trials)}/{n_trials}")
        
        # Enhanced parameter analysis
        if len(study.trials) > 5:
            try:
                # Get parameter importance
                importance = optuna.importance.get_param_importances(study)
                print(f"   📈 Parameter importance: {importance}")
                optimization_history['param_importance'] = importance
            except Exception as e:
                print(f"   ⚠️ Could not calculate parameter importance: {e}")
        
        # Save best parameters with metadata
        self._save_best_params(best_params, optimization_history, target_mode)
        
        return {**best_params, 'optimization_history': optimization_history}
    
    def _save_best_params(self, best_params: Dict[str, Any], 
                         optimization_history: Dict[str, Any],
                         target_mode: str) -> None:
        """Save best parameters with enhanced metadata.
        
        Args:
            best_params: Dictionary with best parameters
            optimization_history: Optimization metadata
            target_mode: Target mode used
        """
        try:
            os.makedirs('configs/optuna', exist_ok=True)
            save_path = f'configs/optuna/best_{self.pair_name}_{target_mode}.yaml'
            
            save_data = {
                'pair_name': self.pair_name,
                'target_mode': target_mode,
                'best_parameters': best_params,
                'optimization_history': optimization_history,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(save_path, 'w') as f:
                yaml.dump(save_data, f, default_flow_style=False, indent=2)
                
            print(f"   💾 Saved params to {save_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save params: {e}")
    
    def load_best_params(self, target_mode: str) -> Dict[str, Any]:
        """Load previously saved best parameters.
        
        Args:
            target_mode: Target mode to load parameters for
            
        Returns:
            Dictionary with best parameters or empty dict if not found
        """
        try:
            load_path = f'configs/optuna/best_{self.pair_name}_{target_mode}.yaml'
            
            if os.path.exists(load_path):
                with open(load_path, 'r') as f:
                    data = yaml.safe_load(f)
                print(f"   📁 Loaded cached params from {load_path}")
                return data.get('best_parameters', {})
            else:
                print(f"   ℹ️ No cached params found for {self.pair_name}_{target_mode}")
                return {}
        except Exception as e:
            print(f"   ⚠️ Failed to load cached params: {e}")
            return {}
    
    def get_optimization_study(self, X: np.ndarray, y: np.ndarray, 
                              target_mode: str = 'binary') -> optuna.Study:
        """Get optimization study for advanced analysis.
        
        Args:
            X: Input features
            y: Target labels
            target_mode: Target mode
            
        Returns:
            Optuna study object
        """
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        return study

def create_optimizer(pair_name: str, device: torch.device) -> OptunaOptimizer:
    """Factory function to create enhanced Optuna optimizer.
    
    Args:
        pair_name: Name of the currency pair
        device: PyTorch device
        
    Returns:
        Initialized optimizer instance
    """
    return OptunaOptimizer(pair_name, device)

__all__ = ['OptunaOptimizer', 'create_optimizer']
