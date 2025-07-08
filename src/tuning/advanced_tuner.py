# src/tuning/advanced_tuner.py
"""Advanced hyperparameter tuning with Optuna for LSTM and Enhanced Transformer models."""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import time
import json
from pathlib import Path

from ..data.preprocess import create_preprocessor
from ..models.factory import create_model
from ..models.lstm import PairSpecificLSTM


class AdvancedTuner:
    """Advanced hyperparameter tuner with support for multiple model types."""
    
    def __init__(self, model_type: str, device: torch.device, config: Dict[str, Any],
                 optuna_trials: int = 50, timeout: int = 3600, 
                 cross_validate: bool = False, tune_architecture: bool = False,
                 tune_preprocessing: bool = False):
        """
        Initialize Advanced Tuner.
        
        Args:
            model_type: Type of model ('lstm', 'enhanced_transformer')
            device: PyTorch device
            config: System configuration
            optuna_trials: Number of optimization trials
            timeout: Maximum tuning time in seconds
            cross_validate: Use cross-validation
            tune_architecture: Tune model architecture
            tune_preprocessing: Tune preprocessing parameters
        """
        self.model_type = model_type
        self.device = device
        self.config = config
        self.optuna_trials = optuna_trials
        self.timeout = timeout
        self.cross_validate = cross_validate
        self.tune_architecture = tune_architecture
        self.tune_preprocessing = tune_preprocessing
        
        # Create output directory
        self.output_dir = Path("tune_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎛️ Advanced Tuner initialized:")
        print(f"   🤖 Model type: {model_type}")
        print(f"   🔬 Trials: {optuna_trials}")
        print(f"   ⏰ Timeout: {timeout}s")
        print(f"   ✅ Cross-validation: {cross_validate}")
        print(f"   🏗️ Architecture tuning: {tune_architecture}")
        print(f"   🔧 Preprocessing tuning: {tune_preprocessing}")
    
    def tune_pair(self, pair_name: str, data: pd.DataFrame, 
                  target_mode: str = 'binary', csv_logger: str = None) -> Tuple[Dict, Dict, Dict]:
        """
        Tune hyperparameters for a specific currency pair.
        
        Args:
            pair_name: Currency pair name
            data: Market data
            target_mode: Target mode ('binary' or 'three_class')
            csv_logger: Path to CSV log file
            
        Returns:
            Tuple of (best_params, study_results, tuning_metrics)
        """
        print(f"🎛️ Starting advanced tuning for {pair_name}...")
        
        start_time = time.time()
        
        # Create preprocessor
        preprocessor = create_preprocessor(pair_name, self.config)
        preprocessor.target_mode = target_mode
        
        if target_mode == 'three_class':
            preprocessor.use_smote = True
        
        # Prepare base data
        features, target = preprocessor.prepare_pair_data(data)
        
        if len(features) < 1000:
            raise ValueError(f"Insufficient data: {len(features)} samples")
        
        X, y = preprocessor.create_sequences(features, target)
        
        if len(X) < 200:
            raise ValueError(f"Insufficient sequences: {len(X)} sequences")
        
        # Create Optuna study
        study_name = f"{pair_name}_{self.model_type}_{target_mode}"
        storage_url = f"sqlite:///tune_results/optuna_studies.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        )
        
        # Store data for objective function
        self.current_data = (X, y, target_mode, pair_name, csv_logger, preprocessor)
        
        try:
            # Run optimization
            study.optimize(
                self.objective_function,
                n_trials=self.optuna_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            # Extract results
            best_params = study.best_params
            best_value = study.best_value
            
            # Compile study results
            study_results = {
                'best_params': best_params,
                'best_value': best_value,
                'trials': [
                    {
                        'number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': trial.state.name,
                        'duration': trial.duration.total_seconds() if trial.duration else None
                    }
                    for trial in study.trials
                ],
                'study_name': study_name
            }
            
            # Calculate tuning metrics
            total_time = time.time() - start_time
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            
            tuning_metrics = {
                'best_val_acc': best_value,
                'total_time': total_time,
                'completed_trials': completed_trials,
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'trials_per_minute': completed_trials / (total_time / 60) if total_time > 0 else 0
            }
            
            print(f"✅ Tuning completed for {pair_name}:")
            print(f"   🏆 Best validation accuracy: {best_value:.2f}%")
            print(f"   ⏱️ Total time: {total_time:.1f}s")
            print(f"   📊 Completed trials: {completed_trials}/{self.optuna_trials}")
            
            # Save detailed results
            self._save_study_results(pair_name, study_results, tuning_metrics)
            
            return best_params, study_results, tuning_metrics
            
        except Exception as e:
            print(f"❌ Tuning failed for {pair_name}: {str(e)}")
            raise e
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy to maximize
        """
        X, y, target_mode, pair_name, csv_logger, preprocessor = self.current_data
        
        try:
            # Suggest hyperparameters based on model type
            if self.model_type == 'enhanced_transformer':
                params = self._suggest_transformer_params(trial)
            else:  # lstm
                params = self._suggest_lstm_params(trial)
            
            # Suggest common training parameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
            
            params.update({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate
            })
            
            # Suggest preprocessing parameters if enabled
            if self.tune_preprocessing:
                seq_len = trial.suggest_categorical('seq_len', [32, 64, 96, 128])
                horizon = trial.suggest_categorical('horizon', [32, 48, 64, 96])
                params.update({
                    'seq_len': seq_len,
                    'horizon': horizon
                })
                
                # Re-prepare data with new parameters
                features, target_new = preprocessor.prepare_pair_data(
                    preprocessor.original_data,  # Need to store this
                    horizon=horizon
                )
                X_new, y_new = preprocessor.create_sequences(features, target_new, seq_len=seq_len)
                X, y = X_new, y_new
            
            # Train and evaluate model
            val_acc = self._train_and_evaluate(X, y, params, target_mode, trial)
            
            # Log to CSV if provided
            if csv_logger:
                self._log_trial_to_csv(csv_logger, pair_name, trial.number, params, val_acc)
            
            return val_acc
            
        except Exception as e:
            print(f"   ❌ Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    def _suggest_transformer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Enhanced Transformer specific parameters."""
        
        # Architecture parameters
        if self.tune_architecture:
            d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8, 16])
            num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 6, 8])
            ff_dim = trial.suggest_categorical('ff_dim', [128, 256, 512, 1024])
            
            # Ensure d_model is divisible by nhead
            while d_model % nhead != 0:
                valid_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
                if valid_heads:
                    nhead = trial.suggest_categorical(f'nhead_fixed_{d_model}', valid_heads)
                    break
                else:
                    d_model = trial.suggest_categorical('d_model_retry', [64, 128, 256])
        else:
            # Use default architecture
            d_model = 128
            nhead = 8
            num_layers = 4
            ff_dim = 256
        
        return {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'ff_dim': ff_dim
        }
    
    def _suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LSTM specific parameters."""
        
        if self.tune_architecture:
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 96, 128, 256])
            num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        else:
            hidden_size = 64
            num_layers = 2
        
        return {
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }
    
    def _train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          params: Dict[str, Any], target_mode: str,
                          trial: optuna.Trial) -> float:
        """
        Train and evaluate model with given parameters.
        
        Args:
            X: Input sequences
            y: Target labels
            params: Hyperparameters
            target_mode: Target mode
            trial: Optuna trial for pruning
            
        Returns:
            Validation accuracy
        """
        try:
            # Prepare data tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            if target_mode == 'binary':
                y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
                criterion = nn.BCEWithLogitsLoss()
            else:  # three_class
                y_tensor = torch.LongTensor(y).to(self.device)
                criterion = nn.CrossEntropyLoss()
            
            # Train/validation split (time-based)
            if self.cross_validate:
                # Use cross-validation
                return self._cross_validate(X_tensor, y_tensor, params, target_mode, criterion, trial)
            else:
                # Simple train/val split
                train_size = int(0.8 * len(X_tensor))
                X_train = X_tensor[:train_size]
                X_val = X_tensor[train_size:]
                y_train = y_tensor[:train_size]
                y_val = y_tensor[train_size:]
                
                return self._single_train_eval(X_train, X_val, y_train, y_val, params, target_mode, criterion, trial)
                
        except Exception as e:
            print(f"   ⚠️ Training error: {str(e)}")
            return 0.0
    
    def _single_train_eval(self, X_train: torch.Tensor, X_val: torch.Tensor,
                          y_train: torch.Tensor, y_val: torch.Tensor,
                          params: Dict[str, Any], target_mode: str,
                          criterion: nn.Module, trial: optuna.Trial) -> float:
        """Single training and evaluation."""
        
        # Create model
        model = self._create_model(params, X_train.shape[2], target_mode)
        
        # Create data loaders
        batch_size = params['batch_size']
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.7)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        patience = 5
        epochs = 25  # Reduced for tuning speed
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                if target_mode == 'binary':
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_correct += (predicted == batch_y).sum().item()
                else:
                    predicted = torch.argmax(outputs, dim=1)
                    train_correct += (predicted == batch_y).sum().item()
                
                train_total += batch_y.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    if target_mode == 'binary':
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        val_correct += (predicted == batch_y).sum().item()
                    else:
                        predicted = torch.argmax(outputs, dim=1)
                        val_correct += (predicted == batch_y).sum().item()
                    
                    val_total += batch_y.size(0)
            
            # Calculate metrics
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Scheduler step
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Optuna pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            if patience_counter >= patience:
                break
        
        return best_val_acc
    
    def _cross_validate(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
                       params: Dict[str, Any], target_mode: str,
                       criterion: nn.Module, trial: optuna.Trial) -> float:
        """Cross-validation training and evaluation."""
        
        n_folds = 3  # Reduced for speed
        fold_size = len(X_tensor) // n_folds
        val_accs = []
        
        for fold in range(n_folds):
            # Create fold splits (time-aware)
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(X_tensor)
            
            # Validation set
            X_val = X_tensor[val_start:val_end]
            y_val = y_tensor[val_start:val_end]
            
            # Training set (everything except validation)
            X_train = torch.cat([X_tensor[:val_start], X_tensor[val_end:]], dim=0)
            y_train = torch.cat([y_tensor[:val_start], y_tensor[val_end:]], dim=0)
            
            # Train and evaluate
            fold_val_acc = self._single_train_eval(X_train, X_val, y_train, y_val, 
                                                  params, target_mode, criterion, trial)
            val_accs.append(fold_val_acc)
            
            # Report intermediate result
            trial.report(np.mean(val_accs), fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(val_accs)
    
    def _create_model(self, params: Dict[str, Any], input_size: int, target_mode: str) -> nn.Module:
        """Create model based on parameters and model type."""
        
        if self.model_type == 'enhanced_transformer':
            # Update config with trial parameters
            temp_config = self.config.copy()
            temp_config['transformer'] = {
                'd_model': params.get('d_model', 128),
                'nhead': params.get('nhead', 8),
                'num_layers': params.get('num_layers', 4),
                'ff_dim': params.get('ff_dim', 256),
                'dropout_rate': params.get('dropout_rate', 0.1),
                'max_seq_len': self.config['data'].get('sequence_length', 64)
            }
            temp_config['model']['target_mode'] = target_mode
            
            # Create Enhanced Transformer
            model = create_model(
                model_type='enhanced_transformer',
                config=temp_config,
                n_features=input_size,
                device=self.device
            )
        else:
            # Create LSTM model
            output_size = 1 if target_mode == 'binary' else 3
            model = PairSpecificLSTM(
                input_size=input_size,
                hidden_size=params.get('hidden_size', 64),
                num_layers=params.get('num_layers', 2),
                output_size=output_size,
                dropout=params.get('dropout_rate', 0.45),
                target_mode=target_mode
            ).to(self.device)
        
        return model
    
    def _log_trial_to_csv(self, csv_path: str, pair_name: str, trial_number: int,
                         params: Dict[str, Any], val_acc: float):
        """Log trial results to CSV."""
        
        trial_time = time.time()  # Simplified timing
        
        # Extract model-specific parameters
        if self.model_type == 'enhanced_transformer':
            d_model = params.get('d_model', 128)
            nhead = params.get('nhead', 8)
            num_layers = params.get('num_layers', 4)
        else:
            d_model = params.get('hidden_size', 64)
            nhead = 0  # Not applicable for LSTM
            num_layers = params.get('num_layers', 2)
        
        with open(csv_path, 'a') as f:
            f.write(f"{pair_name},{trial_number},{self.model_type},{val_acc:.2f},0.0,"
                   f"{params.get('learning_rate', 1e-3):.6f},{params.get('batch_size', 32)},"
                   f"{params.get('dropout_rate', 0.45):.3f},{d_model},{nhead},{num_layers},"
                   f"{params.get('seq_len', 64)},{trial_time:.1f}\n")
    
    def _save_study_results(self, pair_name: str, study_results: Dict, tuning_metrics: Dict):
        """Save detailed study results."""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = self.output_dir / f"{pair_name}_{self.model_type}_study_{timestamp}.json"
        
        full_results = {
            'pair_name': pair_name,
            'model_type': self.model_type,
            'study_results': study_results,
            'tuning_metrics': tuning_metrics,
            'tuning_config': {
                'optuna_trials': self.optuna_trials,
                'timeout': self.timeout,
                'cross_validate': self.cross_validate,
                'tune_architecture': self.tune_architecture,
                'tune_preprocessing': self.tune_preprocessing
            },
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"   💾 Study results saved: {results_file}")
        
        # Save best parameters as YAML config
        best_config_file = self.output_dir / f"{pair_name}_{self.model_type}_best_config_{timestamp}.yaml"
        
        best_config = {
            'pair_name': pair_name,
            'model_type': self.model_type,
            'best_parameters': study_results['best_params'],
            'validation_accuracy': study_results['best_value'],
            'tuning_metrics': tuning_metrics,
            'timestamp': timestamp
        }
        
        # Save using the config module
        from .. import config
        config.save(best_config, str(best_config_file))
        
        print(f"   📋 Best config saved: {best_config_file}")


# Factory function for easy import
def create_advanced_tuner(model_type: str, device: torch.device, config: Dict[str, Any],
                         **kwargs) -> AdvancedTuner:
    """
    Factory function to create AdvancedTuner.
    
    Args:
        model_type: Type of model to tune
        device: PyTorch device
        config: System configuration
        **kwargs: Additional tuner parameters
        
    Returns:
        Initialized AdvancedTuner instance
    """
    return AdvancedTuner(
        model_type=model_type,
        device=device,
        config=config,
        **kwargs
    )


__all__ = ['AdvancedTuner', 'create_advanced_tuner']