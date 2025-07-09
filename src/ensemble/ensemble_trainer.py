# src/ensemble/ensemble_trainer.py - Import düzeltmesi ve basit preprocessing

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Enhanced ensemble trainer for forex prediction models"""
    
    def __init__(self, model_configs: List[Dict], voting_strategy: str = 'confidence_weighted', device: str = 'cuda'):
        """
        Initialize ensemble trainer
        
        Args:
            model_configs: List of model configurations
            voting_strategy: Voting strategy for ensemble
            device: Device to train on
        """
        self.model_configs = model_configs
        self.voting_strategy = voting_strategy
        self.device = torch.device(device)
        self.models = {}
        self.model_weights = {}
        self.training_history = {}
        
        logger.info(f"🏋️ EnsembleTrainer initialized on {self.device}")
    
    def _create_model(self, model_config: Dict, n_features: int) -> nn.Module:
        """Create a model based on configuration"""
        from src.models.factory import create_model
        
        model_type = model_config['type']
        config = model_config['config']
        
        try:
            model = create_model(
                model_type=model_type,
                config=config,
                n_features=n_features,
                device=self.device
            )
            logger.info(f"✅ Created {model_type} model")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to create {model_type} model: {str(e)}")
            raise
    
    def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical features without complex preprocessing"""
        
        # Temel fiyat features
        features = pd.DataFrame(index=data.index)
        
        # OHLCV features
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']
        features['volume'] = data['volume']
        
        # Basit technical indicators
        features['returns'] = data['close'].pct_change()
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['co_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Simple moving averages
        for window in [5, 10, 20]:
            features[f'sma_{window}'] = data['close'].rolling(window=window).mean()
            features[f'sma_{window}_ratio'] = data['close'] / features[f'sma_{window}']
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        features['volatility'] = data['close'].rolling(window=20).std()
        
        # Normalize features
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col] = (features[col] - features[col].mean()) / (features[col].std() + 1e-8)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _create_labels(self, data: pd.DataFrame, target_mode: str = 'binary') -> pd.Series:
        """Create CUDA-safe labels for prediction"""
        
        # Simple return-based labeling
        returns = data['close'].pct_change().shift(-1)  # Next period return
        
        if target_mode == 'binary':
            # Binary: up (1) or down/sideways (0)
            labels = (returns > 0).astype(int)
            # Ensure strict binary values
            labels = labels.clip(0, 1)
        else:
            # Multi-class: up (2), down (0), sideways (1)
            threshold = returns.std() * 0.5  # Dynamic threshold
            labels = np.where(returns > threshold, 2,
                            np.where(returns < -threshold, 0, 1))
            # Ensure strict multi-class values
            labels = pd.Series(labels, index=data.index)
            labels = labels.clip(0, 2)
        
        # Remove any NaN values
        labels = labels.fillna(0)  # Default to class 0 for NaN
        
        return labels

    
    # src/ensemble/ensemble_trainer.py - CUDA assertion fix

    def _prepare_data(self, data: pd.DataFrame, pair: str, target_mode: str = 'binary') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training with CUDA-safe label formatting"""
        
        print(f"🔧 {pair} data preparation ({target_mode})...")
        
        # Create features
        features_df = self._create_basic_features(data)
        
        # Create labels
        labels = self._create_labels(data, target_mode)
        
        # Align features and labels
        min_length = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_length]
        labels = labels.iloc[:min_length]
        
        # Remove any remaining NaN values
        valid_indices = ~(features_df.isna().any(axis=1) | labels.isna())
        features_df = features_df[valid_indices]
        labels = labels[valid_indices]
        
        print(f"   📊 Features shape: {features_df.shape}")
        print(f"   📊 Labels shape: {labels.shape}")
        
        # **CUDA FIX: Ensure labels are properly formatted**
        if target_mode == 'binary':
            # Convert to 0/1 integers
            labels = labels.astype(int)
            # Ensure values are exactly 0 or 1
            labels = labels.clip(0, 1)
            unique_labels = labels.unique()
            print(f"   📊 Binary labels range: {unique_labels.min()} to {unique_labels.max()}")
            
            # Ensure we have both classes
            if len(unique_labels) == 1:
                print(f"   ⚠️ Single class detected: {unique_labels[0]}")
                # Add one sample of the other class to avoid single class issue
                if unique_labels[0] == 0:
                    labels.iloc[0] = 1
                else:
                    labels.iloc[0] = 0
        else:
            # Multi-class: ensure labels are 0, 1, 2
            labels = labels.astype(int)
            labels = labels.clip(0, 2)  # Ensure range 0-2
            unique_labels = labels.unique()
            print(f"   📊 Multi-class labels range: {unique_labels.min()} to {unique_labels.max()}")
        
        print(f"   📊 Target distribution: {labels.value_counts().to_dict()}")
        
        # Convert to tensors with proper dtypes
        X = torch.FloatTensor(features_df.values)
        y = torch.LongTensor(labels.values)  # Use LongTensor for classification
        
        # **CUDA FIX: Additional validation**
        print(f"   🔍 Tensor validation:")
        print(f"      X shape: {X.shape}, dtype: {X.dtype}")
        print(f"      y shape: {y.shape}, dtype: {y.dtype}")
        print(f"      y range: {y.min().item()} to {y.max().item()}")
        
        # Ensure no invalid values
        if target_mode == 'binary':
            if y.min() < 0 or y.max() > 1:
                print(f"   ❌ Invalid binary labels detected! Range: {y.min()} to {y.max()}")
                y = torch.clamp(y, 0, 1)
                print(f"   ✅ Fixed binary labels to range: {y.min()} to {y.max()}")
        else:
            if y.min() < 0 or y.max() > 2:
                print(f"   ❌ Invalid multi-class labels detected! Range: {y.min()} to {y.max()}")
                y = torch.clamp(y, 0, 2)
                print(f"   ✅ Fixed multi-class labels to range: {y.min()} to {y.max()}")
        
        # Train/validation split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"   📊 Data split: Train={len(X_train)}, Val={len(X_val)}")
        
        # Final validation before returning
        print(f"   🔍 Final validation:")
        print(f"      Train labels range: {y_train.min()} to {y_train.max()}")
        print(f"      Val labels range: {y_val.min()} to {y_val.max()}")
        
        return X_train.to(self.device), X_val.to(self.device), y_train.to(self.device), y_val.to(self.device)
    
    def _train_single_model(self, model: nn.Module, model_name: str, X_train: torch.Tensor, y_train: torch.Tensor, 
                        X_val: torch.Tensor, y_val: torch.Tensor, config: Dict) -> Dict:
        """Train a single model with CUDA-safe loss computation"""
        
        print(f"🚀 Training {model_name}...")
        
        # Training parameters
        epochs = config.get('epochs', 10)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        
        # **CUDA FIX: Determine number of classes from data**
        num_classes = len(torch.unique(y_train))
        print(f"   📊 Detected {num_classes} classes in training data")
        print(f"   📊 Train labels unique: {torch.unique(y_train)}")
        print(f"   📊 Val labels unique: {torch.unique(y_val)}")
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Create batches
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_train))
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                # **CUDA FIX: Additional safety check**
                if batch_y.min() < 0 or batch_y.max() >= num_classes:
                    print(f"   ⚠️ Batch {batch_idx}: Invalid labels detected!")
                    batch_y = torch.clamp(batch_y, 0, num_classes - 1)
                
                try:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # **CUDA FIX: Ensure output dimensions match**
                    if outputs.shape[1] != num_classes:
                        print(f"   ❌ Output dimension mismatch: {outputs.shape[1]} != {num_classes}")
                        continue
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
                    
                except RuntimeError as e:
                    print(f"   ❌ Training error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Validation with error handling
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                num_val_batches = (len(X_val) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_val_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(X_val))
                    
                    batch_X = X_val[start_idx:end_idx]
                    batch_y = y_val[start_idx:end_idx]
                    
                    # **CUDA FIX: Additional safety check**
                    if batch_y.min() < 0 or batch_y.max() >= num_classes:
                        batch_y = torch.clamp(batch_y, 0, num_classes - 1)
                    
                    try:
                        outputs = model(batch_X)
                        
                        if outputs.shape[1] != num_classes:
                            continue
                        
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                    except RuntimeError as e:
                        print(f"   ❌ Validation error in batch {batch_idx}: {str(e)}")
                        continue
            
            # Calculate metrics
            if num_batches > 0:
                avg_train_loss = train_loss / num_batches
                train_acc = 100.0 * train_correct / max(train_total, 1)
            else:
                avg_train_loss = 0.0
                train_acc = 0.0
            
            if num_val_batches > 0:
                avg_val_loss = val_loss / num_val_batches
                val_acc = 100.0 * val_correct / max(val_total, 1)
            else:
                avg_val_loss = 0.0
                val_acc = 0.0
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"✅ {model_name} training completed. Best Val Acc: {best_val_acc:.2f}%")
        
        return {
            'model': model,
            'best_val_acc': best_val_acc,
            'history': history
        }
    
    def _calculate_model_weights(self, results: Dict) -> Dict:
        """Calculate ensemble weights based on validation performance"""
        
        if self.voting_strategy == 'simple':
            # Equal weights
            weights = {name: 1.0 / len(results) for name in results.keys()}
        
        elif self.voting_strategy == 'performance_weighted':
            # Weight by validation accuracy
            total_acc = sum(result['best_val_acc'] for result in results.values())
            weights = {name: result['best_val_acc'] / total_acc for name, result in results.items()}
        
        elif self.voting_strategy == 'confidence_weighted':
            # Weight by confidence (placeholder - equal weights for now)
            weights = {name: 1.0 / len(results) for name in results.keys()}
        
        else:
            # Default to equal weights
            weights = {name: 1.0 / len(results) for name in results.keys()}
        
        return weights
    
    def train(self, data: pd.DataFrame, pair: str, target_mode: str = 'binary') -> Optional[Dict]:
        """Train ensemble of models"""
        
        try:
            # Prepare data
            X_train, X_val, y_train, y_val = self._prepare_data(data, pair, target_mode)
            n_features = X_train.shape[1]
            
            print(f"🎯 {pair} features: {n_features}")
            
            # Train individual models
            results = {}
            
            for model_config in self.model_configs:
                model_name = model_config['type']
                config = model_config['config']
                
                try:
                    # Create model
                    model = self._create_model(model_config, n_features)
                    
                    # Train model
                    result = self._train_single_model(
                        model=model,
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        config=config
                    )
                    
                    results[model_name] = result
                    self.models[model_name] = result['model']
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not results:
                logger.error("❌ No models trained successfully")
                return None
            
            # Calculate ensemble weights
            self.model_weights = self._calculate_model_weights(results)
            
            # Store training history
            self.training_history[pair] = results
            
            # Print ensemble summary
            print(f"\n📊 Ensemble Summary for {pair}:")
            for name, weight in self.model_weights.items():
                val_acc = results[name]['best_val_acc']
                print(f"   {name}: Weight={weight:.3f}, Val Acc={val_acc:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ensemble training failed for {pair}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions"""
        
        if not self.models:
            raise ValueError("No trained models available")
        
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(X)
                pred_probs = torch.softmax(pred, dim=1)
                predictions.append(pred_probs)
                weights.append(self.model_weights.get(model_name, 1.0))
        
        # Weighted ensemble prediction
        weights_tensor = torch.tensor(weights, device=X.device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights_tensor[i] * pred
        
        return ensemble_pred
    
    def save_ensemble(self, save_path: Path):
        """Save ensemble models and weights"""
        
        ensemble_data = {
            'model_configs': self.model_configs,
            'model_weights': self.model_weights,
            'voting_strategy': self.voting_strategy,
            'training_history': self.training_history,
            'models': {}
        }
        
        # Save model state dicts
        for name, model in self.models.items():
            ensemble_data['models'][name] = model.state_dict()
        
        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        logger.info(f"✅ Ensemble saved to {save_path}")
    
    def load_ensemble(self, load_path: Path):
        """Load ensemble models and weights"""
        
        with open(load_path, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.model_configs = ensemble_data['model_configs']
        self.model_weights = ensemble_data['model_weights']
        self.voting_strategy = ensemble_data['voting_strategy']
        self.training_history = ensemble_data['training_history']
        
        logger.info(f"✅ Ensemble loaded from {load_path}")


# Voting strategies utility class
class VotingStrategies:
    """Utility class for different voting strategies"""
    
    @staticmethod
    def simple_voting(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple average voting"""
        return torch.stack(list(predictions.values())).mean(dim=0)
    
    @staticmethod
    def weighted_voting(predictions: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
        """Weighted voting based on model weights"""
        weighted_preds = []
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0)
            weighted_preds.append(pred * weight)
            total_weight += weight
        
        return torch.stack(weighted_preds).sum(dim=0) / total_weight
    
    @staticmethod
    def confidence_weighted_voting(predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Confidence-weighted voting based on prediction entropy"""
        confidences = {}
        
        for name, pred in predictions.items():
            # Calculate confidence as inverse entropy
            entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)
            confidence = 1.0 / (1.0 + entropy)  # Inverse entropy
            confidences[name] = confidence
        
        return VotingStrategies.weighted_voting(predictions, confidences)