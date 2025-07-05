"""Enhanced validation strategies for time series models (PDF recommendations)"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import torch

class WalkForwardValidator:
    """Walk-forward validation for time series (PDF Priority #5)"""
    
    def __init__(self, train_window=252*24*4, step_size=7*24*4, purge_window=24*4):
        """
        Args:
            train_window: Training window size (252 trading days of 5-min data)
            step_size: Step size for moving window (1 week)
            purge_window: Purge window to avoid look-ahead bias (1 day)
        """
        self.train_window = train_window
        self.step_size = step_size
        self.purge_window = purge_window
        
        print(f"🔄 Walk-Forward Validator initialized:")
        print(f"   Train window: {train_window:,} samples (~{train_window/(24*4):.0f} days)")
        print(f"   Step size: {step_size:,} samples ({step_size/(24*4):.0f} days)")
        print(f"   Purge window: {purge_window:,} samples ({purge_window/(24*4):.0f} days)")
        
    def split(self, X: np.ndarray, y: np.ndarray, 
              timestamps: pd.DatetimeIndex = None) -> List[Tuple[List[int], List[int]]]:
        """Generate walk-forward splits with purging (PDF implementation)"""
        splits = []
        start_idx = 0
        
        print(f"🔍 Generating walk-forward splits for {len(X):,} samples...")
        
        while start_idx + self.train_window + self.purge_window < len(X):
            # Training period
            train_end = start_idx + self.train_window
            
            # Purge period (PDF critical: avoid look-ahead bias)
            val_start = train_end + self.purge_window
            val_end = min(val_start + self.step_size, len(X))
            
            # Minimum validation size check
            if val_end - val_start > 100:
                train_indices = list(range(start_idx, train_end))
                val_indices = list(range(val_start, val_end))
                splits.append((train_indices, val_indices))
                
                if timestamps is not None and len(splits) <= 3:  # Log first few splits
                    print(f"   Split {len(splits)}: Train={timestamps[start_idx].strftime('%Y-%m-%d')} "
                          f"to {timestamps[train_end-1].strftime('%Y-%m-%d')}, "
                          f"Val={timestamps[val_start].strftime('%Y-%m-%d')} "
                          f"to {timestamps[val_end-1].strftime('%Y-%m-%d')}")
            
            start_idx += self.step_size
        
        print(f"✅ Generated {len(splits)} walk-forward splits")
        return splits
    
    def validate_model(self, model_class, X: np.ndarray, y: np.ndarray, 
                      config: Dict, timestamps: pd.DatetimeIndex = None) -> Dict:
        """Run walk-forward validation on model (PDF implementation)"""
        splits = self.split(X, y, timestamps)
        
        if len(splits) == 0:
            raise ValueError("No valid splits generated - data too small for walk-forward")
        
        print(f"\n🚀 Running walk-forward validation: {len(splits)} splits")
        
        results = {
            'val_accuracies': [],
            'train_accuracies': [],
            'val_losses': [],
            'split_details': [],
            'timing': []
        }
        
        for i, (train_idx, val_idx) in enumerate(splits):
            split_start = datetime.now()
            print(f"\n   📊 Split {i+1}/{len(splits)} - "
                  f"Train: {len(train_idx):,}, Val: {len(val_idx):,}")
            
            try:
                # Split data
                X_train_split = X[train_idx]
                y_train_split = y[train_idx]
                X_val_split = X[val_idx]
                y_val_split = y[val_idx]
                
                # Create and train model for this split
                model = model_class(config)
                
                # Quick training for validation (reduced epochs)
                train_acc, val_acc, val_loss = self._quick_train_validate(
                    model, X_train_split, y_train_split, X_val_split, y_val_split
                )
                
                results['train_accuracies'].append(train_acc)
                results['val_accuracies'].append(val_acc)
                results['val_losses'].append(val_loss)
                
                split_time = (datetime.now() - split_start).total_seconds()
                results['timing'].append(split_time)
                
                results['split_details'].append({
                    'split_id': i,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'overfitting_gap': train_acc - val_acc,
                    'split_time': split_time
                })
                
                print(f"      Train: {train_acc:.3f}, Val: {val_acc:.3f}, "
                      f"Gap: {train_acc-val_acc:.3f}, Time: {split_time:.1f}s")
                
            except Exception as e:
                print(f"      ❌ Split {i+1} failed: {e}")
                continue
        
        # Summary statistics
        if results['val_accuracies']:
            results['mean_val_acc'] = np.mean(results['val_accuracies'])
            results['std_val_acc'] = np.std(results['val_accuracies'])
            results['mean_train_acc'] = np.mean(results['train_accuracies'])
            results['mean_overfitting_gap'] = np.mean([d['overfitting_gap'] for d in results['split_details']])
            results['total_time'] = sum(results['timing'])
            results['avg_time_per_split'] = np.mean(results['timing'])
            
            print(f"\n📊 Walk-Forward Summary:")
            print(f"   Mean Val Accuracy: {results['mean_val_acc']:.3f} ± {results['std_val_acc']:.3f}")
            print(f"   Mean Train Accuracy: {results['mean_train_acc']:.3f}")
            print(f"   Mean Overfitting Gap: {results['mean_overfitting_gap']:.3f}")
            print(f"   Total Time: {results['total_time']:.1f}s")
            print(f"   Avg Time/Split: {results['avg_time_per_split']:.1f}s")
        else:
            print(f"   ❌ No valid splits completed")
        
        return results
    
    def _quick_train_validate(self, model, X_train, y_train, X_val, y_val, 
                            quick_epochs: int = 10) -> Tuple[float, float, float]:
        """Quick training for walk-forward validation"""
        
        # Convert to tensors
        device = next(model.parameters()).device
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        # Determine target mode from model
        if hasattr(model, 'target_mode'):
            target_mode = model.target_mode
        else:
            target_mode = 'binary'  # default
        
        if target_mode == 'binary':
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            y_train_tensor = torch.LongTensor(y_train).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            criterion = torch.nn.CrossEntropyLoss()
        
        # Quick optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Quick training loop
        model.train()
        for epoch in range(quick_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Calculate final accuracies
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(X_train_tensor)
            if target_mode == 'binary':
                train_pred = (torch.sigmoid(train_outputs) > 0.5).float()
                train_acc = (train_pred == y_train_tensor).float().mean().item()
            else:
                train_pred = torch.argmax(train_outputs, dim=1)
                train_acc = (train_pred == y_train_tensor).float().mean().item()
            
            # Validation accuracy
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if target_mode == 'binary':
                val_pred = (torch.sigmoid(val_outputs) > 0.5).float()
                val_acc = (val_pred == y_val_tensor).float().mean().item()
            else:
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()
        
        return train_acc, val_acc, val_loss


class TemporalValidator:
    """PDF: Temporal validation for specific time periods"""
    
    def __init__(self, train_start: str, train_end: str, 
                 val_start: str, val_end: str):
        """
        Args:
            train_start: Training period start (YYYY-MM-DD)
            train_end: Training period end (YYYY-MM-DD)
            val_start: Validation period start (YYYY-MM-DD)
            val_end: Validation period end (YYYY-MM-DD)
        """
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)
        
        print(f"📅 Temporal Validator initialized:")
        print(f"   Train: {self.train_start.strftime('%Y-%m-%d')} to {self.train_end.strftime('%Y-%m-%d')}")
        print(f"   Val: {self.val_start.strftime('%Y-%m-%d')} to {self.val_end.strftime('%Y-%m-%d')}")
    
    def split_by_time(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by time periods (PDF implementation)"""
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for temporal validation")
        
        # Create masks for time periods
        train_mask = (data.index >= self.train_start) & (data.index <= self.train_end)
        val_mask = (data.index >= self.val_start) & (data.index <= self.val_end)
        
        train_data = data[train_mask]
        val_data = data[val_mask]
        
        print(f"📊 Temporal split results:")
        print(f"   Train: {self.train_start.strftime('%Y-%m-%d')} to {self.train_end.strftime('%Y-%m-%d')} "
              f"({len(train_data):,} samples)")
        print(f"   Val: {self.val_start.strftime('%Y-%m-%d')} to {self.val_end.strftime('%Y-%m-%d')} "
              f"({len(val_data):,} samples)")
        
        if len(train_data) == 0:
            raise ValueError(f"No training data found in period {self.train_start} to {self.train_end}")
        if len(val_data) == 0:
            raise ValueError(f"No validation data found in period {self.val_start} to {self.val_end}")
        
        return train_data, val_data
    
    def validate_temporal_model(self, model_class, data: pd.DataFrame, 
                              features_cols: List[str], target_col: str, 
                              config: Dict) -> Dict:
        """Run temporal validation on specific time periods"""
        
        print(f"\n🕒 Running temporal validation...")
        
        # Split data by time
        train_data, val_data = self.split_by_time(data)
        
        # Prepare features and targets
        X_train = train_data[features_cols].values
        y_train = train_data[target_col].values
        X_val = val_data[features_cols].values
        y_val = val_data[target_col].values
        
        # Create and train model
        model = model_class(config)
        
        try:
            # Train on temporal training period
            print(f"   🚀 Training on temporal period...")
            train_acc, val_acc, val_loss = self._temporal_train_validate(
                model, X_train, y_train, X_val, y_val
            )
            
            results = {
                'temporal_train_acc': train_acc,
                'temporal_val_acc': val_acc,
                'temporal_val_loss': val_loss,
                'overfitting_gap': train_acc - val_acc,
                'train_period': f"{self.train_start.strftime('%Y-%m-%d')} to {self.train_end.strftime('%Y-%m-%d')}",
                'val_period': f"{self.val_start.strftime('%Y-%m-%d')} to {self.val_end.strftime('%Y-%m-%d')}",
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            }
            
            print(f"   📊 Temporal Results:")
            print(f"      Train Accuracy: {train_acc:.3f}")
            print(f"      Val Accuracy: {val_acc:.3f}")
            print(f"      Overfitting Gap: {train_acc - val_acc:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Temporal validation failed: {e}")
            return {}
    
    def _temporal_train_validate(self, model, X_train, y_train, X_val, y_val,
                               epochs: int = 30) -> Tuple[float, float, float]:
        """Train and validate model on temporal data"""
        
        # Similar to walk-forward quick training but with more epochs
        device = next(model.parameters()).device
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        # Determine target mode
        if hasattr(model, 'target_mode'):
            target_mode = model.target_mode
        else:
            target_mode = 'binary'
        
        if target_mode == 'binary':
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            y_train_tensor = torch.LongTensor(y_train).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)
            criterion = torch.nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        
        # Training loop
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    if target_mode == 'binary':
                        val_pred = (torch.sigmoid(val_outputs) > 0.5).float()
                        val_acc = (val_pred == y_val_tensor).float().mean().item()
                    else:
                        val_pred = torch.argmax(val_outputs, dim=1)
                        val_acc = (val_pred == y_val_tensor).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(X_train_tensor)
            if target_mode == 'binary':
                train_pred = (torch.sigmoid(train_outputs) > 0.5).float()
                train_acc = (train_pred == y_train_tensor).float().mean().item()
            else:
                train_pred = torch.argmax(train_outputs, dim=1)
                train_acc = (train_pred == y_train_tensor).float().mean().item()
            
            # Validation accuracy
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            if target_mode == 'binary':
                val_pred = (torch.sigmoid(val_outputs) > 0.5).float()
                val_acc = (val_pred == y_val_tensor).float().mean().item()
            else:
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()
        
        return train_acc, val_acc, val_loss


class EnhancedValidator:
    """PDF: Combined validation strategies"""
    
    def __init__(self, validation_type: str = "walk_forward", **kwargs):
        """
        Args:
            validation_type: 'walk_forward', 'temporal', or 'both'
            **kwargs: Parameters for specific validators
        """
        self.validation_type = validation_type
        
        if validation_type in ['walk_forward', 'both']:
            self.walk_forward = WalkForwardValidator(**kwargs.get('walk_forward', {}))
        
        if validation_type in ['temporal', 'both']:
            temporal_params = kwargs.get('temporal', {})
            if temporal_params:
                self.temporal = TemporalValidator(**temporal_params)
            else:
                self.temporal = None
        
        print(f"🔧 Enhanced Validator initialized: {validation_type}")
    
    def validate(self, model_class, X: np.ndarray, y: np.ndarray, 
                config: Dict, data: pd.DataFrame = None, 
                features_cols: List[str] = None, target_col: str = None) -> Dict:
        """Run comprehensive validation"""
        
        results = {'validation_type': self.validation_type}
        
        if self.validation_type in ['walk_forward', 'both']:
            print(f"\n🔄 Running Walk-Forward Validation...")
            wf_results = self.walk_forward.validate_model(model_class, X, y, config)
            results['walk_forward'] = wf_results
        
        if self.validation_type in ['temporal', 'both'] and self.temporal is not None:
            if data is not None and features_cols is not None and target_col is not None:
                print(f"\n🕒 Running Temporal Validation...")
                temp_results = self.temporal.validate_temporal_model(
                    model_class, data, features_cols, target_col, config
                )
                results['temporal'] = temp_results
            else:
                print(f"   ⚠️ Temporal validation skipped - missing data/columns")
        
        return results


def create_validator(validation_type: str = "walk_forward", **kwargs):
    """Factory function for validators (PDF recommendation)"""
    
    if validation_type == "walk_forward":
        return WalkForwardValidator(**kwargs)
    elif validation_type == "temporal":
        return TemporalValidator(**kwargs)
    elif validation_type == "enhanced":
        return EnhancedValidator(**kwargs)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")

__all__ = ['WalkForwardValidator', 'TemporalValidator', 'EnhancedValidator', 'create_validator']
