"""
İyileştirilmiş Optuna optimization utilities with Transformer model support.

Bu modül, Transformer modelleri için hiperparametre aramasını desteklemek üzere
mevcut Optuna optimizasyonunu genişletir. Mimari-specific parametreler,
attention head'leri, model boyutları ve feedforward boyutları dahil olmak üzere
gelişmiş konfigürasyon validasyonu, DataLoader kullanımı ve proper error handling içerir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import optuna
import numpy as np
import yaml
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ..models.transformer_model import TransformerClassifier
from ..models.lstm import PairSpecificLSTM
from ..models.factory import create_model, validate_transformer_config, validate_lstm_config

# Logger yapılandırması
logger = logging.getLogger(__name__)


class EnhancedOptunaOptimizer:
    """
    Transformer desteği ile gelişmiş Optuna tabanlı hiperparametre optimizer'ı.
    
    LSTM ve Transformer mimarileri için Optuna'nın TPE sampler'ını kullanarak
    verimli hiperparametre araması gerçekleştirir. DataLoader kullanımı,
    proper validation ve gelişmiş error handling içerir.
    
    Args:
        pair_name: Döviz çifti adı
        device: Eğitim için PyTorch cihazı
        model_type: Model tipi ('lstm' veya 'transformer')
        reproducibility_seed: Tekrarlanabilirlik için seed
    """
    
    def __init__(self, pair_name: str, device: torch.device, 
                 model_type: str = 'lstm', reproducibility_seed: int = 42):
        self.pair_name = pair_name
        self.device = device
        self.model_type = model_type.lower()
        self.reproducibility_seed = reproducibility_seed
        
        logger.info(f"🔍 EnhancedOptunaOptimizer başlatıldı:")
        logger.info(f"   Pair: {pair_name}")
        logger.info(f"   Model: {model_type.upper()}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Seed: {reproducibility_seed}")
        
    def objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, 
                 target_mode: str = 'binary') -> float:
        """
        Transformer desteği ile gelişmiş Optuna objective fonksiyonu.
        
        Args:
            trial: Optuna trial objesi
            X: Giriş özellikleri
            y: Hedef etiketler
            target_mode: Tahmin görevi tipi
            
        Returns:
            Maksimize edilecek validation accuracy
        """
        try:
            # Reproducibility ayarları her trial başında
            torch.manual_seed(self.reproducibility_seed + trial.number)
            np.random.seed(self.reproducibility_seed + trial.number)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.reproducibility_seed + trial.number)
            
            # Ortak hiperparametreler
            horizon = trial.suggest_categorical('horizon', [32, 48, 64, 96])
            seq_len = trial.suggest_categorical('seq_len', [64, 96, 128])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Model-specific hiperparametreler
            if self.model_type == 'transformer':
                # Transformer-specific parametreler
                d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
                nhead = trial.suggest_categorical('nhead', [4, 8, 16])
                num_layers = trial.suggest_categorical('num_layers', [2, 4, 6, 8])
                dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024])
                
                # d_model ve nhead uyumluluğunu kontrol et
                if d_model % nhead != 0:
                    valid_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
                    if valid_heads:
                        nhead = trial.suggest_categorical(f'nhead_adjusted_{d_model}', valid_heads)
                    else:
                        trial.report(0.0, 0)
                        raise optuna.exceptions.TrialPruned()
                
                # Transformer konfigürasyonu oluştur
                config = {
                    'model': {'dropout_rate': dropout, 'target_mode': target_mode, 'type': 'transformer'},
                    'transformer': {
                        'd_model': d_model,
                        'nhead': nhead,
                        'num_layers': num_layers,
                        'dim_feedforward': dim_feedforward
                    },
                    'data': {'sequence_length': seq_len},
                    'training': {'learning_rate': learning_rate, 'batch_size': batch_size}
                }
                
                # Transformer config validasyonu
                try:
                    config = validate_transformer_config(config)
                except ValueError as e:
                    logger.warning(f"   ⚠️ Trial {trial.number} config validation failed: {e}")
                    trial.report(0.0, 0)
                    raise optuna.exceptions.TrialPruned()
                
            else:
                # LSTM-specific parametreler
                hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 96, 128, 256])
                num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
                
                config = {
                    'model': {
                        'dropout_rate': dropout,
                        'target_mode': target_mode,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'type': 'lstm'
                    },
                    'data': {'sequence_length': seq_len},
                    'training': {'learning_rate': learning_rate, 'batch_size': batch_size}
                }
                
                # LSTM config validasyonu
                try:
                    config = validate_lstm_config(config)
                except ValueError as e:
                    logger.warning(f"   ⚠️ Trial {trial.number} LSTM config validation failed: {e}")
                    trial.report(0.0, 0)
                    raise optuna.exceptions.TrialPruned()
            
            # Model oluştur
            model = create_model(self.model_type, config, X.shape[2], self.device)
            
            # DataLoader'ları oluştur
            train_loader, val_loader = self._create_data_loaders(
                X, y, batch_size, target_mode
            )
            
            # Loss ve optimizer setup
            criterion = self._get_enhanced_criterion(y, target_mode)
            optimizer, scheduler = self._get_optimizer_and_scheduler(
                model, config, len(train_loader)
            )
            
            # Training loop
            best_val_acc = 0.0
            best_state_dict = None
            patience_counter = 0
            max_patience = 8 if self.model_type == 'transformer' else 5
            
            epochs = 25  # Hızlı optimizasyon için
            for epoch in range(epochs):
                # Training epoch
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, optimizer, criterion, scheduler
                )
                
                # Validation epoch
                val_loss, val_acc = self._validate_epoch(
                    model, val_loader, criterion
                )
                
                # Scheduler step (model tipine göre)
                self._update_scheduler(scheduler, val_loss, epoch)
                
                # Best model checkpoint
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state_dict = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Overfitting gap kontrolü
                train_val_gap = abs(train_acc - val_acc)
                
                # Enhanced early stopping criteria
                if (val_acc >= 60 and train_val_gap <= 15) or patience_counter >= max_patience:
                    if val_acc >= 60 and train_val_gap <= 15:
                        logger.info(f"   🎯 Trial {trial.number} early success: val_acc={val_acc:.3f}")
                    break
                
                # Optuna pruning
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    logger.info(f"   ✂️ Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()
            
            # En iyi model ağırlıklarını geri yükle
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
            
            return best_val_acc
            
        except optuna.exceptions.TrialPruned:
            raise  # Pruning exception'ını yeniden fırlat
        except Exception as e:
            logger.exception(f"   ❌ Optuna trial {trial.number} failed")
            return 0.0
    
    def _create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                           batch_size: int, target_mode: str) -> tuple:
        """DataLoader'ları oluştur."""
        total_size = len(X)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        
        # TensorDataset'ler oluştur
        train_dataset = data_utils.TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = data_utils.TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        # DataLoader'lar oluştur
        train_loader = data_utils.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = data_utils.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def _get_enhanced_criterion(self, y: np.ndarray, target_mode: str) -> nn.Module:
        """Gelişmiş loss criterion al."""
        if target_mode == 'binary':
            # BCEWithLogitsLoss kullan (daha stabil)
            y_flat = y.astype(int).flatten()
            class_counts = np.bincount(y_flat)
            
            if len(class_counts) > 1 and class_counts[1] > 0:
                pos_weight = torch.FloatTensor([class_counts[0] / class_counts[1]]).to(self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:  # three_class
            y_flat = y.astype(int).flatten()
            unique_classes, counts = np.unique(y_flat, return_counts=True)
            total_samples = len(y_flat)
            
            if len(unique_classes) >= 2:
                class_weights = torch.FloatTensor([
                    total_samples / (len(unique_classes) * count) 
                    for count in counts
                ]).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def _get_optimizer_and_scheduler(self, model: nn.Module, config: Dict[str, Any], 
                                   steps_per_epoch: int) -> tuple:
        """Model-specific optimizer ve scheduler."""
        learning_rate = config['training']['learning_rate']
        
        if self.model_type == 'transformer':
            # Transformer-specific optimizer
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=0.01
            )
            
            # OneCycleLR scheduler
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate * 3,
                epochs=25,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
        else:
            # LSTM-specific optimizer
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=0.01
            )
            
            # ReduceLROnPlateau scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                patience=2, 
                factor=0.5
            )
        
        return optimizer, scheduler
    
    def _train_epoch(self, model: nn.Module, train_loader: data_utils.DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    scheduler: Optional[optim.lr_scheduler._LRScheduler]) -> tuple:
        """Tek training epoch."""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Target boyut ayarlama
            if len(batch_y.shape) == 1:
                if hasattr(criterion, 'pos_weight'):  # BCEWithLogitsLoss
                    batch_y = batch_y.unsqueeze(1)
                elif isinstance(criterion, nn.CrossEntropyLoss):
                    batch_y = batch_y.long()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # OneCycleLR adımı (sadece Transformer için)
            if (self.model_type == 'transformer' and 
                hasattr(scheduler, 'step') and 'OneCycle' in str(type(scheduler))):
                scheduler.step()
            
            total_loss += loss.item()
            
            # Accuracy hesaplama
            with torch.no_grad():
                if hasattr(criterion, 'pos_weight'):  # Binary BCEWithLogitsLoss
                    predictions = torch.sigmoid(outputs) > 0.5
                    correct_predictions += (predictions == batch_y).sum().item()
                else:  # CrossEntropyLoss
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                
                total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: data_utils.DataLoader,
                       criterion: nn.Module) -> tuple:
        """Tek validation epoch."""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Target boyut ayarlama
                if len(batch_y.shape) == 1:
                    if hasattr(criterion, 'pos_weight'):  # BCEWithLogitsLoss
                        batch_y = batch_y.unsqueeze(1)
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        batch_y = batch_y.long()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Accuracy hesaplama
                if hasattr(criterion, 'pos_weight'):  # Binary BCEWithLogitsLoss
                    predictions = torch.sigmoid(outputs) > 0.5
                    correct_predictions += (predictions == batch_y).sum().item()
                else:  # CrossEntropyLoss
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                
                total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _update_scheduler(self, scheduler: Optional[optim.lr_scheduler._LRScheduler], 
                         val_loss: float, epoch: int) -> None:
        """Scheduler'ı güncelle (model tipine göre)."""
        if scheduler is None:
            return
        
        # ReduceLROnPlateau adımı (sadece LSTM için)
        if (self.model_type == 'lstm' and 
            hasattr(scheduler, 'step') and 'ReduceLR' in str(type(scheduler))):
            scheduler.step(val_loss)
    
    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 40, 
                target_mode: str = 'binary') -> Dict[str, Any]:
        """
        Model-specific arama alanı ile gelişmiş Optuna optimizasyonu çalıştır.
        
        Args:
            X: Giriş özellikleri
            y: Hedef etiketler
            n_trials: Çalıştırılacak trial sayısı
            target_mode: Tahmin görevi tipi
            
        Returns:
            En iyi parametreler ve optimizasyon geçmişi içeren sözlük
        """
        model_name = self.model_type.upper()
        logger.info(f"🔍 {self.pair_name} Gelişmiş {model_name} Optuna optimizasyonu")
        logger.info(f"   Trials: {n_trials}, Mode: {target_mode}")
        
        # Pruning ile study oluştur
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.reproducibility_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
        )
        
        # Timeout koruması ile optimize et
        try:
            study.optimize(
                lambda trial: self.objective(trial, X, y, target_mode), 
                n_trials=n_trials,
                timeout=3600  # 60 dakika timeout
            )
        except KeyboardInterrupt:
            logger.info(f"   ⏹️ Kullanıcı tarafından durduruldu")
        
        best_params = study.best_params
        optimization_history = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'optimization_direction': 'maximize',
            'model_type': self.model_type
        }
        
        logger.info(f"   ✅ En iyi {model_name} parametreleri: {best_params}")
        logger.info(f"   🎯 En iyi skor: {study.best_value:.4f}")
        logger.info(f"   📊 Tamamlanan trial'lar: {len(study.trials)}/{n_trials}")
        
        # Parameter importance analizi (güvenli)
        if len(study.trials) > 5:
            try:
                importance = optuna.importance.get_param_importances(study)
                logger.info(f"   📈 Parametre önemi: {importance}")
                optimization_history['param_importance'] = importance
            except Exception as e:
                logger.warning(f"   ⚠️ Parametre önemi hesaplanamadı: {e}")
                optimization_history['param_importance'] = {}
        else:
            optimization_history['param_importance'] = {}
        
        # En iyi parametreleri kaydet
        self._save_best_params(best_params, optimization_history, target_mode)
        
        return {**best_params, 'optimization_history': optimization_history}
    
    def _save_best_params(self, best_params: Dict[str, Any], 
                         optimization_history: Dict[str, Any],
                         target_mode: str) -> None:
        """Gelişmiş metadata ile en iyi parametreleri kaydet."""
        try:
            os.makedirs('configs/optuna', exist_ok=True)
            save_path = f'configs/optuna/best_{self.pair_name}_{self.model_type}_{target_mode}.yaml'
            
            save_data = {
                'pair_name': self.pair_name,
                'model_type': self.model_type,
                'target_mode': target_mode,
                'best_parameters': best_params,
                'optimization_history': optimization_history,
                'timestamp': datetime.now().isoformat(),  # pandas dependency kaldırıldı
                'reproducibility_seed': self.reproducibility_seed
            }
            
            with open(save_path, 'w') as f:
                yaml.dump(save_data, f, default_flow_style=False, indent=2)
                
            logger.info(f"   💾 {self.model_type.upper()} parametreleri kaydedildi: {save_path}")
        except Exception as e:
            logger.error(f"   ❌ Parametre kaydetme başarısız: {e}")
    
    def load_best_params(self, target_mode: str) -> Dict[str, Any]:
        """Önceden kaydedilmiş en iyi parametreleri yükle."""
        try:
            load_path = f'configs/optuna/best_{self.pair_name}_{self.model_type}_{target_mode}.yaml'
            
            if os.path.exists(load_path):
                with open(load_path, 'r') as f:
                    data = yaml.safe_load(f)
                logger.info(f"   📁 Önbelleğe alınmış parametreler yüklendi: {load_path}")
                return data.get('best_parameters', {})
            else:
                logger.info(f"   ℹ️ {self.pair_name}_{self.model_type}_{target_mode} için önbellek bulunamadı")
                return {}
        except Exception as e:
            logger.warning(f"   ⚠️ Önbelleğe alınmış parametre yükleme başarısız: {e}")
            return {}


def create_enhanced_optimizer(pair_name: str, device: torch.device, 
                            model_type: str = 'lstm', 
                            reproducibility_seed: int = 42) -> EnhancedOptunaOptimizer:
    """
    Model tipi desteği ile gelişmiş Optuna optimizer factory fonksiyonu.
    
    Args:
        pair_name: Döviz çifti adı
        device: PyTorch cihazı
        model_type: Model tipi ('lstm' veya 'transformer')
        reproducibility_seed: Tekrarlanabilirlik için seed
        
    Returns:
        Başlatılmış optimizer instance'ı
    """
    return EnhancedOptunaOptimizer(pair_name, device, model_type, reproducibility_seed)


__all__ = ['EnhancedOptunaOptimizer', 'create_enhanced_optimizer']