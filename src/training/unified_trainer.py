"""
İyileştirilmiş unified training module with best practices.

Bu modül, LSTM, Transformer ve Enhanced Transformer modelleri için en iyi uygulamalarla
eğitim yeteneklerini sağlar. DataLoader kullanımı, uygun scheduler yönetimi
ve gelişmiş checkpoint sistemi içerir.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Union
from tqdm import tqdm

from ..models.transformer_model import (
    TransformerClassifier, 
    create_data_loaders, 
    set_reproducibility,
    train_transformer_model
)
from ..models.enhanced_transformer import (  # NEW IMPORT
    train_enhanced_transformer,
    EnhancedTransformer
)
from ..models.lstm import PairSpecificLSTM
from ..models.factory import create_model, get_model_info
from ..models.losses import get_loss_function

# Logging yapılandırması
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """
    LSTM, Transformer ve Enhanced Transformer modelleri için birleşik trainer.
    
    Her mimari için uygun optimizer'lar, scheduler'lar ve
    eğitim stratejileri ile model-specific optimizasyonları işler.
    
    Args:
        device: Eğitim için PyTorch cihazı
        model_type: Model tipi ('lstm', 'transformer' veya 'enhanced_transformer')
        use_focal_loss: Focal loss kullanılıp kullanılmayacağı
        target_mode: Tahmin tipi ('binary' veya 'three_class')
        reproducibility_seed: Tekrarlanabilirlik için seed
    """
    
    def __init__(self, device: torch.device, model_type: str = 'lstm',
                 use_focal_loss: bool = True, target_mode: str = 'binary',
                 reproducibility_seed: int = 42):
        self.device = device
        self.model_type = model_type.lower()
        self.use_focal_loss = use_focal_loss
        self.target_mode = target_mode
        self.pair_models = {}
        self.pair_histories = {}
        
        # Tekrarlanabilirlik ayarla
        set_reproducibility(reproducibility_seed)
        
        logger.info(f"🔧 UnifiedTrainer başlatıldı:")
        logger.info(f"   Model type: {self.model_type.upper()}")
        logger.info(f"   Target mode: {self.target_mode}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Reproducibility seed: {reproducibility_seed}")
        
    def train_pair_model(self, pair_name: str, X: np.ndarray, y: np.ndarray, 
                        config: Dict[str, Any], epochs: int = 120, 
                        batch_size: int = 32) -> Tuple[Union[PairSpecificLSTM, TransformerClassifier, EnhancedTransformer], Dict]:
        """
        Belirli bir döviz çifti için model eğit.
        
        Args:
            pair_name: Döviz çifti adı
            X: Giriş sekansları
            y: Hedef etiketler  
            config: Model yapılandırması
            epochs: Eğitim epoch sayısı
            batch_size: Batch boyutu
            
        Returns:
            (eğitilmiş_model, eğitim_geçmişi) tuple'ı
        """
        logger.info(f"\n🚀 {pair_name} {self.model_type.upper()} Eğitimi ({self.target_mode})...")
        
        # Model oluştur
        model = create_model(self.model_type, config, X.shape[2], self.device)
        model_info = get_model_info(model)
        logger.info(f"   🤖 {pair_name} Model: {model_info['total_parameters']:,} parametreler")
        
        # Model tipine göre veri bölme oranları
        if self.model_type in ['transformer', 'enhanced_transformer']:
            train_split, val_split = 0.65, 0.20  # Transformer'lar için daha fazla validation
        else:
            train_split, val_split = 0.70, 0.15  # Standart bölme
        
        # DataLoader'ları oluştur
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, 
            train_split=train_split, 
            val_split=val_split,
            batch_size=batch_size, 
            shuffle_train=True,
            num_workers=2 if torch.cuda.is_available() else 0
        )
        
        # Model kayıt yolu
        model_save_path = f'{pair_name}_best_{self.model_type}_model.pth'
        
        # ==================== ENHANCED TRANSFORMER BRANCH ====================
        if self.model_type == 'enhanced_transformer':
            # Enhanced Transformer için özel eğitim fonksiyonu
            logger.info("   ⚡ Enhanced Transformer eğitimi başlıyor...")
            
            # Y_train hesapla (class imbalance için)
            train_size = int(len(X) * train_split)
            y_train = y[:train_size] if not config.get('shuffle_train', True) else None
            
            history = train_enhanced_transformer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                y_train=y_train,
                epochs=epochs,
                patience=12,  # Enhanced Transformer için daha fazla sabır
                min_delta=0.0005,
                model_save_path=model_save_path,
                use_focal_loss=self.use_focal_loss,
                target_mode=self.target_mode
            )
        # ==================== STANDARD TRANSFORMER BRANCH ====================
        elif self.model_type == 'transformer':
            # Model-specific loss fonksiyonu ve optimizer
            criterion = self._get_enhanced_criterion(y, pair_name)
            optimizer, scheduler = self._get_enhanced_optimizer_and_scheduler(
                model, config, len(train_loader), epochs
            )
            
            history = train_transformer_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                epochs=epochs,
                patience=8,
                min_delta=0.001,
                scheduler=scheduler,
                model_save_path=model_save_path
            )
        # ==================== LSTM BRANCH ==================== 
        else:
            # Model-specific loss fonksiyonu ve optimizer
            criterion = self._get_enhanced_criterion(y, pair_name)
            optimizer, scheduler = self._get_enhanced_optimizer_and_scheduler(
                model, config, len(train_loader), epochs
            )
            
            # LSTM için geleneksel eğitim
            history = self._train_lstm_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=epochs,
                patience=5,
                model_save_path=model_save_path
            )
        
        # Sonuçları sakla
        self.pair_models[pair_name] = model
        self.pair_histories[pair_name] = history
        
        # Final istatistikler
        best_val_acc = max(history['val_acc'])
        final_train_acc = history['train_acc'][-1]
        final_overfitting_gap = final_train_acc - best_val_acc
        
        logger.info(f"   ✅ {pair_name} {self.model_type.upper()} Eğitimi tamamlandı!")
        logger.info(f"   📊 En İyi Val Acc: {best_val_acc:.2f}% | Son Train: {final_train_acc:.2f}%")
        logger.info(f"   📈 Overfitting Gap: {final_overfitting_gap:.2f}pp")
        
        return model, history

    # ... (Diğer metodlar _get_enhanced_criterion, _get_enhanced_optimizer_and_scheduler, 
    # _train_lstm_model, vb. aynı kalacak) ...


def create_unified_trainer(config: Dict[str, Any], device: torch.device) -> UnifiedTrainer:
    """
    Yapılandırmadan unified trainer oluştur.
    
    Args:
        config: Yapılandırma sözlüğü
        device: PyTorch cihazı
        
    Returns:
        Başlatılmış trainer instance'ı
    """
    model_config = config.get('model', {})
    
    return UnifiedTrainer(
        device=device,
        model_type=model_config.get('type', 'lstm'),
        use_focal_loss=model_config.get('use_focal_loss', True),
        target_mode=model_config.get('target_mode', 'binary'),
        reproducibility_seed=42
    )


__all__ = ['UnifiedTrainer', 'create_unified_trainer']
