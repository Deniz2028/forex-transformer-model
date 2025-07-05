"""
Optimized Transformer-based model for financial time series classification.

Bu modül, forex fiyat hareketi tahmini için geliştirilmiş Transformer mimarisi içerir.
Model kapasitesi artırılmış, cosine scheduler eklanmiş ve kayıp fonksiyonu iyileştirilmiştir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinüsoidal pozisyonel encoding implementasyonu.
    
    Args:
        d_model: Model boyutu (embedding boyutu)
        max_seq_length: Desteklenecek maksimum sekans uzunluğu
        dropout: Regularizasyon için dropout oranı
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pozisyonel encoding matrisini oluştur
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Sinüsoidal encoding için div_term oluştur
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Çift indekslere sine uygula
        pe[:, 0::2] = torch.sin(position * div_term)
        # Tek indekslere cosine uygula
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Batch boyutu ekle ve buffer olarak kaydet
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_seq_length, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Giriş embedding'lerine pozisyonel encoding ekle.
        
        Args:
            x: [seq_len, batch_size, d_model] şeklinde giriş tensörü
            
        Returns:
            Pozisyonel encoding eklenmiş tensör
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EnhancedTransformer(nn.Module):
    """
    Geliştirilmiş Transformer modeli ile sınıflandırma.
    
    Değişiklikler:
    - Model kapasitesi artırıldı (d_model, nhead, num_layers, ff_dim)
    - Pre-norm mimarisi eklendi (daha stabil eğitim)
    - Çoklu sınıf desteği (binary/three_class)
    
    Args:
        input_size: Giriş özellik sayısı
        d_model: Model boyutu (embedding boyutu)
        nhead: Attention head sayısı
        num_layers: Transformer katman sayısı
        ff_dim: Feedforward ağ boyutu
        dropout: Dropout oranı
        target_mode: 'binary' veya 'three_class'
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 12,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1,
        target_mode: str = 'binary'
    ):
        super(EnhancedTransformer, self).__init__()
        
        # Parametreleri sakla
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.target_mode = target_mode
        
        # Giriş projeksiyon katmanı
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Pozisyonel encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, 
            max_seq_length=5000,
            dropout=dropout
        )
        
        # Transformer encoder katmanları (Pre-norm mimarisi)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,  # Giriş [seq_len, batch, d_model]
            norm_first=True     # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Sınıflandırıcı çıkış katmanları
        if target_mode == 'three_class':
            output_size = 3
        else:
            output_size = 1  # Binary classification
            
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        # Ağırlıkları başlat
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot başlatma ile model ağırlıklarını başlat."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Geliştirilmiş Transformer modeli için forward pass.
        
        Args:
            x: [batch_size, seq_len, input_size] şeklinde giriş tensörü
            
        Returns:
            Sınıflandırma çıktıları (logits)
        """
        batch_size, seq_len, _ = x.shape
        
        # Giriş projeksiyonu
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)  # Ölçeklendirme
        
        # Transformer için boyutları düzenle [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        # Pozisyonel encoding ekle
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer(x)  # [seq_len, batch_size, d_model]
        
        # Son zaman adımının çıktısını al
        last_output = x[-1]  # [batch_size, d_model]
        
        # Sınıflandırıcı
        logits = self.classifier(last_output)  # [batch_size, output_size]
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Model bilgilerini döndürür.
        
        Returns:
            Model parametrelerini içeren sözlük
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'EnhancedTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'target_mode': self.target_mode
        }


def create_data_loaders(X: np.ndarray, y: np.ndarray, 
                       train_split: float = 0.7, val_split: float = 0.15,
                       batch_size: int = 32, shuffle_train: bool = True,
                       num_workers: int = 0) -> Tuple[data_utils.DataLoader, ...]:
    """
    TensorDataset ve DataLoader kullanarak veri yükleyicileri oluştur.
    
    Args:
        X: Giriş özellikleri
        y: Hedef etiketler
        train_split: Eğitim verisi oranı
        val_split: Doğrulama verisi oranı
        batch_size: Batch boyutu
        shuffle_train: Eğitim verisini karıştır
        num_workers: DataLoader worker sayısı
        
    Returns:
        (train_loader, val_loader, test_loader) tuple'ı
    """
    total_size = len(X)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    # Zaman sırasını koruyarak böl (finansal veriler için kritik)
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    # TensorDataset'ler oluştur
    train_dataset = data_utils.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = data_utils.TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    test_dataset = data_utils.TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    # DataLoader'lar oluştur
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = data_utils.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = data_utils.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"📊 DataLoaders oluşturuldu:")
    logger.info(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def set_reproducibility(seed: int = 42):
    """
    Tekrarlanabilirlik için tüm rastgelelik kaynaklarını ayarla.
    
    Args:
        seed: Rastgelelik tohumu
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CUDA deterministic ayarları
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"🔧 Tekrarlanabilirlik ayarlandı (seed={seed})")


def get_cosine_scheduler(optimizer: torch.optim.Optimizer, 
                         max_epochs: int, 
                         warmup_epochs: int = 5) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup dönemi içeren cosine learning rate scheduler.
    
    Args:
        optimizer: Optimize edici
        max_epochs: Toplam epoch sayısı
        warmup_epochs: Warmup için epoch sayısı
        
    Returns:
        Cosine scheduler
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_weighted_bce_loss(y_train: np.ndarray) -> nn.Module:
    """
    Sınıf dağılımına göre ağırlıklandırılmış BCE loss.
    
    Args:
        y_train: Eğitim etiketleri
        
    Returns:
        Ağırlıklandırılmış BCEWithLogitsLoss
    """
    pos_count = (y_train == 1).sum().item()
    neg_count = (y_train == 0).sum().item()
    total = len(y_train)
    
    # Pozitif sınıf için ağırlık
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)])
    
    logger.info(f"⚠️ Weighted BCE Loss: pos_weight={pos_weight.item():.2f} "
               f"(pos={pos_count}, neg={neg_count})")
    
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_epoch(model: EnhancedTransformer, 
                train_loader: data_utils.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float, float]:
    """
    Tek eğitim epoch'u çalıştır.
    
    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        optimizer: Optimize edici
        criterion: Loss fonksiyonu
        device: Hesaplama cihazı
        scheduler: Learning rate scheduler (opsiyonel)
        
    Returns:
        (ortalama_loss, doğruluk_oranı, roc_auc) tuple'ı
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_probs = []
    
    # Progress bar için tqdm kullan
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_X, batch_y in pbar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Hedef boyutlarını ayarla
        if len(batch_y.shape) == 1:
            batch_y = batch_y.unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        
        # Loss hesapla
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (Transformer'lar için kritik)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Scheduler güncelleme (batch bazlı değil)
        if scheduler:
            scheduler.step()
        
        # Metrikleri hesapla
        total_loss += loss.item()
        
        # Tahminler ve olasılıklar
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)
            
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Progress bar güncelle
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * correct_predictions / total_samples:.1f}%'
        })
    
    # ROC-AUC hesapla
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception as e:
        logger.warning(f"ROC-AUC hesaplanamadı: {str(e)}")
        roc_auc = 0.5
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy, roc_auc


def validate_epoch(model: EnhancedTransformer,
                   val_loader: data_utils.DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, float]:
    """
    Doğrulama epoch'u çalıştır.
    
    Args:
        model: Değerlendirilecek model
        val_loader: Doğrulama veri yükleyicisi
        criterion: Loss fonksiyonu
        device: Hesaplama cihazı
        
    Returns:
        (ortalama_loss, doğruluk_oranı, roc_auc) tuple'ı
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Hedef boyutlarını ayarla
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Metrikleri hesapla
            total_loss += loss.item()
            
            # Tahminler ve olasılıklar
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)
            
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Progress bar güncelle
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.1f}%'
            })
    
    # ROC-AUC hesapla
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception as e:
        logger.warning(f"ROC-AUC hesaplanamadı: {str(e)}")
        roc_auc = 0.5
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    return avg_loss, accuracy, roc_auc


def train_enhanced_transformer(
    model: EnhancedTransformer,
    train_loader: data_utils.DataLoader,
    val_loader: data_utils.DataLoader,
    device: torch.device,
    y_train: np.ndarray,
    epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.001,
    model_save_path: str = "best_model.pth"
) -> Dict[str, list]:
    """
    Geliştirilmiş Transformer modelini eğit.
    
    Değişiklikler:
    - AdamW optimizer ve düşük learning rate
    - Cosine scheduler with warmup
    - Weighted BCE loss
    - ROC-AUC metrik desteği
    
    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        device: Hesaplama cihazı
        y_train: Eğitim etiketleri (loss ağırlıkları için)
        epochs: Maksimum epoch sayısı
        patience: Early stopping için sabır
        min_delta: İyileşme için minimum değişim
        model_save_path: Model kayıt yolu
        
    Returns:
        Eğitim geçmişi sözlüğü
    """
    logger.info(f"🚀 Geliştirilmiş Transformer eğitimi başlıyor...")
    logger.info(f"   Epochs: {epochs}, Patience: {patience}")
    logger.info(f"   Device: {device}")
    
    # Optimizer (AdamW with Transformer-optimized params)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-2
    )
    
    # Cosine scheduler with warmup
    scheduler = get_cosine_scheduler(
        optimizer, 
        max_epochs=epochs, 
        warmup_epochs=5
    )
    
    # Weighted BCE loss
    criterion = get_weighted_bce_loss(y_train)
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_roc_auc': [],
        'val_roc_auc': [],
        'learning_rates': []
    }
    
    # Early stopping ve checkpoint değişkenleri
    best_val_acc = 0.0
    best_val_roc_auc = 0.5
    best_state_dict = None
    patience_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"\n📊 Epoch {epoch + 1}/{epochs}")
        
        # Eğitim epoch'u
        train_loss, train_acc, train_roc_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Doğrulama epoch'u
        val_loss, val_acc, val_roc_auc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        
        # History güncelle
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_roc_auc'].append(train_roc_auc)
        history['val_roc_auc'].append(val_roc_auc)
        history['learning_rates'].append(current_lr)
        
        # Early stopping ve best model checkpoint
        val_improvement = val_roc_auc - best_val_roc_auc
        
        if val_improvement > min_delta:
            best_val_acc = val_acc
            best_val_roc_auc = val_roc_auc
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
            
            # Model kaydet
            torch.save(best_state_dict, model_save_path)
            logger.info(f"   💾 Best model saved: Acc={val_acc:.1f}%, ROC-AUC={val_roc_auc:.4f}")
        else:
            patience_counter += 1
        
        # Epoch sonuçlarını logla
        logger.info(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%, ROC-AUC={train_roc_auc:.4f}")
        logger.info(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}%, ROC-AUC={val_roc_auc:.4f}")
        logger.info(f"   LR: {current_lr:.2e}, Best ROC-AUC: {best_val_roc_auc:.4f}")
        
        # Early stopping kontrolü
        if patience_counter >= patience:
            logger.info(f"   ⏰ Early stopping: {patience} epoch boyunca iyileşme yok")
            break
        
        # Learning rate minimum kontrolü
        if current_lr < 1e-7:
            logger.info(f"   📉 Stopping: Learning rate çok düşük ({current_lr:.2e})")
            break
    
    # En iyi model ağırlıklarını geri yükle
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"   ✅ En iyi model yüklendi (Val Acc: {best_val_acc:.1f}%, ROC-AUC: {best_val_roc_auc:.4f})")
    
    logger.info(f"🎉 Eğitim tamamlandı! En iyi doğrulama ROC-AUC: {best_val_roc_auc:.4f}")
    
    return history


def create_enhanced_transformer(config: Dict[str, Any], 
                               n_features: int, 
                               device: torch.device) -> EnhancedTransformer:
    """
    Yapılandırmadan geliştirilmiş Transformer modeli oluştur.
    
    Args:
        config: Model yapılandırma sözlüğü
        n_features: Giriş özellik sayısı
        device: Model oluşturulacak cihaz
        
    Returns:
        Başlatılmış EnhancedTransformer modeli
    """
    model_config = config.get('model', {})
    transformer_config = config.get('transformer', {})
    
    # Geliştirilmiş parametreler
    d_model = transformer_config.get('d_model', 256)
    nhead = transformer_config.get('nhead', 12)
    num_layers = transformer_config.get('num_layers', 6)
    ff_dim = transformer_config.get('ff_dim', 512)
    dropout = model_config.get('dropout_rate', 0.1)
    target_mode = model_config.get('target_mode', 'binary')
    
    # nhead bölünebilirlik doğrulaması
    if d_model % nhead != 0:
        raise ValueError(f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı")
    
    model = EnhancedTransformer(
        input_size=n_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        target_mode=target_mode
    )
    
    return model.to(device)


def add_time_features(df):
    """
    Zaman bazlı özellikler ekler (feature engineering).
    
    Args:
        df: Finansal veri çerçevesi (DateTime index'li)
        
    Returns:
        Özellik eklenmiş veri çerçevesi
    """
    # Saat bilgisi (sin-cos encoding)
    hour = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Hafta içi/sonu
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)
    
    # ATR normalizasyonu
    if 'atr' in df.columns:
        df['atr_normalized'] = df['atr'] / df['close']
    
    return df


__all__ = [
    'EnhancedTransformer',
    'PositionalEncoding', 
    'create_enhanced_transformer',
    'create_data_loaders',
    'set_reproducibility',
    'train_enhanced_transformer',
    'add_time_features',
    'get_weighted_bce_loss'
]
