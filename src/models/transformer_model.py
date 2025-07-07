"""
Optimized Transformer-based models for financial time series classification.

Bu modül, forex fiyat hareketi tahmini için geliştirilmiş iki Transformer mimarisi içerir:
1. TransformerClassifier: Standart Transformer implementasyonu
2. EnhancedTransformer: Gelişmiş özelliklerle optimize edilmiş Transformer

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


class TransformerClassifier(nn.Module):
    """
    Standard Transformer-based classifier for financial time series.
    
    Bu sınıf, EnhancedTransformer'dan farklı olarak daha basit ve standart
    Transformer mimarisi kullanır. Mevcut proje yapısıyla uyumlu şekilde tasarlanmıştır.
    
    Args:
        input_size: Giriş özellik sayısı
        d_model: Model boyutu (embedding boyutu) 
        nhead: Attention head sayısı
        num_layers: Transformer katman sayısı
        ff_dim: Feedforward ağ boyutu
        dropout: Dropout oranı
        target_mode: 'binary' veya 'three_class'
        max_seq_length: Maksimum sekans uzunluğu
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8, 
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        target_mode: str = 'binary',
        max_seq_length: int = 1000
    ):
        super(TransformerClassifier, self).__init__()
        
        # Parametreleri sakla
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.target_mode = target_mode
        self.input_size = input_size
        
        # d_model'in nhead'e bölünebilir olduğunu kontrol et
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        # Giriş projeksiyon katmanı
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Pozisyonel encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Transformer encoder katmanları
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,  # [seq_len, batch, d_model] format
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Sınıflandırıcı katmanları
        if target_mode == 'three_class':
            output_size = 3
        else:
            output_size = 1
            
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        # Ağırlıkları başlat
        self._init_weights()
    
    def _init_weights(self):
        """Xavier başlatma ile model ağırlıklarını başlat."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_len, input_size] şeklinde giriş tensörü
            
        Returns:
            Sınıflandırma çıktısı
        """
        batch_size, seq_len, _ = x.shape
        
        # Giriş projeksiyon: [batch, seq_len, input_size] -> [batch, seq_len, d_model]
        x = self.input_projection(x)
        
        # Transformer format: [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        
        # Pozisyonel encoding ekle
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer(x)  # [seq_len, batch, d_model]
        
        # Global average pooling: [seq_len, batch, d_model] -> [batch, d_model] 
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]
        
        # Sınıflandırma
        output = self.classifier(x)
        
        # Binary classification için sigmoid uygula
        if self.target_mode == 'binary':
            output = torch.sigmoid(output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Attention ağırlıklarını çıkar (analiz için).
        
        Args:
            x: Giriş tensörü
            
        Returns:
            Her katmandaki attention ağırlıkları listesi
        """
        attention_weights = []
        
        # Giriş projeksiyon
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        
        # Her transformer katmanından attention ağırlıklarını çıkar
        for layer in self.transformer.layers:
            # Self-attention
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights.append(attn_weights)
            
            # Normalization ve feedforward
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))
        
        return attention_weights


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
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass için gelişmiş implementasyon.
        
        Args:
            x: [batch_size, seq_len, input_size] şeklinde giriş tensörü
            
        Returns:
            Sınıflandırma çıktısı [batch_size, output_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Giriş projeksiyon: input_size -> d_model
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Transformer beklenen format: [seq_len, batch, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        
        # Pozisyonel encoding ekle
        x = self.pos_encoder(x)
        
        # Transformer encoder katmanları
        transformer_output = self.transformer(x)  # [seq_len, batch, d_model]
        
        # Global pooling: Son timestep'i al (veya ortalama)
        # Son timestep: [batch, d_model]
        pooled_output = transformer_output[-1]  # [batch, d_model]
        
        # Sınıflandırıcı
        logits = self.classifier(pooled_output)  # [batch, output_size]
        
        # Binary classification için sigmoid
        if self.target_mode == 'binary':
            return torch.sigmoid(logits)
        else:
            return logits  # Cross-entropy loss softmax'ı kendisi uygular


# Factory Functions
def create_transformer_model(config: Dict[str, Any], n_features: int, device: torch.device) -> TransformerClassifier:
    """
    TransformerClassifier modeli oluşturan factory fonksiyonu.
    
    Args:
        config: Model konfigürasyonu
        n_features: Giriş özellik sayısı
        device: PyTorch cihazı
        
    Returns:
        Oluşturulan TransformerClassifier modeli
    """
    # Transformer specific config
    transformer_config = config.get('transformer', {})
    
    model = TransformerClassifier(
        input_size=n_features,
        d_model=transformer_config.get('d_model', 128),
        nhead=transformer_config.get('nhead', 8),
        num_layers=transformer_config.get('num_layers', 3),
        ff_dim=transformer_config.get('ff_dim', 256),
        dropout=transformer_config.get('dropout_rate', 0.1),
        target_mode=config.get('model', {}).get('target_mode', 'binary'),
        max_seq_length=transformer_config.get('max_seq_len', 1000)
    )
    
    return model.to(device)


def create_enhanced_transformer(config: Dict[str, Any], n_features: int, device: torch.device) -> EnhancedTransformer:
    """
    EnhancedTransformer modeli oluşturan factory fonksiyonu.
    
    Args:
        config: Model konfigürasyonu
        n_features: Giriş özellik sayısı
        device: PyTorch cihazı
        
    Returns:
        Oluşturulan EnhancedTransformer modeli
    """
    # Enhanced Transformer specific config
    transformer_config = config.get('transformer', {})
    
    model = EnhancedTransformer(
        input_size=n_features,
        d_model=transformer_config.get('d_model', 256),
        nhead=transformer_config.get('nhead', 12),
        num_layers=transformer_config.get('num_layers', 6),
        ff_dim=transformer_config.get('ff_dim', 512),
        dropout=transformer_config.get('dropout_rate', 0.1),
        target_mode=config.get('model', {}).get('target_mode', 'binary')
    )
    
    return model.to(device)


# Utility Functions
def create_data_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                       train_split: float = 0.8) -> Tuple[data_utils.DataLoader, data_utils.DataLoader]:
    """
    PyTorch DataLoader'ları oluştur.
    
    Args:
        X: Özellik matrisi
        y: Hedef değişken
        batch_size: Batch boyutu
        train_split: Eğitim veri oranı
        
    Returns:
        (train_loader, val_loader) tuple'ı
    """
    # Veriyi böl
    split_idx = int(len(X) * train_split)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Tensor'a çevir
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Dataset'ler oluştur
    train_dataset = data_utils.TensorDataset(X_train, y_train)
    val_dataset = data_utils.TensorDataset(X_val, y_val)
    
    # DataLoader'lar oluştur
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = data_utils.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader


def set_reproducibility(seed: int = 42):
    """
    Tekrarlanabilirlik için seed ayarla.
    
    Args:
        seed: Rastgelelik seed'i
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_transformer_model(model: TransformerClassifier, train_loader: data_utils.DataLoader,
                           val_loader: data_utils.DataLoader, config: Dict[str, Any],
                           device: torch.device) -> Tuple[TransformerClassifier, Dict[str, List]]:
    """
    TransformerClassifier modelini eğiten fonksiyon.
    
    Args:
        model: Eğitilecek model
        train_loader: Eğitim DataLoader'ı
        val_loader: Validasyon DataLoader'ı
        config: Eğitim konfigürasyonu
        device: PyTorch cihazı
        
    Returns:
        (eğitilmiş_model, eğitim_geçmişi) tuple'ı
    """
    # Eğitim parametreleri
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 30)
    learning_rate = training_config.get('learning_rate', 1e-4)
    patience = training_config.get('patience', 10)
    
    # Optimizer ve loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    if model.target_mode == 'binary':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )
    
    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"🚀 TransformerClassifier eğitimi başlıyor...")
    logger.info(f"   Epochs: {epochs}, LR: {learning_rate}")
    
    for epoch in range(epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            if model.target_mode == 'binary':
                loss = criterion(output.squeeze(), target)
                pred = (output.squeeze() > 0.5).float()
            else:
                loss = criterion(output, target.long())
                pred = output.argmax(dim=1).float()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # İstatistikler
            train_loss += loss.item()
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Validasyon
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if model.target_mode == 'binary':
                    loss = criterion(output.squeeze(), target)
                    pred = (output.squeeze() > 0.5).float()
                else:
                    loss = criterion(output, target.long())
                    pred = output.argmax(dim=1).float()
                
                val_loss += loss.item()
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # Ortalamalar
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Geçmişe ekle
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping kontrolü
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, history
