"""
Geliştirilmiş model factory for creating different model architectures.

Bu modül, LSTM, EnhancedTransformer ve diğer modelleri oluşturmak için birleşik arayüz sağlar.
Model seçimi, parametre validasyonu, cihaz yerleştirme ve gelişmiş hata yönetimi
ile kapsamlı konfigürasyon doğrulaması içerir.
"""

import torch
import copy
import logging
import importlib
from typing import Dict, Any, Union, List, Tuple, Type

# Logger yapılandırması - EN BAŞTA TANIMLA
logger = logging.getLogger(__name__)

# LSTM modeli için import (her zaman mevcut)
from .lstm import PairSpecificLSTM, create_model as create_lstm_model

# Transformer modeli için koşullu import
TRANSFORMER_AVAILABLE = False
ENHANCED_TRANSFORMER_V2_AVAILABLE = False
EnhancedTransformer = None
TransformerClassifier = None
create_transformer_model = None
create_enhanced_transformer = None
EnhancedTransformerV2 = None
create_enhanced_transformer_v2 = None

# Transformer modülünü dinamik olarak yükleme
try:
    transformer_module = importlib.import_module('.transformer_model', package='src.models')
    
    # EnhancedTransformer var mı?
    if hasattr(transformer_module, 'EnhancedTransformer'):
        from .transformer_model import EnhancedTransformer, create_enhanced_transformer
        TRANSFORMER_AVAILABLE = True
        logger.info("✅ EnhancedTransformer import başarılı")
    
    # TransformerClassifier var mı?
    if hasattr(transformer_module, 'TransformerClassifier'):
        from .transformer_model import TransformerClassifier, create_transformer_model
        logger.info("✅ TransformerClassifier import başarılı")
    else:
        logger.warning("⚠️ TransformerClassifier bulunamadı")
        
except ImportError as e:
    logger.warning(f"⚠️ Transformer model import hatası: {e}")
except Exception as e:
    logger.error(f"❌ Beklenmeyen hata (transformer_model): {e}")

# Enhanced Transformer V2 için import
try:
    enhanced_module = importlib.import_module('.enhanced_transformer', package='src.models')
    
    if hasattr(enhanced_module, 'EnhancedTransformer'):
        from .enhanced_transformer import EnhancedTransformer as EnhancedTransformerV2
        from .enhanced_transformer import create_enhanced_transformer as create_enhanced_transformer_v2
        ENHANCED_TRANSFORMER_V2_AVAILABLE = True
        logger.info("✅ Enhanced Transformer V2 import başarılı")
        
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Transformer V2 import hatası: {e}")
except Exception as e:
    logger.error(f"❌ Beklenmeyen hata (enhanced_transformer): {e}")

# Model tipi sabitleri - GÜVENLİ HALE GETİR
SUPPORTED_MODELS = ['lstm', 'pairspecificlstm']

# Mevcut modellere göre desteklenen tipleri ekle
if TRANSFORMER_AVAILABLE:
    SUPPORTED_MODELS.extend(['transformer', 'enhanced_transformer'])
    logger.info("✅ Transformer modelleri desteklenen listeye eklendi")

if ENHANCED_TRANSFORMER_V2_AVAILABLE:
    SUPPORTED_MODELS.append('enhanced_transformer_v2')
    logger.info("✅ Enhanced Transformer V2 desteklenen listeye eklendi")

logger.info(f"📋 Desteklenen modeller: {SUPPORTED_MODELS}")

# Model alias'ları
LSTM_ALIASES = ['lstm', 'pairspecificlstm']
TRANSFORMER_ALIASES = ['transformer', 'enhanced_transformer'] if TRANSFORMER_AVAILABLE else []
ENHANCED_TRANSFORMER_V2_ALIASES = ['enhanced_transformer_v2'] if ENHANCED_TRANSFORMER_V2_AVAILABLE else []

# ModelInstance tipini güvenli hale getir
available_types = [PairSpecificLSTM]

if TRANSFORMER_AVAILABLE:
    if EnhancedTransformer:
        available_types.append(EnhancedTransformer)
    if TransformerClassifier:
        available_types.append(TransformerClassifier)

if ENHANCED_TRANSFORMER_V2_AVAILABLE and EnhancedTransformerV2:
    available_types.append(EnhancedTransformerV2)

if len(available_types) == 0:
    logger.error("⚠️ Hiçbir model tipi kullanılamıyor! Sadece LSTM kullanılacak.")
    available_types = [PairSpecificLSTM]

ModelInstance = Union[tuple(available_types)]

# Tip alias'ları (daha açıklayıcı tip notasyonları için)
ModelConfig = Dict[str, Any]
TransformerConfig = Dict[str, Any]
LSTMConfig = Dict[str, Any]


def create_model(
    model_type: str,
    config: ModelConfig, 
    n_features: int, 
    device: torch.device
) -> ModelInstance:
    """
    Tip ve konfigürasyona göre model oluşturan factory fonksiyonu.
    
    Args:
        model_type: Oluşturulacak model tipi ('lstm', 'transformer', 'enhanced_transformer', 'enhanced_transformer_v2')
        config: Model konfigürasyon sözlüğü
        n_features: Giriş özellik sayısı
        device: Model oluşturulacak cihaz
        
    Returns:
        Başlatılmış model instance'ı
        
    Raises:
        ValueError: Model tipi desteklenmiyorsa veya konfigürasyon geçersizse
        RuntimeError: Model oluşturma sırasında beklenmeyen hata
    """
    model_type_normalized = model_type.lower().strip()
    
    logger.info(f"🏭 Model factory: {model_type_normalized} modeli oluşturuluyor...")
    logger.info(f"   Giriş özellikleri: {n_features}")
    logger.info(f"   Cihaz: {device}")
    
    # Model tipi validasyonu
    if model_type_normalized not in SUPPORTED_MODELS:
        raise ValueError(
            f"Desteklenmeyen model tipi: '{model_type}'. "
            f"Desteklenen modeller: {SUPPORTED_MODELS}"
        )
    
    try:
        if model_type_normalized in LSTM_ALIASES:
            # LSTM konfigürasyonu validasyonu
            validated_config = validate_lstm_config(config)
            model = create_lstm_model(validated_config, n_features, device)
            logger.info(f"   ✅ LSTM modeli başarıyla oluşturuldu")
            
        elif model_type_normalized in TRANSFORMER_ALIASES:
            if not TRANSFORMER_AVAILABLE:
                raise RuntimeError("Transformer modeli desteklenmiyor (import hatası)")
                
            # Transformer konfigürasyonu validasyonu
            validated_config = validate_transformer_config(config)
            
            if model_type_normalized == 'transformer' and create_transformer_model:
                model = create_transformer_model(validated_config, n_features, device)
                logger.info(f"   ✅ Transformer modeli başarıyla oluşturuldu")
            elif model_type_normalized == 'enhanced_transformer' and create_enhanced_transformer:
                model = create_enhanced_transformer(validated_config, n_features, device)
                logger.info(f"   ✅ EnhancedTransformer modeli başarıyla oluşturuldu")
            else:
                raise RuntimeError(f"İstenen model tipi desteklenmiyor: {model_type}")
                
        elif model_type_normalized in ENHANCED_TRANSFORMER_V2_ALIASES:
            if not ENHANCED_TRANSFORMER_V2_AVAILABLE or not create_enhanced_transformer_v2:
                raise RuntimeError("Enhanced Transformer V2 modeli desteklenmiyor (import hatası)")
                
            # EnhancedTransformer V2 konfigürasyonu validasyonu
            validated_config = validate_enhanced_transformer_config(config)
            model = create_enhanced_transformer_v2(validated_config, n_features, device)
            logger.info(f"   ✅ EnhancedTransformer V2 modeli başarıyla oluşturuldu")
        
        else:
            # Bu duruma teorik olarak gelmemeli ama güvenlik için
            raise ValueError(f"Model tipi '{model_type}' işlenemiyor")
        
        # Model bilgilerini logla
        model_info = get_model_info(model)
        logger.info(f"   📊 Model parametreleri: {model_info['total_parameters']:,}")
        
        return model
        
    except ValueError as e:
        logger.error(f"   ❌ Model oluşturma hatası: {e}")
        raise
    except Exception as e:
        logger.error(f"   ❌ Beklenmeyen hata: {e}")
        raise RuntimeError(f"Model oluşturma başarısız: {e}") from e


def validate_transformer_config(config: TransformerConfig) -> TransformerConfig:
    """
    Transformer konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    # Orijinal sözlüğü bozmamak için deep copy
    config = copy.deepcopy(config)
    
    transformer_config = config.get('transformer', {})
    
    d_model = transformer_config.get('d_model', 128)
    nhead = transformer_config.get('nhead', 8)
    num_layers = transformer_config.get('num_layers', 4)
    dim_feedforward = transformer_config.get('dim_feedforward', 256)
    dropout = config.get('model', {}).get('dropout_rate', 0.1)
    
    logger.info(f"🔍 Transformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # d_model validasyonu
    if not isinstance(d_model, int) or d_model < 32 or d_model > 1024:
        raise ValueError(
            f"d_model geçersiz: {d_model}. "
            f"32 ile 1024 arasında integer olmalı"
        )
    
    # nhead validasyonu
    if not isinstance(nhead, int) or nhead < 1 or nhead > 32:
        raise ValueError(
            f"nhead geçersiz: {nhead}. "
            f"1 ile 32 arasında integer olmalı"
        )
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        logger.warning(f"   ⚠️ d_model ({d_model}) nhead ({nhead}) ile bölünemiyor")
        
        # Otomatik düzeltme denemesi
        valid_heads = [h for h in [1, 2, 4, 8, 16, 32] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            transformer_config['nhead'] = nhead
            logger.warning(f"   🔧 nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(
                f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı. "
                f"d_model % nhead == 0 koşulu sağlanmalı. "
                f"Geçerli nhead değerleri: {[h for h in [1, 2, 4, 8, 16, 32] if d_model % h == 0]}"
            )
    
    # num_layers validasyonu
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 12:
        raise ValueError(
            f"num_layers geçersiz: {num_layers}. "
            f"1 ile 12 arasında integer olmalı"
        )
    
    # dim_feedforward validasyonu
    if not isinstance(dim_feedforward, int) or dim_feedforward < 64 or dim_feedforward > 4096:
        raise ValueError(
            f"dim_feedforward geçersiz: {dim_feedforward}. "
            f"64 ile 4096 arasında integer olmalı"
        )
    
    # dim_feedforward ve d_model ilişkisi
    if dim_feedforward < d_model:
        logger.warning(
            f"   ⚠️ dim_feedforward ({dim_feedforward}) < d_model ({d_model}). "
            f"Genellikle dim_feedforward >= d_model olması önerilir"
        )
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.9:
        raise ValueError(
            f"dropout geçersiz: {dropout}. "
            f"0.0 ile 0.9 arasında float olmalı"
        )
    
    # Performans uyarıları
    if d_model > 512:
        logger.warning(f"   ⚠️ Büyük d_model ({d_model}) önemli GPU belleği gerektirebilir")
    
    if num_layers > 8:
        logger.warning(f"   ⚠️ Çok katman ({num_layers}) eğitimi yavaşlatabilir")
    
    if nhead > 16:
        logger.warning(f"   ⚠️ Çok attention head ({nhead}) hesaplama maliyeti artırabilir")
    
    logger.info(f"   ✅ Transformer konfigürasyonu doğrulandı")
    
    return config


def validate_enhanced_transformer_config(config: TransformerConfig) -> TransformerConfig:
    """
    EnhancedTransformer konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    # Orijinal sözlüğü bozmamak için deep copy
    config = copy.deepcopy(config)
    
    transformer_config = config.get('transformer', {})
    
    d_model = transformer_config.get('d_model', 256)
    nhead = transformer_config.get('nhead', 12)
    num_layers = transformer_config.get('num_layers', 6)
    ff_dim = transformer_config.get('ff_dim', 512)
    dropout = config.get('model', {}).get('dropout_rate', 0.1)
    target_mode = config.get('model', {}).get('target_mode', 'binary')
    
    logger.info(f"🔍 EnhancedTransformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # d_model validasyonu
    if not isinstance(d_model, int) or d_model < 64 or d_model > 2048:
        raise ValueError(
            f"d_model geçersiz: {d_model}. "
            f"64 ile 2048 arasında integer olmalı"
        )
    
    # nhead validasyonu
    if not isinstance(nhead, int) or nhead < 1 or nhead > 32:
        raise ValueError(
            f"nhead geçersiz: {nhead}. "
            f"1 ile 32 arasında integer olmalı"
        )
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        logger.warning(f"   ⚠️ d_model ({d_model}) nhead ({nhead}) ile bölünemiyor")
        
        # Otomatik düzeltme denemesi
        valid_heads = [h for h in [1, 2, 4, 8, 16, 32] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            transformer_config['nhead'] = nhead
            logger.warning(f"   🔧 nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(
                f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı. "
                f"d_model % nhead == 0 koşulu sağlanmalı. "
                f"Geçerli nhead değerleri: {[h for h in [1, 2, 4, 8, 16, 32] if d_model % h == 0]}"
            )
    
    # num_layers validasyonu
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 16:
        raise ValueError(
            f"num_layers geçersiz: {num_layers}. "
            f"1 ile 16 arasında integer olmalı"
        )
    
    # ff_dim validasyonu
    if not isinstance(ff_dim, int) or ff_dim < 128 or ff_dim > 8192:
        raise ValueError(
            f"ff_dim geçersiz: {ff_dim}. "
            f"128 ile 8192 arasında integer olmalı"
        )
    
    # ff_dim ve d_model ilişkisi
    if ff_dim < d_model * 2:
        logger.warning(
            f"   ⚠️ ff_dim ({ff_dim}) < d_model*2 ({d_model*2}). "
            f"Genellikle ff_dim >= d_model*2 olması önerilir"
        )
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.9:
        raise ValueError(
            f"dropout geçersiz: {dropout}. "
            f"0.0 ile 0.9 arasında float olmalı"
        )
    
    # target_mode validasyonu
    if target_mode not in ['binary', 'three_class']:
        raise ValueError(
            f"target_mode geçersiz: {target_mode}. "
            f"'binary' veya 'three_class' olmalı"
        )
    
    # Performans uyarıları
    if d_model > 512:
        logger.warning(f"   ⚠️ Büyük d_model ({d_model}) önemli GPU belleği gerektirebilir")
    
    if num_layers > 8:
        logger.warning(f"   ⚠️ Çok katman ({num_layers}) eğitimi yavaşlatabilir")
    
    if nhead > 16:
        logger.warning(f"   ⚠️ Çok attention head ({nhead}) hesaplama maliyeti artırabilir")
    
    logger.info(f"   ✅ EnhancedTransformer konfigürasyonu doğrulandı")
    
    return config


def validate_lstm_config(config: LSTMConfig) -> LSTMConfig:
    """
    LSTM konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    # Orijinal sözlüğü bozmamak için deep copy
    config = copy.deepcopy(config)
    
    model_config = config.get('model', {})
    
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout_rate', 0.45)
    
    logger.info(f"🔍 LSTM konfigürasyonu doğrulanıyor...")
    logger.info(f"   hidden_size: {hidden_size}, layers: {num_layers}, dropout: {dropout}")
    
    # hidden_size validasyonu
    if not isinstance(hidden_size, int) or hidden_size < 16 or hidden_size > 512:
        raise ValueError(
            f"hidden_size geçersiz: {hidden_size}. "
            f"16 ile 512 arasında integer olmalı"
        )
    
    # num_layers validasyonu  
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 4:
        raise ValueError(
            f"num_layers geçersiz: {num_layers}. "
            f"1 ile 4 arasında integer olmalı"
        )
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.8:
        raise ValueError(
            f"dropout geçersiz: {dropout}. "
            f"0.0 ile 0.8 arasında float olmalı"
        )
    
    # Performans uyarıları
    if hidden_size > 256:
        logger.warning(f"   ⚠️ Büyük hidden_size ({hidden_size}) overfitting riskini artırabilir")
    
    if num_layers > 3:
        logger.warning(f"   ⚠️ Çok katman ({num_layers}) gradient vanishing problemine yol açabilir")
    
    logger.info(f"   ✅ LSTM konfigürasyonu doğrulandı")
    
    return config


def get_model_info(model: ModelInstance) -> Dict[str, Any]:
    """
    Farklı mimarilerde birleşik model bilgisi al.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Model bilgilerini içeren sözlük
        
    Raises:
        RuntimeError: Model bilgisi alınamıyorsa
    """
    try:
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        else:
            # get_model_info metodu olmayan modeller için fallback
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"   ℹ️ Model get_model_info metoduna sahip değil, temel bilgiler kullanılıyor")
            
            return {
                'model_type': type(model).__name__,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'architecture': 'Unknown (fallback info)'
            }
    except Exception as e:
        logger.error(f"   ❌ Model bilgisi alınamadı: {e}")
        raise RuntimeError(f"Model bilgisi alınırken hata: {e}") from e


def get_model_complexity_score(model: ModelInstance) -> float:
    """
    Parametre sayısı ve mimariye göre model karmaşıklık skoru hesapla.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Karmaşıklık skoru (yüksek = daha karmaşık)
        
    Raises:
        RuntimeError: Skor hesaplanamıyorsa
    """
    try:
        info = get_model_info(model)
        param_count = info['total_parameters']
        
        # Temel karmaşıklık parametre sayısından
        complexity = param_count / 100000  # Makul aralığa normalize et
        
        # Mimari-specific ayarlamalar
        if TRANSFORMER_AVAILABLE and isinstance(model, (TransformerClassifier, EnhancedTransformer)):
            # Transformer'lar attention mekanizması nedeniyle daha karmaşık
            complexity *= 1.5
            
            # Head ve katman sayısına göre karmaşıklık ekle
            if hasattr(model, 'nhead'):
                complexity += model.nhead * 0.1
            if hasattr(model, 'num_layers'):
                complexity += model.num_layers * 0.2
            
            # EnhancedTransformer için ek karmaşıklık
            if hasattr(model, 'target_mode') and model.target_mode == 'three_class':
                complexity *= 1.1
            
            logger.info(f"   📊 Transformer karmaşıklık faktörleri uygulandı")
            
        elif ENHANCED_TRANSFORMER_V2_AVAILABLE and isinstance(model, EnhancedTransformerV2):
            # EnhancedTransformer V2 için karmaşıklık
            complexity *= 1.8
            
            if hasattr(model, 'nhead'):
                complexity += model.nhead * 0.15
            if hasattr(model, 'num_layers'):
                complexity += model.num_layers * 0.25
            if hasattr(model, 'target_mode') and model.target_mode == 'three_class':
                complexity *= 1.15
                
            logger.info(f"   📊 EnhancedTransformer V2 karmaşıklık faktörleri uygulandı")
            
        elif isinstance(model, PairSpecificLSTM):
            # LSTM karmaşıklığı katman ve hidden size'a göre
            if hasattr(model, 'num_layers'):
                complexity += model.num_layers * 0.1
            if hasattr(model, 'hidden_size'):
                complexity += model.hidden_size / 1000
            logger.info(f"   📊 LSTM karmaşıklık faktörleri uygulandı")
            
        else:
            # Bilinmeyen model tipi için default davranış
            logger.info(f"   ℹ️ Bilinmeyen model tipi ({type(model).__name__}), default skor kullanıldı")
        
        final_score = round(complexity, 2)
        logger.info(f"   📊 Model karmaşıklık skoru: {final_score}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"   ❌ Karmaşıklık skoru hesaplanamadı: {e}")
        raise RuntimeError(f"Karmaşıklık skoru hesaplama hatası: {e}") from e


def suggest_training_params(model: ModelInstance) -> Dict[str, Any]:
    """
    Model karmaşıklığı ve tipine göre eğitim parametreleri öner.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Önerilen eğitim parametreleri
        
    Raises:
        RuntimeError: Öneri oluşturulamıyorsa
    """
    try:
        complexity = get_model_complexity_score(model)
        info = get_model_info(model)
        
        logger.info(f"🎯 Eğitim parametreleri öneriliyor...")
        logger.info(f"   Model karmaşıklığı: {complexity}")
        
        if TRANSFORMER_AVAILABLE and isinstance(model, EnhancedTransformer):
            # EnhancedTransformer-specific öneriler
            
            # Model detaylarından ek bilgiler
            num_layers = getattr(model, 'num_layers', 6)
            nhead = getattr(model, 'nhead', 12)
            d_model = getattr(model, 'd_model', 256)
            target_mode = getattr(model, 'target_mode', 'binary')
            
            logger.info(f"   EnhancedTransformer detayları: layers={num_layers}, heads={nhead}, d_model={d_model}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 3.0 and num_layers <= 6:
                base_lr = 1e-4
                batch_size = 64
                warmup = 1000
            elif complexity < 8.0 or num_layers <= 8:
                base_lr = 5e-5
                batch_size = 32
                warmup = 2000
            else:
                base_lr = 1e-5
                batch_size = 16
                warmup = 4000
            
            # Attention head sayısına göre LR ayarlaması
            if nhead > 16:
                base_lr *= 0.8  # Çok head varsa LR düşür
                logger.info(f"   📉 Çok attention head nedeniyle LR düşürüldü")
            
            # d_model'e göre batch size ayarlaması
            if d_model > 512:
                batch_size = max(8, batch_size // 2)
                logger.info(f"   📦 Büyük d_model nedeniyle batch_size düşürüldü")
            
            # Çoklu sınıf için ayarlama
            if target_mode == 'three_class':
                batch_size = max(16, batch_size)
                logger.info(f"   🎯 Üç sınıflı mod için batch_size ayarlandı")
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'warmup_steps': warmup,
                'weight_decay': 0.01 if complexity > 5.0 else 0.005,
                'scheduler': 'CosineAnnealingWarmRestarts',
                'optimizer': 'AdamW',
                'gradient_clip': 0.5,
                'model_type': 'enhanced_transformer'
            }
            
        elif ENHANCED_TRANSFORMER_V2_AVAILABLE and isinstance(model, EnhancedTransformerV2):
            # EnhancedTransformer V2-specific öneriler
            
            # Model detaylarından ek bilgiler
            num_layers = getattr(model, 'num_layers', 8)
            nhead = getattr(model, 'nhead', 16)
            d_model = getattr(model, 'd_model', 512)
            target_mode = getattr(model, 'target_mode', 'binary')
            
            logger.info(f"   EnhancedTransformer V2 detayları: layers={num_layers}, heads={nhead}, d_model={d_model}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 5.0 and num_layers <= 8:
                base_lr = 8e-5
                batch_size = 48
                warmup = 1500
            elif complexity < 10.0 or num_layers <= 12:
                base_lr = 3e-5
                batch_size = 24
                warmup = 3000
            else:
                base_lr = 8e-6
                batch_size = 12
                warmup = 5000
            
            # Attention head sayısına göre LR ayarlaması
            if nhead > 24:
                base_lr *= 0.7
                logger.info(f"   📉 Çok attention head nedeniyle LR düşürüldü")
            
            # d_model'e göre batch size ayarlaması
            if d_model > 768:
                batch_size = max(6, batch_size // 2)
                logger.info(f"   📦 Büyük d_model nedeniyle batch_size düşürüldü")
            
            # Çoklu sınıf için ayarlama
            if target_mode == 'three_class':
                batch_size = max(12, batch_size)
                logger.info(f"   🎯 Üç sınıflı mod için batch_size ayarlandı")
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'warmup_steps': warmup,
                'weight_decay': 0.015 if complexity > 8.0 else 0.008,
                'scheduler': 'CosineAnnealingWarmRestarts',
                'optimizer': 'AdamW',
                'gradient_clip': 0.4,
                'model_type': 'enhanced_transformer_v2'
            }
            
        elif TRANSFORMER_AVAILABLE and isinstance(model, TransformerClassifier):
            # Transformer-specific öneriler
            
            # Model detaylarından ek bilgiler
            num_layers = getattr(model, 'num_layers', 4)
            nhead = getattr(model, 'nhead', 8)
            d_model = getattr(model, 'd_model', 128)
            
            logger.info(f"   Transformer detayları: layers={num_layers}, heads={nhead}, d_model={d_model}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 2.0 and num_layers <= 4:
                base_lr = 1e-3
                batch_size = 64
                warmup = 1000
            elif complexity < 5.0 or num_layers <= 6:
                base_lr = 5e-4
                batch_size = 32
                warmup = 2000
            else:
                base_lr = 1e-4
                batch_size = 16
                warmup = 4000
            
            # Attention head sayısına göre LR ayarlaması
            if nhead > 16:
                base_lr *= 0.8  # Çok head varsa LR düşür
                logger.info(f"   📉 Çok attention head nedeniyle LR düşürüldü")
            
            # d_model'e göre batch size ayarlaması
            if d_model > 256:
                batch_size = max(8, batch_size // 2)
                logger.info(f"   📦 Büyük d_model nedeniyle batch_size düşürüldü")
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'warmup_steps': warmup,
                'weight_decay': 0.01 if complexity > 3.0 else 0.005,
                'scheduler': 'OneCycleLR',
                'optimizer': 'AdamW',
                'gradient_clip': 1.0,
                'model_type': 'transformer'
            }
            
        else:
            # LSTM-specific öneriler
            
            # Model detaylarından ek bilgiler
            num_layers = getattr(model, 'num_layers', 2)
            hidden_size = getattr(model, 'hidden_size', 64)
            
            logger.info(f"   LSTM detayları: layers={num_layers}, hidden_size={hidden_size}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 1.0 and num_layers <= 2:
                base_lr = 1e-3
                batch_size = 128
            elif complexity < 3.0 or num_layers <= 3:
                base_lr = 8e-4
                batch_size = 64
            else:
                base_lr = 5e-4
                batch_size = 32
            
            # Hidden size'a göre ayarlama
            if hidden_size > 128:
                base_lr *= 0.8  # Büyük hidden size için LR düşür
                logger.info(f"   📉 Büyük hidden_size nedeniyle LR düşürüldü")
            
            # Katman sayısına göre weight decay ayarlaması
            weight_decay = 0.0001 if num_layers <= 2 else 0.001
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'scheduler': 'ReduceLROnPlateau',
                'optimizer': 'AdamW',
                'gradient_clip': 0.5,
                'patience': 5,
                'model_type': 'lstm'
            }
        
        logger.info(f"   ✅ Eğitim parametreleri önerildi: LR={suggestions['learning_rate']:.2e}, BS={suggestions['batch_size']}")
        
        return suggestions
        
    except Exception as e:
        logger.error(f"   ❌ Parametre önerisi oluşturulamadı: {e}")
        raise RuntimeError(f"Eğitim parametresi önerisi hatası: {e}") from e


def get_supported_models() -> List[str]:
    """
    Desteklenen model tiplerinin listesini döndür.
    
    Returns:
        Desteklenen model tipleri listesi
    """
    return SUPPORTED_MODELS.copy()


def validate_model_compatibility(model_type: str, config: ModelConfig) -> Tuple[bool, List[str]]:
    """
    Model tipi ve konfigürasyon uyumluluğunu kontrol et.
    
    Args:
        model_type: Model tipi
        config: Konfigürasyon sözlüğü
        
    Returns:
        (uyumlu_mu, uyarı_listesi) tuple'ı
    """
    warnings = []
    is_compatible = True
    
    try:
        if model_type.lower() in TRANSFORMER_ALIASES:
            validate_transformer_config(config)
        elif model_type.lower() in ENHANCED_TRANSFORMER_V2_ALIASES:
            validate_enhanced_transformer_config(config)
        elif model_type.lower() in LSTM_ALIASES:
            validate_lstm_config(config)
        else:
            warnings.append(f"Bilinmeyen model tipi: {model_type}")
            is_compatible = False
            
    except ValueError as e:
        warnings.append(str(e))
        is_compatible = False
    
    return is_compatible, warnings


__all__ = [
    'create_model',
    'get_model_info',
    'validate_transformer_config',
    'validate_enhanced_transformer_config',
    'validate_lstm_config',
    'get_model_complexity_score',
    'suggest_training_params',
    'get_supported_models',
    'validate_model_compatibility',
    'SUPPORTED_MODELS',
    'ModelConfig',
    'TransformerConfig', 
    'LSTMConfig',
    'ModelInstance'
]
