"""
Geliştirilmiş model factory for creating different model architectures.

Bu modül, LSTM, EnhancedTransformer, HybridLSTMTransformer ve diğer modelleri oluşturmak için birleşik arayüz sağlar.
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
HYBRID_MODEL_AVAILABLE = False
EnhancedTransformer = None
TransformerClassifier = None
create_transformer_model = None
create_enhanced_transformer = None
EnhancedTransformerV2 = None
create_enhanced_transformer_v2 = None
HybridLSTMTransformer = None

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

# Hibrit LSTM-Transformer modeli için import
try:
    hybrid_module = importlib.import_module('.hybrid_model', package='src.models')
    
    if hasattr(hybrid_module, 'HybridLSTMTransformer'):
        from .hybrid_model import HybridLSTMTransformer
        HYBRID_MODEL_AVAILABLE = True
        logger.info("✅ HybridLSTMTransformer import başarılı")
        
except ImportError as e:
    logger.warning(f"⚠️ Hybrid model import hatası: {e}")
except Exception as e:
    logger.error(f"❌ Beklenmeyen hata (hybrid_model): {e}")

# Model tipi sabitleri - GÜVENLİ HALE GETİR
SUPPORTED_MODELS = ['lstm', 'pairspecificlstm']

# Mevcut modellere göre desteklenen tipleri ekle
if TRANSFORMER_AVAILABLE:
    SUPPORTED_MODELS.extend(['transformer', 'enhanced_transformer'])
    logger.info("✅ Transformer modelleri desteklenen listeye eklendi")

if ENHANCED_TRANSFORMER_V2_AVAILABLE:
    SUPPORTED_MODELS.append('enhanced_transformer_v2')
    logger.info("✅ Enhanced Transformer V2 desteklenen listeye eklendi")

if HYBRID_MODEL_AVAILABLE:
    SUPPORTED_MODELS.append('hybrid_lstm_transformer')
    logger.info("✅ Hibrit LSTM-Transformer desteklenen listeye eklendi")

logger.info(f"📋 Desteklenen modeller: {SUPPORTED_MODELS}")

# Model alias'ları
LSTM_ALIASES = ['lstm', 'pairspecificlstm']
TRANSFORMER_ALIASES = ['transformer', 'enhanced_transformer'] if TRANSFORMER_AVAILABLE else []
ENHANCED_TRANSFORMER_V2_ALIASES = ['enhanced_transformer_v2'] if ENHANCED_TRANSFORMER_V2_AVAILABLE else []
HYBRID_ALIASES = ['hybrid_lstm_transformer'] if HYBRID_MODEL_AVAILABLE else []

# ModelInstance tipini güvenli hale getir
available_types = [PairSpecificLSTM]

if TRANSFORMER_AVAILABLE:
    if EnhancedTransformer:
        available_types.append(EnhancedTransformer)
    if TransformerClassifier:
        available_types.append(TransformerClassifier)

if ENHANCED_TRANSFORMER_V2_AVAILABLE and EnhancedTransformerV2:
    available_types.append(EnhancedTransformerV2)

if HYBRID_MODEL_AVAILABLE and HybridLSTMTransformer:
    available_types.append(HybridLSTMTransformer)

if len(available_types) == 0:
    logger.error("⚠️ Hiçbir model tipi kullanılamıyor! Sadece LSTM kullanılacak.")
    available_types = [PairSpecificLSTM]

ModelInstance = Union[tuple(available_types)]

# Tip alias'ları (daha açıklayıcı tip notasyonları için)
ModelConfig = Dict[str, Any]
TransformerConfig = Dict[str, Any]
LSTMConfig = Dict[str, Any]
HybridConfig = Dict[str, Any]


def create_model(
    model_type: str,
    config: ModelConfig, 
    n_features: int, 
    device: torch.device
) -> ModelInstance:
    """
    Tip ve konfigürasyona göre model oluşturan factory fonksiyonu.
    
    Args:
        model_type: Oluşturulacak model tipi ('lstm', 'transformer', 'hybrid_lstm_transformer', ...)
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
        
        elif model_type_normalized in HYBRID_ALIASES:
            if not HYBRID_MODEL_AVAILABLE or not HybridLSTMTransformer:
                raise RuntimeError("Hibrit LSTM-Transformer modeli desteklenmiyor (import hatası)")
                
            # Hibrit model konfigürasyonu validasyonu
            validated_config = validate_hybrid_config(config)
            model = create_hybrid_model(validated_config, n_features, device)
            logger.info(f"   ✅ Hibrit LSTM-Transformer modeli başarıyla oluşturuldu")
        
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


def create_hybrid_model(config: HybridConfig, n_features: int, device: torch.device) -> HybridLSTMTransformer:
    """
    Hibrit LSTM-Transformer modeli oluşturur
    
    Args:
        config: Hibrit model konfigürasyonu
        n_features: Giriş özellik sayısı
        device: Modelin yerleştirileceği cihaz
        
    Returns:
        Oluşturulan hibrit model instance'ı
    """
    # Hibrit-specific config
    hybrid_config = config.get('hybrid', {})
    
    # Parametreleri çıkar veya default değerler kullan
    lstm_hidden = hybrid_config.get('lstm_hidden', 96)
    d_model = hybrid_config.get('d_model', 512)
    nhead = hybrid_config.get('nhead', 8)
    num_layers = hybrid_config.get('num_layers', 4)
    dropout = hybrid_config.get('dropout', 0.1)
    
    model = HybridLSTMTransformer(
        input_dim=n_features,
        lstm_hidden=lstm_hidden,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    logger.info(f"   ✅ Hibrit LSTM-Transformer modeli oluşturuldu")
    logger.info(f"      LSTM hidden: {lstm_hidden}, Transformer d_model: {d_model}")
    
    return model


def validate_hybrid_config(config: HybridConfig) -> HybridConfig:
    """
    Hibrit model konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    config = copy.deepcopy(config)
    hybrid_config = config.get('hybrid', {})
    
    lstm_hidden = hybrid_config.get('lstm_hidden', 96)
    d_model = hybrid_config.get('d_model', 512)
    nhead = hybrid_config.get('nhead', 8)
    num_layers = hybrid_config.get('num_layers', 4)
    dropout = hybrid_config.get('dropout', 0.1)
    
    logger.info(f"🔍 Hibrit model konfigürasyonu doğrulanıyor...")
    logger.info(f"   LSTM hidden: {lstm_hidden}, d_model: {d_model}, layers: {num_layers}")
    
    # LSTM hidden size validasyonu
    if not isinstance(lstm_hidden, int) or lstm_hidden < 16 or lstm_hidden > 512:
        raise ValueError(
            f"LSTM hidden_size geçersiz: {lstm_hidden}. "
            f"16 ile 512 arasında integer olmalı"
        )
    
    # Transformer d_model validasyonu
    if not isinstance(d_model, int) or d_model < 64 or d_model > 1024:
        raise ValueError(
            f"d_model geçersiz: {d_model}. "
            f"64 ile 1024 arasında integer olmalı"
        )
    
    # nhead validasyonu
    if not isinstance(nhead, int) or nhead < 1 or nhead > 16:
        raise ValueError(
            f"nhead geçersiz: {nhead}. "
            f"1 ile 16 arasında integer olmalı"
        )
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        logger.warning(f"   ⚠️ d_model ({d_model}) nhead ({nhead}) ile bölünemiyor")
        
        # Otomatik düzeltme denemesi
        valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            hybrid_config['nhead'] = nhead
            logger.warning(f"   🔧 nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(
                f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı. "
                f"d_model % nhead == 0 koşulu sağlanmalı. "
                f"Geçerli nhead değerleri: {[h for h in [1, 2, 4, 8, 16] if d_model % h == 0]}"
            )
    
    # num_layers validasyonu
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 8:
        raise ValueError(
            f"num_layers geçersiz: {num_layers}. "
            f"1 ile 8 arasında integer olmalı"
        )
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.5:
        raise ValueError(
            f"dropout geçersiz: {dropout}. "
            f"0.0 ile 0.5 arasında float olmalı"
        )
    
    # Performans uyarıları
    if lstm_hidden > 256:
        logger.warning(f"   ⚠️ Büyük LSTM hidden_size ({lstm_hidden}) overfitting riskini artırabilir")
    
    if d_model > 512:
        logger.warning(f"   ⚠️ Büyük d_model ({d_model}) önemli GPU belleği gerektirebilir")
    
    if num_layers > 6:
        logger.warning(f"   ⚠️ Çok katman ({num_layers}) eğitimi yavaşlatabilir")
    
    logger.info(f"   ✅ Hibrit model konfigürasyonu doğrulandı")
    
    return config


def validate_transformer_config(config: TransformerConfig) -> TransformerConfig:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

def validate_enhanced_transformer_config(config: TransformerConfig) -> TransformerConfig:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

def validate_lstm_config(config: LSTMConfig) -> LSTMConfig:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

def get_model_info(model: ModelInstance) -> Dict[str, Any]:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

def get_model_complexity_score(model: ModelInstance) -> float:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

def suggest_training_params(model: ModelInstance) -> Dict[str, Any]:
    try:
        complexity = get_model_complexity_score(model)
        info = get_model_info(model)
        
        logger.info(f"🎯 Eğitim parametreleri öneriliyor...")
        logger.info(f"   Model karmaşıklığı: {complexity}")
        
        # YENİ: Hibrit model için öneriler
        if HYBRID_MODEL_AVAILABLE and isinstance(model, HybridLSTMTransformer):
            # Hibrit model detayları
            lstm_hidden = getattr(model, 'lstm_hidden_size', 96)
            d_model = getattr(model, 'd_model', 512)
            num_layers = getattr(model, 'num_layers', 4)
            
            logger.info(f"   Hibrit model detayları: LSTM hidden={lstm_hidden}, d_model={d_model}, layers={num_layers}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 4.0 and num_layers <= 4:
                base_lr = 7e-5
                batch_size = 48
                warmup = 1200
            elif complexity < 10.0 or num_layers <= 6:
                base_lr = 3e-5
                batch_size = 32
                warmup = 2500
            else:
                base_lr = 1e-5
                batch_size = 16
                warmup = 4000
            
            # LSTM hidden size'a göre ayarlama
            if lstm_hidden > 128:
                base_lr *= 0.85
                logger.info(f"   📉 Büyük LSTM hidden_size nedeniyle LR düşürüldü")
            
            # d_model'e göre batch size ayarlaması
            if d_model > 512:
                batch_size = max(8, batch_size // 2)
                logger.info(f"   📦 Büyük d_model nedeniyle batch_size düşürüldü")
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'warmup_steps': warmup,
                'weight_decay': 0.01,
                'scheduler': 'CosineAnnealingWarmRestarts',
                'optimizer': 'AdamW',
                'gradient_clip': 0.3,
                'model_type': 'hybrid_lstm_transformer'
            }
        
        elif TRANSFORMER_AVAILABLE and isinstance(model, EnhancedTransformer):
            # ... (Mevcut kod aynı kalır) ...
        elif ENHANCED_TRANSFORMER_V2_AVAILABLE and isinstance(model, EnhancedTransformerV2):
            # ... (Mevcut kod aynı kalır) ...
        elif TRANSFORMER_AVAILABLE and isinstance(model, TransformerClassifier):
            # ... (Mevcut kod aynı kalır) ...
        else:
            # ... (Mevcut kod aynı kalır) ...
        
        logger.info(f"   ✅ Eğitim parametreleri önerildi: LR={suggestions['learning_rate']:.2e}, BS={suggestions['batch_size']}")
        return suggestions
        
    except Exception as e:
        logger.error(f"   ❌ Parametre önerisi oluşturulamadı: {e}")
        raise RuntimeError(f"Eğitim parametresi önerisi hatası: {e}") from e


def get_supported_models() -> List[str]:
    # ... (Mevcut kod aynı kalır, değişiklik yok) ...

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
        elif model_type.lower() in HYBRID_ALIASES:
            validate_hybrid_config(config)  # YENİ: Hibrit model validasyonu
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
    'validate_hybrid_config',  # YENİ
    'get_model_complexity_score',
    'suggest_training_params',
    'get_supported_models',
    'validate_model_compatibility',
    'SUPPORTED_MODELS',
    'ModelConfig',
    'TransformerConfig', 
    'LSTMConfig',
    'HybridConfig',  # YENİ
    'ModelInstance'
]
