# src/models/factory.py
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

# Enhanced Transformer için import (mevcut dosyadan)
try:
    from .enhanced_transformer import EnhancedTransformer, create_enhanced_transformer
    TRANSFORMER_AVAILABLE = True
    logger.info("✅ EnhancedTransformer import başarılı")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Transformer import hatası: {e}")
except Exception as e:
    logger.error(f"❌ Beklenmeyen hata (enhanced_transformer): {e}")

# Transformer modülünü dinamik olarak yükleme
try:
    transformer_module = importlib.import_module('.transformer_model', package='src.models')
    
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

# Hibrit LSTM-Transformer modeli için import
try:
    hybrid_module = importlib.import_module('.hybrid_model', package='src.models')
    
    if hasattr(hybrid_module, 'HybridLSTMTransformer'):
        from .hybrid_model import HybridLSTMTransformer, create_hybrid_model, validate_hybrid_config
        HYBRID_MODEL_AVAILABLE = True
        logger.info("✅ HybridLSTMTransformer import başarılı")
        
except ImportError as e:
    logger.warning(f"⚠️ Hybrid model import hatası: {e}")
    HYBRID_MODEL_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Beklenmeyen hata (hybrid_model): {e}")
    HYBRID_MODEL_AVAILABLE = False

# Model tipi sabitleri - GÜVENLİ HALE GETİR
SUPPORTED_MODELS = ['lstm', 'pairspecificlstm']

# Mevcut modellere göre desteklenen tipleri ekle
if TRANSFORMER_AVAILABLE:
    SUPPORTED_MODELS.extend(['transformer', 'enhanced_transformer'])
    logger.info("✅ Transformer modelleri desteklenen listeye eklendi")

if HYBRID_MODEL_AVAILABLE:
    SUPPORTED_MODELS.extend(['hybrid_lstm_transformer', 'hybrid'])
    logger.info("✅ Hibrit LSTM-Transformer desteklenen listeye eklendi")

logger.info(f"📋 Desteklenen modeller: {SUPPORTED_MODELS}")

# Model alias'ları
LSTM_ALIASES = ['lstm', 'pairspecificlstm']
TRANSFORMER_ALIASES = ['transformer', 'enhanced_transformer'] if TRANSFORMER_AVAILABLE else []
HYBRID_ALIASES = ['hybrid_lstm_transformer', 'hybrid'] if HYBRID_MODEL_AVAILABLE else []

# ModelInstance tipini güvenli hale getir
available_types = [PairSpecificLSTM]

if TRANSFORMER_AVAILABLE:
    if EnhancedTransformer:
        available_types.append(EnhancedTransformer)
    if TransformerClassifier:
        available_types.append(TransformerClassifier)

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
        model_type: Oluşturulacak model tipi ('lstm', 'enhanced_transformer', 'hybrid_lstm_transformer', ...)
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
            
            if model_type_normalized == 'enhanced_transformer' and create_enhanced_transformer:
                model = create_enhanced_transformer(validated_config, n_features, device)
                logger.info(f"   ✅ EnhancedTransformer modeli başarıyla oluşturuldu")
            elif model_type_normalized == 'transformer' and create_transformer_model:
                model = create_transformer_model(validated_config, n_features, device)
                logger.info(f"   ✅ Transformer modeli başarıyla oluşturuldu")
            else:
                raise RuntimeError(f"İstenen model tipi desteklenmiyor: {model_type}")
                
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
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    d_model = model_config.get('d_model', 256)
    nhead = model_config.get('nhead', 8)
    num_layers = model_config.get('num_layers', 4)
    dropout = model_config.get('dropout_rate', 0.1)
    
    logger.info(f"🔍 Transformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # d_model validasyonu
    if not isinstance(d_model, int) or d_model < 64 or d_model > 1024:
        raise ValueError(f"d_model geçersiz: {d_model}. 64 ile 1024 arasında integer olmalı")
    
    # nhead validasyonu
    if not isinstance(nhead, int) or nhead < 1 or nhead > 16:
        raise ValueError(f"nhead geçersiz: {nhead}. 1 ile 16 arasında integer olmalı")
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            model_config['nhead'] = nhead
            logger.warning(f"   🔧 nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(f"d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı")
    
    # num_layers validasyonu
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 8:
        raise ValueError(f"num_layers geçersiz: {num_layers}. 1 ile 8 arasında integer olmalı")
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.5:
        raise ValueError(f"dropout geçersiz: {dropout}. 0.0 ile 0.5 arasında float olmalı")
    
    logger.info(f"   ✅ Transformer konfigürasyonu doğrulandı")
    return config


def validate_enhanced_transformer_config(config: TransformerConfig) -> TransformerConfig:
    """
    Enhanced Transformer konfigürasyon parametrelerini doğrula ve ayarla.
    
    Args:
        config: Konfigürasyon sözlüğü
        
    Returns:
        Doğrulanmış konfigürasyon
        
    Raises:
        ValueError: Konfigürasyon geçersizse
    """
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    d_model = model_config.get('d_model', 512)
    nhead = model_config.get('nhead', 8)
    num_layers = model_config.get('num_layers', 6)
    dropout = model_config.get('dropout_rate', 0.1)
    
    logger.info(f"🔍 Enhanced Transformer konfigürasyonu doğrulanıyor...")
    logger.info(f"   d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
    
    # Enhanced transformer için daha yüksek minimum değerler
    if not isinstance(d_model, int) or d_model < 128 or d_model > 1024:
        raise ValueError(f"Enhanced d_model geçersiz: {d_model}. 128 ile 1024 arasında integer olmalı")
    
    if not isinstance(nhead, int) or nhead < 4 or nhead > 16:
        raise ValueError(f"Enhanced nhead geçersiz: {nhead}. 4 ile 16 arasında integer olmalı")
    
    # d_model ve nhead uyumluluğu
    if d_model % nhead != 0:
        valid_heads = [h for h in [4, 8, 16] if d_model % h == 0 and h <= d_model]
        if valid_heads:
            old_nhead = nhead
            nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[-1]
            model_config['nhead'] = nhead
            logger.warning(f"   🔧 Enhanced nhead otomatik düzeltildi: {old_nhead} → {nhead}")
        else:
            raise ValueError(f"Enhanced d_model ({d_model}) nhead ({nhead}) ile bölünebilir olmalı")
    
    if not isinstance(num_layers, int) or num_layers < 2 or num_layers > 12:
        raise ValueError(f"Enhanced num_layers geçersiz: {num_layers}. 2 ile 12 arasında integer olmalı")
    
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.3:
        raise ValueError(f"Enhanced dropout geçersiz: {dropout}. 0.0 ile 0.3 arasında float olmalı")
    
    logger.info(f"   ✅ Enhanced Transformer konfigürasyonu doğrulandı")
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
    config = copy.deepcopy(config)
    model_config = config.get('model', {})
    
    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout_rate', 0.45)
    
    logger.info(f"🔍 LSTM konfigürasyonu doğrulanıyor...")
    logger.info(f"   hidden_size: {hidden_size}, layers: {num_layers}, dropout: {dropout}")
    
    # hidden_size validasyonu
    if not isinstance(hidden_size, int) or hidden_size < 16 or hidden_size > 512:
        raise ValueError(f"hidden_size geçersiz: {hidden_size}. 16 ile 512 arasında integer olmalı")
    
    # num_layers validasyonu  
    if not isinstance(num_layers, int) or num_layers < 1 or num_layers > 4:
        raise ValueError(f"num_layers geçersiz: {num_layers}. 1 ile 4 arasında integer olmalı")
    
    # dropout validasyonu
    if not isinstance(dropout, (int, float)) or dropout < 0.0 or dropout > 0.8:
        raise ValueError(f"dropout geçersiz: {dropout}. 0.0 ile 0.8 arasında float olmalı")
    
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
        Model bilgi sözlüğü
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model tipine göre detaylar
        model_type = type(model).__name__
        
        info = {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'device': str(next(model.parameters()).device),
        }
        
        # Model-specific bilgiler
        if hasattr(model, 'hidden_size'):
            info['hidden_size'] = model.hidden_size
        if hasattr(model, 'num_layers'):
            info['num_layers'] = model.num_layers
        if hasattr(model, 'd_model'):
            info['d_model'] = model.d_model
        if hasattr(model, 'nhead'):
            info['nhead'] = model.nhead
        if hasattr(model, 'lstm_hidden'):
            info['lstm_hidden'] = model.lstm_hidden
        
        return info
        
    except Exception as e:
        logger.error(f"   ❌ Model bilgisi alınamadı: {e}")
        return {
            'model_type': type(model).__name__,
            'total_parameters': 0,
            'trainable_parameters': 0,
            'error': str(e)
        }


def get_model_complexity_score(model: ModelInstance) -> float:
    """
    Model karmaşıklık skorunu hesapla.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Karmaşıklık skoru (million parametreler cinsinden)
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        complexity = total_params / 1e6  # Million parametreler
        
        return complexity
        
    except Exception as e:
        logger.error(f"   ❌ Karmaşıklık skoru hesaplanamadı: {e}")
        return 1.0


def suggest_training_params(model: ModelInstance) -> Dict[str, Any]:
    """
    Model kompleksitesine göre eğitim parametreleri öner.
    
    Args:
        model: Model instance'ı
        
    Returns:
        Önerilen eğitim parametreleri
    """
    try:
        complexity = get_model_complexity_score(model)
        info = get_model_info(model)
        
        logger.info(f"🎯 Eğitim parametreleri öneriliyor...")
        logger.info(f"   Model karmaşıklığı: {complexity:.2f}M parameters")
        
        # Hibrit model için öneriler
        if HYBRID_MODEL_AVAILABLE and isinstance(model, HybridLSTMTransformer):
            # Hibrit model detayları
            lstm_hidden = getattr(model, 'lstm_hidden', 96)
            d_model = getattr(model, 'd_model', 512)
            num_layers = getattr(model, 'num_layers', 4)
            
            logger.info(f"   Hibrit model detayları: LSTM hidden={lstm_hidden}, d_model={d_model}, layers={num_layers}")
            
            # Karmaşıklığa ve mimari detaylarına göre öneri
            if complexity < 4.0 and num_layers <= 4:
                base_lr = 7e-4
                batch_size = 48
                warmup = 1200
            elif complexity < 10.0 or num_layers <= 6:
                base_lr = 3e-4
                batch_size = 32
                warmup = 2500
            else:
                base_lr = 1e-4
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
            # Enhanced Transformer için öneriler
            d_model = getattr(model, 'd_model', 256)
            nhead = getattr(model, 'nhead', 8)
            num_layers = getattr(model, 'num_layers', 6)
            
            logger.info(f"   Enhanced Transformer detayları: d_model={d_model}, heads={nhead}, layers={num_layers}")
            
            # Karmaşıklığa göre öneri
            if complexity < 2.0:
                base_lr = 5e-4
                batch_size = 64
                warmup = 800
            elif complexity < 8.0:
                base_lr = 2e-4
                batch_size = 32
                warmup = 1600
            else:
                base_lr = 8e-5
                batch_size = 16
                warmup = 3200
            
            # Attention head'e göre ayarlama
            if nhead > 12:
                base_lr *= 0.8
                logger.info(f"   📉 Çok attention head nedeniyle LR düşürüldü")
            
            suggestions = {
                'learning_rate': base_lr,
                'batch_size': batch_size,
                'warmup_steps': warmup,
                'weight_decay': 0.01,
                'scheduler': 'OneCycleLR',
                'optimizer': 'AdamW',
                'gradient_clip': 1.0,
                'model_type': 'enhanced_transformer'
            }
        
        else:
            # LSTM için öneriler
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
                base_lr *= 0.8
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
        elif model_type.lower() in LSTM_ALIASES:
            validate_lstm_config(config)
        elif model_type.lower() in HYBRID_ALIASES:
            validate_hybrid_config(config) if HYBRID_MODEL_AVAILABLE else warnings.append("Hibrit model mevcut değil")
        else:
            warnings.append(f"Bilinmeyen model tipi: {model_type}")
            is_compatible = False
            
    except ValueError as e:
        warnings.append(str(e))
        is_compatible = False
    
    return is_compatible, warnings


# Test fonksiyonu da ekleyelim
def test_hybrid_model_creation():
    """Test hibrit model oluşturma"""
    if not HYBRID_MODEL_AVAILABLE:
        logger.warning("⚠️ Hibrit model mevcut değil, test atlanıyor")
        return False
        
    try:
        from .. import config
        
        # Test konfigürasyonu
        test_config = config.get_model_config('hybrid_lstm_transformer', 'three_class')
        device = torch.device('cpu')
        n_features = 20
        
        logger.info(f"🧪 Hibrit model testi başlıyor...")
        logger.info(f"   Config: {test_config}")
        
        model = create_model('hybrid_lstm_transformer', test_config, n_features, device)
        
        # Test input
        batch_size, seq_len = 8, 50
        test_input = torch.randn(batch_size, seq_len, n_features)
        
        logger.info(f"   Test input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            
        expected_classes = 3 if test_config.get('target_mode') == 'three_class' else 2
        assert output.shape == (batch_size, expected_classes), f"Wrong output shape: {output.shape}"
        
        logger.info(f"✅ Hibrit model testi başarılı!")
        logger.info(f"   Input: {test_input.shape} → Output: {output.shape}")
        logger.info(f"   Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hibrit model testi başarısız: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


__all__ = [
    'create_model',
    'get_model_info',
    'validate_transformer_config',
    'validate_enhanced_transformer_config',
    'validate_lstm_config',
    'validate_hybrid_config' if HYBRID_MODEL_AVAILABLE else 'validate_lstm_config',
    'get_model_complexity_score',
    'suggest_training_params',
    'get_supported_models',
    'validate_model_compatibility',
    'test_hybrid_model_creation',
    'SUPPORTED_MODELS',
    'ModelConfig',
    'TransformerConfig', 
    'LSTMConfig',
    'HybridConfig',
    'ModelInstance'
]
