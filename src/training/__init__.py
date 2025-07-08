# src/training/__init__.py

# Mevcut trainer import'ları
try:
    from .trainer import Trainer
except ImportError:
    pass

# Yeni HybridForexTrainer import'u
try:
    from .trainer import HybridForexTrainer
    print("✅ HybridForexTrainer successfully imported")
except ImportError as e:
    print(f"❌ HybridForexTrainer import failed: {e}")

# Diğer trainer'lar (eğer varsa)
try:
    from .trainer import EnhancedMultiPairTrainer
except ImportError:
    pass

# Export edilen sınıflar
__all__ = ['Trainer', 'HybridForexTrainer']