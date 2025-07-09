# src/ensemble/__init__.py - Basitleştirilmiş ve güvenli import

"""
Ensemble module for multi-model predictions
Simplified version to avoid import errors
"""

# Önce mevcut dosyaları kontrol edelim
import os
from pathlib import Path

# Mevcut dosyalar
current_dir = Path(__file__).parent
available_modules = []

# ensemble_trainer.py var mı?
if (current_dir / 'ensemble_trainer.py').exists():
    try:
        from .ensemble_trainer import EnsembleTrainer, VotingStrategies
        available_modules.extend(['EnsembleTrainer', 'VotingStrategies'])
        print("✅ EnsembleTrainer imported successfully")
    except ImportError as e:
        print(f"⚠️ EnsembleTrainer import failed: {e}")

# base_ensemble.py var mı?
if (current_dir / 'base_ensemble.py').exists():
    try:
        from .base_ensemble import BaseEnsemble
        available_modules.append('BaseEnsemble')
        print("✅ BaseEnsemble imported successfully")
    except ImportError as e:
        print(f"⚠️ BaseEnsemble import failed: {e}")

# ensemble_classifier.py var mı?
if (current_dir / 'ensemble_classifier.py').exists():
    try:
        from .ensemble_classifier import EnsembleClassifier
        available_modules.append('EnsembleClassifier')
        print("✅ EnsembleClassifier imported successfully")
    except ImportError as e:
        print(f"⚠️ EnsembleClassifier import failed: {e}")

# dynamic_selector.py var mı?
if (current_dir / 'dynamic_selector.py').exists():
    try:
        from .dynamic_selector import DynamicModelSelector
        available_modules.append('DynamicModelSelector')
        print("✅ DynamicModelSelector imported successfully")
    except ImportError as e:
        print(f"⚠️ DynamicModelSelector import failed: {e}")

# meta_learner.py var mı?
if (current_dir / 'meta_learner.py').exists():
    try:
        from .meta_learner import MetaLearner, AdvancedEnsemble
        available_modules.extend(['MetaLearner', 'AdvancedEnsemble'])
        print("✅ MetaLearner imported successfully")
    except ImportError as e:
        print(f"⚠️ MetaLearner import failed: {e}")

# Sadece başarılı import'ları __all__'a ekle
__all__ = available_modules

# Eğer hiçbir modül import edilememişse, minimal fallback
if not available_modules:
    print("⚠️ No ensemble modules could be imported, creating fallback")
    
    class FallbackEnsembleTrainer:
        """Fallback ensemble trainer for when imports fail"""
        def __init__(self, *args, **kwargs):
            print("📋 Using fallback EnsembleTrainer")
        
        def train(self, *args, **kwargs):
            print("📋 Fallback training - placeholder")
            return None
    
    EnsembleTrainer = FallbackEnsembleTrainer
    __all__ = ['EnsembleTrainer']

print(f"📋 Available ensemble modules: {available_modules}")