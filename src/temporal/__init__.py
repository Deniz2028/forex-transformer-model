# src/temporal/__init__.py
"""
Temporal training and backtesting module.

Bu modül, veri sızıntısı olmaksızın temporal eğitim ve sinyal üretimi sağlar.
Belirli tarih aralıklarında eğitim yapıp, sonraki periyotlarda sinyal üretimi yapar.
"""

from .temporal_trainer import TemporalTrainer, run_temporal_training_pipeline

__all__ = [
    'TemporalTrainer',
    'run_temporal_training_pipeline'
]

__version__ = "1.0.0"
__author__ = "LSTM Team - Temporal Extension"