# src/config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class OptimizationConfig:
    """OneCycleLR ve Focal Loss konfigürasyonları"""
    
    # OneCycleLR parametreleri
    use_onecycle: bool = True
    max_lr: float = 0.01
    pct_start: float = 0.25
    div_factor: float = 10.0
    final_div_factor: float = 1000.0
    three_phase: bool = True
    
    # Focal Loss parametreleri
    loss_type: str = 'focal'  # 'focal', 'adaptive_focal', 'forex_focal', 'cross_entropy'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Memory optimization
    accumulation_steps: int = 1
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # Learning rate ve optimizer
    base_lr: float = 0.001
    weight_decay: float = 1e-4
    optimizer_type: str = 'adam'  # 'adam', 'adamw', 'sgd'
    
    # Scheduler alternatifleri
    scheduler_type: str = 'onecycle'  # 'onecycle', 'cosine', 'step', 'plateau'
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

@dataclass
class ModelConfig:
    """Model konfigürasyonu"""
    model_type: str = 'lstm'  # 'lstm', 'transformer', 'hybrid'
    hidden_size: int = 512
    num_layers: int = 4
    dropout: float = 0.15
    bidirectional: bool = True
    
    # LSTM specific
    lstm_layers: int = 2
    
    # Transformer specific (gelecek için)
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6

@dataclass
class DataConfig:
    """Data konfigürasyonu"""
    sequence_length: int = 60
    prediction_horizon: int = 1
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_sentiment_data: bool = False
    normalize_features: bool = True

@dataclass
class TrainingConfig:
    """Ana training konfigürasyonu"""
    
    # Temel parametreler
    epochs: int = 50
    batch_size: int = 32
    val_batch_size: int = 64
    
    # Alt konfigürasyonlar
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    
    # Monitoring ve logging
    log_interval: int = 100
    save_interval: int = 5
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    # Device ve reproducibility
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    seed: int = 42
    deterministic: bool = True
    
    # Evaluation
    eval_metrics: List[str] = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Mevcut model config ile uyumlu hale getir
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.optimization, dict):
            self.optimization = OptimizationConfig(**self.optimization)
    
    def to_dict(self) -> Dict[str, Any]:
        """Config'i dictionary'ye çevir"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'val_batch_size': self.val_batch_size,
            'model': {
                'model_type': self.model.model_type,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'bidirectional': self.model.bidirectional,
                'lstm_layers': self.model.lstm_layers
            },
            'data': {
                'sequence_length': self.data.sequence_length,
                'prediction_horizon': self.data.prediction_horizon,
                'train_split': self.data.train_split,
                'val_split': self.data.val_split,
                'test_split': self.data.test_split
            },
            'optimization': {
                'use_onecycle': self.optimization.use_onecycle,
                'max_lr': self.optimization.max_lr,
                'pct_start': self.optimization.pct_start,
                'div_factor': self.optimization.div_factor,
                'final_div_factor': self.optimization.final_div_factor,
                'loss_type': self.optimization.loss_type,
                'focal_alpha': self.optimization.focal_alpha,
                'focal_gamma': self.optimization.focal_gamma,
                'accumulation_steps': self.optimization.accumulation_steps,
                'use_mixed_precision': self.optimization.use_mixed_precision,
                'base_lr': self.optimization.base_lr,
                'weight_decay': self.optimization.weight_decay,
            },
            'log_interval': self.log_interval,
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device,
            'seed': self.seed
        }

# Preset konfigürasyonlar
def get_forex_configs():
    """Forex trading için önceden tanımlanmış konfigürasyonlar"""
    
    configs = {
        'conservative': TrainingConfig(
            epochs=100,
            batch_size=32,
            model=ModelConfig(
                hidden_size=256,
                num_layers=2,
                dropout=0.2
            ),
            optimization=OptimizationConfig(
                max_lr=0.005,
                pct_start=0.3,
                div_factor=25,
                focal_alpha=0.25,
                focal_gamma=2.0,
                loss_type='focal',
                accumulation_steps=1
            )
        ),
        
        'balanced': TrainingConfig(
            epochs=75,
            batch_size=24,
            model=ModelConfig(
                hidden_size=512,
                num_layers=4,
                dropout=0.15
            ),
            optimization=OptimizationConfig(
                max_lr=0.01,
                pct_start=0.25,
                div_factor=10,
                focal_alpha=0.25,
                focal_gamma=2.0,
                loss_type='adaptive_focal',
                accumulation_steps=2,
                use_mixed_precision=True
            )
        ),
        
        'aggressive': TrainingConfig(
            epochs=50,
            batch_size=16,
            model=ModelConfig(
                hidden_size=768,
                num_layers=6,
                dropout=0.1
            ),
            optimization=OptimizationConfig(
                max_lr=0.02,
                pct_start=0.2,
                div_factor=5,
                focal_alpha=0.3,
                focal_gamma=2.5,
                loss_type='forex_focal',
                accumulation_steps=4,
                use_mixed_precision=True
            )
        ),
        
        'memory_optimized': TrainingConfig(
            epochs=60,
            batch_size=8,
            model=ModelConfig(
                hidden_size=256,
                num_layers=2,
                dropout=0.15
            ),
            optimization=OptimizationConfig(
                max_lr=0.005,
                pct_start=0.25,
                div_factor=15,
                focal_alpha=0.25,
                focal_gamma=2.0,
                loss_type='focal',
                accumulation_steps=8,
                use_mixed_precision=True
            )
        )
    }
    
    return configs

def get_quick_test_config():
    """Hızlı test için minimal konfigürasyon"""
    return TrainingConfig(
        epochs=5,
        batch_size=16,
        model=ModelConfig(
            hidden_size=128,
            num_layers=1,
            dropout=0.1
        ),
        optimization=OptimizationConfig(
            max_lr=0.01,
            pct_start=0.3,
            div_factor=10,
            loss_type='focal',
            accumulation_steps=1,
            use_mixed_precision=False
        ),
        log_interval=10
    )

# Backward compatibility için default config
def get_default_config():
    """Mevcut sisteminizle uyumlu default config"""
    return TrainingConfig(
        epochs=50,
        batch_size=32,
        model=ModelConfig(
            model_type='lstm',
            hidden_size=512,
            num_layers=4,
            dropout=0.15,
            bidirectional=True,
            lstm_layers=2
        ),
        data=DataConfig(
            sequence_length=60,
            prediction_horizon=1
        ),
        optimization=OptimizationConfig(
            use_onecycle=True,
            max_lr=0.01,
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        )
    )