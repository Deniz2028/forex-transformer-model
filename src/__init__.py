# src/__init__.py
"""Enhanced Modular LSTM/Transformer System for Forex Prediction with Temporal Training."""

__version__ = "2.0.0"  # ✅ Version bump for temporal features
__author__ = "LSTM Team"

import logging

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Core module availability checks
def check_core_availability():
    """Check if core modules are available."""
    try:
        from .data import fetcher, preprocess
        from .models import lstm, factory
        from .training import trainer
        return True
    except ImportError as e:
        logger.warning(f"Core modules not fully available: {e}")
        return False

def check_transformer_availability():
    """Check if transformer modules are available."""
    try:
        from .models import enhanced_transformer
        return True
    except ImportError:
        return False

def check_temporal_availability():
    """Check if temporal training modules are available."""
    try:
        from .temporal import TemporalTrainer
        from .utils import temporal_plotting, training_timer
        return True
    except ImportError:
        return False

def check_inference_availability():
    """Check if inference modules are available."""
    try:
        from .inference import predictor
        return True
    except ImportError:
        return False

# Check module availability
CORE_AVAILABLE = check_core_availability()
TRANSFORMER_AVAILABLE = check_transformer_availability()
TEMPORAL_AVAILABLE = check_temporal_availability()
INFERENCE_AVAILABLE = check_inference_availability()

# Print system status
def print_system_status():
    """Print system module availability status."""
    print("🚀 ENHANCED LSTM/TRANSFORMER SYSTEM v2.0.0")
    print("="*50)
    print(f"📊 Core Modules: {'✅' if CORE_AVAILABLE else '❌'}")
    print(f"🤖 Enhanced Transformer: {'✅' if TRANSFORMER_AVAILABLE else '❌'}")
    print(f"🕒 Temporal Training: {'✅' if TEMPORAL_AVAILABLE else '❌'}")
    print(f"🔮 Inference: {'✅' if INFERENCE_AVAILABLE else '❌'}")
    print("="*50)
    
    if TEMPORAL_AVAILABLE:
        print("🎉 Temporal training is ready!")
        print("💡 Use: python -m src.cli temporal --pairs EUR_USD --cache")
    else:
        print("⚠️ Temporal training not available")
        print("💡 Install temporal components to enable")
    
    print()

# Feature flags for conditional imports
FEATURES = {
    'core': CORE_AVAILABLE,
    'transformer': TRANSFORMER_AVAILABLE,
    'temporal': TEMPORAL_AVAILABLE,
    'inference': INFERENCE_AVAILABLE,
    'plotting': TEMPORAL_AVAILABLE,  # Plotting depends on temporal utils
    'time_tracking': TEMPORAL_AVAILABLE  # Time tracking depends on temporal utils
}

# Export availability info
__all__ = [
    '__version__',
    '__author__',
    'FEATURES',
    'print_system_status',
    'check_temporal_availability',
    'TEMPORAL_AVAILABLE'
]

# Auto-print status when imported (optional - can be commented out)
if __name__ != "__main__":
    # Only print during direct import, not during testing
    import sys
    if 'pytest' not in sys.modules:
        print_system_status()

# Conditional imports for convenience
if TEMPORAL_AVAILABLE:
    try:
        from .temporal import TemporalTrainer, run_temporal_training_pipeline
        __all__.extend(['TemporalTrainer', 'run_temporal_training_pipeline'])
    except ImportError:
        pass

if INFERENCE_AVAILABLE:
    try:
        from .inference import EnhancedTransformerPredictor, run_inference_cli
        __all__.extend(['EnhancedTransformerPredictor', 'run_inference_cli'])
    except ImportError:
        pass

# Quick access functions
def get_available_features():
    """Get list of available features."""
    return [feature for feature, available in FEATURES.items() if available]

def is_feature_available(feature_name: str) -> bool:
    """Check if a specific feature is available."""
    return FEATURES.get(feature_name, False)

def get_temporal_example_command():
    """Get example temporal training command."""
    return "python -m src.cli temporal --pairs EUR_USD GBP_USD --cache --temporal_epochs 50"

def get_inference_example_command():
    """Get example inference command."""
    return "python -m src.cli inference --pairs EUR_USD --model_dir models --cache"

# Add convenience functions to exports
__all__.extend([
    'get_available_features',
    'is_feature_available', 
    'get_temporal_example_command',
    'get_inference_example_command'
])

# Development helper
def create_temporal_directories():
    """Create necessary directories for temporal training."""
    from pathlib import Path
    
    directories = [
        'temporal_results',
        'temporal_reports', 
        'temporal_signals',
        'temporal_plots',
        'temporal_models',
        'logs/temporal'
    ]
    
    created = []
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    
    if created:
        logger.info(f"📁 Created directories: {', '.join(created)}")
    else:
        logger.info("📁 All temporal directories already exist")
    
    return created

# Add to exports
__all__.append('create_temporal_directories')

