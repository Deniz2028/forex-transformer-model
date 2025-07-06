# src/cli.py - ENHANCED VERSION with ENSEMBLE, TUNE, INFERENCE and TEMPORAL support
"""Enhanced command-line interface with ENSEMBLE, TUNE, INFERENCE and TEMPORAL support."""

import argparse
import torch
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Import modular components
from . import config
from .data.fetcher import create_fetcher
from .data.preprocess import create_preprocessor
from .training.trainer import create_trainer
from .training.optuna_utils import create_optimizer
from .utils.viz import plot_training_results, print_summary_report

# Enhanced Transformer factory import
from .models.factory import create_model

# INFERENCE import
from .inference.predictor import run_inference_cli

# Ensemble imports
from .ensemble.ensemble_trainer import EnsembleTrainer
from .ensemble.ensemble_classifier import EnsembleClassifier
from .ensemble.ensemble_monitor import EnsembleMonitor

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cli.log')
    ]
)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with Enhanced Transformer, TUNE, INFERENCE, TEMPORAL and ENSEMBLE support."""
    parser = argparse.ArgumentParser(
        description='Modular Multi-Pair LSTM/Enhanced Transformer System with Ensemble, Advanced Tuning, Inference and Temporal Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ===== TRAIN COMMAND =====
    train_parser = subparsers.add_parser('train', help='Train LSTM/Enhanced Transformer models')
    add_common_args(train_parser)
    add_training_args(train_parser)
    add_transformer_args(train_parser)
    add_enhanced_transformer_args(train_parser)  # PDF optimizations
    add_ensemble_args(train_parser)  # Ensemble support
    
    # ===== ENSEMBLE COMMAND =====
    ensemble_parser = subparsers.add_parser('ensemble', help='Train ensemble of models')
    add_common_args(ensemble_parser)
    add_training_args(ensemble_parser)
    add_ensemble_args(ensemble_parser)
    
    # ===== TUNE COMMAND =====
    tune_parser = subparsers.add_parser('tune', help='Advanced hyperparameter tuning')
    add_common_args(tune_parser)
    add_training_args(tune_parser)
    add_transformer_args(tune_parser)
    add_tune_args(tune_parser)  # Tune-specific arguments
    add_enhanced_transformer_args(tune_parser)  # PDF optimizations
    
    # ===== BACKTEST COMMAND =====
    backtest_parser = subparsers.add_parser('backtest', help='Run walk-forward backtesting')
    add_common_args(backtest_parser)
    add_training_args(backtest_parser)
    add_backtest_args_fixed(backtest_parser)
    
    # ===== EVALUATE COMMAND =====
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate existing models')
    eval_parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained models')
    eval_parser.add_argument('--test_data', type=str, help='Path to test data')
    add_common_args(eval_parser)
    
    # ===== INFERENCE COMMAND =====
    inference_parser = subparsers.add_parser('inference', help='Generate predictions CSV')
    add_inference_args(inference_parser)
    add_ensemble_args(inference_parser)  # Ensemble support for inference
    
    # ===== TEMPORAL COMMAND - NEW! =====
    temporal_parser = subparsers.add_parser('temporal', help='Temporal training and signal generation')
    add_common_args(temporal_parser)
    add_training_args(temporal_parser)
    add_transformer_args(temporal_parser)
    add_temporal_args(temporal_parser)
    add_enhanced_transformer_args(temporal_parser)  # PDF optimizations
    
    # If no command provided, default to train
    args = parser.parse_args()
    if args.command is None:
        args.command = 'train'
    
    # Validate Enhanced Transformer arguments
    args = validate_transformer_args(args)
    
    return args

def add_common_args(parser):
    """Add common arguments shared across commands."""
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    # Model type selection - ENHANCED TRANSFORMER ADDED
    parser.add_argument(
        '--model', 
        type=str, 
        default='lstm',
        choices=['lstm', 'transformer', 'enhanced_transformer'],
        help='Model architecture to use'
    )
    
    # Data parameters
    parser.add_argument(
        '--pairs', 
        nargs='+',
        default=['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'XAU_USD'],
        help='Currency pairs to process'
    )
    parser.add_argument(
        '--granularity', 
        type=str, 
        default='M15',
        choices=['M1', 'M5', 'M15', 'H1', 'H4', 'D'],
        help='Candle granularity'
    )
    parser.add_argument(
        '--lookback_candles', 
        type=int, 
        default=50000,
        help='Maximum number of candles to fetch'
    )
    parser.add_argument(
        '--cache', 
        action='store_true',
        default=False,
        help='Enable data caching to disk'
    )
    
    # Model parameters
    parser.add_argument(
        '--target_mode', 
        type=str, 
        default='binary',
        choices=['binary', 'three_class'],
        help='Target prediction mode'
    )
    parser.add_argument(
        '--use_focal_loss', 
        action='store_true',
        default=True,
        help='Use focal loss for class imbalance'
    )
    parser.add_argument(
        '--dropout_rate', 
        type=float, 
        default=0.45,
        help='Dropout rate for regularization'
    )
    parser.add_argument(
        '--dropout_upper', 
        type=float, 
        default=0.70,
        help='Upper limit for dropout sweep'
    )
    
    # Output parameters
    parser.add_argument(
        '--save_models', 
        action='store_true',
        default=True,
        help='Save trained models'
    )
    parser.add_argument(
        '--save_plots', 
        action='store_true',
        default=True,
        help='Save visualization plots'
    )
    
    # MLflow parameters
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default='file:./mlruns',
        help='MLflow tracking URI'
    )

def add_training_args(parser):
    """Add training-specific arguments."""
    parser.add_argument(
        '--model_type',
        type=str,
        default='lstm',
        choices=['lstm', 'transformer', 'enhanced_transformer', 'hybrid_lstm_transformer'],
        help='Type of model to train (default: lstm)'
    )
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=None,  # Will be set based on model type
        help='Learning rate (auto-adjusted for model type)'
    )
    parser.add_argument(
        '--oversample_smote', 
        action='store_true',
        default=False,
        help='Apply SMOTE oversampling'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--optuna_runs', 
        type=int, 
        default=0,
        help='Number of Optuna optimization trials'
    )
    parser.add_argument(
        '--dynamic_k_aggressive', 
        action='store_true',
        default=True,
        help='Use dynamic k-pips calculation'
    )
    
    # Sequence parameters
    parser.add_argument(
        '--seq_len', 
        type=int, 
        default=None,
        help='Override sequence length'
    )

def add_ensemble_args(parser):
    """Add ensemble-specific arguments to CLI"""
    ensemble_group = parser.add_argument_group('Ensemble Options')
    
    ensemble_group.add_argument('--ensemble_mode', action='store_true',
                               help='Train ensemble of models instead of single model')
    
    ensemble_group.add_argument('--ensemble_size', type=int, default=5,
                               help='Number of models in ensemble (default: 5)')
    
    ensemble_group.add_argument('--ensemble_strategy', 
                               choices=['simple_average', 'weighted_average', 'confidence_weighted', 
                                       'meta_learning', 'dynamic'],
                               default='confidence_weighted',
                               help='Ensemble voting strategy (default: confidence_weighted)')
    
    ensemble_group.add_argument('--model_diversity', type=float, default=0.2,
                               help='Diversity factor for ensemble training (default: 0.2)')
    
    ensemble_group.add_argument('--ensemble_models', nargs='+',
                               choices=['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer'],
                               default=['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer'],
                               help='Model types to include in ensemble')
    
    ensemble_group.add_argument('--meta_learning', action='store_true',
                               help='Use meta-learning for ensemble combination')
    
    ensemble_group.add_argument('--dynamic_selection', action='store_true',
                               help='Use dynamic model selection based on performance')
    
    ensemble_group.add_argument('--ensemble_save_dir', type=str, default='ensemble_models',
                               help='Directory to save ensemble models')
    
    ensemble_group.add_argument('--ensemble_load_dir', type=str,
                               help='Directory to load pre-trained ensemble from')

def add_transformer_args(parser):
    """Add Enhanced Transformer specific arguments."""
    transformer_group = parser.add_argument_group('Enhanced Transformer Parameters')
    
    transformer_group.add_argument(
        '--d_model',
        type=int,
        default=128,
        help='Transformer model dimension (must be divisible by nhead)'
    )
    
    transformer_group.add_argument(
        '--nhead',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    
    transformer_group.add_argument(
        '--num_layers',
        type=int,
        default=4,
        help='Number of transformer layers'
    )
    
    transformer_group.add_argument(
        '--ff_dim',
        type=int,
        default=256,
        help='Feed-forward dimension'
    )

def add_enhanced_transformer_args(parser):
    """PDF'deki Enhanced Transformer specific arguments."""
    et_group = parser.add_argument_group('PDF Enhanced Transformer Optimizations')
    
    # Pair-specific configs
    et_group.add_argument(
        '--pair_specific_config',
        action='store_true',
        default=False,
        help='Use pair-specific configurations from PDF recommendations'
    )
    
    # Advanced features from PDF
    et_group.add_argument(
        '--multi_timeframe_features',
        action='store_true',
        default=True,
        help='Enable multi-timeframe feature engineering'
    )
    
    et_group.add_argument(
        '--dynamic_focal_loss',
        action='store_true',
        default=True,
        help='Use dynamic focal loss with volatility adaptation'
    )
    
    et_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation steps for larger effective batch size'
    )
    
    # Attention visualization from PDF
    et_group.add_argument(
        '--save_attention_plots',
        action='store_true',
        default=False,
        help='Save attention pattern visualizations'
    )

def add_tune_args(parser):
    """Add TUNE mode specific arguments."""
    tune_group = parser.add_argument_group('Advanced Tuning Parameters')
    
    # Optuna trials
    tune_group.add_argument(
        '--optuna_trials',
        type=int,
        default=50,
        help='Number of Optuna trials for tuning'
    )
    
    # Scheduler type
    tune_group.add_argument(
        '--scheduler',
        type=str,
        default='OneCycleLR',
        choices=['OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealing'],
        help='Learning rate scheduler type'
    )
    
    # Early stopping
    tune_group.add_argument(
        '--early_stop',
        type=int,
        default=7,
        help='Early stopping patience'
    )
    
    # Search space ranges
    tune_group.add_argument(
        '--lr_range',
        nargs=2,
        type=float,
        default=[1e-5, 1e-2],
        help='Learning rate search range (min max)'
    )
    
    tune_group.add_argument(
        '--batch_range',
        nargs='+',
        type=int,
        default=[16, 32, 64, 128],
        help='Batch size options'
    )
    
    tune_group.add_argument(
        '--dropout_range',
        nargs=2,
        type=float,
        default=[0.1, 0.7],
        help='Dropout rate search range (min max)'
    )
    
    # Advanced tuning options
    tune_group.add_argument(
        '--tune_architecture',
        action='store_true',
        default=False,
        help='Tune model architecture (layers, hidden size)'
    )
    
    tune_group.add_argument(
        '--tune_preprocessing',
        action='store_true',
        default=False,
        help='Tune preprocessing parameters'
    )
    
    tune_group.add_argument(
        '--cross_validate',
        action='store_true',
        default=False,
        help='Use cross-validation for robust tuning'
    )
    
    tune_group.add_argument(
        '--save_all_models',
        action='store_true',
        default=False,
        help='Save all tuned models (not just best)'
    )
    
    tune_group.add_argument(
        '--tune_timeout',
        type=int,
        default=3600,  # 1 hour
        help='Maximum tuning time in seconds'
    )

def add_inference_args(parser):
    """Add inference-specific arguments."""
    # Pairs to predict
    parser.add_argument(
        '--pairs',
        nargs='+',
        default=['EUR_USD', 'GBP_USD', 'USD_JPY'],
        help='Currency pairs for prediction'
    )
    
    # Data parameters
    parser.add_argument(
        '--lookback_candles',
        type=int,
        default=30000,
        help='Number of candles to fetch'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        default='M15',
        choices=['M1', 'M5', 'M15', 'H1', 'H4', 'D'],
        help='Data granularity'
    )
    
    parser.add_argument(
        '--seq_len',
        type=int,
        default=64,
        help='Sequence length for predictions'
    )
    
    # Model and output directories
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='predictions',
        help='Directory to save prediction CSVs'
    )
    
    # Cache option
    parser.add_argument(
        '--cache',
        action='store_true',
        default=False,
        help='Use cached data instead of fetching from API'
    )
    
    # Filtering options
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.1,
        help='Minimum confidence threshold for predictions'
    )
    
    parser.add_argument(
        '--filter_flat',
        action='store_true',
        default=True,
        help='Filter out flat (neutral) predictions'
    )

def add_backtest_args_fixed(parser):
    """Add backtesting-specific arguments."""
    # Walk-forward parameters
    parser.add_argument('--train_window_days', type=int, default=90, help='Training window in days')
    parser.add_argument('--retrain_freq_days', type=int, default=14, help='Retrain frequency in days')
    
    # Multi-timeframe setup
    parser.add_argument('--signal_tf', type=str, default='M15', choices=['M1', 'M5', 'M15', 'H1', 'H4'], help='Signal timeframe')
    parser.add_argument('--exec_tf', type=str, default='M5', choices=['M1', 'M5', 'M15', 'H1'], help='Execution timeframe')
    
    # Fine-tuning options
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Use fine-tuning for retraining')
    parser.add_argument('--epochs_fine', type=int, default=5, help='Epochs for fine-tuning')
    
    # Signal generation parameters
    parser.add_argument('--signal_threshold', type=float, default=0.4, help='Signal threshold')
    parser.add_argument('--k_pips_mult', type=float, default=0.8, help='K-pips multiplier')
    
    # Trading simulation
    parser.add_argument('--spread_multiplier', type=float, default=1.2, help='Spread multiplier')
    parser.add_argument('--commission_pips', type=float, default=0.0, help='Commission in pips')
    parser.add_argument('--slippage_pips', type=float, default=0.1, help='Slippage in pips')
    parser.add_argument('--max_positions', type=int, default=1, help='Max concurrent positions')
    
    # Time range
    parser.add_argument('--start_date', type=str, default=None, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Backtest end date (YYYY-MM-DD)')
    
    # Risk management
    parser.add_argument('--position_size', type=float, default=0.01, help='Position size (lot size)')
    parser.add_argument('--stop_loss_pips', type=float, default=None, help='Stop loss in pips')
    parser.add_argument('--take_profit_pips', type=float, default=None, help='Take profit in pips')

def add_temporal_args(parser):
    """Add temporal training arguments."""
    temporal_group = parser.add_argument_group('Temporal Training Parameters')
    
    temporal_group.add_argument(
        '--train_start',
        type=str,
        default='2024-01-01',
        help='Training period start date (YYYY-MM-DD)'
    )
    
    temporal_group.add_argument(
        '--train_end',
        type=str,
        default='2024-06-30',
        help='Training period end date (YYYY-MM-DD)'
    )
    
    temporal_group.add_argument(
        '--signal_start',
        type=str,
        default='2024-07-01',
        help='Signal generation period start date (YYYY-MM-DD)'
    )
    
    temporal_group.add_argument(
        '--signal_end',
        type=str,
        default='2024-12-31',
        help='Signal generation period end date (YYYY-MM-DD)'
    )
    
    temporal_group.add_argument(
        '--temporal_epochs',
        type=int,
        default=50,
        help='Number of epochs for temporal training'
    )
    
    temporal_group.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold for signals'
    )

def validate_transformer_args(args):
    """Validate and fix Enhanced Transformer arguments."""
    # Skip validation for inference command (doesn't have training parameters)
    if getattr(args, 'command', None) in ['inference', 'temporal']:
        return args
    
    if getattr(args, 'model', 'lstm') == 'enhanced_transformer':
        d_model = getattr(args, 'd_model', 128)
        nhead = getattr(args, 'nhead', 8)
        
        # Check d_model % nhead == 0
        if d_model % nhead != 0:
            print(f"⚠️ d_model ({d_model}) not divisible by nhead ({nhead})!")
            
            # Auto-fix nhead
            valid_heads = [h for h in [1, 2, 4, 8, 16, 32] if d_model % h == 0 and h <= d_model]
            if valid_heads:
                old_nhead = nhead
                args.nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[0]
                print(f"🔧 nhead auto-fixed: {old_nhead} → {args.nhead}")
            else:
                # Fix d_model instead
                old_d_model = d_model
                args.d_model = d_model + (nhead - (d_model % nhead))
                print(f"🔧 d_model auto-fixed: {old_d_model} → {args.d_model}")
        
        # Set Enhanced Transformer specific defaults (only for training commands)
        if hasattr(args, 'learning_rate') and args.learning_rate is None:
            args.learning_rate = 5e-4  # Lower LR for transformer
            print(f"🔧 Learning rate set for Enhanced Transformer: {args.learning_rate}")
        
        if hasattr(args, 'batch_size') and args.batch_size == 32:  # Default batch size
            args.batch_size = 16  # Smaller batch for transformer
            print(f"🔧 Batch size adjusted for Enhanced Transformer: {args.batch_size}")
        
        if hasattr(args, 'dropout_rate') and args.dropout_rate == 0.45:  # Default LSTM dropout
            args.dropout_rate = 0.1  # Lower dropout for transformer
            print(f"🔧 Dropout rate adjusted for Enhanced Transformer: {args.dropout_rate}")
    
    else:
        # Set LSTM defaults (only for training commands)
        if hasattr(args, 'learning_rate') and args.learning_rate is None:
            args.learning_rate = 1e-3
    
    return args

def setup_device() -> torch.device:
    """Setup PyTorch device with GPU memory check."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔧 Using GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 4:
            print(f"   ⚠️ Limited GPU memory - consider smaller models")
    else:
        device = torch.device('cpu')
        print(f"🔧 Using CPU")
    
    return device

def create_output_directories() -> str:
    """Create output directories and return timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for directory in ['logs', 'models', 'results', 'backtest_results', 'tune_results', 'predictions', 'ensemble_models']:
        os.makedirs(directory, exist_ok=True)
    
    return timestamp

def setup_csv_logging(timestamp: str, command: str) -> str:
    """Setup CSV logging for detailed metrics."""
    if command == 'backtest':
        csv_path = f'logs/backtest_run_{timestamp}.csv'
        header = "pair,window_start,window_end,train_samples,val_acc,total_trades,winning_trades,total_pnl,k_pips,ending_equity\n"
    elif command == 'tune':
        csv_path = f'logs/tune_run_{timestamp}.csv'
        header = "pair,trial,model_type,val_acc,train_acc,lr,batch_size,dropout,d_model,nhead,num_layers,seq_len,trial_time\n"
    elif command == 'ensemble':
        csv_path = f'logs/ensemble_run_{timestamp}.csv'
        header = "pair,ensemble_size,strategy,val_acc,best_individual,improvement,diversity\n"
    else:
        csv_path = f'logs/training_run_{timestamp}.csv'
        header = "pair,horizon,seq_len,min_ratio,train_acc,val_acc,overfit_gap,model_type\n"
    
    with open(csv_path, 'w') as f:
        f.write(header)
    
    print(f"📝 CSV logging: {csv_path}")
    return csv_path

def main():
    """Main pipeline dispatcher."""
    print("🚀 MODULAR MULTI-PAIR LSTM/ENHANCED TRANSFORMER SYSTEM WITH ENSEMBLE")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()

    if args.command == 'train':
        if getattr(args, 'ensemble_mode', False):
            return train_ensemble_command(args)
        else:
            return train_single_model(args)

    elif args.command == 'ensemble':
        return train_ensemble_command(args)

    elif args.command == 'inference':
        if getattr(args, 'ensemble_load_dir', None):
            return ensemble_inference_command(args)
        else:
            return run_inference_pipeline(args)

    elif args.command == 'tune':
        return run_tuning_pipeline(args)

    elif args.command == 'backtest':
        return run_backtest_pipeline(args)
    
    elif args.command == 'evaluate':
        return run_evaluation_pipeline(args)
    
    elif args.command == 'temporal':
        return run_temporal_pipeline(args)

    else:
        print("❌ Invalid command. Use --help for available options.")
        return 1

def train_ensemble_command(args):
    """Train ensemble of models"""
    print(f"\n🎭 ENSEMBLE TRAINING PIPELINE")
    print(f"{'='*60}")
    
    # Setup
    device = setup_device()
    timestamp = create_output_directories()
    csv_path = setup_csv_logging(timestamp, 'ensemble')
    
    print(f"🎯 Ensemble Configuration:")
    print(f"   Strategy: {args.ensemble_strategy}")
    print(f"   Size: {args.ensemble_size}")
    print(f"   Models: {', '.join(args.ensemble_models)}")
    print(f"   Diversity Factor: {args.model_diversity}")
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = config.load(args.config)
        print(f"📁 Loaded config from {args.config}")
    else:
        print(f"⚠️ Config file not found, using defaults")
        cfg = config.get_default_config()
    
    # Create components
    print("\n🔧 Initializing components...")
    fetcher = create_fetcher(cfg, cache_enabled=args.cache)
    
    # Fetch data
    print("\n📊 Fetching market data...")
    pair_data = fetcher.fetch_all_pairs(
        granularity=cfg['data']['granularity'], 
        count=3000, 
        lookback_candles=cfg['data']['lookback_candles'],
        pairs=args.pairs
    )
    
    if len(pair_data) == 0:
        print("❌ No pair data fetched!")
        return
    
    results = {}
    
    for pair_name, data in pair_data.items():
        print(f"\n{'='*50}")
        print(f"🎯 Processing {pair_name}...")
        print(f"{'='*50}")
        
        # Apply rescue mode for USD_JPY & XAU_USD
        current_target_mode = cfg['model']['target_mode']
        if pair_name in ['USD_JPY', 'XAU_USD']:
            current_target_mode = 'three_class'
            print(f"   🛡️ USD_JPY/XAU_USD rescue: forcing three_class")
        
        # Create preprocessor
        preprocessor = create_preprocessor(pair_name, cfg)
        preprocessor.target_mode = current_target_mode
        
        # Prepare data
        features, target = preprocessor.prepare_pair_data(
            data, 
            dynamic_k_aggressive=args.dynamic_k_aggressive,
            seq_len=args.seq_len
        )
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(target if target.ndim == 1 else np.argmax(target, axis=1))
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"   Data split: Train={len(X_train)}, Val={len(X_val)}")
        
        # Initialize ensemble trainer
        trainer = EnsembleTrainer(device=device)
        
        # Generate diverse model configurations
        base_config = config.get_model_config('lstm', current_target_mode)
        model_configs = trainer.generate_diverse_configs(
            base_config=base_config,
            n_models=args.ensemble_size,
            diversity_factor=args.model_diversity
        )
        
        # Override with specific model types if provided
        if args.ensemble_models:
            for i, model_type in enumerate(args.ensemble_models):
                if i < len(model_configs):
                    model_configs[i] = config.get_model_config(model_type, current_target_mode)
                    model_configs[i]['model_type'] = model_type
        
        # Train ensemble models
        print(f"🏋️ Training ensemble models...")
        models = trainer.train_diverse_models(
            X_train=X_train,
            y_train=y_train,
            model_configs=model_configs,
            epochs=args.epochs
        )
        
        if not models:
            print(f"❌ No models successfully trained for {pair_name}")
            continue
        
        # Calculate model weights
        print("📊 Calculating model weights...")
        model_weights = trainer.calculate_model_weights(models, X_val, y_val)
        
        # Create ensemble classifier
        ensemble = EnsembleClassifier(
            models=models,
            device=device,
            strategy=args.ensemble_strategy
        )
        ensemble.set_performance_weights(model_weights)
        
        # Evaluate ensemble
        print("🎯 Evaluating ensemble...")
        ensemble.eval()
        with torch.no_grad():
            ensemble_pred = ensemble.predict(X_val)
            individual_preds = ensemble.get_individual_predictions(X_val)
            
            # Calculate metrics
            if ensemble_pred.dim() > 1:
                ensemble_classes = torch.argmax(ensemble_pred, dim=1)
            else:
                ensemble_classes = (ensemble_pred > 0.5).long()
            
            ensemble_accuracy = (ensemble_classes == y_val).float().mean().item()
            
            # Individual accuracies
            individual_accuracies = {}
            for name, pred in individual_preds.items():
                if pred.dim() > 1:
                    pred_classes = torch.argmax(pred, dim=1)
                else:
                    pred_classes = (pred > 0.5).long()
                acc = (pred_classes == y_val).float().mean().item()
                individual_accuracies[name] = acc
            
            best_individual = max(individual_accuracies.values())
            improvement = ensemble_accuracy - best_individual
            
            print(f"   ✅ Ensemble Results:")
            print(f"      Accuracy: {ensemble_accuracy:.4f}")
            print(f"      Best Individual: {best_individual:.4f}")
            print(f"      Improvement: +{improvement:.4f}")
        
        # Save ensemble
        save_dir = Path(args.ensemble_save_dir) / pair_name
        trainer.save_ensemble(
            models=models,
            save_dir=str(save_dir),
            ensemble_config={
                'strategy': args.ensemble_strategy,
                'model_weights': model_weights,
                'model_configs': model_configs,
                'ensemble_accuracy': ensemble_accuracy,
                'individual_accuracies': individual_accuracies
            }
        )
        
        # Diversity metrics
        diversity = trainer.calculate_diversity_metrics(models, X_val)
        
        results[pair_name] = {
            'ensemble': ensemble,
            'accuracy': ensemble_accuracy,
            'improvement': improvement,
            'model_count': len(models),
            'diversity': diversity
        }
        
        # Log to CSV
        with open(csv_path, 'a') as f:
            f.write(f"{pair_name},{args.ensemble_size},{args.ensemble_strategy},{ensemble_accuracy:.4f},{best_individual:.4f},{improvement:.4f},{diversity:.4f}\n")
    
    # Overall summary
    print(f"\n🎉 Ensemble Training Complete!")
    print(f"   Pairs processed: {len(results)}")
    
    if results:
        total_accuracy = np.mean([r['accuracy'] for r in results.values()])
        total_improvement = np.mean([r['improvement'] for r in results.values()])
        
        print(f"   Average Ensemble Accuracy: {total_accuracy:.4f}")
        print(f"   Average Improvement: +{total_improvement:.4f}")
    
    return results

def ensemble_inference_command(args):
    """Run inference with pre-trained ensemble"""
    print(f"\n🔮 ENSEMBLE INFERENCE PIPELINE")
    print(f"{'='*60}")
    
    if not args.ensemble_load_dir:
        print("❌ --ensemble_load_dir required for ensemble inference")
        return
    
    device = setup_device()
    
    print(f"📂 Loading ensemble from: {args.ensemble_load_dir}")
    
    # Load ensemble configuration
    load_path = Path(args.ensemble_load_dir)
    
    results = {}
    for pair_dir in load_path.iterdir():
        if pair_dir.is_dir():
            pair_name = pair_dir.name
            print(f"🔄 Loading ensemble for {pair_name}...")
            
            # Load ensemble config
            config_file = pair_dir / "ensemble_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    ensemble_config = json.load(f)
                
                model_configs = ensemble_config.get('model_configs', [])
                
                # Load models
                trainer = EnsembleTrainer(device=device)
                models = trainer.load_ensemble(str(pair_dir), model_configs)
                
                if models:
                    # Create ensemble classifier
                    ensemble = EnsembleClassifier(
                        models=models,
                        device=device,
                        strategy=ensemble_config.get('strategy', 'confidence_weighted')
                    )
                    
                    # Set weights if available
                    model_weights = ensemble_config.get('model_weights', {})
                    if model_weights:
                        ensemble.set_performance_weights(model_weights)
                    
                    results[pair_name] = {
                        'ensemble': ensemble,
                        'config': ensemble_config
                    }
                    
                    print(f"   ✅ Ensemble loaded: {len(models)} models")
                else:
                    print(f"   ⚠️ Failed to load models for {pair_name}")
            else:
                print(f"   ⚠️ No config found for {pair_name}")
    
    print(f"\n🎭 Loaded {len(results)} ensemble(s)")
    
    # TODO: Actual inference would go here
    print("\n🚧 Inference functionality will be implemented in the next version")
    
    return results

def run_inference_pipeline(args):
    """Run inference pipeline to generate prediction CSVs."""
    print(f"\n🔮 ENHANCED TRANSFORMER INFERENCE PIPELINE")
    print(f"{'='*60}")
    
    # Validate model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    # Check for trained models
    model_files = list(model_dir.glob("*enhanced_transformer_model_*.pth"))
    if not model_files:
        print(f"❌ No Enhanced Transformer models found in {model_dir}")
        print("💡 Train models first using: python -m src.cli train --model enhanced_transformer")
        return
    
    print(f"📁 Found {len(model_files)} trained models")
    
    # Run inference
    try:
        results = run_inference_cli(
            pairs=args.pairs,
            lookback_candles=args.lookback_candles,
            granularity=args.granularity,
            seq_len=args.seq_len,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            cache_enabled=args.cache
        )
        
        if results:
            print(f"\n🎉 Inference completed successfully!")
            print(f"📁 CSV files generated in: {args.output_dir}")
            
            # Show file details
            for pair_name, csv_path in results.items():
                csv_file = Path(csv_path)
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    print(f"   📈 {pair_name}: {len(df)} predictions → {csv_file.name}")
        else:
            print(f"❌ No predictions generated")
            
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")

def run_tuning_pipeline(args):
    """Enhanced hyperparameter tuning pipeline with advanced features."""
    print(f"\n🎛️ ADVANCED HYPERPARAMETER TUNING PIPELINE")
    print(f"{'='*60}")
    
    # Setup
    device = setup_device()
    timestamp = create_output_directories()
    csv_path = setup_csv_logging(timestamp, 'tune')
    
    model_type = getattr(args, 'model', 'lstm')
    
    print(f"🎯 Tuning Configuration:")
    print(f"   🤖 Model type: {model_type.upper()}")
    print(f"   🔬 Optuna trials: {getattr(args, 'optuna_trials', 50)}")
    print(f"   ⏰ Timeout: {getattr(args, 'tune_timeout', 3600)}s")
    print(f"   📊 Scheduler: {getattr(args, 'scheduler', 'OneCycleLR')}")
    print(f"   🛑 Early stop: {getattr(args, 'early_stop', 7)} epochs")
    print(f"   🏗️ Architecture tuning: {getattr(args, 'tune_architecture', False)}")
    print(f"   🔧 Preprocessing tuning: {getattr(args, 'tune_preprocessing', False)}")
    print(f"   ✅ Cross-validation: {getattr(args, 'cross_validate', False)}")
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = config.load(args.config)
        print(f"📁 Loaded config from {args.config}")
    else:
        print(f"⚠️ Config file not found, using defaults")
        cfg = config.get_default_config()
    
    # Override config with CLI arguments
    cfg = override_config_with_args(cfg, args)
    
    # Apply PDF optimizations
    cfg = apply_pdf_optimizations(args, cfg)
    
    # Create components
    print("\n🔧 Initializing tuning components...")
    fetcher = create_fetcher(cfg, cache_enabled=args.cache)
    
    # Fetch data
    print("\n📊 Fetching market data for tuning...")
    pair_data = fetcher.fetch_all_pairs(
        granularity=cfg['data']['granularity'], 
        count=3000, 
        lookback_candles=cfg['data']['lookback_candles'],
        pairs=args.pairs
    )
    
    if len(pair_data) == 0:
        print("❌ No pair data fetched!")
        return
    
    # Initialize advanced tuner
    from .tuning.advanced_tuner import AdvancedTuner
    
    tuner = AdvancedTuner(
        model_type=model_type,
        device=device,
        config=cfg,
        optuna_trials=getattr(args, 'optuna_trials', 50),
        timeout=getattr(args, 'tune_timeout', 3600),
        cross_validate=getattr(args, 'cross_validate', False),
        tune_architecture=getattr(args, 'tune_architecture', False),
        tune_preprocessing=getattr(args, 'tune_preprocessing', False)
    )
    
    tuning_results = {}
    
    for pair_name, data in pair_data.items():
        print(f"\n{'='*60}")
        print(f"🎛️ Advanced Tuning: {pair_name} with {model_type.upper()}")
        print(f"{'='*60}")
        
        # Apply rescue mode for USD_JPY & XAU_USD
        current_target_mode = cfg['model']['target_mode']
        if pair_name in ['USD_JPY', 'XAU_USD']:
            current_target_mode = 'three_class'
            print(f"   🛡️ USD_JPY/XAU_USD rescue: forcing three_class")
        
        try:
            # Run advanced tuning
            best_params, study_results, tuning_metrics = tuner.tune_pair(
                pair_name=pair_name,
                data=data,
                target_mode=current_target_mode,
                csv_logger=csv_path
            )
            
            tuning_results[pair_name] = {
                'status': 'success',
                'best_params': best_params,
                'study_results': study_results,
                'tuning_metrics': tuning_metrics,
                'model_type': model_type,
                'target_mode': current_target_mode
            }
            
            print(f"✅ {pair_name} tuning completed!")
            print(f"   🏆 Best validation accuracy: {tuning_metrics.get('best_val_acc', 0):.2f}%")
            print(f"   📊 Trials completed: {len(study_results.get('trials', []))}")
            
        except Exception as e:
            print(f"❌ {pair_name} tuning failed: {str(e)}")
            tuning_results[pair_name] = {
                'status': 'failed',
                'error': str(e),
                'model_type': model_type
            }
    
    # Generate tuning report
    print(f"\n📊 TUNING RESULTS SUMMARY")
    print(f"{'='*50}")
    
    successful_pairs = []
    failed_pairs = []
    
    for pair_name, result in tuning_results.items():
        if result['status'] == 'success':
            successful_pairs.append(pair_name)
            metrics = result['tuning_metrics']
            print(f"✅ {pair_name}: {metrics.get('best_val_acc', 0):.2f}% val_acc, "
                  f"{len(result['study_results'].get('trials', []))} trials")
        else:
            failed_pairs.append(pair_name)
            print(f"❌ {pair_name}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📈 Success Rate: {len(successful_pairs)}/{len(tuning_results)} "
          f"({100*len(successful_pairs)/len(tuning_results):.1f}%)")
    
    # Save comprehensive tuning results
    save_tuning_results(tuning_results, timestamp, args)
    
    # Generate recommendations
    generate_tuning_recommendations(tuning_results, model_type)
    
    print(f"\n🎉 Advanced tuning pipeline completed!")
    print(f"📁 Results saved to: tune_results/")
    print(f"📊 CSV log: {csv_path}")

def save_tuning_results(tuning_results: Dict, timestamp: str, args):
    """Save comprehensive tuning results."""
    
    # Save JSON results
    results_file = f'tune_results/tuning_results_{timestamp}.json'
    
    # Convert results to JSON-serializable format
    json_results = {}
    for pair_name, result in tuning_results.items():
        json_results[pair_name] = {
            'status': result['status'],
            'model_type': result.get('model_type', 'lstm'),
            'target_mode': result.get('target_mode', 'binary')
        }
        
        if result['status'] == 'success':
            json_results[pair_name].update({
                'best_params': result['best_params'],
                'best_val_acc': result['tuning_metrics'].get('best_val_acc', 0),
                'trials_completed': len(result['study_results'].get('trials', [])),
                'tuning_time': result['tuning_metrics'].get('total_time', 0)
            })
        else:
            json_results[pair_name]['error'] = result.get('error', 'Unknown')
    
    # Add metadata
    metadata = {
        'timestamp': timestamp,
        'command_args': {
            'model': getattr(args, 'model', 'lstm'),
            'optuna_trials': getattr(args, 'optuna_trials', 50),
            'tune_timeout': getattr(args, 'tune_timeout', 3600),
            'scheduler': getattr(args, 'scheduler', 'OneCycleLR'),
            'early_stop': getattr(args, 'early_stop', 7),
            'tune_architecture': getattr(args, 'tune_architecture', False),
            'tune_preprocessing': getattr(args, 'tune_preprocessing', False),
            'cross_validate': getattr(args, 'cross_validate', False)
        },
        'results': json_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"📄 Tuning results saved: {results_file}")
    
    # Save best parameters as config files
    for pair_name, result in tuning_results.items():
        if result['status'] == 'success':
            best_config = {
                'pair_name': pair_name,
                'model_type': result['model_type'],
                'target_mode': result['target_mode'],
                'best_parameters': result['best_params'],
                'validation_accuracy': result['tuning_metrics'].get('best_val_acc', 0),
                'timestamp': timestamp
            }
            
            config_file = f'tune_results/best_config_{pair_name}_{timestamp}.yaml'
            config.save(best_config, config_file)

def generate_tuning_recommendations(tuning_results: Dict, model_type: str):
    """Generate recommendations based on tuning results."""
    
    print(f"\n💡 TUNING RECOMMENDATIONS")
    print(f"{'='*50}")
    
    successful_results = [r for r in tuning_results.values() if r['status'] == 'success']
    
    if not successful_results:
        print("❌ No successful tuning results to analyze")
        return
    
    # Analyze best parameters
    all_best_params = [r['best_params'] for r in successful_results]
    
    # Learning rate analysis
    lrs = [p.get('learning_rate', 1e-3) for p in all_best_params]
    avg_lr = np.mean(lrs)
    print(f"📊 Average optimal learning rate: {avg_lr:.6f}")
    
    # Batch size analysis
    batch_sizes = [p.get('batch_size', 32) for p in all_best_params]
    most_common_batch = max(set(batch_sizes), key=batch_sizes.count)
    print(f"📊 Most common optimal batch size: {most_common_batch}")
    
    # Dropout analysis
    dropouts = [p.get('dropout_rate', 0.45) for p in all_best_params]
    avg_dropout = np.mean(dropouts)
    print(f"📊 Average optimal dropout rate: {avg_dropout:.3f}")
    
    if model_type == 'enhanced_transformer':
        # Transformer-specific analysis
        d_models = [p.get('d_model', 128) for p in all_best_params]
        avg_d_model = np.mean(d_models)
        print(f"📊 Average optimal d_model: {avg_d_model:.0f}")
        
        nheads = [p.get('nhead', 8) for p in all_best_params]
        most_common_nhead = max(set(nheads), key=nheads.count)
        print(f"📊 Most common optimal nhead: {most_common_nhead}")
    
    # Performance analysis
    val_accs = [r['tuning_metrics'].get('best_val_acc', 0) for r in successful_results]
    avg_val_acc = np.mean(val_accs)
    best_val_acc = max(val_accs)
    
    print(f"📊 Average validation accuracy: {avg_val_acc:.2f}%")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Generate command recommendation
    print(f"\n🎯 RECOMMENDED TRAINING COMMAND:")
    recommended_cmd = (
        f"python -m src.cli train "
        f"--model {model_type} "
        f"--learning_rate {avg_lr:.6f} "
        f"--batch_size {most_common_batch} "
        f"--dropout_rate {avg_dropout:.3f} "
    )
    
    if model_type == 'enhanced_transformer':
        recommended_cmd += (
            f"--d_model {int(avg_d_model)} "
            f"--nhead {most_common_nhead} "
        )
    
    recommended_cmd += "--cache --save_models --save_plots"
    
    print(f"   {recommended_cmd}")

def run_training_pipeline(args):
    """Enhanced training pipeline with Enhanced Transformer support."""
    # Setup
    device = setup_device()
    timestamp = create_output_directories()
    csv_path = setup_csv_logging(timestamp, 'train')
    
    # Model type
    model_type = getattr(args, 'model', 'lstm')
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = config.load(args.config)
        print(f"📁 Loaded config from {args.config}")
    else:
        print(f"⚠️ Config file not found, using defaults")
        cfg = config.get_default_config()
    
    # Override config with CLI arguments
    cfg = override_config_with_args(cfg, args)
    
    # Apply PDF optimizations
    cfg = apply_pdf_optimizations(args, cfg)
    
    print(f"📋 Configuration:")
    print(f"   🤖 Model type: {model_type.upper()}")
    print(f"   🎯 Target mode: {cfg['model']['target_mode']}")
    print(f"   📊 Granularity: {cfg['data']['granularity']}")
    print(f"   📈 Lookback candles: {cfg['data']['lookback_candles']}")
    
    if model_type == 'enhanced_transformer':
        trans_cfg = cfg.get('transformer', {})
        print(f"   🤖 Enhanced Transformer config:")
        print(f"      d_model: {trans_cfg.get('d_model', 128)}")
        print(f"      heads: {trans_cfg.get('nhead', 8)}")
        print(f"      layers: {trans_cfg.get('num_layers', 4)}")
        print(f"      ff_dim: {trans_cfg.get('ff_dim', 256)}")
        print(f"   ⚙️ PDF Optimizations:")
        print(f"      Multi-timeframe: {cfg.get('features', {}).get('multi_timeframe', False)}")
        print(f"      Dynamic focal loss: {cfg.get('features', {}).get('dynamic_focal_loss', False)}")
        print(f"      Gradient accumulation: {cfg.get('features', {}).get('gradient_accumulation', 1)}")
    
    # Create components
    print("\n🔧 Initializing components...")
    fetcher = create_fetcher(cfg, cache_enabled=args.cache)
    trainer = create_trainer(cfg, device)
    
    # Fetch data
    print("\n📊 Fetching market data...")
    pair_data = fetcher.fetch_all_pairs(
        granularity=cfg['data']['granularity'], 
        count=3000, 
        lookback_candles=cfg['data']['lookback_candles'],
        pairs=args.pairs
    )
    
    if len(pair_data) == 0:
        print("❌ No pair data fetched!")
        return
    
    results_summary = {}
    
    for pair_name, data in pair_data.items():
        print(f"\n{'='*50}")
        print(f"🎯 Processing {pair_name} with {model_type.upper()}...")
        print(f"{'='*50}")
        
        # Apply rescue mode for USD_JPY & XAU_USD
        current_target_mode = cfg['model']['target_mode']
        current_smote = args.oversample_smote
        
        if pair_name in ['USD_JPY', 'XAU_USD']:
            current_target_mode = 'three_class'
            current_smote = True
            print(f"   🛡️ USD_JPY/XAU_USD rescue: forcing three_class + SMOTE")
        
        # Create preprocessor
        preprocessor = create_preprocessor(pair_name, cfg)
        preprocessor.target_mode = current_target_mode
        preprocessor.use_smote = current_smote
        
        # Apply pair-specific configs from PDF
        if model_type == 'enhanced_transformer' and args.pair_specific_config:
            apply_pair_specific_configs(cfg, pair_name)
            print(f"   🔧 Applied PDF pair-specific config for {pair_name}")
        
        # Optuna optimization
        best_params = {}
        if args.optuna_runs > 0:
            print(f"🔍 Running {model_type.upper()} Optuna optimization...")
            features, target = preprocessor.prepare_pair_data(
                data, 
                dynamic_k_aggressive=args.dynamic_k_aggressive,
                seq_len=args.seq_len
            )
            
            if len(features) >= 1000:
                X_temp, y_temp = preprocessor.create_sequences(features, target)
                optimizer = create_optimizer(pair_name, device)
                best_params = optimizer.optimize(X_temp, y_temp, args.optuna_runs, current_target_mode)
        
        # Prepare final data
        horizon = max(32, best_params.get('horizon', 64))
        final_seq_len = best_params.get('seq_len', args.seq_len)
        
        features, target = preprocessor.prepare_pair_data(
            data, 
            horizon=horizon,
            dynamic_k_aggressive=args.dynamic_k_aggressive,
            seq_len=final_seq_len
        )
        
        if len(features) < 1000:
            print(f"❌ {pair_name}: Insufficient data!")
            results_summary[pair_name] = {'status': 'insufficient_data'}
            continue
        
        # Create sequences
        X, y = preprocessor.create_sequences(features, target)
        
        if len(X) < 200:
            print(f"❌ {pair_name}: Insufficient sequences!")
            results_summary[pair_name] = {'status': 'insufficient_sequences'}
            continue
        
        # Check for single class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2 and current_target_mode == 'binary':
            print(f"❌ {pair_name}: Single class problem - skipping training")
            results_summary[pair_name] = {'status': 'single_class_problem'}
            continue
        elif len(unique_classes) < 2 and current_target_mode == 'three_class':
            print(f"⚠️ {pair_name}: Single class in three_class mode - proceeding with SMOTE")
        
        # Train model using the unified trainer
        try:
            model, history = trainer.train_pair_model(
                pair_name=pair_name,
                X=X,
                y=y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout=args.dropout_rate,
                model_type=model_type
            )
            
            # Optional walk-forward validation
            if getattr(args, 'walk_forward_validation', False):
                print(f"\n🔄 Running Walk-Forward Validation for {pair_name}...")
                wf_results = trainer.run_walk_forward_validation(pair_name, X, y, model_type)
                if wf_results:
                    print(f"   📊 Walk-Forward Val Acc: {wf_results.get('mean_val_acc', 0):.3f}")
                    
        except Exception as e:
            print(f"❌ {pair_name} training failed: {str(e)}")
            results_summary[pair_name] = {'status': 'training_failed', 'error': str(e)}
            continue
        
        # Save model if requested
        if args.save_models:
            model_data = {
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'input_size': X.shape[2],
                'history': history,
                'feature_columns': preprocessor.feature_columns,
                'pair_name': pair_name,
                'target_mode': current_target_mode,
                'best_params': best_params,
                'config': cfg
            }
            
            torch.save(model_data, f'models/{pair_name}_{model_type}_model_{timestamp}.pth')
        
        # Calculate metrics
        best_val_acc = max(history['val_acc'])
        final_train_acc = history['train_acc'][-1]
        
        # Calculate minority ratio
        if current_target_mode == 'binary':
            final_class_counts = np.bincount(y.astype(int))
            minority_ratio = final_class_counts.min() / len(y) if len(final_class_counts) > 1 else 0.0
        else:
            unique, counts = np.unique(y.astype(int), return_counts=True)
            minority_ratio = counts.min() / len(y) if len(counts) > 1 else 0.0
        
        overfit_gap = final_train_acc - best_val_acc
        
        # Log to CSV
        with open(csv_path, 'a') as f:
            f.write(f"{pair_name},{horizon},{preprocessor.sequence_length},{minority_ratio:.3f},{final_train_acc:.2f},{best_val_acc:.2f},{overfit_gap:.2f},{model_type}\n")
        
        # Store results
        results_summary[pair_name] = {
            'status': 'success',
            'model_type': model_type,
            'best_val_acc': best_val_acc,
            'final_train_acc': final_train_acc,
            'best_params': best_params,
            'data_points': len(X),
            'minority_ratio': minority_ratio,
            'overfit_gap': overfit_gap,
            'target_mode': current_target_mode
        }
        
        print(f"✅ {pair_name} {model_type.upper()} model completed!")
    
    # Generate summary report
    print_summary_report(results_summary, {})
    
    # Save final configuration
    final_config = {
        'system_params': {
            'command': 'train',
            'model_type': model_type,
            'target_mode': args.target_mode,
            'use_focal_loss': args.use_focal_loss,
            'oversample_smote': args.oversample_smote,
            'dropout_rate': args.dropout_rate,
            'granularity': cfg['data']['granularity'],
            'lookback_candles': cfg['data']['lookback_candles'],
            'dynamic_k_aggressive': args.dynamic_k_aggressive,
            'seq_len_override': args.seq_len
        },
        'results_summary': results_summary,
        'timestamp': datetime.now().isoformat()
    }
    
    config.save(final_config, f'results/final_config_{timestamp}.yaml')
    
    # Final summary
    successful_pairs = sum(1 for r in results_summary.values() if r['status'] == 'success')
    total_pairs = len(results_summary)
    
    print(f"\n🎉 Training pipeline completed!")
    print(f"📁 {successful_pairs} models trained successfully")
    print(f"📈 Success rate: {successful_pairs}/{total_pairs} ({100*successful_pairs/total_pairs:.1f}%)")
    print(f"📝 CSV metrics: {csv_path}")
    print(f"🔧 Configuration saved to: results/final_config_{timestamp}.yaml")

def apply_pdf_optimizations(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """PDF'deki tüm optimizasyonları uygula"""
    
    if args.model == 'enhanced_transformer':
        # PDF'deki scheduler optimization
        if hasattr(args, 'scheduler') and args.scheduler == 'OneCycleLR':
            cfg['training']['scheduler'] = 'OneCycleLR'
            cfg['training']['scheduler_params'] = {
                'max_lr': args.learning_rate or 2e-4,
                'pct_start': 0.3,
                'div_factor': 25.0,
                'final_div_factor': 10000.0
            }
        
        # PDF'deki pair-specific configurations
        if args.pair_specific_config:
            apply_pair_specific_configs(cfg, args.pairs)
        
        # PDF'deki advanced features
        cfg['features'] = {
            'multi_timeframe': getattr(args, 'multi_timeframe_features', True),
            'dynamic_focal_loss': getattr(args, 'dynamic_focal_loss', True),
            'gradient_accumulation': getattr(args, 'gradient_accumulation_steps', 4),
            'save_attention_plots': getattr(args, 'save_attention_plots', False)
        }
    
    return cfg

def apply_pair_specific_configs(cfg: Dict[str, Any], pairs: List[str]):
    """PDF'deki pair-specific konfigürasyonları uygula"""
    # PDF'deki pair-specific öneriler
    pair_configs = {
        'EUR_USD': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'ff_dim': 512,
            'seq_len': 128
        },
        'USD_JPY': {
            'd_model': 192,
            'nhead': 8,
            'num_layers': 3,
            'ff_dim': 384,
            'seq_len': 96
        },
        'GBP_USD': {
            'd_model': 224,
            'nhead': 8,
            'num_layers': 4,
            'ff_dim': 448,
            'seq_len': 112
        },
        'XAU_USD': {
            'd_model': 192,
            'nhead': 8,
            'num_layers': 3,
            'ff_dim': 384,
            'seq_len': 96
        },
        'AUD_USD': {
            'd_model': 192,
            'nhead': 8,
            'num_layers': 3,
            'ff_dim': 384,
            'seq_len': 96
        }
    }
    
    # Uygula sadece belirtilen pair'ler için
    for pair in pairs:
        if pair in pair_configs:
            print(f"   🔧 Applying PDF config for {pair}")
            pair_cfg = pair_configs[pair]
            cfg['transformer']['d_model'] = pair_cfg['d_model']
            cfg['transformer']['nhead'] = pair_cfg['nhead']
            cfg['transformer']['num_layers'] = pair_cfg['num_layers']
            cfg['transformer']['ff_dim'] = pair_cfg['ff_dim']
            cfg['data']['sequence_length'] = pair_cfg['seq_len']
    
    return cfg

def run_backtest_pipeline(args):
    """Run the walk-forward backtesting pipeline."""
    print("⚠️ Backtesting pipeline not yet implemented for Enhanced Transformer")

def run_evaluation_pipeline(args):
    """Run model evaluation pipeline."""
    print("⚠️ Evaluation pipeline not yet implemented")

def run_temporal_pipeline(args):
    """Run temporal training and signal generation pipeline."""
    print(f"\n🕒 TEMPORAL TRAINING & SIGNAL GENERATION")
    print(f"{'='*60}")
    
    # Setup
    device = setup_device()
    timestamp = create_output_directories()
    
    print(f"📅 Temporal Configuration:")
    print(f"   🎯 Training period: {args.train_start} → {args.train_end}")
    print(f"   📡 Signal period: {args.signal_start} → {args.signal_end}")
    print(f"   🤖 Model: Enhanced Transformer")
    print(f"   📊 Pairs: {args.pairs}")
    print(f"   ⏰ Epochs: {args.temporal_epochs}")
    print(f"   🎪 Confidence threshold: {args.confidence_threshold}")
    
    # Load configuration
    if os.path.exists(args.config):
        cfg = config.load(args.config)
        print(f"📁 Loaded config from {args.config}")
    else:
        print(f"⚠️ Config file not found, using defaults")
        cfg = config.get_default_config()
    
    # Override with temporal settings
    cfg = override_config_with_temporal_args(cfg, args)
    
    # Apply PDF optimizations
    cfg = apply_pdf_optimizations(args, cfg)
    
    # Import and run temporal module
    try:
        from .temporal.temporal_trainer import TemporalTrainer
        
        # Create temporal trainer with custom periods
        temporal_trainer = TemporalTrainer(cfg, cache_enabled=args.cache)
        temporal_trainer.train_start = args.train_start
        temporal_trainer.train_end = args.train_end
        temporal_trainer.signal_start = args.signal_start
        temporal_trainer.signal_end = args.signal_end
        
        # Run pipeline
        results = temporal_trainer.run_temporal_pipeline(args.pairs)
        
        # Display results
        print_temporal_results(results)
        
        # Save detailed report
        save_temporal_report(results, timestamp)
        
        print(f"\n🎉 Temporal pipeline completed successfully!")
        print(f"📁 Results saved in temporal_results/ and temporal_signals/")
        
        return results
        
    except ImportError as e:
        print(f"❌ Temporal module import error: {e}")
        print("💡 Make sure temporal_trainer.py is in src/temporal/ directory")
    except Exception as e:
        print(f"❌ Temporal pipeline error: {e}")
        return None

def override_config_with_temporal_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config with temporal-specific arguments."""
    
    # Model settings for temporal training
    cfg['model']['type'] = 'enhanced_transformer'
    cfg['model']['target_mode'] = args.target_mode
    cfg['model']['use_focal_loss'] = args.use_focal_loss
    cfg['model']['dropout_rate'] = args.dropout_rate
    
    # Training settings
    cfg['training']['epochs'] = args.temporal_epochs
    cfg['training']['batch_size'] = args.batch_size
    cfg['training']['learning_rate'] = args.learning_rate or 2e-4  # Transformer default
    
    # Data settings
    cfg['data']['granularity'] = args.granularity
    cfg['data']['lookback_candles'] = args.lookback_candles
    cfg['data']['sequence_length'] = args.seq_len or 64
    
    # Enhanced Transformer settings
    cfg['transformer'] = {
        'd_model': getattr(args, 'd_model', 256),
        'nhead': getattr(args, 'nhead', 8),
        'num_layers': getattr(args, 'num_layers', 4),
        'ff_dim': getattr(args, 'ff_dim', 512),
        'max_seq_len': cfg['data']['sequence_length']
    }
    
    # Temporal-specific settings
    cfg['temporal'] = {
        'train_start': args.train_start,
        'train_end': args.train_end,
        'signal_start': args.signal_start,
        'signal_end': args.signal_end,
        'confidence_threshold': args.confidence_threshold
    }
    
    return cfg

def print_temporal_results(results: Dict[str, Any]):
    """Print detailed temporal results."""
    print(f"\n📊 TEMPORAL RESULTS SUMMARY")
    print(f"{'='*50}")
    
    summary = results.get('pipeline_summary', {})
    pipeline_info = summary.get('pipeline_info', {})
    training_summary = summary.get('training_summary', {})
    signal_summary = summary.get('signal_summary', {})
    
    # Pipeline info
    print(f"🕒 Training Period: {pipeline_info.get('train_period', 'N/A')}")
    print(f"📡 Signal Period: {pipeline_info.get('signal_period', 'N/A')}")
    print(f"✅ Success Rate: {pipeline_info.get('successful_training', 0)}/{pipeline_info.get('total_pairs_processed', 0)} pairs")
    
    # Training results
    if training_summary:
        print(f"\n🎯 TRAINING RESULTS:")
        for pair, stats in training_summary.items():
            val_acc = stats.get('best_val_acc', 0)
            train_acc = stats.get('final_train_acc', 0)
            gap = stats.get('overfitting_gap', 0)
            epochs = stats.get('epochs_trained', 0)
            
            print(f"   📈 {pair}: Val={val_acc:.1f}%, Train={train_acc:.1f}%, Gap={gap:.1f}pp, Epochs={epochs}")
    
    # Signal results
    if signal_summary:
        print(f"\n📡 SIGNAL RESULTS:")
        total_signals = 0
        total_long = 0
        total_short = 0
        
        for pair, stats in signal_summary.items():
            signals = stats.get('total_signals', 0)
            long_sig = stats.get('long_signals', 0)
            short_sig = stats.get('short_signals', 0)
            confidence = stats.get('avg_confidence', 0)
            
            total_signals += signals
            total_long += long_sig
            total_short += short_sig
            
            print(f"   📊 {pair}: {signals} signals (L:{long_sig}, S:{short_sig}) Conf:{confidence:.3f}")
        
        print(f"\n📈 OVERALL SIGNAL STATS:")
        print(f"   Total Signals: {total_signals}")
        print(f"   Long Signals: {total_long} ({100*total_long/max(total_signals,1):.1f}%)")
        print(f"   Short Signals: {total_short} ({100*total_short/max(total_signals,1):.1f}%)")

def save_temporal_report(results: Dict[str, Any], timestamp: str):
    """Save detailed temporal report."""
    try:
        import json
        from pathlib import Path
        
        # Create reports directory
        reports_dir = Path("temporal_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON report
        report_path = reports_dir / f"temporal_report_{timestamp}.json"
        
        # Prepare serializable data
        report_data = {
            'pipeline_summary': results.get('pipeline_summary', {}),
            'timestamp': timestamp,
            'pairs_processed': list(results.get('training_results', {}).keys()),
            'signal_files': []
        }
        
        # Collect signal file paths
        signal_results = results.get('signal_results', {})
        for pair, signals in signal_results.items():
            if signals and 'csv_path' in signals:
                report_data['signal_files'].append({
                    'pair': pair,
                    'csv_path': signals['csv_path'],
                    'signal_count': signals.get('signals_count', 0)
                })
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📋 Detailed report saved: {report_path}")
        
        # Create summary CSV
        summary_csv = reports_dir / f"temporal_summary_{timestamp}.csv"
        create_temporal_summary_csv(results, summary_csv)
        
    except Exception as e:
        print(f"⚠️ Failed to save temporal report: {e}")

def create_temporal_summary_csv(results: Dict[str, Any], csv_path: Path):
    """Create summary CSV for temporal results."""
    try:
        summary_data = []
        training_summary = results.get('pipeline_summary', {}).get('training_summary', {})
        signal_summary = results.get('pipeline_summary', {}).get('signal_summary', {})
        
        for pair in training_summary.keys():
            train_stats = training_summary.get(pair, {})
            signal_stats = signal_summary.get(pair, {})
            
            row = {
                'pair': pair,
                'best_val_acc': train_stats.get('best_val_acc', 0),
                'final_train_acc': train_stats.get('final_train_acc', 0),
                'overfitting_gap': train_stats.get('overfitting_gap', 0),
                'epochs_trained': train_stats.get('epochs_trained', 0),
                'total_signals': signal_stats.get('total_signals', 0),
                'long_signals': signal_stats.get('long_signals', 0),
                'short_signals': signal_stats.get('short_signals', 0),
                'avg_confidence': signal_stats.get('avg_confidence', 0),
                'signal_csv': signal_stats.get('signal_csv', '')
            }
            summary_data.append(row)
        
        if summary_data:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
            print(f"📊 Summary CSV saved: {csv_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create summary CSV: {e}")

def override_config_with_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration with command-line arguments."""
    # Data parameters
    cfg['data']['granularity'] = args.granularity
    cfg['data']['lookback_candles'] = args.lookback_candles
    if args.seq_len:
        cfg['data']['sequence_length'] = args.seq_len
    
    # Model parameters
    cfg['model']['target_mode'] = args.target_mode
    cfg['model']['use_focal_loss'] = args.use_focal_loss
    cfg['model']['dropout_rate'] = args.dropout_rate
    
    # Training parameters
    cfg['training']['epochs'] = args.epochs
    cfg['training']['batch_size'] = args.batch_size
    cfg['training']['learning_rate'] = args.learning_rate
    cfg['training']['use_smote'] = args.oversample_smote
    
    # Enhanced Transformer specific configuration
    if getattr(args, 'model', 'lstm') == 'enhanced_transformer':
        cfg['transformer'] = {
            'd_model': getattr(args, 'd_model', 128),
            'nhead': getattr(args, 'nhead', 8),
            'num_layers': getattr(args, 'num_layers', 4),
            'ff_dim': getattr(args, 'ff_dim', 256),
            'max_seq_len': cfg['data'].get('sequence_length', 64)
        }
        
        # Set system model type for factory
        cfg['system'] = cfg.get('system', {})
        cfg['system']['model_type'] = 'enhanced_transformer'
        
        print(f"🔧 Enhanced Transformer config created:")
        print(f"   d_model: {cfg['transformer']['d_model']}")
        print(f"   nhead: {cfg['transformer']['nhead']}")
        print(f"   num_layers: {cfg['transformer']['num_layers']}")
        print(f"   ff_dim: {cfg['transformer']['ff_dim']}")
    
    return cfg

if __name__ == "__main__":
    main()
