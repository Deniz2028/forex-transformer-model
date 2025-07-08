# src/cli.py - ENHANCED VERSION with ENSEMBLE, TUNE, INFERENCE, TEMPORAL and PDF HYBRID support
"""Enhanced command-line interface with ENSEMBLE, TUNE, INFERENCE, TEMPORAL and PDF HYBRID support."""

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
import yaml

# Import modular components
from . import config
from .data.fetcher import create_fetcher
from .data.preprocess import create_preprocessor
from .training.trainer import create_trainer
from .training.optuna_utils import create_optimizer
from .utils.viz import plot_training_results, print_summary_report

# Enhanced Transformer and PDF Hybrid factory import
from .models.factory import create_model, SUPPORTED_MODELS, test_pdf_hybrid_model_creation
from .models.pdf_hybrid import test_pdf_hybrid_model

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

# Check PDF Hybrid availability
try:
    PDF_HYBRID_AVAILABLE = 'pdf_hybrid' in SUPPORTED_MODELS
except NameError:
    PDF_HYBRID_AVAILABLE = False

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with Enhanced Transformer, TUNE, INFERENCE, TEMPORAL, ENSEMBLE and PDF HYBRID support."""
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
    add_pdf_hybrid_args(train_parser)  # PDF Hybrid support
    
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
    
    # ===== TEST COMMAND - NEW! =====
    test_parser = subparsers.add_parser('test', help='Test model functionality')
    test_parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS, help='Model to test')
    test_parser.add_argument('--test-pdf-hybrid', action='store_true', help='Test PDF Hybrid model specifically')
    
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
    
    # Model type selection - PDF HYBRID ADDED
    model_choices = SUPPORTED_MODELS if 'SUPPORTED_MODELS' in globals() else [
        'lstm', 'transformer', 'enhanced_transformer', 'hybrid_lstm_transformer', 'pdf_hybrid'
    ]
    parser.add_argument(
        '--model', 
        type=str, 
        default='lstm',
        choices=model_choices,
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

def add_pdf_hybrid_args(parser):
    """Add PDF Hybrid specific arguments to CLI"""
    pdf_group = parser.add_argument_group('PDF Hybrid Parameters',
                                         description='Parameters specific to the PDF Hybrid LSTM-Transformer model')
    
    pdf_group.add_argument(
        '--pdf-config',
        type=str,
        default='configs/pdf_hybrid.yaml',
        help='PDF Hybrid model configuration file'
    )
    
    pdf_group.add_argument(
        '--pdf-target-mode',
        type=str,
        choices=['binary', 'three_class'],
        default='binary',
        help='PDF Hybrid target mode: binary or three_class'
    )
    
    pdf_group.add_argument(
        '--pdf-d-model',
        type=int,
        default=512,
        help='PDF Hybrid d_model parameter (default: 512)'
    )
    
    pdf_group.add_argument(
        '--pdf-nhead',
        type=int,
        default=8,
        help='PDF Hybrid attention heads (default: 8)'
    )
    
    pdf_group.add_argument(
        '--pdf-layers',
        type=int,
        default=4,
        help='PDF Hybrid transformer layers (default: 4)'
    )

def add_training_args(parser):
    """Add training-specific arguments."""
    parser.add_argument(
        '--model_type',
        type=str,
        default='lstm',
        choices=['lstm', 'transformer', 'enhanced_transformer', 'hybrid_lstm_transformer', 'pdf_hybrid'],
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

    # === FOCAL LOSS PARAMETERS ===
    focal_group = parser.add_argument_group('Focal Loss Parameters')
    
    focal_group.add_argument(
        '--focal_alpha',
        type=float,
        default=0.25,
        help='Focal loss alpha parameter (class balancing)'
    )
    
    focal_group.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter (focusing parameter)'
    )
    
    focal_group.add_argument(
        '--loss_type',
        type=str,
        default='focal',
        choices=['focal', 'dynamic_focal', 'bce', 'weighted_bce', 'crossentropy', 'forex_focal'],
        help='Loss function type'
    )
    
    focal_group.add_argument(
        '--use_dynamic_focal',
        action='store_true',
        default=False,
        help='Use volatility-adaptive focal loss'
    )
    
    # === ONECYCLR SCHEDULER PARAMETERS ===
    scheduler_group = parser.add_argument_group('OneCycleLR Scheduler Parameters')
    
    scheduler_group.add_argument(
        '--use_onecycle',
        action='store_true',
        default=True,
        help='Use OneCycleLR scheduler'
    )
    
    scheduler_group.add_argument(
        '--max_lr',
        type=float,
        default=0.01,
        help='OneCycleLR maximum learning rate'
    )
    
    scheduler_group.add_argument(
        '--pct_start',
        type=float,
        default=0.25,
        help='Percentage of cycle for learning rate increase'
    )
    
    scheduler_group.add_argument(
        '--div_factor',
        type=float,
        default=10.0,
        help='Initial learning rate division factor (initial_lr = max_lr/div_factor)'
    )
    
    scheduler_group.add_argument(
        '--final_div_factor',
        type=float,
        default=1000.0,
        help='Final learning rate division factor (final_lr = initial_lr/final_div_factor)'
    )
    
    scheduler_group.add_argument(
        '--scheduler_type',
        type=str,
        default='OneCycleLR',
        choices=['OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealing', 'StepLR'],
        help='Learning rate scheduler type'
    )
    
    # === ADVANCED OPTIMIZATION PARAMETERS ===
    optimization_group = parser.add_argument_group('Advanced Optimization Parameters')
    
    optimization_group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps for larger effective batch size'
    )
    
    optimization_group.add_argument(
        '--use_mixed_precision',
        action='store_true',
        default=False,
        help='Use mixed precision training (FP16)'
    )
    
    optimization_group.add_argument(
        '--gradient_clip_norm',
        type=float,
        default=1.0,
        help='Gradient clipping norm value'
    )
    
    optimization_group.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer'
    )
    
    optimization_group.add_argument(
        '--optimizer_type',
        type=str,
        default='adam',
        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
        help='Optimizer type'
    )
    
    # === FOREX-SPECIFIC PARAMETERS ===
    forex_group = parser.add_argument_group('Forex-Specific Parameters')
    
    forex_group.add_argument(
        '--market_regime_adaptive',
        action='store_true',
        default=False,
        help='Use market regime adaptive loss functions'
    )
    
    forex_group.add_argument(
        '--volatility_window',
        type=int,
        default=20,
        help='Window size for volatility calculation'
    )
    
    forex_group.add_argument(
        '--class_balancing_strategy',
        type=str,
        default='auto',
        choices=['auto', 'smote', 'weighted', 'focal', 'none'],
        help='Class balancing strategy'
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
                               choices=['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer', 'pdf_hybrid'],
                               default=['lstm', 'enhanced_transformer', 'hybrid_lstm_transformer', 'pdf_hybrid'],
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
    
    #et_group.add_argument(
    #    '--gradient_accumulation_steps',
    #    type=int,
    #    default=4,
    #    help='Gradient accumulation steps for larger effective batch size'
    #)
    
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
    
    elif args.model == 'pdf_hybrid':
        # Set PDF Hybrid specific defaults
        if hasattr(args, 'learning_rate') and args.learning_rate is None:
            args.learning_rate = 2e-4  # Default for PDF Hybrid
            print(f"🔧 Learning rate set for PDF Hybrid: {args.learning_rate}")
        
        if hasattr(args, 'batch_size') and args.batch_size == 32:  # Default batch size
            args.batch_size = 24  # Smaller batch for PDF Hybrid
            print(f"🔧 Batch size adjusted for PDF Hybrid: {args.batch_size}")
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
    print("🚀 MODULAR MULTI-PAIR LSTM/ENHANCED TRANSFORMER SYSTEM WITH ENSEMBLE AND PDF HYBRID")
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
        
    elif args.command == 'test':
        return run_test_command(args)

    else:
        print("❌ Invalid command. Use --help for available options.")
        return 1

def run_test_command(args):
    """Handle test command functionality."""
    print(f"\n🧪 MODEL TESTING PIPELINE")
    print(f"{'='*60}")
    
    if args.test_pdf_hybrid or args.model == 'pdf_hybrid':
        if not PDF_HYBRID_AVAILABLE:
            print("❌ PDF Hybrid model not available!")
            print("💡 Ensure src/models/pdf_hybrid.py exists")
            return 1
        
        print("🧪 Testing PDF Hybrid model functionality...")
        
        # Factory test
        factory_test = test_pdf_hybrid_model_creation()
        if not factory_test:
            print("❌ PDF Hybrid factory test failed!")
            return 1
        
        # Model test
        model_test = test_pdf_hybrid_model()
        if not model_test:
            print("❌ PDF Hybrid model test failed!")
            return 1
        
        print("✅ PDF Hybrid model tests passed successfully!")
        return 0
    
    else:
        print("⚠️ No specific model tests requested")
        print("💡 Use --test-pdf-hybrid to test PDF Hybrid model")
        return 0

def train_ensemble_command(args):
    """Train ensemble of models - FIXED DataFrame conversion"""
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
    
    # Override config for PDF Hybrid if needed
    if 'pdf_hybrid' in args.ensemble_models:
        try:
            cfg = override_config_for_pdf_hybrid(cfg, args)
        except AttributeError as e:
            print(f"⚠️ PDF config override skipped: {e}")
    
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
        return {}
    
    print(f"✅ {len(pair_data)} pairs fetched successfully")
    
    # Process each pair
    ensemble_results = {}
    
    for pair_name, data in pair_data.items():
        print(f"\n{'='*50}")
        print(f"🎯 Processing {pair_name}...")
        print(f"{'='*50}")
        
        # Preprocess data
        preprocessor = create_preprocessor(
            pair_name=pair_name,
            config=cfg,
            target_mode=args.target_mode
        )

        # SMOTE ayarını manuel olarak yap
        if getattr(args, 'use_smote', False):
            preprocessor.use_smote = True
        elif args.target_mode == 'three_class':
            preprocessor.use_smote = True  # Three-class için otomatik SMOTE
        
        features, target = preprocessor.prepare_pair_data(data)
        
        # 🔧 CRITICAL FIX: DataFrame → NumPy conversion
        print(f"   📊 Raw data types: features={type(features)}, target={type(target)}")
        
        # Convert DataFrame to NumPy if needed
        if hasattr(features, 'values'):  # It's a DataFrame
            print(f"   🔄 Converting features DataFrame to NumPy...")
            features = features.values
        if hasattr(target, 'values'):    # It's a DataFrame
            print(f"   🔄 Converting target DataFrame to NumPy...")
            target = target.values
        
        # Ensure they are numpy arrays
        features = np.array(features)
        target = np.array(target)
        
        print(f"   📊 NumPy shapes: features={features.shape}, target={target.shape}")
        
        # Convert to tensors - NOW SAFE
        X = torch.FloatTensor(features)
        
        # Smart target conversion
        if target.ndim > 1:
            if target.shape[1] > 1:  # One-hot encoded
                y = torch.LongTensor(np.argmax(target, axis=1))
                print(f"   🎯 One-hot target converted to class indices")
            else:  # Single column
                y = torch.LongTensor(target.flatten())
                print(f"   🎯 Single column target flattened")
        else:  # Already 1D
            y = torch.LongTensor(target)
            print(f"   🎯 1D target used directly")
        
        print(f"   📊 Final tensor shapes: X={X.shape}, y={y.shape}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"   📊 Data split: Train={len(X_train)}, Val={len(X_val)}")
        
        # Initialize ensemble trainer
        trainer = EnsembleTrainer(device=device)
        
        # Generate diverse model configurations
        model_configs = []
        for model_type in args.ensemble_models:
            base_config = config.get_model_config(model_type, args.target_mode)
            base_config['model_type'] = model_type  # 🔧 EKLENEN SATIR
            if model_type == 'pdf_hybrid':
                base_config = override_config_for_pdf_hybrid(base_config, args)
            model_configs.append(base_config)
        # Train ensemble models
        print(f"🏋️ Training ensemble models for {pair_name}...")
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
            ensemble_pred = ensemble.predict(X_val.to(device))
            individual_preds = ensemble.get_individual_predictions(X_val.to(device))

            # Calculate metrics
            if ensemble_pred.dim() > 1:
                ensemble_classes = torch.argmax(ensemble_pred, dim=1)
            else:
                ensemble_classes = (ensemble_pred > 0.5).long()

            # Align y_val with device
            y_val = y_val.to(device)
            ensemble_classes = ensemble_classes.to(device)

            ensemble_accuracy = (ensemble_classes == y_val).float().mean().item()

            # Individual accuracies
            individual_accuracies = {}
            y_val_aligned = y_val.to(device)  # Döngüden önce cihaz uyumlu hale getir

            for name, pred in individual_preds.items():
                if pred.dim() > 1:
                    pred_classes = torch.argmax(pred, dim=1)
                else:
                    pred_classes = (pred > 0.5).long()

                pred_classes = pred_classes.to(device)

                acc = (pred_classes == y_val_aligned).float().mean().item()
                individual_accuracies[name] = acc
            
            best_individual = max(individual_accuracies.values())
            improvement = ensemble_accuracy - best_individual
            
            print(f"   ✅ Ensemble Results for {pair_name}:")
            print(f"      Ensemble Accuracy: {ensemble_accuracy:.4f}")
            print(f"      Best Individual: {best_individual:.4f}")
            print(f"      Improvement: +{improvement:.4f} ({improvement/best_individual*100:.2f}%)")
        
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
        
        ensemble_results[pair_name] = {
            'ensemble': ensemble,
            'accuracy': ensemble_accuracy,
            'improvement': improvement,
            'model_count': len(models)
        }
    
    # Overall summary
    print(f"\n🎉 Ensemble Training Complete!")
    print(f"   Pairs processed: {len(ensemble_results)}")
    
    if ensemble_results:
        total_accuracy = np.mean([r['accuracy'] for r in ensemble_results.values()])
        total_improvement = np.mean([r['improvement'] for r in ensemble_results.values()])
        
        print(f"   Average Ensemble Accuracy: {total_accuracy:.4f}")
        print(f"   Average Improvement: +{total_improvement:.4f}")
    
    return ensemble_results

def override_config_for_pdf_hybrid(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """Override config for PDF Hybrid model specific parameters."""
    
    # 🔧 FIX: Check attributes with getattr to avoid AttributeError
    if getattr(args, 'pdf_target_mode', None):
        cfg['model']['target_mode'] = args.pdf_target_mode
    
    if getattr(args, 'pdf_dropout', None):
        cfg['model']['dropout_rate'] = args.pdf_dropout
    
    if getattr(args, 'pdf_hidden_size', None):
        cfg['model']['hidden_size'] = args.pdf_hidden_size
    
    if getattr(args, 'pdf_num_layers', None):
        cfg['model']['num_layers'] = args.pdf_num_layers
    
    if getattr(args, 'pdf_learning_rate', None):
        cfg['training']['learning_rate'] = args.pdf_learning_rate
    
    if getattr(args, 'pdf_batch_size', None):
        cfg['training']['batch_size'] = args.pdf_batch_size
    
    if getattr(args, 'pdf_scheduler', None):
        cfg['training']['scheduler'] = args.pdf_scheduler
    
    # PDF Hybrid specific transformer settings
    if getattr(args, 'pdf_d_model', None):
        if 'transformer' not in cfg:
            cfg['transformer'] = {}
        cfg['transformer']['d_model'] = args.pdf_d_model
    
    if getattr(args, 'pdf_nhead', None):
        if 'transformer' not in cfg:
            cfg['transformer'] = {}
        cfg['transformer']['nhead'] = args.pdf_nhead
    
    if getattr(args, 'pdf_transformer_layers', None):
        if 'transformer' not in cfg:
            cfg['transformer'] = {}
        cfg['transformer']['num_layers'] = args.pdf_transformer_layers
    
    # Loss function settings
    if getattr(args, 'pdf_loss_type', None):
        if 'loss' not in cfg:
            cfg['loss'] = {}
        cfg['loss']['type'] = args.pdf_loss_type
    
    if getattr(args, 'pdf_focal_alpha', None):
        if 'loss' not in cfg:
            cfg['loss'] = {}
        cfg['loss']['focal_alpha'] = args.pdf_focal_alpha
    
    if getattr(args, 'pdf_focal_gamma', None):
        if 'loss' not in cfg:
            cfg['loss'] = {}
        cfg['loss']['focal_gamma'] = args.pdf_focal_gamma
    
    # Scheduler settings
    if getattr(args, 'pdf_max_lr', None):
        if 'scheduler' not in cfg:
            cfg['scheduler'] = {}
        cfg['scheduler']['max_lr'] = args.pdf_max_lr
    
    if getattr(args, 'pdf_pct_start', None):
        if 'scheduler' not in cfg:
            cfg['scheduler'] = {}
        cfg['scheduler']['pct_start'] = args.pdf_pct_start
    
    return cfg

def display_pdf_hybrid_info(config: Dict[str, Any]):
    """Display PDF Hybrid model information."""
    print("🔥 PDF Hybrid LSTM-Transformer Model:")
    print(f"   📐 Architecture: LSTM + Transformer + MLP")
    print(f"   🎯 Target mode: {config.get('model', {}).get('target_mode', 'binary')}")
    print(f"   📊 d_model: {config.get('transformer', {}).get('d_model', 512)}")
    print(f"   📊 nhead: {config.get('transformer', {}).get('nhead', 8)}")
    print(f"   📊 layers: {config.get('transformer', {}).get('num_layers', 4)}")
    print(f"   🎯 Target accuracy: 70-75% (PDF specification)")
    print(f"   ⚡ OneCycleLR scheduler optimized")

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
    """Enhanced training pipeline with Enhanced Transformer and PDF Hybrid support."""
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
    
    # Apply PDF Hybrid specific config if needed
    if model_type == 'pdf_hybrid':
        cfg = override_config_for_pdf_hybrid(cfg, args)
    
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
    elif model_type == 'pdf_hybrid':
        display_pdf_hybrid_info(cfg)
    
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

# Config Override Fix - src/cli.py dosyasında override_config_with_args fonksiyonunu düzeltin

def override_config_with_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration with command-line arguments - FIXED for config file priority."""
    
    # 🔧 FIX: Only override if CLI args are explicitly provided (not defaults)
    
    # Data parameters - override only if explicitly set
    if hasattr(args, 'granularity') and args.granularity != 'M15':  # Check if not default
        cfg['data']['granularity'] = args.granularity
    if hasattr(args, 'lookback_candles') and args.lookback_candles != 50000:
        cfg['data']['lookback_candles'] = args.lookback_candles
    if hasattr(args, 'seq_len') and args.seq_len is not None:
        cfg['data']['sequence_length'] = args.seq_len
    
    # Model parameters - only override if explicitly set
    if hasattr(args, 'target_mode') and args.target_mode != 'binary':
        cfg['model']['target_mode'] = args.target_mode
    if hasattr(args, 'use_focal_loss') and args.use_focal_loss != True:
        cfg['model']['use_focal_loss'] = args.use_focal_loss
    if hasattr(args, 'dropout_rate') and args.dropout_rate != 0.45:  # Check if not LSTM default
        cfg['model']['dropout_rate'] = args.dropout_rate
    
    # Training parameters - respect config file values
    if hasattr(args, 'epochs') and args.epochs != 30:
        cfg['training']['epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size != 32:
        cfg['training']['batch_size'] = args.batch_size
    
    # 🔧 CRITICAL FIX: Only override learning_rate if explicitly provided
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        # User explicitly set learning rate
        cfg['training']['learning_rate'] = args.learning_rate
        print(f"🔧 Learning rate OVERRIDE: {args.learning_rate}")
    else:
        # Keep config file value
        print(f"🔧 Learning rate from CONFIG: {cfg['training'].get('learning_rate', 'not set')}")
    
    if hasattr(args, 'oversample_smote') and args.oversample_smote != False:
        cfg['training']['use_smote'] = args.oversample_smote
    
    # === FOCAL LOSS CONFIGURATION - RESPECT CONFIG FILE ===
    if 'loss' not in cfg:
        cfg['loss'] = {}
    
    # Only override if CLI args are explicitly different from defaults
    if hasattr(args, 'loss_type') and args.loss_type != 'focal':
        cfg['loss']['type'] = args.loss_type
    
    if hasattr(args, 'focal_alpha') and args.focal_alpha != 0.25:
        cfg['loss']['focal_alpha'] = args.focal_alpha
        print(f"🔧 Focal alpha OVERRIDE: {args.focal_alpha}")
    
    if hasattr(args, 'focal_gamma') and args.focal_gamma != 2.0:
        cfg['loss']['focal_gamma'] = args.focal_gamma
    
    if hasattr(args, 'use_dynamic_focal') and args.use_dynamic_focal != False:
        cfg['loss']['use_dynamic_focal'] = args.use_dynamic_focal
    
    if hasattr(args, 'use_focal_loss') and args.use_focal_loss != True:
        cfg['loss']['use_focal_loss'] = args.use_focal_loss
    
    # === SCHEDULER CONFIGURATION - RESPECT CONFIG FILE ===
    if 'scheduler' not in cfg:
        cfg['scheduler'] = {}
    
    if hasattr(args, 'scheduler_type') and args.scheduler_type != 'OneCycleLR':
        cfg['scheduler']['type'] = args.scheduler_type
    
    if hasattr(args, 'use_onecycle') and args.use_onecycle != True:
        cfg['scheduler']['use_onecycle'] = args.use_onecycle
    
    # 🔧 CRITICAL FIX: Only override max_lr if explicitly provided
    if hasattr(args, 'max_lr') and args.max_lr != 0.01:
        cfg['scheduler']['max_lr'] = args.max_lr
        print(f"🔧 Max LR OVERRIDE: {args.max_lr}")
    else:
        print(f"🔧 Max LR from CONFIG: {cfg['scheduler'].get('max_lr', 'not set')}")
    
    if hasattr(args, 'pct_start') and args.pct_start != 0.25:
        cfg['scheduler']['pct_start'] = args.pct_start
    if hasattr(args, 'div_factor') and args.div_factor != 10.0:
        cfg['scheduler']['div_factor'] = args.div_factor
    if hasattr(args, 'final_div_factor') and args.final_div_factor != 1000.0:
        cfg['scheduler']['final_div_factor'] = args.final_div_factor
    
    # === OPTIMIZATION CONFIGURATION - RESPECT CONFIG FILE ===
    if 'optimization' not in cfg:
        cfg['optimization'] = {}
    
    if hasattr(args, 'optimizer_type') and args.optimizer_type != 'adam':
        cfg['optimization']['optimizer_type'] = args.optimizer_type
    
    # 🔧 CRITICAL FIX: Only override weight_decay if explicitly provided  
    if hasattr(args, 'weight_decay') and args.weight_decay != 1e-4:
        cfg['optimization']['weight_decay'] = args.weight_decay
        print(f"🔧 Weight decay OVERRIDE: {args.weight_decay}")
    
    if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps != 1:
        cfg['optimization']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision != False:
        cfg['optimization']['use_mixed_precision'] = args.use_mixed_precision
    
    # 🔧 CRITICAL FIX: Only override gradient_clip if explicitly provided
    if hasattr(args, 'gradient_clip_norm') and args.gradient_clip_norm != 1.0:
        cfg['optimization']['gradient_clip_norm'] = args.gradient_clip_norm
        print(f"🔧 Gradient clip OVERRIDE: {args.gradient_clip_norm}")

    # === FOREX-SPECIFIC CONFIGURATION ===
    if 'forex' not in cfg:
        cfg['forex'] = {}
        
    if hasattr(args, 'market_regime_adaptive') and args.market_regime_adaptive != False:
        cfg['forex']['market_regime_adaptive'] = args.market_regime_adaptive
    if hasattr(args, 'volatility_window') and args.volatility_window != 20:
        cfg['forex']['volatility_window'] = args.volatility_window
    if hasattr(args, 'class_balancing_strategy') and args.class_balancing_strategy != 'auto':
        cfg['forex']['class_balancing_strategy'] = args.class_balancing_strategy

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

def train_single_model(args):
    """Train single model with enhanced parameters."""
    print(f"\n🎯 ENHANCED TRAINING PIPELINE")
    print(f"{'='*60}")
    
    # Enhanced parameter display
    print(f"🎯 Enhanced Parameters:")
    print(f"   🔥 Loss Type: {getattr(args, 'loss_type', 'focal')}")
    print(f"   📊 Focal Alpha: {getattr(args, 'focal_alpha', 0.25)}")
    print(f"   🎪 Focal Gamma: {getattr(args, 'focal_gamma', 2.0)}")
    print(f"   🔄 Scheduler: {getattr(args, 'scheduler_type', 'OneCycleLR')}")
    print(f"   📈 Max LR: {getattr(args, 'max_lr', 0.01)}")
    print(f"   ⏱️ PCT Start: {getattr(args, 'pct_start', 0.25)}")
    print(f"   🔢 Div Factor: {getattr(args, 'div_factor', 10.0)}")
    print(f"   🏁 Final Div Factor: {getattr(args, 'final_div_factor', 1000.0)}")
    print(f"   💾 Grad Accumulation: {getattr(args, 'gradient_accumulation_steps', 1)}")
    print(f"   🚀 Mixed Precision: {getattr(args, 'use_mixed_precision', False)}")
    
    return run_training_pipeline(args)    

# Updated help display function
def display_enhanced_help():
    """Display enhanced parameter help."""
    print("\n🚀 ENHANCED CLI PARAMETERS")
    print("="*50)
    
    focal_examples = [
        "# Basic focal loss",
        "python -m src.cli train --loss_type focal --focal_alpha 0.25 --focal_gamma 2.0",
        "",
        "# Dynamic focal loss with volatility adaptation", 
        "python -m src.cli train --loss_type dynamic_focal --use_dynamic_focal",
        "",
        "# Forex-specific focal loss",
        "python -m src.cli train --loss_type forex_focal --market_regime_adaptive"
    ]
    
    onecycle_examples = [
        "# OneCycleLR with custom parameters",
        "python -m src.cli train --use_onecycle --max_lr 0.02 --pct_start 0.3 --div_factor 15",
        "",
        "# Different scheduler types",
        "python -m src.cli train --scheduler_type CosineAnnealing",
        "python -m src.cli train --scheduler_type ReduceLROnPlateau"
    ]
    
    advanced_examples = [
        "# Mixed precision with gradient accumulation",
        "python -m src.cli train --use_mixed_precision --gradient_accumulation_steps 4",
        "",
        "# Custom optimizer with weight decay",
        "python -m src.cli train --optimizer_type adamw --weight_decay 1e-3",
        "",
        "# Full advanced optimization",
        "python -m src.cli train --loss_type focal --focal_alpha 0.3 --focal_gamma 2.5 \\",
        "  --use_onecycle --max_lr 0.015 --pct_start 0.25 --div_factor 12 \\",
        "  --gradient_accumulation_steps 2 --use_mixed_precision \\",
        "  --optimizer_type adamw --weight_decay 1e-4"
    ]
    
    print("📊 FOCAL LOSS EXAMPLES:")
    for line in focal_examples:
        print(f"   {line}")
    
    print("\n🔄 ONECYCLR EXAMPLES:")
    for line in onecycle_examples:
        print(f"   {line}")
    
    print("\n🚀 ADVANCED OPTIMIZATION EXAMPLES:")
    for line in advanced_examples:
        print(f"   {line}")
    
    print("\n💡 RECOMMENDED COMBINATIONS:")
    recommendations = [
        "# For Enhanced Transformer:",
        "python -m src.cli train --model enhanced_transformer --loss_type focal \\",
        "  --focal_alpha 0.25 --focal_gamma 2.0 --use_onecycle --max_lr 5e-4",
        "",
        "# For PDF Hybrid:",
        "python -m src.cli train --model pdf_hybrid --pdf-d-model 768 --pdf-nhead 12 \\",
        "  --use_onecycle --max_lr 2e-4 --pct_start 0.3",
        "",
        "# For LSTM with three-class:",
        "python -m src.cli train --model lstm --target_mode three_class \\",
        "  --loss_type focal --focal_alpha 0.3 --focal_gamma 2.5",
        "",
        "# For high volatility pairs:",
        "python -m src.cli train --loss_type forex_focal --market_regime_adaptive \\",
        "  --volatility_window 20 --use_dynamic_focal"
    ]
    
    for line in recommendations:
        print(f"   {line}")

if __name__ == "__main__":
    # Show enhanced help if requested
    import sys
    if '--enhanced-help' in sys.argv:
        display_enhanced_help()
        sys.exit(0)
    
    main()
