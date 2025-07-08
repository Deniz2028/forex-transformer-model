# src/cli_enhanced.py
"""Enhanced CLI functions for Transformer support."""

def add_transformer_arguments(parser):
    """Add Transformer-specific arguments to CLI parser."""
    transformer_group = parser.add_argument_group('Transformer Parameters')
    
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
        help='Number of transformer encoder layers'
    )
    
    transformer_group.add_argument(
        '--dim_feedforward',
        type=int,
        default=256,
        help='Dimension of feedforward network'
    )

def enhance_existing_cli_arguments(parser):
    """Enhance existing CLI arguments for better Transformer support."""
    # Learning rate için Transformer'a uygun range
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (automatically adjusted based on model type)'
    )
    
    # Warmup steps for Transformer
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=1000,
        help='Warmup steps for OneCycleLR scheduler (Transformer only)'
    )

def create_model_config(args):
    """Create complete model configuration from CLI arguments."""
    
    # Base configuration
    config = {
        'model': {
            'type': getattr(args, 'model', 'lstm'),
            'target_mode': args.target_mode,
            'use_focal_loss': args.use_focal_loss,
            'dropout_rate': args.dropout_rate
        },
        'data': {
            'granularity': args.granularity,
            'lookback_candles': args.lookback_candles,
            'sequence_length': getattr(args, 'seq_len', 64)
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': getattr(args, 'learning_rate', 0.001),
            'use_smote': args.oversample_smote
        },
        'api': {
            'api_key': '8d8619f4119fec7e59d73c61b76b480d-d0947fd967a22401c1e48bc1516ad0eb',
            'account_id': '101-004-35700665-002',
            'environment': 'practice'
        }
    }
    
    # Transformer-specific configuration
    if getattr(args, 'model', 'lstm') == 'transformer':
        config['transformer'] = {
            'd_model': getattr(args, 'd_model', 128),
            'nhead': getattr(args, 'nhead', 8),
            'num_layers': getattr(args, 'num_layers', 4),
            'dim_feedforward': getattr(args, 'dim_feedforward', 256),
            'warmup_steps': getattr(args, 'warmup_steps', 1000)
        }
        
        # Transformer için batch size ayarlaması
        if not hasattr(args, 'batch_size') or args.batch_size == 32:  # Default değeri
            config['training']['batch_size'] = 16  # Transformer için daha küçük
    
    return config

def validate_model_arguments(args):
    """Validate and fix model-specific arguments."""
    
    if getattr(args, 'model', 'lstm') == 'transformer':
        d_model = getattr(args, 'd_model', 128)
        nhead = getattr(args, 'nhead', 8)
        
        # d_model ve nhead uyumluluğunu kontrol et
        if d_model % nhead != 0:
            print(f"⚠️ d_model ({d_model}) nhead ({nhead}) ile bölünemiyor!")
            
            # Otomatik düzeltme
            valid_heads = [h for h in [2, 4, 8, 16, 32] if d_model % h == 0 and h <= d_model]
            if valid_heads:
                old_nhead = nhead
                args.nhead = max([h for h in valid_heads if h <= nhead]) or valid_heads[0]
                print(f"🔧 nhead otomatik düzeltildi: {old_nhead} → {args.nhead}")
            else:
                # d_model'i düzelt
                old_d_model = d_model
                args.d_model = d_model + (nhead - (d_model % nhead))
                print(f"🔧 d_model otomatik düzeltildi: {old_d_model} → {args.d_model}")
        
        # Transformer için learning rate ayarlaması
        if not hasattr(args, 'learning_rate') or args.learning_rate == 0.001:  # Default
            args.learning_rate = 0.0005  # Transformer için daha düşük
            print(f"🔧 Transformer için learning rate ayarlandı: {args.learning_rate}")
    
    return args

def print_model_comparison():
    """Print detailed model architecture comparison."""
    print("""
🤖 MODEL ARCHITECTURE COMPARISON

┌─────────────────────┬─────────────────┬─────────────────────┐
│       Feature       │      LSTM       │    Transformer      │
├─────────────────────┼─────────────────┼─────────────────────┤
│ Training Speed      │     ⚡ Fast     │    🐌 Medium        │
│ Memory Usage        │   💾 Low (1GB)  │  💾 High (2-4GB)    │
│ Long Sequences      │   📊 Limited    │   📊 Excellent      │
│ Parallelization     │     ❌ No       │      ✅ Yes         │
│ Attention Mechanism │     ❌ No       │      ✅ Yes         │
│ Interpretability    │   🔍 Medium     │    🔍 High          │
│ Overfitting Risk    │   ⚠️ Medium     │    ⚠️ High          │
│ Best For            │ Quick training  │ Complex patterns    │
│ GPU Requirements    │   🎮 Basic      │   🎮 Modern         │
│ Batch Size          │   📦 64-128     │   📦 16-32          │
└─────────────────────┴─────────────────┴─────────────────────┘

💡 RECOMMENDATIONS:
   • Use LSTM for: Quick experiments, limited GPU memory
   • Use Transformer for: Better accuracy, longer sequences, more data
   
⚙️ TRANSFORMER TIPS:
   • Start with d_model=128, nhead=8, num_layers=4
   • Use smaller batch sizes (16-32)
   • Enable gradient clipping (max_norm=1.0)
   • Monitor GPU memory usage
""")

def get_model_specific_defaults(model_type: str) -> dict:
    """Get model-specific default parameters."""
    
    if model_type.lower() == 'transformer':
        return {
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 80,
            'dropout_rate': 0.1,
            'warmup_steps': 1000
        }
    else:  # LSTM
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 120,
            'dropout_rate': 0.45,
            'warmup_steps': 0
        }

__all__ = [
    'add_transformer_arguments',
    'enhance_existing_cli_arguments', 
    'create_model_config',
    'validate_model_arguments',
    'print_model_comparison',
    'get_model_specific_defaults'
]
