# Enhanced Modular Multi-Pair LSTM System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)](https://mlflow.org/)

An advanced, modular LSTM-based forex trading system with enhanced features including **three-class classification**, **dynamic k-pips calculation**, **extensive hyperparameter optimization**, and **comprehensive backtesting**.

## 🚀 Key Features

### Enhanced Data Processing
- **Large Data Support**: Up to 180,000 candles per pair
- **Intelligent Caching**: Disk caching with automatic invalidation
- **Dynamic K-Pips**: Enhanced formula `max(0.2 * ATR, 0.8 * spread)` 
- **Multi-Granularity**: M1, M5, M15, H1, H4, D support

### Advanced Machine Learning
- **Dual Target Modes**: Binary and three-class (Long/Flat/Short) classification
- **Enhanced SMOTE**: Forced oversampling for three-class mode
- **Focal Loss**: Advanced loss function for class imbalance
- **Overfitting Protection**: Automatic dropout adjustment and model restart

### Hyperparameter Optimization
- **Expanded Optuna Search**: 32-128 hidden units, 0.40-0.70 dropout
- **Early Stopping**: Intelligent convergence detection
- **Parameter Caching**: Reuse best parameters across runs
- **Multi-Objective**: Balance accuracy and overfitting

### Production Features
- **MLflow Integration**: Comprehensive experiment tracking
- **Backtesting Engine**: Full trading simulation with metrics
- **Enhanced Visualizations**: Detailed performance analysis
- **Modular Architecture**: Easy to extend and maintain

## 📊 Training Pipeline Schema

```mermaid
graph TD
    A[Data Fetching] --> B{Cache Check}
    B -->|Hit| C[Load Cached Data]
    B -->|Miss| D[Fetch from OANDA API]
    D --> E[Save to Cache]
    C --> F[Data Preprocessing]
    E --> F
    
    F --> G[Feature Engineering]
    G --> H[Dynamic K-Pips Calculation]
    H --> I{Target Mode}
    
    I -->|Binary| J[Binary Labels]
    I -->|Three-Class| K[Three-Class Labels]
    
    J --> L{Single Class?}
    K --> L
    L -->|Yes| M[Auto K-Pips Adjustment]
    L -->|No| N[Apply SMOTE if needed]
    M --> N
    
    N --> O{Optuna Enabled?}
    O -->|Yes| P[Hyperparameter Optimization]
    O -->|No| Q[Default Parameters]
    P --> Q
    
    Q --> R[LSTM Training]
    R --> S{Overfitting Detected?}
    S -->|Yes| T[Increase Dropout & Restart]
    S -->|No| U[Continue Training]
    T --> R
    U --> V[Model Evaluation]
    
    V --> W[MLflow Logging]
    W --> X[Save Model & Results]
    X --> Y[Generate Reports]
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- OANDA API account

### Setup
```bash
# Clone repository
git clone <repository-url>
cd lstm_project

# Install dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[dev]"

# Setup project structure
python setup_modular_project.py
```

## ⚙️ Configuration

### Basic Configuration (`configs/default.yaml`)
```yaml
data:
  granularity: "M15"
  lookback_candles: 180000  # Up to 180k candles
  sequence_length: 64

model:
  target_mode: "binary"     # or "three_class"
  use_focal_loss: true
  dropout_rate: 0.45
  hidden_size: 96
  num_layers: 2

training:
  epochs: 120
  batch_size: 128
  learning_rate: 0.0008

api:
  api_key: "your_oanda_api_key"
  account_id: "your_account_id"
  environment: "practice"
```

### Big Data Configuration (`configs/big_run.yaml`)
For high-volume training with enhanced features:
```yaml
data:
  granularity: "M5"         # Higher frequency
  lookback_candles: 180000  # Maximum data
  sequence_length: 128      # Longer sequences

training:
  epochs: 200
  batch_size: 256           # Larger batches
  patience: 15             # More patience

optimization:
  optuna_enabled: true
  n_trials: 50             # Extensive search
```

## 🚀 Usage

### Training Models

#### Basic Training
```bash
# Train with default settings
python -m src.cli train

# Train specific pairs with caching
python -m src.cli train \
  --pairs EUR_USD GBP_USD \
  --cache \
  --granularity M15 \
  --lookback_candles 50000
```

#### Enhanced Training with Three-Class Mode
```bash
# Three-class mode with extensive optimization
python -m src.cli train \
  --target_mode three_class \
  --granularity M5 \
  --lookback_candles 180000 \
  --optuna_runs 40 \
  --epochs 150 \
  --cache \
  --mlflow_uri "http://localhost:5000"
```

#### High-Performance Training
```bash
# Big data training with all enhancements
python -m src.cli train \
  --config configs/big_run.yaml \
  --granularity M1 \
  --lookback_candles 180000 \
  --target_mode three_class \
  --optuna_runs 100 \
  --cache \
  --dropout_upper 0.80 \
  --hidden_size 128
```

### Backtesting

#### Basic Backtest
```bash
# Backtest a trained model
python -m src.cli backtest \
  --model_path models/EUR_USD_model_20250630_120000.pth \
  --start_date 2024-01-01 \
  --end_date 2024-12-31
```

#### Advanced Backtest with Custom Data
```bash
# Backtest with external data
python -m src.cli backtest \
  --model_path models/EUR_USD_model_20250630_120000.pth \
  --test_data data/eur_usd_test.csv \
  --output_dir custom_reports \
  --mlflow_uri "http://localhost:5000"
```

## 📈 MLflow Integration

### Start MLflow Server
```bash
# Local MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Access UI at http://localhost:5000
```

### Experiment Tracking
```bash
# Train with MLflow logging
python -m src.cli train \
  --mlflow_uri "http://localhost:5000" \
  --experiment_name "Production_Run_2024" \
  --optuna_runs 50
```

## 🔧 Advanced Features

### Dynamic K-Pips Calculation
The enhanced k-pips formula automatically adapts to market conditions:
```python
# Enhanced formula
k_pips = max(0.2 * ATR(horizon//2), 0.8 * avg_spread)

# Auto-adjustment for single-class problems
k_pips = auto_adjust_k_pips(data, horizon, initial_k_pips, max_iterations=3)
```

### Overfitting Protection
Automatic detection and mitigation:
- **Gap Detection**: Monitor train-val accuracy difference
- **Dropout Adjustment**: Automatic increase when overfitting detected
- **Model Restart**: Complete restart with higher regularization
- **Early Stopping**: Intelligent convergence detection

### Three-Class Mode
Advanced classification for better market understanding:
- **Long (0)**: Strong upward movement
- **Flat (1)**: Sideways/neutral movement  
- **Short (2)**: Strong downward movement
- **Forced SMOTE**: Automatic oversampling for balanced classes

## 📊 Performance Monitoring

### Key Metrics
- **Validation Accuracy**: Primary performance indicator
- **Overfitting Gap**: Train-validation accuracy difference
- **Minority Ratio**: Class imbalance measure
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline

### Visualization
```python
# Generate comprehensive plots
python -m src.cli train --save_plots

# Custom visualization
from src.utils.viz import plot_training_results
plot_training_results(histories, results, "custom_plot.png")
```

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Quick smoke test
pytest -q tests/test_smoke.py

# Test specific module
pytest tests/test_data_fetcher.py -v
```

## 📝 API Reference

### Core Classes

#### `MultiPairOANDAFetcher`
Enhanced data fetcher with caching support.
```python
fetcher = MultiPairOANDAFetcher(
    api_key="your_key",
    account_id="your_account", 
    cache_enabled=True
)
data = fetcher.fetch_all_pairs(lookback_candles=180000)
```

#### `PairSpecificPreprocessor`
Advanced preprocessing with three-class support.
```python
preprocessor = PairSpecificPreprocessor(
    pair_name="EUR_USD",
    target_mode="three_class",
    use_smote=True
)
features, target = preprocessor.prepare_pair_data(data)
```

#### `MultiPairTrainer`
Enhanced trainer with overfitting protection.
```python
trainer = MultiPairTrainer(
    device=device,
    target_mode="three_class",
    use_focal_loss=True
)
model, history = trainer.train_pair_model(pair_name, X, y)
```

#### `BacktestRunner`
Comprehensive backtesting engine.
```python
runner = BacktestRunner(
    model_path="model.pth",
    initial_balance=10000
)
results = runner.execute_backtest(data, signals)
```

## 🔍 Troubleshooting

### Common Issues

#### Cache Problems
```bash
# Clear all cache
python -c "from src.data.fetcher import MultiPairOANDAFetcher; f = MultiPairOANDAFetcher('','', cache_enabled=True); f.clear_cache()"

# Check cache status
python -c "from src.data.fetcher import MultiPairOANDAFetcher; f = MultiPairOANDAFetcher('','', cache_enabled=True); print(f.get_cache_info())"
```

#### Single Class Problems
```bash
# Use auto k-pips adjustment
python -m src.cli train --dynamic_k_aggressive

# Force three-class mode for problematic pairs
python -m src.cli train --target_mode three_class
```

#### Memory Issues
```bash
# Reduce batch size
python -m src.cli train --batch_size 64

# Use gradient accumulation
python -m src.cli train --batch_size 32 --lookback_candles 50000
```

## 🚧 Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement with proper docstrings (Google style)
3. Add tests: `pytest tests/test_new_feature.py`
4. Update documentation
5. Submit PR

### Code Style
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourrepo/issues)
- **Documentation**: [Wiki](https://github.com/yourrepo/wiki)
- **Discord**: [Community Server](https://discord.gg/yourserver)

## 🎯 Roadmap

- [ ] **Q2 2024**: Real-time trading integration
- [ ] **Q3 2024**: Multi-asset support (crypto, stocks)
- [ ] **Q4 2024**: Ensemble models and meta-learning
- [ ] **Q1 2025**: Cloud deployment and auto-scaling

---

**Example Working Command:**
```bash
python -m src.cli train \
  --granularity M5 \
  --lookback_candles 180000 \
  --target_mode three_class \
  --optuna_runs 40 \
  --epochs 30 \
  --cache \
  --mlflow_uri "http://localhost:5000"
```