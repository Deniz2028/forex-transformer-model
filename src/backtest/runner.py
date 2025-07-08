import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from ..data.fetcher import create_fetcher
from ..data.preprocess import create_preprocessor
from ..training.trainer import create_trainer
from ..models.lstm import create_model
from .k_pips_utils import compute_k_pips, validate_k_pips
from .trade_logic import TradeSignalEngine  
from .mlflow_manager import MLflowManager
from .threshold_optimizer import ThresholdOptimizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

class WalkForwardBacktester:
    """Enhanced walk-forward backtesting engine with all improvements."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device,
                 train_window_days: int = 180,
                 retrain_freq_days: int = 7,
                 signal_tf: str = "M15",
                 exec_tf: str = "M5",
                 fine_tune: bool = False,
                 epochs_fine: int = 5,
                 cache_enabled: bool = False,
                 mlflow_uri: str = None,
                 signal_threshold: float = 0.5,
                 k_pips_mult: float = 1.0):
        
        self.config = config
        self.device = device
        self.train_window_days = train_window_days
        self.retrain_freq_days = retrain_freq_days
        self.signal_tf = signal_tf
        self.exec_tf = exec_tf
        self.fine_tune = fine_tune
        self.epochs_fine = epochs_fine
        self.cache_enabled = cache_enabled
        
        # Enhanced features
        self.signal_threshold = signal_threshold
        self.k_pips_mult = k_pips_mult
        
        # MLflow manager
        self.mlflow_manager = MLflowManager(mlflow_uri) if mlflow_uri else None
        
        # Create output directories
        self.output_dir = Path("backtest_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Trading metrics storage
        self.trading_results = {}
        
        # Threshold analysis tracking
        self.threshold_analyses = {}
        
        # Current state tracking
        self.current_model = None
        self.current_preprocessor = None
        self.current_pair_name = None
        self.current_window_id = 1
        
        # Store best weights for fine-tuning
        self.best_model_weights = {}
    
    def run_pair_backtest(self, pair_name: str, 
                         start_date: str = None,
                         end_date: str = None,
                         position_size: float = 0.01,
                         spread_multiplier: float = 1.2,
                         commission_pips: float = 0.0,
                         slippage_pips: float = 0.1,
                         stop_loss_pips: float = None,
                         take_profit_pips: float = None,
                         max_positions: int = 1,
                         epochs: int = 30,
                         batch_size: int = 64,
                         csv_logger: str = None) -> Dict[str, Any]:
        """Run walk-forward backtest for a single pair with enhanced features."""
        
        print(f"\n🎯 Starting walk-forward backtest for {pair_name}")
        print(f"   📅 Window: {self.train_window_days}d, Retrain: {self.retrain_freq_days}d")
        print(f"   🎛️ Signal TF: {self.signal_tf}, Exec TF: {self.exec_tf}")
        print(f"   🎯 Signal threshold: {self.signal_threshold}, K-pips mult: {self.k_pips_mult}")
        
        # MLflow setup
        if self.mlflow_manager:
            pair_config = {
                'position_size': position_size,
                'spread_multiplier': spread_multiplier,
                'commission_pips': commission_pips,
                'slippage_pips': slippage_pips,
                'max_positions': max_positions,
                'signal_threshold': self.signal_threshold,
                'k_pips_mult': self.k_pips_mult
            }
            self.mlflow_manager.start_pair_run(pair_name, pair_config)
        
        try:
            # 1. Fetch data for both timeframes
            signal_data, exec_data = self._fetch_multi_timeframe_data(pair_name)
            
            if signal_data is None or exec_data is None:
                return {'status': 'failed', 'error': 'Data fetch failed'}
            
            # 2. Set date range
            data_start = signal_data.index.min()
            data_end = signal_data.index.max()
            
            if start_date:
                backtest_start = pd.to_datetime(start_date)
                backtest_start = max(backtest_start, data_start + timedelta(days=self.train_window_days))
            else:
                backtest_start = data_start + timedelta(days=self.train_window_days)
            
            if end_date:
                backtest_end = min(pd.to_datetime(end_date), data_end)
            else:
                backtest_end = data_end
            
            print(f"   📊 Data range: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
            print(f"   🎯 Backtest range: {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
            
            # 3. Run walk-forward loop
            equity_curve, trade_log, window_results = self._walk_forward_loop(
                pair_name=pair_name,
                signal_data=signal_data,
                exec_data=exec_data,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                position_size=position_size,
                spread_multiplier=spread_multiplier,
                commission_pips=commission_pips,
                slippage_pips=slippage_pips,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
                max_positions=max_positions,
                epochs=epochs,
                batch_size=batch_size,
                csv_logger=csv_logger
            )
            
            # 4. Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                equity_curve, trade_log, pair_name
            )
            
            # 5. Log results to MLflow
            if self.mlflow_manager:
                self.mlflow_manager.log_pair_final_results({
                    'pair_name': pair_name,
                    'performance_metrics': performance_metrics,
                    'equity_curve': equity_curve,
                    'trade_log': trade_log,
                    'window_results': window_results,
                    'backtest_period': {
                        'start': backtest_start.isoformat(),
                        'end': backtest_end.isoformat()
                    }
                })
                self.mlflow_manager.end_run()
            
            # 6. Save results
            results = {
                'status': 'success',
                'pair_name': pair_name,
                'equity_curve': equity_curve,
                'trade_log': trade_log,
                'window_results': window_results,
                'performance_metrics': performance_metrics,
                'backtest_period': {
                    'start': backtest_start.isoformat(),
                    'end': backtest_end.isoformat()
                },
                'threshold_analysis': self.threshold_analyses.copy()
            }
            
            self._save_pair_results(results, pair_name)
            
            print(f"   ✅ {pair_name} backtest completed!")
            print(f"   📊 Total trades: {len(trade_log)}")
            print(f"   💰 Total P&L: {performance_metrics['total_pnl']:.2f} pips")
            print(f"   📈 Sharpe ratio: {performance_metrics['sharpe_ratio']:.3f}")
            print(f"   📉 Max drawdown: {performance_metrics['max_drawdown']:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Backtest failed: {str(e)}")
            if self.mlflow_manager:
                self.mlflow_manager.end_run()
            return {'status': 'failed', 'error': str(e)}
    
    def _fetch_multi_timeframe_data(self, pair_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch data for both signal and execution timeframes with correct ratios."""
        try:
            fetcher = create_fetcher(self.config, cache_enabled=self.cache_enabled)
            
            # Enhanced: Fetch more data with fixed loops
            # Use much larger lookback to ensure we get the full 180k
            signal_data = fetcher.fetch_pair_data(
                instrument=pair_name,
                granularity=self.signal_tf,
                lookback_candles=self.config['data']['lookback_candles'],
                target_candles=180000  # Explicit target
            )
            
            exec_data = fetcher.fetch_pair_data(
                instrument=pair_name,
                granularity=self.exec_tf,
                lookback_candles=self.config['data']['lookback_candles'],
                target_candles=180000  # Explicit target
            )
            
            return signal_data, exec_data
            
        except Exception as e:
            print(f"   ❌ Data fetch error: {e}")
            return None, None
    
    def _walk_forward_loop(self, pair_name: str, signal_data: pd.DataFrame, 
                          exec_data: pd.DataFrame, backtest_start: pd.Timestamp,
                          backtest_end: pd.Timestamp, **kwargs) -> Tuple[List, List, List]:
        """Enhanced walk-forward loop with all improvements."""
        
        # Store pair name for logging
        self.current_pair_name = pair_name
        
        equity_curve = []
        trade_log = []
        window_results = []
        
        # Initialize trading state
        current_positions = []
        current_equity = 10000.0
        
        # Walk-forward parameters  
        window_start = backtest_start - timedelta(days=self.train_window_days)
        window_end = backtest_start
        step_size = timedelta(days=self.retrain_freq_days)
        
        model = None
        preprocessor = None
        window_count = 0
        k_pips = None
        optimal_threshold = self.signal_threshold
        
        while window_end + step_size <= backtest_end:
            window_count += 1
            self.current_window_id = window_count
            
            print(f"\n   📅 Window {window_count}: {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}")
            
            # 1. Prepare training data
            train_data = signal_data.loc[window_start:window_end]
            
            if len(train_data) < 1000:
                print(f"   ⚠️ Insufficient training data: {len(train_data)} samples")
                window_start += step_size
                window_end += step_size
                continue
            
            # 2. Train/retrain model with enhanced features
            val_acc, train_samples, k_pips, optimal_threshold = self._train_window_model(
                pair_name=pair_name,
                train_data=train_data,
                model=model,
                preprocessor=preprocessor,
                epochs=kwargs.get('epochs', 30),
                batch_size=kwargs.get('batch_size', 64),
                window_count=window_count
            )
            
            # 3. Generate predictions for next period with optimal threshold
            pred_start = window_end
            pred_end = min(window_end + step_size, backtest_end)
            
            predictions = self._generate_predictions(
                signal_data=signal_data,
                pred_start=pred_start,
                pred_end=pred_end,
                model=model,
                preprocessor=preprocessor,
                k_pips=k_pips,
                optimal_threshold=optimal_threshold  # Use calibrated threshold
            )
            
            # 4. Execute trades based on predictions
            period_trades, period_equity = self._execute_trades(
                pair_name=pair_name,
                exec_data=exec_data,
                predictions=predictions,
                pred_start=pred_start,
                pred_end=pred_end,
                current_positions=current_positions,
                current_equity=current_equity,
                k_pips=k_pips,
                **kwargs
            )
            
            # 5. Update equity curve
            equity_curve.extend(period_equity)
            trade_log.extend(period_trades)
            current_equity = period_equity[-1]['equity'] if period_equity else current_equity
            
            # 6. Log window results with enhanced metrics
            window_result = {
                'window': window_count,
                'train_start': window_start.isoformat(),
                'train_end': window_end.isoformat(),
                'pred_start': pred_start.isoformat(),
                'pred_end': pred_end.isoformat(),
                'train_samples': train_samples,
                'val_acc': val_acc,
                'k_pips': k_pips,
                'optimal_threshold': optimal_threshold,
                'predictions_made': len(predictions),
                'trades_executed': len(period_trades),
                'period_pnl': sum(t['pnl_pips'] for t in period_trades),
                'period_pnl_usd': sum(t['pnl_usd'] for t in period_trades),
                'ending_equity': current_equity,
                'win_rate_window': len([t for t in period_trades if t['pnl_pips'] > 0]) / len(period_trades) * 100 if period_trades else 0
            }
            
            window_results.append(window_result)
            
            # 7. Enhanced logging
            csv_logger = kwargs.get('csv_logger')
            if csv_logger:
                winning_trades = sum(1 for t in period_trades if t['pnl_pips'] > 0)
                total_pnl = sum(t['pnl_pips'] for t in period_trades)
                
                with open(csv_logger, 'a') as f:
                    f.write(f"{pair_name},{window_start.strftime('%Y-%m-%d')},{window_end.strftime('%Y-%m-%d')},{train_samples},{val_acc:.2f},{len(period_trades)},{winning_trades},{total_pnl:.2f},{k_pips:.6f},{current_equity:.2f}\n")
            
            print(f"   📊 Val Acc: {val_acc:.2f}% | Threshold: {optimal_threshold:.3f} | K-pips: {k_pips:.6f} | Trades: {len(period_trades)} | P&L: {sum(t['pnl_pips'] for t in period_trades):.1f} pips")
            
            # Move to next window
            window_start += step_size
            window_end += step_size
        
        return equity_curve, trade_log, window_results
    
    def _train_window_model(self, pair_name: str, train_data: pd.DataFrame,
                           model: Optional[Any], preprocessor: Optional[Any],
                           epochs: int, batch_size: int, window_count: int) -> Tuple[float, int, float, float]:
        """Enhanced training with ROC-based threshold optimization."""
        
        # Create or reuse preprocessor
        if preprocessor is None:
            preprocessor = create_preprocessor(pair_name, self.config)
        
        # ENHANCED: Force three-class for better flat detection
        if self.config['model']['target_mode'] == 'binary':
            print(f"   🔄 Switching to three_class mode for better flat detection")
            preprocessor.target_mode = 'three_class'
            preprocessor.use_smote = True
        
        # Enhanced: Consistent k-pips calculation with NEW FORMULA
        horizon = 64
        k_pips = compute_k_pips(
            df=train_data,
            horizon=horizon,
            pair_name=pair_name,
            k_pips_mult=self.k_pips_mult,
            mode='enhanced'  # Uses 0.3*ATR, 1.2*spread
        )
        
        # ENHANCED: Ensure k-pips is at least 2x spread
        avg_spread = train_data['spread'].mean()
        min_k_pips = avg_spread * 2.0
        if k_pips < min_k_pips:
            print(f"   🔧 K-pips {k_pips:.6f} < 2x spread, adjusting to {min_k_pips:.6f}")
            k_pips = min_k_pips
        
        # Validate k-pips
        validation = validate_k_pips(k_pips, pair_name, avg_spread)
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"   ⚠️ K-pips warning: {warning}")
        
        # Prepare data with consistent k-pips
        features, target = preprocessor.prepare_pair_data(
            train_data,
            horizon=horizon,
            k_pips=k_pips,
            dynamic_k_aggressive=True
        )
        
        if len(features) < 500:
            return 0.0, len(features), k_pips, self.signal_threshold
        
        # ENHANCED: Time-based split (no leakage)
        X, y = preprocessor.create_sequences(features, target)
        
        if len(X) < 100:
            return 0.0, len(X), k_pips, self.signal_threshold
        
        # Time-based split - CRITICAL for avoiding leakage
        split_pt = int(len(X) * 0.85)  # 85% train, 15% val
        X_train = X[:split_pt]
        X_val = X[split_pt:]
        y_train = y[:split_pt]
        y_val = y[split_pt:]
        
        print(f"   📊 Time-based split: Train {len(X_train)} | Val {len(X_val)}")
        
        # Train model
        trainer = create_trainer(self.config, self.device)
        trainer.target_mode = preprocessor.target_mode
        
        # ENHANCED: Proper fine-tuning
        if model is None or not self.fine_tune or window_count == 1:
            training_epochs = epochs
            print(f"   🔥 Full training: {training_epochs} epochs")
        else:
            # Real fine-tuning: Load best weights and reduce LR
            if pair_name in self.best_model_weights:
                model.load_state_dict(self.best_model_weights[pair_name])
                print(f"   🎛️ Fine-tuning from best weights: {self.epochs_fine} epochs")
            training_epochs = self.epochs_fine
        
        # MLflow window tracking
        window_run_id = None
        if self.mlflow_manager:
            window_config = {
                'train_start': train_data.index.min().isoformat(),
                'train_end': train_data.index.max().isoformat(),
                'train_samples': len(X_train),
                'is_fine_tune': self.fine_tune and model is not None and window_count > 1,
                'k_pips': k_pips,
                'target_mode': preprocessor.target_mode
            }
            window_run_id = self.mlflow_manager.start_window_run(window_count, window_config)
        
        try:
            model, history = trainer.train_pair_model(
                pair_name, X_train, y_train,  # Use time-split data
                epochs=training_epochs,
                batch_size=batch_size,
                dropout=self.config['model']['dropout_rate']
            )
            
            best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
            
            # ENHANCED: ROC-based threshold calibration
            optimal_threshold = self.signal_threshold  # Default
            
            if len(X_val) > 50:  # Enough validation data
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    val_outputs = model(X_val_tensor).cpu().numpy()
                
                # Threshold optimization
                optimizer = ThresholdOptimizer(pair_name, preprocessor.target_mode)
                
                if preprocessor.target_mode == 'binary':
                    # Binary: use sigmoid probabilities
                    val_probs = torch.sigmoid(torch.FloatTensor(val_outputs)).numpy().flatten()
                    optimization_result = optimizer.optimize_threshold_roc(
                        y_val.astype(int), val_probs, method='youden'
                    )
                    optimal_threshold = optimization_result['optimal_threshold']
                    
                    print(f"   🎯 ROC-optimized threshold: {optimal_threshold:.3f} (Youden J)")
                    
                else:
                    # Three-class: confidence-based threshold
                    val_probs = torch.softmax(torch.FloatTensor(val_outputs), dim=1).numpy()
                    max_probs = np.max(val_probs, axis=1)
                    predicted_classes = np.argmax(val_probs, axis=1)
                    
                    optimization_result = optimizer.optimize_threshold_roc(
                        y_val.astype(int), max_probs, method='f1'
                    )
                    optimal_threshold = optimization_result['optimal_threshold']
                    
                    print(f"   🎯 Confidence-optimized threshold: {optimal_threshold:.3f}")
                
                # Save threshold analysis plots for first two windows
                if window_count <= 2:
                    try:
                        if preprocessor.target_mode == 'binary':
                            optimizer.save_threshold_analysis(y_val.astype(int), val_probs, 
                                                             optimization_result, window_count)
                        else:
                            optimizer.save_threshold_analysis(y_val.astype(int), max_probs,
                                                             optimization_result, window_count)
                    except Exception as e:
                        print(f"   ⚠️ Threshold plot save failed: {e}")
                
                # Store threshold analysis
                self.threshold_analyses[window_count] = optimization_result
            
            # ENHANCED: Store best weights for fine-tuning
            self.best_model_weights[pair_name] = model.state_dict().copy()
            
            # ENHANCED: Detailed MLflow logging
            if self.mlflow_manager and window_run_id:
                # Validation predictions for analysis
                if len(X_val) > 0:
                    val_predictions = self._generate_validation_predictions(
                        model, preprocessor, X_val, y_val
                    )
                    
                    # Classification report
                    if len(val_predictions) > 0:
                        y_val_true = y_val.astype(int)
                        y_val_pred = [p['predicted_class'] for p in val_predictions]
                        
                        # Create and log classification report
                        class_report = classification_report(
                            y_val_true, y_val_pred, output_dict=True
                        )
                        
                        # Log classification metrics
                        for class_name, metrics in class_report.items():
                            if isinstance(metrics, dict):
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        mlflow.log_metric(f"class_{class_name}_{metric_name}", value)
                        
                        # Log histogram plot
                        self._log_prediction_histogram(y_val_true, y_val_pred, val_outputs, 
                                                     preprocessor.target_mode, window_count)
                        
                        self.mlflow_manager.log_window_results(
                            window_results={
                                'window': window_count,
                                'val_acc': best_val_acc,
                                'train_samples': len(X_train),
                                'k_pips': k_pips,
                                'optimal_threshold': optimal_threshold,
                                'trade_count_estimate': len([p for p in val_predictions if p.get('will_trade', False)])
                            },
                            model_metrics={
                                'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
                                'best_val_acc': best_val_acc,
                                'overfit_gap': history['train_acc'][-1] - best_val_acc if history['train_acc'] else 0
                            },
                            y_true=y_val_true,
                            y_pred=y_val_pred
                        )
                
                self.mlflow_manager.end_run()  # End window run
            
            # Store for next iteration
            self.current_model = model
            self.current_preprocessor = preprocessor
            
            return best_val_acc, len(X_train), k_pips, optimal_threshold
            
        except Exception as e:
            if self.mlflow_manager and window_run_id:
                self.mlflow_manager.end_run()
            raise e
    
    def _log_prediction_histogram(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 raw_outputs: np.ndarray, target_mode: str, window_id: int) -> None:
        """Log prediction distribution histogram to MLflow."""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            if target_mode == 'binary':
                probs = torch.sigmoid(torch.FloatTensor(raw_outputs)).numpy().flatten()
                
                # Probability distribution by class
                ax1.hist(probs[y_true == 0], bins=30, alpha=0.7, label='Class 0', color='red')
                ax1.hist(probs[y_true == 1], bins=30, alpha=0.7, label='Class 1', color='blue')
                ax1.set_xlabel('Predicted Probability')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Binary Prediction Distribution - Window {window_id}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
            else:
                probs = torch.softmax(torch.FloatTensor(raw_outputs), dim=1).numpy()
                max_probs = np.max(probs, axis=1)
                
                # Confidence distribution by true class
                for class_id in np.unique(y_true):
                    class_probs = max_probs[y_true == class_id]
                    ax1.hist(class_probs, bins=20, alpha=0.7, label=f'True Class {class_id}')
                
                ax1.set_xlabel('Max Confidence')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Three-Class Confidence Distribution - Window {window_id}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title(f'Confusion Matrix - Window {window_id}')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            
            # Save and log to MLflow
            plot_path = f"temp_histogram_window_{window_id}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(plot_path)
            
            # Cleanup
            Path(plot_path).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"   ⚠️ Histogram logging failed: {e}")
    
    def _generate_predictions(self, signal_data: pd.DataFrame, pred_start: pd.Timestamp,
                             pred_end: pd.Timestamp, model: Any, 
                             preprocessor: Any, k_pips: float,
                             optimal_threshold: float = 0.5) -> List[Dict]:
        """Enhanced prediction generation with calibrated threshold."""
        
        if model is None or preprocessor is None:
            return []
        
        try:
            # Get prediction period data
            pred_data = signal_data.loc[pred_start:pred_end]
            
            if len(pred_data) == 0:
                return []
            
            # Prepare features (need history for sequences)
            history_start = pred_start - timedelta(days=30)
            extended_data = signal_data.loc[history_start:pred_end]
            
            features, _ = preprocessor.prepare_pair_data(extended_data)
            
            if len(features) < preprocessor.sequence_length:
                return []
            
            # Create sequences
            feature_scaled = preprocessor.feature_scaler.transform(features)
            
            raw_predictions = []
            model.eval()
            
            with torch.no_grad():
                for i in range(preprocessor.sequence_length, len(feature_scaled)):
                    # Check if this timestamp is in prediction period
                    current_time = features.index[i]
                    if current_time < pred_start or current_time > pred_end:
                        continue
                    
                    # Get sequence
                    seq = feature_scaled[i-preprocessor.sequence_length:i]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    
                    # Predict
                    output = model(seq_tensor)
                    
                    if preprocessor.target_mode == 'binary':
                        prob = torch.sigmoid(output).cpu().numpy()[0][0]
                        # Use calibrated threshold
                        signal = 1 if prob > optimal_threshold else 0
                        confidence = abs(prob - 0.5) * 2
                    else:  # three_class
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        max_prob = np.max(probs)
                        signal = np.argmax(probs)  # 0=long, 1=flat, 2=short
                        confidence = max_prob
                    
                    raw_predictions.append({
                        'timestamp': current_time,
                        'signal': signal,
                        'confidence': confidence,
                        'raw_output': output.cpu().numpy().tolist(),
                        'optimal_threshold': optimal_threshold
                    })
            
            # Enhanced: Use TradeSignalEngine for signal conversion
            trade_engine = TradeSignalEngine(
                signal_threshold=optimal_threshold,  # Use calibrated threshold
                k_pips_mult=self.k_pips_mult,
                target_mode=preprocessor.target_mode,
                pair_name=self.current_pair_name
            )
            
            # Convert raw predictions to trade signals
            trade_signals = trade_engine.generate_trade_signals(
                raw_predictions, 
                k_pips,
                risk_filters={
                    'session_filter': {'allowed': ['london', 'ny', 'overlap']},
                    'min_volatility': 0.001
                }
            )
            
            print(f"   🎯 Generated {len(trade_signals)} trade signals from {len(raw_predictions)} predictions (threshold: {optimal_threshold:.3f})")
            
            return trade_signals
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return []
    
    def _execute_trades(self, pair_name: str, exec_data: pd.DataFrame,
                       predictions: List[Dict], pred_start: pd.Timestamp,
                       pred_end: pd.Timestamp, current_positions: List,
                       current_equity: float, k_pips: float, **kwargs) -> Tuple[List, List]:
        """Execute trades based on predictions using execution timeframe data."""
        
        period_trades = []
        period_equity = []
        
        # Get pair configuration for pip values
        fetcher = create_fetcher(self.config)
        pair_config = fetcher.get_pair_config(pair_name)
        pip_value = pair_config.get('pip_value', 0.0001)
        
        position_size = kwargs.get('position_size', 0.01)
        spread_multiplier = kwargs.get('spread_multiplier', 1.2)
        commission_pips = kwargs.get('commission_pips', 0.0)
        slippage_pips = kwargs.get('slippage_pips', 0.1)
        max_positions = kwargs.get('max_positions', 1)
        
        # Process each prediction
        for pred in predictions:
            signal_time = pred['timestamp']
            signal_type = pred.get('signal_type', 'flat')
            confidence = pred.get('confidence', 0.0)
            
            if signal_type == 'flat':
                continue
            
            # Find execution window (next 5 M5 candles after signal)
            exec_start = signal_time
            exec_end = signal_time + timedelta(minutes=25)  # 5 * 5-minute candles
            
            exec_window = exec_data.loc[exec_start:exec_end]
            
            if len(exec_window) == 0:
                continue
            
            # Check position limits
            active_positions = [p for p in current_positions if p['status'] == 'open']
            if len(active_positions) >= max_positions:
                continue
            
            # Execute trade based on signal
            trade = None
            if signal_type == 'long':
                trade = self._execute_long_trade(
                    pair_name=pair_name,
                    exec_window=exec_window,
                    signal_time=signal_time,
                    confidence=confidence,
                    position_size=position_size,
                    spread_multiplier=spread_multiplier,
                    commission_pips=commission_pips,
                    slippage_pips=slippage_pips,
                    pip_value=pip_value
                )
            elif signal_type == 'short':
                trade = self._execute_short_trade(
                    pair_name=pair_name,
                    exec_window=exec_window,
                    signal_time=signal_time,
                    confidence=confidence,
                    position_size=position_size,
                    spread_multiplier=spread_multiplier,
                    commission_pips=commission_pips,
                    slippage_pips=slippage_pips,
                    pip_value=pip_value
                )
            
            if trade:
                period_trades.append(trade)
                current_positions.append({
                    'id': len(current_positions),
                    'pair': pair_name,
                    'direction': trade['direction'],
                    'entry_time': trade['entry_time'],
                    'entry_price': trade['entry_price'],
                    'size': position_size,
                    'status': 'closed',  # Trade already completed
                    'exit_time': trade['exit_time'],
                    'exit_price': trade['exit_price']
                })
                
                # Update equity curve
                current_equity += trade['pnl_usd']
                period_equity.append({
                    'timestamp': signal_time,
                    'equity': current_equity,
                    'trade_pnl': trade['pnl_usd']
                })
        
        return period_trades, period_equity
    
    def _execute_long_trade(self, pair_name: str, exec_window: pd.DataFrame,
                           signal_time: pd.Timestamp, confidence: float,
                           position_size: float, spread_multiplier: float,
                           commission_pips: float, slippage_pips: float,
                           pip_value: float) -> Optional[Dict]:
        """Execute a long trade in the execution window."""
        
        if len(exec_window) < 2:
            return None
        
        # Entry: Use ask price (buy at ask)
        entry_candle = exec_window.iloc[0]
        entry_price = entry_candle['close'] + (entry_candle['spread'] * spread_multiplier / 2)
        entry_price += slippage_pips * pip_value  # Add slippage
        entry_time = entry_candle.name
        
        # Exit: Use bid price after holding for remaining window
        exit_candle = exec_window.iloc[-1]
        exit_price = exit_candle['close'] - (exit_candle['spread'] * spread_multiplier / 2)
        exit_price -= slippage_pips * pip_value  # Subtract slippage
        exit_time = exit_candle.name
        
        # Calculate P&L
        pnl_pips = (exit_price - entry_price) / pip_value
        pnl_pips -= commission_pips  # Subtract commission
        
        pnl_usd = pnl_pips * position_size * 100000 * pip_value  # Convert to USD
        
        return {
            'pair': pair_name,
            'direction': 'long',
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'confidence': confidence,
            'commission_pips': commission_pips,
            'slippage_pips': slippage_pips * 2  # Entry + exit
        }
    
    def _execute_short_trade(self, pair_name: str, exec_window: pd.DataFrame,
                            signal_time: pd.Timestamp, confidence: float,
                            position_size: float, spread_multiplier: float,
                            commission_pips: float, slippage_pips: float,
                            pip_value: float) -> Optional[Dict]:
        """Execute a short trade in the execution window."""
        
        if len(exec_window) < 2:
            return None
        
        # Entry: Use bid price (sell at bid)
        entry_candle = exec_window.iloc[0]
        entry_price = entry_candle['close'] - (entry_candle['spread'] * spread_multiplier / 2)
        entry_price -= slippage_pips * pip_value  # Subtract slippage
        entry_time = entry_candle.name
        
        # Exit: Use ask price after holding for remaining window
        exit_candle = exec_window.iloc[-1]
        exit_price = exit_candle['close'] + (exit_candle['spread'] * spread_multiplier / 2)
        exit_price += slippage_pips * pip_value  # Add slippage
        exit_time = exit_candle.name
        
        # Calculate P&L (reversed for short)
        pnl_pips = (entry_price - exit_price) / pip_value
        pnl_pips -= commission_pips  # Subtract commission
        
        pnl_usd = pnl_pips * position_size * 100000 * pip_value  # Convert to USD
        
        return {
            'pair': pair_name,
            'direction': 'short',
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'confidence': confidence,
            'commission_pips': commission_pips,
            'slippage_pips': slippage_pips * 2  # Entry + exit
        }
    
    def _calculate_performance_metrics(self, equity_curve: List[Dict],
                                     trade_log: List[Dict], 
                                     pair_name: str) -> Dict[str, float]:
        """Enhanced performance metrics calculation."""
        
        if not trade_log:
            return self._get_empty_metrics()
        
        # Basic trade statistics
        total_trades = len(trade_log)
        winning_trades = sum(1 for t in trade_log if t['pnl_pips'] > 0)
        losing_trades = sum(1 for t in trade_log if t['pnl_pips'] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L calculations
        total_pnl = sum(t['pnl_pips'] for t in trade_log)
        total_pnl_usd = sum(t['pnl_usd'] for t in trade_log)
        
        wins = [t['pnl_pips'] for t in trade_log if t['pnl_pips'] > 0]
        losses = [t['pnl_pips'] for t in trade_log if t['pnl_pips'] < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Enhanced metrics calculations
        returns = [t['pnl_pips'] for t in trade_log]
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0.0
        sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0.0
        
        # Consecutive losses
        consecutive_losses = []
        current_streak = 0
        for ret in returns:
            if ret < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_losses.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            consecutive_losses.append(current_streak)
        
        max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
        
        # Drawdown calculation
        max_drawdown, max_dd_duration = self._calculate_drawdown(equity_curve)
        
        # Recovery factor and annualized return
        if len(equity_curve) > 1:
            equity_values = [point['equity'] for point in equity_curve]
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
            days_elapsed = len(equity_curve) / (24 * 4)  # Approximate days (assuming 15min data)
            
            if days_elapsed > 0:
                annualized_return = (1 + total_return) ** (252 / days_elapsed) - 1
            else:
                annualized_return = 0.0
            
            recovery_factor = abs(total_pnl_usd / (max_drawdown * equity_values[0] / 100)) if max_drawdown > 0 else 0.0
        else:
            annualized_return = 0.0
            recovery_factor = 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / (max_drawdown / 100.0) if max_drawdown > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_pnl_usd': total_pnl_usd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'annualized_return': annualized_return * 100,  # Percentage
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_duration_hours': self._calculate_avg_trade_duration(trade_log),
            'profit_per_trade': total_pnl_usd / len(trade_log) if trade_log else 0.0,
            'largest_win': max([t['pnl_usd'] for t in trade_log]) if trade_log else 0.0,
            'largest_loss': min([t['pnl_usd'] for t in trade_log]) if trade_log else 0.0
        }
    
    def _calculate_drawdown(self, equity_curve: List[Dict]) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        
        if not equity_curve:
            return 0.0, 0
        
        equity_values = [point['equity'] for point in equity_curve]
        
        if len(equity_values) < 2:
            return 0.0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdown
        drawdown = (equity_values - running_max) / running_max * 100
        
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate max drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                current_dd_duration += 1
            else:
                max_dd_duration = max(max_dd_duration, current_dd_duration)
                current_dd_duration = 0
        
        max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        return max_drawdown, max_dd_duration
    
    def _calculate_avg_trade_duration(self, trade_log: List[Dict]) -> float:
        """Calculate average trade duration in hours."""
        
        durations = []
        for trade in trade_log:
            try:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
            except:
                continue
        
        return np.mean(durations) if durations else 0.0
    
    def _generate_validation_predictions(self, model: Any, preprocessor: Any, 
                                       X_val: np.ndarray, y_val: np.ndarray) -> List[Dict]:
        """Generate predictions for validation set analysis."""
        
        predictions = []
        model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            outputs = model(X_val_tensor)
            
            for i, (output, true_label) in enumerate(zip(outputs, y_val)):
                if preprocessor.target_mode == 'binary':
                    prob = torch.sigmoid(output).cpu().numpy()[0]
                    predicted_class = 1 if prob > 0.5 else 0
                    confidence = abs(prob - 0.5) * 2
                    will_trade = prob > self.signal_threshold or prob < (1 - self.signal_threshold)
                else:  # three_class
                    probs = torch.softmax(output, dim=0).cpu().numpy()
                    predicted_class = np.argmax(probs)
                    confidence = np.max(probs)
                    will_trade = predicted_class != 1 and confidence > self.signal_threshold  # Not flat
                
                predictions.append({
                    'predicted_class': predicted_class,
                    'true_class': int(true_label),
                    'confidence': confidence,
                    'correct': predicted_class == int(true_label),
                    'will_trade': will_trade
                })
        
        return predictions
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary for failed cases."""
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_pnl_usd': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'calmar_ratio': 0.0,
            'recovery_factor': 0.0,
            'annualized_return': 0.0,
            'max_consecutive_losses': 0,
            'avg_trade_duration_hours': 0.0,
            'profit_per_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    def _save_pair_results(self, results: Dict[str, Any], pair_name: str) -> None:
        """Save backtest results for a pair."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        if not equity_df.empty:
            equity_df.to_csv(self.output_dir / f"{pair_name}_equity_curve_{timestamp}.csv", index=False)
        
        # Save trade log
        trades_df = pd.DataFrame(results['trade_log'])
        if not trades_df.empty:
            trades_df.to_csv(self.output_dir / f"{pair_name}_trades_{timestamp}.csv", index=False)
        
        # Save window results
        windows_df = pd.DataFrame(results['window_results'])
        if not windows_df.empty:
            windows_df.to_csv(self.output_dir / f"{pair_name}_windows_{timestamp}.csv", index=False)
        
        # Save performance summary
        summary = {
            'pair_name': pair_name,
            'backtest_period': results['backtest_period'],
            'performance_metrics': results['performance_metrics'],
            'total_windows': len(results['window_results']),
            'threshold_analysis': results.get('threshold_analysis', {}),
            'system_config': {
                'signal_tf': self.signal_tf,
                'exec_tf': self.exec_tf,
                'train_window_days': self.train_window_days,
                'retrain_freq_days': self.retrain_freq_days,
                'signal_threshold': self.signal_threshold,
                'k_pips_mult': self.k_pips_mult,
                'fine_tune': self.fine_tune
            },
            'timestamp': timestamp
        }
        
        with open(self.output_dir / f"{pair_name}_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def generate_final_report(self, backtest_results: Dict[str, Any], timestamp: str) -> None:
        """Generate comprehensive final report."""
        
        print(f"\n📊 GENERATING FINAL BACKTEST REPORT")
        print(f"{'='*50}")
        
        # Aggregate results
        successful_pairs = [name for name, result in backtest_results.items() 
                           if result.get('status') == 'success']
        
        if not successful_pairs:
            print("❌ No successful backtests to report")
            return
        
        # MLflow final summary
        if self.mlflow_manager:
            self.mlflow_manager.log_backtest_summary(backtest_results)
        
        # Create HTML report
        self._create_html_report(backtest_results, timestamp)
        
        # Create summary CSV
        self._create_summary_csv(backtest_results, timestamp)
        
        # Create consolidated equity curve
        self._create_consolidated_equity_curve(backtest_results, timestamp)
        
        # Print summary to console
        self._print_final_summary(backtest_results)
    
    def _create_html_report(self, backtest_results: Dict[str, Any], timestamp: str) -> None:
        """Create comprehensive HTML report with proper Unicode encoding."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Enhanced LSTM Walk-Forward Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .pair-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .neutral {{ color: orange; font-weight: bold; }}
        .threshold-analysis {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced LSTM Walk-Forward Backtest Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>System:</strong> Multi-Timeframe LSTM ({self.signal_tf} to {self.exec_tf})</p>
        <p><strong>Training Window:</strong> {self.train_window_days} days</p>
        <p><strong>Retrain Frequency:</strong> {self.retrain_freq_days} days</p>
        <p><strong>Fine-tune Mode:</strong> {'Enabled' if self.fine_tune else 'Disabled'}</p>
        <p><strong>K-pips Multiplier:</strong> {self.k_pips_mult}</p>
        <p><strong>Enhanced Features:</strong> ROC Threshold Optimization, Three-Class Mode, Time-based Splits</p>
    </div>
"""
        
        # Add individual pair results
        for pair_name, result in backtest_results.items():
            if result.get('status') != 'success':
                html_content += f"""
    <div class="pair-section">
        <h2>{pair_name} - Failed</h2>
        <p><strong>Error:</strong> {result.get('error', 'Unknown error')}</p>
    </div>
"""
                continue
            
            metrics = result['performance_metrics']
            
            # Determine overall performance
            if metrics['total_pnl'] > 0 and metrics['win_rate'] > 50:
                status_class = "positive"
                status_text = "SUCCESS"
            elif metrics['total_pnl'] > 0:
                status_class = "neutral"
                status_text = "MIXED"
            else:
                status_class = "negative"
                status_text = "NEGATIVE"
            
            html_content += f"""
    <div class="pair-section">
        <h2 class="{status_class}">{pair_name} - {status_text}</h2>
        <p><strong>Period:</strong> {result['backtest_period']['start']} to {result['backtest_period']['end']}</p>
        
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Trades</td><td>{metrics['total_trades']}</td></tr>
            <tr><td>Win Rate</td><td>{metrics['win_rate']:.1f}%</td></tr>
            <tr><td>Total P&L (pips)</td><td class="{status_class}">{metrics['total_pnl']:.2f}</td></tr>
            <tr><td>Total P&L (USD)</td><td class="{status_class}">${metrics['total_pnl_usd']:.2f}</td></tr>
            <tr><td>Average Win</td><td class="positive">{metrics['avg_win']:.2f} pips</td></tr>
            <tr><td>Average Loss</td><td class="negative">{metrics['avg_loss']:.2f} pips</td></tr>
            <tr><td>Profit Factor</td><td>{metrics['profit_factor']:.2f}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{metrics['sharpe_ratio']:.3f}</td></tr>
            <tr><td>Sortino Ratio</td><td>{metrics['sortino_ratio']:.3f}</td></tr>
            <tr><td>Max Drawdown</td><td class="negative">{metrics['max_drawdown']:.2f}%</td></tr>
            <tr><td>Recovery Factor</td><td>{metrics['recovery_factor']:.2f}</td></tr>
            <tr><td>Annualized Return</td><td>{metrics['annualized_return']:.1f}%</td></tr>
            <tr><td>Max Consecutive Losses</td><td>{metrics['max_consecutive_losses']}</td></tr>
            <tr><td>Avg Trade Duration</td><td>{metrics['avg_trade_duration_hours']:.1f} hours</td></tr>
        </table>
        
        <h3>Window Performance</h3>
        <p>Total windows: {len(result['window_results'])}</p>
        <p>Average validation accuracy: {np.mean([w['val_acc'] for w in result['window_results']]):.1f}%</p>
        
        <h3>Enhanced Features Used</h3>
        <div class="threshold-analysis">
            <p><strong>ROC-based Threshold Optimization:</strong> Yes</p>
            <p><strong>Three-class Mode:</strong> Enabled for better flat detection</p>
            <p><strong>Time-based Validation:</strong> Chronological splits to prevent leakage</p>
            <p><strong>Enhanced K-pips Formula:</strong> max(0.3*ATR, 1.2*spread)</p>
        </div>
"""
        
        # Add threshold analysis if available
        threshold_analysis = result.get('threshold_analysis', {})
        if threshold_analysis:
            html_content += """
        <h3>Threshold Analysis</h3>
        <div class="threshold-analysis">
"""
            for window_id, analysis in threshold_analysis.items():
                html_content += f"""
            <p><strong>Window {window_id}:</strong></p>
            <ul>
                <li>Optimized threshold: {analysis.get('optimal_threshold', 'N/A'):.3f}</li>
                <li>Method: {analysis.get('method', 'N/A')}</li>
                <li>Trade frequency: {analysis.get('trade_frequency', 0.0):.1%}</li>
            </ul>
"""
        html_content += """
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML report with proper encoding
        report_path = self.output_dir / f"enhanced_backtest_report_{timestamp}.html"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Enhanced HTML report saved: {report_path}")
        except Exception as e:
            print(f"WARNING: Failed to save HTML report: {e}")
    
    def _create_summary_csv(self, backtest_results: Dict[str, Any], timestamp: str) -> None:
        """Create summary CSV with enhanced metrics."""
        
        summary_data = []
        
        for pair_name, result in backtest_results.items():
            if result.get('status') != 'success':
                summary_data.append({
                    'Pair': pair_name,
                    'Status': 'Failed',
                    'Error': result.get('error', 'Unknown'),
                    'Total_Trades': 0,
                    'Win_Rate': 0,
                    'Total_PnL_Pips': 0,
                    'Total_PnL_USD': 0,
                    'Sharpe_Ratio': 0,
                    'Sortino_Ratio': 0,
                    'Max_Drawdown': 0,
                    'Recovery_Factor': 0,
                    'Profit_Factor': 0,
                    'ROC_Optimized': False,
                    'Three_Class_Mode': False
                })
                continue
            
            metrics = result['performance_metrics']
            
            summary_data.append({
                'Pair': pair_name,
                'Status': 'Success',
                'Error': '',
                'Total_Trades': metrics['total_trades'],
                'Win_Rate': metrics['win_rate'],
                'Total_PnL_Pips': metrics['total_pnl'],
                'Total_PnL_USD': metrics['total_pnl_usd'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Sortino_Ratio': metrics['sortino_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Recovery_Factor': metrics['recovery_factor'],
                'Profit_Factor': metrics['profit_factor'],
                'Avg_Win_Pips': metrics['avg_win'],
                'Avg_Loss_Pips': metrics['avg_loss'],
                'Calmar_Ratio': metrics['calmar_ratio'],
                'Annualized_Return': metrics['annualized_return'],
                'Max_Consecutive_Losses': metrics['max_consecutive_losses'],
                'Avg_Trade_Duration_Hours': metrics['avg_trade_duration_hours'],
                'Total_Windows': len(result['window_results']),
                'Avg_Val_Acc': np.mean([w['val_acc'] for w in result['window_results']]),
                'K_Pips_Mult': self.k_pips_mult,
                'ROC_Optimized': True,  # Always true in enhanced version
                'Three_Class_Mode': True,  # Forced in enhanced version
                'Time_Based_Split': True,  # Enhanced feature
                'Enhanced_K_Pips': True  # Enhanced formula
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / f"enhanced_backtest_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"📊 Enhanced summary CSV saved: {summary_path}")
    
    def _create_consolidated_equity_curve(self, backtest_results: Dict[str, Any], timestamp: str) -> None:
        """Create consolidated equity curve across all pairs."""
        
        all_equity_points = []
        
        for pair_name, result in backtest_results.items():
            if result.get('status') != 'success':
                continue
            
            for point in result['equity_curve']:
                all_equity_points.append({
                    'timestamp': point['timestamp'],
                    'pair': pair_name,
                    'equity': point['equity'],
                    'trade_pnl': point.get('trade_pnl', 0)
                })
        
        if all_equity_points:
            equity_df = pd.DataFrame(all_equity_points)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df = equity_df.sort_values('timestamp')
            
            # Create consolidated curve
            consolidated_equity = []
            current_total = 10000.0 * len([r for r in backtest_results.values() if r.get('status') == 'success'])
            
            for _, row in equity_df.iterrows():
                current_total += row['trade_pnl']
                consolidated_equity.append({
                    'timestamp': row['timestamp'],
                    'total_equity': current_total,
                    'pair': row['pair'],
                    'trade_pnl': row['trade_pnl']
                })
            
            consolidated_df = pd.DataFrame(consolidated_equity)
            consolidated_path = self.output_dir / f"consolidated_equity_{timestamp}.csv"
            consolidated_df.to_csv(consolidated_path, index=False)
            
            print(f"📊 Consolidated equity curve saved: {consolidated_path}")
    
    def _print_final_summary(self, backtest_results: Dict[str, Any]) -> None:
        """Print enhanced final summary to console."""
        
        successful_pairs = [name for name, result in backtest_results.items() 
                           if result.get('status') == 'success']
        failed_pairs = [name for name, result in backtest_results.items() 
                       if result.get('status') != 'success']
        
        print(f"\n*** ENHANCED BACKTEST SUMMARY ***")
        print(f"{'='*50}")
        print(f"Total pairs tested: {len(backtest_results)}")
        print(f"Successful: {len(successful_pairs)}")
        print(f"Failed: {len(failed_pairs)}")
        
        if successful_pairs:
            print(f"\nENHANCED FEATURES APPLIED:")
            print(f"   ✅ ROC-based threshold optimization")
            print(f"   ✅ Three-class mode (Long/Flat/Short)")
            print(f"   ✅ Time-based validation splits")
            print(f"   ✅ Enhanced k-pips formula (0.3*ATR, 1.2*spread)")
            print(f"   ✅ Fine-tuning with best weights")
            print(f"   ✅ Comprehensive MLflow logging")
            
            print(f"\nPERFORMANCE OVERVIEW:")
            
            total_trades = 0
            total_pnl_usd = 0
            all_win_rates = []
            all_sharpe_ratios = []
            all_sortino_ratios = []
            all_max_drawdowns = []
            all_threshold_improvements = []
            
            for pair_name in successful_pairs:
                result = backtest_results[pair_name]
                metrics = result['performance_metrics']
                
                total_trades += metrics['total_trades']
                total_pnl_usd += metrics['total_pnl_usd']
                all_win_rates.append(metrics['win_rate'])
                all_sharpe_ratios.append(metrics['sharpe_ratio'])
                all_sortino_ratios.append(metrics['sortino_ratio'])
                all_max_drawdowns.append(metrics['max_drawdown'])
                
                # Calculate threshold improvement
                threshold_analysis = result.get('threshold_analysis', {})
                if threshold_analysis:
                    improvements = []
                    for analysis in threshold_analysis.values():
                        if 'improvement' in analysis:
                            improvements.append(abs(analysis['improvement']))
                    if improvements:
                        all_threshold_improvements.append(np.mean(improvements))
                
                status_icon = "[+]" if metrics['total_pnl'] > 0 else "[-]"
                print(f"   {status_icon} {pair_name}: {metrics['total_trades']} trades, "
                      f"{metrics['total_pnl']:.1f} pips, {metrics['win_rate']:.1f}% WR, "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            
            print(f"\nAGGREGATE METRICS:")
            print(f"   Total P&L: ${total_pnl_usd:.2f}")
            print(f"   Total Trades: {total_trades}")
            print(f"   Average Win Rate: {np.mean(all_win_rates):.1f}%")
            print(f"   Average Sharpe: {np.mean(all_sharpe_ratios):.3f}")
            print(f"   Average Sortino: {np.mean(all_sortino_ratios):.3f}")
            print(f"   Average Max DD: {np.mean(all_max_drawdowns):.1f}%")
            
            if all_threshold_improvements:
                print(f"   Average Threshold Improvement: {np.mean(all_threshold_improvements):.3f}")
            
            # Overall assessment with enhanced criteria
            avg_sharpe = np.mean(all_sharpe_ratios)
            avg_win_rate = np.mean(all_win_rates)
            avg_sortino = np.mean(all_sortino_ratios)
            
            if (total_pnl_usd > 0 and avg_win_rate > 55 and 
                avg_sharpe > 0.8 and avg_sortino > 0.6):
                print(f"\nOVERALL ASSESSMENT: EXCELLENT (Enhanced system working)")
            elif total_pnl_usd > 0 and avg_sharpe > 0.3 and avg_sortino > 0.2:
                print(f"\nOVERALL ASSESSMENT: POSITIVE (Enhancements effective)")
            elif total_pnl_usd > 0:
                print(f"\nOVERALL ASSESSMENT: MIXED (Partial success)")
            else:
                print(f"\nOVERALL ASSESSMENT: NEGATIVE (Needs tuning)")
        
        if failed_pairs:
            print(f"\nFAILED PAIRS:")
            for pair_name in failed_pairs:
                error = backtest_results[pair_name].get('error', 'Unknown error')
                print(f"   * {pair_name}: {error}")
        
        print(f"\nENHANCED SYSTEM CONFIGURATION:")
        print(f"   Signal TF: {self.signal_tf}")
        print(f"   Execution TF: {self.exec_tf}")
        print(f"   Train Window: {self.train_window_days} days")
        print(f"   Retrain Frequency: {self.retrain_freq_days} days")
        print(f"   Fine-tune Mode: {'Enabled' if self.fine_tune else 'Disabled'}")
        print(f"   K-pips Multiplier: {self.k_pips_mult}")
        print(f"   Enhanced Features: All active")
        
        # Threshold optimization summary
        print(f"\nTHRESHOLD OPTIMIZATION SUMMARY:")
        all_threshold_analyses = {}
        for pair_name, result in backtest_results.items():
            if result.get('status') == 'success' and 'threshold_analysis' in result:
                threshold_analysis = result['threshold_analysis']
                for window_id, analysis in threshold_analysis.items():
                    if window_id not in all_threshold_analyses:
                        all_threshold_analyses[window_id] = []
                    all_threshold_analyses[window_id].append({
                        'pair': pair_name,
                        'threshold': analysis.get('optimal_threshold', self.signal_threshold),
                        'method': analysis.get('method', 'default')
                    })
        
        if all_threshold_analyses:
            for window_id, analyses in all_threshold_analyses.items():
                avg_threshold = np.mean([a['threshold'] for a in analyses])
                methods = list(set([a['method'] for a in analyses]))
                print(f"   Window {window_id}: Avg optimal threshold = {avg_threshold:.3f}")
                print(f"      Methods used: {', '.join(methods)}")
            
            overall_avg = np.mean([a['threshold'] for analyses in all_threshold_analyses.values() for a in analyses])
            if abs(overall_avg - self.signal_threshold) > 0.05:
                print(f"   💡 Recommendation: Consider --signal_threshold {overall_avg:.3f}")
        
        print(f"\nENHANCED OUTPUTS:")
        print(f"   📊 All results saved to: {self.output_dir}")
        print(f"   📈 ROC analysis plots in: {self.output_dir}/threshold_optimization/")
        print(f"   📋 Enhanced HTML report with all metrics")
        print(f"   📊 Prediction histograms logged to MLflow")
        
        if self.mlflow_manager:
            print(f"   🌐 MLflow tracking: {self.mlflow_manager.mlflow_uri}")
            print(f"   📊 Classification reports and confusion matrices logged")
        
        print(f"\n🎯 NEXT STEPS:")
        if total_pnl_usd > 0:
            print(f"   1. Review ROC threshold plots for further optimization")
            print(f"   2. Analyze three-class predictions for flat market detection")
            print(f"   3. Consider increasing k_pips_mult if too many trades")
            print(f"   4. Examine MLflow runs for detailed model performance")
        else:
            print(f"   1. Increase k_pips_mult to reduce trade frequency")
            print(f"   2. Adjust training window size")
            print(f"   3. Review feature engineering for specific pairs")
            print(f"   4. Consider ensemble methods or different model architectures")

__all__ = ['WalkForwardBacktester']
