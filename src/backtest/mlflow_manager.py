# src/backtest/mlflow_manager.py
"""MLflow run management for walk-forward backtesting."""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

logger = logging.getLogger(__name__)

class MLflowManager:
    """MLflow experiment tracking manager for backtesting."""
    
    def __init__(self, mlflow_uri: str = None, experiment_name: str = "LSTM_Walk_Forward_Backtest"):
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.main_run_id = None
        self.current_pair_run_id = None
        
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"📊 MLflow initialized: {mlflow_uri}")
    
    def start_backtest_run(self, config: Dict[str, Any]) -> str:
        """Start main backtest run."""
        
        if not self.mlflow_uri:
            return None
        
        try:
            self._ensure_no_active_run()
            
            run = mlflow.start_run(run_name="walk_forward_backtest")
            self.main_run_id = run.info.run_id
            
            # Log main configuration
            mlflow.log_params({
                "system_type": "walk_forward_backtest",
                "train_window_days": config.get('train_window_days', 180),
                "retrain_freq_days": config.get('retrain_freq_days', 7),
                "signal_tf": config.get('signal_tf', 'M15'),
                "exec_tf": config.get('exec_tf', 'M5'),
                "fine_tune": config.get('fine_tune', False),
                "target_mode": config.get('target_mode', 'binary'),
                "lookback_candles": config.get('lookback_candles', 180000)
            })
            
            logger.info(f"📊 Started main backtest run: {self.main_run_id}")
            return self.main_run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start backtest run: {e}")
            return None
    
    def start_pair_run(self, pair_name: str, pair_config: Dict[str, Any]) -> str:
        """Start nested run for individual pair."""
        
        if not self.mlflow_uri or not self.main_run_id:
            return None
        
        try:
            self._ensure_no_active_run()
            
            run = mlflow.start_run(run_name=f"{pair_name}_backtest", nested=True)
            self.current_pair_run_id = run.info.run_id
            
            # Log pair-specific parameters
            mlflow.log_params({
                "pair_name": pair_name,
                "position_size": pair_config.get('position_size', 0.01),
                "spread_multiplier": pair_config.get('spread_multiplier', 1.2),
                "commission_pips": pair_config.get('commission_pips', 0.0),
                "slippage_pips": pair_config.get('slippage_pips', 0.1),
                "max_positions": pair_config.get('max_positions', 1)
            })
            
            logger.info(f"📊 Started pair run for {pair_name}: {self.current_pair_run_id}")
            return self.current_pair_run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start pair run for {pair_name}: {e}")
            return None
    
    def start_window_run(self, window_id: int, window_config: Dict[str, Any]) -> str:
        """Start nested run for training window."""
        
        if not self.mlflow_uri or not self.current_pair_run_id:
            return None
        
        try:
            run = mlflow.start_run(run_name=f"window_{window_id}", nested=True)
            window_run_id = run.info.run_id
            
            # Log window parameters
            mlflow.log_params({
                "window_id": window_id,
                "train_start": window_config.get('train_start'),
                "train_end": window_config.get('train_end'),
                "pred_start": window_config.get('pred_start'),
                "pred_end": window_config.get('pred_end'),
                "train_samples": window_config.get('train_samples', 0),
                "is_fine_tune": window_config.get('is_fine_tune', False)
            })
            
            return window_run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start window run {window_id}: {e}")
            return None
    
    def log_window_results(self, window_results: Dict[str, Any], 
                          model_metrics: Dict[str, Any] = None,
                          y_true: np.ndarray = None,
                          y_pred: np.ndarray = None) -> None:
        """Log training window results to MLflow."""
        
        if not self.mlflow_uri:
            return
        
        try:
            # Log basic metrics
            mlflow.log_metrics({
                "val_accuracy": window_results.get('val_acc', 0.0),
                "train_samples": window_results.get('train_samples', 0),
                "predictions_made": window_results.get('predictions_made', 0),
                "trades_executed": window_results.get('trades_executed', 0),
                "period_pnl": window_results.get('period_pnl', 0.0),
                "ending_equity": window_results.get('ending_equity', 0.0)
            })
            
            # Log model performance metrics if available
            if model_metrics:
                mlflow.log_metrics({
                    f"model_{k}": v for k, v in model_metrics.items() 
                    if isinstance(v, (int, float))
                })
            
            # Log classification report and confusion matrix
            if y_true is not None and y_pred is not None:
                self._log_classification_artifacts(y_true, y_pred, window_results.get('window', 0))
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log window results: {e}")
    
    def log_pair_final_results(self, pair_results: Dict[str, Any]) -> None:
        """Log final results for a currency pair."""
        
        if not self.mlflow_uri:
            return
        
        try:
            performance_metrics = pair_results.get('performance_metrics', {})
            
            # Log performance metrics
            mlflow.log_metrics({
                "total_trades": performance_metrics.get('total_trades', 0),
                "winning_trades": performance_metrics.get('winning_trades', 0),
                "win_rate": performance_metrics.get('win_rate', 0.0),
                "total_pnl_pips": performance_metrics.get('total_pnl', 0.0),
                "total_pnl_usd": performance_metrics.get('total_pnl_usd', 0.0),
                "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0.0),
                "max_drawdown": performance_metrics.get('max_drawdown', 0.0),
                "profit_factor": performance_metrics.get('profit_factor', 0.0),
                "calmar_ratio": performance_metrics.get('calmar_ratio', 0.0),
                "avg_win_pips": performance_metrics.get('avg_win', 0.0),
                "avg_loss_pips": performance_metrics.get('avg_loss', 0.0)
            })
            
            # Log backtest period
            backtest_period = pair_results.get('backtest_period', {})
            mlflow.log_params({
                "backtest_start": backtest_period.get('start'),
                "backtest_end": backtest_period.get('end')
            })
            
            # Save equity curve and trade log as artifacts
            self._save_trading_artifacts(pair_results)
            
            # Create and save performance plots
            self._create_performance_plots(pair_results)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log pair final results: {e}")
    
    def log_backtest_summary(self, all_results: Dict[str, Any]) -> None:
        """Log final backtest summary."""
        
        if not self.mlflow_uri or not self.main_run_id:
            return
        
        try:
            # Switch back to main run
            with mlflow.start_run(run_id=self.main_run_id):
                
                # Aggregate metrics across all pairs
                successful_pairs = [name for name, result in all_results.items() 
                                  if result.get('status') == 'success']
                
                if successful_pairs:
                    total_trades = sum(
                        all_results[pair]['performance_metrics']['total_trades'] 
                        for pair in successful_pairs
                    )
                    total_pnl = sum(
                        all_results[pair]['performance_metrics']['total_pnl_usd'] 
                        for pair in successful_pairs
                    )
                    avg_sharpe = np.mean([
                        all_results[pair]['performance_metrics']['sharpe_ratio'] 
                        for pair in successful_pairs
                    ])
                    avg_max_dd = np.mean([
                        all_results[pair]['performance_metrics']['max_drawdown'] 
                        for pair in successful_pairs
                    ])
                    
                    # Log aggregate metrics
                    mlflow.log_metrics({
                        "total_pairs_tested": len(all_results),
                        "successful_pairs": len(successful_pairs),
                        "success_rate": len(successful_pairs) / len(all_results),
                        "aggregate_total_trades": total_trades,
                        "aggregate_total_pnl_usd": total_pnl,
                        "average_sharpe_ratio": avg_sharpe,
                        "average_max_drawdown": avg_max_dd
                    })
                
                # Save summary artifacts
                self._save_summary_artifacts(all_results)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to log backtest summary: {e}")
    
    def _log_classification_artifacts(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    window_id: int) -> None:
        """Log classification report and confusion matrix."""
        
        try:
            # Create temporary directory for artifacts
            artifact_dir = Path("temp_mlflow_artifacts")
            artifact_dir.mkdir(exist_ok=True)
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True)
            report_path = artifact_dir / f"classification_report_window_{window_id}.json"
            
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            
            mlflow.log_artifact(str(report_path))
            
            # Confusion matrix plot
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Window {window_id}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = artifact_dir / f"confusion_matrix_window_{window_id}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(cm_path))
            
            # Cleanup
            report_path.unlink(missing_ok=True)
            cm_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log classification artifacts: {e}")
    
    def _save_trading_artifacts(self, pair_results: Dict[str, Any]) -> None:
        """Save trading-related artifacts."""
        
        try:
            artifact_dir = Path("temp_mlflow_artifacts")
            artifact_dir.mkdir(exist_ok=True)
            
            pair_name = pair_results['pair_name']
            
            # Save equity curve
            if 'equity_curve' in pair_results and pair_results['equity_curve']:
                equity_df = pd.DataFrame(pair_results['equity_curve'])
                equity_path = artifact_dir / f"{pair_name}_equity_curve.csv"
                equity_df.to_csv(equity_path, index=False)
                mlflow.log_artifact(str(equity_path))
                equity_path.unlink(missing_ok=True)
            
            # Save trade log
            if 'trade_log' in pair_results and pair_results['trade_log']:
                trades_df = pd.DataFrame(pair_results['trade_log'])
                trades_path = artifact_dir / f"{pair_name}_trades.csv"
                trades_df.to_csv(trades_path, index=False)
                mlflow.log_artifact(str(trades_path))
                trades_path.unlink(missing_ok=True)
            
            # Save window results
            if 'window_results' in pair_results and pair_results['window_results']:
                windows_df = pd.DataFrame(pair_results['window_results'])
                windows_path = artifact_dir / f"{pair_name}_windows.csv"
                windows_df.to_csv(windows_path, index=False)
                mlflow.log_artifact(str(windows_path))
                windows_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trading artifacts: {e}")
    
    def _create_performance_plots(self, pair_results: Dict[str, Any]) -> None:
        """Create and save performance visualization plots."""
        
        try:
            artifact_dir = Path("temp_mlflow_artifacts")
            artifact_dir.mkdir(exist_ok=True)
            
            pair_name = pair_results['pair_name']
            
            # Equity curve plot
            if 'equity_curve' in pair_results and pair_results['equity_curve']:
                equity_data = pair_results['equity_curve']
                
                if equity_data:
                    df = pd.DataFrame(equity_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Plot equity curve
                    plt.subplot(2, 1, 1)
                    plt.plot(df['timestamp'], df['equity'])
                    plt.title(f'{pair_name} - Equity Curve')
                    plt.ylabel('Equity (USD)')
                    plt.grid(True, alpha=0.3)
                    
                    # Plot drawdown
                    equity_values = df['equity'].values
                    running_max = np.maximum.accumulate(equity_values)
                    drawdown = (equity_values - running_max) / running_max * 100
                    
                    plt.subplot(2, 1, 2)
                    plt.fill_between(df['timestamp'], drawdown, 0, alpha=0.3, color='red')
                    plt.plot(df['timestamp'], drawdown, color='red')
                    plt.title(f'{pair_name} - Drawdown')
                    plt.ylabel('Drawdown (%)')
                    plt.xlabel('Date')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    plot_path = artifact_dir / f"{pair_name}_performance.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    mlflow.log_artifact(str(plot_path))
                    plot_path.unlink(missing_ok=True)
            
            # Window performance plot
            if 'window_results' in pair_results and pair_results['window_results']:
                windows_data = pair_results['window_results']
                
                if windows_data:
                    df = pd.DataFrame(windows_data)
                    
                    plt.figure(figsize=(12, 10))
                    
                    # Validation accuracy over windows
                    plt.subplot(3, 1, 1)
                    plt.plot(df['window'], df['val_acc'], marker='o')
                    plt.title(f'{pair_name} - Validation Accuracy by Window')
                    plt.ylabel('Val Accuracy (%)')
                    plt.grid(True, alpha=0.3)
                    
                    # Trades per window
                    plt.subplot(3, 1, 2)
                    plt.bar(df['window'], df['trades_executed'])
                    plt.title(f'{pair_name} - Trades per Window')
                    plt.ylabel('Number of Trades')
                    plt.grid(True, alpha=0.3)
                    
                    # P&L per window
                    plt.subplot(3, 1, 3)
                    colors = ['green' if pnl >= 0 else 'red' for pnl in df['period_pnl']]
                    plt.bar(df['window'], df['period_pnl'], color=colors, alpha=0.7)
                    plt.title(f'{pair_name} - P&L per Window')
                    plt.ylabel('P&L (pips)')
                    plt.xlabel('Window')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    plot_path = artifact_dir / f"{pair_name}_windows.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    mlflow.log_artifact(str(plot_path))
                    plot_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create performance plots: {e}")
    
    def _save_summary_artifacts(self, all_results: Dict[str, Any]) -> None:
        """Save summary artifacts for all pairs."""
        
        try:
            artifact_dir = Path("temp_mlflow_artifacts")
            artifact_dir.mkdir(exist_ok=True)
            
            # Create summary DataFrame
            summary_data = []
            for pair_name, result in all_results.items():
                if result.get('status') == 'success':
                    metrics = result['performance_metrics']
                    summary_data.append({
                        'pair': pair_name,
                        'total_trades': metrics['total_trades'],
                        'win_rate': metrics['win_rate'],
                        'total_pnl_pips': metrics['total_pnl'],
                        'total_pnl_usd': metrics['total_pnl_usd'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown': metrics['max_drawdown'],
                        'profit_factor': metrics['profit_factor']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = artifact_dir / "backtest_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                mlflow.log_artifact(str(summary_path))
                summary_path.unlink(missing_ok=True)
                
                # Create summary plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Total P&L by pair
                axes[0, 0].bar(summary_df['pair'], summary_df['total_pnl_usd'])
                axes[0, 0].set_title('Total P&L by Pair')
                axes[0, 0].set_ylabel('P&L (USD)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Sharpe ratio by pair
                axes[0, 1].bar(summary_df['pair'], summary_df['sharpe_ratio'])
                axes[0, 1].set_title('Sharpe Ratio by Pair')
                axes[0, 1].set_ylabel('Sharpe Ratio')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Win rate by pair
                axes[1, 0].bar(summary_df['pair'], summary_df['win_rate'])
                axes[1, 0].set_title('Win Rate by Pair')
                axes[1, 0].set_ylabel('Win Rate (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Max drawdown by pair
                axes[1, 1].bar(summary_df['pair'], summary_df['max_drawdown'])
                axes[1, 1].set_title('Max Drawdown by Pair')
                axes[1, 1].set_ylabel('Max Drawdown (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                
                summary_plot_path = artifact_dir / "summary_performance.png"
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                mlflow.log_artifact(str(summary_plot_path))
                summary_plot_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save summary artifacts: {e}")
    
    def _ensure_no_active_run(self) -> None:
        """Ensure no MLflow run is currently active."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.debug("🔄 Ended previous active MLflow run")
        except Exception as e:
            logger.debug(f"⚠️ Error ending active run: {e}")
    
    def end_run(self) -> None:
        """Safely end current MLflow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("✅ MLflow run ended successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error ending MLflow run: {e}")
    
    def end_all_runs(self) -> None:
        """End all MLflow runs and reset state."""
        try:
            while mlflow.active_run():
                mlflow.end_run()
            
            self.main_run_id = None
            self.current_pair_run_id = None
            logger.info("✅ All MLflow runs ended")
            
        except Exception as e:
            logger.warning(f"⚠️ Error ending all MLflow runs: {e}")

__all__ = ['MLflowManager']
