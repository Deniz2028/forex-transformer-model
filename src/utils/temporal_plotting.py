# src/utils/temporal_plotting.py
"""
Temporal training plotting utilities.
Validation curves, training metrics ve performance plots.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TemporalPlotter:
    """
    Temporal eğitim için plotting sınıfı.
    Validation curves, training metrics ve signal analysis plots.
    """
    
    def __init__(self, output_dir: str = "temporal_plots", format: str = "png"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.format = format.lower()
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        logger.info(f"📊 TemporalPlotter initialized: {self.output_dir}")
    
    def plot_training_history(self, history: Dict[str, List], pair_name: str, 
                            model_type: str = "enhanced_transformer",
                            save: bool = True) -> Optional[str]:
        """
        Training history plot (train/val accuracy ve loss curves).
        
        Args:
            history: Training history dictionary
            pair_name: Currency pair name
            model_type: Model type for filename
            save: Whether to save the plot
            
        Returns:
            Plot file path if saved
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(history['train_acc']) + 1)
            
            # Accuracy plot
            ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax1.set_title(f'{pair_name} - Accuracy Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax2.set_title(f'{pair_name} - Loss Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Overfitting gap
            if 'train_acc' in history and 'val_acc' in history:
                overfitting_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
                ax3.plot(epochs, overfitting_gap, 'orange', linewidth=2)
                ax3.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold (15pp)')
                ax3.set_title(f'{pair_name} - Overfitting Gap')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Train - Val Accuracy (pp)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Learning rate (if available)
            if 'learning_rates' in history and history['learning_rates']:
                ax4.plot(epochs, history['learning_rates'], 'green', linewidth=2)
                ax4.set_title(f'{pair_name} - Learning Rate Schedule')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
            else:
                # Show final metrics as text
                final_train_acc = history['train_acc'][-1]
                final_val_acc = history['val_acc'][-1]
                best_val_acc = max(history['val_acc'])
                final_gap = final_train_acc - final_val_acc
                
                ax4.text(0.5, 0.7, f'Final Train Acc: {final_train_acc:.2f}%', 
                        transform=ax4.transAxes, fontsize=12, ha='center')
                ax4.text(0.5, 0.5, f'Final Val Acc: {final_val_acc:.2f}%', 
                        transform=ax4.transAxes, fontsize=12, ha='center')
                ax4.text(0.5, 0.3, f'Best Val Acc: {best_val_acc:.2f}%', 
                        transform=ax4.transAxes, fontsize=12, ha='center')
                ax4.text(0.5, 0.1, f'Overfitting Gap: {final_gap:.2f}pp', 
                        transform=ax4.transAxes, fontsize=12, ha='center')
                ax4.set_title(f'{pair_name} - Final Metrics')
                ax4.axis('off')
            
            plt.suptitle(f'{pair_name} {model_type.upper()} Training Analysis', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{pair_name}_training_history_{timestamp}.{self.format}"
                filepath = self.output_dir / filename
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"📈 Training history plot saved: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to plot training history for {pair_name}: {e}")
            plt.close()
            return None
    
    def plot_signal_analysis(self, signals_df: pd.DataFrame, pair_name: str,
                           train_period: tuple, signal_period: tuple,
                           price_data: Optional[pd.DataFrame] = None,
                           save: bool = True) -> Optional[str]:
        """
        Signal analysis plot with price chart and signals.
        
        Args:
            signals_df: Signals DataFrame
            pair_name: Currency pair name
            train_period: (start, end) for training period
            signal_period: (start, end) for signal period
            price_data: Optional price data for background
            save: Whether to save the plot
            
        Returns:
            Plot file path if saved
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            
            # Convert datetime columns
            signals_df['datetime'] = pd.to_datetime(signals_df['datetime'])
            
            # 1. Price chart with signals
            if price_data is not None and 'close' in price_data.columns:
                # Filter price data to signal period
                signal_start, signal_end = signal_period
                price_mask = (price_data.index >= signal_start) & (price_data.index <= signal_end)
                price_filtered = price_data[price_mask]
                
                ax1.plot(price_filtered.index, price_filtered['close'], 
                        'b-', alpha=0.7, linewidth=1, label='Price')
                
                # Plot signals on price chart
                long_signals = signals_df[signals_df['signal'] == 'LONG']
                short_signals = signals_df[signals_df['signal'] == 'SHORT']
                
                if not long_signals.empty:
                    ax1.scatter(long_signals['datetime'], long_signals['price'], 
                               color='green', marker='^', s=50, alpha=0.8, 
                               label=f'Long Signals ({len(long_signals)})')
                
                if not short_signals.empty:
                    ax1.scatter(short_signals['datetime'], short_signals['price'], 
                               color='red', marker='v', s=50, alpha=0.8, 
                               label=f'Short Signals ({len(short_signals)})')
            
            ax1.set_title(f'{pair_name} - Price Chart with Signals')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            
            # 2. Signal count by day
            signals_df['date'] = signals_df['datetime'].dt.date
            daily_signals = signals_df.groupby(['date', 'signal']).size().unstack(fill_value=0)
            
            if 'LONG' in daily_signals.columns:
                ax2.bar(daily_signals.index, daily_signals['LONG'], 
                       alpha=0.7, color='green', label='Long Signals')
            if 'SHORT' in daily_signals.columns:
                ax2.bar(daily_signals.index, daily_signals['SHORT'], 
                       alpha=0.7, color='red', label='Short Signals', bottom=daily_signals.get('LONG', 0))
            
            ax2.set_title(f'{pair_name} - Daily Signal Distribution')
            ax2.set_ylabel('Signal Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Confidence distribution
            ax3.hist(signals_df['confidence'], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(signals_df['confidence'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {signals_df["confidence"].mean():.3f}')
            ax3.set_title(f'{pair_name} - Signal Confidence Distribution')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'{pair_name} Signal Analysis\n'
                        f'Training: {train_period[0]} → {train_period[1]} | '
                        f'Signals: {signal_period[0]} → {signal_period[1]}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{pair_name}_signal_analysis_{timestamp}.{self.format}"
                filepath = self.output_dir / filename
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"📊 Signal analysis plot saved: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to plot signal analysis for {pair_name}: {e}")
            plt.close()
            return None
    
    def plot_pipeline_summary(self, pipeline_results: Dict[str, Any], 
                            save: bool = True) -> Optional[str]:
        """
        Pipeline summary plot with all pairs performance.
        
        Args:
            pipeline_results: Complete pipeline results
            save: Whether to save the plot
            
        Returns:
            Plot file path if saved
        """
        try:
            training_summary = pipeline_results.get('pipeline_summary', {}).get('training_summary', {})
            signal_summary = pipeline_results.get('pipeline_summary', {}).get('signal_summary', {})
            
            if not training_summary:
                logger.warning("No training summary available for pipeline plot")
                return None
            
            pairs = list(training_summary.keys())
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Validation accuracy comparison
            val_accs = [training_summary[pair].get('best_val_acc', 0) for pair in pairs]
            bars1 = ax1.bar(pairs, val_accs, alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_title('Best Validation Accuracy by Pair')
            ax1.set_ylabel('Validation Accuracy (%)')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars1, val_accs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.1f}%', ha='center', va='bottom')
            
            # 2. Overfitting gap comparison
            overfit_gaps = [training_summary[pair].get('overfitting_gap', 0) for pair in pairs]
            colors = ['green' if gap < 10 else 'orange' if gap < 15 else 'red' for gap in overfit_gaps]
            bars2 = ax2.bar(pairs, overfit_gaps, alpha=0.7, color=colors, edgecolor='black')
            ax2.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
            ax2.set_title('Overfitting Gap by Pair')
            ax2.set_ylabel('Train - Val Accuracy (pp)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, gap in zip(bars2, overfit_gaps):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{gap:.1f}pp', ha='center', va='bottom')
            
            # 3. Signal count comparison
            if signal_summary:
                signal_counts = [signal_summary.get(pair, {}).get('total_signals', 0) for pair in pairs]
                bars3 = ax3.bar(pairs, signal_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
                ax3.set_title('Total Signals Generated by Pair')
                ax3.set_ylabel('Signal Count')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, count in zip(bars3, signal_counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                            f'{count}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No Signal Data Available', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=14)
                ax3.set_title('Signal Generation')
            
            # 4. Training epochs comparison
            epochs_trained = [training_summary[pair].get('epochs_trained', 0) for pair in pairs]
            bars4 = ax4.bar(pairs, epochs_trained, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax4.set_title('Training Epochs by Pair')
            ax4.set_ylabel('Epochs')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, epochs in zip(bars4, epochs_trained):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{epochs}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.suptitle('Temporal Training Pipeline Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pipeline_summary_{timestamp}.{self.format}"
                filepath = self.output_dir / filename
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"📋 Pipeline summary plot saved: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to plot pipeline summary: {e}")
            plt.close()
            return None
    
    def plot_temporal_timeline(self, train_period: tuple, signal_period: tuple,
                             pairs: List[str], save: bool = True) -> Optional[str]:
        """
        Temporal timeline plot showing training and signal periods.
        
        Args:
            train_period: (start, end) for training
            signal_period: (start, end) for signals
            pairs: List of currency pairs
            save: Whether to save the plot
            
        Returns:
            Plot file path if saved
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            
            train_start, train_end = pd.to_datetime(train_period[0]), pd.to_datetime(train_period[1])
            signal_start, signal_end = pd.to_datetime(signal_period[0]), pd.to_datetime(signal_period[1])
            
            # Training period
            ax.barh(0, (train_end - train_start).days, left=train_start, 
                   height=0.8, color='skyblue', alpha=0.7, label='Training Period')
            
            # Signal period
            ax.barh(0, (signal_end - signal_start).days, left=signal_start, 
                   height=0.8, color='lightcoral', alpha=0.7, label='Signal Period')
            
            # Add gap indicator
            gap_days = (signal_start - train_end).days
            if gap_days > 0:
                ax.barh(0, gap_days, left=train_end, 
                       height=0.4, color='yellow', alpha=0.5, label=f'Gap ({gap_days} days)')
            
            # Labels
            ax.text(train_start + (train_end - train_start)/2, 0, 
                   f'TRAINING\n{train_start.strftime("%Y-%m-%d")} to {train_end.strftime("%Y-%m-%d")}', 
                   ha='center', va='center', fontweight='bold')
            
            ax.text(signal_start + (signal_end - signal_start)/2, 0, 
                   f'SIGNAL GENERATION\n{signal_start.strftime("%Y-%m-%d")} to {signal_end.strftime("%Y-%m-%d")}', 
                   ha='center', va='center', fontweight='bold')
            
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Date')
            ax.set_title(f'Temporal Training Timeline\nPairs: {", ".join(pairs)}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"temporal_timeline_{timestamp}.{self.format}"
                filepath = self.output_dir / filename
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"📅 Temporal timeline plot saved: {filepath}")
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to plot temporal timeline: {e}")
            plt.close()
            return None

def create_temporal_plots(pipeline_results: Dict[str, Any], 
                         output_dir: str = "temporal_plots",
                         format: str = "png") -> Dict[str, str]:
    """
    Create all temporal plots for pipeline results.
    
    Args:
        pipeline_results: Complete pipeline results
        output_dir: Output directory for plots
        format: Plot format (png, svg, pdf)
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    plotter = TemporalPlotter(output_dir, format)
    plot_paths = {}
    
    try:
        # Pipeline summary plot
        summary_path = plotter.plot_pipeline_summary(pipeline_results)
        if summary_path:
            plot_paths['pipeline_summary'] = summary_path
        
        # Individual training history plots
        training_results = pipeline_results.get('training_results', {})
        for pair_name, history in training_results.items():
            if history and 'train_acc' in history:
                history_path = plotter.plot_training_history(history, pair_name)
                if history_path:
                    plot_paths[f'{pair_name}_training_history'] = history_path
        
        # Signal analysis plots
        signal_results = pipeline_results.get('signal_results', {})
        pipeline_info = pipeline_results.get('pipeline_summary', {}).get('pipeline_info', {})
        
        train_period = pipeline_info.get('train_period', '').split(' → ')
        signal_period = pipeline_info.get('signal_period', '').split(' → ')
        
        if len(train_period) == 2 and len(signal_period) == 2:
            for pair_name, signals in signal_results.items():
                if signals and 'signals_df' in signals:
                    signal_path = plotter.plot_signal_analysis(
                        signals['signals_df'], pair_name, 
                        tuple(train_period), tuple(signal_period)
                    )
                    if signal_path:
                        plot_paths[f'{pair_name}_signal_analysis'] = signal_path
            
            # Temporal timeline plot
            pairs = list(training_results.keys())
            if pairs:
                timeline_path = plotter.plot_temporal_timeline(
                    tuple(train_period), tuple(signal_period), pairs
                )
                if timeline_path:
                    plot_paths['temporal_timeline'] = timeline_path
        
        logger.info(f"📊 Created {len(plot_paths)} temporal plots")
        return plot_paths
        
    except Exception as e:
        logger.error(f"❌ Failed to create temporal plots: {e}")
        return plot_paths

__all__ = [
    'TemporalPlotter',
    'create_temporal_plots'
]