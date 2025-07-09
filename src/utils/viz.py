"""Visualization utilities for training results."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def plot_training_results(trainer_histories: Dict[str, Dict], 
                         results_summary: Dict[str, Dict],
                         save_path: str = 'training_results.png') -> None:
    """Create comprehensive training results visualization.
    
    Args:
        trainer_histories: Dictionary of training histories per pair
        results_summary: Dictionary of results summary per pair
        save_path: Path to save the plot
    """
    if len(trainer_histories) < 1:
        print("⚠️ No training histories to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    
    # Get successful pairs
    pairs = [p for p in results_summary.keys() if results_summary[p]['status'] == 'success']
    
    # 1. Validation accuracy evolution
    ax1 = axes[0, 0]
    for pair_name, history in trainer_histories.items():
        ax1.plot(history['val_acc'], label=f'{pair_name}', linewidth=2)
    ax1.set_title('Validation Accuracy Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final performance bar chart
    ax2 = axes[0, 1]
    val_accs = [results_summary[p]['best_val_acc'] for p in pairs]
    colors = ['green' if acc > 60 else 'orange' if acc > 55 else 'red' for acc in val_accs]
    
    ax2.bar(pairs, val_accs, color=colors)
    ax2.set_title('Best Validation Accuracy by Pair')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting gap analysis
    ax3 = axes[0, 2]
    overfit_gaps = [results_summary[p].get('overfit_gap', 0) for p in pairs]
    colors_gap = ['red' if gap > 20 else 'orange' if gap > 10 else 'green' for gap in overfit_gaps]
    
    ax3.bar(pairs, overfit_gaps, color=colors_gap)
    ax3.set_title('Overfitting Gap (Train - Val Acc)')
    ax3.set_ylabel('Gap (pp)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Cool-down threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Minority ratio analysis
    ax4 = axes[1, 0]
    minority_ratios = [results_summary[p].get('minority_ratio', 0) for p in pairs]
    colors_ratio = ['red' if ratio < 0.15 else 'orange' if ratio < 0.3 else 'green' for ratio in minority_ratios]
    
    ax4.bar(pairs, minority_ratios, color=colors_ratio)
    ax4.set_title('Minority Class Ratio by Pair')
    ax4.set_ylabel('Minority Ratio')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='SMOTE threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Target mode distribution
    ax5 = axes[1, 1]
    target_modes = [results_summary[p].get('target_mode', 'binary') for p in pairs]
    if target_modes:
        mode_counts = pd.Series(target_modes).value_counts()
        ax5.pie(mode_counts.values, labels=mode_counts.index, autopct='%1.1f%%')
        ax5.set_title('Target Mode Distribution')
    
    # 6. Training epochs vs performance
    ax6 = axes[1, 2]
    epochs_used = [len(trainer_histories[p]['train_acc']) for p in pairs if p in trainer_histories]
    val_accs_scatter = [results_summary[p]['best_val_acc'] for p in pairs if p in trainer_histories]
    overfit_gaps_scatter = [results_summary[p].get('overfit_gap', 0) for p in pairs if p in trainer_histories]
    
    if epochs_used and val_accs_scatter:
        scatter = ax6.scatter(epochs_used, val_accs_scatter, c=overfit_gaps_scatter, 
                             cmap='RdYlGn_r', s=100)
        plt.colorbar(scatter, ax=ax6, label='Overfit Gap')
        ax6.set_title('Epochs vs Performance (Color = Overfit)')
        ax6.set_xlabel('Training Epochs')
        ax6.set_ylabel('Best Val Acc (%)')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {save_path}")

def plot_loss_curves(trainer_histories: Dict[str, Dict], 
                     save_path: str = 'loss_curves.png') -> None:
    """Plot training and validation loss curves.
    
    Args:
        trainer_histories: Dictionary of training histories per pair
        save_path: Path to save the plot
    """
    if len(trainer_histories) < 1:
        print("⚠️ No training histories to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    for pair_name, history in trainer_histories.items():
        ax1.plot(history['train_loss'], label=f'{pair_name}', linewidth=2)
    ax1.set_title('Training Loss Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation loss
    for pair_name, history in trainer_histories.items():
        ax2.plot(history['val_loss'], label=f'{pair_name}', linewidth=2)
    ax2.set_title('Validation Loss Evolution')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Loss curves saved to: {save_path}")

def plot_accuracy_comparison(trainer_histories: Dict[str, Dict], 
                           save_path: str = 'accuracy_comparison.png') -> None:
    """Plot training vs validation accuracy comparison.
    
    Args:
        trainer_histories: Dictionary of training histories per pair
        save_path: Path to save the plot
    """
    if len(trainer_histories) < 1:
        print("⚠️ No training histories to visualize")
        return
    
    n_pairs = len(trainer_histories)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 5))
    
    if n_pairs == 1:
        axes = [axes]
    
    for i, (pair_name, history) in enumerate(trainer_histories.items()):
        ax = axes[i]
        ax.plot(history['train_acc'], label='Training', linewidth=2)
        ax.plot(history['val_acc'], label='Validation', linewidth=2)
        ax.set_title(f'{pair_name} Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Accuracy comparison saved to: {save_path}")

def create_summary_table(results_summary: Dict[str, Dict]) -> pd.DataFrame:
    """Create a summary table of training results.
    
    Args:
        results_summary: Dictionary of results summary per pair
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for pair_name, results in results_summary.items():
        if results['status'] == 'success':
            summary_data.append({
                'Pair': pair_name,
                'Best Val Acc (%)': f"{results['best_val_acc']:.2f}",
                'Final Train Acc (%)': f"{results['final_train_acc']:.2f}",
                'Overfit Gap (pp)': f"{results.get('overfit_gap', 0):.2f}",
                'Minority Ratio': f"{results.get('minority_ratio', 0):.3f}",
                'Data Points': f"{results['data_points']:,}",
                'Target Mode': results.get('target_mode', 'binary'),
                'Status': '✅ Success'
            })
        else:
            summary_data.append({
                'Pair': pair_name,
                'Best Val Acc (%)': '-',
                'Final Train Acc (%)': '-',
                'Overfit Gap (pp)': '-',
                'Minority Ratio': '-',
                'Data Points': '-',
                'Target Mode': '-',
                'Status': f"❌ {results['status']}"
            })
    
    df = pd.DataFrame(summary_data)
    return df

def print_summary_report(results_summary: Dict[str, Dict], 
                        trainer_histories: Dict[str, Dict]) -> None:
    """Print a comprehensive summary report.
    
    Args:
        results_summary: Dictionary of results summary per pair
        trainer_histories: Dictionary of training histories per pair
    """
    print(f"\n🏆 MODULAR LSTM TRAINING SUMMARY:")
    print("="*80)
    
    successful_pairs = 0
    total_pairs = len(results_summary)
    
    for pair_name, results in results_summary.items():
        print(f"📊 {pair_name}:")
        
        if results['status'] == 'success':
            successful_pairs += 1
            best_val_acc = results['best_val_acc']
            final_train_acc = results['final_train_acc']
            minority_ratio = results.get('minority_ratio', 0)
            overfit_gap = results.get('overfit_gap', 0)
            target_mode = results.get('target_mode', 'binary')
            
            print(f"   🎯 Best Val Acc: {best_val_acc:.2f}% ({target_mode})")
            print(f"   📈 Final Train Acc: {final_train_acc:.2f}%")
            print(f"   📊 Data points: {results['data_points']:,}")
            print(f"   ⚖️ Minority ratio: {minority_ratio:.3f}")
            print(f"   🔥 Overfit gap: {overfit_gap:.2f}pp")
            
            if results.get('best_params'):
                print(f"   🔍 Optimized params: {results['best_params']}")
            
            if best_val_acc > 65:
                print(f"   🟢 Excellent!")
            elif best_val_acc > 60:
                print(f"   🟡 Good")
            else:
                print(f"   🔴 Needs improvement")
        else:
            print(f"   ❌ Status: {results['status']}")
        print()
    
    print(f"📈 Success Rate: {successful_pairs}/{total_pairs} ({100*successful_pairs/total_pairs:.1f}%)")

__all__ = [
    'plot_training_results', 
    'plot_loss_curves', 
    'plot_accuracy_comparison',
    'create_summary_table',
    'print_summary_report'
]
