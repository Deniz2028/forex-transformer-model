# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import json

# Focal Loss imports (eğer ayrı dosya oluşturduysanız)
try:
    from ..losses.focal_loss import FocalLoss, VolatilityAdaptiveFocalLoss, ForexFocalLoss
except ImportError:
    # Focal Loss inline implementation
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, weight=None, size_average=True):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.weight = weight
            self.size_average = size_average

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()
    
    class VolatilityAdaptiveFocalLoss(nn.Module):
        def __init__(self, base_alpha=0.25, base_gamma=2.0):
            super().__init__()
            self.base_alpha = base_alpha
            self.base_gamma = base_gamma
            
        def forward(self, inputs, targets, volatility_factor=1.0):
            alpha = self.base_alpha * (0.5 + 0.5 * volatility_factor)
            gamma = self.base_gamma * (0.8 + 0.4 * volatility_factor)
            
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = alpha * (1 - pt)**gamma * ce_loss
            
            return focal_loss.mean()
    
    class ForexFocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.class_weights = class_weights
            
        def forward(self, inputs, targets, market_regime='normal'):
            if market_regime == 'high_volatility':
                gamma = self.gamma * 1.25
            elif market_regime == 'low_volatility':
                gamma = self.gamma * 0.8
            else:
                gamma = self.gamma
            
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            
            if self.class_weights is not None:
                alpha_t = self.class_weights[targets]
            else:
                alpha_t = self.alpha
            
            focal_loss = alpha_t * (1 - pt)**gamma * ce_loss
            return focal_loss.mean()

class HybridForexTrainer:
    """
    OneCycleLR ve Focal Loss ile optimize edilmiş Forex trainer
    Mevcut proje yapınıza uyumlu tam implementation
    """
    
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Config attribute'larına güvenli erişim
        self.epochs = getattr(config, 'epochs', 50)
        self.batch_size = getattr(config, 'batch_size', 32)
        
        # Optimization config'e güvenli erişim
        if hasattr(config, 'optimization'):
            opt_config = config.optimization
        else:
            # Backward compatibility için default değerler
            class OptConfig:
                use_onecycle = True
                max_lr = 0.01
                pct_start = 0.25
                div_factor = 10.0
                final_div_factor = 1000.0
                loss_type = 'focal'
                focal_alpha = 0.25
                focal_gamma = 2.0
                accumulation_steps = 1
                use_mixed_precision = True
                gradient_clip_norm = 1.0
                base_lr = 0.001
                weight_decay = 1e-4
                optimizer_type = 'adam'
                patience = 10
                
            opt_config = OptConfig()
        
        self.opt_config = opt_config
        
        # Optimizer setup
        self.setup_optimizer()
        
        # Loss function setup
        self.setup_loss_function()
        
        # Scheduler (OneCycleLR ile değiştirilecek)
        self.scheduler = None
        self.onecycle_scheduler = None
        
        # Training tracking
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr_history': [],
            'epoch_times': []
        }
        
        # Memory optimization
        self.accumulation_steps = getattr(opt_config, 'accumulation_steps', 1)
        self.use_mixed_precision = getattr(opt_config, 'use_mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Early stopping
        self.patience = getattr(opt_config, 'patience', 10)
        self.patience_counter = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trainer initialized with device: {device}")
        
    def setup_optimizer(self):
        """Optimizer konfigürasyonu"""
        optimizer_type = getattr(self.opt_config, 'optimizer_type', 'adam')
        base_lr = getattr(self.opt_config, 'base_lr', 0.001)
        weight_decay = getattr(self.opt_config, 'weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=base_lr, 
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=base_lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=base_lr, 
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
            
        self.logger.info(f"Optimizer set to: {optimizer_type}")
        
    def setup_loss_function(self):
        """Loss function konfigürasyonu"""
        loss_type = getattr(self.opt_config, 'loss_type', 'focal')
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=getattr(self.opt_config, 'focal_alpha', 0.25),
                gamma=getattr(self.opt_config, 'focal_gamma', 2.0)
            )
        elif loss_type == 'adaptive_focal':
            self.criterion = VolatilityAdaptiveFocalLoss(
                base_alpha=getattr(self.opt_config, 'focal_alpha', 0.25),
                base_gamma=getattr(self.opt_config, 'focal_gamma', 2.0)
            )
        elif loss_type == 'forex_focal':
            self.criterion = ForexFocalLoss(
                alpha=getattr(self.opt_config, 'focal_alpha', 0.25),
                gamma=getattr(self.opt_config, 'focal_gamma', 2.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.logger.info(f"Loss function set to: {loss_type}")
        
    def setup_onecycle_scheduler(self, steps_per_epoch, epochs):
        """OneCycleLR scheduler kurulumu"""
        if not getattr(self.opt_config, 'use_onecycle', True):
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
            self.logger.info("Using ReduceLROnPlateau scheduler")
            return
            
        total_steps = epochs * steps_per_epoch
        
        max_lr = getattr(self.opt_config, 'max_lr', 0.01)
        pct_start = getattr(self.opt_config, 'pct_start', 0.25)
        div_factor = getattr(self.opt_config, 'div_factor', 10.0)
        final_div_factor = getattr(self.opt_config, 'final_div_factor', 1000.0)
        
        self.onecycle_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=True,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
        
        self.lr_history = []
        
        self.logger.info(f"OneCycleLR scheduler setup: total_steps={total_steps}, max_lr={max_lr}")
        
    def train_step(self, batch, batch_idx):
        """Tek training step"""
        self.model.train()
        
        # Batch'i device'a taşı
        if isinstance(batch, dict):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device) if 'targets' in batch else batch['labels'].to(self.device)
        else:
            # Tuple veya list formatı (features, targets)
            features, targets = batch
            features = features.to(self.device)
            targets = targets.to(self.device)
        
        # Mixed precision training
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                predictions = self.model(features)
                loss = self.calculate_loss(predictions, targets, batch)
        else:
            predictions = self.model(features)
            loss = self.calculate_loss(predictions, targets, batch)
        
        # Gradient accumulation
        loss = loss / self.accumulation_steps
        
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % self.accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=getattr(self.opt_config, 'gradient_clip_norm', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=getattr(self.opt_config, 'gradient_clip_norm', 1.0)
                )
                self.optimizer.step()
            
            # OneCycleLR step (kritik: her batch sonrası)
            if self.onecycle_scheduler:
                self.onecycle_scheduler.step()
                current_lr = self.onecycle_scheduler.get_last_lr()[0]
                self.lr_history.append(current_lr)
                
            self.optimizer.zero_grad()
            self.current_step += 1
            
        return loss.item() * self.accumulation_steps
    
    def calculate_loss(self, predictions, targets, batch):
        """Loss hesaplama - volatilite adaptasyonu ile"""
        if isinstance(self.criterion, VolatilityAdaptiveFocalLoss):
            # Volatilite faktörünü hesapla
            volatility_factor = self.calculate_volatility_factor(batch)
            return self.criterion(predictions, targets, volatility_factor)
        elif isinstance(self.criterion, ForexFocalLoss):
            # Market rejimini belirle
            market_regime = self.determine_market_regime(batch)
            return self.criterion(predictions, targets, market_regime)
        else:
            return self.criterion(predictions, targets)
    
    def calculate_volatility_factor(self, batch):
        """Batch için volatilite faktörü hesaplama"""
        if isinstance(batch, dict) and 'returns' in batch:
            returns = batch['returns']
            volatility = torch.std(returns, dim=1)
            volatility_percentile = torch.quantile(volatility, 0.7)
            return torch.clamp(volatility / volatility_percentile, 0.5, 2.0).mean()
        return torch.tensor(1.0, device=self.device)
    
    def determine_market_regime(self, batch):
        """Market rejimi belirleme"""
        if isinstance(batch, dict) and 'atr' in batch:
            atr = batch['atr'].mean()
            if atr > 1.5:  # Threshold değerleri konfigürasyona alınabilir
                return 'high_volatility'
            elif atr < 0.5:
                return 'low_volatility'
        return 'normal'
    
    def validation_step(self, batch):
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Batch handling
            if isinstance(batch, dict):
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device) if 'targets' in batch else batch['labels'].to(self.device)
            else:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
            
            predictions = self.model(features)
            loss = self.calculate_loss(predictions, targets, batch)
            
            # Accuracy calculation
            predicted_classes = torch.argmax(predictions, dim=1)
            correct = (predicted_classes == targets).float()
            accuracy = correct.mean()
            
        return {
            'val_loss': loss.item(),
            'val_acc': accuracy.item(),
            'predictions': predicted_classes.cpu(),
            'targets': targets.cpu()
        }
    
    def train_epoch(self, train_dataloader, val_dataloader=None):
        """Tek epoch training"""
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        # OneCycleLR setup (ilk epoch'ta)
        if self.current_epoch == 0 and self.onecycle_scheduler is None:
            self.setup_onecycle_scheduler(num_batches, self.epochs)
        
        # Training loop
        for batch_idx, batch in enumerate(train_dataloader):
            batch_loss = self.train_step(batch, batch_idx)
            total_loss += batch_loss
            
            # Log progress
            log_interval = getattr(self.config, 'log_interval', 100)
            if batch_idx % log_interval == 0:
                current_lr = self.get_current_lr()
                self.logger.info(
                    f"Epoch {self.current_epoch + 1}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {batch_loss:.4f}, LR: {current_lr:.6f}"
                )
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        val_metrics = {}
        if val_dataloader:
            val_metrics = self.validate(val_dataloader)
        
        # Update history
        self.training_history['train_loss'].append(avg_train_loss)
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_acc'].append(val_metrics['val_acc'])
        
        # ReduceLROnPlateau step (OneCycleLR kullanılmıyorsa)
        if self.scheduler and not self.onecycle_scheduler and val_metrics:
            self.scheduler.step(val_metrics['val_loss'])
        
        self.current_epoch += 1
        epoch_time = time.time() - epoch_start_time
        self.training_history['epoch_times'].append(epoch_time)
        
        self.logger.info(
            f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s, "
            f"Train Loss: {avg_train_loss:.4f}"
        )
        
        if val_metrics:
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")
            
            # Early stopping check
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        return {
            'train_loss': avg_train_loss,
            **val_metrics
        }
    
    def validate(self, val_dataloader):
        """Full validation"""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in val_dataloader:
            metrics = self.validation_step(batch)
            
            # Batch size hesaplama
            if isinstance(batch, dict):
                batch_size = batch['features'].size(0)
            else:
                batch_size = batch[0].size(0)
            
            total_loss += metrics['val_loss'] * batch_size
            total_correct += metrics['val_acc'] * batch_size
            total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
    
    def get_current_lr(self):
        """Mevcut learning rate'i al"""
        if self.onecycle_scheduler:
            return self.onecycle_scheduler.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def should_stop_early(self):
        """Early stopping kontrolü"""
        return self.patience_counter >= self.patience
    
    def save_checkpoint(self, filepath, epoch=None, val_acc=None, additional_info=None):
        """Model checkpoint kaydetme"""
        if epoch is None:
            epoch = self.current_epoch
        if val_acc is None:
            val_acc = self.best_val_acc
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'training_history': self.training_history,
            'current_step': self.current_step,
            'patience_counter': self.patience_counter
        }
        
        if self.onecycle_scheduler:
            checkpoint['scheduler_state_dict'] = self.onecycle_scheduler.state_dict()
        elif self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if additional_info:
            checkpoint.update(additional_info)
        
        # Checkpoint dizinini oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Checkpoint yükleme"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.current_step = checkpoint.get('current_step', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Scheduler state'i yükle
        if 'scheduler_state_dict' in checkpoint:
            if self.onecycle_scheduler:
                self.onecycle_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}, Epoch: {self.current_epoch}")
        
        return checkpoint
    
    def plot_training_history(self, save_path=None):
        """Training history görselleştirmesi"""
        if not self.training_history['train_loss']:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curve
        if self.training_history['val_acc']:
            axes[0, 1].plot(self.training_history['val_acc'], color='green')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
        
        # Learning rate schedule
        if self.lr_history:
            axes[1, 0].plot(self.lr_history, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Epoch times
        if self.training_history['epoch_times']:
            axes[1, 1].plot(self.training_history['epoch_times'], color='purple')
            axes[1, 1].set_title('Epoch Training Times')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved: {save_path}")
        
        plt.show()
    
    def get_training_summary(self):
        """Training özeti"""
        summary = {
            'total_epochs': self.current_epoch,
            'total_steps': self.current_step,
            'best_val_acc': self.best_val_acc,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'final_val_acc': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else None,
            'total_training_time': sum(self.training_history['epoch_times']) if self.training_history['epoch_times'] else 0,
            'avg_epoch_time': np.mean(self.training_history['epoch_times']) if self.training_history['epoch_times'] else 0
        }
        
        return summary
    
    def print_training_summary(self):
        """Training özetini yazdır"""
        summary = self.get_training_summary()
        
        print("=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total Epochs: {summary['total_epochs']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Best Validation Accuracy: {summary['best_val_acc']:.4f}")
        
        if summary['final_train_loss']:
            print(f"Final Training Loss: {summary['final_train_loss']:.4f}")
        if summary['final_val_loss']:
            print(f"Final Validation Loss: {summary['final_val_loss']:.4f}")
        if summary['final_val_acc']:
            print(f"Final Validation Accuracy: {summary['final_val_acc']:.4f}")
        
        print(f"Total Training Time: {summary['total_training_time']:.2f} seconds")
        print(f"Average Epoch Time: {summary['avg_epoch_time']:.2f} seconds")
        print("=" * 50)
    
    def save_training_summary(self, filepath):
        """Training özetini JSON olarak kaydet"""
        summary = self.get_training_summary()
        
        # Datetime bilgisi ekle
        summary['training_completed_at'] = datetime.now().isoformat()
        summary['device'] = str(self.device)
        
        # Config bilgisini ekle
        if hasattr(self.config, 'to_dict'):
            summary['config'] = self.config.to_dict()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved: {filepath}")
    
    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        """Ana training metodu"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Accumulation steps: {self.accumulation_steps}")
        self.logger.info(f"Mixed precision: {self.use_mixed_precision}")
        self.logger.info(f"Loss type: {getattr(self.opt_config, 'loss_type', 'cross_entropy')}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.epochs):
                # Train epoch
                metrics = self.train_epoch(train_dataloader, val_dataloader)
                
                # Early stopping check
                if self.should_stop_early():
                    self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
                
                # Save checkpoint
                save_interval = getattr(self.config, 'save_interval', 5)
                if (epoch + 1) % save_interval == 0:
                    checkpoint_dir = getattr(self.config, 'checkpoint_dir', './checkpoints')
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                    self.save_checkpoint(checkpoint_path, epoch, metrics.get('val_acc', 0.0))
                
                # Save best model
                if val_dataloader and metrics['val_acc'] > self.best_val_acc:
                    best_model_path = os.path.join(
                        getattr(self.config, 'checkpoint_dir', './checkpoints'), 
                        'best_model.pt'
                    )
                    self.save_checkpoint(best_model_path, epoch, metrics['val_acc'])
                    self.best_val_acc = metrics['val_acc']
            
            # Final evaluation
            if test_dataloader:
                self.logger.info("Running final test evaluation...")
                test_metrics = self.validate(test_dataloader)
                self.logger.info(f"Test Accuracy: {test_metrics['val_acc']:.4f}")
                self.logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            
            # Print summary
            self.print_training_summary()
            
            return {
                'best_val_acc': self.best_val_acc,
                'training_history': self.training_history,
                'total_time': total_time
            }
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            
            # Save interrupted training state
            interrupt_path = os.path.join(
                getattr(self.config, 'checkpoint_dir', './checkpoints'), 
                'interrupted_training.pt'
            )
            self.save_checkpoint(interrupt_path, self.current_epoch, self.best_val_acc)
            
            return {
                'best_val_acc': self.best_val_acc,
                'training_history': self.training_history,
                'interrupted': True
            }
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise

# Utility functions
def create_trainer(model, config, device=None):
    """Trainer oluşturma helper fonksiyonu"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = HybridForexTrainer(model, config, device)
    return trainer

def resume_training(checkpoint_path, model, config, train_dataloader, val_dataloader=None):
    """Checkpoint'ten training'i devam ettirme"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = HybridForexTrainer(model, config, device)
    
    # Checkpoint yükle
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    
    # Kalan epoch sayısını hesapla
    remaining_epochs = config.epochs - trainer.current_epoch
    if remaining_epochs > 0:
        trainer.epochs = remaining_epochs
        trainer.logger.info(f"Resuming training for {remaining_epochs} more epochs")
        
        # Training'i devam ettir
        return trainer.train(train_dataloader, val_dataloader)
    else:
        trainer.logger.info("Training already completed")
        return trainer.get_training_summary()

# Backward compatibility
class Trainer(HybridForexTrainer):
    """Mevcut kodunuzla uyumluluk için alias"""
    pass
